"""Remote object stores for checkpoints.

:class:`ObjectStore` is the only interface the checkpoint logic depends on, so the
Hugging Face Hub backend used in production and the local backend used in tests are
interchangeable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hamic.provenance import file_sha256

__all__ = ["HuggingFaceObjectStore", "LocalObjectStore", "ObjectStore", "RemoteFileInfo"]


@dataclass(frozen=True)
class RemoteFileInfo:
    """Size and, when the backend knows it, the SHA-256 of a remote file."""

    size: int
    sha256: str | None = None


class ObjectStore(Protocol):
    """Minimal object-store operations used by checkpoint publishing and resolution."""

    def put_files(
        self,
        files: Mapping[str, Path],
        message: str,
        copies: Mapping[str, str] | None = None,
    ) -> None:
        """Upload ``{remote_path: local_file}`` as one atomic batch where possible.

        ``copies`` maps new remote paths to existing remote paths copied inside the store
        (no upload) in the same batch.
        """
        ...

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        """Upload a small object."""
        ...

    def read_bytes(self, remote_path: str) -> bytes | None:
        """Return the object's bytes, or ``None`` if it does not exist."""
        ...

    def download(self, remote_path: str, destination: Path) -> Path:
        """Download an object to ``destination`` and return the path."""
        ...

    def download_prefix(self, prefix: str, destination: Path, max_workers: int = 8) -> Path:
        """Download every object under ``prefix`` into a new directory ``destination``."""
        ...

    def list_files(self, prefix: str) -> list[str]:
        """List object paths under ``prefix``."""
        ...

    def stat(self, remote_paths: list[str]) -> dict[str, RemoteFileInfo]:
        """Return info for the paths that exist."""
        ...


class LocalObjectStore:
    """A directory acting as an object store (tests and offline drills)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, remote_path: str) -> Path:
        target = (self.root / remote_path).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            raise ValueError(f"Path escapes store root: {remote_path}")
        return target

    def put_files(
        self,
        files: Mapping[str, Path],
        message: str,
        copies: Mapping[str, str] | None = None,
    ) -> None:
        """Copy files into the store; ``copies`` duplicate existing store objects."""
        del message
        for remote, existing in (copies or {}).items():
            target = self._path(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._path(existing), target)
        for remote, local in files.items():
            target = self._path(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        """Write bytes into the store."""
        del message
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def read_bytes(self, remote_path: str) -> bytes | None:
        """Read bytes, or ``None`` when absent."""
        target = self._path(remote_path)
        return target.read_bytes() if target.is_file() else None

    def download(self, remote_path: str, destination: Path) -> Path:
        """Copy an object out of the store."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), destination)
        return destination

    def download_prefix(self, prefix: str, destination: Path, max_workers: int = 8) -> Path:
        """Copy every object under ``prefix`` into ``destination``."""
        del max_workers
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        source = self._path(prefix)
        if not source.is_dir():
            raise FileNotFoundError(f"Nothing found under {prefix!r}")
        shutil.copytree(source, destination)
        return destination

    def list_files(self, prefix: str) -> list[str]:
        """List files under ``prefix``."""
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())

    def stat(self, remote_paths: list[str]) -> dict[str, RemoteFileInfo]:
        """Return size and SHA-256 of existing paths."""
        info: dict[str, RemoteFileInfo] = {}
        for remote in remote_paths:
            target = self._path(remote)
            if target.is_file():
                info[remote] = RemoteFileInfo(target.stat().st_size, file_sha256(target))
        return info


class HuggingFaceObjectStore:
    """A (private) Hugging Face Hub repository used as an object store.

    Each ``put_*`` call is one Hub commit, so a batch of payload files appears atomically.
    The token is taken from ``token`` when given (e.g. typed into a password field), else
    from ``HF_TOKEN`` or the cached login. This class never logs or writes the token.

    Args:
        repo_id: e.g. ``"<namespace>/hamic-runs"``.
        repo_type: ``"model"`` (default) or ``"dataset"``.
        private: Create the repository as private if it does not exist.
        create: Create the repository when missing. An existing repository is never
            touched, so a fine-grained token scoped to that one repository is enough.
        token: Hugging Face access token held in memory only.
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = True,
        create: bool = True,
        token: str | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._api = HfApi(token=token or None)
        if create and not self._api.repo_exists(repo_id, repo_type=repo_type):
            self._api.create_repo(repo_id, repo_type=repo_type, private=private)

    def __repr__(self) -> str:
        return f"HuggingFaceObjectStore(repo_id={self.repo_id!r}, repo_type={self.repo_type!r})"

    def whoami(self) -> str:
        """Return the account name the token belongs to (verifies the token works)."""
        return str(self._api.whoami()["name"])

    def put_files(
        self,
        files: Mapping[str, Path],
        message: str,
        copies: Mapping[str, str] | None = None,
    ) -> None:
        """Upload files in a single commit; ``copies`` are server-side copies (no upload)."""
        from huggingface_hub import CommitOperationAdd, CommitOperationCopy

        operations: list[Any] = [
            CommitOperationCopy(src_path_in_repo=existing, path_in_repo=remote)
            for remote, existing in (copies or {}).items()
        ]
        operations += [
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for remote, local in files.items()
        ]
        self._api.create_commit(
            self.repo_id, operations=operations, commit_message=message, repo_type=self.repo_type
        )

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        """Upload bytes in a single commit."""
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=message,
        )

    def read_bytes(self, remote_path: str) -> bytes | None:
        """Download a small file into memory, or return ``None`` when absent."""
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

        try:
            path = self._api.hf_hub_download(
                self.repo_id, remote_path, repo_type=self.repo_type, force_download=True
            )
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            return None
        return Path(path).read_bytes()

    def _staging(self, destination: Path) -> Path:
        # Same filesystem as the destination, so the final move is a rename, not a copy.
        destination.parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))

    def download(self, remote_path: str, destination: Path) -> Path:
        """Download a file straight to ``destination`` (no Hub cache, no extra copy)."""
        staging = self._staging(destination)
        try:
            fetched = self._api.hf_hub_download(
                self.repo_id, remote_path, repo_type=self.repo_type, local_dir=staging
            )
            os.replace(fetched, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return destination

    def download_prefix(self, prefix: str, destination: Path, max_workers: int = 8) -> Path:
        """Download every file under ``prefix`` in parallel into ``destination``."""
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        staging = self._staging(destination)
        try:
            self._api.snapshot_download(
                self.repo_id,
                repo_type=self.repo_type,
                allow_patterns=[f"{prefix.rstrip('/')}/*"],
                local_dir=staging,
                max_workers=max_workers,
            )
            source = staging / prefix.rstrip("/")
            if not source.is_dir():
                raise FileNotFoundError(f"Nothing found under {prefix!r} in {self.repo_id}")
            os.replace(source, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return destination

    def list_files(self, prefix: str) -> list[str]:
        """List repository files under ``prefix``."""
        files = self._api.list_repo_files(self.repo_id, repo_type=self.repo_type)
        return sorted(f for f in files if f.startswith(prefix))

    def stat(self, remote_paths: list[str]) -> dict[str, RemoteFileInfo]:
        """Return sizes and LFS SHA-256 values (``None`` for small non-LFS files)."""
        infos: list[Any] = self._api.get_paths_info(
            self.repo_id, remote_paths, expand=True, repo_type=self.repo_type
        )
        result: dict[str, RemoteFileInfo] = {}
        for item in infos:
            size = getattr(item, "size", None)
            if size is None:
                continue  # a folder
            lfs = getattr(item, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            result[item.path] = RemoteFileInfo(size=int(size), sha256=sha)
        return result
