"""Remote object stores for checkpoints.

:class:`ObjectStore` is the only interface the checkpoint logic depends on, so the
Hugging Face Hub backend used in production and the local backend used in tests are
interchangeable.
"""

from __future__ import annotations

import shutil
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

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        """Upload ``{remote_path: local_file}`` as one atomic batch where possible."""
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

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        """Copy files into the store."""
        del message
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
    Authentication comes from the ``HF_TOKEN`` environment variable or the cached login;
    tokens are never passed as arguments or written to disk by this class.

    Args:
        repo_id: e.g. ``"HAMIC/hamic-runs"``.
        repo_type: ``"model"`` (default) or ``"dataset"``.
        private: Create the repository as private if it does not exist.
        create: Create the repository when missing.
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = True,
        create: bool = True,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._api = HfApi()
        if create:
            self._api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        """Upload files in a single commit."""
        from huggingface_hub import CommitOperationAdd

        operations = [
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

    def download(self, remote_path: str, destination: Path) -> Path:
        """Download a file to ``destination``."""
        cached = self._api.hf_hub_download(self.repo_id, remote_path, repo_type=self.repo_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
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
