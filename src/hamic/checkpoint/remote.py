"""Publish checkpoints to an object store and resolve the newest valid one.

Publishing order (each step only after the previous one is verified):

1. ``manifest.json`` is built locally over every payload file.
2. Payload files and the manifest are uploaded in one batch.
3. Remote sizes and hashes are verified against the manifest.
4. ``COMMITTED.json`` is uploaded; only now is the checkpoint resumable.
5. ``LATEST.json`` is updated. It is a hint, never the source of truth.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hamic.checkpoint.manifest import (
    COMMIT_NAME,
    MANIFEST_NAME,
    CheckpointError,
    ManifestEntry,
    build_manifest,
    commit_marker,
    parse_manifest,
    verify_local,
)
from hamic.checkpoint.storage import ObjectStore
from hamic.provenance import file_sha256

__all__ = ["CheckpointPublisher", "CheckpointResolver", "RemoteCheckpoint"]

LATEST_NAME = "LATEST.json"


def checkpoint_name(step: int) -> str:
    """Return the canonical checkpoint directory name for an optimizer step."""
    return f"step-{step:09d}"


@dataclass(frozen=True)
class RemoteCheckpoint:
    """A committed checkpoint in the store."""

    prefix: str
    metadata: dict[str, Any]

    @property
    def step(self) -> int:
        """Global optimizer step recorded in the metadata."""
        return int(self.metadata["global_step"])


class CheckpointPublisher:
    """Upload a local checkpoint directory with the publish-last protocol."""

    def __init__(self, store: ObjectStore, run_prefix: str) -> None:
        self.store = store
        self.run_prefix = run_prefix.rstrip("/")

    def publish(self, local_dir: str | Path, metadata: dict[str, Any]) -> RemoteCheckpoint:
        """Publish ``local_dir`` as checkpoint ``step-<global_step>``.

        Raises:
            CheckpointError: If the checkpoint already exists remotely or verification fails.
        """
        root = Path(local_dir)
        name = checkpoint_name(int(metadata["global_step"]))
        prefix = f"{self.run_prefix}/checkpoints/{name}"
        if self.store.read_bytes(f"{prefix}/{COMMIT_NAME}") is not None:
            raise CheckpointError(f"{prefix} is already committed; checkpoints are immutable")

        build_manifest(root, metadata)
        manifest_path = root / MANIFEST_NAME
        entries = parse_manifest(manifest_path.read_bytes())
        files = {f"{prefix}/{e.path}": root / e.path for e in entries}
        files[f"{prefix}/{MANIFEST_NAME}"] = manifest_path
        self.store.put_files(files, message=f"checkpoint payload {name}")

        self._verify_remote(prefix, entries, manifest_path)
        marker = commit_marker(file_sha256(manifest_path))
        self.store.put_bytes(
            f"{prefix}/{COMMIT_NAME}", _json_bytes(marker), message=f"commit {name}"
        )
        if self.store.read_bytes(f"{prefix}/{COMMIT_NAME}") is None:
            raise CheckpointError(f"Commit marker for {prefix} is not visible after upload")
        (root / COMMIT_NAME).write_bytes(_json_bytes(marker))
        latest = {"checkpoint": prefix, "global_step": metadata["global_step"]}
        self.store.put_bytes(
            f"{self.run_prefix}/{LATEST_NAME}", _json_bytes(latest), message=f"latest -> {name}"
        )
        return RemoteCheckpoint(prefix=prefix, metadata=metadata)

    def _verify_remote(
        self, prefix: str, entries: list[ManifestEntry], manifest_path: Path
    ) -> None:
        wanted = [f"{prefix}/{e.path}" for e in entries] + [f"{prefix}/{MANIFEST_NAME}"]
        info = self.store.stat(wanted)
        expected = {f"{prefix}/{e.path}": (e.size, e.sha256) for e in entries}
        expected[f"{prefix}/{MANIFEST_NAME}"] = (
            manifest_path.stat().st_size,
            file_sha256(manifest_path),
        )
        for remote, (size, sha) in expected.items():
            found = info.get(remote)
            if found is None or found.size != size:
                raise CheckpointError(f"Remote file missing or wrong size: {remote}")
            if found.sha256 is not None and found.sha256 != sha:
                raise CheckpointError(f"Remote SHA-256 mismatch: {remote}")


class CheckpointResolver:
    """Find, verify and download committed checkpoints of a run."""

    def __init__(self, store: ObjectStore, run_prefix: str) -> None:
        self.store = store
        self.run_prefix = run_prefix.rstrip("/")

    def committed(self) -> list[RemoteCheckpoint]:
        """Committed checkpoints, oldest first (incomplete prefixes are ignored)."""
        found: list[RemoteCheckpoint] = []
        for path in self.store.list_files(f"{self.run_prefix}/checkpoints/"):
            if not path.endswith(f"/{COMMIT_NAME}"):
                continue
            prefix = path[: -len(COMMIT_NAME) - 1]
            manifest = self.store.read_bytes(f"{prefix}/{MANIFEST_NAME}")
            if manifest is None:
                continue
            metadata = json.loads(manifest)["metadata"]
            found.append(RemoteCheckpoint(prefix=prefix, metadata=metadata))
        return sorted(found, key=lambda c: c.step)

    def download(self, checkpoint: RemoteCheckpoint, destination: str | Path) -> Path:
        """Download ``checkpoint`` and verify it locally; raise on any mismatch."""
        target = Path(destination)
        try:
            # One parallel transfer for the whole prefix; integrity is checked afterwards.
            self.store.download_prefix(checkpoint.prefix, target)
        except FileNotFoundError as exc:
            raise CheckpointError(f"{checkpoint.prefix} has no files") from exc
        verify_local(target)
        return target

    def restore_latest(self, destination: str | Path) -> RemoteCheckpoint | None:
        """Download the newest checkpoint that verifies, falling back to older ones.

        Returns:
            The restored checkpoint, or ``None`` when the run has no valid checkpoint.
        """
        for candidate in reversed(self.committed()):
            with tempfile.TemporaryDirectory(dir=Path(destination).parent) as staging:
                try:
                    staged = self.download(candidate, Path(staging) / "ckpt")
                except CheckpointError:
                    continue  # corrupt generation: try the previous one
                final = Path(destination)
                if final.exists():
                    raise FileExistsError(f"Restore destination already exists: {final}")
                staged.rename(final)
                return candidate
        return None


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
