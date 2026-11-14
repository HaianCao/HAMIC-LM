"""Immutable checkpoint manifests and commit markers.

A checkpoint directory is resumable only when it holds ``manifest.json`` (every payload
file with size and SHA-256) and ``COMMITTED.json`` (the manifest hash), written last.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hamic.provenance import file_sha256, utc_now

__all__ = [
    "COMMIT_NAME",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "CheckpointError",
    "ManifestEntry",
    "build_manifest",
    "commit_marker",
    "payload_files",
    "verify_local",
]

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMMIT_NAME = "COMMITTED.json"
_CONTROL = {MANIFEST_NAME, COMMIT_NAME}


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, corrupt or unsafe to use."""


@dataclass(frozen=True)
class ManifestEntry:
    """One payload file in a manifest."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a temporary file and an atomic rename."""
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def payload_files(directory: Path) -> list[Path]:
    """Return payload files (everything except control files), sorted by relative path."""
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise CheckpointError(f"Refusing symlink inside checkpoint: {path}")
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.parent == directory and path.name in _CONTROL:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(directory).as_posix())


def build_manifest(directory: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Hash every payload file and write ``manifest.json`` into ``directory``.

    Raises:
        CheckpointError: If the directory is already committed or has no payload.
    """
    root = Path(directory)
    if (root / COMMIT_NAME).exists():
        raise CheckpointError(f"{root} is already committed and must stay immutable")
    files = payload_files(root)
    if not files:
        raise CheckpointError(f"{root} has no payload files")
    json.dumps(metadata, allow_nan=False)  # metadata must be portable JSON
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "metadata": metadata,
        "files": [
            ManifestEntry(
                path=p.relative_to(root).as_posix(), size=p.stat().st_size, sha256=file_sha256(p)
            ).to_dict()
            for p in files
        ],
    }
    atomic_write_json(root / MANIFEST_NAME, manifest)
    return manifest


def commit_marker(manifest_sha256: str) -> dict[str, Any]:
    """Return the content of ``COMMITTED.json`` for a manifest hash."""
    return {
        "schema_version": SCHEMA_VERSION,
        "committed_at_utc": utc_now(),
        "manifest_sha256": manifest_sha256,
    }


def parse_manifest(data: bytes) -> list[ManifestEntry]:
    """Parse and validate the file list of a manifest."""
    try:
        manifest = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Manifest is not valid JSON: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("Unsupported manifest schema version")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise CheckpointError("Manifest lists no payload files")
    seen: set[str] = set()
    result: list[ManifestEntry] = []
    for item in entries:
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or not path or path in seen:
            raise CheckpointError(f"Malformed or duplicate manifest entry: {item!r}")
        if path.startswith("/") or ".." in Path(path).parts:
            raise CheckpointError(f"Manifest path escapes checkpoint: {path}")
        seen.add(path)
        result.append(ManifestEntry(path=path, size=int(item["size"]), sha256=str(item["sha256"])))
    return result


def verify_local(directory: str | Path) -> dict[str, Any]:
    """Verify a committed checkpoint directory and return its manifest metadata.

    Raises:
        CheckpointError: On any missing marker, hash mismatch or unexpected file.
    """
    root = Path(directory)
    manifest_path, marker_path = root / MANIFEST_NAME, root / COMMIT_NAME
    if not manifest_path.is_file() or not marker_path.is_file():
        raise CheckpointError(f"{root} is incomplete: {MANIFEST_NAME} and {COMMIT_NAME} required")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != file_sha256(manifest_path):
        raise CheckpointError("Manifest hash does not match COMMITTED.json")
    entries = parse_manifest(manifest_path.read_bytes())
    for entry in entries:
        candidate = root / entry.path
        if not candidate.is_file() or candidate.stat().st_size != entry.size:
            raise CheckpointError(f"Missing or truncated payload file: {entry.path}")
        if file_sha256(candidate) != entry.sha256:
            raise CheckpointError(f"SHA-256 mismatch: {entry.path}")
    actual = {p.relative_to(root).as_posix() for p in payload_files(root)}
    expected = {e.path for e in entries}
    if actual != expected:
        raise CheckpointError(
            f"Payload changed: unexpected={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )
    metadata: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]
    return metadata
