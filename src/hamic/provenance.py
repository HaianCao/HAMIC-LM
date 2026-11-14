"""Provenance helpers: hashes, code revision and environment fingerprints."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "environment_fingerprint",
    "file_sha256",
    "git_revision",
    "json_sha256",
    "utc_now",
]

_TRACKED_PACKAGES = ("hamic", "torch", "transformers", "tokenizers", "datasets", "huggingface-hub")


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 with a ``Z`` suffix."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON encoding of ``value``."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision(cwd: str | Path | None = None) -> dict[str, Any]:
    """Return the current commit and whether the work tree has uncommitted changes.

    Returns ``{"commit": None, "dirty": None}`` outside a Git repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def environment_fingerprint() -> dict[str, Any]:
    """Return Python and key package versions (never secrets or paths)."""
    versions: dict[str, str | None] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": versions,
    }
