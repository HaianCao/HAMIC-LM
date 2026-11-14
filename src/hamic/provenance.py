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
    "code_revision",
    "environment_fingerprint",
    "file_sha256",
    "git_revision",
    "installed_revision",
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


def installed_revision(package: str = "hamic") -> dict[str, Any]:
    """Return the Git commit pip recorded when ``package`` was installed from a VCS URL.

    pip writes ``direct_url.json`` (PEP 610) with the resolved ``commit_id`` even when
    the requested revision was a branch such as ``main``.
    """
    try:
        raw = metadata.distribution(package).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        raw = None
    data = json.loads(raw) if raw else {}
    vcs = data.get("vcs_info", {})
    return {
        "commit": vcs.get("commit_id"),
        "requested_revision": vcs.get("requested_revision"),
        "url": data.get("url"),
    }


def code_revision() -> dict[str, Any]:
    """Return the commit of the HAMIC code that is running, wherever it came from.

    A source checkout is identified with Git (run inside the package directory, never the
    caller's working directory); an installed package falls back to pip's record.

    Returns:
        ``{"commit", "dirty", "source", "requested_revision"}`` where ``source`` is
        ``"git"``, ``"pip"`` or ``"unknown"``.
    """
    revision = git_revision(Path(__file__).resolve().parent)
    if revision["commit"]:
        return {**revision, "source": "git", "requested_revision": None}
    installed = installed_revision()
    if installed["commit"]:
        return {
            "commit": installed["commit"],
            "dirty": False,
            "source": "pip",
            "requested_revision": installed["requested_revision"],
        }
    return {"commit": None, "dirty": None, "source": "unknown", "requested_revision": None}


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
