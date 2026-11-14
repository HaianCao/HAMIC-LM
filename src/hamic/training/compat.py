"""Compatibility gate for strict resume.

Strict resume continues the same run, so code and configuration must be the ones the
checkpoint was written with. Anything else is a warm start and needs a new run id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["ResumeCompatibilityError", "check_strict_resume"]


class ResumeCompatibilityError(RuntimeError):
    """The checkpoint was produced by different code or configuration."""


def _commit(metadata: Mapping[str, Any]) -> str | None:
    revision = metadata.get("code_revision") or {}
    commit = revision.get("commit") if isinstance(revision, Mapping) else None
    return str(commit) if commit else None


def check_strict_resume(checkpoint: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Refuse a strict resume across code or configuration changes.

    Args:
        checkpoint: Metadata stored in the checkpoint manifest.
        current: Metadata of the process that wants to resume.

    Returns:
        Warnings for checks that could not be made (e.g. unknown commit).

    Raises:
        ResumeCompatibilityError: If the configuration hash or the code commit differs.
    """
    problems: list[str] = []
    warnings: list[str] = []
    if checkpoint.get("config_sha256") != current.get("config_sha256"):
        problems.append("configuration changed")
    saved, running = _commit(checkpoint), _commit(current)
    if saved and running and saved != running:
        problems.append(f"code changed ({saved[:12]} -> {running[:12]})")
    elif not saved or not running:
        warnings.append("code commit unknown on one side; continuing without a code check")
    if problems:
        raise ResumeCompatibilityError(
            "Cannot strictly resume: "
            + "; ".join(problems)
            + ". Install the checkpoint's commit, or start a new run id (warm start)."
        )
    return warnings
