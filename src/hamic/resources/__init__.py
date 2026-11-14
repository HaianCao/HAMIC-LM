"""Versioned data files shipped with the package (teencode lexicon, tokenizer probes)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

__all__ = ["Resource", "load_resource"]


@dataclass(frozen=True)
class Resource:
    """A parsed data file and its identity.

    Attributes:
        name: File name inside ``hamic.resources``.
        version: ``version`` field of the file.
        sha256: Hash of the raw bytes; reports record it so results can be traced to the
            exact data that produced them.
        data: Parsed YAML content.
    """

    name: str
    version: int
    sha256: str
    data: dict[str, Any]

    def identity(self) -> dict[str, Any]:
        """Return ``{"name", "version", "sha256"}`` for reports."""
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


def load_resource(name: str) -> Resource:
    """Load ``name`` (e.g. ``"teencode.yaml"``) from the package."""
    raw = resources.files(__name__).joinpath(name).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"Resource {name} must be a mapping with a 'version' field")
    return Resource(
        name=name, version=int(data["version"]), sha256=hashlib.sha256(raw).hexdigest(), data=data
    )
