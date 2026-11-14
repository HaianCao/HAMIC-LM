"""Source Registry: the gate every data source passes before it reaches the corpus.

The registry fails closed. A source whose review status does not allow the requested use
is rejected with :class:`SourceNotApprovedError`, and so is a source that is not
registered at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "SourceNotApprovedError",
    "SourceRecord",
    "SourceRegistry",
    "SourceStatus",
    "UseCase",
]


class SourceStatus(str, Enum):
    """Review status of a data source."""

    PENDING = "pending"
    APPROVED_RESEARCH = "approved_research"
    APPROVED_COMMERCIAL = "approved_commercial"
    REJECTED = "rejected"
    REVOKED = "revoked"


class UseCase(str, Enum):
    """Intended use of the data. HAMIC is currently research-only."""

    RESEARCH = "research"
    COMMERCIAL = "commercial"


_ALLOWED: Mapping[UseCase, frozenset[SourceStatus]] = {
    UseCase.RESEARCH: frozenset(
        {SourceStatus.APPROVED_RESEARCH, SourceStatus.APPROVED_COMMERCIAL}
    ),
    UseCase.COMMERCIAL: frozenset({SourceStatus.APPROVED_COMMERCIAL}),
}


class SourceNotApprovedError(PermissionError):
    """Raised when a source may not be used for the requested purpose."""


@dataclass(frozen=True)
class SourceRecord:
    """One entry of the Source Registry.

    Attributes:
        source_id: Stable identifier, e.g. ``"wikipedia-vi"``.
        publisher: Owner or publisher of the content.
        location: Dataset id, URL or object path.
        revision: Immutable snapshot, config or commit of the source.
        license: License identifier or a short description of the terms.
        status: Review status.
        reviewer: Person or team who made the decision (empty while pending).
        reviewed_on: Date of the decision.
        rationale: Why the decision was made.
        attribution: Attribution text that releases must carry, if any.
        pii_risk: ``"low"``, ``"medium"`` or ``"high"``.
        notes: Free-form notes.
    """

    source_id: str
    publisher: str
    location: str
    revision: str
    license: str
    status: SourceStatus = SourceStatus.PENDING
    reviewer: str = ""
    reviewed_on: dt.date | None = None
    rationale: str = ""
    attribution: str = ""
    pii_risk: str = "medium"
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.pii_risk not in {"low", "medium", "high"}:
            raise ValueError(f"pii_risk must be low/medium/high, got {self.pii_risk!r}")
        approved = self.status in {
            SourceStatus.APPROVED_RESEARCH,
            SourceStatus.APPROVED_COMMERCIAL,
        }
        if approved and (not self.reviewer or self.reviewed_on is None):
            raise ValueError(f"{self.source_id}: an approved source needs reviewer and date")

    def allows(self, use: UseCase) -> bool:
        """Return whether this source may be used for ``use``."""
        return self.status in _ALLOWED[use]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SourceRecord:
        """Build a record from a YAML/JSON mapping."""
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        values = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        values["status"] = SourceStatus(values.get("status", SourceStatus.PENDING.value))
        reviewed_on = values.get("reviewed_on")
        if isinstance(reviewed_on, str):
            values["reviewed_on"] = dt.date.fromisoformat(reviewed_on)
        return cls(**values, extra=extra)


class SourceRegistry:
    """A collection of :class:`SourceRecord` loaded from a YAML file."""

    def __init__(self, records: Mapping[str, SourceRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SourceRegistry:
        """Load a registry file whose top-level key ``sources`` is a list of records."""
        with Path(path).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        records: dict[str, SourceRecord] = {}
        for item in document.get("sources", []):
            record = SourceRecord.from_mapping(item)
            if record.source_id in records:
                raise ValueError(f"Duplicate source_id in registry: {record.source_id}")
            records[record.source_id] = record
        return cls(records)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._records

    def __iter__(self) -> Iterator[SourceRecord]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def get(self, source_id: str) -> SourceRecord:
        """Return the record for ``source_id`` or raise :class:`KeyError`."""
        return self._records[source_id]

    def require(self, source_id: str, use: UseCase = UseCase.RESEARCH) -> SourceRecord:
        """Return the record if it is approved for ``use``; otherwise fail closed.

        Raises:
            SourceNotApprovedError: If the source is unknown or not approved for ``use``.
        """
        record = self._records.get(source_id)
        if record is None:
            raise SourceNotApprovedError(f"Source {source_id!r} is not in the Source Registry")
        if not record.allows(use):
            raise SourceNotApprovedError(
                f"Source {source_id!r} has status {record.status.value!r}, "
                f"which does not allow {use.value} use"
            )
        return record
