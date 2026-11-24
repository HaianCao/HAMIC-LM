"""Corpus pipeline configuration: every threshold in one versioned file.

The manifest of each corpus records the full configuration and its hash, so a corpus can
always be traced to the exact thresholds that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from hamic.data.stages import (
    ContaminationFilterStage,
    ExactDedupStage,
    IndexedExactDedupStage,
    IndexedNearDedupStage,
    NearDedupStage,
    NormalizeStage,
    PiiFilterStage,
    QualityFilterConfig,
    QualityFilterStage,
    SplitConfig,
    SplitStage,
    Stage,
)
from hamic.provenance import json_sha256
from hamic.text.normalization import NormalizationConfig, VietnameseNormalizer

if TYPE_CHECKING:
    from hamic.data.dedup_index import DedupIndex

__all__ = ["ContaminationConfig", "NearDedupConfig", "PipelineConfig"]


@dataclass(frozen=True)
class NearDedupConfig:
    """MinHash-LSH settings."""

    threshold: float = 0.8
    num_perm: int = 128
    shingle_size: int = 5


@dataclass(frozen=True)
class ContaminationConfig:
    """Benchmark contamination removal (:mod:`hamic.data.contamination`).

    Attributes:
        benchmarks: File written by ``hamic corpus benchmark-texts``.
        sha256: Its checksum; the build refuses another file, so the manifest pins the
            exact benchmark texts.
        ngram: Syllable run compared between documents and benchmark texts.
        splits: Benchmark splits checked (train text is not evaluation data).
        min_coverage: A document is contaminated when it contains at least this share of
            the n-grams of one benchmark text (or equals it).
    """

    benchmarks: str
    sha256: str
    ngram: int = 13
    splits: tuple[str, ...] = ("dev", "test")
    min_coverage: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    """All settings of a corpus build.

    Attributes:
        name: Configuration id, recorded in the manifest.
        normalization: Text normalization options.
        pii_filter: Run the PII/secret stage.
        quality: Quality filter thresholds.
        near_dedup: MinHash-LSH settings.
        split: Train/validation/test fractions and seed.
        shard_docs: Documents per output shard.
        contamination: Drop documents repeating downstream benchmark text (``None``:
            off; left out of the hash, so older configurations keep theirs).
    """

    name: str = "default"
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    pii_filter: bool = True
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    near_dedup: NearDedupConfig = field(default_factory=NearDedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    shard_docs: int = 50_000
    contamination: ContaminationConfig | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build from a mapping; unknown keys raise ``TypeError`` instead of being ignored."""
        values = dict(data)
        values.pop("version", None)
        sections = {
            "normalization": NormalizationConfig,
            "quality": QualityFilterConfig,
            "near_dedup": NearDedupConfig,
            "split": SplitConfig,
            "contamination": ContaminationConfig,
        }
        for key, kind in sections.items():
            if key in values:
                values[key] = kind(**(values[key] or {}))
        contamination = values.get("contamination")
        if contamination is not None and not isinstance(contamination.splits, tuple):
            values["contamination"] = ContaminationConfig(
                **{**asdict(contamination), "splits": tuple(contamination.splits)}
            )
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a YAML configuration file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream) or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        data = asdict(self)
        # Options added after v0.12 stay out of the hash while unused, so earlier
        # configurations keep their hash.
        if data["contamination"] is None:
            del data["contamination"]
        if data["quality"]["max_legacy_font_share"] is None:
            del data["quality"]["max_legacy_font_share"]
        return data

    def sha256(self) -> str:
        """Hash of the configuration (independent of file formatting)."""
        return json_sha256(self.to_dict())

    def build_stages(self, dedup_index: DedupIndex | None = None) -> list[Stage]:
        """Return the canonical stage sequence configured by this object.

        Args:
            dedup_index: Persistent index of earlier corpus versions. When given, the
                dedup stages also drop duplicates of those versions and add the kept
                documents to the index (corpus deltas).
        """
        stages: list[Stage] = [NormalizeStage(VietnameseNormalizer(self.normalization))]
        if self.pii_filter:
            stages.append(PiiFilterStage())
        stages.append(QualityFilterStage(self.quality))
        if self.contamination is not None:
            stages.append(ContaminationFilterStage(self._benchmark_index()))
        if dedup_index is not None:
            expected = (self.near_dedup.num_perm, self.near_dedup.shingle_size)
            if (dedup_index.num_perm, dedup_index.shingle_size) != expected:
                raise ValueError("The dedup index was built with other MinHash settings")
            stages += [IndexedExactDedupStage(dedup_index), IndexedNearDedupStage(dedup_index)]
        else:
            stages += [
                ExactDedupStage(),
                NearDedupStage(
                    threshold=self.near_dedup.threshold,
                    num_perm=self.near_dedup.num_perm,
                    shingle_size=self.near_dedup.shingle_size,
                ),
            ]
        stages.append(SplitStage(self.split))
        return stages

    def _benchmark_index(self) -> Any:
        from hamic.data.contamination import BenchmarkIndex
        from hamic.provenance import file_sha256

        assert self.contamination is not None
        path = Path(self.contamination.benchmarks)
        if file_sha256(path) != self.contamination.sha256:
            raise ValueError(f"{path} does not match contamination.sha256")
        return BenchmarkIndex.from_file(
            path,
            self.contamination.ngram,
            splits=self.contamination.splits,
            min_coverage=self.contamination.min_coverage,
        )
