"""Composable corpus-cleaning stages.

Every stage consumes and yields :class:`~hamic.data.documents.Document` objects and keeps
:class:`StageStats`, so the pipeline report can show how many documents each stage kept
and why the rest were rejected. Order matters and is fixed by the pipeline:
normalize -> quality filter -> exact dedup -> near dedup -> contamination -> split.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from hamic.data.documents import Document
from hamic.text.normalization import VietnameseNormalizer

__all__ = [
    "ContaminationFilterStage",
    "ExactDedupStage",
    "NearDedupStage",
    "NormalizeStage",
    "QualityFilterConfig",
    "QualityFilterStage",
    "SplitConfig",
    "SplitStage",
    "Stage",
    "StageStats",
]

_SYLLABLE = re.compile(r"[\w]+", flags=re.UNICODE)
_VIETNAMESE_MARKED = re.compile("[đĐăĂâÂêÊôÔơƠưƯ]")


def _has_vietnamese_marks(token: str) -> bool:
    if _VIETNAMESE_MARKED.search(token):
        return True
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", token))


@dataclass
class StageStats:
    """Counters for one stage."""

    seen: int = 0
    kept: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly view."""
        return {"seen": self.seen, "kept": self.kept, "rejected": dict(self.rejected)}


class Stage(ABC):
    """Base class for pipeline stages."""

    name: str = "stage"

    def __init__(self, audit_samples: int = 20, seed: int = 0) -> None:
        self.stats = StageStats()
        self._audit_limit = audit_samples
        self._rng = random.Random(seed)
        self.rejected_samples: dict[str, list[Document]] = {}

    def __call__(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Run the stage over ``documents``."""
        for document in documents:
            self.stats.seen += 1
            result = self.process(document)
            if isinstance(result, Document):
                self.stats.kept += 1
                yield result
            else:
                self._reject(document, result)

    @abstractmethod
    def process(self, document: Document) -> Document | str:
        """Return the (possibly modified) document, or a rejection reason string."""

    def _reject(self, document: Document, reason: str) -> None:
        self.stats.rejected[reason] += 1
        bucket = self.rejected_samples.setdefault(reason, [])
        count = self.stats.rejected[reason]
        if len(bucket) < self._audit_limit:
            bucket.append(document)
        else:  # reservoir sampling keeps an unbiased audit sample
            slot = self._rng.randrange(count)
            if slot < self._audit_limit:
                bucket[slot] = document


class NormalizeStage(Stage):
    """Apply :class:`VietnameseNormalizer` and drop documents that become empty."""

    name = "normalize"

    def __init__(self, normalizer: VietnameseNormalizer | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._normalizer = normalizer or VietnameseNormalizer()

    def process(self, document: Document) -> Document | str:
        """Normalize the text."""
        text = self._normalizer(document.text)
        return document.with_text(text) if text else "empty_after_normalization"


@dataclass(frozen=True)
class QualityFilterConfig:
    """Thresholds for :class:`QualityFilterStage`.

    The defaults are provisional and must be validated by a human audit of kept and
    rejected samples. They are calibrated for Vietnamese: the mean *syllable* length of
    normal Vietnamese prose is about 3 characters, so the English-centric rule
    "mean word length 3-10" would reject ordinary sentences such as
    "Hôm nay tôi đi ăn phở với mẹ".
    """

    min_chars: int = 200
    min_letter_ratio: float = 0.6
    min_mean_syllable_chars: float = 1.5
    max_mean_syllable_chars: float = 8.0
    max_duplicate_line_ratio: float = 0.3


class QualityFilterStage(Stage):
    """Heuristic quality filter adapted to Vietnamese.

    Each kept document gets ``meta["vi_mark_ratio"]``: the share of syllables carrying a
    Vietnamese diacritic or letter. It is recorded for audit and mixture analysis, not
    used as a filter, because text without diacritics is real Vietnamese usage.
    """

    name = "quality_filter"

    def __init__(self, config: QualityFilterConfig | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.config = config or QualityFilterConfig()

    def process(self, document: Document) -> Document | str:
        """Apply the quality rules."""
        cfg = self.config
        text = document.text
        if len(text) < cfg.min_chars:
            return "too_short"
        visible = [c for c in text if not c.isspace()]
        letters = sum(1 for c in visible if c.isalpha())
        if not visible or letters / len(visible) < cfg.min_letter_ratio:
            return "low_letter_ratio"
        syllables = _SYLLABLE.findall(text)
        if not syllables:
            return "no_syllables"
        mean_len = sum(len(s) for s in syllables) / len(syllables)
        if not cfg.min_mean_syllable_chars <= mean_len <= cfg.max_mean_syllable_chars:
            return "abnormal_syllable_length"
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            duplicates = len(lines) - len(set(lines))
            if duplicates / len(lines) > cfg.max_duplicate_line_ratio:
                return "repeated_lines"
        marked = sum(1 for s in syllables if _has_vietnamese_marks(s))
        return document.with_meta(vi_mark_ratio=round(marked / len(syllables), 4))


class ExactDedupStage(Stage):
    """Drop documents whose normalized text was already seen."""

    name = "exact_dedup"

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._seen: dict[str, str] = {}

    def process(self, document: Document) -> Document | str:
        """Keep the first occurrence of each text hash."""
        digest = document.text_sha256
        if digest in self._seen:
            return "exact_duplicate"
        self._seen[digest] = document.doc_id
        return document


class NearDedupStage(Stage):
    """MinHash-LSH near-duplicate removal over lower-cased syllable n-grams.

    The first document of a near-duplicate cluster is kept and becomes the cluster id,
    which :class:`SplitStage` uses so that a cluster never spans two splits.
    """

    name = "near_dedup"

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 5,
        **kwargs: int,
    ) -> None:
        from datasketch import MinHashLSH

        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def _signature(self, text: str) -> object:
        from datasketch import MinHash

        syllables = _SYLLABLE.findall(text.lower())
        size = self.shingle_size
        shingles = (
            {" ".join(syllables[i : i + size]) for i in range(len(syllables) - size + 1)}
            if len(syllables) >= size
            else {" ".join(syllables)}
        )
        signature = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            signature.update(shingle.encode("utf-8"))
        return signature

    def process(self, document: Document) -> Document | str:
        """Keep the document unless it is near-identical to one already kept."""
        signature = self._signature(document.text)
        if self._lsh.query(signature):
            return "near_duplicate"
        self._lsh.insert(document.doc_id, signature)
        return document.with_meta(cluster_id=document.doc_id)


class ContaminationFilterStage(Stage):
    """Drop documents sharing a long n-gram with any benchmark text."""

    name = "contamination"

    def __init__(self, benchmark_texts: Iterable[str], ngram: int = 13, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.ngram = ngram
        self._index = {g for text in benchmark_texts for g in self._ngrams(text)}

    def _ngrams(self, text: str) -> set[str]:
        tokens = _SYLLABLE.findall(unicodedata.normalize("NFC", text).lower())
        n = self.ngram
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def process(self, document: Document) -> Document | str:
        """Reject on any shared n-gram."""
        if self._index and not self._index.isdisjoint(self._ngrams(document.text)):
            return "benchmark_overlap"
        return document


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test fractions and the seed of the hash split."""

    validation: float = 0.01
    test: float = 0.01
    seed: str = "hamic-split-v1"

    def __post_init__(self) -> None:
        if self.validation < 0 or self.test < 0 or self.validation + self.test >= 1:
            raise ValueError("validation and test fractions must be >= 0 and sum to < 1")


class SplitStage(Stage):
    """Assign ``meta["split"]`` deterministically from the cluster id."""

    name = "split"

    def __init__(self, config: SplitConfig | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.config = config or SplitConfig()

    def assign(self, cluster_id: str) -> str:
        """Return ``"train"``, ``"validation"`` or ``"test"`` for ``cluster_id``."""
        digest = hashlib.sha256(f"{self.config.seed}:{cluster_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        if bucket < self.config.test:
            return "test"
        if bucket < self.config.test + self.config.validation:
            return "validation"
        return "train"

    def process(self, document: Document) -> Document | str:
        """Tag the document with its split."""
        cluster_id = str(document.meta.get("cluster_id", document.doc_id))
        return document.with_meta(split=self.assign(cluster_id))
