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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from hamic.data.documents import Document
from hamic.text.normalization import VietnameseNormalizer

__all__ = [
    "ContaminationFilterStage",
    "ExactDedupStage",
    "NearDedupStage",
    "NormalizeStage",
    "PiiFilterStage",
    "QualityFilterConfig",
    "QualityFilterStage",
    "SplitConfig",
    "SplitStage",
    "Stage",
    "StageStats",
    "minhash_signature",
]

_SYLLABLE = re.compile(r"[\w]+", flags=re.UNICODE)
_LETTER = re.compile(r"[^\W\d_]")
_WHITESPACE = re.compile(r"\s")


def _marked_letter_class() -> str:
    """Regex class of Latin letters carrying a diacritic, plus combining marks and đ/Đ."""
    chars = {"đ", "Đ"}
    for code in (*range(0x00C0, 0x0250), *range(0x1E00, 0x1F00)):
        char = chr(code)
        if char.isalpha() and any(
            unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", char)
        ):
            chars.add(char)
    return "[" + "".join(sorted(chars)) + "̀-ͯ]"


_MARKED = re.compile(_marked_letter_class())


def minhash_signature(text: str, num_perm: int = 128, shingle_size: int = 5) -> Any:
    """Return MinHash hash values of lower-cased syllable shingles of ``text``.

    A pure function, so it can run in worker processes; the result feeds
    :class:`NearDedupStage` through :attr:`NearDedupStage.precomputed`.
    """
    from datasketch import MinHash

    syllables = _SYLLABLE.findall(text.lower())
    size = shingle_size
    shingles = (
        {" ".join(syllables[i : i + size]) for i in range(len(syllables) - size + 1)}
        if len(syllables) >= size
        else {" ".join(syllables)}
    )
    signature = MinHash(num_perm=num_perm)
    signature.update_batch([s.encode("utf-8") for s in shingles])
    return signature.hashvalues


@dataclass
class StageStats:
    """Counters for one stage."""

    seen: int = 0
    kept: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    modified: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly view."""
        return {
            "seen": self.seen,
            "kept": self.kept,
            "rejected": dict(self.rejected),
            "modified": dict(self.modified),
        }


class Stage(ABC):
    """Base class for pipeline stages.

    A stage with ``parallel_safe = True`` keeps no state between documents, so the
    pipeline may run it in worker processes and merge its statistics afterwards.
    """

    name: str = "stage"
    parallel_safe: bool = False

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
                self.record_kept(result)
                yield result
            else:
                self.record_rejected(document, result)

    @abstractmethod
    def process(self, document: Document) -> Document | str:
        """Return the (possibly modified) document, or a rejection reason string."""

    def modification_counts(self, document: Document) -> Mapping[str, int]:
        """Return what this stage changed in a kept ``document`` (for statistics)."""
        del document
        return {}

    def record_kept(self, document: Document) -> None:
        """Account for a document this stage kept (used by serial and parallel runs)."""
        self.stats.kept += 1
        self.stats.modified.update(self.modification_counts(document))

    def record_rejected(self, document: Document, reason: str) -> None:
        """Account for a document this stage rejected and keep an audit sample."""
        self._reject(document, reason)

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
    parallel_safe = True

    def __init__(self, normalizer: VietnameseNormalizer | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._normalizer = normalizer or VietnameseNormalizer()

    def process(self, document: Document) -> Document | str:
        """Normalize the text."""
        text = self._normalizer(document.text)
        return document.with_text(text) if text else "empty_after_normalization"


_SECRET_PATTERNS: dict[str, str] = {
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "openai_style_key": r"\bsk-[A-Za-z0-9_-]{32,}\b",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "private_key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
}
# One pass over the text instead of one per pattern; ``lastgroup`` names the kind.
_SECRETS = re.compile("|".join(f"(?P<{k}>{p})" for k, p in _SECRET_PATTERNS.items()))
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# Vietnamese numbers written with a leading 0 or +84 and optional space/dot/hyphen
# separators. Current plan: mobile 03/05/07/08/09 + 8 digits, landline 02 + 9 digits.
# Pre-2018 plan, still common in web text: mobile 01x + 8 digits (11 digits total),
# landline 02-08 + 8-9 digits. A number followed by ":" is a time, not a phone number.
_PHONE = re.compile(
    r"(?<![\w.,-])(?:\+84[ .-]?|0)"
    r"(?:[35789](?:[ .-]?\d){8}|1(?:[ .-]?\d){9}|[2-8](?:[ .-]?\d){8,9})"
    r"(?![\w:]|[.,-]?\d)"
)
# Book identifiers look like phone numbers ("ISBN 0-8018-...", "978-0-19-...").
_ISBN_CONTEXT = re.compile(r"(?:ISBN(?:-1[03])?:?\s*(?:97[89][ -]?)?|97[89][ -])$", re.IGNORECASE)
# IPv4 with valid octets and no leading zeros. Vietnamese writes money with dots as
# thousands separators ("1.500.000.000"); those groups are three digits and often
# "000" or above 255, so they do not match.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.]*\d)")


class PiiFilterStage(Stage):
    """Drop documents that contain credentials and anonymize common personal data.

    Credentials (API keys, tokens, private keys) cause the whole document to be dropped.
    E-mail addresses, Vietnamese phone numbers and IPv4 addresses are replaced with fixed
    dummy values, which keeps the text natural for tokenizer training. Counts of each
    replacement are reported in ``StageStats.modified`` and ``meta["pii_redactions"]``.

    This is a heuristic first pass, not a privacy guarantee: names, addresses and ID
    numbers are not detected and must be covered by the human audit.
    """

    name = "pii_filter"
    parallel_safe = True

    EMAIL_REPLACEMENT = "email@example.com"
    PHONE_REPLACEMENT = "0900000000"
    IP_REPLACEMENT = "0.0.0.0"

    def process(self, document: Document) -> Document | str:
        """Reject on secrets, otherwise redact personal data."""
        text = document.text
        secret = _SECRETS.search(text)
        if secret:
            return f"secret_{secret.lastgroup}"
        counts: dict[str, int] = {}
        for kind, pattern, replacement in (
            ("email", _EMAIL, self.EMAIL_REPLACEMENT),
            ("phone", _PHONE, self.PHONE_REPLACEMENT),
            ("ipv4", _IPV4, self.IP_REPLACEMENT),
        ):
            text, found = self._redact(text, pattern, replacement, isbn_guard=kind == "phone")
            if found:
                counts[kind] = found
        if not counts:
            return document
        return document.with_text(text).with_meta(pii_redactions=counts)

    def modification_counts(self, document: Document) -> Mapping[str, int]:
        """Redactions made in ``document``."""
        counts: Mapping[str, int] = document.meta.get("pii_redactions", {})
        return counts

    @staticmethod
    def _redact(
        text: str, pattern: re.Pattern[str], replacement: str, isbn_guard: bool
    ) -> tuple[str, int]:
        found = 0

        def substitute(match: re.Match[str]) -> str:
            nonlocal found
            before = text[max(0, match.start() - 16) : match.start()]
            if isbn_guard and _ISBN_CONTEXT.search(before):
                return match.group(0)
            found += 1
            return replacement

        return pattern.sub(substitute, text), found


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
    parallel_safe = True

    def __init__(self, config: QualityFilterConfig | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.config = config or QualityFilterConfig()

    def process(self, document: Document) -> Document | str:
        """Apply the quality rules."""
        cfg = self.config
        text = document.text
        if len(text) < cfg.min_chars:
            return "too_short"
        visible = len(text) - len(_WHITESPACE.findall(text))
        letters = len(_LETTER.findall(text))
        if not visible or letters / visible < cfg.min_letter_ratio:
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
        marked = sum(1 for s in syllables if _MARKED.search(s))
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
        from datasketch import MinHash, MinHashLSH

        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        reference = MinHash(num_perm=num_perm)
        self._seed = reference.seed
        self._scheme = reference.scheme
        self.precomputed: dict[str, tuple[int, Any]] = {}
        """``doc_id -> (len(text), hash values)`` filled by parallel workers."""

    def _signature(self, document: Document) -> Any:
        # LeanMinHash rebuilds from hash values without regenerating permutations.
        from datasketch import LeanMinHash

        cached = self.precomputed.pop(document.doc_id, None)
        if cached is not None and cached[0] == len(document.text):
            values = cached[1]
        else:
            values = minhash_signature(document.text, self.num_perm, self.shingle_size)
        return LeanMinHash(seed=self._seed, hashvalues=values, scheme=self._scheme)

    def process(self, document: Document) -> Document | str:
        """Keep the document unless it is near-identical to one already kept."""
        signature = self._signature(document)
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
