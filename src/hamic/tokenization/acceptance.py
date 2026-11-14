"""Tokenizer acceptance suite.

Metrics are reported per input, never as a single "best tokenizer" score. Fertility is
normalized by syllables, characters and UTF-8 bytes so tokenizers with different
vocabularies can be compared on the same text.
"""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Protocol

from hamic.tokenization.probes import PROBES

__all__ = ["AcceptanceReport", "CorpusMetrics", "EncodedText", "TokenizerEvaluator"]


@dataclass(frozen=True)
class EncodedText:
    """Tokenizer output needed by the suite."""

    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]


class _Encoder(Protocol):
    def encode(self, text: str) -> EncodedText: ...
    def decode(self, ids: list[int]) -> str: ...
    def token_bytes(self, token_id: int) -> bytes: ...


class _TokenizersAdapter:
    """Adapts a ``tokenizers.Tokenizer`` (no special tokens added) to the suite."""

    def __init__(self, tokenizer: Any) -> None:
        from tokenizers import decoders

        self._tok = tokenizer
        self._byte_decoder = decoders.ByteLevel()

    def encode(self, text: str) -> EncodedText:
        enc = self._tok.encode(text, add_special_tokens=False)
        return EncodedText(list(enc.ids), list(enc.tokens), [tuple(o) for o in enc.offsets])

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=False))

    def token_bytes(self, token_id: int) -> bytes:
        token = self._tok.id_to_token(token_id)
        mapping = _byte_level_inverse()
        return bytes(mapping[ch] for ch in token if ch in mapping)


def _byte_level_inverse() -> dict[str, int]:
    # GPT-2 byte-to-unicode table, inverted.
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, codes, strict=True)}


def _percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class CorpusMetrics:
    """Metrics over held-out documents."""

    documents: int = 0
    tokens: int = 0
    syllables: int = 0
    characters: int = 0
    utf8_bytes: int = 0
    roundtrip_failures: int = 0
    nfd_mismatches: int = 0
    offset_errors: int = 0
    partial_byte_tokens: int = 0
    tokens_per_doc_p50: float = 0.0
    tokens_per_doc_p95: float = 0.0
    tokens_per_second: float = 0.0

    @property
    def tokens_per_syllable(self) -> float:
        """Fertility per whitespace-separated syllable."""
        return self.tokens / self.syllables if self.syllables else 0.0

    @property
    def chars_per_token(self) -> float:
        """Characters covered by one token on average."""
        return self.characters / self.tokens if self.tokens else 0.0

    @property
    def bytes_per_token(self) -> float:
        """UTF-8 bytes covered by one token on average."""
        return self.utf8_bytes / self.tokens if self.tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view including derived ratios."""
        data = asdict(self)
        data.update(
            tokens_per_syllable=round(self.tokens_per_syllable, 4),
            chars_per_token=round(self.chars_per_token, 4),
            bytes_per_token=round(self.bytes_per_token, 4),
        )
        return data


@dataclass
class AcceptanceReport:
    """Full acceptance report for one tokenizer."""

    tokenizer_name: str
    vocab_size: int
    corpus: CorpusMetrics
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed_hard_gates(self) -> bool:
        """Round-trip, NFC/NFD invariance and offsets must be perfect."""
        c = self.corpus
        return c.roundtrip_failures == 0 and c.nfd_mismatches == 0 and c.offset_errors == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        return {
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "passed_hard_gates": self.passed_hard_gates,
            "corpus": self.corpus.to_dict(),
            "probes": self.probes,
        }


class TokenizerEvaluator:
    """Run the acceptance suite against a ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._enc: _Encoder = _TokenizersAdapter(tokenizer)
        self._vocab_size = int(tokenizer.get_vocab_size())
        self._name = name

    @staticmethod
    def _check_offsets(text: str, encoded: EncodedText) -> int:
        """Count offset defects.

        Byte-level tokens that split one multi-byte character legitimately share or
        overlap character spans, so overlap is allowed. A defect is a span outside the
        text, a start that moves backwards, or a non-space character no span covers
        (which would silently shift NER/QA labels).
        """
        errors = 0
        covered = [False] * len(text)
        previous_start = 0
        for start, end in encoded.offsets:
            if not 0 <= start <= end <= len(text) or start < previous_start:
                errors += 1
                continue
            previous_start = start
            for index in range(start, end):
                covered[index] = True
        errors += sum(1 for i, ch in enumerate(text) if not ch.isspace() and not covered[i])
        return errors

    def evaluate_corpus(self, texts: Iterable[str]) -> CorpusMetrics:
        """Measure fertility, invariances and offsets on held-out ``texts``."""
        metrics = CorpusMetrics()
        per_doc: list[int] = []
        elapsed = 0.0
        for raw in texts:
            text = unicodedata.normalize("NFC", raw)
            started = time.perf_counter()
            encoded = self._enc.encode(text)
            elapsed += time.perf_counter() - started
            metrics.documents += 1
            metrics.tokens += len(encoded.ids)
            metrics.syllables += len(text.split())
            metrics.characters += len(text)
            metrics.utf8_bytes += len(text.encode("utf-8"))
            per_doc.append(len(encoded.ids))
            if self._enc.decode(encoded.ids) != text:
                metrics.roundtrip_failures += 1
            if self._enc.encode(unicodedata.normalize("NFD", text)).ids != encoded.ids:
                metrics.nfd_mismatches += 1
            metrics.offset_errors += self._check_offsets(text, encoded)
            metrics.partial_byte_tokens += sum(
                1 for i in encoded.ids if not self._is_complete_utf8(i)
            )
        metrics.tokens_per_doc_p50 = _percentile(per_doc, 0.5)
        metrics.tokens_per_doc_p95 = _percentile(per_doc, 0.95)
        metrics.tokens_per_second = metrics.tokens / elapsed if elapsed else 0.0
        return metrics

    def _is_complete_utf8(self, token_id: int) -> bool:
        try:
            self._enc.token_bytes(token_id).decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    def evaluate_probes(
        self, probes: Mapping[str, Sequence[str]] = PROBES
    ) -> dict[str, dict[str, Any]]:
        """Return per-category fertility and examples of how probes are split."""
        result: dict[str, dict[str, Any]] = {}
        for category, items in probes.items():
            ratios: list[float] = []
            examples: list[dict[str, Any]] = []
            roundtrip_ok = True
            for item in items:
                text = unicodedata.normalize("NFC", item)
                encoded = self._enc.encode(text)
                syllables = max(1, len(text.split()))
                ratios.append(len(encoded.ids) / syllables)
                roundtrip_ok &= self._enc.decode(encoded.ids) == text
                pieces = [text[s:e] for s, e in encoded.offsets]
                examples.append({"text": text, "tokens": len(encoded.ids), "pieces": pieces})
            result[category] = {
                "tokens_per_syllable": round(mean(ratios), 3) if ratios else 0.0,
                "roundtrip_ok": roundtrip_ok,
                "examples": examples,
            }
        return result

    def run(self, texts: Iterable[str]) -> AcceptanceReport:
        """Run corpus metrics and probes."""
        return AcceptanceReport(
            tokenizer_name=self._name,
            vocab_size=self._vocab_size,
            corpus=self.evaluate_corpus(texts),
            probes=self.evaluate_probes(),
        )
