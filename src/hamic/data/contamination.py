"""Benchmark contamination: find corpus text that repeats downstream evaluation text.

Text is compared as lower-cased NFC syllables (runs of word characters), so spacing,
punctuation and case do not hide a copy. A text is contaminated when it equals a
benchmark text, or shares a run of ``ngram`` syllables with one. N-grams are kept as
64-bit hashes, so the index of all HAMIC downstream tasks fits in a few hundred MB.

Two uses, two rules:

- rows of social-media sources, checked before they are grouped into documents
  (:class:`hamic.data.documents.GroupedSource`): every split of every task, exact match or
  any shared 8-gram. The ViSoBERT corpus contains ViHSD and VSMEC comments verbatim;
- documents (:class:`hamic.data.stages.ContaminationFilterStage`): the dev and test splits,
  13-grams, and only when the document covers at least half of the n-grams of one
  benchmark text. A single shared n-gram is not enough: the national motto heading every
  administrative document ("Cộng hòa xã hội chủ nghĩa Việt Nam, Độc lập - Tự do - Hạnh
  phúc") also occurs in benchmark comments.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path

__all__ = ["BenchmarkIndex", "benchmark_syllables", "read_benchmark_texts"]

_SYLLABLE = re.compile(r"\w+")


def benchmark_syllables(text: str) -> list[str]:
    """Lower-cased NFC syllables of ``text`` (the unit of every comparison)."""
    return _SYLLABLE.findall(unicodedata.normalize("NFC", text).lower())


def _key(parts: Iterable[str]) -> int:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def read_benchmark_texts(path: str | Path, splits: Collection[str] | None = None) -> Iterator[str]:
    """Texts of a benchmark file written by ``hamic corpus benchmark-texts``.

    Args:
        path: The file.
        splits: Keep only these splits (``"train"``, ``"dev"``, ``"test"``); all by default.
    """
    with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if splits is None or row["split"] in splits:
                    yield str(row["text"])


class BenchmarkIndex:
    """Exact texts and ``ngram``-syllable runs of a set of benchmark texts.

    Args:
        texts: Benchmark texts.
        ngram: Length of the shared run that marks contamination.
        min_exact_syllables: Shortest benchmark text matched exactly (shorter ones, such
            as "ok" or "hay quá", are ordinary phrases, not copies).
        min_coverage: ``None``: any shared n-gram is a match. Otherwise the share of one
            benchmark text's n-grams a text must contain.
    """

    def __init__(
        self,
        texts: Iterable[str],
        ngram: int,
        min_exact_syllables: int = 4,
        min_coverage: float | None = None,
    ) -> None:
        if ngram < 2:
            raise ValueError("ngram must be at least 2")
        if min_coverage is not None and not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        self.ngram = ngram
        self.min_exact_syllables = min_exact_syllables
        self.min_coverage = min_coverage
        self._exact: set[int] = set()
        self._grams: set[int] = set()
        self._owners: dict[int, list[int]] = {}  # with min_coverage: n-gram -> texts
        self._sizes: list[int] = []  # with min_coverage: distinct n-grams per text
        self.texts = 0
        for text in texts:
            syllables = benchmark_syllables(text)
            if not syllables:
                continue
            self.texts += 1
            if len(syllables) >= min_exact_syllables:
                self._exact.add(_key(syllables))
            grams = self._ngrams(syllables)
            if min_coverage is None:
                self._grams.update(grams)
            elif grams:
                for gram in grams:
                    self._owners.setdefault(gram, []).append(len(self._sizes))
                self._sizes.append(len(grams))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        ngram: int,
        splits: Collection[str] | None = None,
        min_coverage: float | None = None,
    ) -> BenchmarkIndex:
        """Index the texts of a benchmark file (optionally only some splits)."""
        return cls(read_benchmark_texts(path, splits), ngram, min_coverage=min_coverage)

    def _ngrams(self, syllables: list[str]) -> set[int]:
        n = self.ngram
        return {_key(syllables[i : i + n]) for i in range(len(syllables) - n + 1)}

    def __len__(self) -> int:
        return len(self._grams) + len(self._owners)

    def matches(self, text: str) -> bool:
        """Whether ``text`` equals a benchmark text or shares enough n-grams with one."""
        syllables = benchmark_syllables(text)
        if not syllables:
            return False
        if len(syllables) >= self.min_exact_syllables and _key(syllables) in self._exact:
            return True
        if len(syllables) < self.ngram:
            return False
        grams = self._ngrams(syllables)
        if self.min_coverage is None:
            return not self._grams.isdisjoint(grams)
        hits: Counter[int] = Counter()
        for gram in grams:
            hits.update(self._owners.get(gram, ()))
        return any(n >= self.min_coverage * self._sizes[t] for t, n in hits.items())
