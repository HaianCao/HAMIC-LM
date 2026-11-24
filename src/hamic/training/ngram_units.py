"""PMI n-gram masking units (E6c) and whole-word joining.

:class:`~hamic.training.masking.PmiUnits` joins neighbours whose *pair* is a strong
collocation and chains such pairs, so "học sinh viên" becomes one unit when both
"học sinh" and "sinh viên" are listed. This module follows the original PMI-Masking
recipe (Levine et al., 2021) more closely:

* n-grams of 2 to ``max_n`` complete syllables are mined from the train split;
* an n-gram is scored by its **minimum PMI over every split into contiguous parts**,
  ``min_σ log p(w) / Π p(part)``, so a sequence is cohesive only if no coarser split
  explains it ("học sinh | viên" explains "học sinh viên");
* n-grams that occur in too few documents are dropped (boilerplate repeated inside one
  page is not a collocation);
* n-grams that are mostly a fragment of one longer sequence are dropped: when a single
  left or right extension accounts for at least ``max_extension_share`` of the
  occurrences ("lực lượng an" → "lực lượng an ninh", "Chí Minh" ← "Hồ Chí Minh"), the
  n-gram is not a unit of its own. This is the min-entropy form of the branching-entropy
  test used in unsupervised word discovery;
* at masking time, listed n-grams are matched longest first and never overlap.

Only *complete syllables* take part in n-grams: a letter-only token that starts a word
(leading space, or after a non-word token) and is not followed by a continuation piece.
Digits never take part ("năm 1844" is not a collocation worth masking whole).
Syllables split by the tokenizer ("tr" + "í") are handled by :class:`WholeWordUnits`,
which always keeps continuation pieces in the unit of the token before them.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hamic.training.masking import SelectionUnits, _units_from_starts, decoded_vocabulary

__all__ = [
    "NgramTables",
    "NgramUnits",
    "TokenClasses",
    "WholeWordUnits",
    "mine_pmi_ngrams",
]

MAX_N = 3
"""Longest unit. Counting goes one longer, to test extensions, and must fit int64 keys."""


@dataclass(frozen=True)
class TokenClasses:
    """Per-id flags derived from the tokenizer vocabulary.

    Attributes:
        word: Token is letters/digits only, with at most one leading space.
        spaced: Token text starts with a space.
        alpha: Token is letters only (n-gram members), with at most one leading space.
    """

    word: np.ndarray[Any, Any]
    spaced: np.ndarray[Any, Any]
    alpha: np.ndarray[Any, Any]

    @classmethod
    def from_tokenizer_file(cls, path: str | Path) -> TokenClasses:
        """Classify every id of a byte-level ``tokenizer.json``."""
        return cls.from_texts(decoded_vocabulary(path))

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> TokenClasses:
        """Classify decoded token texts (index = id)."""
        word = np.zeros(len(texts), dtype=bool)
        spaced = np.zeros(len(texts), dtype=bool)
        alpha = np.zeros(len(texts), dtype=bool)
        for index, text in enumerate(texts):
            core = text[1:] if text.startswith(" ") else text
            word[index] = bool(core) and core.isalnum()
            alpha[index] = bool(core) and core.isalpha()
            spaced[index] = text.startswith(" ")
        return cls(word=word, spaced=spaced, alpha=alpha)


def _continuation(ids: np.ndarray[Any, Any], classes: TokenClasses) -> np.ndarray[Any, Any]:
    """Positions that continue the word of the previous token."""
    word = classes.word[ids]
    cont: np.ndarray[Any, Any] = word & ~classes.spaced[ids]
    cont[1:] &= word[:-1]
    if cont.size:
        cont[0] = False
    return cont


def _complete_starts(
    ids: np.ndarray[Any, Any], classes: TokenClasses, special: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Positions holding a whole syllable (letter token, not continued, not continuing)."""
    cont = _continuation(ids, classes)
    complete: np.ndarray[Any, Any] = classes.alpha[ids] & ~cont & ~special[ids]
    complete[:-1] &= ~cont[1:]
    return complete


def _ngram_keys(ids: np.ndarray[Any, Any], n: int, base: int) -> np.ndarray[Any, Any]:
    keys = np.zeros(ids.size - n + 1, dtype=np.int64)
    for offset in range(n):
        keys = keys * base + ids[offset : ids.size - n + 1 + offset]
    return keys


def _valid_windows(complete: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
    valid: np.ndarray[Any, Any] = complete[: complete.size - n + 1].copy()
    for offset in range(1, n):
        valid &= complete[offset : complete.size - n + 1 + offset]
    return valid


class _Counter:
    """Exact (key, count, document count) totals with periodic in-memory compaction."""

    def __init__(self, compact_rows: int) -> None:
        self.keys: list[np.ndarray[Any, Any]] = []
        self.counts: list[np.ndarray[Any, Any]] = []
        self.docs: list[np.ndarray[Any, Any]] = []
        self.rows = 0
        self.compact_rows = compact_rows

    def add_document(self, keys: np.ndarray[Any, Any]) -> None:
        if keys.size == 0:
            return
        unique, counts = np.unique(keys, return_counts=True)
        self.keys.append(unique)
        self.counts.append(counts.astype(np.int64))
        self.docs.append(np.ones(unique.size, dtype=np.int64))
        self.rows += unique.size
        if self.rows >= self.compact_rows:
            self._compact()

    def _compact(self) -> None:
        if len(self.keys) <= 1:
            return
        keys = np.concatenate(self.keys)
        counts = np.concatenate(self.counts)
        docs = np.concatenate(self.docs)
        order = np.argsort(keys, kind="stable")
        keys, counts, docs = keys[order], counts[order], docs[order]
        first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        self.keys = [keys[first]]
        self.counts = [np.add.reduceat(counts, first)]
        self.docs = [np.add.reduceat(docs, first)]
        self.rows = first.size
        # Grow the threshold so compaction stays amortised when most keys are distinct.
        self.compact_rows = max(self.compact_rows, 2 * self.rows)

    def totals(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if not self.keys:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty
        self.compact_rows = 0
        self._compact()
        return self.keys[0], self.counts[0], self.docs[0]


def _compositions(n: int) -> list[tuple[int, ...]]:
    """Every split of ``n`` into at least two positive contiguous part lengths."""
    result = []
    for cuts in range(1, n):
        for positions in itertools.combinations(range(1, n), cuts):
            bounds = (0, *positions, n)
            result.append(tuple(b - a for a, b in itertools.pairwise(bounds)))
    return result


def _dominant_extension(
    keys: np.ndarray[Any, Any],
    counts: np.ndarray[Any, Any],
    longer: np.ndarray[Any, Any],
    longer_counts: np.ndarray[Any, Any],
    n: int,
    base: int,
) -> np.ndarray[Any, Any]:
    """Largest share of an n-gram's occurrences taken by one left or right neighbour."""
    dominant = np.zeros(keys.size)
    if longer.size == 0:
        return dominant
    # A longer key "w x" has prefix w (right extension); "x w" has suffix w (left one).
    for part in (longer // base, longer % base**n):
        order = np.argsort(part, kind="stable")
        part, part_counts = part[order], longer_counts[order]
        first = np.flatnonzero(np.r_[True, part[1:] != part[:-1]])
        groups, best = part[first], np.maximum.reduceat(part_counts, first)
        where = np.searchsorted(groups, keys).clip(max=groups.size - 1)
        hit = groups[where] == keys
        dominant[hit] = np.maximum(dominant[hit], best[where[hit]] / counts[hit])
    return dominant


@dataclass(frozen=True)
class NgramTables:
    """Mined n-gram keys per length (sorted ``int64``, ids packed in base ``vocab_size``)."""

    vocab_size: int
    tables: dict[int, np.ndarray[Any, Any]]

    def save(self, path: str | Path) -> None:
        """Write an ``.npz`` file."""
        arrays: dict[str, Any] = {f"n{n}": keys for n, keys in self.tables.items()}
        arrays["vocab_size"] = np.int64(self.vocab_size)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NgramTables:
        """Read a file written by :meth:`save`."""
        with np.load(path) as data:
            tables = {int(name[1:]): data[name] for name in data.files if name.startswith("n")}
            return cls(vocab_size=int(data["vocab_size"]), tables=tables)

    def __len__(self) -> int:
        return sum(keys.size for keys in self.tables.values())

    def decode(self, n: int, key: int) -> list[int]:
        """Token ids of one packed key."""
        ids = []
        for _ in range(n):
            key, rest = divmod(key, self.vocab_size)
            ids.append(rest)
        return ids[::-1]


def mine_pmi_ngrams(
    documents: Iterable[np.ndarray[Any, Any]],
    vocab_size: int,
    special_ids: Iterable[int],
    classes: TokenClasses,
    max_n: int = 3,
    max_tokens: int = 100_000_000,
    min_count: int = 50,
    min_doc_count: int = 5,
    min_pmi: float = 3.0,
    max_units: int = 200_000,
    discount_k: float = 0.0,
    max_extension_share: float = 0.6,
    compact_rows: int = 20_000_000,
) -> NgramTables:
    """Mine cohesive n-grams of complete syllables.

    Args:
        documents: Token ids per document, read until ``max_tokens`` are seen.
        vocab_size: Vocabulary size (key base).
        special_ids: Ids that never take part.
        classes: Word/space flags of the vocabulary (:class:`TokenClasses`).
        max_n: Longest n-gram (2 to :data:`MAX_N`).
        max_tokens: Tokens to read.
        min_count: Fewest occurrences of a kept n-gram.
        min_doc_count: Fewest distinct documents containing a kept n-gram.
        min_pmi: Lowest minimum-over-splits PMI (nats) of a kept n-gram.
        max_units: Most n-grams kept over all lengths, best score first.
        discount_k: Ranking uses ``PMI * count / (count + k)`` when positive.
        max_extension_share: Drop an n-gram when one left or right neighbour follows it in
            at least this share of its occurrences (``1.0`` disables the test).
        compact_rows: Rows buffered before an in-memory reduction.

    Returns:
        The kept n-grams per length.
    """
    if not 2 <= max_n <= MAX_N:
        raise ValueError(f"max_n must be in [2, {MAX_N}]")
    if vocab_size ** (max_n + 1) >= 2**63:
        raise ValueError("vocab_size ** (max_n + 1) does not fit in int64 keys")
    if min(max_tokens, min_count, min_doc_count, max_units) < 1 or discount_k < 0:
        raise ValueError("counts must be positive and discount_k non-negative")
    if not 0 < max_extension_share <= 1:
        raise ValueError("max_extension_share must be in (0, 1]")
    count_n = max_n + 1
    special = np.zeros(vocab_size, dtype=bool)
    special[list(set(special_ids))] = True
    counters = {n: _Counter(compact_rows) for n in range(1, count_n + 1)}
    windows = dict.fromkeys(range(1, count_n + 1), 0)
    seen = 0
    for document in documents:
        ids = np.asarray(document, dtype=np.int64)
        if ids.size == 0:
            continue
        complete = _complete_starts(ids, classes, special)
        for n in range(1, count_n + 1):
            if ids.size < n:
                break
            valid = _valid_windows(complete, n)
            windows[n] += int(valid.sum())
            counters[n].add_document(_ngram_keys(ids, n, vocab_size)[valid])
        seen += ids.size
        if seen >= max_tokens:
            break
    totals = {n: counters[n].totals() for n in counters}
    log_p = {}
    for n, (keys, counts, _) in totals.items():
        log_p[n] = (keys, np.log(np.maximum(counts, 1)) - np.log(max(windows[n], 1)))

    def lookup(n: int, keys: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Log p of each packed key; ``inf`` when unseen (the split then rules it out)."""
        table, values = log_p[n]
        result = np.full(keys.size, np.inf)
        if table.size:
            where = np.searchsorted(table, keys).clip(max=table.size - 1)
            hit = table[where] == keys
            result[hit] = values[where[hit]]
        return result

    kept_keys, kept_n, kept_score = [], [], []
    for n in range(2, max_n + 1):
        keys, counts, docs = totals[n]
        keep = (counts >= min_count) & (docs >= min_doc_count)
        keys, counts = keys[keep], counts[keep]
        if max_extension_share < 1:
            dominant = _dominant_extension(keys, counts, *totals[n + 1][:2], n, vocab_size)
            free = dominant < max_extension_share
            keys, counts = keys[free], counts[free]
        if keys.size == 0:
            continue
        whole = log_p[n][1][np.searchsorted(log_p[n][0], keys)]
        pmi = np.full(keys.size, np.inf)
        for parts in _compositions(n):
            parts_log_p = np.zeros(keys.size)
            end = n
            for length in reversed(parts):
                start = end - length
                part = (keys // vocab_size ** (n - end)) % vocab_size**length
                parts_log_p += lookup(length, part)
                end = start
            pmi = np.minimum(pmi, whole - parts_log_p)
        strong = pmi >= min_pmi
        score = pmi if discount_k == 0 else pmi * counts / (counts + discount_k)
        kept_keys.append(keys[strong])
        kept_n.append(np.full(int(strong.sum()), n))
        kept_score.append(score[strong])
    tables: dict[int, np.ndarray[Any, Any]] = {
        n: np.zeros(0, dtype=np.int64) for n in range(2, max_n + 1)
    }
    if kept_keys:
        keys = np.concatenate(kept_keys)
        lengths = np.concatenate(kept_n)
        score = np.concatenate(kept_score)
        best = np.lexsort((keys, lengths, -score))[:max_units]
        for n in tables:
            tables[n] = np.sort(keys[best][lengths[best] == n])
    return NgramTables(vocab_size=vocab_size, tables=tables)


def _torch_continuation(
    ids: torch.Tensor, word: torch.Tensor, spaced: torch.Tensor
) -> torch.Tensor:
    is_word = word[ids]
    cont = is_word & ~spaced[ids]
    cont[:, 1:] &= is_word[:, :-1]
    cont[:, 0] = False
    return cont


class WholeWordUnits:
    """Wrap other units so word-continuation pieces join the unit before them.

    Args:
        inner: Units to refine (token, span or PMI units).
        classes: Word/space flags of the vocabulary.
        special_ids: Ids that never join.
    """

    def __init__(
        self, inner: SelectionUnits, classes: TokenClasses, special_ids: Iterable[int]
    ) -> None:
        self.inner = inner
        word = classes.word.copy()
        word[list(set(special_ids))] = False
        self.word = torch.from_numpy(word)
        self.spaced = torch.from_numpy(classes.spaced.copy())

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Inner units, with continuation positions merged into the preceding unit."""
        unit = self.inner(ids, generator)
        ids = ids.long()
        starts = torch.ones_like(ids, dtype=torch.bool)
        starts[:, 1:] = unit[:, 1:] != unit[:, :-1]
        starts &= ~_torch_continuation(ids, self.word, self.spaced)
        return _units_from_starts(starts)


class NgramUnits:
    """Mask listed n-grams of complete syllables as one unit, longest match first.

    Each n-gram occurrence joins with probability ``keep`` (drawn before matching, so a
    skipped long n-gram lets a shorter one inside it match). Overlapping matches of the
    same length keep the leftmost; a match whose predecessor was itself dropped may also
    be dropped, which slightly lowers coverage but never produces overlapping units.
    Continuation pieces always stay with their word.

    Args:
        tables: Mined n-grams (:func:`mine_pmi_ngrams`).
        classes: Word/space flags of the vocabulary.
        special_ids: Ids that never join.
        keep: Probability that a listed occurrence is used.
    """

    def __init__(
        self,
        tables: NgramTables,
        classes: TokenClasses,
        special_ids: Iterable[int],
        keep: float = 1.0,
    ) -> None:
        if not 0 < keep <= 1:
            raise ValueError("keep must be in (0, 1]")
        self.base = tables.vocab_size
        self.tables = {
            n: torch.from_numpy(keys.astype(np.int64))
            for n, keys in sorted(tables.tables.items(), reverse=True)
            if keys.size
        }
        word = classes.word.copy()
        word[list(set(special_ids))] = False
        self.word = torch.from_numpy(word)
        self.spaced = torch.from_numpy(classes.spaced.copy())
        self.keep = keep

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Units for ``[batch, seq]`` ids."""
        ids = ids.long()
        batch, length = ids.shape
        cont = _torch_continuation(ids, self.word, self.spaced)
        complete = self.word[ids] & ~cont
        complete[:, :-1] &= ~cont[:, 1:]
        joined = cont.clone()
        occupied = torch.zeros((batch, length), dtype=torch.bool)
        for n, table in self.tables.items():
            if length < n:
                continue
            width = length - n + 1
            keys = torch.zeros((batch, width), dtype=torch.long)
            valid = torch.ones((batch, width), dtype=torch.bool)
            free = torch.ones((batch, width), dtype=torch.bool)
            for offset in range(n):
                keys = keys * self.base + ids[:, offset : offset + width]
                valid &= complete[:, offset : offset + width]
                free &= ~occupied[:, offset : offset + width]
            match = valid & free & torch.isin(keys, table)
            if self.keep < 1:
                match &= torch.rand((batch, width), generator=generator) < self.keep
            blocked = torch.zeros_like(match)
            for back in range(1, n):
                blocked[:, back:] |= match[:, :-back]
            accepted = match & ~blocked
            for offset in range(n):
                occupied[:, offset : offset + width] |= accepted
                if offset:
                    joined[:, offset : offset + width] |= accepted
        starts = ~joined
        return _units_from_starts(starts)
