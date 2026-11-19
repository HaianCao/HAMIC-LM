"""Building blocks of the masked-LM objective.

* **Selection units** decide what is masked together. :class:`TokenUnits` treats every
  token alone (BERT); :class:`SpanUnits` cuts rows into geometric spans (SpanBERT);
  :class:`PmiUnits` joins neighbouring tokens whose pair is a frequent, high-PMI
  collocation mined from the training split, so Vietnamese multi-syllable words and names
  ("thủ đô", "Hồ Chí Minh") tend to be masked whole without an external word segmenter.
  Every token belongs to exactly one unit, and each unit is selected with probability
  equal to the mask rate, so the expected share of selected tokens equals the rate.
* :class:`MaskRateSchedule` moves the rate linearly over training.
* :class:`VariantTable` lists, for each vocabulary entry, other entries that spell the
  same Vietnamese syllable differently: another tone, no diacritics, teencode or another
  casing. Replacing a selected token by such a variant gives a hard, linguistically
  plausible corruption that the model must undo from context.

All randomness comes from a ``torch.Generator`` supplied by the caller, so masking stays
reproducible across resumes.
"""

from __future__ import annotations

import heapq
import json
import math
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from hamic.text.robustness import strip_diacritics, to_teencode

__all__ = [
    "MaskRateSchedule",
    "PmiUnits",
    "SelectionUnits",
    "SpanUnits",
    "TokenUnits",
    "VariantTable",
    "decoded_vocabulary",
    "mine_pmi_pairs",
    "remove_tone",
    "word_tokens",
]

# Combining marks of the five Vietnamese tones (grave, acute, tilde, hook above, dot below).
_TONE_MARKS = frozenset("̣̀́̃̉")


class SelectionUnits(Protocol):
    """Partition each row of ids into contiguous units masked as a whole."""

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Return ``[batch, seq]`` unit numbers, non-decreasing along each row from 0."""
        ...


def _units_from_starts(starts: torch.Tensor) -> torch.Tensor:
    starts = starts.clone()
    starts[:, 0] = True
    return torch.cumsum(starts.long(), dim=1) - 1


class TokenUnits:
    """Every token is its own unit (standard MLM)."""

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """One unit per position."""
        del generator
        return torch.arange(ids.shape[1]).expand_as(ids).contiguous()


class SpanUnits:
    """Geometric span lengths, truncated to ``[1, max_length]``.

    SpanBERT uses ``p = 0.2`` and at most 10 tokens (mean ~3.8) for English. HAMIC's
    default of 5 (mean ~2.6) keeps spans near Vietnamese word length: with 10, about half
    of the masked tokens would sit in spans of 6-10 tokens (three or four words).

    Args:
        p: Success probability of the geometric length distribution.
        max_length: Longest span.
    """

    def __init__(self, p: float = 0.2, max_length: int = 5) -> None:
        if not 0 < p <= 1 or max_length < 1:
            raise ValueError("need 0 < p <= 1 and max_length >= 1")
        self.p = p
        self.max_length = max_length

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Cut rows at the ends of consecutive spans of sampled lengths."""
        batch, length = ids.shape
        # Enough spans to cover any row: each span is at least one token long. Lengths come
        # from the geometric distribution truncated to [1, max_length] by inverse CDF.
        uniform = torch.rand((batch, length), generator=generator)
        if self.p >= 1:
            lengths = torch.ones((batch, length), dtype=torch.long)
        else:
            tail = 1.0 - (1.0 - self.p) ** self.max_length
            lengths = torch.floor(torch.log1p(-uniform * tail) / np.log1p(-self.p)).long() + 1
            lengths = lengths.clamp(1, self.max_length)
        ends = torch.cumsum(lengths, dim=1)
        starts = torch.zeros((batch, length + 1), dtype=torch.bool)
        starts.scatter_(1, (ends - lengths).clamp_max(length), True)
        return _units_from_starts(starts[:, :length])


class PmiUnits:
    """Join neighbours whose id pair appears in a mined collocation table.

    Args:
        pairs: Sorted ``int64`` keys ``left * vocab_size + right`` (see
            :func:`mine_pmi_pairs`).
        vocab_size: Vocabulary size used to build the keys.
        special_ids: Ids that never join a unit.
        keep: Probability that each listed pair occurrence joins its tokens. ``1.0`` masks
            collocations whole every time; below 1 a collocation is sometimes split, so the
            model also practises predicting one part of a word while the rest is visible
            (pure PMI units never do, which costs token- and span-level accuracy).
        max_length: Most tokens in one joined unit. ``None`` preserves unrestricted legacy
            behaviour. A finite limit prevents chains of individually strong pairs from
            turning an entire phrase into one masking unit.
    """

    def __init__(
        self,
        pairs: torch.Tensor,
        vocab_size: int,
        special_ids: Iterable[int],
        keep: float = 1.0,
        max_length: int | None = None,
    ) -> None:
        if not 0 < keep <= 1:
            raise ValueError("keep must be in (0, 1]")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be >= 2 or None")
        self.pairs = pairs.to(torch.int64)
        self.vocab_size = vocab_size
        self.special = torch.tensor(sorted(set(special_ids)), dtype=torch.long)
        self.keep = keep
        self.max_length = max_length

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """A position starts a new unit unless it continues a listed (kept) pair."""
        ids = ids.long()
        keys = ids[:, :-1] * self.vocab_size + ids[:, 1:]
        joined = torch.isin(keys, self.pairs)
        regular = ~torch.isin(ids, self.special)
        joined &= regular[:, :-1] & regular[:, 1:]
        if self.keep < 1:
            joined &= torch.rand(keys.shape, generator=generator) < self.keep
        starts = torch.ones_like(ids, dtype=torch.bool)
        starts[:, 1:] = ~joined
        if self.max_length is not None:
            # Split long chains without a Python loop over sequence positions. ``last_start``
            # gives the start offset of each unrestricted unit; every max_length-th token in
            # that unit becomes an additional boundary.
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            last_start = torch.where(starts, positions, 0).cummax(dim=1).values
            offset = positions - last_start
            starts |= (offset > 0) & (offset.remainder(self.max_length) == 0)
        return _units_from_starts(starts)


def mine_pmi_pairs(
    documents: Iterable[np.ndarray[Any, Any]],
    vocab_size: int,
    special_ids: Iterable[int],
    max_tokens: int = 100_000_000,
    min_count: int = 50,
    min_pmi: float = 3.0,
    max_pairs: int = 200_000,
    eligible: np.ndarray[Any, Any] | None = None,
    discount_k: float = 0.0,
    scratch_dir: str | Path | None = None,
    chunk_tokens: int | None = None,
) -> np.ndarray[Any, Any]:
    """Mine frequent, strongly associated adjacent token pairs.

    PMI is ``log(p(a, b) / (p(a) p(b)))`` over adjacent pairs inside documents. Pairs seen
    at least ``min_count`` times with PMI at least ``min_pmi`` are kept, the highest PMI
    first, at most ``max_pairs``. When ``discount_k`` is positive, ranking uses
    ``PMI * count / (count + discount_k)`` after the raw-PMI threshold. This reduces the
    tendency of barely frequent pairs to outrank well-supported collocations.

    Args:
        documents: Token ids per document, read until ``max_tokens`` are seen.
        vocab_size: Vocabulary size (key base).
        special_ids: Ids never paired.
        max_tokens: Tokens to read.
        min_count: Fewest occurrences of a kept pair.
        min_pmi: Lowest PMI (nats) of a kept pair.
        max_pairs: Most pairs returned.
        eligible: Optional boolean mask over the vocabulary; only pairs of two eligible
            tokens are kept (e.g. :func:`word_tokens`, so punctuation never joins a unit).
        discount_k: Reliability discount used for ranking. ``0`` is legacy raw PMI.
        scratch_dir: Parent directory for temporary sorted count runs. Required with
            ``chunk_tokens``; temporary files are removed after the table is selected.
        chunk_tokens: Approximate adjacent pairs reduced per in-memory chunk. ``None`` uses
            the legacy all-in-memory implementation.

    Returns:
        Sorted ``int64`` keys ``left * vocab_size + right``.
    """
    if max_tokens < 1 or min_count < 1 or max_pairs < 1:
        raise ValueError("max_tokens, min_count and max_pairs must be positive")
    if discount_k < 0:
        raise ValueError("discount_k must be non-negative")
    if (scratch_dir is None) != (chunk_tokens is None):
        raise ValueError("scratch_dir and chunk_tokens must be passed together")
    if chunk_tokens is not None and chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if chunk_tokens is not None:
        assert scratch_dir is not None
        return _mine_pmi_pairs_spilled(
            documents,
            vocab_size,
            special_ids,
            max_tokens,
            min_count,
            min_pmi,
            max_pairs,
            eligible,
            discount_k,
            Path(scratch_dir),
            chunk_tokens,
        )
    special = np.array(sorted(set(special_ids)), dtype=np.int64)
    unigram = np.zeros(vocab_size, dtype=np.int64)
    chunks: list[np.ndarray[Any, Any]] = []
    seen = 0
    for document in documents:
        ids = np.asarray(document, dtype=np.int64)
        if ids.size == 0:
            continue
        unigram += np.bincount(ids, minlength=vocab_size)
        seen += ids.size
        if ids.size > 1:
            keep = ~(np.isin(ids[:-1], special) | np.isin(ids[1:], special))
            chunks.append((ids[:-1] * vocab_size + ids[1:])[keep])
        if seen >= max_tokens:
            break
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    keys, counts = np.unique(np.concatenate(chunks), return_counts=True)
    total_pairs = counts.sum()
    total_tokens = unigram.sum()
    keep = counts >= min_count
    if eligible is not None:
        keep &= eligible[keys // vocab_size] & eligible[keys % vocab_size]
    keys, counts = keys[keep], counts[keep]
    left, right = keys // vocab_size, keys % vocab_size
    pmi = np.log(counts / total_pairs) - np.log(
        unigram[left] / total_tokens * unigram[right] / total_tokens
    )
    strong = pmi >= min_pmi
    keys, counts, pmi = keys[strong], counts[strong], pmi[strong]
    score = pmi if discount_k == 0 else pmi * counts / (counts + discount_k)
    best = np.argsort(-score, kind="stable")[:max_pairs]
    return np.sort(keys[best])


def _write_count_run(
    directory: Path, index: int, chunks: list[np.ndarray[Any, Any]]
) -> tuple[Path, Path]:
    """Reduce pair occurrences to one sorted key/count run on disk."""
    keys, counts = np.unique(np.concatenate(chunks), return_counts=True)
    key_path = directory / f"run-{index:05d}.keys.npy"
    count_path = directory / f"run-{index:05d}.counts.npy"
    np.save(key_path, keys.astype(np.int64, copy=False), allow_pickle=False)
    np.save(count_path, counts.astype(np.int64, copy=False), allow_pickle=False)
    return key_path, count_path


def _mine_pmi_pairs_spilled(
    documents: Iterable[np.ndarray[Any, Any]],
    vocab_size: int,
    special_ids: Iterable[int],
    max_tokens: int,
    min_count: int,
    min_pmi: float,
    max_pairs: int,
    eligible: np.ndarray[Any, Any] | None,
    discount_k: float,
    scratch_dir: Path,
    chunk_tokens: int,
) -> np.ndarray[Any, Any]:
    """Exact PMI mining with bounded occurrence memory and disk-backed sorted runs."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    special = np.array(sorted(set(special_ids)), dtype=np.int64)
    unigram = np.zeros(vocab_size, dtype=np.int64)
    seen = 0
    pending = 0
    chunks: list[np.ndarray[Any, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hamic-pmi-", dir=scratch_dir) as temporary:
        directory = Path(temporary)
        paths: list[tuple[Path, Path]] = []
        for document in documents:
            ids = np.asarray(document, dtype=np.int64)
            if ids.size == 0:
                continue
            unigram += np.bincount(ids, minlength=vocab_size)
            seen += ids.size
            if ids.size > 1:
                regular = ~(np.isin(ids[:-1], special) | np.isin(ids[1:], special))
                pairs = (ids[:-1] * vocab_size + ids[1:])[regular]
                if pairs.size:
                    chunks.append(pairs)
                    pending += pairs.size
            if pending >= chunk_tokens:
                paths.append(_write_count_run(directory, len(paths), chunks))
                chunks, pending = [], 0
            if seen >= max_tokens:
                break
        if chunks:
            paths.append(_write_count_run(directory, len(paths), chunks))
        if not paths:
            return np.zeros(0, dtype=np.int64)
        return _select_from_count_runs(
            paths,
            unigram,
            eligible,
            vocab_size,
            min_count,
            min_pmi,
            max_pairs,
            discount_k,
        )


def _select_from_count_runs(
    paths: list[tuple[Path, Path]],
    unigram: np.ndarray[Any, Any],
    eligible: np.ndarray[Any, Any] | None,
    vocab_size: int,
    min_count: int,
    min_pmi: float,
    max_pairs: int,
    discount_k: float,
) -> np.ndarray[Any, Any]:
    """K-way merge exact counts and retain only the strongest pairs in memory."""
    key_runs = [np.load(key_path, mmap_mode="r") for key_path, _ in paths]
    count_runs = [np.load(count_path, mmap_mode="r") for _, count_path in paths]
    total_pairs = sum(int(counts.sum()) for counts in count_runs)
    total_tokens = int(unigram.sum())
    merge: list[tuple[int, int, int]] = []
    for run, keys in enumerate(key_runs):
        if keys.size:
            heapq.heappush(merge, (int(keys[0]), run, 0))
    strongest: list[tuple[float, int]] = []
    while merge:
        key, run, position = heapq.heappop(merge)
        count = int(count_runs[run][position])
        next_position = position + 1
        if next_position < key_runs[run].size:
            heapq.heappush(
                merge, (int(key_runs[run][next_position]), run, next_position)
            )
        while merge and merge[0][0] == key:
            _, other_run, other_position = heapq.heappop(merge)
            count += int(count_runs[other_run][other_position])
            other_next = other_position + 1
            if other_next < key_runs[other_run].size:
                heapq.heappush(
                    merge,
                    (int(key_runs[other_run][other_next]), other_run, other_next),
                )
        left, right = divmod(key, vocab_size)
        if count < min_count or (
            eligible is not None and not (eligible[left] and eligible[right])
        ):
            continue
        pmi = math.log(count / total_pairs) - math.log(
            unigram[left] / total_tokens * unigram[right] / total_tokens
        )
        if pmi < min_pmi:
            continue
        score = pmi if discount_k == 0 else pmi * count / (count + discount_k)
        # Lower keys win exact score ties, matching stable ranking of the in-memory path.
        candidate = (score, -key)
        if len(strongest) < max_pairs:
            heapq.heappush(strongest, candidate)
        elif candidate > strongest[0]:
            heapq.heapreplace(strongest, candidate)
    return np.sort(np.asarray([-negative_key for _, negative_key in strongest], dtype=np.int64))


@dataclass(frozen=True)
class MaskRateSchedule:
    """Mask rate per micro-step: constant, or linear from ``start_rate`` to ``end_rate``.

    Args:
        start_rate: Rate before ``start_step``.
        end_rate: Rate at ``end_step`` and after (``None``: constant ``start_rate``).
        start_step: Micro-step where the change begins.
        end_step: Last micro-step of training.
    """

    start_rate: float
    end_rate: float | None = None
    start_step: int = 0
    end_step: int = 1

    def __call__(self, step: int) -> float:
        """Rate at micro-step ``step``."""
        if self.end_rate is None or step <= self.start_step:
            return self.start_rate
        progress = min(1.0, (step - self.start_step) / max(1, self.end_step - self.start_step))
        return self.start_rate + (self.end_rate - self.start_rate) * progress


def remove_tone(text: str) -> str:
    """Drop the tone mark but keep vowel marks (``"mạ"`` → ``"ma"``, ``"mẹ"`` → ``"me"``)."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))


class VariantTable:
    """Single-token spelling variants of every vocabulary entry.

    Built from the vocabulary only, so a variant always is one existing token with the same
    leading space. Kinds: same syllable with another tone (``má`` → ``mà``, ``mạ``...), no
    diacritics (``Hà`` → ``Ha``), teencode (``không`` → ``ko``) and other casing
    (``nội`` → ``Nội``, ``NỘI``).

    Args:
        offsets: ``[vocab_size + 1]`` CSR offsets into ``values``.
        values: Variant ids.
    """

    def __init__(self, offsets: torch.Tensor, values: torch.Tensor) -> None:
        self.offsets = offsets.to(torch.int64)
        self.values = values.to(torch.int64)
        self.counts = self.offsets[1:] - self.offsets[:-1]

    @classmethod
    def from_vocabulary(cls, tokens: list[str]) -> VariantTable:
        """Build from decoded token strings indexed by id (special tokens: empty string)."""
        lookup: dict[str, int] = {}
        by_toneless: dict[str, list[int]] = defaultdict(list)
        for index, text in enumerate(tokens):
            if _is_word(text):
                lookup.setdefault(text, index)
                by_toneless[remove_tone(text)].append(index)
        variants: list[list[int]] = [[] for _ in tokens]
        for index, text in enumerate(tokens):
            if not _is_word(text):
                continue
            space, core = ("", text) if not text.startswith(" ") else (" ", text[1:])
            candidates = {space + form for form in _spellings(core)}
            found = {lookup[c] for c in candidates if c in lookup}
            found.update(by_toneless[remove_tone(text)])
            found.discard(index)
            variants[index] = sorted(found)
        counts = torch.tensor([len(v) for v in variants], dtype=torch.int64)
        offsets = torch.zeros(len(tokens) + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts, dim=0)
        values = torch.tensor([i for v in variants for i in v], dtype=torch.int64)
        return cls(offsets, values)

    @classmethod
    def from_tokenizer_file(cls, path: str | Path) -> VariantTable:
        """Build from a byte-level ``tokenizer.json`` (added tokens get no variants)."""
        return cls.from_vocabulary(decoded_vocabulary(path))

    def __len__(self) -> int:
        return int(self.counts.numel())

    def sample(
        self, ids: torch.Tensor, uniform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick one variant per position.

        Args:
            ids: Token ids.
            uniform: Uniform ``[0, 1)`` draws with the shape of ``ids``.

        Returns:
            ``(variants, available)``: a variant id per position (meaningless where no
            variant exists) and a mask of positions that have one.
        """
        counts = self.counts[ids]
        available = counts > 0
        pick = (uniform * counts).long().clamp_max((counts - 1).clamp_min(0))
        index = (self.offsets[ids] + pick).clamp_max(max(0, self.values.numel() - 1))
        variants = self.values[index] if self.values.numel() else ids
        return torch.where(available, variants, ids), available


def decoded_vocabulary(path: str | Path) -> list[str]:
    """Text of every id of a byte-level ``tokenizer.json``.

    Added tokens, and byte pieces that are not valid UTF-8 on their own, map to ``""``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    byte_of = {char: byte for byte, char in _byte_alphabet().items()}
    vocab: dict[str, int] = data["model"]["vocab"]
    tokens = [""] * (max(vocab.values()) + 1)
    for piece, index in vocab.items():
        raw = bytes(byte_of[c] for c in piece if c in byte_of)
        tokens[index] = raw.decode("utf-8") if _valid_utf8(raw) else ""
    for added in data.get("added_tokens", []):
        if added["id"] < len(tokens):
            tokens[added["id"]] = ""
    return tokens


def word_tokens(path: str | Path) -> np.ndarray[Any, Any]:
    """Boolean mask of ids made only of letters and digits (one optional leading space)."""
    tokens = decoded_vocabulary(path)
    return np.array([_is_word(t) or (t.strip().isalnum() and t.strip() != "") for t in tokens])


def _is_word(text: str) -> bool:
    core = text[1:] if text.startswith(" ") else text
    return bool(core) and core.isalpha()


def _spellings(core: str) -> set[str]:
    forms = {
        strip_diacritics(core),
        core.lower(),
        core.upper(),
        core[:1].upper() + core[1:].lower(),
        to_teencode(core),
    }
    forms.discard(core)
    return forms


def _valid_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _byte_alphabet() -> dict[int, str]:
    """GPT-2 byte-to-character map used by byte-level BPE vocabularies."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, chars, strict=True)}
