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

import json
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
    """Geometric span lengths (SpanBERT: ``p = 0.2``, at most 10 tokens, mean ~3.8).

    Args:
        p: Success probability of the geometric length distribution.
        max_length: Longest span.
    """

    def __init__(self, p: float = 0.2, max_length: int = 10) -> None:
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
    """

    def __init__(self, pairs: torch.Tensor, vocab_size: int, special_ids: Iterable[int]) -> None:
        self.pairs = pairs.to(torch.int64)
        self.vocab_size = vocab_size
        self.special = torch.tensor(sorted(set(special_ids)), dtype=torch.long)

    def __call__(self, ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """A position starts a new unit unless it continues a listed pair."""
        del generator
        ids = ids.long()
        keys = ids[:, :-1] * self.vocab_size + ids[:, 1:]
        joined = torch.isin(keys, self.pairs)
        regular = ~torch.isin(ids, self.special)
        joined &= regular[:, :-1] & regular[:, 1:]
        starts = torch.ones_like(ids, dtype=torch.bool)
        starts[:, 1:] = ~joined
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
) -> np.ndarray[Any, Any]:
    """Mine frequent, strongly associated adjacent token pairs.

    PMI is ``log(p(a, b) / (p(a) p(b)))`` over adjacent pairs inside documents. Pairs seen
    at least ``min_count`` times with PMI at least ``min_pmi`` are kept, the highest PMI
    first, at most ``max_pairs``.

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

    Returns:
        Sorted ``int64`` keys ``left * vocab_size + right``.
    """
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
    keys, pmi = keys[strong], pmi[strong]
    best = np.argsort(-pmi, kind="stable")[:max_pairs]
    return np.sort(keys[best])


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
