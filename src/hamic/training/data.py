"""Packed training sequences from tokenized shards, resumable to the exact token.

Documents are visited in a seeded permutation per epoch. Each document contributes
``<s> tokens </s>``, and these pieces are concatenated and cut into rows of exactly
``seq_len`` tokens, so no position is padding. Whether attention may cross document
boundaries inside a row is a training option (see ``hamic.training.attention``).

Optional per-source weights repeat documents within an epoch: weight ``2.0`` visits each
document of that source twice, ``0.5`` visits a seeded half of them. Without weights the
epoch order is the plain seeded permutation of earlier runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hamic.data.tokenized import TokenizedCorpus
from hamic.training.state import DataCursor

__all__ = ["PackedBatches"]


class PackedBatches:
    """Deterministic, resumable batches of packed sequences.

    Args:
        corpus: Tokenized corpus.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        seq_len: Tokens per row.
        batch_size: Rows per batch.
        bos_id: Id prepended to every document.
        eos_id: Id appended to every document.
        seed: Seed of the per-epoch document permutation.
        shuffle: Visit documents in a seeded permutation (``False`` keeps corpus order).
        cursor: Position to resume from (``None`` starts at the beginning).
        source_weights: Visits per document of each source id per epoch (sources not
            listed keep 1.0). Needs per-document sources in the tokenized shards.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus,
        split: str,
        seq_len: int,
        batch_size: int,
        bos_id: int,
        eos_id: int,
        seed: int = 0,
        shuffle: bool = True,
        cursor: DataCursor | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if seq_len < 2 or batch_size < 1:
            raise ValueError("seq_len must be >= 2 and batch_size >= 1")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.seed = seed
        self.shuffle = shuffle
        self._shards = list(corpus.shards(split))
        self._starts = np.cumsum([0] + [len(s) for s in self._shards])
        self.documents = int(self._starts[-1])
        if self.documents == 0:
            raise ValueError(f"Split {split!r} has no documents")
        self._visits = self._document_weights(corpus, source_weights) if source_weights else None
        start = cursor or DataCursor()
        self.epoch = start.epoch
        self.position = start.offset
        self.token_offset = start.token_offset
        self.consumed_tokens = start.consumed_tokens
        self._order = self._permutation(self.epoch)

    def _document_weights(
        self, corpus: TokenizedCorpus, source_weights: dict[str, float]
    ) -> np.ndarray[Any, Any]:
        source_ids = corpus.source_ids
        if source_ids is None or any(s.sources is None for s in self._shards):
            raise ValueError(
                "source_weights need per-document sources; run `hamic corpus annotate-sources`"
            )
        unknown = set(source_weights) - set(source_ids)
        if unknown:
            raise ValueError(f"Unknown sources in source_weights: {sorted(unknown)}")
        table = np.array([source_weights.get(s, 1.0) for s in source_ids], dtype=np.float64)
        sources = np.concatenate([s.sources for s in self._shards if s.sources is not None])
        weights: np.ndarray[Any, Any] = table[sources]
        return weights

    def _permutation(self, epoch: int) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng([self.seed, epoch])
        if self._visits is not None:
            whole = np.floor(self._visits)
            extra = rng.random(self.documents) < self._visits - whole
            order = np.repeat(np.arange(self.documents), (whole + extra).astype(np.int64))
            if order.size == 0:
                raise ValueError("source_weights leave no documents to visit")
            return rng.permutation(order) if self.shuffle else order
        if not self.shuffle:
            return np.arange(self.documents)
        return rng.permutation(self.documents)

    def _piece(self, global_index: int) -> np.ndarray[Any, Any]:
        shard = int(np.searchsorted(self._starts, global_index, side="right")) - 1
        tokens = self._shards[shard].document(global_index - int(self._starts[shard]))
        piece = np.empty(tokens.size + 2, dtype=np.int64)
        piece[0], piece[-1] = self.bos_id, self.eos_id
        piece[1:-1] = tokens
        return piece

    def _advance_document(self) -> None:
        self.position += 1
        self.token_offset = 0
        if self.position >= self._order.size:
            self.epoch += 1
            self.position = 0
            self._order = self._permutation(self.epoch)

    def next_batch(self) -> torch.Tensor:
        """Return the next ``[batch_size, seq_len]`` tensor of token ids."""
        out = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for row in range(self.batch_size):
            filled = 0
            while filled < self.seq_len:
                piece = self._piece(int(self._order[self.position]))
                take = min(self.seq_len - filled, piece.size - self.token_offset)
                out[row, filled : filled + take] = piece[
                    self.token_offset : self.token_offset + take
                ]
                filled += take
                self.token_offset += take
                if self.token_offset >= piece.size:
                    self._advance_document()
        self.consumed_tokens += out.size
        return torch.from_numpy(out)

    def cursor(self) -> DataCursor:
        """Return the position of the next token to be served."""
        return DataCursor(
            offset=self.position,
            consumed_tokens=self.consumed_tokens,
            epoch=self.epoch,
            token_offset=self.token_offset,
        )
