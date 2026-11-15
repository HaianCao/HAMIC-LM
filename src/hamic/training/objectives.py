"""Pretraining objectives: causal LM for decoders, dynamic masked LM for encoders.

An objective turns a batch of token ids into the keyword arguments of the model's
``forward`` (``input_ids`` and ``labels``). Masking draws from a generator seeded by the
global micro-step, so a resumed run masks exactly as an uninterrupted one.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

__all__ = ["CausalLM", "MaskedLM", "Objective"]

IGNORE_INDEX = -100


class Objective(Protocol):
    """Builds model inputs from a batch of token ids."""

    name: str

    def __call__(self, input_ids: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        """Return ``forward`` keyword arguments for micro-step ``step``."""
        ...


class CausalLM:
    """Next-token prediction; Transformers shifts the labels inside the model."""

    name = "clm"

    def __call__(self, input_ids: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        """Labels are the inputs themselves."""
        del step
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class MaskedLM:
    """Dynamic MLM with the BERT 80/10/10 replacement policy.

    Args:
        mask_rate: Share of maskable positions selected for prediction (0.15 or 0.30).
        mask_id: Id of ``<mask>``.
        vocab_size: Vocabulary size, for random replacements.
        special_ids: Ids never selected (``<s>``, ``</s>``, ``<pad>``...).
        seed: Base seed; combined with the step so masks are reproducible.
        mask_share: Share of selected positions replaced by ``<mask>``.
        random_share: Share replaced by a random non-special token; the rest are kept.
    """

    name = "mlm"

    def __init__(
        self,
        mask_rate: float,
        mask_id: int,
        vocab_size: int,
        special_ids: Iterable[int],
        seed: int = 0,
        mask_share: float = 0.8,
        random_share: float = 0.1,
    ) -> None:
        if not 0 < mask_rate < 1:
            raise ValueError("mask_rate must be in (0, 1)")
        if mask_share + random_share > 1:
            raise ValueError("mask_share + random_share must be <= 1")
        self.mask_rate = mask_rate
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.special = torch.tensor(sorted(set(special_ids)), dtype=torch.long)
        self.first_regular_id = int(self.special.max()) + 1
        self.seed = seed
        self.mask_share = mask_share
        self.random_share = random_share

    def __call__(self, input_ids: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        """Select, replace and label positions for micro-step ``step``."""
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        ids = input_ids.cpu()
        maskable = ~torch.isin(ids, self.special)
        selected = (torch.rand(ids.shape, generator=generator) < self.mask_rate) & maskable
        labels = torch.where(selected, ids, torch.full_like(ids, IGNORE_INDEX))
        action = torch.rand(ids.shape, generator=generator)
        random_ids = torch.randint(
            self.first_regular_id, self.vocab_size, ids.shape, generator=generator
        )
        masked = ids.clone()
        to_mask = selected & (action < self.mask_share)
        to_random = (
            selected & (action >= self.mask_share) & (action < self.mask_share + self.random_share)
        )
        masked[to_mask] = self.mask_id
        masked[to_random] = random_ids[to_random]
        return {"input_ids": masked, "labels": labels}
