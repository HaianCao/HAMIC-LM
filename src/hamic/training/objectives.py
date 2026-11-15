"""Pretraining objectives: causal LM for decoders, dynamic masked LM for encoders.

An objective turns a batch of token ids into the keyword arguments of the model's
``forward`` (``input_ids`` and ``labels``). Masking draws from a generator seeded by the
global micro-step, so a resumed run masks exactly as an uninterrupted one.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

from hamic.training.masking import (
    MaskRateSchedule,
    SelectionUnits,
    TokenUnits,
    VariantTable,
)

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
    """Dynamic MLM with the BERT 80/10/10 replacement policy and HAMIC extensions.

    Positions are grouped into selection units (single tokens by default; spans or PMI
    collocations optionally); each unit is selected with probability equal to the current
    mask rate, and every position of a selected unit gets the unit's action: ``<mask>``,
    a random token, a Vietnamese spelling variant of the original token, or unchanged.
    Labels always hold the original ids. With default arguments the draws, and so the
    masks, are identical to the original token-level implementation.

    Args:
        mask_rate: Share of maskable positions selected for prediction, or a schedule
            mapping the micro-step to that share.
        mask_id: Id of ``<mask>``.
        vocab_size: Vocabulary size, for random replacements.
        special_ids: Ids never selected (``<s>``, ``</s>``, ``<pad>``...).
        seed: Base seed; combined with the step so masks are reproducible.
        mask_share: Share of selected positions replaced by ``<mask>``.
        random_share: Share replaced by a random non-special token; the rest are kept.
        variant_share: Share replaced by a spelling variant (needs ``variants``); taken
            from ``random_share`` first, then from ``mask_share``. Positions whose token
            has no variant get a random token instead.
        units: Selection units (default: every token alone).
        variants: Variant table for ``variant_share``.
    """

    name = "mlm"

    def __init__(
        self,
        mask_rate: float | MaskRateSchedule,
        mask_id: int,
        vocab_size: int,
        special_ids: Iterable[int],
        seed: int = 0,
        mask_share: float = 0.8,
        random_share: float = 0.1,
        variant_share: float = 0.0,
        units: SelectionUnits | None = None,
        variants: VariantTable | None = None,
    ) -> None:
        schedule = mask_rate if isinstance(mask_rate, MaskRateSchedule) else None
        rates = (
            [schedule.start_rate] + ([schedule.end_rate] if schedule.end_rate else [])
            if schedule
            else [float(mask_rate)]  # type: ignore[arg-type]
        )
        if not all(0 < r < 1 for r in rates):
            raise ValueError("mask_rate must be in (0, 1)")
        if mask_share + random_share > 1:
            raise ValueError("mask_share + random_share must be <= 1")
        if variant_share < 0 or variant_share > mask_share + random_share:
            raise ValueError("variant_share must be in [0, mask_share + random_share]")
        if variant_share and variants is None:
            raise ValueError("variant_share needs a VariantTable")
        self.schedule = schedule or MaskRateSchedule(float(mask_rate))  # type: ignore[arg-type]
        self.mask_rate = self.schedule.start_rate
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.special = torch.tensor(sorted(set(special_ids)), dtype=torch.long)
        self.first_regular_id = int(self.special.max()) + 1
        self.seed = seed
        taken_from_random = min(variant_share, random_share)
        self.random_share = random_share - taken_from_random
        self.mask_share = mask_share - (variant_share - taken_from_random)
        self.variant_share = variant_share
        self.units = units or TokenUnits()
        self.variants = variants

    def __call__(self, input_ids: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        """Select, replace and label positions for micro-step ``step``."""
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        ids = input_ids.cpu()
        maskable = ~torch.isin(ids, self.special)
        unit = self.units(ids, generator)
        rate = self.schedule(step)
        selected = torch.gather(torch.rand(ids.shape, generator=generator) < rate, 1, unit)
        selected &= maskable
        labels = torch.where(selected, ids, torch.full_like(ids, IGNORE_INDEX))
        action = torch.gather(torch.rand(ids.shape, generator=generator), 1, unit)
        random_ids = torch.randint(
            self.first_regular_id, self.vocab_size, ids.shape, generator=generator
        )
        masked = ids.clone()
        random_end = self.mask_share + self.random_share
        to_mask = selected & (action < self.mask_share)
        to_random = selected & (action >= self.mask_share) & (action < random_end)
        if self.variant_share and self.variants is not None:
            to_variant = (
                selected & (action >= random_end) & (action < random_end + self.variant_share)
            )
            variant_ids, available = self.variants.sample(
                ids, torch.rand(ids.shape, generator=generator)
            )
            to_random |= to_variant & ~available
            masked[to_variant & available] = variant_ids[to_variant & available]
        masked[to_mask] = self.mask_id
        masked[to_random] = random_ids[to_random]
        return {"input_ids": masked, "labels": labels}
