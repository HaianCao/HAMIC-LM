"""Runtime parameter counting; the only source for parameter numbers in model cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hamic.modeling.factory import build_model
from hamic.modeling.presets import ModelPreset

__all__ = ["ParameterReport", "count_parameters", "parameter_report"]


@dataclass(frozen=True)
class ParameterReport:
    """Parameter counts of an instantiated model (tied weights counted once)."""

    name: str
    total: int
    trainable: int
    embedding: int
    label: float
    tolerance: float

    @property
    def non_embedding(self) -> int:
        """Parameters outside the token embedding table."""
        return self.total - self.embedding

    @property
    def deviation(self) -> float:
        """Relative deviation of :attr:`total` from the name label."""
        return (self.total - self.label) / self.label

    @property
    def within_label(self) -> bool:
        """Whether the runtime count matches the name within tolerance."""
        return abs(self.deviation) <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        return {
            "name": self.name,
            "total_parameters": self.total,
            "trainable_parameters": self.trainable,
            "embedding_parameters": self.embedding,
            "non_embedding_parameters": self.non_embedding,
            "label_parameters": self.label,
            "deviation": round(self.deviation, 4),
            "within_label": self.within_label,
        }


def count_parameters(model: Any) -> tuple[int, int, int]:
    """Return ``(total, trainable, embedding)`` for a ``transformers`` model."""
    params = list(model.parameters())  # deduplicates tied tensors
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    embedding = model.get_input_embeddings().weight.numel()
    return total, trainable, embedding


def parameter_report(preset: ModelPreset) -> ParameterReport:
    """Build ``preset`` on the meta device and count its parameters."""
    model = build_model(preset, device="meta")
    total, trainable, embedding = count_parameters(model)
    return ParameterReport(
        name=preset.name,
        total=total,
        trainable=trainable,
        embedding=embedding,
        label=preset.label_parameters,
        tolerance=preset.label_tolerance,
    )
