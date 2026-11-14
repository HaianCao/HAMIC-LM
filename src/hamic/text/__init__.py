"""Vietnamese text utilities."""

from hamic.text.normalization import NormalizationConfig, VietnameseNormalizer
from hamic.text.robustness import (
    Perturbation,
    decompose,
    perturbations,
    strip_diacritics,
    to_teencode,
)

__all__ = [
    "NormalizationConfig",
    "Perturbation",
    "VietnameseNormalizer",
    "decompose",
    "perturbations",
    "strip_diacritics",
    "to_teencode",
]
