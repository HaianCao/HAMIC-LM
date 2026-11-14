"""Text perturbations used by the Vietnamese robustness evaluation suite.

These transforms produce the variants a Vietnamese model meets in the wild: text without
diacritics, decomposed Unicode, social-media abbreviations and English code-switching.
They are evaluation tools only and are never applied to training data.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_TEENCODE",
    "Perturbation",
    "decompose",
    "perturbations",
    "strip_diacritics",
    "to_teencode",
]

DEFAULT_TEENCODE: Mapping[str, str] = {
    "không": "ko",
    "được": "dc",
    "gì": "j",
    "biết": "bít",
    "vậy": "z",
    "nhé": "nha",
    "anh": "a",
    "em": "e",
}


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese tone and vowel marks, mapping ``đ``/``Đ`` to ``d``/``D``."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")


def decompose(text: str) -> str:
    """Return the NFD form, which renders identically but differs code point by code point."""
    return unicodedata.normalize("NFD", text)


def to_teencode(text: str, mapping: Mapping[str, str] = DEFAULT_TEENCODE) -> str:
    """Replace whole Vietnamese words with common social-media abbreviations."""
    if not mapping:
        return text
    pattern = re.compile(
        r"(?<!\w)("
        + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        return mapping[match.group(0).lower()]

    return pattern.sub(_replace, unicodedata.normalize("NFC", text))


@dataclass(frozen=True)
class Perturbation:
    """A named text transform in the robustness suite."""

    name: str
    transform: Callable[[str], str] = field(compare=False)

    def __call__(self, text: str) -> str:
        """Apply the transform."""
        return self.transform(text)


def perturbations() -> tuple[Perturbation, ...]:
    """Return the standard robustness perturbations in reporting order."""
    return (
        Perturbation("no_diacritics", strip_diacritics),
        Perturbation("nfd", decompose),
        Perturbation("teencode", to_teencode),
    )
