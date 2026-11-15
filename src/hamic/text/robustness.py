"""Text perturbations used by the Vietnamese robustness evaluation suite.

These transforms produce the variants a Vietnamese model meets in the wild: text without
diacritics, decomposed Unicode and social-media spelling ("teencode"). They build the
robustness test sets, and ``hamic.training.masking.VariantTable`` uses them to list
spelling variants of vocabulary entries for the optional variant corruption of MLM
pretraining (training text itself is never rewritten). The teencode lexicon and rules
live in ``hamic/resources/teencode.yaml``; randomness is always seeded so a perturbed
test set is reproducible.
"""

from __future__ import annotations

import functools
import random
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hamic.resources import Resource, load_resource

__all__ = [
    "Perturbation",
    "TeencodeRule",
    "Teencoder",
    "decompose",
    "perturbations",
    "strip_diacritics",
    "teencode_identity",
    "to_teencode",
]

_SYLLABLE = re.compile(r"\w+")
_VOWELS = frozenset("aeiouy")


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese tone and vowel marks, mapping ``đ``/``Đ`` to ``d``/``D``."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")


def decompose(text: str) -> str:
    """Return the NFD form, which renders identically but differs code point by code point."""
    return unicodedata.normalize("NFD", text)


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper() and replacement:
        return replacement[0].upper() + replacement[1:]
    return replacement


@dataclass(frozen=True)
class TeencodeRule:
    """One syllable-level spelling rule.

    Attributes:
        name: Rule id.
        kind: ``"replace"`` (regex substitution), ``"strip_diacritics"`` or ``"elongate"``.
        pattern: Regex applied to the lower-cased syllable (``"replace"`` only).
        replace: Replacement text (``"replace"`` only).
        min_repeat: Fewest extra copies of the last letter (``"elongate"`` only).
        max_repeat: Most extra copies of the last letter (``"elongate"`` only).
        weight: Multiplier on the caller's ``rule_rate`` for this rule.
    """

    name: str
    kind: str = "replace"
    pattern: str = ""
    replace: str = ""
    min_repeat: int = 2
    max_repeat: int = 4
    weight: float = 1.0

    def apply(self, syllable: str, rng: random.Random) -> str:
        """Return ``syllable`` transformed by this rule."""
        if self.kind == "strip_diacritics":
            return strip_diacritics(syllable)
        if self.kind == "elongate":
            # People stretch vowels ("quáaaa", "nhaaa"), not final consonants.
            if not syllable or strip_diacritics(syllable[-1]).lower() not in _VOWELS:
                return syllable
            return syllable + syllable[-1] * rng.randint(self.min_repeat, self.max_repeat)
        changed = re.sub(self.pattern, self.replace, syllable.lower())
        return _match_case(syllable, changed) if changed != syllable.lower() else syllable


class Teencoder:
    """Rewrite Vietnamese text in social-media spelling.

    Args:
        resource: Parsed ``teencode.yaml``; the packaged file is used by default.
    """

    def __init__(self, resource: Resource | None = None) -> None:
        res = resource or load_resource("teencode.yaml")
        self.identity = res.identity()
        entries: dict[str, list[str]] = {}
        for section in ("phrases", "words"):
            for key, variants in (res.data.get(section) or {}).items():
                entries[unicodedata.normalize("NFC", str(key)).lower()] = [
                    str(v) for v in variants
                ]
        self._entries = entries
        keys = sorted(entries, key=len, reverse=True)
        self._pattern = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE
        )
        self.rules = [TeencodeRule(**rule) for rule in res.data.get("rules") or []]

    def __len__(self) -> int:
        return len(self._entries)

    def __call__(
        self,
        text: str,
        *,
        seed: int | None = None,
        word_rate: float = 1.0,
        rule_rate: float = 0.0,
    ) -> str:
        """Return ``text`` in teencode.

        Args:
            text: Input text.
            seed: ``None`` gives the deterministic, most common form (first variant, no
                rules). An integer samples variants and rules reproducibly.
            word_rate: Probability of replacing each lexicon match (seeded mode only).
            rule_rate: Probability of applying each syllable rule (seeded mode only).
        """
        rng = random.Random(seed) if seed is not None else None
        normalized = unicodedata.normalize("NFC", text)

        def lexicon(match: re.Match[str]) -> str:
            variants = self._entries[match.group(0).lower()]
            if rng is None:
                return _match_case(match.group(0), variants[0])
            if rng.random() >= word_rate:
                return match.group(0)
            return _match_case(match.group(0), rng.choice(variants))

        result = self._pattern.sub(lexicon, normalized)
        if rng is None or rule_rate <= 0 or not self.rules:
            return result

        def syllable(match: re.Match[str]) -> str:
            word = match.group(0)
            if word.isdigit():
                return word
            for rule in self.rules:
                if rng.random() < rule_rate * rule.weight:
                    word = rule.apply(word, rng)
            return word

        return _SYLLABLE.sub(syllable, result)


@functools.cache
def _default_teencoder() -> Teencoder:
    return Teencoder()


def to_teencode(
    text: str, *, seed: int | None = None, word_rate: float = 1.0, rule_rate: float = 0.0
) -> str:
    """Rewrite ``text`` with the packaged teencode lexicon (see :class:`Teencoder`)."""
    return _default_teencoder()(text, seed=seed, word_rate=word_rate, rule_rate=rule_rate)


@dataclass(frozen=True)
class Perturbation:
    """A named text transform in the robustness suite."""

    name: str
    transform: Callable[[str], str] = field(compare=False)

    def __call__(self, text: str) -> str:
        """Apply the transform."""
        return self.transform(text)


def _seeded_teencode(text: str) -> str:
    return to_teencode(text, seed=0, word_rate=0.8, rule_rate=0.15)


def perturbations() -> tuple[Perturbation, ...]:
    """Return the standard robustness perturbations in reporting order."""
    return (
        Perturbation("no_diacritics", strip_diacritics),
        Perturbation("nfd", decompose),
        Perturbation("teencode", _seeded_teencode),
    )


def teencode_identity() -> dict[str, Any]:
    """Version and hash of the teencode lexicon in use (for reports)."""
    return _default_teencoder().identity
