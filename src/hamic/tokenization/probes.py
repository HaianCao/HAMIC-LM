"""Vietnamese probe strings for the tokenizer acceptance suite.

The strings come from ``hamic/resources/probes.yaml``. Two kinds of category are built in
code so they stay consistent with their sources instead of being hand-written:

* ``person_names``: deterministic combinations of common surname/middle/given names;
* ``no_diacritics`` and ``teencode``: derived from the file's ``base_sentences`` with
  :func:`~hamic.text.robustness.strip_diacritics` and seeded
  :func:`~hamic.text.robustness.to_teencode`.

Every report records the version and SHA-256 of the probe file and the teencode lexicon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hamic.resources import Resource, load_resource
from hamic.text.robustness import strip_diacritics, teencode_identity, to_teencode

__all__ = ["PROBES", "ProbeSet", "load_probes"]

TEENCODE_WORD_RATE = 0.8
TEENCODE_RULE_RATE = 0.15


@dataclass(frozen=True)
class ProbeSet:
    """Probe strings grouped by category, plus the identity of the data behind them."""

    categories: Mapping[str, tuple[str, ...]]
    identity: Mapping[str, Any]

    def __len__(self) -> int:
        return sum(len(items) for items in self.categories.values())


def _person_names(parts: Mapping[str, Any]) -> list[str]:
    surnames: Sequence[str] = parts["surnames"]
    middle: Sequence[str] = parts["middle"]
    given: Sequence[str] = parts["given"]
    names: list[str] = []
    for i in range(int(parts["count"])):
        names.append(
            f"{surnames[i % len(surnames)]} {middle[(3 * i + 1) % len(middle)]} "
            f"{given[(7 * i + 2) % len(given)]}"
        )
    return names


def load_probes(resource: Resource | None = None) -> ProbeSet:
    """Build the probe set from ``probes.yaml`` (or a given resource)."""
    res = resource or load_resource("probes.yaml")
    data = res.data
    categories: dict[str, tuple[str, ...]] = {
        "person_names": tuple(_person_names(data["person_name_parts"]))
    }
    for name, items in data["categories"].items():
        categories[name] = tuple(str(item) for item in items)
    base = [str(s) for s in data["base_sentences"]]
    categories["no_diacritics"] = tuple(strip_diacritics(s) for s in base)
    categories["teencode"] = tuple(
        to_teencode(s, seed=i, word_rate=TEENCODE_WORD_RATE, rule_rate=TEENCODE_RULE_RATE)
        for i, s in enumerate(base)
    )
    identity = {
        "probes": res.identity(),
        "teencode": teencode_identity(),
        "teencode_rates": {"word": TEENCODE_WORD_RATE, "rule": TEENCODE_RULE_RATE},
    }
    return ProbeSet(categories=categories, identity=identity)


PROBES: Mapping[str, tuple[str, ...]] = load_probes().categories
"""Default probe categories (kept for backwards compatibility)."""
