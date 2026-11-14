"""Tokenizer specification and the shared special-token contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DigitMode", "InputMode", "SpecialTokens", "TokenizerSpec"]


class InputMode(str, Enum):
    """How text reaches the tokenizer."""

    RAW = "raw"
    """Raw NFC text; whitespace separates syllables (HAMIC default)."""
    SEGMENTED = "segmented"
    """Word-segmented text where syllables of a word are joined by ``_`` (control arm)."""


class DigitMode(str, Enum):
    """How digit runs are pre-split before BPE."""

    GROUPED = "grouped"
    """Keep digit runs together (GPT-2/PhoGPT behaviour)."""
    INDIVIDUAL = "individual"
    """Split every digit into its own pre-token."""


@dataclass(frozen=True)
class SpecialTokens:
    """Special tokens shared by every HAMIC topology.

    The order fixes the ids: ``<s>`` = 0, ``<pad>`` = 1, ``</s>`` = 2, ``<unk>`` = 3,
    ``<mask>`` = 4 (RoBERTa-compatible), followed by span-corruption sentinels. The
    contract must be frozen before any long run; changing it creates a new tokenizer.
    """

    bos: str = "<s>"
    pad: str = "<pad>"
    eos: str = "</s>"
    unk: str = "<unk>"
    mask: str = "<mask>"
    sentinel_template: str = "<extra_id_{index}>"
    num_sentinels: int = 32

    def __post_init__(self) -> None:
        if self.num_sentinels < 0:
            raise ValueError("num_sentinels must be >= 0")
        if "{index}" not in self.sentinel_template:
            raise ValueError("sentinel_template must contain '{index}'")

    @property
    def sentinels(self) -> list[str]:
        """Sentinel tokens in id order."""
        return [self.sentinel_template.format(index=i) for i in range(self.num_sentinels)]

    @property
    def ordered(self) -> list[str]:
        """All special tokens in id order."""
        return [self.bos, self.pad, self.eos, self.unk, self.mask, *self.sentinels]


@dataclass(frozen=True)
class TokenizerSpec:
    """Everything that defines a tokenizer's identity.

    Attributes:
        name: Experiment id, e.g. ``"t1-bbpe"``.
        vocab_size: Total vocabulary including special tokens.
        input_mode: Raw or word-segmented input.
        digit_mode: Digit pre-splitting policy.
        min_frequency: Minimum pair frequency for a BPE merge.
        model_max_length: Default maximum sequence length written to the config.
        special_tokens: The shared special-token contract.
    """

    name: str
    vocab_size: int
    input_mode: InputMode = InputMode.RAW
    digit_mode: DigitMode = DigitMode.GROUPED
    min_frequency: int = 2
    model_max_length: int = 2048
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

    def __post_init__(self) -> None:
        minimum = 256 + len(self.special_tokens.ordered)
        if self.vocab_size < minimum:
            raise ValueError(f"vocab_size must be at least {minimum} (bytes + specials)")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TokenizerSpec:
        """Build a spec from a mapping (e.g. parsed YAML)."""
        values = dict(data)
        values["input_mode"] = InputMode(values.get("input_mode", InputMode.RAW.value))
        values["digit_mode"] = DigitMode(values.get("digit_mode", DigitMode.GROUPED.value))
        values["special_tokens"] = SpecialTokens(**values.get("special_tokens", {}))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TokenizerSpec:
        """Load a spec from a YAML file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        data = asdict(self)
        data["input_mode"] = self.input_mode.value
        data["digit_mode"] = self.digit_mode.value
        return data
