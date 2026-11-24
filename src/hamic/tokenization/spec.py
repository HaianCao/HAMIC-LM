"""Tokenizer specification and the shared special-token contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SpecialTokens", "TokenizerSpec"]


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

    HAMIC has one tokenizer pipeline (:func:`hamic.tokenization.build_tokenizer`); a spec
    only sets its size and training details. The pipeline:

    - text first goes through :func:`hamic.tokenization.prepare_text`, which folds
      Cyrillic, Greek and small-capital look-alikes next to a Latin letter ("пgười" to
      "người", one character for one so offsets hold). It runs in Python everywhere HAMIC
      tokenizes (tokenizer training, corpus tokenization, evaluation, fine-tuning): as a
      ``tokenizers`` normalizer it would take one regex pass per letter;
    - normalizer: the corpus pipeline's cleanup (line endings, zero-width characters,
      runs of whitespace incl. NBSP and tabs, blank-line runs, outer whitespace), the
      ``always`` map of ``resources/confusables.yaml`` (``Ð`` to ``Đ``, emoji variation
      selectors removed), NFKC (fullwidth punctuation, "fancy" mathematical letters,
      circled digits; Vietnamese letters unchanged), and a virtual space between a
      newline or an opening bracket/quote and a following letter or digit, so "(Hà Nội)"
      uses the same "␣Hà" as running text (the decoder removes it again);
    - byte-level pre-tokenizer with a prefix space (RoBERTa), so the first word of a text
      uses its running-text token;
    - digits: runs of up to four stay whole (years, the ``000`` of ``1.000.000``); longer
      runs are cut into pairs counted from the right (``09|12|34|56|78``);
    - the strings of ``resources/required_tokens.yaml`` (frequent emoji, symbols,
      emoticons; every Vietnamese letter with a diacritic) and every character at least
      ``CHARACTER_SHARE`` frequent in the training text are added to the training text,
      so each becomes a single token and is never cut into UTF-8 byte pieces.

    Attributes:
        name: Experiment id, e.g. ``"t1b-bbpe-20544"``.
        vocab_size: Total vocabulary including special tokens.
        min_frequency: Minimum pair frequency for a BPE merge.
        model_max_length: Default maximum sequence length written to the config.
        special_tokens: The shared special-token contract.
        required_count: Synthetic occurrences per form (alone, after a space) of each
            required string, Vietnamese letter and frequent character in the training
            text.
    """

    name: str
    vocab_size: int
    min_frequency: int = 2
    model_max_length: int = 2048
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    required_count: int = 50_000

    def __post_init__(self) -> None:
        minimum = 256 + len(self.special_tokens.ordered)
        if self.vocab_size < minimum:
            raise ValueError(f"vocab_size must be at least {minimum} (bytes + specials)")
        if self.required_count < 1:
            raise ValueError("required_count must be positive")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TokenizerSpec:
        """Build a spec from a mapping (e.g. parsed YAML)."""
        values = dict(data)
        values["special_tokens"] = SpecialTokens(**values.get("special_tokens", {}))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TokenizerSpec:
        """Load a spec from a YAML file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)
