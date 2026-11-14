"""Model presets: the HAMIC-level description of a model shape."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from hamic.tokenization.spec import SpecialTokens

__all__ = ["Architecture", "ModelFamily", "ModelPreset"]


class ModelFamily(str, Enum):
    """Topology of the model."""

    ENCODER = "encoder"
    DECODER = "decoder"


class Architecture(str, Enum):
    """Transformers architecture a preset is exported as.

    Every value maps to a class shipped with ``transformers``, so released checkpoints
    load with the ``Auto*`` classes and do not need ``trust_remote_code``.
    """

    ROBERTA = "roberta"
    """Encoder reference profile: post-LN, GELU, learned absolute positions."""
    MODERNBERT = "modernbert"
    """Encoder modern profile: RoPE, GeGLU, pre-LN, no linear bias."""
    LLAMA = "llama"
    """Decoder reference profile: GQA, RoPE, SwiGLU, pre-RMSNorm."""
    QWEN3 = "qwen3"
    """Decoder modern profile: Llama-style plus QK-Norm."""

    @property
    def family(self) -> ModelFamily:
        """Encoder or decoder."""
        if self in {Architecture.ROBERTA, Architecture.MODERNBERT}:
            return ModelFamily.ENCODER
        return ModelFamily.DECODER


@dataclass(frozen=True)
class ModelPreset:
    """Shape and identity of one HAMIC model.

    Attributes:
        name: Canonical model name.
        architecture: Transformers architecture used for export.
        label_parameters: Nominal scale encoded in the name; the runtime count
            must be within :attr:`label_tolerance` of it.
        num_layers: Transformer blocks.
        hidden_size: Model width.
        num_heads: Attention (query) heads.
        intermediate_size: FFN width (per branch for GLU variants).
        vocab_size: Embedding rows; must equal the tokenizer's vocabulary size.
        max_position_embeddings: Maximum context the config allows.
        num_kv_heads: Key/value heads for grouped-query attention (decoders).
        head_dim: Per-head width; defaults to ``hidden_size // num_heads``.
        rope_theta: RoPE base frequency (ignored by RoBERTa).
        norm_eps: Normalization epsilon.
        label_tolerance: Allowed relative deviation from ``label_parameters``.
    """

    name: str
    architecture: Architecture
    label_parameters: float
    num_layers: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    label_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads and self.head_dim is None:
            raise ValueError("hidden_size must be divisible by num_heads unless head_dim is set")
        kv = self.kv_heads
        if self.num_heads % kv:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.architecture.family is ModelFamily.ENCODER and kv != self.num_heads:
            raise ValueError("Encoder presets do not support grouped-query attention")

    @property
    def family(self) -> ModelFamily:
        """Encoder or decoder."""
        return self.architecture.family

    @property
    def kv_heads(self) -> int:
        """Effective number of key/value heads."""
        return self.num_kv_heads or self.num_heads

    @property
    def resolved_head_dim(self) -> int:
        """Effective head dimension."""
        return self.head_dim or self.hidden_size // self.num_heads

    @property
    def special_tokens(self) -> SpecialTokens:
        """The shared special-token contract (ids are fixed by its order)."""
        return SpecialTokens()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelPreset:
        """Build a preset from a mapping (e.g. parsed YAML)."""
        values = dict(data)
        values["architecture"] = Architecture(values["architecture"])
        values["label_parameters"] = float(values["label_parameters"])
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelPreset:
        """Load a preset from a YAML file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        data = asdict(self)
        data["architecture"] = self.architecture.value
        return data
