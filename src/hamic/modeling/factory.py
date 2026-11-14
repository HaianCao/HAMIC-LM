"""Turn a :class:`ModelPreset` into a standard Transformers config and model.

Each architecture has one builder registered in :data:`CONFIG_BUILDERS`. Supporting a new
architecture means adding a builder; existing builders do not change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from hamic.modeling.presets import Architecture, ModelFamily, ModelPreset

__all__ = ["CONFIG_BUILDERS", "build_config", "build_model"]

ConfigBuilder = Callable[[ModelPreset], Any]
CONFIG_BUILDERS: dict[Architecture, ConfigBuilder] = {}


def _register(architecture: Architecture) -> Callable[[ConfigBuilder], ConfigBuilder]:
    def decorator(builder: ConfigBuilder) -> ConfigBuilder:
        CONFIG_BUILDERS[architecture] = builder
        return builder

    return decorator


class _TokenIds(TypedDict):
    bos_token_id: int
    pad_token_id: int
    eos_token_id: int


def _token_ids(preset: ModelPreset) -> _TokenIds:
    ordered = preset.special_tokens.ordered
    st = preset.special_tokens
    return {
        "bos_token_id": ordered.index(st.bos),
        "pad_token_id": ordered.index(st.pad),
        "eos_token_id": ordered.index(st.eos),
    }


@_register(Architecture.ROBERTA)
def _roberta(preset: ModelPreset) -> Any:
    from transformers import RobertaConfig

    ids = _token_ids(preset)
    return RobertaConfig(
        vocab_size=preset.vocab_size,
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_layers,
        num_attention_heads=preset.num_heads,
        intermediate_size=preset.intermediate_size,
        hidden_act="gelu",
        # RoBERTa offsets positions by padding_idx + 1.
        max_position_embeddings=preset.max_position_embeddings + ids["pad_token_id"] + 1,
        type_vocab_size=1,
        layer_norm_eps=preset.norm_eps,
        tie_word_embeddings=True,
        **ids,
    )


@_register(Architecture.MODERNBERT)
def _modernbert(preset: ModelPreset) -> Any:
    from transformers import ModernBertConfig

    ids = _token_ids(preset)
    rope = {"rope_type": "default", "rope_theta": preset.rope_theta}
    return ModernBertConfig(
        vocab_size=preset.vocab_size,
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        num_hidden_layers=preset.num_layers,
        num_attention_heads=preset.num_heads,
        hidden_activation="gelu",
        max_position_embeddings=preset.max_position_embeddings,
        norm_eps=preset.norm_eps,
        norm_bias=False,
        attention_bias=False,
        mlp_bias=False,
        # Full bidirectional attention in every layer: HAMIC v1 contexts are <= 2K.
        layer_types=["full_attention"] * preset.num_layers,
        rope_parameters={"full_attention": dict(rope), "sliding_attention": dict(rope)},
        cls_token_id=ids["bos_token_id"],
        sep_token_id=ids["eos_token_id"],
        tie_word_embeddings=True,
        **ids,
    )


def _decoder_kwargs(preset: ModelPreset) -> dict[str, Any]:
    return {
        "vocab_size": preset.vocab_size,
        "hidden_size": preset.hidden_size,
        "intermediate_size": preset.intermediate_size,
        "num_hidden_layers": preset.num_layers,
        "num_attention_heads": preset.num_heads,
        "num_key_value_heads": preset.kv_heads,
        "head_dim": preset.resolved_head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": preset.max_position_embeddings,
        "rms_norm_eps": preset.norm_eps,
        "attention_bias": False,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": preset.rope_theta},
        **_token_ids(preset),
    }


@_register(Architecture.LLAMA)
def _llama(preset: ModelPreset) -> Any:
    from transformers import LlamaConfig

    return LlamaConfig(mlp_bias=False, **_decoder_kwargs(preset))


@_register(Architecture.QWEN3)
def _qwen3(preset: ModelPreset) -> Any:
    from transformers import Qwen3Config

    return Qwen3Config(
        use_sliding_window=False,
        layer_types=["full_attention"] * preset.num_layers,
        **_decoder_kwargs(preset),
    )


def build_config(preset: ModelPreset) -> Any:
    """Return the Transformers config for ``preset``."""
    try:
        builder = CONFIG_BUILDERS[preset.architecture]
    except KeyError as exc:
        raise NotImplementedError(f"No builder for {preset.architecture.value}") from exc
    config = builder(preset)
    config.architectures = [
        _model_class(preset.family, config).__name__,
    ]
    return config


def _model_class(family: ModelFamily, config: Any) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

    auto = AutoModelForMaskedLM if family is ModelFamily.ENCODER else AutoModelForCausalLM
    return auto._model_mapping[type(config)]


def build_model(preset: ModelPreset, device: str = "cpu", seed: int | None = None) -> Any:
    """Instantiate the pretraining model (masked LM or causal LM) for ``preset``.

    Args:
        preset: Model preset.
        device: ``"meta"`` builds shapes only (no memory) for parameter counting.
        seed: Optional seed for reproducible initialization.
    """
    import torch

    config = build_config(preset)
    model_class = _model_class(preset.family, config)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.device(device):
        return model_class(config)
