"""Export a trained model as a standard Hugging Face model directory.

The result loads with plain Transformers, exactly like BERT or RoBERTa::

    from transformers import pipeline
    fill = pipeline("fill-mask", model="path/or/hub-id")
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hamic.modeling.parameters import count_parameters
from hamic.modeling.presets import ModelFamily, ModelPreset
from hamic.provenance import utc_now

__all__ = ["TOKENIZER_FILES", "export_pretrained"]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def export_pretrained(
    model: Any,
    preset: ModelPreset,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    lineage: dict[str, Any] | None = None,
) -> Path:
    """Write weights (safetensors), config, tokenizer and a metadata file.

    Args:
        model: A trained ``transformers`` model built from ``preset``.
        preset: The model preset.
        tokenizer_dir: Directory produced by :func:`hamic.tokenization.export_tokenizer`.
        output_dir: Target directory (must not exist yet).
        lineage: Run id, checkpoint, corpus and tokenizer revisions for the model card.

    Raises:
        ValueError: If the tokenizer vocabulary does not match the model.
    """
    from transformers import AutoTokenizer

    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"{out} already exists; exports are immutable")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if len(tokenizer) != preset.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} entries but the model expects {preset.vocab_size}"
        )
    if getattr(model.config, "sparse_prediction", False):
        # Training-time shortcut only; released models predict every position.
        model.config.sparse_prediction = False
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    metadata_file = Path(tokenizer_dir) / "hamic_tokenizer.json"
    if metadata_file.exists():
        shutil.copyfile(metadata_file, out / metadata_file.name)
    if preset.family is ModelFamily.DECODER:
        model.generation_config.save_pretrained(out)

    total, trainable, embedding = count_parameters(model)
    info = {
        "name": preset.name,
        "preset": preset.to_dict(),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "embedding_parameters": embedding,
        "exported_at_utc": utc_now(),
        "lineage": lineage or {},
    }
    (out / "hamic_model.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
