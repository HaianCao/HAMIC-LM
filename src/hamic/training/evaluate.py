"""Offline masked-LM evaluation of published checkpoints under several masking views.

Validation loss with token-level masking favours models trained with token-level masking:
a model trained to recover whole words or spans never practises predicting one syllable
of a word while the rest of the word is visible (see PMI-Masking, Levine et al. 2021,
Table 4, where single-token perplexity ranks masking schemes opposite to their downstream
scores). This module scores any number of runs on the same validation sequences with the
same seeded masks under three views:

* ``token``: each token alone (the trainer's validation loss);
* ``span``: geometric spans of at most 5 tokens;
* ``pmi``: units joined by high-PMI pairs (whole Vietnamese words and names).

No view is neutral; together they show the trade-off. Downstream tasks decide.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from hamic.checkpoint.manifest import MANIFEST_NAME
from hamic.checkpoint.remote import CheckpointResolver
from hamic.checkpoint.storage import ObjectStore
from hamic.data.tokenized import TokenizedCorpus
from hamic.modeling.factory import build_model
from hamic.modeling.presets import Architecture, ModelPreset
from hamic.provenance import file_sha256
from hamic.training.attention import DocumentMask, document_ids
from hamic.training.config import PretrainConfig
from hamic.training.data import PackedBatches
from hamic.training.masking import (
    MaskRateSchedule,
    PmiUnits,
    SelectionUnits,
    SpanUnits,
    TokenUnits,
    mine_pmi_pairs,
    word_tokens,
)
from hamic.training.objectives import MaskedLM

__all__ = ["PublishedModel", "evaluate_views", "load_published_model", "masking_views"]

_EVAL_SEED_OFFSET = 1 << 40
_INDEX_NAME = "checkpoint_index.json"


class PublishedModel:
    """A run's latest committed weights with the configuration it was trained with.

    Attributes:
        run_id: Run identity.
        step: Optimizer step of the checkpoint.
        config: Training configuration.
        preset: Model preset.
        model: The model in evaluation mode.
    """

    def __init__(
        self, run_id: str, step: int, config: PretrainConfig, preset: ModelPreset, model: Any
    ) -> None:
        self.run_id = run_id
        self.step = step
        self.config = config
        self.preset = preset
        self.model = model


def _config_from_metadata(data: Mapping[str, Any]) -> PretrainConfig:
    names = {f.name for f in fields(PretrainConfig)}
    return PretrainConfig.from_mapping({k: v for k, v in data.items() if k in names})


def load_published_model(
    store: ObjectStore, run_id: str, device: str = "cpu", step: int | None = None
) -> PublishedModel:
    """Download only the weights of a committed checkpoint and build the model.

    Args:
        store: Store holding ``runs/<run_id>/``.
        run_id: Run to load.
        device: Torch device for the model.
        step: Checkpoint step (default: the newest committed one).

    Raises:
        FileNotFoundError: The run has no (matching) committed checkpoint.
        ValueError: A downloaded file does not match the checkpoint manifest.
    """
    committed = CheckpointResolver(store, f"runs/{run_id}").committed()
    if step is not None:
        committed = [c for c in committed if c.step == step]
    if not committed:
        raise FileNotFoundError(f"No committed checkpoint for {run_id!r}")
    checkpoint = committed[-1]
    manifest = json.loads(store.read_bytes(f"{checkpoint.prefix}/{MANIFEST_NAME}") or b"{}")
    checksums = {f["path"]: f["sha256"] for f in manifest["files"]}
    metadata = manifest["metadata"]
    config = _config_from_metadata(metadata["config"])
    preset = ModelPreset.from_mapping(metadata["model"])
    # Sparse prediction gives the same loss with the MLM head on labelled positions only.
    model = build_model(
        preset,
        attn_implementation="sdpa",
        sparse_prediction=preset.architecture is Architecture.MODERNBERT,
    )
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        index_path = store.download(f"{checkpoint.prefix}/{_INDEX_NAME}", root / _INDEX_NAME)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weights: dict[str, torch.Tensor] = {}
        for name in [_INDEX_NAME, *index["weights"]["files"]]:
            path = root / name
            if not path.is_file():
                store.download(f"{checkpoint.prefix}/{name}", path)
            if file_sha256(path) != checksums.get(name):
                raise ValueError(f"{checkpoint.prefix}/{name} does not match its manifest")
            if name != _INDEX_NAME:
                weights.update(load_file(str(path)))
        for alias, name in index["weights"]["aliases"].items():
            weights[alias] = weights[name]
        model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return PublishedModel(run_id, checkpoint.step, config, preset, model)


def masking_views(
    corpus: TokenizedCorpus,
    preset: ModelPreset,
    rate: float = 0.15,
    seed: int = 1234,
    pmi_tokens: int = 100_000_000,
    cache_dir: str | Path | None = None,
    views: Iterable[str] = ("token", "span", "pmi"),
) -> dict[str, MaskedLM]:
    """Seeded validation objectives, one per masking view, shared by every evaluated run."""
    special = range(len(preset.special_tokens.ordered))
    units: dict[str, SelectionUnits] = {}
    for view in views:
        if view == "token":
            units[view] = TokenUnits()
        elif view == "span":
            units[view] = SpanUnits(max_length=5)
        elif view == "pmi":
            units[view] = PmiUnits(
                torch.from_numpy(_pmi_pairs(corpus, preset, pmi_tokens, cache_dir)),
                preset.vocab_size,
                special,
            )
        else:
            raise ValueError(f"Unknown masking view: {view}")
    ordered = preset.special_tokens.ordered
    return {
        view: MaskedLM(
            mask_rate=MaskRateSchedule(rate),
            mask_id=ordered.index(preset.special_tokens.mask),
            vocab_size=preset.vocab_size,
            special_ids=special,
            seed=seed,
            units=unit,
        )
        for view, unit in units.items()
    }


def _pmi_pairs(
    corpus: TokenizedCorpus, preset: ModelPreset, tokens: int, cache_dir: str | Path | None
) -> np.ndarray[Any, Any]:
    digest = corpus.manifest["source_corpus"]["manifest_sha256"][:16]
    cache = Path(cache_dir) / f"pmi-pairs-{digest}.npy" if cache_dir else None
    if cache is not None and cache.is_file():
        cached: np.ndarray[Any, Any] = np.load(cache)
        return cached
    pairs = mine_pmi_pairs(
        corpus.documents("train"),
        preset.vocab_size,
        special_ids=range(len(preset.special_tokens.ordered)),
        max_tokens=tokens,
        eligible=word_tokens(corpus.tokenizer_file()),
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, pairs, allow_pickle=False)
    return pairs


@torch.no_grad()
def evaluate_views(
    published: PublishedModel,
    corpus: TokenizedCorpus,
    views: Mapping[str, MaskedLM],
    sequences: int = 512,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, float]:
    """Mean validation loss of ``published`` under each masking view.

    The first ``sequences`` packed validation rows are used in corpus order, and every view
    draws its masks from fixed seeds, so all runs see identical inputs. Runs trained with
    attention restricted to documents are evaluated the same way.
    """
    preset = published.preset
    ordered = preset.special_tokens.ordered
    bos = ordered.index(preset.special_tokens.bos)
    loader = PackedBatches(
        corpus,
        "validation",
        published.config.seq_len,
        batch_size,
        bos_id=bos,
        eos_id=ordered.index(preset.special_tokens.eos),
        shuffle=False,
    )
    batches = [loader.next_batch() for _ in range(math.ceil(sequences / batch_size))]
    layer_types = tuple(
        getattr(published.model.config, "layer_types", None) or ("full_attention",)
    )
    document_mask = (
        None if published.config.cross_document_attention else DocumentMask("sdpa", layer_types)
    )
    results: dict[str, float] = {}
    for view, objective in views.items():
        total = 0.0
        for index, batch in enumerate(batches):
            inputs: dict[str, Any] = {
                k: v.to(device) for k, v in objective(batch, _EVAL_SEED_OFFSET + index).items()
            }
            if document_mask is not None:
                inputs["attention_mask"] = document_mask(document_ids(batch, bos).to(device))
            total += float(published.model(**inputs).loss)
        results[view] = round(total / len(batches), 6)
    return results
