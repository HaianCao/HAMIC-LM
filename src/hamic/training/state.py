"""Full training state for strict resume, stored as size-capped shards.

A strict resume restores model weights, optimizer, scheduler, every RNG and the data
cursor. Anything less is a warm start and must use a new run id.

Weights and optimizer state are split into shards of at most ``max_shard_mib`` so a
checkpoint can be downloaded with several parallel transfers (one Hub file transfers
far slower than several). ``checkpoint_index.json`` lists the shards. Checkpoints written
before sharding (one weights file, one trainer-state file) still load.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, load_model, save_file

__all__ = [
    "SAVED_STATE",
    "DataCursor",
    "capture_rng",
    "load_training_state",
    "restore_rng",
    "save_training_state",
]

INDEX_NAME = "checkpoint_index.json"
INDEX_FORMAT = "hamic-sharded-v1"
TRAINER_META_NAME = "trainer_meta.pt"
CURSOR_NAME = "data_cursor.json"
LEGACY_WEIGHTS_NAME = "model.safetensors"
LEGACY_TRAINER_STATE_NAME = "trainer_state.pt"
DEFAULT_MAX_SHARD_MIB = 1024.0
SAVED_STATE = {
    "model": True,
    "optimizer": True,
    "scheduler": True,
    "rng": True,
    "data_position": True,
}


@dataclass
class DataCursor:
    """Position in the training stream.

    Attributes:
        shard_index: Index of the current shard.
        offset: Next sequence/batch index inside the shard.
        consumed_tokens: Real (non-padding) training tokens seen so far.
        epoch: Pass over the data.
    """

    shard_index: int = 0
    offset: int = 0
    consumed_tokens: int = 0
    epoch: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-friendly view."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataCursor:
        """Rebuild from :meth:`to_dict` output."""
        return cls(**{k: int(v) for k, v in data.items()})


def capture_rng() -> dict[str, Any]:
    """Snapshot Python, NumPy, Torch CPU and all CUDA generators."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Restore generators captured by :func:`capture_rng`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _chunks(sizes: Iterable[tuple[Any, int]], limit: int) -> Iterator[list[Any]]:
    """Group ``(key, nbytes)`` items into consecutive chunks of at most ``limit`` bytes."""
    chunk: list[Any] = []
    total = 0
    for key, nbytes in sizes:
        if chunk and total + nbytes > limit:
            yield chunk
            chunk, total = [], 0
        chunk.append(key)
        total += nbytes
    if chunk:
        yield chunk


def _storage_key(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        tensor.untyped_storage().data_ptr(),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _save_weights(model: torch.nn.Module, root: Path, limit: int) -> dict[str, Any]:
    """Save unique weight tensors in shards; tied names are recorded as aliases."""
    canonical: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    unique: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    for name, tensor in model.state_dict().items():
        key = _storage_key(tensor)
        if key in canonical:
            aliases[name] = canonical[key]
        else:
            canonical[key] = name
            unique[name] = tensor.detach().cpu().contiguous()
    groups = list(_chunks(((n, t.nbytes) for n, t in unique.items()), limit))
    files = []
    for index, names in enumerate(groups):
        filename = f"weights-{index + 1:05d}-of-{len(groups):05d}.safetensors"
        save_file({n: unique[n] for n in names}, str(root / filename))
        files.append(filename)
    return {"files": files, "aliases": aliases}


def _save_optimizer(optimizer: torch.optim.Optimizer, root: Path, limit: int) -> dict[str, Any]:
    """Save per-parameter optimizer state in shards; param groups go to the meta file."""
    state = optimizer.state_dict()["state"]

    def nbytes(entry: dict[str, Any]) -> int:
        return sum(v.nbytes for v in entry.values() if isinstance(v, torch.Tensor))

    groups = list(_chunks(((idx, nbytes(entry)) for idx, entry in state.items()), limit))
    files = []
    for index, keys in enumerate(groups):
        filename = f"optimizer-{index + 1:05d}-of-{len(groups):05d}.pt"
        torch.save({k: state[k] for k in keys}, root / filename)
        files.append(filename)
    return {"files": files}


def save_training_state(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    cursor: DataCursor,
    max_shard_mib: float = DEFAULT_MAX_SHARD_MIB,
) -> Path:
    """Write the complete training state into ``directory`` (which must not exist).

    Args:
        directory: Target directory.
        model: Model being trained.
        optimizer: Its optimizer.
        scheduler: Learning-rate scheduler, if any.
        cursor: Data position.
        max_shard_mib: Upper bound for one weights or optimizer shard. A tensor larger
            than the bound gets a shard of its own.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    limit = max(1, int(max_shard_mib * 2**20))
    index = {
        "format": INDEX_FORMAT,
        "weights": _save_weights(model, root, limit),
        "optimizer": _save_optimizer(optimizer, root, limit),
        "trainer_meta": TRAINER_META_NAME,
        "cursor": CURSOR_NAME,
    }
    torch.save(
        {
            "param_groups": optimizer.state_dict()["param_groups"],
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": capture_rng(),
        },
        root / TRAINER_META_NAME,
    )
    (root / CURSOR_NAME).write_text(json.dumps(cursor.to_dict(), indent=2), encoding="utf-8")
    (root / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _load_sharded(
    root: Path, index: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    if index.get("format") != INDEX_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {index.get('format')!r}")
    weights: dict[str, torch.Tensor] = {}
    for filename in index["weights"]["files"]:
        weights.update(load_file(str(root / filename)))
    for alias, name in index["weights"]["aliases"].items():
        weights[alias] = weights[name]
    model.load_state_dict(weights, strict=True)

    meta: dict[str, Any] = torch.load(root / index["trainer_meta"], weights_only=False)
    state: dict[Any, Any] = {}
    for filename in index["optimizer"]["files"]:
        state.update(torch.load(root / filename, weights_only=False))
    optimizer.load_state_dict({"state": state, "param_groups": meta["param_groups"]})
    return meta


def _load_legacy(
    root: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    load_model(model, str(root / LEGACY_WEIGHTS_NAME))
    meta: dict[str, Any] = torch.load(root / LEGACY_TRAINER_STATE_NAME, weights_only=False)
    optimizer.load_state_dict(meta["optimizer"])
    return meta


def load_training_state(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
) -> DataCursor:
    """Restore state written by :func:`save_training_state` and return the data cursor.

    Only call this on a checkpoint whose manifest has been verified: optimizer and
    trainer metadata are pickles (they hold NumPy RNG state) and must be trusted.
    """
    root = Path(directory)
    index_path = root / INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        meta = _load_sharded(root, index, model, optimizer)
    else:
        meta = _load_legacy(root, model, optimizer)
    if scheduler is not None:
        if meta["scheduler"] is None:
            raise ValueError("Checkpoint has no scheduler state but a scheduler was given")
        scheduler.load_state_dict(meta["scheduler"])
    restore_rng(meta["rng"])
    return DataCursor.from_dict(json.loads((root / CURSOR_NAME).read_text(encoding="utf-8")))
