"""Full training state for strict resume.

A strict resume restores model weights, optimizer, scheduler, every RNG and the data
cursor. Anything less is a warm start and must use a new run id.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_model, save_model

__all__ = [
    "SAVED_STATE",
    "DataCursor",
    "capture_rng",
    "load_training_state",
    "restore_rng",
    "save_training_state",
]

WEIGHTS_NAME = "model.safetensors"
TRAINER_STATE_NAME = "trainer_state.pt"
CURSOR_NAME = "data_cursor.json"
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


def save_training_state(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    cursor: DataCursor,
) -> Path:
    """Write the complete training state into ``directory`` (which must not exist)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    save_model(model, str(root / WEIGHTS_NAME))  # handles tied weights
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": capture_rng(),
        },
        root / TRAINER_STATE_NAME,
    )
    (root / CURSOR_NAME).write_text(json.dumps(cursor.to_dict(), indent=2), encoding="utf-8")
    return root


def load_training_state(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
) -> DataCursor:
    """Restore state written by :func:`save_training_state` and return the data cursor.

    Only call this on a checkpoint whose manifest has been verified: the trainer state is
    a pickle (it holds NumPy RNG state) and must come from a trusted source.
    """
    root = Path(directory)
    load_model(model, str(root / WEIGHTS_NAME))
    state = torch.load(root / TRAINER_STATE_NAME, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("Checkpoint has no scheduler state but a scheduler was given")
        scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return DataCursor.from_dict(json.loads((root / CURSOR_NAME).read_text(encoding="utf-8")))
