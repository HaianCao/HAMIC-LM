"""A tiny, deterministic training loop that exercises checkpoint and strict resume.

It is the recovery drill required before any long run: train, publish checkpoints to the
object store, stop, start again from the store only, and check that the continued loss
curve is identical to an uninterrupted run.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from hamic.checkpoint.remote import CheckpointPublisher, CheckpointResolver
from hamic.checkpoint.storage import ObjectStore
from hamic.modeling.factory import build_model
from hamic.modeling.presets import Architecture, ModelPreset
from hamic.provenance import code_revision, environment_fingerprint, json_sha256
from hamic.training.compat import check_strict_resume
from hamic.training.state import (
    SAVED_STATE,
    DataCursor,
    load_training_state,
    save_training_state,
)

__all__ = ["SmokeConfig", "SmokeResult", "SyntheticTokenStream", "run_smoke"]


def _tiny_decoder() -> ModelPreset:
    return ModelPreset(
        name="HAMIC-smoke",
        architecture=Architecture.QWEN3,
        label_parameters=1.0e5,
        label_tolerance=10.0,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=512,
        max_position_embeddings=128,
    )


@dataclass(frozen=True)
class SmokeConfig:
    """Settings of the smoke run."""

    run_id: str = "smoke"
    steps: int = 12
    checkpoint_every: int = 4
    max_shard_mib: float = 1024.0
    batch_size: int = 4
    seq_len: int = 32
    learning_rate: float = 1e-3
    seed: int = 1234
    preset: ModelPreset = field(default_factory=_tiny_decoder)

    def fingerprint(self) -> str:
        """Hash of the configuration (part of the run identity)."""
        data = asdict(self)
        data["preset"] = self.preset.to_dict()
        return json_sha256(data)


class SyntheticTokenStream:
    """Deterministic batches addressed by index, so a cursor fully defines the position."""

    def __init__(self, vocab_size: int, batch_size: int, seq_len: int, seed: int) -> None:
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed

    def batch(self, index: int) -> torch.Tensor:
        """Return batch number ``index``."""
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        return torch.randint(
            5, self.vocab_size, (self.batch_size, self.seq_len), generator=generator
        )


@dataclass
class SmokeResult:
    """Outcome of :func:`run_smoke`."""

    losses: dict[int, float]
    resumed_from_step: int | None
    final_step: int
    published_steps: list[int]
    code_commit: str | None = None
    warnings: list[str] = field(default_factory=list)


def run_smoke(
    store: ObjectStore,
    workdir: str | Path,
    config: SmokeConfig | None = None,
    stop_after_step: int | None = None,
) -> SmokeResult:
    """Train (or strictly resume) the smoke model, publishing checkpoints to ``store``.

    Args:
        store: Object store holding the run.
        workdir: Local scratch directory (treated as disposable).
        config: Run settings.
        stop_after_step: Return early after this step, simulating a session ending.
    """
    cfg = config or SmokeConfig()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    run_prefix = f"runs/{cfg.run_id}"

    model = build_model(cfg.preset, seed=cfg.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, cfg.steps // 4))
    )
    stream = SyntheticTokenStream(cfg.preset.vocab_size, cfg.batch_size, cfg.seq_len, cfg.seed)

    restore_dir = work / "restore"
    if restore_dir.exists():
        shutil.rmtree(restore_dir)
    restored = CheckpointResolver(store, run_prefix).restore_latest(restore_dir)
    current = _metadata(cfg, 0, None)
    warnings: list[str] = []
    if restored is not None:
        warnings = check_strict_resume(restored.metadata, current)
        cursor = load_training_state(restore_dir, model, optimizer, scheduler)
        start = restored.step
    else:
        torch.manual_seed(cfg.seed)
        cursor, start = DataCursor(), 0

    publisher = CheckpointPublisher(store, run_prefix)
    losses: dict[int, float] = {}
    published: list[int] = []
    model.train()
    for step in range(start, cfg.steps):
        batch = stream.batch(cursor.offset)
        output = model(input_ids=batch, labels=batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        cursor.offset += 1
        cursor.consumed_tokens += int(batch.numel())
        global_step = step + 1
        losses[global_step] = float(output.loss.detach())

        if global_step % cfg.checkpoint_every == 0 or global_step == cfg.steps:
            local = work / f"step-{global_step:09d}"
            if local.exists():
                shutil.rmtree(local)
            save_training_state(
                local, model, optimizer, scheduler, cursor, max_shard_mib=cfg.max_shard_mib
            )
            publisher.publish(local, _metadata(cfg, global_step, restored))
            published.append(global_step)
        if stop_after_step is not None and global_step >= stop_after_step:
            break

    return SmokeResult(
        losses=losses,
        resumed_from_step=restored.step if restored else None,
        final_step=max(losses, default=start),
        published_steps=published,
        code_commit=current["code_revision"]["commit"],
        warnings=warnings,
    )


def _metadata(cfg: SmokeConfig, global_step: int, restored: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": cfg.run_id,
        "global_step": global_step,
        "config_sha256": cfg.fingerprint(),
        "code_revision": code_revision(),
        "environment": environment_fingerprint(),
        "resumed_from": restored.prefix if restored else None,
        "saved_state": SAVED_STATE,
    }
