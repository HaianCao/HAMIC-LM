"""Pretraining loop for HAMIC encoders (MLM) and decoders (causal LM).

One :class:`Pretrainer` run is one session: it restores the newest committed checkpoint
of the run (strict resume), trains until the token budget is reached or the session's
soft deadline, publishes checkpoints to the object store along the way, and marks the run
``COMPLETE`` at the end.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hamic.checkpoint.remote import CheckpointPublisher, CheckpointResolver
from hamic.checkpoint.storage import ObjectStore
from hamic.data.tokenized import TokenizedCorpus
from hamic.modeling.factory import build_model
from hamic.modeling.presets import Architecture, ModelPreset
from hamic.provenance import code_revision, environment_fingerprint, utc_now
from hamic.runtime.session import HOUR, SessionClock
from hamic.training.attention import DocumentMask, document_ids
from hamic.training.compat import check_strict_resume
from hamic.training.config import PretrainConfig
from hamic.training.data import PackedBatches
from hamic.training.masking import (
    MaskRateSchedule,
    PmiUnits,
    SelectionUnits,
    SpanUnits,
    TokenUnits,
    VariantTable,
    mine_pmi_pairs,
    word_tokens,
)
from hamic.training.objectives import CausalLM, MaskedLM, Objective
from hamic.training.state import SAVED_STATE, load_training_state, save_training_state

__all__ = ["Pretrainer", "TrainResult"]

COMPLETE_NAME = "COMPLETE.json"
_EVAL_SEED_OFFSET = 1 << 40
_PMI_TOKENS = 100_000_000


@dataclass
class TrainResult:
    """Outcome of one session.

    Attributes:
        status: ``"completed"``, ``"paused"`` (soft deadline), ``"interrupted"`` (stopped
            without a checkpoint, used by drills) or ``"already_complete"``.
        final_step: Last optimizer step reached in this session.
        resumed_from_step: Step restored at the start (``None`` for a fresh run).
        losses: Training loss per optimizer step of this session.
        evaluations: Validation loss per evaluated step.
        eval_bits_per_byte: Validation bits per UTF-8 byte per evaluated step (causal LM
            with a byte-level tokenizer copy next to the shards); comparable across
            tokenizers, unlike loss per token.
        published_steps: Checkpoints committed in this session.
        warnings: Non-fatal resume warnings.
    """

    status: str
    final_step: int
    resumed_from_step: int | None
    losses: dict[int, float] = field(default_factory=dict)
    evaluations: dict[int, float] = field(default_factory=dict)
    eval_bits_per_byte: dict[int, float] = field(default_factory=dict)
    published_steps: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def lr_schedule(config: PretrainConfig) -> Callable[[int], float]:
    """Learning-rate multiplier per optimizer step (for ``LambdaLR``).

    ``cosine``: linear warmup, then cosine from the peak to ``min_lr_ratio``.
    ``wsd``: linear warmup, constant peak, then a linear decay to ``min_lr_ratio`` over the
    last ``decay_ratio`` of the steps. The constant phase does not depend on the total, so a
    run can be extended (or a decay branched off a checkpoint) without changing its past.
    """
    total, warmup, floor = config.total_steps, config.warmup_steps, config.min_lr_ratio
    decay_start = max(warmup, total - max(1, round(total * config.decay_ratio)))

    def cosine(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    def wsd(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start + 1) / max(1, total - decay_start))
        return 1.0 - (1.0 - floor) * progress

    return wsd if config.lr_schedule == "wsd" else cosine


class Pretrainer:
    """Run (or resume) one pretraining session.

    Args:
        config: Run configuration.
        store: Object store holding ``runs/<run_id>/``.
        workdir: Local scratch directory (checkpoints are staged here, then uploaded).
        clock: Session clock; defaults to one built from the configuration.
        log: Callable receiving one metrics dict per logged step (default: print JSON).
    """

    def __init__(
        self,
        config: PretrainConfig,
        store: ObjectStore,
        workdir: str | Path,
        clock: SessionClock | None = None,
        log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SessionClock(
            hard_limit=config.session_hard_hours * HOUR,
            soft_limit=config.session_soft_hours * HOUR,
        )
        self.log = log or (lambda record: print(json.dumps(record), flush=True))
        self.run_prefix = f"runs/{config.run_id}"
        self.preset = ModelPreset.from_yaml(config.model_preset)
        self.corpus = TokenizedCorpus(config.tokenized, verify=True)
        if self.preset.vocab_size != self.corpus.vocab_size:
            raise ValueError(
                f"Preset vocab {self.preset.vocab_size} != tokenizer vocab "
                f"{self.corpus.vocab_size}"
            )
        expected = "mlm" if self.preset.family.value == "encoder" else "clm"
        if config.objective != expected:
            raise ValueError(f"{self.preset.name} is trained with {expected!r}")
        wanted = config.device
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if wanted == "auto" else wanted
        )
        if not config.cross_document_attention and (
            self.preset.architecture is not Architecture.MODERNBERT
        ):
            raise ValueError("cross_document_attention=False is supported for ModernBERT only")
        self._pending_metrics: list[dict[str, Any]] = []
        self._byte_lengths: torch.Tensor | None = None
        if config.objective == "clm":
            try:
                self._byte_lengths = torch.from_numpy(self.corpus.token_byte_lengths())
            except FileNotFoundError:
                self._byte_lengths = None

    # -- construction ------------------------------------------------------------------

    def _objective(self) -> Objective:
        """Training objective with every configured masking option."""
        cfg = self.config
        if cfg.objective == "clm":
            return CausalLM()
        micro_steps = cfg.total_steps * cfg.grad_accum_steps
        schedule = MaskRateSchedule(
            start_rate=cfg.mask_rate,
            end_rate=cfg.mask_rate_final,
            start_step=round(cfg.mask_schedule_start * micro_steps),
            end_step=micro_steps,
        )
        return self._masked_lm(
            schedule,
            units=self._selection_units(),
            variant_share=cfg.variant_share,
            variants=VariantTable.from_tokenizer_file(self.corpus.tokenizer_file())
            if cfg.variant_share
            else None,
        )

    def _eval_objective(self) -> Objective:
        """Plain token-level MLM at a fixed rate, so validation loss stays comparable."""
        cfg = self.config
        if cfg.objective == "clm":
            return CausalLM()
        return self._masked_lm(MaskRateSchedule(cfg.eval_mask_rate or cfg.mask_rate))

    def _masked_lm(self, schedule: MaskRateSchedule, **options: Any) -> MaskedLM:
        tokens = self.preset.special_tokens
        ordered = tokens.ordered
        return MaskedLM(
            mask_rate=schedule,
            mask_id=ordered.index(tokens.mask),
            vocab_size=self.preset.vocab_size,
            special_ids=range(len(ordered)),
            seed=self.config.seed,
            **options,
        )

    def _selection_units(self) -> SelectionUnits:
        special = range(len(self.preset.special_tokens.ordered))
        if self.config.masking == "span":
            return SpanUnits(max_length=self.config.span_max_length)
        if self.config.masking == "pmi":
            pairs = torch.from_numpy(self._pmi_pairs())
            return PmiUnits(
                pairs, self.preset.vocab_size, special, keep=self.config.pmi_unit_share
            )
        return TokenUnits()

    def _pmi_pairs(self) -> np.ndarray[Any, Any]:
        """Collocations mined from the first train tokens; cached per corpus revision."""
        digest = self.corpus.manifest["source_corpus"]["manifest_sha256"][:16]
        cache = self.workdir / f"pmi-pairs-{digest}.npy"
        if cache.is_file():
            cached: np.ndarray[Any, Any] = np.load(cache)
            return cached
        pairs = mine_pmi_pairs(
            self.corpus.documents("train"),
            self.preset.vocab_size,
            special_ids=range(len(self.preset.special_tokens.ordered)),
            max_tokens=_PMI_TOKENS,
            eligible=word_tokens(self.corpus.tokenizer_file()),
        )
        np.save(cache, pairs, allow_pickle=False)
        return pairs

    def _optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=self.config.betas,
            fused=self.device.type == "cuda",
        )

    def _metadata(self, step: int, consumed: int, restored: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.config.run_id,
            "global_step": step,
            "consumed_tokens": consumed,
            "config_sha256": self.config.identity_sha256(),
            "config": self.config.to_dict(),
            "model": self.preset.to_dict(),
            "tokenizer": self.corpus.manifest["tokenizer"],
            "source_corpus": self.corpus.manifest["source_corpus"],
            "code_revision": code_revision(),
            "environment": environment_fingerprint(),
            "resumed_from": restored.prefix if restored else None,
            "saved_state": SAVED_STATE,
            "created_at_utc": utc_now(),
        }

    # -- main loop -----------------------------------------------------------------------

    def run(self, stop_after_steps: int | None = None) -> TrainResult:
        """Train until done or the soft deadline; return what this session did.

        Args:
            stop_after_steps: Stop after this many steps of this session **without** a
                checkpoint, simulating a crash (used by recovery drills and tests).
        """
        cfg = self.config
        if self.store.read_bytes(f"{self.run_prefix}/{COMPLETE_NAME}") is not None:
            return TrainResult(
                "already_complete", final_step=cfg.total_steps, resumed_from_step=None
            )
        torch.use_deterministic_algorithms(cfg.deterministic)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = build_model(
            self.preset,
            seed=cfg.seed,
            attn_implementation=cfg.attention,
            sparse_prediction=cfg.sparse_prediction,
        ).to(self.device)
        layer_types = tuple(getattr(model.config, "layer_types", None) or ("full_attention",))
        document_mask = (
            None if cfg.cross_document_attention else DocumentMask(cfg.attention, layer_types)
        )
        # The compiled wrapper shares parameters with ``model``; state is saved from and
        # evaluation runs on the eager module.
        train_model = torch.compile(model, dynamic=False) if cfg.compile else model
        optimizer = self._optimizer(model)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule(cfg))
        objective = self._objective()
        eval_objective = self._eval_objective()

        restore_dir = self.workdir / "restore"
        shutil.rmtree(restore_dir, ignore_errors=True)
        restored = CheckpointResolver(self.store, self.run_prefix).restore_latest(restore_dir)
        warnings: list[str] = []
        cursor = None
        step = 0
        if restored is not None:
            warnings = check_strict_resume(restored.metadata, self._metadata(0, 0, None))
            cursor = load_training_state(restore_dir, model, optimizer, scheduler)
            step = restored.step
            shutil.rmtree(restore_dir, ignore_errors=True)
        else:
            torch.manual_seed(cfg.seed)

        special = self.preset.special_tokens
        ordered = special.ordered
        data = PackedBatches(
            self.corpus,
            "train",
            cfg.seq_len,
            cfg.micro_batch_size,
            bos_id=ordered.index(special.bos),
            eos_id=ordered.index(special.eos),
            seed=cfg.seed,
            shuffle=cfg.shuffle,
            cursor=cursor,
            source_weights=cfg.source_weights,
        )
        bos_id = ordered.index(special.bos)
        eval_batches = self._eval_batches(bos_id, ordered.index(special.eos))
        autocast = cfg.precision == "bf16" and self.device.type == "cuda"
        publisher = CheckpointPublisher(self.store, self.run_prefix)
        result = TrainResult(
            "completed",
            final_step=step,
            resumed_from_step=restored.step if restored else None,
            warnings=warnings,
        )
        last_checkpoint_step = step
        last_checkpoint_time = time.monotonic()
        session_steps = 0
        model.train()

        # Per-step values stay on the device and are read back in one transfer per log
        # window, so the host never waits for the GPU between steps.
        pending: list[tuple[int, torch.Tensor, torch.Tensor, float, int]] = []
        window = (time.perf_counter(), data.consumed_tokens)
        pin = self.device.type == "cuda"

        while step < cfg.total_steps:
            loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            for micro in range(cfg.grad_accum_steps):
                batch = data.next_batch()
                inputs = self._to_device(
                    objective(batch, step * cfg.grad_accum_steps + micro),
                    batch,
                    bos_id,
                    document_mask,
                    pin,
                )
                with torch.autocast(self.device.type, torch.bfloat16, enabled=autocast):
                    loss = train_model(**inputs).loss / cfg.grad_accum_steps
                loss.backward()
                loss_sum += loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            session_steps += 1
            pending.append(
                (
                    step,
                    loss_sum,
                    grad_norm.detach(),
                    float(scheduler.get_last_lr()[0]),
                    data.consumed_tokens,
                )
            )

            finished = step >= cfg.total_steps
            interrupted = stop_after_steps is not None and session_steps >= stop_after_steps
            due = (cfg.checkpoint_every_steps and step % cfg.checkpoint_every_steps == 0) or (
                cfg.checkpoint_every_minutes
                and time.monotonic() - last_checkpoint_time >= cfg.checkpoint_every_minutes * 60
            )
            pause = self.clock.should_stop()
            evaluate = bool(cfg.eval_every_steps and step % cfg.eval_every_steps == 0) or (
                finished and not interrupted
            )
            log_now = step % cfg.log_every_steps == 0 or finished
            if not (log_now or evaluate or interrupted or due or pause):
                continue

            record, window = self._flush(pending, result, window)
            if evaluate:
                record.update(
                    self._evaluate(
                        model, eval_objective, eval_batches, autocast, bos_id, document_mask
                    )
                )
                self._store_eval(result, step, record)
                window = (time.perf_counter(), data.consumed_tokens)  # exclude eval time
            if log_now or evaluate:
                self.log(record)
            pause = pause or self.clock.should_stop()
            if interrupted:
                result.status = "interrupted"
                break
            if (finished or due or pause) and step != last_checkpoint_step:
                self._checkpoint(model, optimizer, scheduler, data, step, restored, publisher)
                result.published_steps.append(step)
                last_checkpoint_step, last_checkpoint_time = step, time.monotonic()
                window = (time.perf_counter(), data.consumed_tokens)
            if pause and not finished:
                result.status = "paused"
                break

        result.final_step = step
        if result.status == "completed":
            marker = {"run_id": cfg.run_id, "final_step": step, "completed_at_utc": utc_now()}
            self.store.put_bytes(
                f"{self.run_prefix}/{COMPLETE_NAME}",
                (json.dumps(marker, indent=2) + "\n").encode("utf-8"),
                message=f"run {cfg.run_id} complete",
            )
        return result

    # -- helpers ---------------------------------------------------------------------------

    def _to_device(
        self,
        inputs: dict[str, torch.Tensor],
        batch: torch.Tensor,
        bos_id: int,
        document_mask: DocumentMask | None,
        pin: bool,
    ) -> dict[str, Any]:
        moved: dict[str, Any] = {
            k: (v.pin_memory() if pin else v).to(self.device, non_blocking=pin)
            for k, v in inputs.items()
        }
        if document_mask is not None:
            documents = document_ids(batch, bos_id)
            moved["attention_mask"] = document_mask(
                (documents.pin_memory() if pin else documents).to(self.device, non_blocking=pin)
            )
        return moved

    def _eval_batches(self, bos_id: int, eos_id: int) -> list[torch.Tensor]:
        cfg = self.config
        loader = PackedBatches(
            self.corpus,
            "validation",
            cfg.seq_len,
            cfg.micro_batch_size,
            bos_id=bos_id,
            eos_id=eos_id,
            shuffle=False,
        )
        count = max(1, math.ceil(cfg.eval_sequences / cfg.micro_batch_size))
        return [loader.next_batch() for _ in range(count)]

    def _flush(
        self,
        pending: list[tuple[int, torch.Tensor, torch.Tensor, float, int]],
        result: TrainResult,
        window: tuple[float, int],
    ) -> tuple[dict[str, Any], tuple[float, int]]:
        """Read back pending step values in one transfer; return the last record."""
        values = torch.stack([torch.stack([loss, norm.double()]) for _, loss, norm, *_ in pending])
        now = time.perf_counter()
        record: dict[str, Any] = {}
        for (step, _, _, lr, tokens), (loss, norm) in zip(pending, values.tolist(), strict=True):
            result.losses[step] = loss
            record = {
                "step": step,
                "loss": round(loss, 6),
                "lr": lr,
                "grad_norm": round(norm, 4),
                "tokens": tokens,
            }
            self._pending_metrics.append(record)
        started, start_tokens = window
        record["tokens_per_second"] = round((record["tokens"] - start_tokens) / (now - started), 1)
        if self.device.type == "cuda":
            record["max_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        pending.clear()
        return record, (now, record["tokens"])

    @staticmethod
    def _store_eval(result: TrainResult, step: int, record: dict[str, Any]) -> None:
        result.evaluations[step] = record["eval_loss"]
        if "eval_bpb" in record:
            result.eval_bits_per_byte[step] = record["eval_bpb"]

    @torch.no_grad()
    def _evaluate(
        self,
        model: torch.nn.Module,
        objective: Objective,
        batches: list[torch.Tensor],
        autocast: bool,
        bos_id: int,
        document_mask: DocumentMask | None,
    ) -> dict[str, float]:
        model.eval()
        total = 0.0
        nats = 0.0
        target_bytes = 0
        for index, batch in enumerate(batches):
            inputs = self._to_device(
                objective(batch, _EVAL_SEED_OFFSET + index), batch, bos_id, document_mask, False
            )
            with torch.autocast(self.device.type, torch.bfloat16, enabled=autocast):
                loss = float(model(**inputs).loss)
            total += loss
            if self._byte_lengths is not None:
                # Causal LM: the mean runs over every position but the first of each row.
                targets = batch[:, 1:]
                nats += loss * targets.numel()
                target_bytes += int(self._byte_lengths[targets].sum())
        model.train()
        metrics = {"eval_loss": round(total / len(batches), 6)}
        if target_bytes:
            metrics["eval_bpb"] = round(nats / (math.log(2) * target_bytes), 6)
        return metrics

    def _checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        data: PackedBatches,
        step: int,
        restored: Any,
        publisher: CheckpointPublisher,
    ) -> None:
        started = time.monotonic()
        local = self.workdir / f"step-{step:09d}"
        shutil.rmtree(local, ignore_errors=True)
        save_training_state(
            local, model, optimizer, scheduler, data.cursor(), self.config.max_shard_mib
        )
        with (local / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for record in self._pending_metrics:
                stream.write(json.dumps(record) + "\n")
        publisher.publish(local, self._metadata(step, data.consumed_tokens, restored))
        shutil.rmtree(local, ignore_errors=True)
        self._pending_metrics = []
        self.clock.record_checkpoint(time.monotonic() - started)
