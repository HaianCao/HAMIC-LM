"""Pretraining run configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from hamic.provenance import json_sha256

__all__ = ["PretrainConfig"]

# Fields that change the training trajectory. Logging, evaluation, checkpoint cadence and
# session limits are operational and may change between sessions of the same run.
_IDENTITY_FIELDS = (
    "model_preset",
    "objective",
    "mask_rate",
    "seq_len",
    "micro_batch_size",
    "grad_accum_steps",
    "total_tokens",
    "learning_rate",
    "min_lr_ratio",
    "warmup_ratio",
    "betas",
    "weight_decay",
    "grad_clip",
    "precision",
    "seed",
    "shuffle",
)

# Trajectory fields added after the first runs. They join the identity only when they
# differ from their default, so the hashes of earlier runs stay valid.
_OPTIONAL_IDENTITY_FIELDS = (
    "lr_schedule",
    "decay_ratio",
    "cross_document_attention",
    "source_weights",
    "mask_rate_final",
    "mask_schedule_start",
    "masking",
    "span_max_length",
    "pmi_unit_share",
    "variant_share",
)

LR_SCHEDULES = ("cosine", "wsd")
MASKING_UNITS = ("token", "span", "pmi")
ATTENTION_BACKENDS = ("sdpa", "flex_attention")


@dataclass(frozen=True)
class PretrainConfig:
    """Settings of one pretraining run.

    Attributes:
        run_id: Run identity; checkpoints live under ``runs/<run_id>/``.
        model_preset: Path of the model preset YAML.
        tokenized: Directory written by ``hamic corpus tokenize``.
        objective: ``"mlm"`` (encoders) or ``"clm"`` (decoders).
        mask_rate: MLM selection rate (0.15 control, 0.30 candidate).
        seq_len: Tokens per packed sequence.
        micro_batch_size: Sequences per forward pass.
        grad_accum_steps: Micro-batches per optimizer step.
        total_tokens: Training budget in tokens; sets the number of optimizer steps.
        learning_rate: Peak learning rate.
        min_lr_ratio: Final learning rate as a fraction of the peak (cosine floor).
        warmup_ratio: Share of steps with linear warmup.
        betas: AdamW betas.
        weight_decay: AdamW weight decay (not applied to biases and norms).
        grad_clip: Global gradient-norm clip.
        precision: ``"bf16"`` (autocast on CUDA) or ``"fp32"``.
        seed: Seed for initialization, data order and masking.
        shuffle: Shuffle documents per epoch.
        cross_document_attention: Packed rows attend across document boundaries. ``False``
            restricts attention to each document (ModernBERT encoders only).
        lr_schedule: ``"cosine"`` (warmup then cosine to the floor) or ``"wsd"``
            (warmup, constant peak, then a linear decay to the floor over the last
            :attr:`decay_ratio` of the steps).
        decay_ratio: Share of steps in the WSD decay phase.
        source_weights: Per-source sampling multipliers for the train split, e.g.
            ``{"wikipedia-vi-20231101": 2.0}`` visits each Wikipedia document twice per
            epoch (fractions are sampled per epoch). Needs per-document source ids in the
            tokenized shards.
        mask_rate_final: MLM rate at the last step; the rate moves linearly from
            :attr:`mask_rate` to this value, starting at :attr:`mask_schedule_start`
            (share of steps). ``None`` keeps the rate constant.
        mask_schedule_start: Share of steps before the mask rate starts to change.
        span_max_length: Longest span of ``masking: span`` (geometric, ``p = 0.2``).
        pmi_unit_share: ``masking: pmi`` only: probability that each collocation occurrence
            is masked as one unit (else its tokens are independent units). ``1.0`` is pure
            PMI masking; ``0.5`` mixes word-level and token-level practice.
        masking: MLM selection unit: ``"token"``, ``"span"`` (geometric spans, SpanBERT)
            or ``"pmi"`` (runs of tokens joined by high-PMI pairs mined from the train
            split, a segmenter-free proxy for Vietnamese multi-syllable words).
        variant_share: Share of selected MLM positions replaced by a Vietnamese variant of
            the same token (other tone, no diacritics, teencode, other casing) while the
            label stays the original. Taken first from the 10% random replacements, then
            from the 80% ``<mask>`` replacements.
        attention: Attention kernel, ``"sdpa"`` or ``"flex_attention"`` (operational).
        sparse_prediction: Compute the MLM head only at labelled positions (ModernBERT;
            same loss, less compute; operational).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        deterministic: Force deterministic kernels (slower; used by tests and drills).
        compile: Wrap the model in ``torch.compile`` for training (fused kernels; the first
            steps pay the compilation time). Operational: it does not change the math.
        log_every_steps: Metric logging cadence.
        eval_every_steps: Validation cadence (0 disables periodic evaluation).
        eval_sequences: Validation sequences per evaluation.
        eval_mask_rate: MLM rate used for validation (``None``: :attr:`mask_rate`). A fixed
            value keeps validation loss comparable between runs with different mask rates.
        checkpoint_every_steps: Checkpoint cadence in optimizer steps (0 disables).
        checkpoint_every_minutes: Checkpoint cadence in wall-clock minutes (0 disables).
        max_shard_mib: Checkpoint shard size.
        session_soft_hours: Stop taking new steps after this long (see ``SessionClock``).
        session_hard_hours: Hard session limit of the runtime.
    """

    run_id: str
    model_preset: str
    tokenized: str
    objective: str
    seq_len: int
    micro_batch_size: int
    total_tokens: int
    learning_rate: float
    grad_accum_steps: int = 1
    mask_rate: float = 0.15
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "bf16"
    seed: int = 1234
    shuffle: bool = True
    cross_document_attention: bool = True
    lr_schedule: str = "cosine"
    decay_ratio: float = 0.2
    source_weights: dict[str, float] | None = None
    mask_rate_final: float | None = None
    mask_schedule_start: float = 0.0
    masking: str = "token"
    span_max_length: int = 5
    pmi_unit_share: float = 1.0
    variant_share: float = 0.0
    attention: str = "sdpa"
    sparse_prediction: bool = False
    device: str = "auto"
    deterministic: bool = False
    compile: bool = False
    log_every_steps: int = 10
    eval_every_steps: int = 500
    eval_sequences: int = 256
    eval_mask_rate: float | None = None
    checkpoint_every_steps: int = 0
    checkpoint_every_minutes: float = 45.0
    max_shard_mib: float = 1024.0
    session_soft_hours: float = 10.5
    session_hard_hours: float = 12.0

    def __post_init__(self) -> None:
        if self.objective not in {"mlm", "clm"}:
            raise ValueError("objective must be 'mlm' or 'clm'")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be 'bf16' or 'fp32'")
        if self.total_tokens < self.tokens_per_step:
            raise ValueError("total_tokens is smaller than one optimizer step")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if not 0 < self.decay_ratio <= 1:
            raise ValueError("decay_ratio must be in (0, 1]")
        if not 0 < self.pmi_unit_share <= 1:
            raise ValueError("pmi_unit_share must be in (0, 1]")
        if self.span_max_length < 1:
            raise ValueError("span_max_length must be >= 1")
        if self.masking not in MASKING_UNITS:
            raise ValueError(f"masking must be one of {MASKING_UNITS}")
        if self.attention not in ATTENTION_BACKENDS:
            raise ValueError(f"attention must be one of {ATTENTION_BACKENDS}")
        if not 0 <= self.mask_schedule_start < 1:
            raise ValueError("mask_schedule_start must be in [0, 1)")
        if self.mask_rate_final is not None and not 0 < self.mask_rate_final < 1:
            raise ValueError("mask_rate_final must be in (0, 1)")
        if not 0 <= self.variant_share <= 0.9:
            raise ValueError("variant_share must be in [0, 0.9]")
        if any(w < 0 for w in (self.source_weights or {}).values()):
            raise ValueError("source_weights must be non-negative")
        mlm_only = (
            self.masking != "token"
            or self.variant_share
            or self.mask_rate_final is not None
            or self.sparse_prediction
        )
        if mlm_only and self.objective != "mlm":
            raise ValueError("masking options apply to the MLM objective only")

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step."""
        return self.seq_len * self.micro_batch_size * self.grad_accum_steps

    @property
    def total_steps(self) -> int:
        """Optimizer steps needed to reach :attr:`total_tokens`."""
        return math.ceil(self.total_tokens / self.tokens_per_step)

    @property
    def warmup_steps(self) -> int:
        """Linear-warmup steps (at least one)."""
        return max(1, round(self.total_steps * self.warmup_ratio))

    def identity(self) -> dict[str, Any]:
        """Fields that define the trajectory; their hash gates strict resume."""
        data = asdict(self)
        identity = {key: data[key] for key in _IDENTITY_FIELDS}
        defaults = {f.name: f.default for f in fields(self)}
        for key in _OPTIONAL_IDENTITY_FIELDS:
            if data[key] != defaults[key]:
                identity[key] = data[key]
        return identity

    def identity_sha256(self) -> str:
        """Hash of :meth:`identity`."""
        return json_sha256(self.identity())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view with derived step counts."""
        data = asdict(self)
        data.update(
            tokens_per_step=self.tokens_per_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )
        return data

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PretrainConfig:
        """Build from a mapping; unknown keys raise ``TypeError``."""
        values = dict(data)
        values.pop("version", None)
        if "betas" in values:
            values["betas"] = tuple(float(b) for b in values["betas"])
        if values.get("source_weights") is not None:
            values["source_weights"] = {
                str(k): float(v) for k, v in values["source_weights"].items()
            }
        for key in ("total_tokens",):
            if key in values:
                values[key] = int(float(values[key]))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainConfig:
        """Load a YAML configuration file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))
