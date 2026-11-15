"""Pretraining loop, data loading, objectives, state and recovery drills."""

from hamic.training.compat import ResumeCompatibilityError, check_strict_resume
from hamic.training.config import PretrainConfig
from hamic.training.data import PackedBatches
from hamic.training.objectives import CausalLM, MaskedLM, Objective
from hamic.training.smoke import SmokeConfig, SmokeResult, SyntheticTokenStream, run_smoke
from hamic.training.state import (
    DataCursor,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from hamic.training.trainer import Pretrainer, TrainResult

__all__ = [
    "CausalLM",
    "DataCursor",
    "MaskedLM",
    "Objective",
    "PackedBatches",
    "PretrainConfig",
    "Pretrainer",
    "ResumeCompatibilityError",
    "SmokeConfig",
    "SmokeResult",
    "SyntheticTokenStream",
    "TrainResult",
    "capture_rng",
    "check_strict_resume",
    "load_training_state",
    "restore_rng",
    "run_smoke",
    "save_training_state",
]
