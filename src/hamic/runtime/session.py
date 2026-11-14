"""Session time budget for runtimes with a hard wall-clock limit (e.g. 12 hours)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["SessionClock"]

HOUR = 3600.0


@dataclass
class SessionClock:
    """Decides when to stop taking new work so a final checkpoint still fits.

    ``soft_stop = min(start + soft_limit, start + hard_limit - max(min_margin,
    2 * worst_checkpoint_seconds))``. The hard limit is never used as a stop mechanism.

    Attributes:
        hard_limit: Session hard limit in seconds.
        soft_limit: Default usable training window in seconds.
        min_margin: Minimum margin kept before the hard limit.
        clock: Monotonic time source (injectable for tests).
    """

    hard_limit: float = 12 * HOUR
    soft_limit: float = 10.5 * HOUR
    min_margin: float = 0.5 * HOUR
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    worst_checkpoint_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not 0 < self.soft_limit <= self.hard_limit:
            raise ValueError("Require 0 < soft_limit <= hard_limit")
        self.started_at = self.clock()

    def record_checkpoint(self, seconds: float) -> None:
        """Record how long a save + upload + verify took."""
        self.worst_checkpoint_seconds = max(self.worst_checkpoint_seconds, seconds)

    @property
    def elapsed(self) -> float:
        """Seconds since the session started."""
        return self.clock() - self.started_at

    @property
    def soft_stop_at(self) -> float:
        """Elapsed time at which no new work should start."""
        margin = max(self.min_margin, 2 * self.worst_checkpoint_seconds)
        return min(self.soft_limit, self.hard_limit - margin)

    def should_stop(self) -> bool:
        """Whether to stop scheduling work and write the final checkpoint."""
        return self.elapsed >= self.soft_stop_at

    @property
    def remaining(self) -> float:
        """Seconds left before :attr:`soft_stop_at` (never negative)."""
        return max(0.0, self.soft_stop_at - self.elapsed)
