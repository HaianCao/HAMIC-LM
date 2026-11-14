"""Run the stateless prefix of a corpus pipeline in worker processes.

Stateless stages (``parallel_safe = True``: normalize, PII, quality) and the MinHash
signature run in a process pool. Results come back **in input order**, and every stage's
statistics and audit samples are updated in the main process exactly as a serial run
would update them, so a parallel corpus is identical to a serial one.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from hamic.data.documents import Document
from hamic.data.stages import NearDedupStage, Stage, minhash_signature

__all__ = ["ParallelPrefix", "split_parallel_prefix"]


@dataclass(frozen=True)
class _Outcome:
    """Result of the stateless prefix for one document."""

    document: Document
    rejected_at: int | None = None
    reason: str = ""
    signature: Any = None


_WORKER_STAGES: Sequence[Stage] = ()
_WORKER_NEAR: tuple[int, int] | None = None


def _init_worker(stages: Sequence[Stage], near: tuple[int, int] | None) -> None:
    global _WORKER_STAGES, _WORKER_NEAR
    _WORKER_STAGES = stages
    _WORKER_NEAR = near


def _process_batch(batch: list[Document]) -> list[_Outcome]:
    outcomes: list[_Outcome] = []
    for document in batch:
        current = document
        for index, stage in enumerate(_WORKER_STAGES):
            result = stage.process(current)
            if isinstance(result, str):
                outcomes.append(_Outcome(current, rejected_at=index, reason=result))
                break
            current = result
        else:
            signature = minhash_signature(current.text, *_WORKER_NEAR) if _WORKER_NEAR else None
            outcomes.append(_Outcome(current, signature=signature))
    return outcomes


def _batches(documents: Iterable[Document], size: int) -> Iterator[list[Document]]:
    iterator = iter(documents)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def split_parallel_prefix(stages: Sequence[Stage]) -> int:
    """Return how many leading stages are ``parallel_safe``."""
    count = 0
    for stage in stages:
        if not stage.parallel_safe:
            break
        count += 1
    return count


class ParallelPrefix:
    """Apply leading stateless stages with a process pool.

    Args:
        stages: The stateless stages, in pipeline order.
        near: The pipeline's near-dedup stage, whose signatures are precomputed in the
            workers, or ``None``.
        workers: Worker processes.
        batch_size: Documents per task.
        window: Maximum tasks in flight per worker (bounds memory on long streams).
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        near: NearDedupStage | None,
        workers: int,
        batch_size: int = 256,
        window: int = 4,
    ) -> None:
        if workers < 2:
            raise ValueError("ParallelPrefix needs at least two workers")
        self.stages = list(stages)
        self.near = near
        self.workers = workers
        self.batch_size = batch_size
        self.window = window

    def __call__(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield documents that pass every stateless stage, in input order."""
        near_params = (self.near.num_perm, self.near.shingle_size) if self.near else None
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(self.stages, near_params),
        ) as pool:
            pending: deque[Future[list[_Outcome]]] = deque()
            for batch in _batches(documents, self.batch_size):
                pending.append(pool.submit(_process_batch, batch))
                if len(pending) >= self.workers * self.window:
                    yield from self._merge(pending.popleft().result())
            while pending:
                yield from self._merge(pending.popleft().result())

    def _merge(self, outcomes: list[_Outcome]) -> Iterator[Document]:
        for outcome in outcomes:
            passed = len(self.stages) if outcome.rejected_at is None else outcome.rejected_at
            for stage in self.stages[: passed + (outcome.rejected_at is not None)]:
                stage.stats.seen += 1
            for stage in self.stages[:passed]:
                stage.record_kept(outcome.document)
            if outcome.rejected_at is not None:
                self.stages[outcome.rejected_at].record_rejected(outcome.document, outcome.reason)
                continue
            if self.near is not None and outcome.signature is not None:
                self.near.precomputed[outcome.document.doc_id] = (
                    len(outcome.document.text),
                    outcome.signature,
                )
            yield outcome.document
