"""Corpus pipeline: registry gate -> stages -> split shards -> immutable manifest."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from hamic.data.config import PipelineConfig
from hamic.data.dedup_index import DedupIndex
from hamic.data.documents import Document, DocumentSource
from hamic.data.parallel import ParallelPrefix, split_parallel_prefix
from hamic.data.registry import SourceRegistry, UseCase
from hamic.data.stages import NearDedupStage, SplitStage, Stage
from hamic.provenance import code_revision, environment_fingerprint, file_sha256, utc_now

__all__ = ["CorpusPipeline", "PipelineResult", "default_stages"]

SPLITS = ("train", "validation", "test")
MANIFEST_NAME = "manifest.json"


def default_stages() -> list[Stage]:
    """Return the canonical stage order with default settings."""
    return PipelineConfig().build_stages()


@dataclass(frozen=True)
class PipelineResult:
    """Summary of one pipeline run."""

    output_dir: Path
    manifest: dict[str, Any]

    @property
    def counts(self) -> dict[str, int]:
        """Documents written per split."""
        return {name: info["documents"] for name, info in self.manifest["splits"].items()}


class _ShardWriter:
    """Writes documents into size-capped gzip JSONL shards for one split."""

    def __init__(self, directory: Path, max_docs: int) -> None:
        self.directory = directory
        self.max_docs = max_docs
        self.files: list[Path] = []
        self.documents = 0
        self.characters = 0
        self._stream: TextIO | None = None
        self._in_shard = 0

    def write(self, document: Document) -> None:
        if self._stream is None or self._in_shard >= self.max_docs:
            self._rotate()
        assert self._stream is not None
        self._stream.write(document.to_json() + "\n")
        self._in_shard += 1
        self.documents += 1
        self.characters += len(document.text)

    def _rotate(self) -> None:
        self.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"part-{len(self.files):05d}.jsonl.gz"
        # mtime=0 keeps the gzip bytes, and therefore the checksum, reproducible. Level 3
        # compresses Vietnamese UTF-8 about 3.5x faster than level 9 for ~12% larger files;
        # at level 9 compression was the pipeline bottleneck.
        raw = gzip.GzipFile(filename=str(path), mode="wb", mtime=0, compresslevel=3)
        self._stream = io.TextIOWrapper(raw, encoding="utf-8")
        self.files.append(path)
        self._in_shard = 0

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class CorpusPipeline:
    """Run registered sources through cleaning stages and write a frozen corpus.

    Args:
        registry: Source Registry used to gate every source (fails closed).
        stages: Custom ordered stages (the last must assign ``meta["split"]``). Mutually
            exclusive with ``config``; the manifest then records no configuration.
        use: Intended use checked against the registry.
        shard_docs: Maximum documents per output shard (overrides ``config``).
        workers: Processes for the stateless stages and MinHash signatures; ``1`` runs
            everything in this process. The output is identical either way.
        config: Pipeline configuration; defaults to :class:`PipelineConfig` defaults.
        dedup_index: Persistent index of earlier corpus versions; the build then writes a
            delta (documents new relative to every indexed version) and adds the kept
            documents to the index. The caller saves the index after a successful run.
        base_corpus: Previous version this delta extends (recorded in the manifest).
    """

    def __init__(
        self,
        registry: SourceRegistry,
        stages: Sequence[Stage] | None = None,
        use: UseCase = UseCase.RESEARCH,
        shard_docs: int | None = None,
        workers: int = 1,
        config: PipelineConfig | None = None,
        dedup_index: DedupIndex | None = None,
        base_corpus: dict[str, Any] | None = None,
    ) -> None:
        if stages is not None and config is not None:
            raise ValueError("Pass either custom stages or a config, not both")
        self.config: PipelineConfig | None
        if stages is not None:
            self.config = None
            self.stages = list(stages)
        else:
            self.config = config or PipelineConfig()
            if shard_docs is not None:
                self.config = replace(self.config, shard_docs=shard_docs)
            self.stages = self.config.build_stages(dedup_index)
        if not self.stages or not isinstance(self.stages[-1], SplitStage):
            raise ValueError("The last stage must be a SplitStage")
        self.registry = registry
        self.use = use
        self.shard_docs = shard_docs or (self.config.shard_docs if self.config else 50_000)
        self.workers = max(1, workers)
        self.dedup_index = dedup_index
        self.base_corpus = base_corpus

    def _stream(self, documents: Iterable[Document]) -> Iterable[Document]:
        stream = documents
        prefix = split_parallel_prefix(self.stages) if self.workers > 1 else 0
        if prefix:
            near = next((s for s in self.stages if isinstance(s, NearDedupStage)), None)
            stream = ParallelPrefix(self.stages[:prefix], near, self.workers)(stream)
        for stage in self.stages[prefix:]:
            stream = stage(stream)
        return stream

    def run(
        self, sources: Iterable[DocumentSource], output_dir: str | Path, corpus_version: str
    ) -> PipelineResult:
        """Process ``sources`` into ``output_dir`` and write ``manifest.json`` last.

        Raises:
            FileExistsError: If ``output_dir`` already holds a manifest (corpora are
                immutable; write a new version instead).
            SourceNotApprovedError: If a source is not approved for :attr:`use`.
        """
        out = Path(output_dir)
        if (out / MANIFEST_NAME).exists():
            raise FileExistsError(f"{out} already contains a frozen corpus")
        source_list = list(sources)
        records = [self.registry.require(s.source_id, self.use) for s in source_list]

        writers = {name: _ShardWriter(out / name, self.shard_docs) for name in SPLITS}
        stream = self._stream(doc for source in source_list for doc in source)
        try:
            for document in stream:
                writers[document.meta["split"]].write(document)
        finally:
            for writer in writers.values():
                writer.close()

        self._write_audit_samples(out)
        manifest = {
            "schema_version": 1,
            "corpus_version": corpus_version,
            "created_at_utc": utc_now(),
            "use": self.use.value,
            "workers": self.workers,
            "config": self.config.to_dict() if self.config else None,
            "config_sha256": self.config.sha256() if self.config else None,
            "sources": [
                {
                    "source_id": r.source_id,
                    "location": r.location,
                    "revision": r.revision,
                    "license": r.license,
                    "status": r.status.value,
                }
                for r in records
            ],
            "stages": [{"name": s.name, **s.stats.to_dict()} for s in self.stages],
            "base_corpus": self.base_corpus,
            "dedup_index": (
                {"versions": list(self.dedup_index.versions), **self.dedup_index.settings()}
                if self.dedup_index is not None
                else None
            ),
            "splits": {
                name: {
                    "documents": w.documents,
                    "characters": w.characters,
                    "files": [
                        {"path": p.relative_to(out).as_posix(), "sha256": file_sha256(p)}
                        for p in w.files
                    ],
                }
                for name, w in writers.items()
            },
            "code": code_revision(),
            "environment": environment_fingerprint(),
        }
        with (out / MANIFEST_NAME).open("w", encoding="utf-8") as stream_out:
            json.dump(manifest, stream_out, ensure_ascii=False, indent=2, sort_keys=True)
        return PipelineResult(output_dir=out, manifest=manifest)

    def _write_audit_samples(self, out: Path) -> None:
        audit_dir = out / "audit"
        for stage in self.stages:
            for reason, documents in stage.rejected_samples.items():
                if not documents:
                    continue
                audit_dir.mkdir(parents=True, exist_ok=True)
                path = audit_dir / f"rejected-{stage.name}-{reason}.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for document in documents:
                        stream.write(document.to_json() + "\n")
