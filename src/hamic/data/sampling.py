"""Deterministic, bounded-memory samples from frozen corpus deltas."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from hamic.data.documents import Document, read_jsonl
from hamic.provenance import file_sha256, utc_now

if TYPE_CHECKING:
    from hamic.checkpoint.storage import ObjectStore

__all__ = ["SAMPLE_MANIFEST", "sample_corpora", "sample_published_tokenized"]

SAMPLE_MANIFEST = "sample_manifest.json"
_HASH_SPACE = 1 << 64


@dataclass
class _SampleWriter:
    """Write deterministic gzip JSONL shards without retaining documents in memory."""

    directory: Path
    shard_documents: int

    def __post_init__(self) -> None:
        self.files: list[dict[str, Any]] = []
        self._stream: TextIO | None = None
        self._raw: gzip.GzipFile | None = None
        self._path: Path | None = None
        self._documents = 0
        self._characters = 0

    def write(self, document: Document) -> None:
        """Append one selected document and rotate at the configured shard size."""
        if self._stream is None or self._documents >= self.shard_documents:
            self._rotate()
        assert self._stream is not None
        self._stream.write(document.to_json() + "\n")
        self._documents += 1
        self._characters += len(document.text)

    def _rotate(self) -> None:
        self.close_shard()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / f"part-{len(self.files):05d}.jsonl.gz"
        self._raw = gzip.GzipFile(
            filename=str(self._path), mode="wb", mtime=0, compresslevel=3
        )
        self._stream = io.TextIOWrapper(self._raw, encoding="utf-8")
        self._documents = 0
        self._characters = 0

    def close_shard(self) -> None:
        """Commit the open shard to the file list."""
        if self._stream is None:
            return
        self._stream.close()
        assert self._path is not None
        self.files.append(
            {
                "path": self._path.relative_to(self.directory.parent).as_posix(),
                "documents": self._documents,
                "characters": self._characters,
                "sha256": file_sha256(self._path),
            }
        )
        self._stream = None
        self._raw = None
        self._path = None


def _selected(seed: str, version: str, document: Document, probability: float) -> bool:
    """Select by a stable 64-bit hash; decisions do not depend on corpus traversal order."""
    if probability >= 1:
        return True
    identity = "\0".join((seed, version, document.source_id, document.doc_id)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return value < int(probability * _HASH_SPACE)


def sample_corpora(
    corpus_dirs: list[str | Path],
    output_dir: str | Path,
    target_documents: int,
    *,
    seed: str = "hamic-tokenizer-sample-v1",
    source_weights: dict[str, float] | None = None,
    shard_documents: int = 50_000,
) -> dict[str, Any]:
    """Sample train documents from immutable base/delta corpora in one streaming pass.

    ``target_documents`` determines the base selection probability from the document totals
    recorded in the input manifests. Source weights multiply that probability, so the final
    document count is approximate when non-unit weights are used. Hash sampling avoids a
    prefix bias, is reproducible across machines and needs constant memory.

    Args:
        corpus_dirs: Frozen corpus base and delta directories, each with ``manifest.json``.
        output_dir: New directory for sample shards and ``sample_manifest.json``.
        target_documents: Expected selected documents before source weighting.
        seed: Stable sampling identity.
        source_weights: Optional positive selection multiplier per source.
        shard_documents: Maximum selected documents per compressed output shard.

    Returns:
        The sample manifest written after all shards complete.
    """
    if target_documents < 1 or shard_documents < 1:
        raise ValueError("target_documents and shard_documents must be positive")
    weights = dict(source_weights or {})
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("source weights must be positive")

    roots = [Path(path) for path in corpus_dirs]
    if not roots:
        raise ValueError("at least one corpus directory is required")
    inputs: list[tuple[Path, Path, dict[str, Any]]] = []
    versions: set[str] = set()
    total_documents = 0
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = str(manifest["corpus_version"])
        if version in versions:
            raise ValueError(f"duplicate corpus version: {version}")
        versions.add(version)
        total_documents += int(manifest["splits"]["train"]["documents"])
        inputs.append((root, manifest_path, manifest))
    if total_documents < 1:
        raise ValueError("input corpora contain no train documents")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    writer = _SampleWriter(output / "train", shard_documents)
    base_probability = min(1.0, target_documents / total_documents)
    source_counts: dict[str, int] = {}
    source_characters: dict[str, int] = {}
    selected_documents = 0
    selected_characters = 0
    input_records: list[dict[str, Any]] = []

    for root, manifest_path, manifest in inputs:
        version = str(manifest["corpus_version"])
        split = manifest["splits"]["train"]
        input_records.append(
            {
                "corpus_version": version,
                "manifest_sha256": file_sha256(manifest_path),
                "documents": int(split["documents"]),
                "characters": int(split["characters"]),
            }
        )
        for entry in split["files"]:
            for document in read_jsonl(root / entry["path"]):
                probability = min(1.0, base_probability * weights.get(document.source_id, 1.0))
                if not _selected(seed, version, document, probability):
                    continue
                writer.write(document)
                selected_documents += 1
                selected_characters += len(document.text)
                source_counts[document.source_id] = source_counts.get(document.source_id, 0) + 1
                source_characters[document.source_id] = (
                    source_characters.get(document.source_id, 0) + len(document.text)
                )
    writer.close_shard()

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "format": "hamic-corpus-sample-v1",
        "split": "train",
        "selection": {
            "method": "sha256-threshold-v1",
            "seed": seed,
            "target_documents": target_documents,
            "base_probability": base_probability,
            "source_weights": weights,
        },
        "inputs": input_records,
        "documents": selected_documents,
        "characters": selected_characters,
        "sources": {
            source: {
                "documents": source_counts[source],
                "characters": source_characters[source],
            }
            for source in sorted(source_counts)
        },
        "files": writer.files,
    }
    target = output / SAMPLE_MANIFEST
    staging = output / f"{SAMPLE_MANIFEST}.tmp"
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    staging.replace(target)
    return manifest


def _quantile_indices(count: int, selected: int) -> list[int]:
    """Evenly spaced shard indices, centered inside equal-width corpus intervals."""
    if selected >= count:
        return list(range(count))
    return [min(count - 1, int((i + 0.5) * count / selected)) for i in range(selected)]


def _weighted_copies(weight: float, seed: str, identity: str) -> int:
    """Convert a positive fractional source weight to deterministic document copies."""
    whole = math.floor(weight)
    fraction = weight - whole
    if fraction == 0:
        return whole
    probe = Document(identity, "", "")
    return whole + int(_selected(f"{seed}-source-weight", "", probe, fraction))


def _remote_path(
    relative: str, prefix: str, remote_files: dict[str, str]
) -> str:
    """Resolve an assembled shard to its published base or current-version location."""
    return remote_files.get(relative, f"{prefix.rstrip('/')}/{relative}")


def sample_published_tokenized(
    tokenized_dir: str | Path,
    store: ObjectStore,
    prefix: str,
    output_dir: str | Path,
    target_tokens: int,
    *,
    seed: str = "hamic-tokenizer-shards-v1",
    source_weights: dict[str, float] | None = None,
    shard_documents: int = 50_000,
    batch_size: int = 256,
    shard_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Decode a distributed sample of published token shards back to cleaned text.

    The old byte-level BPE is only a lossless transport here: every decoded document is
    immediately encoded again and must reproduce the exact original id sequence. Selected
    shards are downloaded one at a time and removed after decoding, bounding local storage.

    Args:
        tokenized_dir: Local lightweight assembled directory containing the manifest,
            tokenizer and optional ``remote_files.json`` map.
        store: Object store holding published tokenized shards.
        prefix: Published folder of the current complete version.
        output_dir: New directory for decoded gzip JSONL shards.
        target_tokens: Approximate number of original token ids to cover before weighting.
        seed: Identity used for deterministic fractional source weights.
        source_weights: Repetition/thinning weight per recorded source id.
        shard_documents: Documents per decoded output shard.
        batch_size: Documents decoded and round-trip checked together.
        shard_indices: Exact train-shard indices to decode. ``None`` selects evenly spaced
            quantiles from ``target_tokens``. This supports explicit source-coverage shards.

    Returns:
        The sample manifest written after every selected shard passes round-trip checks.
    """
    from tokenizers import Tokenizer

    if target_tokens < 1 or shard_documents < 1 or batch_size < 1:
        raise ValueError("target_tokens, shard_documents and batch_size must be positive")
    weights = dict(source_weights or {})
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("source weights must be positive")
    root = Path(tokenized_dir)
    manifest_path = root / "tokenized_manifest.json"
    tokenizer_path = root / "tokenizer.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(tokenizer_path) != manifest["tokenizer"]["sha256"]:
        raise ValueError("local tokenizer does not match the tokenized manifest")
    remote_path = root / "remote_files.json"
    remote_files: dict[str, str] = (
        json.loads(remote_path.read_text(encoding="utf-8")) if remote_path.is_file() else {}
    )
    shards = manifest["splits"]["train"]["shards"]
    total_tokens = sum(int(shard["tokens"]) for shard in shards)
    if not shards or total_tokens < 1:
        raise ValueError("tokenized corpus contains no train shards")
    if shard_indices is None:
        average = total_tokens / len(shards)
        shard_count = min(len(shards), max(1, math.ceil(target_tokens / average)))
        indices = _quantile_indices(len(shards), shard_count)
        selection_method = "even-shard-quantiles-v1"
    else:
        indices = sorted(set(shard_indices))
        if not indices or indices[0] < 0 or indices[-1] >= len(shards):
            raise ValueError("shard_indices must identify at least one existing train shard")
        selection_method = "explicit-shard-indices-v1"

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    writer = _SampleWriter(output / "train", shard_documents)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_ids = list(manifest.get("source_ids") or [])
    source_counts: dict[str, int] = {}
    source_characters: dict[str, int] = {}
    decoded_documents = 0
    decoded_characters = 0
    covered_tokens = 0
    selected_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="hamic-token-sample-", dir=output) as temporary:
        staging = Path(temporary)
        for shard_index in indices:
            shard = shards[shard_index]
            files: dict[str, Any] = shard["files"]
            local: dict[str, Path] = {}
            published: dict[str, str] = {}
            for kind in ("tokens", "offsets", "sources"):
                if kind not in files:
                    if kind == "sources":
                        continue
                    raise ValueError(f"shard {shard['name']} has no {kind} file")
                relative = f"train/{files[kind]['path']}"
                remote = _remote_path(relative, prefix, remote_files)
                destination = staging / files[kind]["path"]
                store.download(remote, destination)
                if file_sha256(destination) != files[kind]["sha256"]:
                    raise ValueError(f"checksum mismatch after downloading {remote}")
                local[kind] = destination
                published[kind] = remote

            tokens = np.load(local["tokens"], mmap_mode="r")
            offsets = np.load(local["offsets"], mmap_mode="r")
            sources = np.load(local["sources"], mmap_mode="r") if "sources" in local else None
            documents = int(offsets.size - 1)
            if documents != int(shard["documents"]):
                raise ValueError(f"document count mismatch in shard {shard['name']}")
            for start in range(0, documents, batch_size):
                stop = min(documents, start + batch_size)
                id_batches = [
                    tokens[offsets[index] : offsets[index + 1]].astype(np.int64).tolist()
                    for index in range(start, stop)
                ]
                texts = tokenizer.decode_batch(id_batches, skip_special_tokens=False)
                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                for offset, (ids, text, restored) in enumerate(
                    zip(id_batches, texts, encoded, strict=True), start=start
                ):
                    if restored.ids != ids:
                        raise ValueError(
                            f"tokenizer round-trip failed at {shard['name']} document {offset}"
                        )
                    source = (
                        source_ids[int(sources[offset])]
                        if sources is not None and source_ids
                        else "unknown"
                    )
                    identity = f"{shard['name']}:{offset}"
                    copies = _weighted_copies(weights.get(source, 1.0), seed, identity)
                    for copy in range(copies):
                        writer.write(
                            Document(
                                doc_id=f"sample-{identity}:{copy}",
                                text=text,
                                source_id=source,
                                meta={
                                    "sample_shard": shard["name"],
                                    "sample_document": offset,
                                },
                            )
                        )
                        decoded_documents += 1
                        decoded_characters += len(text)
                        source_counts[source] = source_counts.get(source, 0) + 1
                        source_characters[source] = source_characters.get(source, 0) + len(text)
            covered_tokens += int(shard["tokens"])
            selected_records.append(
                {
                    "index": shard_index,
                    "name": shard["name"],
                    "tokens": int(shard["tokens"]),
                    "documents": int(shard["documents"]),
                    "remote_files": published,
                }
            )
            del tokens, offsets, sources
            for path in local.values():
                path.unlink()

    writer.close_shard()
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "format": "hamic-decoded-token-sample-v1",
        "split": "train",
        "selection": {
            "method": selection_method,
            "seed": seed,
            "target_tokens": target_tokens,
            "covered_tokens": covered_tokens,
            "source_weights": weights,
            "roundtrip_checked": True,
        },
        "input": {
            "prefix": prefix,
            "manifest_sha256": file_sha256(manifest_path),
            "tokenizer": manifest["tokenizer"],
        },
        "selected_shards": selected_records,
        "documents": decoded_documents,
        "characters": decoded_characters,
        "sources": {
            source: {
                "documents": source_counts[source],
                "characters": source_characters[source],
            }
            for source in sorted(source_counts)
        },
        "files": writer.files,
    }
    target = output / SAMPLE_MANIFEST
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
