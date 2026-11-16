"""Grow the corpus in versions: deltas on disk, complete versions on the Hub.

A version ``v0.(n+1)`` is the previous version plus a *delta* of new documents. Locally
only the delta is stored (``data/corpus/v0.3-delta``), deduplicated against every earlier
version through a persistent :class:`~hamic.data.dedup_index.DedupIndex`. For training and
publishing, :func:`assemble_tokenized` builds a complete tokenized version from the
previous complete version and the delta with **hard links**, so it costs no extra disk
space; the Hub then holds one self-contained folder per version.

FineWeb-2 is read file by file in a seeded random order (:func:`shuffled_files`): its
parquet files follow crawl time, so taking files in order would skew a version towards a
few years of the web.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hamic.data.dedup_index import DedupIndex
from hamic.data.documents import read_jsonl
from hamic.data.stages import minhash_signature
from hamic.data.tokenized import (
    MANIFEST_NAME,
    REMOTE_FILES_NAME,
    SPLITS,
    TOKENIZER_NAME,
    TokenizedCorpus,
    _write_manifest,
)
from hamic.provenance import code_revision, file_sha256, json_sha256, utc_now

__all__ = ["assemble_tokenized", "index_corpus", "shuffled_files"]

CORPUS_MANIFEST = "manifest.json"


def shuffled_files(files: Sequence[str], seed: str) -> list[str]:
    """Seeded random order of ``files`` (stable across Python versions and platforms)."""
    order = sorted(files)
    random.Random(seed).shuffle(order)
    return order


def _signature_batch(args: tuple[list[str], int, int]) -> list[Any]:
    texts, num_perm, shingle_size = args
    return [minhash_signature(text, num_perm, shingle_size) for text in texts]


def _documents(corpus: Path) -> Iterator[tuple[str, str]]:
    manifest = json.loads((corpus / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    for split in SPLITS:
        for entry in manifest["splits"][split]["files"]:
            for document in read_jsonl(corpus / entry["path"]):
                yield document.text_sha256, document.text


def _batches(items: Iterable[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_corpus(
    corpus_dir: str | Path,
    index: DedupIndex,
    version: str,
    workers: int = 1,
    batch_size: int = 512,
) -> int:
    """Add every document of a frozen corpus to ``index``.

    Signatures are recomputed from the stored (already normalized) texts, which are the
    texts the pipeline hashed, so the keys equal those of a build that kept them.

    Returns:
        Documents indexed.
    """
    corpus = Path(corpus_dir)
    count = 0
    settings = (index.num_perm, index.shingle_size)
    batches = _batches(_documents(corpus), batch_size)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = []
            for batch in batches:
                texts = [t for _, t in batch]
                pending.append((batch, pool.submit(_signature_batch, (texts, *settings))))
                if len(pending) >= workers * 4:
                    count += _absorb(index, *pending.pop(0))
            for item in pending:
                count += _absorb(index, *item)
    else:
        for batch in batches:
            count += _absorb(index, batch, _signature_batch(([t for _, t in batch], *settings)))
    if version not in index.versions:
        index.versions.append(version)
    return count


def _absorb(index: DedupIndex, batch: list[tuple[str, str]], signatures: Any) -> int:
    values = signatures.result() if hasattr(signatures, "result") else signatures
    for (digest, _), signature in zip(batch, values, strict=True):
        index.add(digest, signature)
    return len(batch)


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def assemble_tokenized(
    base_dir: str | Path | None,
    delta_dir: str | Path,
    output_dir: str | Path,
    version: str,
    *,
    base_manifest: dict[str, Any] | None = None,
    base_prefix: str | None = None,
) -> dict[str, Any]:
    """Build a complete tokenized version from the previous version and a delta.

    Shards of the base keep their names; delta shards follow with continuing numbers.
    Delta files are hard-linked (no extra disk space); delta source files are rewritten
    when the delta numbers its sources differently from the base.

    The base is either a local directory (``base_dir``; its files are hard-linked too) or a
    version already published to a store (``base_manifest`` read from ``base_prefix``).
    In the second case only the delta is written locally, and ``remote_files.json`` maps
    every base file to its published path, so :func:`~hamic.data.publish_tokenized` copies
    those on the server and the previous versions need not stay on disk.

    Raises:
        ValueError: Base and delta differ in tokenizer, or sources are missing.
        FileExistsError: ``output_dir`` already holds a manifest.
    """
    if (base_dir is None) == (base_manifest is None):
        raise ValueError("Pass either base_dir or base_manifest (with base_prefix)")
    if base_manifest is not None and not base_prefix:
        raise ValueError("base_manifest needs base_prefix")
    local_base = TokenizedCorpus(base_dir, verify=True) if base_dir is not None else None
    base_info = local_base.manifest if local_base is not None else dict(base_manifest or {})
    delta = TokenizedCorpus(delta_dir, verify=True)
    out = Path(output_dir)
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError(f"{out} already contains tokenized shards")
    if base_info["tokenizer"] != delta.manifest["tokenizer"]:
        raise ValueError("Base and delta were tokenized with different tokenizers")
    for ids, name in ((base_info.get("source_ids"), "base"), (delta.source_ids, "delta")):
        if ids is None:
            raise ValueError(f"The {name} shards have no sources; run annotate-sources")
    source_ids = list(base_info["source_ids"])
    for source_id in delta.source_ids or []:
        if source_id not in source_ids:
            source_ids.append(source_id)
    remap = np.array([source_ids.index(s) for s in delta.source_ids or []], dtype=np.uint8)

    remote: dict[str, str] = {}
    splits: dict[str, Any] = {}
    for split in SPLITS:
        shards: list[dict[str, Any]] = []
        for shard in base_info["splits"][split]["shards"]:
            name = f"{split}-{len(shards):05d}"
            files: dict[str, Any] = {}
            for kind, entry in shard["files"].items():
                target_name = f"{name}.{kind}.npy"
                if local_base is not None:
                    target = out / split / target_name
                    _link(local_base.directory / split / entry["path"], target)
                    files[kind] = {"path": target_name, "sha256": file_sha256(target)}
                else:
                    remote[f"{split}/{target_name}"] = f"{base_prefix}/{split}/{entry['path']}"
                    files[kind] = {"path": target_name, "sha256": entry["sha256"]}
            shards.append({**shard, "name": name, "files": files})
        for shard in delta.manifest["splits"][split]["shards"]:
            name = f"{split}-{len(shards):05d}"
            files = {}
            for kind, entry in shard["files"].items():
                source = delta.directory / split / entry["path"]
                target = out / split / f"{name}.{kind}.npy"
                if kind == "sources":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, remap[np.load(source)], allow_pickle=False)
                else:
                    _link(source, target)
                files[kind] = {"path": target.name, "sha256": file_sha256(target)}
            shards.append({**shard, "name": name, "files": files})
        splits[split] = {
            "documents": sum(s["documents"] for s in shards),
            "tokens": sum(s["tokens"] for s in shards),
            "shards": shards,
        }

    parts = [base_info["source_corpus"], delta.manifest["source_corpus"]]
    manifest = {
        **{k: base_info[k] for k in ("schema_version", "format", "dtype", "tokenizer")},
        "special_tokens_added": base_info.get("special_tokens_added", False),
        "created_at_utc": utc_now(),
        "source_corpus": {
            "corpus_version": version,
            "assembled_from": parts,
            "manifest_sha256": json_sha256(parts),
        },
        "source_ids": source_ids,
        "splits": splits,
        "code": code_revision(),
    }
    _link(delta.directory / TOKENIZER_NAME, out / TOKENIZER_NAME)
    if remote:
        (out / REMOTE_FILES_NAME).write_text(json.dumps(remote, indent=2), encoding="utf-8")
    _write_manifest(out, manifest)
    return manifest
