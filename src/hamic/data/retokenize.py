"""Re-tokenize a published tokenized corpus with a new tokenizer, shard by shard.

The text of v0.2–v0.12 exists only as token ids of the old tokenizer (T1) on the Hub. T1 is
a byte-level BPE whose normalizer only applies NFC, so its ids decode to the cleaned corpus
text exactly; decoded documents (all, or every n-th) are encoded with the old tokenizer
again and must give back the same ids.

Shards are converted one to one (same names, documents and order; the ``sources`` file is
identical and copied on the server). Each worker downloads one old shard, converts it and
deletes the download; the main process uploads finished shards in batches and deletes them
locally, so local storage stays bounded however large the corpus is. A progress file makes
an interrupted run resume where it stopped.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hamic.data.tokenized import (
    MANIFEST_NAME,
    REMOTE_FILES_NAME,
    SPLITS,
    TOKENIZER_NAME,
    _dtype_for,
    _text_preparation,
    _write_manifest,
)
from hamic.provenance import code_revision, file_sha256, utc_now

if TYPE_CHECKING:
    from hamic.checkpoint.storage import ObjectStore

__all__ = ["retokenize_published"]

PROGRESS_NAME = "retokenize_progress.jsonl"
_STATE: dict[str, Any] = {}


def _published(relative: str, prefix: str, remote_files: dict[str, str]) -> str:
    return remote_files.get(relative, f"{prefix.rstrip('/')}/{relative}")


def _init_worker(store: ObjectStore, old_tokenizer: str, new_tokenizer: str, threads: int) -> None:
    # Before the first tokenizers call: each worker's Rust pool gets its share of cores.
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    from tokenizers import Tokenizer

    _STATE.update(
        store=store,
        old=Tokenizer.from_file(old_tokenizer),
        new=Tokenizer.from_file(new_tokenizer),
    )


def _documents(
    tokens: np.ndarray[Any, Any], offsets: np.ndarray[Any, Any], batch_size: int
) -> Iterator[list[list[int]]]:
    documents = int(offsets.size - 1)
    for start in range(0, documents, batch_size):
        stop = min(documents, start + batch_size)
        yield [
            tokens[offsets[i] : offsets[i + 1]].astype(np.int64).tolist()
            for i in range(start, stop)
        ]


def _convert(job: dict[str, Any]) -> dict[str, Any]:
    """Download one old shard, verify, decode, re-encode and write the new shard."""
    from hamic.tokenization import prepare_text

    store, old, new = _STATE["store"], _STATE["old"], _STATE["new"]
    shard, split, out = job["shard"], job["split"], Path(job["out"])
    staging = out / "_staging" / f"{split}-{shard['name']}"
    staging.mkdir(parents=True, exist_ok=True)
    local: dict[str, Path] = {}
    for kind in ("tokens", "offsets"):
        entry = shard["files"][kind]
        path = staging / entry["path"]
        store.download(job["remote"][kind], path)
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch after downloading {job['remote'][kind]}")
        local[kind] = path
    tokens = np.load(local["tokens"], mmap_mode="r")
    offsets = np.load(local["offsets"], mmap_mode="r")
    if int(offsets.size - 1) != int(shard["documents"]):
        raise ValueError(f"document count mismatch in {split}/{shard['name']}")

    dtype = np.dtype(job["dtype"])
    chunks: list[np.ndarray[Any, Any]] = []
    lengths: list[int] = []
    characters = 0
    for index, ids in enumerate(_documents(tokens, offsets, job["batch_size"])):
        texts = old.decode_batch(ids, skip_special_tokens=False)
        every = job["check_every"]
        checked = list(range(0, len(texts), every))
        again = old.encode_batch([texts[i] for i in checked], add_special_tokens=False)
        for offset, restored in zip(checked, again, strict=True):
            if restored.ids != ids[offset]:
                document = index * job["batch_size"] + offset
                raise ValueError(
                    f"old tokenizer round-trip failed: {split}/{shard['name']} document {document}"
                )
        prepared = [prepare_text(text) for text in texts]
        characters += sum(len(text) for text in prepared)
        for encoding in new.encode_batch(prepared, add_special_tokens=False):
            chunks.append(np.asarray(encoding.ids, dtype=dtype))
            lengths.append(len(encoding.ids))
    del tokens, offsets
    shutil.rmtree(staging, ignore_errors=True)

    directory = out / split
    directory.mkdir(parents=True, exist_ok=True)
    name = shard["name"]
    new_tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype)
    new_offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=new_offsets[1:])
    token_path = directory / f"{name}.tokens.npy"
    offset_path = directory / f"{name}.offsets.npy"
    np.save(token_path, new_tokens, allow_pickle=False)
    np.save(offset_path, new_offsets, allow_pickle=False)
    files: dict[str, Any] = {
        "tokens": {"path": token_path.name, "sha256": file_sha256(token_path)},
        "offsets": {"path": offset_path.name, "sha256": file_sha256(offset_path)},
    }
    if "sources" in shard["files"]:
        files["sources"] = dict(shard["files"]["sources"])
    return {
        "split": split,
        "index": job["index"],
        "shard": {
            "name": name,
            "documents": len(lengths),
            "tokens": int(new_tokens.size),
            "files": files,
        },
        "old_tokens": int(shard["tokens"]),
        "characters": characters,
        "sources_remote": job["remote"].get("sources"),
    }


def retokenize_published(
    tokenized_dir: str | Path,
    store: ObjectStore,
    prefix: str,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    push_prefix: str,
    *,
    workers: int = 4,
    batch_size: int = 512,
    push_every: int = 16,
    max_shards: int | None = None,
    check_every: int = 50,
) -> dict[str, Any]:
    """Convert every shard of a published tokenized version to a new tokenizer.

    Args:
        tokenized_dir: Local lightweight directory of the published version (manifest,
            ``tokenizer.json`` and optional ``remote_files.json``).
        store: Store holding the published shards; finished shards are written to it.
        prefix: Published folder of the version, e.g. ``tokenized/v0.12/t1-bbpe-20544``.
        tokenizer_dir: Directory with the new ``tokenizer.json`` (and
            ``hamic_tokenizer.json``).
        output_dir: Local working directory; ends holding only the manifest, the tokenizer
            copy and the progress file.
        push_prefix: Folder the new version is published to.
        workers: Shards converted in parallel (each also downloads its own input).
        batch_size: Documents decoded and encoded together.
        push_every: Finished shards uploaded per commit.
        max_shards: Stop after this many shards (trial runs); no manifest is written.
        check_every: Re-encode every n-th decoded document with the old tokenizer and
            require its original ids (1 checks all; the re-encode costs about 30% of the
            run, and T1 decoding has never failed).

    Returns:
        The manifest written (and published) at the end, or a progress summary when
        ``max_shards`` stopped the run early.
    """
    root = Path(tokenized_dir)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_tokenizer = root / TOKENIZER_NAME
    if file_sha256(old_tokenizer) != manifest["tokenizer"]["sha256"]:
        raise ValueError("local tokenizer does not match the tokenized manifest")
    remote_path = root / REMOTE_FILES_NAME
    remote_files: dict[str, str] = (
        json.loads(remote_path.read_text(encoding="utf-8")) if remote_path.is_file() else {}
    )
    new_tokenizer = Path(tokenizer_dir) / TOKENIZER_NAME
    metadata = json.loads((Path(tokenizer_dir) / "hamic_tokenizer.json").read_text("utf-8"))
    from tokenizers import Tokenizer

    vocab_size = int(Tokenizer.from_file(str(new_tokenizer)).get_vocab_size())
    dtype = _dtype_for(vocab_size)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError(f"{out} already holds a finished manifest")
    shutil.rmtree(out / "_staging", ignore_errors=True)
    progress_path = out / PROGRESS_NAME
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[(record["split"], record["index"])] = record

    jobs: list[dict[str, Any]] = []
    for split in SPLITS:
        for index, shard in enumerate(manifest["splits"][split]["shards"]):
            if (split, index) in done:
                continue
            remote = {
                kind: _published(f"{split}/{entry['path']}", prefix, remote_files)
                for kind, entry in shard["files"].items()
            }
            jobs.append(
                {
                    "split": split,
                    "index": index,
                    "shard": shard,
                    "remote": remote,
                    "out": str(out),
                    "dtype": dtype.name,
                    "batch_size": batch_size,
                    "check_every": check_every,
                }
            )
    if max_shards is not None:
        jobs = jobs[:max_shards]

    pending: list[dict[str, Any]] = []

    def push() -> None:
        uploads: dict[str, Path] = {}
        copies: dict[str, str] = {}
        for record in pending:
            split, files = record["split"], record["shard"]["files"]
            for kind in ("tokens", "offsets"):
                uploads[f"{push_prefix}/{split}/{files[kind]['path']}"] = (
                    out / split / files[kind]["path"]
                )
            if "sources" in files:
                copies[f"{push_prefix}/{split}/{files['sources']['path']}"] = record[
                    "sources_remote"
                ]
        store.put_files(uploads, message=f"retokenized shards {push_prefix}", copies=copies)
        with progress_path.open("a", encoding="utf-8") as stream:
            for record in pending:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[(record["split"], record["index"])] = record
        for path in uploads.values():
            path.unlink()
        pending.clear()

    threads = max(1, (os.cpu_count() or 2) // max(1, workers))
    context = get_context("spawn")
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(store, str(old_tokenizer), str(new_tokenizer), threads),
    ) as pool:
        for count, record in enumerate(pool.imap_unordered(_convert, jobs), start=1):
            pending.append(record)
            print(
                f"retokenized {record['split']}/{record['shard']['name']} "
                f"({count}/{len(jobs)}, {len(done) + len(pending)} total): "
                f"{record['old_tokens']} -> {record['shard']['tokens']} tokens",
                flush=True,
            )
            if len(pending) >= push_every:
                push()
    if pending:
        push()
    shutil.rmtree(out / "_staging", ignore_errors=True)

    total = sum(len(manifest["splits"][s]["shards"]) for s in SPLITS)
    if len(done) < total:
        return {"finished": False, "shards_done": len(done), "shards_total": total}

    splits: dict[str, Any] = {}
    for split in SPLITS:
        records = [done[(split, i)] for i in range(len(manifest["splits"][split]["shards"]))]
        shards = [r["shard"] for r in records]
        splits[split] = {
            "documents": sum(s["documents"] for s in shards),
            "tokens": sum(s["tokens"] for s in shards),
            "characters": sum(r["characters"] for r in records),
            "shards": shards,
        }
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "format": "hamic-tokenized-npy-v1",
        "special_tokens_added": False,
        "dtype": dtype.name,
        "tokenizer": {
            "name": metadata["spec"]["name"],
            "vocab_size": vocab_size,
            "sha256": file_sha256(new_tokenizer),
            "text_preparation": _text_preparation(),
        },
        "source_corpus": {
            **manifest["source_corpus"],
            "retokenized_from": {
                "prefix": prefix,
                "manifest_sha256": file_sha256(manifest_path),
                "tokenizer": manifest["tokenizer"],
                "roundtrip_checked_every": check_every,
            },
        },
        "source_ids": manifest.get("source_ids"),
        "splits": splits,
        "code": code_revision(),
    }
    shutil.copyfile(new_tokenizer, out / TOKENIZER_NAME)
    _write_manifest(out, result)
    store.put_files(
        {f"{push_prefix}/{TOKENIZER_NAME}": out / TOKENIZER_NAME},
        message=f"retokenized tokenizer {push_prefix}",
    )
    store.put_files(
        {f"{push_prefix}/{MANIFEST_NAME}": out / MANIFEST_NAME},
        message=f"retokenized manifest {push_prefix}",
    )
    return result
