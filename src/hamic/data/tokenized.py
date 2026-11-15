"""Tokenized, memory-mappable shards used as training input.

Each shard is a pair of ``.npy`` files:

* ``<name>.tokens.npy``: all token ids of the shard's documents, concatenated, as
  ``uint16`` when the vocabulary fits (HAMIC's ~20K/32K vocabularies) else ``uint32``;
* ``<name>.offsets.npy``: ``int64`` document boundaries, ``documents + 1`` entries, so
  document ``i`` is ``tokens[offsets[i]:offsets[i + 1]]``.

Ids carry **no special tokens**: adding ``<s>``/``</s>``, packing and masking belong to the
training data loader. ``tokenized_manifest.json`` (written last) records the tokenizer and
source-corpus hashes and a checksum for every file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hamic.data.documents import read_jsonl
from hamic.provenance import code_revision, file_sha256, utc_now

__all__ = ["TokenizedCorpus", "TokenizedShard", "tokenize_corpus"]

MANIFEST_NAME = "tokenized_manifest.json"
SOURCE_MANIFEST = "manifest.json"
SPLITS = ("train", "validation", "test")


def _dtype_for(vocab_size: int) -> np.dtype[Any]:
    return (
        np.dtype(np.uint16) if vocab_size <= np.iinfo(np.uint16).max + 1 else np.dtype(np.uint32)
    )


class _ShardSink:
    """Accumulates token ids and writes a shard whenever ``shard_tokens`` is reached."""

    def __init__(
        self, directory: Path, split: str, dtype: np.dtype[Any], shard_tokens: int
    ) -> None:
        self.directory = directory
        self.split = split
        self.dtype = dtype
        self.shard_tokens = shard_tokens
        self.shards: list[dict[str, Any]] = []
        self._chunks: list[np.ndarray[Any, Any]] = []
        self._lengths: list[int] = []
        self._pending = 0

    def add(self, ids: list[int]) -> None:
        self._chunks.append(np.asarray(ids, dtype=self.dtype))
        self._lengths.append(len(ids))
        self._pending += len(ids)
        if self._pending >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._lengths:
            return
        name = f"{self.split}-{len(self.shards):05d}"
        tokens = np.concatenate(self._chunks) if self._chunks else np.zeros(0, self.dtype)
        offsets = np.zeros(len(self._lengths) + 1, dtype=np.int64)
        np.cumsum(self._lengths, out=offsets[1:])
        self.directory.mkdir(parents=True, exist_ok=True)
        token_path = self.directory / f"{name}.tokens.npy"
        offset_path = self.directory / f"{name}.offsets.npy"
        np.save(token_path, tokens, allow_pickle=False)
        np.save(offset_path, offsets, allow_pickle=False)
        self.shards.append(
            {
                "name": name,
                "documents": len(self._lengths),
                "tokens": int(tokens.size),
                "files": {
                    "tokens": {"path": token_path.name, "sha256": file_sha256(token_path)},
                    "offsets": {"path": offset_path.name, "sha256": file_sha256(offset_path)},
                },
            }
        )
        self._chunks, self._lengths, self._pending = [], [], 0


def _batched_texts(files: Iterable[Path], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for path in files:
        for document in read_jsonl(path):
            batch.append(document.text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def tokenize_corpus(
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    shard_tokens: int = 50_000_000,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Tokenize every split of a frozen corpus into memory-mappable shards.

    Args:
        corpus_dir: Directory written by :class:`~hamic.data.CorpusPipeline`.
        tokenizer_dir: Directory holding ``tokenizer.json`` (e.g. an exported tokenizer).
        output_dir: Target directory; must not already contain a tokenized manifest.
        shard_tokens: Approximate tokens per shard (a shard closes after the document
            that crosses this size).
        batch_size: Documents per ``encode_batch`` call (the Rust tokenizer runs each
            batch on all CPU cores).

    Returns:
        The manifest that was written.
    """
    from tokenizers import Tokenizer

    corpus = Path(corpus_dir)
    source_manifest_path = corpus / SOURCE_MANIFEST
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"{corpus} has no {SOURCE_MANIFEST}; freeze the corpus first")
    out = Path(output_dir)
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError(f"{out} already contains tokenized shards")

    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    vocab_size = int(tokenizer.get_vocab_size())
    dtype = _dtype_for(vocab_size)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    splits: dict[str, Any] = {}
    for split in SPLITS:
        files = [corpus / f["path"] for f in source_manifest["splits"][split]["files"]]
        sink = _ShardSink(out / split, split, dtype, shard_tokens)
        for texts in _batched_texts(files, batch_size):
            for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
                sink.add(encoding.ids)
        sink.flush()
        splits[split] = {
            "documents": sum(s["documents"] for s in sink.shards),
            "tokens": sum(s["tokens"] for s in sink.shards),
            "shards": sink.shards,
        }

    metadata_file = Path(tokenizer_dir) / "hamic_tokenizer.json"
    tokenizer_name = (
        json.loads(metadata_file.read_text(encoding="utf-8"))["spec"]["name"]
        if metadata_file.is_file()
        else Path(tokenizer_dir).name
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "format": "hamic-tokenized-npy-v1",
        "special_tokens_added": False,
        "dtype": dtype.name,
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": vocab_size,
            "sha256": file_sha256(tokenizer_file),
        },
        "source_corpus": {
            "corpus_version": source_manifest.get("corpus_version"),
            "manifest_sha256": file_sha256(source_manifest_path),
        },
        "splits": splits,
        "code": code_revision(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class TokenizedShard:
    """One memory-mapped shard."""

    tokens: np.ndarray[Any, Any]
    offsets: np.ndarray[Any, Any]

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def document(self, index: int) -> np.ndarray[Any, Any]:
        """Token ids of document ``index``."""
        return self.tokens[self.offsets[index] : self.offsets[index + 1]]


class TokenizedCorpus:
    """Read access to shards written by :func:`tokenize_corpus`.

    Args:
        directory: Output directory of :func:`tokenize_corpus`.
        verify: Check every file's SHA-256 against the manifest before use.
    """

    def __init__(self, directory: str | Path, verify: bool = False) -> None:
        self.directory = Path(directory)
        self.manifest: dict[str, Any] = json.loads(
            (self.directory / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if verify:
            self.verify()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the tokenizer that produced the shards."""
        return int(self.manifest["tokenizer"]["vocab_size"])

    def verify(self) -> None:
        """Raise ``ValueError`` if any shard file differs from its recorded checksum."""
        for split, info in self.manifest["splits"].items():
            for shard in info["shards"]:
                for entry in shard["files"].values():
                    path = self.directory / split / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError(f"Checksum mismatch: {path}")

    def shards(self, split: str) -> Iterator[TokenizedShard]:
        """Yield memory-mapped shards of ``split`` in order."""
        for shard in self.manifest["splits"][split]["shards"]:
            base = self.directory / split
            yield TokenizedShard(
                tokens=np.load(base / shard["files"]["tokens"]["path"], mmap_mode="r"),
                offsets=np.load(base / shard["files"]["offsets"]["path"], mmap_mode="r"),
            )

    def documents(self, split: str) -> Iterator[np.ndarray[Any, Any]]:
        """Yield every document's token ids in ``split``, in corpus order."""
        for shard in self.shards(split):
            for index in range(len(shard)):
                yield shard.document(index)
