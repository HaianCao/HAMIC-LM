"""Tokenized, memory-mappable shards used as training input.

Each shard is a pair of ``.npy`` files:

* ``<name>.tokens.npy``: all token ids of the shard's documents, concatenated, as
  ``uint16`` when the vocabulary fits (HAMIC's ~20K/32K vocabularies) else ``uint32``;
* ``<name>.offsets.npy``: ``int64`` document boundaries, ``documents + 1`` entries, so
  document ``i`` is ``tokens[offsets[i]:offsets[i + 1]]``;
* ``<name>.sources.npy`` (optional): ``uint8`` index of each document's source into the
  manifest's ``source_ids`` list. Written by :func:`tokenize_corpus`; added to older
  directories by :func:`annotate_sources`.

Ids carry **no special tokens**: adding ``<s>``/``</s>``, packing and masking belong to the
training data loader. A copy of ``tokenizer.json`` travels with the shards, so a tokenized
directory is self-contained (e.g. for decoding or bits-per-byte). ``tokenized_manifest.json``
(written last) records the tokenizer and source-corpus hashes and a checksum for every file.
"""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hamic.data.documents import read_jsonl
from hamic.provenance import code_revision, file_sha256, utc_now

if TYPE_CHECKING:
    from hamic.checkpoint.storage import ObjectStore

__all__ = [
    "TokenizedCorpus",
    "TokenizedShard",
    "annotate_sources",
    "fetch_tokenized",
    "publish_tokenized",
    "tokenize_corpus",
    "tokenize_samples",
]

MANIFEST_NAME = "tokenized_manifest.json"
SOURCE_MANIFEST = "manifest.json"
SPLITS = ("train", "validation", "test")
TOKENIZER_NAME = "tokenizer.json"
# Written by ``assemble_tokenized`` on a published base: local path -> published path of
# every file that is not stored locally. Never uploaded.
REMOTE_FILES_NAME = "remote_files.json"


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
        self._sources: list[int] = []
        self._pending = 0

    def add(self, ids: list[int], source: int) -> None:
        self._chunks.append(np.asarray(ids, dtype=self.dtype))
        self._lengths.append(len(ids))
        self._sources.append(source)
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
                    "sources": _write_sources(self.directory, name, self._sources),
                },
            }
        )
        self._chunks, self._lengths, self._sources, self._pending = [], [], [], 0


def _write_sources(directory: Path, name: str, sources: list[int]) -> dict[str, str]:
    path = directory / f"{name}.sources.npy"
    np.save(path, np.asarray(sources, dtype=np.uint8), allow_pickle=False)
    return {"path": path.name, "sha256": file_sha256(path)}


def _source_ids(source_manifest: dict[str, Any]) -> list[str]:
    ids = [str(s["source_id"]) for s in source_manifest.get("sources", [])]
    if len(ids) > np.iinfo(np.uint8).max + 1:
        raise ValueError("At most 256 sources fit the uint8 source index")
    return ids


def _split_documents(
    corpus: Path, source_manifest: dict[str, Any], split: str
) -> Iterator[tuple[str, str]]:
    """Yield ``(text, source_id)`` of a split in the order the tokenizer reads them."""
    for entry in source_manifest["splits"][split]["files"]:
        for document in read_jsonl(corpus / entry["path"]):
            yield document.text, document.source_id


def _batched(
    documents: Iterable[tuple[str, str]], batch_size: int
) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _text_preparation() -> dict[str, Any]:
    """Manifest record of :func:`hamic.tokenization.prepare_text` (its look-alike table)."""
    from hamic.text.lookalikes import lookalike_identity

    return {"lookalikes": lookalike_identity()}


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
    from hamic.tokenization import prepare_text

    vocab_size = int(tokenizer.get_vocab_size())
    dtype = _dtype_for(vocab_size)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_ids = _source_ids(source_manifest)
    source_index = {source_id: i for i, source_id in enumerate(source_ids)}

    splits: dict[str, Any] = {}
    for split in SPLITS:
        sink = _ShardSink(out / split, split, dtype, shard_tokens)
        for batch in _batched(_split_documents(corpus, source_manifest, split), batch_size):
            encodings = tokenizer.encode_batch(
                [prepare_text(t) for t, _ in batch], add_special_tokens=False
            )
            for encoding, (_, source_id) in zip(encodings, batch, strict=True):
                sink.add(encoding.ids, source_index[source_id])
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
            "text_preparation": _text_preparation(),
        },
        "source_corpus": {
            "corpus_version": source_manifest.get("corpus_version"),
            "manifest_sha256": file_sha256(source_manifest_path),
        },
        "source_ids": source_ids,
        "splits": splits,
        "code": code_revision(),
    }
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_file, out / TOKENIZER_NAME)
    _write_manifest(out, manifest)
    return manifest


def tokenize_samples(
    sample_dirs: Sequence[str | Path],
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    shard_tokens: int = 50_000_000,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Tokenize one or more verified decoded-sample directories.

    Each input must contain ``sample_manifest.json``. Its declared split is
    preserved, allowing independent train and held-out samples to be combined
    without copying their compressed JSONL payloads.
    """
    from tokenizers import Tokenizer

    samples: list[tuple[Path, dict[str, Any]]] = []
    source_ids: list[str] = []
    for value in sample_dirs:
        root = Path(value)
        path = root / "sample_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        split = str(manifest["split"])
        if split not in SPLITS:
            raise ValueError(f"invalid sample split: {split}")
        for source_id in manifest.get("sources", {}):
            if source_id not in source_ids:
                source_ids.append(source_id)
        samples.append((root, manifest))
    if not samples:
        raise ValueError("at least one sample directory is required")

    tokenizer_file = Path(tokenizer_dir) / TOKENIZER_NAME
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    from hamic.tokenization import prepare_text

    vocab_size = int(tokenizer.get_vocab_size())
    dtype = _dtype_for(vocab_size)
    source_index = {source_id: i for i, source_id in enumerate(source_ids)}
    out = Path(output_dir)
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError(f"{out} already contains tokenized shards")

    splits: dict[str, Any] = {}
    for split in SPLITS:
        sink = _ShardSink(out / split, split, dtype, shard_tokens)
        documents = (
            (document.text, document.source_id)
            for root, manifest in samples
            if manifest["split"] == split
            for entry in manifest["files"]
            for document in read_jsonl(root / entry["path"])
        )
        for batch in _batched(documents, batch_size):
            encodings = tokenizer.encode_batch(
                [prepare_text(text) for text, _ in batch], add_special_tokens=False
            )
            for encoding, (_, source_id) in zip(encodings, batch, strict=True):
                sink.add(encoding.ids, source_index[source_id])
        sink.flush()
        splits[split] = {
            "documents": sum(item["documents"] for item in sink.shards),
            "tokens": sum(item["tokens"] for item in sink.shards),
            "shards": sink.shards,
        }

    metadata_file = Path(tokenizer_dir) / "hamic_tokenizer.json"
    tokenizer_name = json.loads(metadata_file.read_text(encoding="utf-8"))["spec"]["name"]
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
            "text_preparation": _text_preparation(),
        },
        "source_corpus": {
            "corpus_version": "decoded-samples",
            "samples": [file_sha256(root / "sample_manifest.json") for root, _ in samples],
        },
        "source_ids": source_ids,
        "splits": splits,
        "code": code_revision(),
    }
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_file, out / TOKENIZER_NAME)
    _write_manifest(out, manifest)
    return manifest


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically (it is the last file of a valid directory)."""
    target = directory / MANIFEST_NAME
    staging = directory / f"{MANIFEST_NAME}.tmp"
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    staging.replace(target)


def annotate_sources(corpus_dir: str | Path, tokenized_dir: str | Path) -> dict[str, Any]:
    """Add per-document source ids to a tokenized directory written without them.

    Reads the frozen corpus in the order :func:`tokenize_corpus` read it, checks that the
    document counts of every shard match, writes ``<shard>.sources.npy`` files and rewrites
    the manifest. Token files are untouched, so training on the directory is unchanged.

    Raises:
        ValueError: The corpus is not the one the shards were tokenized from.
    """
    corpus = Path(corpus_dir)
    tokenized = TokenizedCorpus(tokenized_dir, verify=True)
    source_manifest_path = corpus / SOURCE_MANIFEST
    manifest = tokenized.manifest
    if file_sha256(source_manifest_path) != manifest["source_corpus"]["manifest_sha256"]:
        raise ValueError(f"{corpus} is not the corpus these shards were tokenized from")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_ids = _source_ids(source_manifest)
    source_index = {source_id: i for i, source_id in enumerate(source_ids)}
    for split in SPLITS:
        documents = _split_documents(corpus, source_manifest, split)
        for shard in manifest["splits"][split]["shards"]:
            sources = [
                source_index[source_id]
                for _, (_, source_id) in zip(range(shard["documents"]), documents, strict=False)
            ]
            if len(sources) != shard["documents"]:
                raise ValueError(f"Corpus ended before shard {shard['name']} was complete")
            shard["files"]["sources"] = _write_sources(
                tokenized.directory / split, shard["name"], sources
            )
        if next(documents, None) is not None:
            raise ValueError(f"Corpus split {split!r} has more documents than the shards")
    manifest["source_ids"] = source_ids
    _write_manifest(tokenized.directory, manifest)
    return manifest


@dataclass(frozen=True)
class TokenizedShard:
    """One memory-mapped shard."""

    tokens: np.ndarray[Any, Any]
    offsets: np.ndarray[Any, Any]
    sources: np.ndarray[Any, Any] | None = None

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

    @property
    def source_ids(self) -> list[str] | None:
        """Source ids indexed by ``TokenizedShard.sources`` (``None``: not recorded)."""
        ids = self.manifest.get("source_ids")
        return list(ids) if ids is not None else None

    def verify(self) -> None:
        """Raise ``ValueError`` if any shard file differs from its recorded checksum."""
        for split, info in self.manifest["splits"].items():
            for shard in info["shards"]:
                for entry in shard["files"].values():
                    path = self.directory / split / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError(f"Checksum mismatch: {path}")

    def tokenizer_file(self) -> Path:
        """Path of the ``tokenizer.json`` copy, checked against the manifest hash.

        Raises:
            FileNotFoundError: The directory has no tokenizer copy.
            ValueError: The copy does not match the manifest.
        """
        path = self.directory / TOKENIZER_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{self.directory} has no {TOKENIZER_NAME}")
        if file_sha256(path) != self.manifest["tokenizer"]["sha256"]:
            raise ValueError(f"{path} does not match the tokenizer recorded in the manifest")
        return path

    def token_byte_lengths(self) -> np.ndarray[Any, Any]:
        """UTF-8 bytes each token id decodes to (``0`` for special and added tokens).

        Used to report bits per byte, which, unlike loss per token, is comparable across
        tokenizers. Reads the ``tokenizer.json`` copy stored with the shards and checks it
        against the manifest hash.

        HAMIC tokenizers count their virtual spaces (prefix space, after openers) as bytes
        (0.41% of the bytes of v0.12 held-out text plus one per document), so their bits per
        byte read about 0.4% low against a tokenizer without virtual spaces.

        Raises:
            FileNotFoundError: The directory has no tokenizer copy.
            ValueError: The copy does not match the manifest, or is not byte-level BPE.
        """
        data = json.loads(self.tokenizer_file().read_text(encoding="utf-8"))
        decoder = data.get("decoder") or {}
        kinds = {d.get("type") for d in decoder.get("decoders", [])} | {decoder.get("type")}
        if "ByteLevel" not in kinds:
            raise ValueError("Byte lengths need a byte-level tokenizer")
        lengths = np.zeros(self.vocab_size, dtype=np.int64)
        for token, index in data["model"]["vocab"].items():
            lengths[index] = len(token)  # one alphabet character per byte
        for added in data.get("added_tokens", []):
            lengths[added["id"]] = 0
        return lengths

    def shards(self, split: str) -> Iterator[TokenizedShard]:
        """Yield memory-mapped shards of ``split`` in order."""
        for shard in self.manifest["splits"][split]["shards"]:
            base = self.directory / split
            files = shard["files"]
            yield TokenizedShard(
                tokens=np.load(base / files["tokens"]["path"], mmap_mode="r"),
                offsets=np.load(base / files["offsets"]["path"], mmap_mode="r"),
                sources=(np.load(base / files["sources"]["path"]) if "sources" in files else None),
            )

    def documents(self, split: str) -> Iterator[np.ndarray[Any, Any]]:
        """Yield every document's token ids in ``split``, in corpus order."""
        for shard in self.shards(split):
            for index in range(len(shard)):
                yield shard.document(index)


def publish_tokenized(
    directory: str | Path, store: ObjectStore, prefix: str, copy_from: str | None = None
) -> int:
    """Upload a tokenized directory to ``store`` under ``prefix``.

    Shards and the tokenizer copy go in one commit; the manifest follows in a second one,
    so a reader that finds the manifest also finds every file it lists.

    Args:
        directory: Tokenized directory to publish. It may hold only part of its files
            when ``remote_files.json`` (written by ``assemble_tokenized`` on a published
            base) lists where the others are already published; those are copied on the
            server.
        store: Target store.
        prefix: Remote folder, e.g. ``tokenized/v0.3/t1-bbpe-20544``.
        copy_from: Remote folder of an already published version. Files whose checksum
            matches one there are copied inside the store instead of uploaded, so a new
            complete version only uploads its delta. When the store rejects the copies of
            files that exist locally, those are uploaded instead.

    Returns:
        Number of files written under ``prefix``.
    """
    root = Path(directory)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    remote_path = root / REMOTE_FILES_NAME
    remote: dict[str, str] = (
        json.loads(remote_path.read_text(encoding="utf-8")) if remote_path.is_file() else {}
    )
    _verify_local_files(root, manifest, remote)
    skip = {MANIFEST_NAME, REMOTE_FILES_NAME}
    payload = {
        f"{prefix}/{path.relative_to(root).as_posix()}": path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in skip and not path.name.endswith(".tmp")
    }
    required = {f"{prefix}/{local}": published for local, published in remote.items()}
    optional = (
        _published_copies(TokenizedCorpus(root), store, prefix, copy_from) if copy_from else {}
    )
    optional = {k: v for k, v in optional.items() if k in payload and k not in required}
    copies = {**required, **optional}
    uploads = {
        remote_name: local for remote_name, local in payload.items() if remote_name not in copies
    }
    message = f"tokenized shards {prefix}"
    try:
        store.put_files(uploads, message=message, copies=copies)
    except Exception as error:
        if required or not copies:
            raise
        warnings.warn(f"Server-side copy failed ({error}); uploading every file", stacklevel=2)
        store.put_files(payload, message=message)
    store.put_files(
        {f"{prefix}/{MANIFEST_NAME}": root / MANIFEST_NAME},
        message=f"tokenized manifest {prefix}",
    )
    return len(payload) + len(required) + 1


def _verify_local_files(root: Path, manifest: dict[str, Any], remote: dict[str, str]) -> None:
    """Check every listed file that is stored locally; the rest must be in ``remote``."""
    for split, info in manifest["splits"].items():
        for shard in info["shards"]:
            for entry in shard["files"].values():
                relative = f"{split}/{entry['path']}"
                if relative in remote:
                    continue
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(f"{path} is missing and not listed as published")
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")


def _published_copies(
    corpus: TokenizedCorpus, store: ObjectStore, prefix: str, copy_from: str
) -> dict[str, str]:
    """Map files of ``corpus`` to identical files already published under ``copy_from``."""
    raw = store.read_bytes(f"{copy_from}/{MANIFEST_NAME}")
    if raw is None:
        raise FileNotFoundError(f"No tokenized manifest under {copy_from!r}")
    published = json.loads(raw.decode("utf-8"))
    by_sha: dict[str, str] = {}
    for split, info in published["splits"].items():
        for shard in info["shards"]:
            for entry in shard["files"].values():
                by_sha[entry["sha256"]] = f"{copy_from}/{split}/{entry['path']}"
    copies: dict[str, str] = {}
    for split, info in corpus.manifest["splits"].items():
        for shard in info["shards"]:
            for entry in shard["files"].values():
                if entry["sha256"] in by_sha:
                    copies[f"{prefix}/{split}/{entry['path']}"] = by_sha[entry["sha256"]]
    if published["tokenizer"]["sha256"] == corpus.manifest["tokenizer"]["sha256"]:
        copies[f"{prefix}/{TOKENIZER_NAME}"] = f"{copy_from}/{TOKENIZER_NAME}"
    return copies


def fetch_tokenized(store: ObjectStore, prefix: str, destination: str | Path) -> TokenizedCorpus:
    """Download shards published by :func:`publish_tokenized` and verify every checksum.

    An existing ``destination`` is reused when it verifies; a partial or corrupt copy is
    replaced.
    """
    target = Path(destination)
    if (target / MANIFEST_NAME).is_file():
        try:
            return TokenizedCorpus(target, verify=True)
        except (ValueError, FileNotFoundError):
            pass
    if target.exists():
        shutil.rmtree(target)
    if store.read_bytes(f"{prefix}/{MANIFEST_NAME}") is None:
        raise FileNotFoundError(f"No tokenized manifest under {prefix!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    store.download_prefix(prefix, target)
    return TokenizedCorpus(target, verify=True)
