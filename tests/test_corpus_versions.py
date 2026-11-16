"""Corpus versions: persistent dedup index, deltas, assembled versions, server-side copies."""

import datetime as dt
import json
import os
from contextlib import nullcontext

import numpy as np
import pytest

from hamic.checkpoint import LocalObjectStore
from hamic.data import (
    CorpusPipeline,
    DedupIndex,
    Document,
    JsonlSource,
    SourceRecord,
    SourceRegistry,
    SourceStatus,
    TokenizedCorpus,
    assemble_tokenized,
    fetch_tokenized,
    index_corpus,
    publish_tokenized,
    read_jsonl,
    shuffled_files,
    tokenize_corpus,
    write_jsonl,
)
from hamic.data.config import PipelineConfig
from hamic.data.stages import minhash_signature
from hamic.tokenization import TokenizerKind
from tests.conftest import VIETNAMESE_SENTENCES

_SYLLABLES = sorted({w.strip(".,") for s in VIETNAMESE_SENTENCES for w in s.split()} - {""})


def _text(i: int) -> str:
    """A distinct pseudo-Vietnamese document per ``i`` (no shared 5-syllable shingles)."""
    rng = np.random.default_rng(i)
    words = rng.choice(_SYLLABLES, size=60)
    return " ".join(words) + "."


def _registry() -> SourceRegistry:
    record = SourceRecord(
        source_id="web",
        publisher="test",
        location="local",
        revision="v1",
        license="CC-BY-SA-4.0",
        status=SourceStatus.APPROVED_RESEARCH,
        reviewer="HAMIC",
        reviewed_on=dt.date(2026, 11, 16),
        pii_risk="low",
    )
    return SourceRegistry({"web": record})


def _build(tmp_path, name, documents, index=None, base=None):
    source = tmp_path / f"{name}.jsonl"
    write_jsonl(source, documents)
    config = PipelineConfig.from_mapping({"quality": {"min_chars": 20}})
    pipeline = CorpusPipeline(_registry(), config=config, dedup_index=index, base_corpus=base)
    return pipeline.run([JsonlSource("web", [source])], tmp_path / name, name)


def _texts(corpus_dir):
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        d.text
        for split in manifest["splits"].values()
        for f in split["files"]
        for d in read_jsonl(corpus_dir / f["path"])
    }


def test_index_finds_exact_and_near_duplicates(tmp_path):
    index = DedupIndex()
    text = _text(1) * 3
    index.add("ab" * 32, minhash_signature(text))
    assert index.has_exact("ab" * 32)
    assert not index.has_exact("cd" * 32)
    assert index.has_near(minhash_signature(text + " Thêm một câu."))
    assert not index.has_near(minhash_signature(_text(5) * 3))
    index.save(tmp_path / "idx", version="v1")
    loaded = DedupIndex.load(tmp_path / "idx")
    assert len(loaded) == 1
    assert loaded.versions == ["v1"]
    assert loaded.has_exact("ab" * 32)
    assert loaded.has_near(minhash_signature(text + " Thêm một câu."))


def test_delta_keeps_only_new_documents(tmp_path):
    base_docs = [Document(f"b{i}", _text(i) * 2, "web") for i in range(40)]
    base = _build(tmp_path, "v1", base_docs)
    index = DedupIndex()
    assert index_corpus(base.output_dir, index, "v1") == sum(base.counts.values())
    index.save(tmp_path / "idx")

    stored = sorted(_texts(base.output_dir))
    repeated = [Document(f"r{i}", t, "web") for i, t in enumerate(stored[:5])]
    near = [Document(f"n{i}", t + " Một câu thêm.", "web") for i, t in enumerate(stored[5:10])]
    new = [Document(f"x{i}", _text(100 + i) * 2, "web") for i in range(15)]
    index = DedupIndex.load(tmp_path / "idx")
    delta = _build(
        tmp_path, "v2-delta", repeated + near + new, index=index, base={"corpus_version": "v1"}
    )
    index.save(tmp_path / "idx", version="v2")

    kept = _texts(delta.output_dir)
    assert kept.isdisjoint(stored)
    assert len(kept) == 15
    manifest = delta.manifest
    assert manifest["base_corpus"] == {"corpus_version": "v1"}
    assert manifest["dedup_index"]["versions"] == ["v1"]
    stages = {s["name"]: s for s in manifest["stages"]}
    assert stages["exact_dedup"]["rejected"]["exact_duplicate"] == 5
    assert stages["near_dedup"]["rejected"]["near_duplicate"] == 5
    assert DedupIndex.load(tmp_path / "idx").versions == ["v1", "v2"]


@pytest.fixture
def versions(tmp_path, exported_tokenizers):
    tokenizer = exported_tokenizers[TokenizerKind.ENCODER]
    base = _build(tmp_path, "v1", [Document(f"b{i}", _text(i) * 2, "web") for i in range(40)])
    index = DedupIndex()
    index_corpus(base.output_dir, index, "v1")
    delta = _build(
        tmp_path,
        "v2-delta",
        [Document(f"x{i}", _text(100 + i) * 2, "web") for i in range(30)],
        index=index,
    )
    tokenize_corpus(base.output_dir, tokenizer, tmp_path / "tok-v1", shard_tokens=2000)
    tokenize_corpus(delta.output_dir, tokenizer, tmp_path / "tok-v2-delta", shard_tokens=2000)
    return tmp_path


def test_assembled_version_links_base_and_delta(versions):
    out = versions / "tok-v2"
    manifest = assemble_tokenized(versions / "tok-v1", versions / "tok-v2-delta", out, "v2")
    v1 = TokenizedCorpus(versions / "tok-v1")
    delta = TokenizedCorpus(versions / "tok-v2-delta")
    v2 = TokenizedCorpus(out, verify=True)
    for split in ("train", "validation", "test"):
        expected = [d.tolist() for d in v1.documents(split)] + [
            d.tolist() for d in delta.documents(split)
        ]
        assert [d.tolist() for d in v2.documents(split)] == expected
    first = out / "train" / manifest["splits"]["train"]["shards"][0]["files"]["tokens"]["path"]
    assert os.stat(first).st_nlink >= 2  # hard link, no copy
    assert v2.source_ids == ["web"]
    assert v2.manifest["source_corpus"]["corpus_version"] == "v2"
    assert "manifest_sha256" in v2.manifest["source_corpus"]
    with pytest.raises(FileExistsError):
        assemble_tokenized(versions / "tok-v1", versions / "tok-v2-delta", out, "v2")


class _RecordingStore(LocalObjectStore):
    def __init__(self, root, fail_copies=False):
        super().__init__(root)
        self.fail_copies = fail_copies
        self.calls = []

    def put_files(self, files, message, copies=None):
        self.calls.append((dict(files), dict(copies or {})))
        if copies and self.fail_copies:
            raise RuntimeError("copies not supported")
        super().put_files(files, message, copies=copies)


@pytest.mark.parametrize("fail_copies", [False, True])
def test_publish_copies_unchanged_files_on_the_server(versions, fail_copies):
    assemble_tokenized(versions / "tok-v1", versions / "tok-v2-delta", versions / "tok-v2", "v2")
    store = _RecordingStore(versions / "store", fail_copies=fail_copies)
    publish_tokenized(versions / "tok-v1", store, "tokenized/v1")
    store.calls.clear()
    expectation = pytest.warns(UserWarning, match="uploading") if fail_copies else nullcontext()
    with expectation:
        publish_tokenized(versions / "tok-v2", store, "tokenized/v2", copy_from="tokenized/v1")
    uploads, copies = store.calls[0]
    v1_files = {
        f"tokenized/v1/{split}/{entry['path']}"
        for split, info in TokenizedCorpus(versions / "tok-v1").manifest["splits"].items()
        for shard in info["shards"]
        for entry in shard["files"].values()
    }
    # Every v1 file (and the tokenizer) is copied; byte-identical delta files may be too,
    # and identical v1 files may share one source path.
    assert len(copies) >= len(v1_files) + 1
    assert "tokenized/v2/tokenizer.json" in copies
    assert not set(copies) & set(uploads)
    for target, source in copies.items():
        assert (store.root / target).read_bytes() == (store.root / source).read_bytes()
    fetched = fetch_tokenized(store, "tokenized/v2", versions / "fetched")
    original = TokenizedCorpus(versions / "tok-v2")
    for split in ("train", "validation", "test"):
        assert [d.tolist() for d in fetched.documents(split)] == [
            d.tolist() for d in original.documents(split)
        ]


def test_shuffled_files_is_seeded():
    files = [f"data/x/{i:03d}.parquet" for i in range(30)]
    order = shuffled_files(files, "hamic-fineweb2-v1")
    assert sorted(order) == sorted(files)
    assert order == shuffled_files(list(reversed(files)), "hamic-fineweb2-v1")
    assert order != shuffled_files(files, "other-seed")
    assert np.mean([a != b for a, b in zip(order, sorted(files), strict=True)]) > 0.5


def test_assemble_on_published_base_keeps_only_the_delta_locally(versions):
    store = LocalObjectStore(versions / "store")
    publish_tokenized(versions / "tok-v1", store, "tokenized/v1")
    published = json.loads(store.read_bytes("tokenized/v1/tokenized_manifest.json"))
    out = versions / "tok-v2-remote"
    assemble_tokenized(
        None,
        versions / "tok-v2-delta",
        out,
        "v2",
        base_manifest=published,
        base_prefix="tokenized/v1",
    )
    remote = json.loads((out / "remote_files.json").read_text(encoding="utf-8"))
    local = {p.relative_to(out).as_posix() for p in out.rglob("*.npy")}
    assert remote
    assert local.isdisjoint(remote)
    delta_files = sum(
        len(shard["files"])
        for info in TokenizedCorpus(versions / "tok-v2-delta").manifest["splits"].values()
        for shard in info["shards"]
    )
    assert len(local) == delta_files

    publish_tokenized(out, store, "tokenized/v2")
    assert store.read_bytes("tokenized/v2/remote_files.json") is None
    full = versions / "tok-v2-full"
    assemble_tokenized(versions / "tok-v1", versions / "tok-v2-delta", full, "v2")
    fetched = fetch_tokenized(store, "tokenized/v2", versions / "fetched-remote")
    reference = TokenizedCorpus(full)
    for split in ("train", "validation", "test"):
        assert [d.tolist() for d in fetched.documents(split)] == [
            d.tolist() for d in reference.documents(split)
        ]


def test_publish_rejects_missing_files_without_a_remote_map(versions):
    out = versions / "tok-v2"
    assemble_tokenized(versions / "tok-v1", versions / "tok-v2-delta", out, "v2")
    next(out.glob("train/*.tokens.npy")).unlink()
    with pytest.raises(FileNotFoundError, match="not listed as published"):
        publish_tokenized(out, LocalObjectStore(versions / "store"), "tokenized/v2")
