import json
from pathlib import Path

import numpy as np
import pytest

from hamic.data import (
    CorpusPipeline,
    Document,
    JsonlSource,
    NormalizeStage,
    SourceRecord,
    SourceRegistry,
    SourceStatus,
    SplitConfig,
    SplitStage,
    TokenizedCorpus,
    tokenize_corpus,
    write_jsonl,
)
from hamic.tokenization import TokenizerKind
from tests.conftest import VIETNAMESE_SENTENCES


@pytest.fixture
def frozen_corpus(tmp_path: Path) -> Path:
    import datetime as dt

    texts = [" ".join(VIETNAMESE_SENTENCES[i:] + VIETNAMESE_SENTENCES[:i]) * 2 for i in range(8)]
    texts += [t + f" Phiên bản số {n}." for n, t in enumerate(texts * 4)]
    source = tmp_path / "src.jsonl"
    write_jsonl(source, [Document(f"d{i}", t, "s") for i, t in enumerate(texts)])
    record = SourceRecord(
        source_id="s",
        publisher="test",
        location="local",
        revision="v1",
        license="CC-BY-SA-4.0",
        status=SourceStatus.APPROVED_RESEARCH,
        reviewer="HAMIC",
        reviewed_on=dt.date(2026, 11, 15),
        pii_risk="low",
    )
    out = tmp_path / "corpus"
    # No dedup here: the fixture texts are deliberately similar, and this test is about
    # tokenization, not cleaning.
    stages = [NormalizeStage(), SplitStage(SplitConfig(validation=0.2, test=0.2))]
    CorpusPipeline(SourceRegistry({"s": record}), stages=stages).run(
        [JsonlSource("s", [source])], out, "t"
    )
    return out


def test_tokenized_shards_roundtrip(tmp_path, frozen_corpus, exported_tokenizers):
    from tokenizers import Tokenizer

    tok_dir = exported_tokenizers[TokenizerKind.ENCODER]
    out = tmp_path / "tokens"
    manifest = tokenize_corpus(frozen_corpus, tok_dir, out, shard_tokens=500)

    assert manifest["dtype"] == "uint16"
    assert manifest["special_tokens_added"] is False
    train = manifest["splits"]["train"]
    assert len(train["shards"]) > 1  # small shard size forces rotation

    corpus = TokenizedCorpus(out, verify=True)
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    texts = []
    source = json.loads((frozen_corpus / "manifest.json").read_text(encoding="utf-8"))
    from hamic.data import read_jsonl

    for f in source["splits"]["train"]["files"]:
        texts += [d.text for d in read_jsonl(frozen_corpus / f["path"])]
    stored = list(corpus.documents("train"))
    assert len(stored) == len(texts) == train["documents"]
    for ids, text in zip(stored, texts, strict=True):
        assert ids.tolist() == tokenizer.encode(text, add_special_tokens=False).ids
    assert sum(len(d) for d in stored) == train["tokens"]
    lengths = corpus.token_byte_lengths()
    for ids, text in zip(stored, texts, strict=True):
        assert int(lengths[ids].sum()) == len(text.encode("utf-8")) + 1  # + prefix space
    assert lengths[:5].sum() == 0  # special tokens decode to no bytes


def test_tokenized_corpus_detects_tampering(tmp_path, frozen_corpus, exported_tokenizers):
    out = tmp_path / "tokens"
    manifest = tokenize_corpus(frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out)
    assert manifest["tokenizer"]["text_preparation"]["lookalikes"]["name"] == "confusables.yaml"
    shard = manifest["splits"]["train"]["shards"][0]["files"]["tokens"]["path"]
    np.save(out / "train" / shard, np.zeros(3, dtype=np.uint16))
    with pytest.raises(ValueError, match="Checksum"):
        TokenizedCorpus(out, verify=True)
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        TokenizedCorpus(out).token_byte_lengths()
    with pytest.raises(FileExistsError):
        tokenize_corpus(frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out)


def test_large_vocab_uses_uint32():
    from hamic.data.tokenized import _dtype_for

    assert _dtype_for(65536) == np.uint16
    assert _dtype_for(65537) == np.uint32


def test_publish_and_fetch_roundtrip(tmp_path, frozen_corpus, exported_tokenizers):
    from hamic.checkpoint import LocalObjectStore
    from hamic.data import fetch_tokenized, publish_tokenized

    out = tmp_path / "tokens"
    tokenize_corpus(frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out)
    store = LocalObjectStore(tmp_path / "store")
    with pytest.raises(FileNotFoundError):
        fetch_tokenized(store, "tokenized/x", tmp_path / "missing")
    count = publish_tokenized(out, store, "tokenized/t/tok")
    assert "tokenized/t/tok/tokenizer.json" in store.list_files("tokenized/t/tok")
    assert count == len(store.list_files("tokenized/t/tok"))

    local = tmp_path / "fetched"
    fetched = fetch_tokenized(store, "tokenized/t/tok", local)
    assert (
        next(fetched.documents("train")).tolist()
        == next(TokenizedCorpus(out).documents("train")).tolist()
    )
    assert fetched.token_byte_lengths().sum() > 0
    # A corrupt local copy is replaced on the next fetch.
    shard = next((local / "train").glob("*.tokens.npy"))
    np.save(shard, np.zeros(3, dtype=np.uint16))
    fetch_tokenized(store, "tokenized/t/tok", local)
    TokenizedCorpus(local, verify=True)


def test_sample_published_tokenized_roundtrips_to_text(
    tmp_path, frozen_corpus, exported_tokenizers
):
    from hamic.checkpoint import LocalObjectStore
    from hamic.data import publish_tokenized, read_jsonl, sample_published_tokenized

    out = tmp_path / "tokens"
    manifest = tokenize_corpus(
        frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out, shard_tokens=500
    )
    store = LocalObjectStore(tmp_path / "store")
    prefix = "tokenized/t/tok"
    publish_tokenized(out, store, prefix)
    sample = sample_published_tokenized(
        out,
        store,
        prefix,
        tmp_path / "sample",
        target_tokens=manifest["splits"]["train"]["tokens"] // 2,
        source_weights={"s": 2.0},
        shard_documents=5,
        batch_size=3,
    )
    documents = [
        document
        for entry in sample["files"]
        for document in read_jsonl(tmp_path / "sample" / entry["path"])
    ]
    assert sample["selection"]["roundtrip_checked"] is True
    assert sample["selection"]["covered_tokens"] > 0
    assert len(documents) == sample["documents"]
    assert all(document.source_id == "s" for document in documents)
    assert not any((tmp_path / "sample").glob("hamic-token-sample-*"))

    explicit = sample_published_tokenized(
        out,
        store,
        prefix,
        tmp_path / "explicit",
        target_tokens=1,
        shard_indices=[0],
    )
    assert explicit["selection"]["method"] == "explicit-shard-indices-v1"
    assert [item["index"] for item in explicit["selected_shards"]] == [0]

    heldout = sample_published_tokenized(
        out,
        store,
        prefix,
        tmp_path / "heldout",
        target_tokens=1,
        shard_indices=[0],
        split="validation",
    )
    assert heldout["split"] == "validation"
