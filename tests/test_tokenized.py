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


def test_tokenized_corpus_detects_tampering(tmp_path, frozen_corpus, exported_tokenizers):
    out = tmp_path / "tokens"
    manifest = tokenize_corpus(frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out)
    shard = manifest["splits"]["train"]["shards"][0]["files"]["tokens"]["path"]
    np.save(out / "train" / shard, np.zeros(3, dtype=np.uint16))
    with pytest.raises(ValueError, match="Checksum"):
        TokenizedCorpus(out, verify=True)
    with pytest.raises(FileExistsError):
        tokenize_corpus(frozen_corpus, exported_tokenizers[TokenizerKind.ENCODER], out)


def test_large_vocab_uses_uint32():
    from hamic.data.tokenized import _dtype_for

    assert _dtype_for(65536) == np.uint16
    assert _dtype_for(65537) == np.uint32
