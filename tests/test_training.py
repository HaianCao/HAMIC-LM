import datetime as dt
from pathlib import Path

import pytest
import torch
import yaml

from hamic.checkpoint import LocalObjectStore
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
from hamic.runtime import SessionClock
from hamic.tokenization import TokenizerKind
from hamic.training import (
    DataCursor,
    MaskedLM,
    PackedBatches,
    PretrainConfig,
    Pretrainer,
)
from tests.conftest import VIETNAMESE_SENTENCES


@pytest.fixture(scope="module")
def tokenized(tmp_path_factory, exported_tokenizers) -> Path:
    root = tmp_path_factory.mktemp("training")
    texts = [
        " ".join(VIETNAMESE_SENTENCES[i % 8 :] + VIETNAMESE_SENTENCES[: i % 8]) + f" Số {i}."
        for i in range(60)
    ]
    source = root / "src.jsonl"
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
    stages = [NormalizeStage(), SplitStage(SplitConfig(validation=0.15, test=0.1))]
    CorpusPipeline(SourceRegistry({"s": record}), stages=stages).run(
        [JsonlSource("s", [source])], root / "corpus", "t"
    )
    out = root / "tokens"
    tokenize_corpus(root / "corpus", exported_tokenizers[TokenizerKind.ENCODER], out)
    return out


def _preset(path: Path, architecture: str, vocab: int) -> Path:
    decoder = architecture in {"llama", "qwen3"}
    data = {
        "name": f"test-{architecture}",
        "architecture": architecture,
        "label_parameters": 1e5,
        "label_tolerance": 100.0,
        "num_layers": 2,
        "hidden_size": 32,
        "num_heads": 4,
        "head_dim": 8,
        "intermediate_size": 64,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
    }
    if decoder:
        data["num_kv_heads"] = 2
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _config(tmp_path, tokenized, architecture="qwen3", **overrides) -> PretrainConfig:
    vocab = TokenizedCorpus(tokenized).vocab_size
    values = {
        "run_id": "test-run",
        "model_preset": str(_preset(tmp_path / f"{architecture}.yaml", architecture, vocab)),
        "tokenized": str(tokenized),
        "objective": "clm" if architecture in {"llama", "qwen3"} else "mlm",
        "seq_len": 16,
        "micro_batch_size": 2,
        "grad_accum_steps": 2,
        "total_tokens": 16 * 2 * 2 * 8,  # 8 optimizer steps
        "learning_rate": 3e-3,
        "precision": "fp32",
        "device": "cpu",
        "deterministic": True,
        "log_every_steps": 1,
        "eval_every_steps": 4,
        "eval_sequences": 4,
        "checkpoint_every_steps": 0,
        "checkpoint_every_minutes": 0,
    }
    values.update(overrides)
    return PretrainConfig(**values)


# -- data ------------------------------------------------------------------------------


def test_packed_batches_are_full_and_resume_exactly(tokenized):
    corpus = TokenizedCorpus(tokenized)
    loader = PackedBatches(corpus, "train", seq_len=16, batch_size=3, bos_id=0, eos_id=2)
    first = [loader.next_batch() for _ in range(4)]
    cursor = loader.cursor()
    expected = [loader.next_batch() for _ in range(5)]
    resumed = PackedBatches(
        corpus, "train", seq_len=16, batch_size=3, bos_id=0, eos_id=2, cursor=cursor
    )
    assert all(torch.equal(a, resumed.next_batch()) for a in expected)
    assert first[0].shape == (3, 16)
    assert int(first[0][0, 0]) == 0  # every row stream starts with <s>
    assert cursor.consumed_tokens == 4 * 3 * 16


def test_packed_batches_roll_over_epochs_with_new_order(tokenized):
    corpus = TokenizedCorpus(tokenized)
    loader = PackedBatches(corpus, "validation", seq_len=64, batch_size=4, bos_id=0, eos_id=2)
    for _ in range(50):
        loader.next_batch()
    assert loader.epoch >= 1
    assert isinstance(loader.cursor(), DataCursor)


# -- objectives ------------------------------------------------------------------------


def test_masked_lm_is_reproducible_and_skips_specials():
    objective = MaskedLM(mask_rate=0.3, mask_id=4, vocab_size=500, special_ids=range(37), seed=7)
    ids = torch.randint(37, 500, (8, 64))
    ids[:, 0] = 0
    ids[:, -1] = 2
    a, b = objective(ids, step=5), objective(ids, step=5)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert not torch.equal(a["input_ids"], objective(ids, step=6)["input_ids"])
    selected = a["labels"] != -100
    assert not selected[:, 0].any()
    assert not selected[:, -1].any()
    rate = selected[:, 1:-1].float().mean().item()
    assert 0.2 < rate < 0.4
    assert torch.equal(a["labels"][selected], ids[selected])
    assert (a["input_ids"][selected] == 4).float().mean().item() > 0.6


# -- config ----------------------------------------------------------------------------


def test_config_identity_ignores_operational_fields(tmp_path, tokenized):
    base = _config(tmp_path, tokenized)
    tweaked = _config(tmp_path, tokenized, eval_every_steps=1, checkpoint_every_minutes=5)
    changed = _config(tmp_path, tokenized, learning_rate=1e-3)
    assert base.identity_sha256() == tweaked.identity_sha256()
    assert base.identity_sha256() != changed.identity_sha256()
    assert base.total_steps == 8
    with pytest.raises(TypeError):
        PretrainConfig.from_mapping({**base.to_dict(), "learning_rat": 1})


def test_trainer_rejects_mismatched_objective(tmp_path, tokenized):
    config = _config(tmp_path, tokenized, objective="mlm")
    with pytest.raises(ValueError, match="trained with 'clm'"):
        Pretrainer(config, LocalObjectStore(tmp_path / "s"), tmp_path / "w")


# -- training --------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["qwen3", "modernbert"])
def test_training_completes_and_marks_run(tmp_path, tokenized, architecture):
    store = LocalObjectStore(tmp_path / "store")
    config = _config(tmp_path, tokenized, architecture=architecture)
    result = Pretrainer(config, store, tmp_path / "work", log=lambda r: None).run()
    assert result.status == "completed"
    assert result.final_step == 8
    assert result.published_steps == [8]
    assert set(result.evaluations) == {4, 8}
    assert store.read_bytes("runs/test-run/COMPLETE.json") is not None
    metrics = store.read_bytes("runs/test-run/checkpoints/step-000000008/metrics.jsonl")
    assert metrics is not None
    assert metrics.count(b"\n") == 8
    again = Pretrainer(config, store, tmp_path / "work2", log=lambda r: None).run()
    assert again.status == "already_complete"


def test_strict_resume_after_crash_matches_uninterrupted(tmp_path, tokenized):
    config = _config(tmp_path, tokenized, checkpoint_every_steps=3)
    reference = Pretrainer(config, LocalObjectStore(tmp_path / "a"), tmp_path / "wa").run()

    store = LocalObjectStore(tmp_path / "b")
    crashed = Pretrainer(config, store, tmp_path / "wb1", log=lambda r: None).run(
        stop_after_steps=5
    )
    assert crashed.status == "interrupted"
    assert crashed.published_steps == [3]
    resumed = Pretrainer(config, store, tmp_path / "wb2", log=lambda r: None).run()
    assert resumed.resumed_from_step == 3
    assert resumed.status == "completed"
    assert resumed.losses == {k: reference.losses[k] for k in resumed.losses}
    assert resumed.evaluations[8] == reference.evaluations[8]


def test_soft_deadline_pauses_with_checkpoint(tmp_path, tokenized):
    now = [0.0]

    def advance(record):
        if record["step"] == 2:
            now[0] = 11 * 3600.0  # past the 10.5 h soft limit

    store = LocalObjectStore(tmp_path / "store")
    config = _config(tmp_path, tokenized)
    clock = SessionClock(clock=lambda: now[0])
    paused = Pretrainer(config, store, tmp_path / "w1", clock=clock, log=advance).run()
    assert paused.status == "paused"
    assert paused.published_steps == [2]
    finished = Pretrainer(config, store, tmp_path / "w2", log=lambda r: None).run()
    assert finished.resumed_from_step == 2
    assert finished.status == "completed"
