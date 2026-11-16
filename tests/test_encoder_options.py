"""Encoder pretraining options of the E1–E8 proxy round.

Covers the WSD schedule, document-restricted attention, source weights, mask-rate
schedules, span/PMI selection units, Vietnamese spelling-variant corruption and sparse
prediction, plus an end-to-end ModernBERT run with all of them enabled.
"""

import itertools
import json
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch

from hamic.checkpoint import LocalObjectStore
from hamic.data import TokenizedCorpus, annotate_sources
from hamic.modeling import ModelPreset
from hamic.modeling.factory import build_model
from hamic.provenance import json_sha256
from hamic.training import MaskedLM, PackedBatches, Pretrainer
from hamic.training.attention import DocumentMask, document_ids
from hamic.training.config import _IDENTITY_FIELDS
from hamic.training.masking import (
    MaskRateSchedule,
    PmiUnits,
    SpanUnits,
    TokenUnits,
    VariantTable,
    mine_pmi_pairs,
    remove_tone,
)
from hamic.training.trainer import lr_schedule
from tests.test_training import _config, tokenized  # noqa: F401 (pytest fixture)

# -- E1: learning-rate schedule ---------------------------------------------------------


def test_wsd_schedule_warms_holds_and_decays(tmp_path, tokenized):  # noqa: F811
    config = _config(
        tmp_path,
        tokenized,
        total_tokens=16 * 2 * 2 * 100,
        warmup_ratio=0.1,
        lr_schedule="wsd",
        decay_ratio=0.2,
        min_lr_ratio=0.1,
    )
    factor = lr_schedule(config)
    assert config.total_steps == 100
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == pytest.approx(1.0)
    assert all(factor(s) == 1.0 for s in range(10, 80))
    decay = [factor(s) for s in range(80, 100)]
    assert all(a > b for a, b in itertools.pairwise(decay))
    assert factor(99) == pytest.approx(0.1)


def test_default_identity_matches_earlier_runs(tmp_path, tokenized):  # noqa: F811
    config = _config(tmp_path, tokenized)
    data = config.to_dict()
    assert config.identity_sha256() == json_sha256({k: data[k] for k in _IDENTITY_FIELDS})
    wsd = replace(config, lr_schedule="wsd")
    assert wsd.identity()["lr_schedule"] == "wsd"
    assert wsd.identity_sha256() != config.identity_sha256()
    sparse = replace(config, objective="mlm", sparse_prediction=True, attention="sdpa")
    assert "sparse_prediction" not in sparse.identity()


# -- E2: document-restricted attention ----------------------------------------------------


def test_document_ids_follow_bos():
    ids = torch.tensor([[7, 8, 0, 9, 9, 2, 0, 5]])
    assert document_ids(ids, bos_id=0).tolist() == [[0, 0, 1, 1, 1, 1, 2, 2]]


def test_document_mask_isolates_documents(tmp_path, tokenized):  # noqa: F811
    config = _config(tmp_path, tokenized, architecture="modernbert")
    preset = ModelPreset.from_yaml(config.model_preset)
    model = build_model(preset, seed=0, attn_implementation="sdpa").eval()
    mask = DocumentMask("sdpa", tuple(model.config.layer_types))
    first = torch.tensor([[0, 50, 60, 70, 2, 0, 80, 90, 100, 2]])
    second = first.clone()
    second[0, 6:9] = torch.tensor([110, 120, 130])  # change only the second document
    with torch.no_grad():
        a = model(input_ids=first, attention_mask=mask(document_ids(first, 0))).logits
        b = model(input_ids=second, attention_mask=mask(document_ids(second, 0))).logits
        full = model(input_ids=second).logits
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-6)
    assert not torch.allclose(a[0, :5], full[0, :5], atol=1e-4)


def test_document_mask_needs_modernbert(tmp_path, tokenized):  # noqa: F811
    config = _config(tmp_path, tokenized, architecture="roberta", cross_document_attention=False)
    with pytest.raises(ValueError, match="ModernBERT"):
        Pretrainer(config, LocalObjectStore(tmp_path / "s"), tmp_path / "w")


# -- E3: source weights -------------------------------------------------------------------


@pytest.fixture
def annotated(tmp_path, tokenized):  # noqa: F811
    target = tmp_path / "tokens"
    import shutil

    shutil.copytree(tokenized, target)
    manifest = json.loads((target / "tokenized_manifest.json").read_text(encoding="utf-8"))
    for info in manifest["splits"].values():
        for shard in info["shards"]:
            shard["files"].pop("sources", None)
    manifest.pop("source_ids", None)
    (target / "tokenized_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for path in target.rglob("*.sources.npy"):
        path.unlink()
    return target


def test_tokenize_records_sources_and_annotate_restores_them(tokenized, annotated):  # noqa: F811
    original = TokenizedCorpus(tokenized, verify=True)
    assert original.source_ids == ["s"]
    assert all((s.sources == 0).all() for s in original.shards("train"))
    stripped = TokenizedCorpus(annotated)
    assert stripped.source_ids is None
    annotate_sources(tokenized.parent / "corpus", annotated)
    restored = TokenizedCorpus(annotated, verify=True)
    assert restored.source_ids == ["s"]
    for a, b in zip(original.shards("train"), restored.shards("train"), strict=True):
        assert np.array_equal(a.sources, b.sources)
        assert np.array_equal(a.tokens, b.tokens)


def test_source_weights_repeat_documents(tokenized):  # noqa: F811
    corpus = TokenizedCorpus(tokenized)
    plain = PackedBatches(corpus, "train", 16, 2, bos_id=0, eos_id=2, seed=3)
    doubled = PackedBatches(
        corpus, "train", 16, 2, bos_id=0, eos_id=2, seed=3, source_weights={"s": 2.0}
    )
    assert doubled._order.size == 2 * plain._order.size
    assert np.array_equal(np.bincount(doubled._order), np.full(plain.documents, 2))
    with pytest.raises(ValueError, match="Unknown sources"):
        PackedBatches(corpus, "train", 16, 2, bos_id=0, eos_id=2, source_weights={"x": 2.0})


def test_source_weights_need_annotations(annotated):
    with pytest.raises(ValueError, match="annotate-sources"):
        PackedBatches(TokenizedCorpus(annotated), "train", 16, 2, 0, 2, source_weights={"s": 2.0})


# -- E4–E7: masking ----------------------------------------------------------------------


def _reference_mlm(ids, step, rate=0.3, seed=7, vocab=500, special=37, mask_id=4):
    """The token-level implementation used by every run before the E1–E8 round."""
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
    maskable = ~torch.isin(ids, torch.arange(special))
    selected = (torch.rand(ids.shape, generator=generator) < rate) & maskable
    labels = torch.where(selected, ids, torch.full_like(ids, -100))
    action = torch.rand(ids.shape, generator=generator)
    random_ids = torch.randint(special, vocab, ids.shape, generator=generator)
    masked = ids.clone()
    masked[selected & (action < 0.8)] = mask_id
    to_random = selected & (action >= 0.8) & (action < 0.9)
    masked[to_random] = random_ids[to_random]
    return masked, labels


def test_default_masking_is_unchanged():
    objective = MaskedLM(mask_rate=0.3, mask_id=4, vocab_size=500, special_ids=range(37), seed=7)
    ids = torch.randint(37, 500, (4, 64))
    for step in (0, 5, 123):
        masked, labels = _reference_mlm(ids, step)
        out = objective(ids, step)
        assert torch.equal(out["input_ids"], masked)
        assert torch.equal(out["labels"], labels)


def test_mask_rate_schedule_is_linear():
    schedule = MaskRateSchedule(0.3, 0.15, start_step=10, end_step=20)
    assert schedule(0) == schedule(10) == pytest.approx(0.3)
    assert schedule(15) == pytest.approx(0.225)
    assert schedule(20) == schedule(99) == pytest.approx(0.15)
    assert MaskRateSchedule(0.3)(1000) == pytest.approx(0.3)


def _unit_lengths(units: torch.Tensor) -> list[int]:
    lengths = []
    for row in units:
        assert (row[1:] - row[:-1]).ge(0).all()
        assert (row[1:] - row[:-1]).le(1).all()
        lengths.extend(torch.bincount(row).tolist())
    return lengths


def test_span_units_follow_spanbert_lengths():
    generator = torch.Generator().manual_seed(0)
    lengths = _unit_lengths(SpanUnits()(torch.zeros(64, 512, dtype=torch.long), generator))
    assert max(lengths) <= 5
    assert 2.4 < float(np.mean(lengths)) < 2.8
    spanbert = _unit_lengths(
        SpanUnits(max_length=10)(torch.zeros(64, 512, dtype=torch.long), generator)
    )
    assert 3.6 < float(np.mean(spanbert)) < 4.0
    assert _unit_lengths(TokenUnits()(torch.zeros(2, 8, dtype=torch.long), generator)) == [1] * 16


def test_span_masking_keeps_rate_and_masks_whole_spans():
    objective = MaskedLM(0.3, 4, 500, range(37), seed=1, units=SpanUnits())
    ids = torch.randint(37, 500, (32, 256))
    out = objective(ids, step=0)
    selected = out["labels"] != -100
    assert 0.25 < selected.float().mean().item() < 0.35
    runs = (selected[:, 1:] & selected[:, :-1]).float().mean() / selected.float().mean()
    assert runs.item() > 0.5  # neighbours of a selected token are usually selected too


def test_pmi_mining_and_units_join_collocations():
    rng = np.random.default_rng(0)
    documents = []
    for _ in range(400):
        doc = rng.integers(40, 400, 60)
        doc[10:12] = [500, 501]  # a collocation planted in every document
        documents.append(doc)
    pairs = mine_pmi_pairs(documents, 600, special_ids=range(37), min_count=20, min_pmi=2.0)
    assert 500 * 600 + 501 in pairs.tolist()
    units = PmiUnits(torch.from_numpy(pairs), 600, range(37))
    ids = torch.tensor([[0, 100, 500, 501, 200, 2]])
    row = units(ids, torch.Generator())[0].tolist()
    assert row[2] == row[3]
    assert row[1] != row[2]
    assert row[4] != row[3]


def test_variant_table_lists_vietnamese_spellings():
    tokens = ["", " má", " mà", " ma", " mạ", " Ma", " không", " ko", "Hà", "Ha", " nhà"]
    table = VariantTable.from_vocabulary(tokens)
    index = {t: i for i, t in enumerate(tokens)}

    def variants(text):
        i = index[text]
        return {tokens[j] for j in table.values[table.offsets[i] : table.offsets[i + 1]].tolist()}

    assert {" mà", " ma", " mạ"} <= variants(" má")
    assert " Ma" in variants(" ma")
    assert " ko" in variants(" không")
    assert "Ha" in variants("Hà")
    assert variants(" nhà") == set()
    assert remove_tone("mẹ") == "me"
    assert remove_tone("Hà") == "Ha"
    assert remove_tone("ơ") == "ơ"


def test_variant_corruption_replaces_with_variants_and_keeps_labels():
    tokens = ["", " má", " mà", " ma", " mạ", " nhà", " ở"] + [f" x{i}" for i in range(93)]
    table = VariantTable.from_vocabulary(tokens)
    ids = torch.ones((16, 64), dtype=torch.long)  # every position is " má"
    objective = MaskedLM(
        0.5,
        0,
        100,
        [0],
        seed=2,
        mask_share=0.8,
        random_share=0.1,
        variant_share=0.5,
        variants=table,
    )
    out = objective(ids, step=0)
    selected = out["labels"] != -100
    assert (out["labels"][selected] == 1).all()
    replaced = out["input_ids"][selected]
    variant_ids = {2, 3, 4}
    share = np.isin(replaced.numpy(), list(variant_ids)).mean()
    assert 0.4 < share < 0.6
    assert objective.random_share == 0.0
    assert objective.mask_share == pytest.approx(0.4)


def test_variant_table_from_tokenizer_file(tokenized):  # noqa: F811
    table = VariantTable.from_tokenizer_file(TokenizedCorpus(tokenized).tokenizer_file())
    assert len(table) == TokenizedCorpus(tokenized).vocab_size
    assert int(table.counts.sum()) > 0


# -- E8 and end to end --------------------------------------------------------------------


def test_sparse_prediction_gives_the_same_loss(tmp_path, tokenized):  # noqa: F811
    config = _config(tmp_path, tokenized, architecture="modernbert")
    preset = ModelPreset.from_yaml(config.model_preset)
    dense = build_model(preset, seed=0).eval()
    sparse = build_model(preset, seed=0, sparse_prediction=True).eval()
    ids = torch.randint(40, preset.vocab_size, (2, 12))
    labels = torch.full_like(ids, -100)
    labels[:, ::3] = ids[:, ::3]
    with torch.no_grad():
        assert torch.allclose(
            dense(input_ids=ids, labels=labels).loss, sparse(input_ids=ids, labels=labels).loss
        )
    with pytest.raises(ValueError, match="ModernBERT"):
        build_model(
            ModelPreset.from_yaml(_config(tmp_path, tokenized).model_preset),
            sparse_prediction=True,
        )


@pytest.mark.parametrize("masking", ["span", "pmi"])
def test_modernbert_trains_with_every_option(tmp_path, tokenized, masking):  # noqa: F811
    config = _config(
        tmp_path,
        tokenized,
        architecture="modernbert",
        mask_rate=0.3,
        eval_mask_rate=0.15,
        lr_schedule="wsd",
        cross_document_attention=False,
        source_weights={"s": 2.0},
        mask_rate_final=0.15,
        mask_schedule_start=0.5,
        masking=masking,
        variant_share=0.1,
        sparse_prediction=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Pretrainer(
            config, LocalObjectStore(tmp_path / "store"), tmp_path / "work", log=lambda r: None
        ).run()
    assert result.status == "completed"
    assert all(np.isfinite(v) for v in result.losses.values())
    assert set(result.evaluations) == {4, 8}
    if masking == "pmi":
        assert list((tmp_path / "work").glob("pmi-pairs-*.npy"))


def test_offline_evaluation_of_a_published_run(tmp_path, tokenized):  # noqa: F811
    from hamic.training.evaluate import evaluate_views, load_published_model, masking_views

    config = _config(
        tmp_path, tokenized, architecture="modernbert", mask_rate=0.3, eval_mask_rate=0.15
    )
    store = LocalObjectStore(tmp_path / "store")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Pretrainer(config, store, tmp_path / "work", log=lambda r: None).run()
    published = load_published_model(store, config.run_id)
    assert published.step == config.total_steps
    corpus = TokenizedCorpus(tokenized)
    views = masking_views(corpus, published.preset, pmi_tokens=10_000, cache_dir=tmp_path / "c")
    first = evaluate_views(published, corpus, views, sequences=4, batch_size=2)
    again = evaluate_views(published, corpus, views, sequences=4, batch_size=2)
    assert set(first) == {"token", "span", "pmi"}
    assert first == again  # same seeded masks, same inputs
    assert all(np.isfinite(v) and v > 0 for v in first.values())
