"""Downstream suite: metrics, and fine-tuning of every task kind on tiny synthetic data."""

import warnings

import pytest

from hamic.data import TokenizedCorpus
from hamic.downstream import (
    TASKS,
    FinetuneConfig,
    TaskData,
    TaskSpec,
    build_tokenizer,
    run_task,
    save_for_finetuning,
)
from hamic.downstream.metrics import classification_scores, entity_f1, qa_scores
from hamic.modeling import ModelPreset
from hamic.modeling.factory import build_model
from tests.test_training import _config, tokenized  # noqa: F401 (pytest fixture)

# -- metrics --------------------------------------------------------------------------


def test_classification_scores():
    scores = classification_scores([0, 1, 1, 2], [0, 1, 2, 2])
    assert scores["accuracy"] == 75.0
    assert scores["macro_f1"] == pytest.approx(77.78, abs=0.01)
    assert scores["weighted_f1"] == pytest.approx(75.0, abs=0.01)


def test_entity_f1_counts_whole_spans():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    assert entity_f1(gold, gold) == 100.0
    half = [["B-PER", "O", "O", "B-LOC"]]  # truncated PER span is wrong, LOC is right
    assert entity_f1(half, gold) == 50.0


def test_qa_scores_handle_unanswerable_questions():
    scores = qa_scores(["Hà Nội", "", "thủ đô"], [["Hà Nội."], [], ["thủ đô Hà Nội"]])
    assert scores["exact_match"] == pytest.approx(66.67, abs=0.01)
    assert scores["f1"] == pytest.approx(88.89, abs=0.01)


def test_registry_is_pinned_and_complete():
    assert len(TASKS) == 9
    for spec in TASKS.values():
        assert len(spec.revision) == 40
        assert spec.kind in {"sequence", "pair", "token", "qa"}
        assert "train" in spec.splits
        assert "test" in spec.splits


# -- fine-tuning ----------------------------------------------------------------------


@pytest.fixture
def model_dir(tmp_path, tokenized):  # noqa: F811
    config = _config(tmp_path, tokenized, architecture="modernbert")
    model = build_model(ModelPreset.from_yaml(config.model_preset), seed=0)
    return save_for_finetuning(model, tmp_path / "model")


def _cfg() -> FinetuneConfig:
    return FinetuneConfig(
        learning_rates=(1e-3,),
        seeds=(1,),
        batch_size=4,
        max_length=48,
        max_epochs=2,
        device="cpu",
        qa_stride=16,
    )


def _spec(kind: str, metric: str) -> TaskSpec:
    return TaskSpec(f"toy-{kind}", kind, "toy/dataset", "0" * 40, metric=metric)


def _run(model_dir, tokenized, data):  # noqa: F811
    tokenizer = build_tokenizer(TokenizedCorpus(tokenized).tokenizer_file(), max_length=48)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_task(model_dir, data, tokenizer, _cfg(), log=lambda _: None)


def test_sequence_and_pair_tasks_run(model_dir, tokenized):  # noqa: F811
    rows = [
        {"text": f"câu số {i} rất {'tốt' if i % 2 else 'tệ'}", "label": i % 2} for i in range(16)
    ]
    pairs = [dict(r, text_b="đây là câu thứ hai") for r in rows]
    for kind, split_rows in (("sequence", rows), ("pair", pairs)):
        data = TaskData(
            _spec(kind, "weighted_f1"),
            {"train": split_rows, "dev": split_rows[:4], "test": split_rows[:6]},
            ["tệ", "tốt"],
        )
        result = _run(model_dir, tokenized, data)
        assert set(result.test_by_seed[1]) == {"accuracy", "macro_f1", "weighted_f1"}
        assert 0 <= result.mean <= 100


def test_token_task_labels_every_syllable(model_dir, tokenized):  # noqa: F811
    rows = [
        {"tokens": ["Bệnh", "nhân", "ở", "Hà", "Nội"], "tags": ["O", "O", "O", "B-LOC", "I-LOC"]}
    ] * 8
    data = TaskData(
        _spec("token", "entity_f1"),
        {"train": rows, "dev": rows[:2], "test": rows[:2]},
        ["B-LOC", "I-LOC", "O"],
    )
    result = _run(model_dir, tokenized, data)
    assert set(result.test_by_seed[1]) == {"entity_f1", "weighted_f1"}


def test_qa_task_runs_with_unanswerable_questions(model_dir, tokenized):  # noqa: F811
    context = "Hà Nội là thủ đô của Việt Nam. Thành phố có nhiều hồ đẹp."
    answerable = {
        "id": "a",
        "context": context,
        "question": "Thủ đô của Việt Nam là gì?",
        "answers": [{"text": "Hà Nội", "start": 0}],
    }
    impossible = {"id": "b", "context": context, "question": "Dân số là bao nhiêu?", "answers": []}
    rows = [answerable, impossible] * 4
    data = TaskData(_spec("qa", "f1"), {"train": rows, "dev": rows[:2], "test": rows[:2]}, [])
    result = _run(model_dir, tokenized, data)
    assert set(result.test_by_seed[1]) == {"exact_match", "f1"}
