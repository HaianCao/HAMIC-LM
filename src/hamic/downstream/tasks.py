"""The HAMIC Vietnamese downstream suite: nine tasks on eight datasets.

The task set follows HuITBERT's benchmark (so results can be compared with it), with
two documented deviations:

* VLSP-2013 POS is not publicly available; POS uses Universal Dependencies Vietnamese-VTB.
* UIT-ViQuAD2.0's public test answers are hidden; its validation split is the test set and
  10% of training articles (grouped by title, so passages never leak) form the dev set.
  VToC has no dev split; 10% of its training rows (seeded) form the dev set.

Every dataset is pinned to a Hub revision. All of them are used for research evaluation only;
licences are recorded per task.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TASKS", "TaskData", "TaskSpec", "load_task"]


@dataclass(frozen=True)
class TaskSpec:
    """One downstream task.

    Attributes:
        name: Task id.
        kind: ``"sequence"`` (one text), ``"pair"`` (two texts), ``"token"`` (tags per
            syllable) or ``"qa"`` (extractive QA with unanswerable questions).
        dataset: Hugging Face dataset id.
        revision: Pinned dataset commit.
        config: Dataset configuration.
        splits: Split name per role (``train``, ``dev``, ``test``); a missing ``dev`` is
            carved from ``train``.
        fields: Column names: ``text`` (and ``text_b``) for classification, ``tokens``
            for tagging, ``label``.
        metric: Headline metric: ``weighted_f1``, ``entity_f1`` or ``f1`` (QA).
        licence: Licence as published (``None``: not stated).
        note: Deviations from the reference benchmark.
        parquet: Split name -> parquet file path at ``revision``, for datasets whose
            loading script the ``datasets`` library no longer runs (read from the Hub's
            ``refs/convert/parquet`` commit instead).
    """

    name: str
    kind: str
    dataset: str
    revision: str
    config: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    metric: str = "weighted_f1"
    licence: str | None = None
    note: str = ""
    parquet: dict[str, str] = field(default_factory=dict)


_VSFC_PARQUET = {
    "train": "default/train/0000.parquet",
    "validation": "default/validation/0000.parquet",
    "test": "default/test/0000.parquet",
}

TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in [
        TaskSpec(
            "phoner",
            "token",
            "hungphongtrn/PhoNer_Covid19",
            "7803a7d58c2d31a90a14e3776811f176a2e5878a",
            splits={"train": "train", "dev": "validation", "test": "test"},
            fields={"tokens": "words", "label": "tags"},
            metric="entity_f1",
            note="Syllable-level PhoNER_COVID19 (community copy); test has 2000 sentences.",
        ),
        TaskSpec(
            "vsmec",
            "sequence",
            "tridm/UIT-VSMEC",
            "6feda28cba29ea564c76a75b2c77f9af1fbc56e6",
            splits={"train": "train", "dev": "validation", "test": "test"},
            fields={"text": "Sentence", "label": "Emotion"},
        ),
        TaskSpec(
            "vihsd",
            "sequence",
            "ura-hcmut/UIT-ViHSD",
            "16c4f67cf509d4f9f36ca5b63c5503c7c8830557",
            splits={"train": "train", "dev": "validation", "test": "test"},
            fields={"text": "free_text", "label": "label_id"},
            licence="cc-by-nc-sa-4.0",
        ),
        TaskSpec(
            "vinli",
            "pair",
            "presencesw/vinli_4_label",
            "a885fd1bdeb00099f2215b601ddb0df255503cf1",
            splits={"train": "train", "dev": "dev", "test": "test"},
            fields={"text": "sentence1", "text_b": "sentence2", "label": "gold_label"},
        ),
        TaskSpec(
            "viquad2",
            "qa",
            "taidng/UIT-ViQuAD2.0",
            "406f09a45cc106a8f7b7fd0c25078883fe58cb1f",
            splits={"train": "train", "test": "validation"},
            metric="f1",
            note="Public test answers are hidden: validation is the test set; dev is 10% of "
            "training titles.",
        ),
        TaskSpec(
            "pos",
            "token",
            "universal-dependencies/universal_dependencies",
            "c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703",
            config="vi_vtb",
            splits={"train": "train", "dev": "dev", "test": "test"},
            fields={"tokens": "tokens", "label": "upos"},
            licence="cc-by-sa-4.0 (UD_Vietnamese-VTB)",
            note="Replaces VLSP-2013 POS, which is not publicly available.",
        ),
        TaskSpec(
            "vsfc_sentiment",
            "sequence",
            "uitnlp/vietnamese_students_feedback",
            "2d76906157232f87e9883fa37ebfc91ffdee1f83",  # refs/convert/parquet
            splits={"train": "train", "dev": "validation", "test": "test"},
            fields={"text": "sentence", "label": "sentiment"},
            licence="unknown",
            parquet=_VSFC_PARQUET,
            note="Official copy; its loading script is unsupported, so the Hub's parquet "
            "conversion is read.",
        ),
        TaskSpec(
            "vsfc_topic",
            "sequence",
            "uitnlp/vietnamese_students_feedback",
            "2d76906157232f87e9883fa37ebfc91ffdee1f83",  # refs/convert/parquet
            splits={"train": "train", "dev": "validation", "test": "test"},
            fields={"text": "sentence", "label": "topic"},
            licence="unknown",
            parquet=_VSFC_PARQUET,
            note="Official copy; its loading script is unsupported, so the Hub's parquet "
            "conversion is read.",
        ),
        TaskSpec(
            "vtoc",
            "sequence",
            "tmnam20/ViGLUE",
            "ad7df00441b9e8666f107b12f6b3d3b3e16cb470",
            config="vtoc",
            splits={"train": "train", "test": "validation"},
            fields={"text": "sentence", "label": "label"},
            licence="cc-by-nc-nd-4.0 (VToC)",
            note="No dev split: 10% of training rows (seeded) form the dev set.",
        ),
    ]
}


@dataclass
class TaskData:
    """Rows of one task, normalized per kind.

    ``sequence``/``pair`` rows: ``{"text", "text_b"?, "label": int}``; ``token`` rows:
    ``{"tokens": [str], "tags": [str]}``; ``qa`` rows: ``{"id", "context", "question",
    "answers": [{"text", "start"}]}`` (empty for unanswerable questions).
    """

    spec: TaskSpec
    splits: dict[str, list[dict[str, Any]]]
    labels: list[str]


def _bucket(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _carve_dev(
    rows: list[dict[str, Any]], key: str, share: float = 0.1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``rows`` deterministically by a hash of ``row[key]`` (or the row index)."""
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        value = str(row.get(key, i)) if key else str(i)
        (dev if _bucket(value, "hamic-dev-v1") < share else train).append(row)
    return train, dev


def _parse(value: Any) -> Any:
    """Columns stored as Python-literal strings by some community copies."""
    if isinstance(value, str) and value[:1] in "[{":
        return ast.literal_eval(value)
    return value


def load_task(spec: TaskSpec, cache_dir: str | None = None) -> TaskData:
    """Download (pinned) and normalize every split of ``spec``."""
    raw = {role: _load_split(spec, split, cache_dir) for role, split in spec.splits.items()}
    if spec.kind == "qa":
        splits = {role: [_qa_row(r) for r in rows] for role, rows in raw.items()}
        if "dev" not in splits:
            titles = {
                r["id"]: t for r, t in zip(splits["train"], raw["train"]["title"], strict=True)
            }
            for row in splits["train"]:
                row["title"] = titles[row["id"]]
            splits["train"], splits["dev"] = _carve_dev(splits["train"], "title")
        return TaskData(spec, splits, [])

    label_field = spec.fields["label"]
    feature = raw["train"].features[label_field]
    names = _label_names(feature)
    splits = {}
    for role, rows in raw.items():
        if spec.kind == "token":
            splits[role] = [
                {
                    "tokens": list(_parse(r[spec.fields["tokens"]])),
                    "tags": [_name(t, names) for t in _parse(r[label_field])],
                }
                for r in rows
            ]
        else:
            splits[role] = [
                {
                    "text": str(r[spec.fields["text"]]),
                    **(
                        {"text_b": str(r[spec.fields["text_b"]])}
                        if "text_b" in spec.fields
                        else {}
                    ),
                    "label": str(_name(r[label_field], names)),
                }
                for r in rows
            ]
    if "dev" not in splits:
        splits["train"], splits["dev"] = _carve_dev(splits["train"], "")
    if spec.kind == "token":
        labels = sorted({t for row in splits["train"] for t in row["tags"]})
    else:
        labels = sorted({row["label"] for row in splits["train"]})
        index = {label: i for i, label in enumerate(labels)}
        for rows in splits.values():
            for row in rows:
                row["label"] = index.get(row["label"], -1)
    return TaskData(spec, splits, labels)


def _load_split(spec: TaskSpec, split: str, cache_dir: str | None) -> Any:
    from datasets import load_dataset

    if spec.parquet:
        url = f"hf://datasets/{spec.dataset}@{spec.revision}/{spec.parquet[split]}"
        return load_dataset("parquet", data_files={split: url}, split=split, cache_dir=cache_dir)
    return load_dataset(
        spec.dataset, spec.config, split=split, revision=spec.revision, cache_dir=cache_dir
    )


def _label_names(feature: Any) -> list[str] | None:
    inner = getattr(feature, "feature", feature)
    return list(inner.names) if hasattr(inner, "names") else None


def _name(value: Any, names: list[str] | None) -> str:
    return (
        names[int(value)] if names is not None and str(value).lstrip("-").isdigit() else str(value)
    )


def _qa_row(row: dict[str, Any]) -> dict[str, Any]:
    answers = _parse(row.get("answers")) or {}
    impossible = str(row.get("is_impossible", "False")).lower() == "true"
    texts = [] if impossible else list(answers.get("text", []))
    starts = [] if impossible else [int(s) for s in answers.get("answer_start", [])]
    return {
        "id": str(row["id"]),
        "context": str(row["context"]),
        "question": str(row["question"]),
        "answers": [{"text": t, "start": s} for t, s in zip(texts, starts, strict=True)],
    }
