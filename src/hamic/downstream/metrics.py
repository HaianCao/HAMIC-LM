"""Metrics of the downstream suite (no third-party metric libraries).

* Classification: accuracy, macro F1 and weighted F1 (HuITBERT reports weighted F1).
* Sequence labelling: entity-level micro F1 over BIO spans (seqeval convention, the usual
  NER metric) and token-level weighted F1 (what HuITBERT reports for PhoNER and POS).
* Extractive QA with unanswerable questions: SQuAD 2.0 exact match and token F1.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence

__all__ = [
    "classification_scores",
    "entity_f1",
    "qa_scores",
    "token_weighted_f1",
]


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def classification_scores(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    """Accuracy, macro F1 and support-weighted F1 (percentages)."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels differ in length")
    classes = sorted(set(labels) | set(predictions))
    support = Counter(labels)
    per_class = {}
    for c in classes:
        tp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == c and y == c)
        fp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == c and y != c)
        fn = sum(1 for p, y in zip(predictions, labels, strict=True) if p != c and y == c)
        per_class[c] = _f1(tp, fp, fn)
    total = len(labels) or 1
    accuracy = sum(1 for p, y in zip(predictions, labels, strict=True) if p == y) / total
    labelled = [c for c in classes if support[c]]
    macro = sum(per_class[c] for c in labelled) / (len(labelled) or 1)
    weighted = sum(per_class[c] * support[c] for c in labelled) / total
    return {
        "accuracy": round(100 * accuracy, 2),
        "macro_f1": round(100 * macro, 2),
        "weighted_f1": round(100 * weighted, 2),
    }


def token_weighted_f1(
    predictions: Sequence[Sequence[str]], labels: Sequence[Sequence[str]]
) -> float:
    """Support-weighted F1 over every token's tag (percentage)."""
    flat_p = [p for row in predictions for p in row]
    flat_y = [y for row in labels for y in row]
    names = sorted(set(flat_p) | set(flat_y))
    index = {name: i for i, name in enumerate(names)}
    scores = classification_scores([index[p] for p in flat_p], [index[y] for y in flat_y])
    return scores["weighted_f1"]


def _spans(tags: Sequence[str]) -> set[tuple[int, int, str]]:
    """BIO (or IOB2) spans ``(start, end, type)``; a stray ``I-`` opens a new span."""
    spans: set[tuple[int, int, str]] = set()
    start, kind = -1, ""
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and kind == label and start >= 0
        if start >= 0 and not continues:
            spans.add((start, i, kind))
            start, kind = -1, ""
        if prefix in {"B", "I"} and not continues:
            start, kind = i, label
    return spans


def entity_f1(predictions: Sequence[Sequence[str]], labels: Sequence[Sequence[str]]) -> float:
    """Entity-level micro F1 over BIO spans (percentage)."""
    tp = fp = fn = 0
    for pred, gold in zip(predictions, labels, strict=True):
        p, g = _spans(pred), _spans(gold)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return round(100 * _f1(tp, fp, fn), 2)


_PUNCT = re.compile(f"[{re.escape(string.punctuation)}“”‘’…–—]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def qa_scores(predictions: Sequence[str], answers: Sequence[Sequence[str]]) -> dict[str, float]:
    """SQuAD 2.0 exact match and F1 (percentages); an empty answer list means unanswerable.

    Args:
        predictions: Predicted answer text per question (``""`` predicts no answer).
        answers: Gold answer texts per question (empty for unanswerable questions).
    """
    exact = f1 = 0.0
    for pred, golds in zip(predictions, answers, strict=True):
        candidates = [_normalize(g) for g in golds] or [""]
        p = _normalize(pred)
        exact += max(float(p == g) for g in candidates)
        f1 += max(_token_f1(p, g) for g in candidates)
    total = len(predictions) or 1
    return {"exact_match": round(100 * exact / total, 2), "f1": round(100 * f1 / total, 2)}


def _token_f1(prediction: str, gold: str) -> float:
    p, g = prediction.split(), gold.split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)
