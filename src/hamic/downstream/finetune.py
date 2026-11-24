"""Fine-tune an exported HAMIC encoder on the downstream suite.

Protocol (HuITBERT's, so results compare): AdamW, linear schedule with 10% warmup, weight
decay 0.01, gradient clipping 1.0, batch 32, max length 512, up to 10 epochs with early
stopping (patience 2) on the dev metric; the learning rate is chosen per task on dev from a
small grid, and test scores are reported as mean and standard deviation over seeds.

Raw syllable text goes in: token-level datasets are joined with spaces and labels are put
on the first sub-token of every syllable, so no word segmenter is involved.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hamic.downstream.metrics import (
    classification_scores,
    entity_f1,
    qa_scores,
    token_weighted_f1,
)
from hamic.downstream.tasks import TaskData

__all__ = ["FinetuneConfig", "TaskResult", "build_tokenizer", "run_task", "save_for_finetuning"]

IGNORE = -100


@dataclass(frozen=True)
class FinetuneConfig:
    """Fine-tuning settings shared by every task."""

    learning_rates: tuple[float, ...] = (2e-5, 5e-5)
    seeds: tuple[int, ...] = (10, 42, 100)
    batch_size: int = 32
    max_length: int = 512
    max_epochs: int = 10
    patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    qa_stride: int = 128
    qa_max_answer_tokens: int = 64
    device: str = "cuda"


@dataclass
class TaskResult:
    """Scores of one task: per seed, and the mean/std of the headline metric."""

    task: str
    metric: str
    learning_rate: float
    dev_by_rate: dict[float, float]
    test_by_seed: dict[int, dict[str, float]] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def mean(self) -> float:
        """Mean headline test score over seeds."""
        return round(statistics.fmean(r[self.metric] for r in self.test_by_seed.values()), 2)

    @property
    def std(self) -> float:
        """Standard deviation of the headline test score over seeds."""
        values = [r[self.metric] for r in self.test_by_seed.values()]
        return round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "task": self.task,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "dev_by_rate": {str(k): v for k, v in self.dev_by_rate.items()},
            "test_by_seed": {str(k): v for k, v in self.test_by_seed.items()},
            "mean": self.mean,
            "std": self.std,
            "seconds": round(self.seconds, 1),
        }


def build_tokenizer(tokenizer_file: str | Path, max_length: int = 512) -> Any:
    """A Transformers fast tokenizer around HAMIC's ``tokenizer.json`` (encoder template)."""
    from transformers import AddedToken, PreTrainedTokenizerFast

    from hamic.tokenization.spec import SpecialTokens

    st = SpecialTokens()
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_file=str(tokenizer_file),
        bos_token=st.bos,
        eos_token=st.eos,
        unk_token=st.unk,
        pad_token=st.pad,
        cls_token=st.bos,
        sep_token=st.eos,
        mask_token=AddedToken(st.mask, lstrip=True, rstrip=False, special=True),
        model_max_length=max_length,
        clean_up_tokenization_spaces=False,
    )


def save_for_finetuning(model: Any, output_dir: str | Path) -> Path:
    """Save a pretrained (masked-LM) model so task heads can load its encoder.

    Training-only switches are reset (sparse prediction). Task models built with
    ``from_pretrained`` reuse the encoder weights and initialize their own head.
    """
    out = Path(output_dir)
    if getattr(model.config, "sparse_prediction", False):
        model.config.sparse_prediction = False
    model.save_pretrained(out, safe_serialization=True)
    return out


# -- encoding ---------------------------------------------------------------------------


def _encode_classification(
    rows: Sequence[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[dict[str, Any]]:
    texts = [r["text"] for r in rows]
    pairs = [r["text_b"] for r in rows] if rows and "text_b" in rows[0] else None
    encoded = tokenizer(texts, pairs, truncation=True, max_length=max_length)
    return [
        {"input_ids": ids, "labels": row["label"]}
        for ids, row in zip(encoded["input_ids"], rows, strict=True)
    ]


def _encode_tokens(
    rows: Sequence[dict[str, Any]], tokenizer: Any, labels: list[str], max_length: int
) -> list[dict[str, Any]]:
    """Join syllables with spaces; label the first sub-token of each syllable."""
    index = {label: i for i, label in enumerate(labels)}
    features = []
    for row in rows:
        starts, text = [], ""
        for i, token in enumerate(row["tokens"]):
            if i:
                text += " "
            starts.append(len(text))
            text += token
        enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        word_at = {s: i for i, s in enumerate(starts)}
        label_ids, first_token = [], {}
        for position, (start, end) in enumerate(enc["offset_mapping"]):
            # Byte-level tokens may carry the leading space in their offsets.
            while start < end and text[start] == " ":
                start += 1
            word = word_at.get(start) if end > start else None
            if word is not None and word not in first_token:
                first_token[word] = position
                label_ids.append(index.get(row["tags"][word], IGNORE))
            else:
                label_ids.append(IGNORE)
        features.append(
            {"input_ids": enc["input_ids"], "labels": label_ids, "first_token": first_token}
        )
    return features


def _encode_qa(
    rows: Sequence[dict[str, Any]], tokenizer: Any, max_length: int, stride: int
) -> list[dict[str, Any]]:
    """Question first, context second; long contexts become overlapping windows."""
    enc = tokenizer(
        [r["question"] for r in rows],
        [r["context"] for r in rows],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    features = []
    for i, ids in enumerate(enc["input_ids"]):
        example = enc["overflow_to_sample_mapping"][i]
        sequence_ids = enc.sequence_ids(i)
        offsets = [
            off if sid == 1 else None
            for off, sid in zip(enc["offset_mapping"][i], sequence_ids, strict=True)
        ]
        answers = rows[example]["answers"]
        start_pos = end_pos = 0  # position 0 (<s>) means "no answer in this window"
        if answers:
            a_start = answers[0]["start"]
            a_end = a_start + len(answers[0]["text"])
            spans = {j: off for j, off in enumerate(offsets) if off is not None}
            ctx = sorted(spans)
            if ctx and spans[ctx[0]][0] <= a_start and spans[ctx[-1]][1] >= a_end:
                start_pos = next(j for j in ctx if spans[j][1] > a_start)
                end_pos = next(j for j in reversed(ctx) if spans[j][0] < a_end)
        features.append(
            {
                "input_ids": ids,
                "start_positions": start_pos,
                "end_positions": end_pos,
                "example": example,
                "offsets": offsets,
            }
        )
    return features


# -- batching ---------------------------------------------------------------------------


def _batches(
    features: Sequence[dict[str, Any]], batch_size: int, pad_id: int, shuffle: bool, seed: int
) -> list[dict[str, torch.Tensor]]:
    order = list(range(len(features)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        chunk = [features[i] for i in order[start : start + batch_size]]
        length = max(len(f["input_ids"]) for f in chunk)
        ids = torch.full((len(chunk), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), length), dtype=torch.long)
        for row, f in enumerate(chunk):
            ids[row, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
            mask[row, : len(f["input_ids"])] = 1
        batch: dict[str, torch.Tensor] = {"input_ids": ids, "attention_mask": mask}
        if isinstance(chunk[0].get("labels"), list):
            labels = torch.full((len(chunk), length), IGNORE, dtype=torch.long)
            for row, f in enumerate(chunk):
                labels[row, : len(f["labels"])] = torch.tensor(f["labels"])
            batch["labels"] = labels
        elif "labels" in chunk[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in chunk])
        for key in ("start_positions", "end_positions"):
            if key in chunk[0]:
                batch[key] = torch.tensor([f[key] for f in chunk])
        batch["index"] = torch.tensor(order[start : start + batch_size])
        batches.append(batch)
    return batches


# -- training ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train(
    model: Any,
    train: Sequence[dict[str, Any]],
    pad_id: int,
    evaluate: Callable[[Any], float],
    lr: float,
    seed: int,
    cfg: FinetuneConfig,
) -> tuple[Any, float]:
    """Train with early stopping on ``evaluate`` (higher is better); return the best model."""
    device = torch.device(cfg.device)
    model.to(device)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = steps_per_epoch * cfg.max_epochs
    warmup = max(1, round(total * cfg.warmup_ratio))
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / (total - warmup)),
    )
    autocast = device.type == "cuda"
    best_score, best_state, waited = -math.inf, None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in _batches(train, cfg.batch_size, pad_id, True, seed * 1000 + epoch):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "index"}
            with torch.autocast(device.type, torch.bfloat16, enabled=autocast):
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        score = evaluate(model)
        if score > best_score:
            best_score, waited = score, 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


@torch.no_grad()
def _forward(
    model: Any, features: Sequence[dict[str, Any]], pad_id: int, cfg: FinetuneConfig
) -> Any:
    device = torch.device(cfg.device)
    model.eval()
    outputs = []
    for batch in _batches(features, cfg.batch_size * 2, pad_id, False, 0):
        inputs = {
            k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}
        }
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            out = model(**inputs)
        outputs.append((batch, out))
    return outputs


def _predict_classes(
    model: Any, feats: Sequence[dict[str, Any]], pad: int, cfg: FinetuneConfig
) -> list[int]:
    predictions: list[int] = []
    for _, out in _forward(model, feats, pad, cfg):
        predictions.extend(out.logits.float().argmax(-1).tolist())
    return predictions


def _predict_tags(
    model: Any, feats: Sequence[dict[str, Any]], labels: list[str], pad: int, cfg: FinetuneConfig
) -> list[list[str]]:
    rows: list[list[str]] = []
    for batch, out in _forward(model, feats, pad, cfg):
        best = out.logits.float().argmax(-1).cpu()
        for row, index in enumerate(batch["index"].tolist()):
            first = feats[index]["first_token"]
            words = max(first) + 1 if first else 0
            rows.append(
                [labels[int(best[row, first[w]])] if w in first else "O" for w in range(words)]
            )
    return rows


def _predict_answers(
    model: Any,
    feats: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    pad: int,
    cfg: FinetuneConfig,
) -> list[str]:
    best: dict[int, tuple[float, str]] = {}
    null: dict[int, float] = {}
    for batch, out in _forward(model, feats, pad, cfg):
        start_logits = out.start_logits.float().cpu()
        end_logits = out.end_logits.float().cpu()
        for row, index in enumerate(batch["index"].tolist()):
            f = feats[index]
            example = f["example"]
            s, e = start_logits[row], end_logits[row]
            null[example] = min(null.get(example, math.inf), float(s[0] + e[0]))
            context = [j for j, off in enumerate(f["offsets"]) if off is not None]
            if not context:
                continue
            top_s = [j for j in torch.argsort(s, descending=True).tolist() if j in set(context)][
                :20
            ]
            top_e = [j for j in torch.argsort(e, descending=True).tolist() if j in set(context)][
                :20
            ]
            for i in top_s:
                for j in top_e:
                    if j < i or j - i + 1 > cfg.qa_max_answer_tokens:
                        continue
                    score = float(s[i] + e[j])
                    if score > best.get(example, (-math.inf, ""))[0]:
                        text = rows[example]["context"][f["offsets"][i][0] : f["offsets"][j][1]]
                        best[example] = (score, text)
    answers = []
    for example in range(len(rows)):
        span_score, text = best.get(example, (-math.inf, ""))
        answers.append(text if span_score > null.get(example, math.inf) else "")
    return answers


def _prepared(data: TaskData) -> TaskData:
    """``data`` after :func:`hamic.tokenization.prepare_text`, as in pretraining.

    Every text field is prepared; lengths are unchanged, so answer spans stay valid.
    """
    from hamic.tokenization import prepare_text

    def fold(value: Any) -> Any:
        if isinstance(value, str):
            return prepare_text(value)
        if isinstance(value, list):
            return [fold(v) for v in value]
        if isinstance(value, dict):
            return {k: (v if k in ("label", "start", "id") else fold(v)) for k, v in value.items()}
        return value

    splits = {role: [fold(row) for row in rows] for role, rows in data.splits.items()}
    return TaskData(data.spec, splits, data.labels)


# -- one task ---------------------------------------------------------------------------


def run_task(
    model_dir: str | Path,
    data: TaskData,
    tokenizer: Any,
    cfg: FinetuneConfig | None = None,
    log: Callable[[str], None] = print,
) -> TaskResult:
    """Choose the learning rate on dev (first seed), then report test over every seed."""
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    cfg = cfg or FinetuneConfig()
    data = _prepared(data)
    spec = data.spec
    pad = int(tokenizer.pad_token_id)
    started = time.monotonic()
    if spec.kind in {"sequence", "pair"}:
        enc = {
            r: _encode_classification(rows, tokenizer, cfg.max_length)
            for r, rows in data.splits.items()
        }

        def build() -> Any:
            return AutoModelForSequenceClassification.from_pretrained(
                model_dir, num_labels=len(data.labels)
            )

        def scores(model: Any, role: str) -> dict[str, float]:
            gold = [f["labels"] for f in enc[role]]
            return classification_scores(_predict_classes(model, enc[role], pad, cfg), gold)

    elif spec.kind == "token":
        enc = {
            r: _encode_tokens(rows, tokenizer, data.labels, cfg.max_length)
            for r, rows in data.splits.items()
        }

        def build() -> Any:
            return AutoModelForTokenClassification.from_pretrained(
                model_dir, num_labels=len(data.labels)
            )

        def scores(model: Any, role: str) -> dict[str, float]:
            predicted = _predict_tags(model, enc[role], data.labels, pad, cfg)
            gold = [
                row["tags"][: len(p)] for row, p in zip(data.splits[role], predicted, strict=True)
            ]
            return {
                "entity_f1": entity_f1(predicted, gold),
                "weighted_f1": token_weighted_f1(predicted, gold),
            }

    elif spec.kind == "qa":
        enc = {
            r: _encode_qa(rows, tokenizer, cfg.max_length, cfg.qa_stride)
            for r, rows in data.splits.items()
        }

        def build() -> Any:
            return AutoModelForQuestionAnswering.from_pretrained(model_dir)

        def scores(model: Any, role: str) -> dict[str, float]:
            predicted = _predict_answers(model, enc[role], data.splits[role], pad, cfg)
            gold = [[a["text"] for a in row["answers"]] for row in data.splits[role]]
            return qa_scores(predicted, gold)

    else:
        raise ValueError(f"Unknown task kind: {spec.kind}")

    metric = spec.metric
    dev_by_rate: dict[float, float] = {}
    for lr in cfg.learning_rates:
        _set_seed(cfg.seeds[0])
        _, dev = _train(
            build(), enc["train"], pad, lambda m: scores(m, "dev")[metric], lr, cfg.seeds[0], cfg
        )
        dev_by_rate[lr] = dev
        log(f"{spec.name}: lr {lr:g} dev {metric} {dev:.2f}")
    best_lr = max(dev_by_rate, key=lambda k: dev_by_rate[k])
    result = TaskResult(spec.name, metric, best_lr, dev_by_rate)
    for seed in cfg.seeds:
        _set_seed(seed)
        model, _ = _train(
            build(), enc["train"], pad, lambda m: scores(m, "dev")[metric], best_lr, seed, cfg
        )
        result.test_by_seed[seed] = scores(model, "test")
        log(f"{spec.name}: seed {seed} test {result.test_by_seed[seed]}")
        del model
    result.seconds = time.monotonic() - started
    return result
