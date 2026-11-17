"""Side-by-side comparison of tokenizer acceptance reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["DEFAULT_HIDDEN_SIZES", "PROBE_COLUMNS", "comparison_table"]

PROBE_COLUMNS = (
    "person_names",
    "place_names",
    "multi_syllable_terms",
    "no_diacritics",
    "teencode",
    "code_switch",
    "numbers_money_dates",
    "emoji",
)
DEFAULT_HIDDEN_SIZES: Mapping[str, int] = {
    "ES": 768,
    "EM": 1024,
    "D500M": 1280,
    "D1B": 2048,
}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def comparison_table(
    reports: Sequence[Mapping[str, Any]],
    hidden_sizes: Mapping[str, int] = DEFAULT_HIDDEN_SIZES,
) -> str:
    """Return Markdown tables comparing acceptance reports.

    The first table covers held-out corpus metrics and embedding cost; the second shows
    tokens per syllable on each Vietnamese probe category (lower means less splitting).

    Args:
        reports: Parsed ``AcceptanceReport.to_dict()`` outputs.
        hidden_sizes: Model widths used to show the embedding parameter cost.
    """
    widths = list(hidden_sizes.items())
    head = (
        "| Tokenizer | Vocab | Hard gates | Tokens/syllable | Chars/token | Bytes/token "
        "| Tokens/doc p50 | p95 | Partial-byte tokens | "
        + " | ".join(f"Embedding {name} (M)" for name, _ in widths)
        + " |"
    )
    sep = "|" + "---|" * (9 + len(widths))
    rows = [head, sep]
    for report in reports:
        corpus = report["corpus"]
        vocab = int(report["vocab_size"])
        partial = corpus["partial_byte_tokens"] / corpus["tokens"] if corpus["tokens"] else 0.0
        cells = [
            str(report["tokenizer_name"]),
            f"{vocab:,}",
            "pass" if report["passed_hard_gates"] else "FAIL",
            _fmt(corpus["tokens_per_syllable"]),
            _fmt(corpus["chars_per_token"], 2),
            _fmt(corpus["bytes_per_token"], 2),
            f"{corpus['tokens_per_doc_p50']:.0f}",
            f"{corpus['tokens_per_doc_p95']:.0f}",
            f"{partial:.2%}",
            *(f"{vocab * width / 1e6:.1f}" for _, width in widths),
        ]
        rows.append("| " + " | ".join(cells) + " |")

    probe_head = "| Tokenizer | " + " | ".join(PROBE_COLUMNS) + " |"
    probe_rows = [probe_head, "|" + "---|" * (1 + len(PROBE_COLUMNS))]
    for report in reports:
        probes = report.get("probes", {})
        values = [
            _fmt(probes[c]["tokens_per_syllable"], 2) if c in probes else "-"
            for c in PROBE_COLUMNS
        ]
        probe_rows.append(f"| {report['tokenizer_name']} | " + " | ".join(values) + " |")

    return "\n".join(rows) + "\n\nTokens per syllable on probes:\n\n" + "\n".join(probe_rows)
