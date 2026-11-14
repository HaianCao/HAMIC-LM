"""Train byte-level BPE tokenizers and export them in Hugging Face format.

The exported directory loads with ``transformers.AutoTokenizer.from_pretrained`` and
needs no HAMIC code. NFC normalization is embedded in the tokenizer itself, so NFD input
(common from some Vietnamese input methods, macOS and PDF copy) is handled at inference.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.trainers import BpeTrainer

from hamic.provenance import utc_now
from hamic.tokenization.segmentation import Segmenter
from hamic.tokenization.spec import DigitMode, InputMode, TokenizerSpec

__all__ = [
    "TokenizerKind",
    "TokenizerVocabularyError",
    "build_tokenizer",
    "export_tokenizer",
    "train_tokenizer",
]

METADATA_NAME = "hamic_tokenizer.json"


class TokenizerKind(str, Enum):
    """Which model family the exported tokenizer serves."""

    ENCODER = "encoder"
    """``<s> A </s>`` / ``<s> A </s></s> B </s>`` (RoBERTa convention)."""
    DECODER = "decoder"
    """``<s> A`` with no end token appended automatically."""


class TokenizerVocabularyError(RuntimeError):
    """The trained vocabulary does not match the requested size."""


def build_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    """Create an untrained tokenizer pipeline for ``spec``."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    steps: list[Any] = []
    if spec.digit_mode is DigitMode.INDIVIDUAL:
        steps.append(pre_tokenizers.Digits(individual_digits=True))
    steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    return tokenizer


def _prepared(texts: Iterable[str], segmenter: Segmenter | None) -> Iterator[str]:
    for text in texts:
        yield segmenter.segment(text) if segmenter is not None else text


def train_tokenizer(
    spec: TokenizerSpec,
    texts: Iterable[str],
    segmenter: Segmenter | None = None,
    strict_vocab_size: bool = True,
    show_progress: bool = False,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on ``texts``.

    Args:
        spec: Tokenizer definition.
        texts: Training texts (train split only; never validation/test/benchmarks).
        segmenter: Required when ``spec.input_mode`` is ``SEGMENTED``.
        strict_vocab_size: Raise if the corpus is too small to reach ``spec.vocab_size``.
        show_progress: Show the ``tokenizers`` progress bar.

    Raises:
        ValueError: If the segmenter does not match the input mode.
        TokenizerVocabularyError: If ``strict_vocab_size`` and the size is not reached.
    """
    if (spec.input_mode is InputMode.SEGMENTED) != (segmenter is not None):
        raise ValueError("Pass a segmenter if and only if input_mode is 'segmented'")
    tokenizer = build_tokenizer(spec)
    trainer = BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        special_tokens=spec.special_tokens.ordered,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(_prepared(texts, segmenter), trainer=trainer)
    actual = tokenizer.get_vocab_size()
    if strict_vocab_size and actual != spec.vocab_size:
        raise TokenizerVocabularyError(
            f"Trained vocabulary has {actual} entries, expected {spec.vocab_size}; "
            "the training corpus is probably too small"
        )
    for expected_id, token in enumerate(spec.special_tokens.ordered):
        if tokenizer.token_to_id(token) != expected_id:
            raise TokenizerVocabularyError(f"Special token {token!r} is not at id {expected_id}")
    return tokenizer


def _post_processor(spec: TokenizerSpec, kind: TokenizerKind, tokenizer: Tokenizer) -> Any:
    st = spec.special_tokens
    ids = {t: tokenizer.token_to_id(t) for t in (st.bos, st.eos)}
    if kind is TokenizerKind.ENCODER:
        template = processors.TemplateProcessing(
            single=f"{st.bos} $A {st.eos}",
            pair=f"{st.bos} $A {st.eos} {st.eos} $B {st.eos}",
            special_tokens=list(ids.items()),
        )
    else:
        template = processors.TemplateProcessing(
            single=f"{st.bos} $A",
            pair=f"{st.bos} $A $B",
            special_tokens=[(st.bos, ids[st.bos])],
        )
    return processors.Sequence([processors.ByteLevel(trim_offsets=True), template])


def _with_mask_lstrip(tokenizer: Tokenizer, mask: str) -> Tokenizer:
    """Return a copy whose mask token absorbs the preceding space (RoBERTa behaviour).

    With ``lstrip`` the input "Hà <mask>" leaves " Nội" to be predicted as one token,
    matching how the word appears in running text.
    """
    data = json.loads(tokenizer.to_str())
    for token in data.get("added_tokens", []):
        if token["content"] == mask:
            token["lstrip"] = True
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


def export_tokenizer(
    tokenizer: Tokenizer,
    spec: TokenizerSpec,
    output_dir: str | Path,
    kind: TokenizerKind,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Save ``tokenizer`` so that ``AutoTokenizer.from_pretrained(output_dir)`` works.

    Args:
        tokenizer: A trained tokenizer.
        spec: The spec it was trained with.
        output_dir: Target directory.
        kind: Encoder or decoder special-token template.
        provenance: Extra metadata (corpus manifest hash, Git commit...).

    Returns:
        The output directory.
    """
    from transformers import PreTrainedTokenizerFast

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = spec.special_tokens
    configured = _with_mask_lstrip(tokenizer, st.mask)
    configured.post_processor = _post_processor(spec, kind, configured)
    fast = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=configured,
        bos_token=st.bos,
        eos_token=st.eos,
        unk_token=st.unk,
        pad_token=st.pad,
        mask_token=AddedToken(st.mask, lstrip=True, rstrip=False, special=True),
        cls_token=st.bos if kind is TokenizerKind.ENCODER else None,
        sep_token=st.eos if kind is TokenizerKind.ENCODER else None,
        additional_special_tokens=st.sentinels,
        model_max_length=spec.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(out)
    metadata = {
        "spec": spec.to_dict(),
        "kind": kind.value,
        "requires_word_segmentation": spec.input_mode is InputMode.SEGMENTED,
        "exported_at_utc": utc_now(),
        "provenance": provenance or {},
    }
    (out / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
