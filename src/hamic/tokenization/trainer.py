"""Train byte-level BPE tokenizers and export them in Hugging Face format.

HAMIC has one tokenizer pipeline (see :class:`~hamic.tokenization.TokenizerSpec`). The
exported directory loads with ``transformers.AutoTokenizer.from_pretrained``; everything
but :func:`prepare_text` is inside ``tokenizer.json``, including NFKC, so NFD input
(common from some Vietnamese input methods, macOS and PDF copy) is handled at inference.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.trainers import BpeTrainer

from hamic.provenance import utc_now
from hamic.resources import load_resource
from hamic.tokenization.spec import TokenizerSpec

__all__ = [
    "TokenizerKind",
    "TokenizerVocabularyError",
    "build_tokenizer",
    "export_tokenizer",
    "frequent_characters",
    "prepare_text",
    "required_characters",
    "required_strings",
    "train_tokenizer",
]

METADATA_NAME = "hamic_tokenizer.json"

NEWLINE = "\n"

ZERO_WIDTH = "[\u200b\u200c\u200d\u2060\ufeff]"
"""Characters :class:`hamic.text.normalization.VietnameseNormalizer` deletes."""

CONFUSABLES_RESOURCE = "confusables.yaml"
"""Look-alikes: ``always`` (normalizer) and ``latin_context`` (:func:`prepare_text`)."""

REQUIRED_RESOURCE = "required_tokens.yaml"
"""Strings trained to be single tokens (frequent emoji, symbols, emoticons) and the
Vietnamese letters trained to be single bare tokens."""

CHARACTER_SHARE = 1 / 3_000_000
"""A character at least this frequent in the training text becomes a single token.

Byte-level BPE otherwise learns a multi-byte character only inside longer tokens and
cuts it into UTF-8 byte pieces elsewhere. In the 2.48B-character v0.12 sample this
selects about 160 characters (CJK, Cyrillic, symbols). They and the Vietnamese letters
are learned bare only: adding the after-space form too (T1b r8) gave more byte pieces
(2,981 against 2,780 on 5,000 held-out documents) and 1.1693 tokens per syllable
against 1.1689."""

OPENERS = "\n([{“‘«"
"""Characters after which the normalizer inserts a virtual space before a letter or digit.

Straight quotes ``"`` and ``'`` both open and close; a space after a closing quote is
real (in the v0.12 held-out sample 789 of 2,483 ``"`` + letter cases had a space, against
287 of 5,069 for ``(`` and 0 of 49,726 for a newline). They count as openers only after
whitespace, where they cannot be closing quotes or apostrophes."""

# Per opener: the opener (not at the very start of the text) followed by a letter becomes
# opener + space. The inserted space is aligned with the opener, so offsets stay on the
# original characters; at text position 0 the byte-level offset trimming keeps one
# leading space (RoBERTa's prefix-space rule) and would stretch the next token over the
# opener, hence the "(?<=[\s\S])" guard. The decoder removes the space again.
_OPENER_PATTERNS = {"\n": r"\n", "(": r"\(", "[": r"\[", "{": r"\{", "“": "“", "‘": "‘", "«": "«"}
_QUOTE_PATTERNS = {'"': '"', "'": "'"}
_VIRTUAL_SPACE = r"(?<=[\n(\[{“‘«]) (?=[\p{L}\p{N}])|(?<=\s[\"']) (?=[\p{L}\p{N}])"

# Runs at the byte level (after ByteLevel), where ASCII digits keep their own characters
# and a space is "Ġ". A whole run of at most four digits stays one piece (years such as
# 1990 remain learnable as single tokens); a longer run is cut into pairs counted from
# the right: optional space, then 1-2 digits followed by an even number of digits up to
# the end of the run. (Triplets were tried first: the vocabulary learned only 83 of the
# 1,000 inner triplets, so "0912345678" became 0|9|12|3|45|6|78.)
_DIGIT_PAIRS = (
    r"Ġ?(?<![0-9])[0-9]{1,4}(?![0-9])"
    r"|Ġ?[0-9]{1,2}(?=(?:[0-9]{2})*(?![0-9]))"
)


class TokenizerKind(str, Enum):
    """Which model family the exported tokenizer serves."""

    ENCODER = "encoder"
    """``<s> A </s>`` / ``<s> A </s></s> B </s>`` (RoBERTa convention)."""
    DECODER = "decoder"
    """``<s> A`` with no end token appended automatically."""


class TokenizerVocabularyError(RuntimeError):
    """The trained vocabulary does not match the requested size."""


def build_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    """Create the untrained HAMIC tokenizer pipeline (``spec`` sets only training details)."""
    del spec  # one pipeline for every spec; kept for a uniform signature
    tokenizer = Tokenizer(models.BPE())
    rules: list[Any] = [
        # The corpus pipeline's representation cleanup (hamic.text.normalization).
        normalizers.Replace("\r\n", "\n"),
        normalizers.Replace("\r", "\n"),
        normalizers.Replace(Regex(ZERO_WIDTH), ""),
        # Before NFKC, which would turn spacing accents (´ ˜) into a space + combining mark.
        *_confusable_rules(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(r"[^\S\n]+"), " "),
        normalizers.Replace(Regex(r"[^\S\n]*\n[^\S\n]*"), "\n"),
        normalizers.Replace(Regex(r"\n{3,}"), "\n\n"),
        normalizers.Strip(),
    ]
    rules += [
        normalizers.Replace(Regex(r"(?<=[\s\S])" + pattern + r"(?=[\p{L}\p{N}])"), opener + " ")
        for opener, pattern in _OPENER_PATTERNS.items()
    ]
    rules += [
        normalizers.Replace(Regex(r"(?<=\s)" + pattern + r"(?=[\p{L}\p{N}])"), quote + " ")
        for quote, pattern in _QUOTE_PATTERNS.items()
    ]
    tokenizer.normalizer = normalizers.Sequence(rules)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True),
            pre_tokenizers.Split(Regex(_DIGIT_PAIRS), behavior="isolated"),
        ]
    )
    tokenizer.decoder = decoders.Sequence(
        [decoders.ByteLevel(), decoders.Fuse(), decoders.Replace(Regex(_VIRTUAL_SPACE), "")]
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    return tokenizer


def _confusable_rules() -> list[Any]:
    """Normalizer steps for the ``always`` map of ``resources/confusables.yaml``.

    The ``latin_context`` map is not a normalizer step (one regex pass per target letter
    would cost hours over a corpus); :func:`prepare_text` applies it.
    """
    data = load_resource(CONFUSABLES_RESOURCE).data
    return [normalizers.Replace(a, b) for a, b in data["always"].items()]


def prepare_text(text: str) -> str:
    """What HAMIC applies to every text before the tokenizer sees it.

    Folds look-alike letters next to Latin letters (:mod:`hamic.text.lookalikes`), one
    character for one, so offsets into the original text stay valid.
    """
    from hamic.text.lookalikes import fold_lookalikes

    return fold_lookalikes(text)


def required_strings(spec: TokenizerSpec) -> list[str]:
    """Strings trained to be single tokens alone and after a space (deduplicated)."""
    data = load_resource(REQUIRED_RESOURCE).data
    return _normalized(spec, (s for g in data["groups"].values() for s in g))


def required_characters(spec: TokenizerSpec) -> list[str]:
    """Vietnamese letters with a diacritic, trained to be single bare tokens."""
    return _normalized(spec, load_resource(REQUIRED_RESOURCE).data["characters"])


def _normalized(spec: TokenizerSpec, strings: Iterable[object]) -> list[str]:
    # In the form the tokenizer sees them (NFKC, variation selectors removed).
    normalizer = build_tokenizer(spec).normalizer
    return list(dict.fromkeys(s for s in (normalizer.normalize_str(str(x)) for x in strings) if s))


def _bare(string: str, count: int) -> str:
    """``string`` ``count`` times, each occurrence its own pre-token with no space.

    The separator must be its own pre-token and never insert a space: a hyphen for
    letters (the normalizer would put a virtual space after a newline), a newline for
    symbols and emoji.
    """
    separator = "-" if string[:1].isalpha() else NEWLINE
    return separator + separator.join([string] * count)


def _required_documents(spec: TokenizerSpec) -> Iterator[str]:
    """Synthetic text giving each required string ``required_count`` occurrences per form.

    Joining with spaces yields the after-space form. Vietnamese letters get the bare form
    only.
    """
    for string in required_strings(spec):
        yield " ".join([string] * spec.required_count)
        yield _bare(string, spec.required_count)
    for character in required_characters(spec):
        yield _bare(character, spec.required_count)


def frequent_characters(spec: TokenizerSpec, counts: Counter[str]) -> list[str]:
    """Multi-byte characters of ``counts`` frequent enough to need their own token.

    Characters are taken as the normalizer outputs them; combining marks (they would
    join the separator), whitespace, controls and ASCII are left out.
    """
    total = sum(counts.values())
    normalizer = build_tokenizer(spec).normalizer
    selected: dict[str, None] = {}
    for character, count in counts.most_common():
        if count < total * CHARACTER_SHARE:
            break
        normalized = normalizer.normalize_str(character)
        if len(normalized) != 1 or len(normalized.encode("utf-8")) == 1:
            continue
        if unicodedata.category(normalized)[0] in "MZC":
            continue
        selected[normalized] = None
    return list(selected)


def _prepared(texts: Iterable[str], spec: TokenizerSpec) -> Iterator[str]:
    yield from _required_documents(spec)
    counts: Counter[str] = Counter()
    for text in texts:
        text = prepare_text(text)
        counts.update(text)
        yield text
    # BPE counts pairs over the whole iterator, so these may come last.
    for character in frequent_characters(spec, counts):
        yield _bare(character, spec.required_count)


def train_tokenizer(
    spec: TokenizerSpec,
    texts: Iterable[str],
    strict_vocab_size: bool = True,
    show_progress: bool = False,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on ``texts``.

    Args:
        spec: Tokenizer definition.
        texts: Training texts (train split only; never validation/test/benchmarks).
        strict_vocab_size: Raise if the corpus is too small to reach ``spec.vocab_size``.
        show_progress: Show the ``tokenizers`` progress bar.

    Raises:
        TokenizerVocabularyError: If ``strict_vocab_size`` and the size is not reached.
    """
    tokenizer = build_tokenizer(spec)
    trainer = BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        special_tokens=spec.special_tokens.ordered,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(_prepared(texts, spec), trainer=trainer)
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
        "exported_at_utc": utc_now(),
        "provenance": provenance or {},
    }
    (out / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
