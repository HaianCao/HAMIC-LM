"""Vietnamese text normalization.

The normalizer is deliberately conservative: it only removes representation noise
(Unicode composition form, line endings, HTML entities, invisible characters) and never
changes linguistic content. In particular it keeps diacritics, letter case, punctuation,
numbers, code and both tone-mark placements (``hoà`` and ``hòa`` stay distinct).
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

__all__ = ["NormalizationConfig", "UnicodeForm", "VietnameseNormalizer"]

UnicodeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_AROUND_NEWLINE = re.compile(r"[^\S\n]*\n[^\S\n]*")


@dataclass(frozen=True)
class NormalizationConfig:
    """Options for :class:`VietnameseNormalizer`.

    Attributes:
        unicode_form: Unicode normalization form. HAMIC always trains on ``"NFC"``.
        unescape_html: Decode HTML entities such as ``&agrave;``.
        strip_zero_width: Remove zero-width spaces/joiners and byte-order marks.
        collapse_whitespace: Collapse runs of horizontal whitespace into one space and
            limit consecutive blank lines to one.
    """

    unicode_form: UnicodeForm = "NFC"
    unescape_html: bool = True
    strip_zero_width: bool = True
    collapse_whitespace: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"Unknown Unicode normalization form: {self.unicode_form!r}")


class VietnameseNormalizer:
    """Callable that maps raw text to HAMIC's canonical text form.

    Example:
        >>> VietnameseNormalizer()("Tie\u0302\u0301ng Vie\u0323\u0302t")
        'Tiếng Việt'
    """

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def __call__(self, text: str) -> str:
        """Normalize ``text`` according to :attr:`config`."""
        cfg = self.config
        result = text.replace("\r\n", "\n").replace("\r", "\n")
        if cfg.unescape_html:
            result = html.unescape(result)
        if cfg.strip_zero_width:
            result = _ZERO_WIDTH.sub("", result)
        result = unicodedata.normalize(cfg.unicode_form, result)
        if cfg.collapse_whitespace:
            result = _HORIZONTAL_SPACE.sub(" ", result)
            result = _SPACE_AROUND_NEWLINE.sub("\n", result)
            result = _MANY_BLANK_LINES.sub("\n\n", result)
            result = result.strip()
        return result

    @staticmethod
    def is_canonical(text: str) -> bool:
        """Return whether ``text`` is already in NFC form."""
        return unicodedata.is_normalized("NFC", text)
