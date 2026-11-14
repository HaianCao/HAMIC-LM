"""Word segmentation adapters for the segmented-input control arm.

HAMIC's main path reads raw text. Segmented input exists only to measure whether an
external word segmenter helps; segmenters are therefore pluggable behind
:class:`Segmenter`, and heavy third-party ones are optional imports.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

__all__ = ["DictionarySegmenter", "PyviSegmenter", "Segmenter"]


@runtime_checkable
class Segmenter(Protocol):
    """Joins the syllables of each multi-syllable word with ``_``."""

    def segment(self, text: str) -> str:
        """Return ``text`` with words segmented."""
        ...


class DictionarySegmenter:
    """Greedy longest-match segmenter over a word list.

    Intended for tests and small controlled experiments, not as a production segmenter.

    Example:
        >>> DictionarySegmenter(["sinh viên", "đại học"]).segment("sinh viên đại học")
        'sinh_viên đại_học'
    """

    def __init__(self, words: Iterable[str], max_syllables: int = 4) -> None:
        self._words = {w.lower() for w in words if " " in w.strip()}
        self._max = max_syllables

    def segment(self, text: str) -> str:
        """Segment whitespace-separated syllables."""
        syllables = text.split()
        out: list[str] = []
        i = 0
        while i < len(syllables):
            for size in range(min(self._max, len(syllables) - i), 1, -1):
                candidate = syllables[i : i + size]
                if " ".join(candidate).lower() in self._words:
                    out.append("_".join(candidate))
                    i += size
                    break
            else:
                out.append(syllables[i])
                i += 1
        return " ".join(out)


class PyviSegmenter:
    """Adapter for the ``pyvi`` tokenizer (optional dependency, check its license first)."""

    def __init__(self) -> None:
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyviSegmenter needs `pip install pyvi`") from exc
        self._impl = ViTokenizer

    def segment(self, text: str) -> str:
        """Segment with pyvi."""
        return str(self._impl.tokenize(text))
