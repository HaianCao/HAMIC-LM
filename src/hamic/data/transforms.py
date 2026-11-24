"""Source-specific text preparation, applied while reading a source.

A source in the corpus config may name one of :data:`TRANSFORMS` (``transform:``); the
choice is part of the source configuration recorded in the corpus manifest.

- ``html``: HTML to plain text (block elements become line breaks, entities decoded,
  ``<head>``, scripts and styles dropped). Used for the legal-document corpus, which
  stores each document as an HTML page.
- ``spaced_poem``: the thivien.net poetry corpus stores poems lower-cased, with a line
  break wrapped in ``<`` ``>`` and spaces before punctuation. Line breaks and punctuation are
  restored and each line starts with a capital letter, as printed poems do. Proper names
  inside a line stay lower-cased: the casing is lost in the source.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from html.parser import HTMLParser

__all__ = ["TRANSFORMS", "html_to_text", "spaced_poem"]

_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
    "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr",
    "ul",
}  # fmt: skip
_SKIP = {"head", "script", "style", "noscript", "template"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skipping += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Visible text of an HTML document, one block element per line."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (_SPACES.sub(" ", line).strip() for line in "".join(parser.parts).split("\n"))
    text = "\n".join(lines)
    return _BLANK_LINES.sub("\n\n", text).strip()


_LINE_BREAK = re.compile(r"\s*<\n>\s*")  # "<", a real line break, ">"
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…)\]”’])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[“‘])\s+")


def spaced_poem(text: str) -> str:
    """Restore line breaks, punctuation spacing and line-initial capitals of a poem."""
    text = _LINE_BREAK.sub("\n", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        for i, ch in enumerate(line):
            if ch.isalpha():
                line = line[:i] + ch.upper() + line[i + 1 :]
                break
        lines.append(line)
    return "\n".join(lines).strip()


TRANSFORMS: dict[str, Callable[[str], str]] = {"html": html_to_text, "spaced_poem": spaced_poem}
"""Transforms a source may name in the corpus config."""
