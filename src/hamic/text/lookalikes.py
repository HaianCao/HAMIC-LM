"""Fold Cyrillic, Greek and small-capital look-alikes inside Latin words.

Obfuscated Vietnamese ("пgười ѕẽ nàу củα кʜôɴɢ", spam and "fancy font" posts) uses
letters from other scripts that look Latin. A look-alike is replaced only when a Latin
letter touches it, so genuine Russian or Greek words ("Москва", "α-helix") stay unchanged.
Every replacement is one character for one character: offsets computed on the folded
text are valid on the original (NER labels, QA answer spans).

The mapping lives in ``hamic/resources/confusables.yaml`` (``latin_context``). Running it
inside a ``tokenizers`` normalizer would need one regex pass per target letter (measured
0.5M characters/s, hours over a full corpus), so HAMIC applies it in Python, where only
the rare texts that contain a look-alike are touched.
"""

from __future__ import annotations

import functools
import re

from hamic.resources import load_resource

__all__ = ["fold_lookalikes", "lookalike_identity"]

RESOURCE = "confusables.yaml"

# Latin letters (Basic Latin, Latin-1, Extended-A/B, IPA incl. small capitals, phonetic
# extensions, Latin Extended Additional with all Vietnamese precomposed letters).
_LATIN = "A-Za-zÀ-ÖØ-öø-ɏɐ-ʯᴀ-ᴫḀ-ỿ"


@functools.cache
def _folder() -> tuple[re.Pattern[str], dict[str, str]]:
    mapping = {str(k): str(v) for k, v in load_resource(RESOURCE).data["latin_context"].items()}
    if any(len(k) != 1 or len(v) != 1 for k, v in mapping.items()):
        raise ValueError(f"{RESOURCE}: latin_context must map one character to one")
    chars = re.escape("".join(mapping))
    pattern = re.compile(f"(?<=[{_LATIN}])[{chars}]|[{chars}](?=[{_LATIN}])")
    return pattern, mapping


def fold_lookalikes(text: str) -> str:
    """Return ``text`` with look-alikes next to Latin letters replaced (length preserved).

    Two passes, so a pair of look-alikes ("кн" in "кнông") is folded completely.
    """
    pattern, mapping = _folder()

    def replace(match: re.Match[str]) -> str:
        return mapping[match.group()]

    folded, count = pattern.subn(replace, text)
    return pattern.sub(replace, folded) if count else folded


def lookalike_identity() -> dict[str, object]:
    """Name, version and hash of the mapping, for manifests."""
    return load_resource(RESOURCE).identity()
