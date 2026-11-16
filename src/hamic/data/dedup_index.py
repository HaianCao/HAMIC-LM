"""Persistent, compact deduplication index shared by corpus versions.

A corpus delta (v0.3, v0.4, ...) must drop documents that duplicate anything already
published, without keeping earlier texts in memory. The index stores, for every kept
document:

* one 64-bit key of its normalized text (exact duplicates), and
* one 64-bit key per MinHash LSH band (near duplicates): with the pipeline's settings
  (threshold 0.8, 128 permutations) that is 9 bands of 13 rows, the banding datasketch
  picks for the same threshold.

That is 80 bytes per document instead of several kilobytes for an in-memory LSH, so the
whole of FineWeb-2 Vietnamese (~50M documents) fits in a few gigabytes. As in the
pipeline's in-memory stage, a document sharing any band with an indexed one counts as a
near duplicate.

Keys of earlier versions live in sorted ``int64`` arrays (binary search); keys added
during a build live in Python sets until :meth:`DedupIndex.save` merges them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DedupIndex", "band_keys", "text_key"]

INDEX_MANIFEST = "dedup_index.json"


def text_key(text_sha256: str) -> int:
    """64-bit exact-duplicate key from a document's ``text_sha256`` hex digest."""
    return int(text_sha256[:16], 16) - (1 << 63)


def band_keys(hashvalues: Any, bands: int, rows: int) -> list[int]:
    """One signed 64-bit key per LSH band of a MinHash signature."""
    values = np.asarray(hashvalues, dtype=np.uint64)
    keys = []
    for band in range(bands):
        chunk = values[band * rows : (band + 1) * rows].tobytes()
        digest = hashlib.blake2b(chunk, digest_size=8, person=band.to_bytes(16, "big"))
        keys.append(int.from_bytes(digest.digest(), "big", signed=True))
    return keys


@dataclass
class _KeySet:
    """Sorted array of earlier keys plus a set of keys added in this build."""

    frozen: np.ndarray[Any, Any]
    added: set[int]

    def __contains__(self, key: int) -> bool:
        if key in self.added:
            return True
        position = int(np.searchsorted(self.frozen, key))
        return position < self.frozen.size and int(self.frozen[position]) == key

    def merged(self) -> np.ndarray[Any, Any]:
        extra = np.fromiter(self.added, dtype=np.int64, count=len(self.added))
        return np.unique(np.concatenate([self.frozen, extra]))


class DedupIndex:
    """Exact and near-duplicate keys of every document kept so far.

    Args:
        num_perm: MinHash permutations (must match the pipeline).
        shingle_size: Syllable shingle size (must match the pipeline).
        bands: LSH bands.
        rows: Rows per band (``bands * rows <= num_perm``).
        versions: Corpus versions already indexed, oldest first.
    """

    def __init__(
        self,
        num_perm: int = 128,
        shingle_size: int = 5,
        bands: int = 9,
        rows: int = 13,
        versions: Iterable[str] = (),
    ) -> None:
        if bands * rows > num_perm:
            raise ValueError("bands * rows must not exceed num_perm")
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.bands = bands
        self.rows = rows
        self.versions = list(versions)
        empty = np.zeros(0, dtype=np.int64)
        self._exact = _KeySet(empty, set())
        self._bands = [_KeySet(empty, set()) for _ in range(bands)]

    def __len__(self) -> int:
        """Indexed documents (one exact key each)."""
        return int(self._exact.frozen.size) + len(self._exact.added)

    def has_exact(self, text_sha256: str) -> bool:
        """Whether a document with this normalized text is indexed."""
        return text_key(text_sha256) in self._exact

    def has_near(self, hashvalues: Any) -> bool:
        """Whether any LSH band of this signature matches an indexed document."""
        keys = band_keys(hashvalues, self.bands, self.rows)
        return any(key in band for key, band in zip(keys, self._bands, strict=True))

    def add(self, text_sha256: str, hashvalues: Any) -> None:
        """Index one kept document."""
        self._exact.added.add(text_key(text_sha256))
        for key, band in zip(
            band_keys(hashvalues, self.bands, self.rows), self._bands, strict=True
        ):
            band.added.add(key)

    def add_exact(self, text_sha256: str) -> None:
        """Record an exact key alone (near keys come with :meth:`add`)."""
        self._exact.added.add(text_key(text_sha256))

    def settings(self) -> dict[str, int]:
        """Signature settings a pipeline must use with this index."""
        return {
            "num_perm": self.num_perm,
            "shingle_size": self.shingle_size,
            "bands": self.bands,
            "rows": self.rows,
        }

    def save(self, directory: str | Path, version: str | None = None) -> dict[str, Any]:
        """Merge added keys and write the index; ``version`` is appended to its history."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._exact = _KeySet(self._exact.merged(), set())
        self._bands = [_KeySet(band.merged(), set()) for band in self._bands]
        if version is not None and version not in self.versions:
            self.versions.append(version)
        arrays = {"exact": self._exact.frozen}
        arrays.update({f"band_{i}": band.frozen for i, band in enumerate(self._bands)})
        staging = out / "keys.npz.tmp"
        with staging.open("wb") as stream:
            np.savez(stream, **arrays)  # type: ignore[arg-type]
        staging.replace(out / "keys.npz")
        manifest = {"documents": len(self), "versions": self.versions, **self.settings()}
        (out / INDEX_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> DedupIndex:
        """Read an index written by :meth:`save`."""
        root = Path(directory)
        manifest = json.loads((root / INDEX_MANIFEST).read_text(encoding="utf-8"))
        index = cls(
            num_perm=manifest["num_perm"],
            shingle_size=manifest["shingle_size"],
            bands=manifest["bands"],
            rows=manifest["rows"],
            versions=manifest["versions"],
        )
        with np.load(root / "keys.npz") as data:
            index._exact = _KeySet(data["exact"], set())
            index._bands = [_KeySet(data[f"band_{i}"], set()) for i in range(index.bands)]
        return index
