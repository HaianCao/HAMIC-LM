"""Document model and readers for corpus sources."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from hamic.data.contamination import BenchmarkIndex

__all__ = [
    "Document",
    "DocumentSource",
    "GroupedSource",
    "HuggingFaceStreamSource",
    "JsonlSource",
    "read_jsonl",
    "write_jsonl",
]


@dataclass(frozen=True)
class Document:
    """A single text document with provenance.

    Attributes:
        doc_id: Stable identifier, unique within the corpus.
        text: Document text.
        source_id: Source Registry id of the origin.
        meta: Provenance and pipeline metadata (URL, title, flags, cluster id...).
    """

    doc_id: str
    text: str
    source_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def with_text(self, text: str) -> Document:
        """Return a copy with ``text`` replaced."""
        return replace(self, text=text)

    def with_meta(self, **updates: Any) -> Document:
        """Return a copy with ``meta`` updated."""
        return replace(self, meta={**self.meta, **updates})

    @property
    def text_sha256(self) -> str:
        """SHA-256 of the UTF-8 text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize to one JSON line."""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Document:
        """Parse one JSON line."""
        data = json.loads(line)
        return cls(
            doc_id=str(data["doc_id"]),
            text=str(data["text"]),
            source_id=str(data["source_id"]),
            meta=dict(data.get("meta", {})),
        )


class DocumentSource(Protocol):
    """Anything that yields documents for a registered source."""

    source_id: str

    def __iter__(self) -> Iterator[Document]: ...


def _open_text(path: Path, mode: str) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[Document]:
    """Yield documents from a ``.jsonl`` or ``.jsonl.gz`` file."""
    with _open_text(Path(path), "r") as stream:
        for line in stream:
            if line.strip():
                yield Document.from_json(line)


def write_jsonl(path: str | Path, documents: Iterable[Document]) -> int:
    """Write documents as JSON lines and return how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(target, "w") as stream:
        for document in documents:
            stream.write(document.to_json() + "\n")
            count += 1
    return count


@dataclass
class JsonlSource:
    """Documents stored in local JSON-lines files."""

    source_id: str
    paths: list[Path]

    def __iter__(self) -> Iterator[Document]:
        for path in self.paths:
            yield from read_jsonl(path)


def _is_transient(error: BaseException) -> bool:
    """Network failures worth retrying (not missing files, auth or programming errors)."""
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(error, HfHubHTTPError):
        status = getattr(error.response, "status_code", None)
        return status is None or status == 429 or status >= 500
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return type(error).__module__.split(".")[0] in {"httpx", "httpcore", "requests", "urllib3"}


@dataclass
class HuggingFaceStreamSource:
    """Stream documents from a Hugging Face dataset without downloading it whole.

    Attributes:
        source_id: Source Registry id.
        dataset: Dataset repository id, e.g. ``"wikimedia/wikipedia"``.
        config: Dataset configuration, e.g. ``"20231101.vi"``.
        revision: Dataset commit to pin; required for reproducibility.
        split: Dataset split.
        text_field: Column holding the text.
        id_field: Column holding a stable id; a content hash is used when absent.
        meta_fields: Columns copied into ``Document.meta``.
        limit: Stop after this many documents (``None`` streams everything).
        data_files: Repository paths of the files to read, in this order (e.g. a seeded
            selection of FineWeb-2 parquet files for a corpus delta). ``None`` reads the
            whole ``config``.
        download_dir: With ``data_files``: download the files here first (partial files
            resume, failures are retried until the network is back), then read them
            locally. Only the download needs the network, so a long build survives the
            machine sleeping or moving between networks. The result is identical to
            streaming the same files.
        file_format: Format of ``data_files``: ``"parquet"``, ``"csv"`` or ``"json"``
            (JSON lines; ``.zst`` and ``.gz`` files are decompressed while streaming).
        transform: Name of a text preparation in :data:`hamic.data.transforms.TRANSFORMS`
            (``"html"``, ``"spaced_poem"``) applied to each text.
        title_field: Column whose value is put before the text, separated by a blank line
            (e.g. a poem's title).
    """

    source_id: str
    dataset: str
    config: str | None
    revision: str
    split: str = "train"
    text_field: str = "text"
    id_field: str | None = "id"
    meta_fields: tuple[str, ...] = ("url", "title")
    limit: int | None = None
    data_files: tuple[str, ...] | None = None
    download_dir: str | None = None
    file_format: str = "parquet"
    transform: str | None = None
    title_field: str | None = None

    def __post_init__(self) -> None:
        from hamic.data.transforms import TRANSFORMS

        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform {self.transform!r}; known: {sorted(TRANSFORMS)}")

    def download(self, attempts: int = 10_000, wait_seconds: float = 30.0) -> list[Path]:
        """Download ``data_files`` into ``download_dir`` in order; retry until done."""
        import time

        from huggingface_hub import hf_hub_download

        if not self.data_files or not self.download_dir:
            raise ValueError("download needs data_files and download_dir")
        paths = []
        for name in self.data_files:
            for attempt in range(1, attempts + 1):
                try:
                    local = hf_hub_download(
                        self.dataset,
                        name,
                        repo_type="dataset",
                        revision=self.revision,
                        local_dir=self.download_dir,
                    )
                    break
                except Exception as error:
                    if attempt == attempts or not _is_transient(error):
                        raise
                    print(f"download {name}: {error!r}; retry {attempt} in {wait_seconds:.0f}s")
                    time.sleep(wait_seconds)
            paths.append(Path(local))
        return paths

    def __iter__(self) -> Iterator[Document]:
        from datasets import load_dataset

        from hamic.data.transforms import TRANSFORMS

        prepare = TRANSFORMS[self.transform] if self.transform else None

        if self.data_files and self.download_dir:
            local = [str(path) for path in self.download()]
            stream = load_dataset(
                self.file_format, data_files={self.split: local}, split=self.split, streaming=True
            )
        elif self.data_files:
            # Read the listed parquet files directly, pinned to the revision and in order;
            # the repository's own configs cannot be combined with ``data_files``.
            urls = [
                f"hf://datasets/{self.dataset}@{self.revision}/{path}" for path in self.data_files
            ]
            stream = load_dataset(
                self.file_format, data_files={self.split: urls}, split=self.split, streaming=True
            )
        else:
            stream = load_dataset(
                self.dataset,
                self.config,
                split=self.split,
                revision=self.revision,
                streaming=True,
            )
        for index, row in enumerate(stream):
            if self.limit is not None and index >= self.limit:
                break
            text = row[self.text_field]
            if not isinstance(text, str):  # empty CSV cells
                continue
            if prepare is not None:
                text = prepare(text)
            title = row.get(self.title_field) if self.title_field else None
            if isinstance(title, str) and title.strip():
                text = f"{title.strip()}\n\n{text}"
            raw_id = row.get(self.id_field) if self.id_field else None
            doc_id = (
                f"{self.source_id}:{raw_id}"
                if raw_id is not None
                else (f"{self.source_id}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}")
            )
            meta = {name: row[name] for name in self.meta_fields if name in row}
            meta["revision"] = self.revision
            yield Document(doc_id=doc_id, text=text, source_id=self.source_id, meta=meta)


class GroupedSource:
    """Join consecutive short rows of a source into documents of about ``target_chars``.

    Social-media comments average about 80 characters, below the quality filter's 200,
    and an encoder needs longer context. Rows are joined with ``separator`` in source
    order; the last group may be shorter. With ``exclude``, a row equal to or sharing an
    n-gram with a benchmark text is dropped before grouping, so one contaminated comment
    does not cost the whole document.

    Attributes:
        inner: The row source.
        target_chars: A group is closed once it reaches this many characters.
        separator: Joins the rows of a group.
        exclude: Benchmark index applied to each row.
        counts: ``rows``, ``empty``, ``excluded`` and ``documents`` of the last pass.
    """

    def __init__(
        self,
        inner: DocumentSource,
        target_chars: int = 1500,
        separator: str = "\n",
        exclude: BenchmarkIndex | None = None,
    ) -> None:
        if target_chars < 1:
            raise ValueError("target_chars must be positive")
        self.inner = inner
        self.source_id = inner.source_id
        self.target_chars = target_chars
        self.separator = separator
        self.exclude = exclude
        self.counts: dict[str, int] = {}

    def __iter__(self) -> Iterator[Document]:
        counts = {"rows": 0, "empty": 0, "excluded": 0, "documents": 0}
        self.counts = counts
        rows: list[str] = []
        first: Document | None = None
        size = 0
        for row in self.inner:
            counts["rows"] += 1
            text = row.text.strip()
            if not text:
                counts["empty"] += 1
                continue
            if self.exclude is not None and self.exclude.matches(text):
                counts["excluded"] += 1
                continue
            first = first or row
            rows.append(text)
            size += len(text) + len(self.separator)
            if size >= self.target_chars:
                yield self._group(rows, first, counts["documents"])
                counts["documents"] += 1
                rows, first, size = [], None, 0
        if rows and first is not None:
            yield self._group(rows, first, counts["documents"])
            counts["documents"] += 1
        print(f"{self.source_id}: {counts}", flush=True)

    def _group(self, rows: list[str], first: Document, index: int) -> Document:
        meta = {**first.meta, "rows": len(rows), "first_row": first.doc_id}
        return Document(
            doc_id=f"{self.source_id}:group-{index:09d}",
            text=self.separator.join(rows),
            source_id=self.source_id,
            meta=meta,
        )
