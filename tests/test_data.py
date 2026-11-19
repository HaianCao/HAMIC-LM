import datetime as dt
import json
from pathlib import Path

import pytest
import yaml

from hamic.data import (
    CorpusPipeline,
    Document,
    ExactDedupStage,
    JsonlSource,
    NearDedupStage,
    QualityFilterStage,
    SourceNotApprovedError,
    SourceRecord,
    SourceRegistry,
    SourceStatus,
    SplitStage,
    UseCase,
    read_jsonl,
    sample_corpora,
    write_jsonl,
)

LONG_TEXT = (
    "Hôm nay tôi đi ăn phở với mẹ ở một quán nhỏ gần chợ. Quán đông người nhưng phục vụ "
    "nhanh, nước dùng trong và thơm mùi quế hồi. Sau đó hai mẹ con đi dạo quanh hồ, "
    "ngắm người tập thể dục và nói chuyện về kỳ thi sắp tới của tôi."
)


def _record(source_id: str, status: SourceStatus) -> SourceRecord:
    approved = status in {SourceStatus.APPROVED_RESEARCH, SourceStatus.APPROVED_COMMERCIAL}
    return SourceRecord(
        source_id=source_id,
        publisher="test",
        location="local",
        revision="v1",
        license="CC-BY-SA-4.0",
        status=status,
        reviewer="HAMIC" if approved else "",
        reviewed_on=dt.date(2026, 11, 14) if approved else None,
        pii_risk="low",
    )


@pytest.mark.parametrize(
    ("status", "use", "allowed"),
    [
        (SourceStatus.PENDING, UseCase.RESEARCH, False),
        (SourceStatus.REJECTED, UseCase.RESEARCH, False),
        (SourceStatus.REVOKED, UseCase.RESEARCH, False),
        (SourceStatus.APPROVED_RESEARCH, UseCase.RESEARCH, True),
        (SourceStatus.APPROVED_RESEARCH, UseCase.COMMERCIAL, False),
        (SourceStatus.APPROVED_COMMERCIAL, UseCase.COMMERCIAL, True),
    ],
)
def test_registry_fails_closed(status, use, allowed):
    registry = SourceRegistry({"s": _record("s", status)})
    if allowed:
        assert registry.require("s", use).source_id == "s"
    else:
        with pytest.raises(SourceNotApprovedError):
            registry.require("s", use)


def test_unknown_source_is_rejected():
    with pytest.raises(SourceNotApprovedError, match="not in the Source Registry"):
        SourceRegistry({}).require("missing")


def test_approval_requires_reviewer_and_date():
    with pytest.raises(ValueError, match="reviewer and date"):
        SourceRecord(
            source_id="x",
            publisher="p",
            location="l",
            revision="r",
            license="l",
            status=SourceStatus.APPROVED_RESEARCH,
        )


def test_registry_yaml_roundtrip_and_duplicates(tmp_path: Path):
    entry = {
        "source_id": "wiki",
        "publisher": "Wikimedia",
        "location": "wikimedia/wikipedia",
        "revision": "20231101.vi",
        "license": "CC-BY-SA-4.0",
        "status": "approved_research",
        "reviewer": "HAMIC",
        "reviewed_on": "2026-11-14",
        "pii_risk": "low",
        "custom_field": 1,
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"sources": [entry]}), encoding="utf-8")
    registry = SourceRegistry.from_yaml(path)
    assert registry.get("wiki").extra == {"custom_field": 1}
    path.write_text(yaml.safe_dump({"sources": [entry, entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        SourceRegistry.from_yaml(path)


def _doc(i: int, text: str) -> Document:
    return Document(doc_id=f"d{i}", text=text, source_id="s")


def test_quality_filter_keeps_normal_vietnamese_and_records_mark_ratio():
    stage = QualityFilterStage()
    kept = list(stage([_doc(0, LONG_TEXT), _doc(1, "ngắn quá")]))
    assert [d.doc_id for d in kept] == ["d0"]
    assert 0.5 < kept[0].meta["vi_mark_ratio"] <= 1.0
    assert stage.stats.rejected["too_short"] == 1


def test_quality_filter_keeps_text_without_diacritics():
    no_marks = (
        "hom nay toi di an pho voi me o mot quan nho gan cho, quan dong nguoi nhung phuc vu "
        "nhanh, nuoc dung trong va thom mui que hoi, sau do hai me con di dao quanh ho"
    ) * 2
    assert list(QualityFilterStage()([_doc(0, no_marks)]))


def test_exact_and_near_dedup():
    exact = ExactDedupStage()
    assert len(list(exact([_doc(0, LONG_TEXT), _doc(1, LONG_TEXT)]))) == 1
    near = NearDedupStage(threshold=0.7)
    variant = LONG_TEXT.replace("kỳ thi", "bài thi")
    kept = list(near([_doc(0, LONG_TEXT), _doc(1, variant)]))
    assert [d.doc_id for d in kept] == ["d0"]
    assert kept[0].meta["cluster_id"] == "d0"


def test_split_is_deterministic_and_cluster_based():
    stage = SplitStage()
    assert stage.assign("cluster-a") == stage.assign("cluster-a")
    splits = {stage.assign(f"c{i}") for i in range(2000)}
    assert splits == {"train", "validation", "test"}


def test_pipeline_end_to_end(tmp_path: Path):
    source_file = tmp_path / "in.jsonl"
    texts = [LONG_TEXT + f" Đoạn số {i} kể về một ngày khác hẳn." * (i % 3 + 1) for i in range(30)]
    write_jsonl(source_file, [_doc(i, t) for i, t in enumerate(texts)] + [_doc(99, texts[0])])
    registry = SourceRegistry({"s": _record("s", SourceStatus.APPROVED_RESEARCH)})
    out = tmp_path / "corpus"
    result = CorpusPipeline(registry, shard_docs=10).run(
        [JsonlSource("s", [source_file])], out, corpus_version="test-0.1"
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_version"] == "test-0.1"
    assert sum(result.counts.values()) <= 30
    exact = next(s for s in manifest["stages"] if s["name"] == "exact_dedup")
    assert exact["rejected"]["exact_duplicate"] == 1
    assert (out / "audit").is_dir()
    with pytest.raises(FileExistsError):
        CorpusPipeline(registry).run([JsonlSource("s", [source_file])], out, "test-0.1")


def test_pipeline_refuses_unapproved_source(tmp_path: Path):
    registry = SourceRegistry({"s": _record("s", SourceStatus.PENDING)})
    with pytest.raises(SourceNotApprovedError):
        CorpusPipeline(registry).run([JsonlSource("s", [])], tmp_path / "out", "v")


def _frozen_sample_input(path: Path, version: str, documents: list[Document]) -> Path:
    train = path / "train" / "part-00000.jsonl.gz"
    write_jsonl(train, documents)
    manifest = {
        "corpus_version": version,
        "splits": {
            "train": {
                "documents": len(documents),
                "characters": sum(len(document.text) for document in documents),
                "files": [{"path": "train/part-00000.jsonl.gz"}],
            }
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_corpus_sample_is_deterministic_streaming_and_train_only(tmp_path: Path):
    first_docs = [Document(f"a{i}", f"vÄƒn báº£n {i}", "a") for i in range(100)]
    second_docs = [Document(f"b{i}", f"tÃ i liá»‡u {i}", "b") for i in range(100)]
    first = _frozen_sample_input(tmp_path / "v1", "v1", first_docs)
    second = _frozen_sample_input(tmp_path / "v2", "v2", second_docs)
    one = sample_corpora([first, second], tmp_path / "one", 50, seed="fixed", shard_documents=9)
    two = sample_corpora([second, first], tmp_path / "two", 50, seed="fixed", shard_documents=9)

    def ids(root: Path, manifest: dict) -> set[str]:
        return {
            document.doc_id
            for entry in manifest["files"]
            for document in read_jsonl(root / entry["path"])
        }

    assert ids(tmp_path / "one", one) == ids(tmp_path / "two", two)
    assert one["documents"] == sum(item["documents"] for item in one["sources"].values())
    assert len(one["files"]) > 1
    assert (tmp_path / "one" / "sample_manifest.json").is_file()


def test_corpus_sample_source_weight_and_validation(tmp_path: Path):
    documents = [Document(f"d{i}", "x" * 10, "s") for i in range(100)]
    corpus = _frozen_sample_input(tmp_path / "v1", "v1", documents)
    base = sample_corpora([corpus], tmp_path / "base", 25, seed="fixed")
    weighted = sample_corpora(
        [corpus], tmp_path / "weighted", 25, seed="fixed", source_weights={"s": 2.0}
    )
    assert weighted["documents"] >= base["documents"]
    with pytest.raises(FileExistsError):
        sample_corpora([corpus], tmp_path / "base", 25)
    with pytest.raises(ValueError, match="duplicate corpus version"):
        sample_corpora([corpus, corpus], tmp_path / "duplicate", 25)


def test_pii_filter_drops_secrets():
    from hamic.data import PiiFilterStage

    stage = PiiFilterStage()
    leaked = "Token của tôi là hf_" + "a" * 34 + " nhé"
    assert list(stage([_doc(0, leaked)])) == []
    assert stage.stats.rejected["secret_huggingface_token"] == 1


def test_pii_filter_redacts_contacts_but_keeps_vietnamese_money():
    from hamic.data import PiiFilterStage

    text = (
        "Liên hệ nguyen.van.a@gmail.com hoặc 0912 345 678, máy chủ 192.168.1.10. "
        "Giá 1.500.000.000 đồng, dân số 100.200.300 người, ngày 25.12.2026, bản 1.2.3.4.5."
    )
    [doc] = list(PiiFilterStage()([_doc(0, text)]))
    assert "email@example.com" in doc.text
    assert "gmail" not in doc.text
    assert "0900000000" in doc.text
    assert "345 678" not in doc.text
    assert "0.0.0.0" in doc.text
    assert "192.168" not in doc.text
    assert "1.500.000.000 đồng" in doc.text
    assert "100.200.300 người" in doc.text
    assert "25.12.2026" in doc.text
    assert "1.2.3.4.5" in doc.text
    assert doc.meta["pii_redactions"] == {"email": 1, "phone": 1, "ipv4": 1}


@pytest.mark.parametrize(
    "number",
    ["ISBN 0-8018-6842-2", "ISBN 978-0-19-280268-4", "ISBN: 0912345678", "978-0912345678"],
)
def test_pii_filter_keeps_isbn(number):
    from hamic.data import PiiFilterStage

    [doc] = list(PiiFilterStage()([_doc(0, f"Sách tham khảo, {number}, Oxford.")]))
    assert number in doc.text


@pytest.mark.parametrize(
    "phone",
    [
        "0912 345 678",
        "091.234.5678",
        "+84 912 345 678",
        "0243 123 4567",
        "0389-123-456",
        "04-38691066",
        "043.933.6868",
        "0168 123 4567",
    ],
)
def test_pii_filter_redacts_vietnamese_phone_formats(phone):
    from hamic.data import PiiFilterStage

    [doc] = list(PiiFilterStage()([_doc(0, f"Gọi {phone} để đặt lịch.")]))
    assert doc.text == "Gọi 0900000000 để đặt lịch."


@pytest.mark.parametrize("number", ["00912345678", "05 09 2012 08:54:07", "0,5 m"])
def test_pii_filter_ignores_non_vietnamese_prefixes(number):
    from hamic.data import PiiFilterStage

    [doc] = list(PiiFilterStage()([_doc(0, f"Mã số {number} trong bảng.")]))
    assert number in doc.text


def test_parallel_pipeline_matches_serial(tmp_path: Path):
    source_file = tmp_path / "in.jsonl"
    texts = [
        LONG_TEXT + f" Đoạn {i} gọi 0912 345 67{i % 10} để biết thêm." * (i % 4 + 1)
        for i in range(60)
    ]
    docs = [_doc(i, t) for i, t in enumerate(texts)]
    docs += [_doc(100, texts[0]), _doc(101, "ngắn"), _doc(102, LONG_TEXT.replace("mẹ", "bố"))]
    write_jsonl(source_file, docs)
    registry = SourceRegistry({"s": _record("s", SourceStatus.APPROVED_RESEARCH)})

    def run(workers: int, name: str) -> dict:
        result = CorpusPipeline(registry, shard_docs=7, workers=workers).run(
            [JsonlSource("s", [source_file])], tmp_path / name, corpus_version="v"
        )
        return result.manifest

    serial, parallel = run(1, "serial"), run(3, "parallel")
    assert serial["stages"] == parallel["stages"]
    for split in ("train", "validation", "test"):
        assert [f["sha256"] for f in serial["splits"][split]["files"]] == [
            f["sha256"] for f in parallel["splits"][split]["files"]
        ]
    pii = next(s for s in parallel["stages"] if s["name"] == "pii_filter")
    assert pii["modified"]["phone"] > 0


def test_default_config_file_matches_code_defaults():
    from hamic.data import PipelineConfig

    root = Path(__file__).resolve().parents[1]
    assert PipelineConfig.from_yaml(root / "configs/pipeline/default.yaml") == PipelineConfig()


def test_config_rejects_unknown_keys():
    from hamic.data import PipelineConfig

    with pytest.raises(TypeError):
        PipelineConfig.from_mapping({"quality": {"min_char": 100}})


def test_config_changes_behaviour_and_is_recorded(tmp_path: Path):
    from hamic.data import PipelineConfig, QualityFilterConfig

    source_file = tmp_path / "in.jsonl"
    write_jsonl(source_file, [_doc(0, LONG_TEXT)])
    registry = SourceRegistry({"s": _record("s", SourceStatus.APPROVED_RESEARCH)})
    strict = PipelineConfig(name="strict", quality=QualityFilterConfig(min_chars=10_000))
    result = CorpusPipeline(registry, config=strict).run(
        [JsonlSource("s", [source_file])], tmp_path / "out", corpus_version="v"
    )
    assert sum(result.counts.values()) == 0
    assert result.manifest["config"]["quality"]["min_chars"] == 10_000
    assert result.manifest["config_sha256"] == strict.sha256()
    with pytest.raises(ValueError, match="either"):
        CorpusPipeline(registry, stages=strict.build_stages(), config=strict)
