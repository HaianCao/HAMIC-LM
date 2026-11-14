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
