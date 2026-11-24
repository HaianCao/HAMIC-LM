import gzip
import json

import pytest

from hamic.data import (
    BenchmarkIndex,
    ContaminationFilterStage,
    Document,
    GroupedSource,
    JsonlSource,
    PipelineConfig,
    write_jsonl,
)
from hamic.data.transforms import html_to_text, spaced_poem

BENCH = [
    "Mình rất thích sản phẩm này, giao hàng nhanh và đóng gói cẩn thận lắm luôn ạ",
    "Thủ đô của Việt Nam là Hà Nội",
    "ok",
]


def test_index_matches_copies_not_ordinary_text():
    index = BenchmarkIndex(BENCH, ngram=8)
    assert index.matches("THỦ ĐÔ của Việt Nam là   Hà Nội!!")  # exact up to case and punctuation
    assert index.matches("haha mình rất thích sản phẩm này giao hàng nhanh và đóng gói")  # 8-gram
    assert not index.matches("ok")  # too short to count as a copy
    assert not index.matches("Hôm nay mình đi chợ mua rau và cá về nấu canh")


def test_document_rule_needs_coverage_of_one_benchmark_text():
    motto = "Cộng hòa xã hội chủ nghĩa Việt Nam Độc lập Tự do Hạnh phúc"
    comment = motto + " mà sao bọn này làm ăn chán thế không biết nữa trời ơi là trời luôn"
    index = BenchmarkIndex([comment, BENCH[0]], ngram=8, min_coverage=0.5)
    decree = motto + " Quyết định về việc ban hành quy chế quản lý tài sản công của tỉnh"
    assert not index.matches(decree)  # shares a few 8-grams of the comment, not half
    assert index.matches("Đánh giá: " + BENCH[0] + " Năm sao.")  # copies a whole text


def test_contamination_stage_rejects_documents(tmp_path):
    stage = ContaminationFilterStage(BenchmarkIndex(BENCH, ngram=8))
    clean = Document("a", "Một văn bản bình thường không liên quan gì cả.", "s")
    copied = Document("b", "Trích: " + BENCH[0], "s")
    assert stage.process(clean) is clean
    assert stage.process(copied) == "benchmark_overlap"
    assert stage.parallel_safe


def test_grouped_source_joins_rows_and_drops_benchmark_rows(tmp_path):
    rows = [Document(f"r{i}", f"bình luận số {i} vui quá", "social") for i in range(10)]
    rows.insert(3, Document("bad", BENCH[0], "social"))
    rows.insert(5, Document("empty", "   ", "social"))
    write_jsonl(tmp_path / "rows.jsonl", rows)
    source = GroupedSource(
        JsonlSource("social", [tmp_path / "rows.jsonl"]),
        target_chars=60,
        exclude=BenchmarkIndex(BENCH, ngram=8),
    )
    docs = list(source)
    assert source.counts == {"rows": 12, "empty": 1, "excluded": 1, "documents": len(docs)}
    assert all(BENCH[0] not in d.text for d in docs)
    joined = "\n".join(d.text for d in docs).split("\n")
    assert joined == [f"bình luận số {i} vui quá" for i in range(10)]  # order kept
    assert docs[0].doc_id == "social:group-000000000"
    assert docs[0].meta["first_row"] == "r0"


def test_html_to_text_keeps_blocks_and_drops_head():
    html = (
        "<html><head><title>x</title><style>p{}</style></head><body>"
        "<p>Điều 1.&nbsp;Phạm vi</p><div>Nghị định này <b>quy định</b></div>"
        "<table><tr><td>A</td></tr></table></body></html>"
    )
    assert html_to_text(html) == "Điều 1. Phạm vi\n\nNghị định này quy định\n\nA"


def test_spaced_poem_restores_lines_punctuation_and_capitals():
    raw = "đẩy hoa dun lá khỏi tay trời , <\n> nghĩ lại tình duyên luống ngậm ngùi . "
    expected = "Đẩy hoa dun lá khỏi tay trời,\nNghĩ lại tình duyên luống ngậm ngùi."
    assert spaced_poem(raw) == expected


def test_pipeline_config_pins_the_benchmark_file(tmp_path):
    from hamic.provenance import file_sha256

    path = tmp_path / "bench.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for text in BENCH:
            stream.write(json.dumps({"task": "t", "split": "test", "text": text}) + "\n")
    good = {"benchmarks": str(path), "sha256": file_sha256(path), "splits": ["test"]}
    config = PipelineConfig.from_mapping({"contamination": good})
    assert config.contamination is not None
    assert config.contamination.splits == ("test",)
    names = [stage.name for stage in config.build_stages()]
    assert names.index("contamination") == names.index("quality_filter") + 1
    assert "contamination" in config.to_dict()
    assert "contamination" not in PipelineConfig().to_dict()  # older hashes unchanged
    bad = PipelineConfig.from_mapping({"contamination": {**good, "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="sha256"):
        bad.build_stages()


def test_quality_filter_drops_legacy_font_mojibake():
    from hamic.data import QualityFilterConfig, QualityFilterStage

    stage = QualityFilterStage(QualityFilterConfig(max_legacy_font_share=0.05))
    vni = "Chæ Laø Moät Khaùm Nghieäm Ñôn Giaûn Dung Toâi bieát chò noùi raèng " * 5
    clean = "Đây là một văn bản tiếng Việt bình thường, có đủ dấu và đủ dài để qua lọc. " * 4
    assert stage.process(Document("a", vni, "s")) == "legacy_font_mojibake"
    assert isinstance(stage.process(Document("b", clean, "s")), Document)
    assert "max_legacy_font_share" not in PipelineConfig().to_dict()["quality"]
