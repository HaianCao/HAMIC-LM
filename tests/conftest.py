from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from hamic.modeling import Architecture, ModelPreset
from hamic.tokenization import TokenizerKind, TokenizerSpec, export_tokenizer, train_tokenizer

warnings.filterwarnings("ignore", category=UserWarning)

VIETNAMESE_SENTENCES = [
    "Sinh viên Đại học Quốc gia Hà Nội đang ôn thi cuối kỳ.",
    "Thủ đô của Việt Nam là Hà Nội, một thành phố nghìn năm văn hiến.",
    "Hôm qua chúng tôi đi Đà Lạt cùng gia đình và ăn bánh mì.",
    "Giá vé máy bay đi Thành phố Hồ Chí Minh là 1.500.000đ.",
    "Mai check mail rồi gửi file báo cáo cho thầy giáo nhé 😅",
    "Trí tuệ nhân tạo đang thay đổi cách chúng ta học ngôn ngữ.",
    "Người dân Đắk Lắk trồng cà phê trên vùng đất đỏ bazan.",
    "Hoà bình và hòa bình là hai cách đặt dấu đều được dùng.",
]

TINY_VOCAB = 3000


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    return VIETNAMESE_SENTENCES * 40


@pytest.fixture(scope="session")
def tiny_spec() -> TokenizerSpec:
    return TokenizerSpec(
        name="test-bbpe", vocab_size=TINY_VOCAB, min_frequency=1, required_count=200
    )


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_spec: TokenizerSpec, corpus: list[str]):
    return train_tokenizer(tiny_spec, corpus, strict_vocab_size=False)


@pytest.fixture(scope="session")
def exported_tokenizers(
    tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer, tiny_spec: TokenizerSpec
) -> dict[TokenizerKind, Path]:
    root = tmp_path_factory.mktemp("tokenizers")
    return {
        kind: export_tokenizer(tiny_tokenizer, tiny_spec, root / kind.value, kind)
        for kind in TokenizerKind
    }


def tiny_preset(architecture: Architecture, vocab_size: int) -> ModelPreset:
    decoder = architecture in {Architecture.LLAMA, Architecture.QWEN3}
    return ModelPreset(
        name=f"test-{architecture.value}",
        architecture=architecture,
        label_parameters=1e5,
        label_tolerance=100.0,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2 if decoder else None,
        head_dim=16,
        intermediate_size=96,
        vocab_size=vocab_size,
        max_position_embeddings=128,
    )
