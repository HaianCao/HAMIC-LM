import unicodedata

import pytest

from hamic.text import (
    NormalizationConfig,
    VietnameseNormalizer,
    decompose,
    strip_diacritics,
    to_teencode,
)


def test_nfd_input_becomes_nfc():
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert nfd != "Tiếng Việt"
    assert VietnameseNormalizer()(nfd) == "Tiếng Việt"


def test_tone_placement_variants_are_preserved():
    normalizer = VietnameseNormalizer()
    assert normalizer("hoà bình") != normalizer("hòa bình")


def test_case_diacritics_and_numbers_are_kept():
    text = "Đà Nẵng có 1.500.000 dân? Không!"
    assert VietnameseNormalizer()(text) == text


def test_representation_noise_is_removed():
    raw = "Hà​ Nội&nbsp;&amp;  Huế\r\n\r\n\r\n\r\nĐà   Lạt  "
    assert VietnameseNormalizer()(raw) == "Hà Nội & Huế\n\nĐà Lạt"


def test_invalid_unicode_form_is_rejected():
    with pytest.raises(ValueError, match="Unknown Unicode"):
        NormalizationConfig(unicode_form="NFX")  # type: ignore[arg-type]


def test_strip_diacritics_handles_d_bar():
    assert strip_diacritics("Đường phố Hà Nội") == "Duong pho Ha Noi"


def test_decompose_renders_the_same_but_differs():
    text = "Việt"
    assert decompose(text) != text
    assert unicodedata.normalize("NFC", decompose(text)) == text


def test_teencode_replaces_whole_words_only():
    assert to_teencode("Không biết được gì") == "ko bít dc j"
    assert to_teencode("khôngkhí") == "khôngkhí"
