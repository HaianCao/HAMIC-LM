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


def test_teencode_default_is_deterministic_and_whole_word():
    assert to_teencode("Không biết được gì") == "Kbt dc j"  # phrase first, case kept
    assert to_teencode("được không") == "dc ko"
    assert to_teencode("khôngkhí") == "khôngkhí"


def test_teencode_seeded_is_reproducible_and_varies():
    text = "Bây giờ mình không biết phải làm gì, cảm ơn mọi người nhiều nhé"
    first = to_teencode(text, seed=1, rule_rate=0.3)
    assert first == to_teencode(text, seed=1, rule_rate=0.3)
    assert {to_teencode(text, seed=s, rule_rate=0.3) for s in range(6)} - {first}
    assert first != text


def test_teencode_rules_and_numbers():
    from hamic.text import Teencoder

    teencoder = Teencoder()
    assert len(teencoder) > 70
    rules = {r.name for r in teencoder.rules}
    assert {"ph_to_f", "gi_to_j", "qu_to_w", "elongate", "strip_diacritics"} <= rules
    assert to_teencode("phải 2026", seed=3, word_rate=0.0, rule_rate=1.0).endswith("2026")


def test_elongation_only_stretches_vowels():
    import random

    from hamic.text.robustness import TeencodeRule

    rule = TeencodeRule(name="elongate", kind="elongate", min_repeat=2, max_repeat=2)
    rng = random.Random(0)
    assert rule.apply("quá", rng) == "quááá"
    assert rule.apply("trc", rng) == "trc"
    assert rule.apply("đt", rng) == "đt"
