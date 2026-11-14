import unicodedata

import pytest

from hamic.tokenization import (
    DictionarySegmenter,
    InputMode,
    SpecialTokens,
    TokenizerEvaluator,
    TokenizerKind,
    TokenizerSpec,
    TokenizerVocabularyError,
    train_tokenizer,
)


def test_special_tokens_have_fixed_ids(tiny_tokenizer):
    for expected_id, token in enumerate(SpecialTokens().ordered):
        assert tiny_tokenizer.token_to_id(token) == expected_id


def test_nfd_input_tokenizes_like_nfc(tiny_tokenizer):
    text = "Người dân Đắk Lắk trồng cà phê"
    nfd = unicodedata.normalize("NFD", text)
    assert tiny_tokenizer.encode(nfd).ids == tiny_tokenizer.encode(text).ids


def test_acceptance_hard_gates_pass(tiny_tokenizer, corpus):
    report = TokenizerEvaluator(tiny_tokenizer, "tiny").run(corpus[:8])
    assert report.passed_hard_gates
    assert report.corpus.tokens_per_syllable > 0
    assert set(report.probes) >= {"place_names", "teencode", "emoji"}
    assert all(p["roundtrip_ok"] for p in report.probes.values())


def test_strict_vocab_size_detects_small_corpus(corpus):
    spec = TokenizerSpec(name="too-big", vocab_size=50_000, min_frequency=1)
    with pytest.raises(TokenizerVocabularyError):
        train_tokenizer(spec, corpus[:8])


def test_segmented_mode_requires_segmenter(corpus):
    spec = TokenizerSpec(name="seg", vocab_size=600, input_mode=InputMode.SEGMENTED)
    with pytest.raises(ValueError, match="segmenter"):
        train_tokenizer(spec, corpus)
    segmenter = DictionarySegmenter(["sinh viên", "đại học", "hà nội"])
    trained = train_tokenizer(spec, corpus, segmenter=segmenter, strict_vocab_size=False)
    assert trained.get_vocab_size() > 300


def test_dictionary_segmenter_greedy_longest_match():
    segmenter = DictionarySegmenter(["đại học", "đại học quốc gia", "hà nội"])
    assert segmenter.segment("Đại học Quốc gia Hà Nội") == "Đại_học_Quốc_gia Hà_Nội"


def test_exported_encoder_tokenizer_loads_with_autotokenizer(exported_tokenizers):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(exported_tokenizers[TokenizerKind.ENCODER])
    ids = tokenizer("Hà Nội <mask>")["input_ids"]
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.mask_token_id in ids
    assert tokenizer.convert_ids_to_tokens(ids)[-3] != "Ġ"  # <mask> absorbs the space
    assert tokenizer.decode(ids, skip_special_tokens=True).strip() == "Hà Nội"


def test_exported_decoder_tokenizer_has_no_trailing_eos(exported_tokenizers):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(exported_tokenizers[TokenizerKind.DECODER])
    ids = tokenizer("Thủ đô của Việt Nam là")["input_ids"]
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] != tokenizer.eos_token_id


def test_comparison_table_lists_every_report(tiny_tokenizer, corpus):
    from hamic.tokenization import comparison_table

    report = TokenizerEvaluator(tiny_tokenizer, "tiny").run(corpus[:4]).to_dict()
    table = comparison_table([report, {**report, "tokenizer_name": "copy"}])
    assert "| tiny |" in table
    assert "| copy |" in table
    assert "place_names" in table


def test_probe_set_is_built_from_resources():
    from hamic.tokenization import load_probes

    probes = load_probes()
    cats = probes.categories
    assert len(cats["person_names"]) == 48
    assert len(set(cats["person_names"])) == 48
    assert "Đắk Lắk" in cats["place_names"]
    assert len(cats["place_names"]) >= 60
    assert all(s.isascii() or "đ" not in s for s in cats["no_diacritics"])
    assert cats["teencode"] == load_probes().categories["teencode"]  # seeded, reproducible
    assert probes.identity["probes"]["version"] == 1
    assert len(probes.identity["probes"]["sha256"]) == 64


def test_report_records_probe_identity(tiny_tokenizer, corpus):
    report = TokenizerEvaluator(tiny_tokenizer, "tiny").run(corpus[:2]).to_dict()
    assert report["probe_set"]["teencode"]["name"] == "teencode.yaml"
