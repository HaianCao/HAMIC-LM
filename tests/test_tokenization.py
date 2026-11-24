import unicodedata

import pytest

from hamic.tokenization import (
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


# -- the HAMIC pipeline: prefix space, digit pairs, look-alikes, required strings -----


def _pieces(tokenizer, text):
    encoding = tokenizer.encode(text)
    return [text[s:e] for s, e in encoding.offsets], encoding.ids


def test_prefix_space_makes_first_word_match_running_text(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    first = tok.encode("Hà Nội").ids
    inner = tok.encode("ở Hà Nội").ids
    assert first == inner[-len(first) :]


def test_long_digit_runs_split_in_pairs_from_the_right(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    text = "năm 2026 giá 1.000.000 mã 12345 gọi 0912345678"
    words = [w.replace("Ġ", "") for w, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
    digits = ["2026", "1", "000", "000", "1", "23", "45", "09", "12", "34", "56", "78"]
    assert [w for w in words if w.isdigit()] == digits


def test_confusable_eth_folds_to_vietnamese_d(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    assert tok.encode("Ðà Lạt").ids == tok.encode("Đà Lạt").ids


def test_required_strings_are_single_tokens(tiny_spec, tiny_tokenizer):
    from hamic.tokenization.trainer import required_strings

    spec, tok = tiny_spec, tiny_tokenizer
    for string in required_strings(spec):
        assert len(tok.encode("Hà " + string).ids) == 2, string  # after a space
        assert len(tok.encode("Hà" + string).ids) == 2, string  # attached to a word


def test_hamic_pipeline_passes_acceptance(tiny_spec, tiny_tokenizer, corpus):
    tok = tiny_tokenizer
    report = TokenizerEvaluator(tok, "t1b").run([*corpus[:8], "Ðà Nẵng 😍😍 năm 2026"])
    assert report.passed_hard_gates
    assert all(p["roundtrip_ok"] for p in report.probes.values())


def test_exported_tokenizer_masks_like_roberta(tiny_spec, tiny_tokenizer, tmp_path):
    from transformers import AutoTokenizer

    from hamic.tokenization import export_tokenizer

    spec, tok = tiny_spec, tiny_tokenizer
    out = export_tokenizer(tok, spec, tmp_path / "enc", TokenizerKind.ENCODER)
    hf = AutoTokenizer.from_pretrained(out)
    plain = hf("Hà Nội")["input_ids"]
    masked = hf("Hà <mask>")["input_ids"]
    assert masked[1] == plain[1]
    assert hf.mask_token_id in masked
    assert hf.decode(plain, skip_special_tokens=True).strip() == "Hà Nội"


def test_corpus_normalization_matches_the_corpus_pipeline(tiny_spec, tiny_tokenizer):
    from hamic.text.normalization import NormalizationConfig, VietnameseNormalizer

    tok = tiny_tokenizer
    corpus = VietnameseNormalizer(NormalizationConfig(unescape_html=False))
    nbsp, zwsp, tab, cr, lf = chr(0xA0), chr(0x200B), chr(9), chr(13), chr(10)
    acute, dot, circumflex = chr(0x301), chr(0x323), chr(0x302)
    for text in [
        f"  Tôi{nbsp}đi{tab}{tab}học {cr}{lf}{cr}{lf}{cr}{lf}{cr}{lf}bài{zwsp} mới  ",
        f"a {lf} b",
        f"Tie{circumflex}{acute}ng Vie{dot}{circumflex}t",
    ]:
        normalized = tok.normalizer.normalize_str(text).replace(lf + " ", lf)  # virtual space
        assert normalized == corpus(text)
    assert tok.encode(f"Hà{nbsp}Nội").ids == tok.encode("Hà Nội").ids


def test_words_after_openers_use_their_running_text_token(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    lf = chr(10)
    running = tok.encode("ở Hà Nội").ids[-2:]
    for text in ["ở (Hà Nội)", "là “Hà Nội”", f"Thủ đô{lf}Hà Nội", "x [Hà Nội]"]:
        ids = tok.encode(text).ids
        assert any(ids[i : i + 2] == running for i in range(len(ids))), text
        assert tok.decode(ids).lstrip(" ") == text
    for text in ["ở (Hà Nội)", "(Hà Nội) ở đây"]:  # offsets stay on the original characters
        pieces, _ = _pieces(tok, text)
        assert "(" in pieces, (text, pieces)
        assert not any(p.startswith("(") and len(p) > 1 for p in pieces), (text, pieces)


def test_lookalikes_fold_only_inside_latin_words(tiny_spec, tiny_tokenizer):
    from hamic.text.lookalikes import fold_lookalikes
    from hamic.tokenization import prepare_text

    pe, dze, u_cyr, alpha, small_c, small_n, small_h, ka, en = (
        chr(0x43F),
        chr(0x455),
        chr(0x443),
        chr(0x3B1),
        chr(0x1D04),
        chr(0x274),
        chr(0x29C),
        chr(0x43A),
        chr(0x43D),
    )
    obfuscated = (
        f"{pe}gười {dze}ẽ nà{u_cyr} củ{alpha} k{small_h}ô{small_n}g {small_c}ó {ka}{en}ông"
    )
    folded = fold_lookalikes(obfuscated)
    assert folded == "người sẽ này của không có không"
    assert len(folded) == len(obfuscated)  # offsets stay valid
    for genuine in ["Москва", f"{alpha}-helix"]:
        assert fold_lookalikes(genuine) == genuine
    tok = tiny_tokenizer
    assert tok.encode(prepare_text(obfuscated)).ids == tok.encode(folded).ids


def test_finetuning_prepares_text_like_pretraining():
    from hamic.downstream.finetune import _prepared
    from hamic.downstream.tasks import TaskData

    text = chr(0x43F) + "gười"
    data = TaskData(None, {"test": [{"text": text, "label": 0}]}, ["a"])  # type: ignore[arg-type]
    assert _prepared(data).splits["test"][0] == {"text": "người", "label": 0}


def test_nfkc_folds_fancy_and_fullwidth_text(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    bold_n, bold_a, bold_m = chr(0x1D427), chr(0x1D41A), chr(0x1D426)
    fullwidth_comma, vs16 = chr(0xFF0C), chr(0xFE0F)
    assert tok.normalizer.normalize_str(f"Việt {bold_n}{bold_a}{bold_m}") == "Việt nam"
    assert tok.normalizer.normalize_str(f"a{fullwidth_comma}b") == "a,b"
    assert tok.normalizer.normalize_str(chr(0x2764) + vs16) == chr(0x2764)


def test_opening_straight_quote_gets_a_virtual_space(tiny_spec, tiny_tokenizer):
    tok = tiny_tokenizer
    running = tok.encode("ở Hà Nội").ids[-2:]
    text = 'gọi là "Hà Nội" ở đây'
    ids = tok.encode(text).ids
    assert any(ids[i : i + 2] == running for i in range(len(ids)))
    assert tok.decode(ids).lstrip(" ") == text
    assert tok.normalizer.normalize_str("l'Auxerrois") == "l'Auxerrois"  # apostrophe


def test_acceptance_restores_quote_with_space_at_text_start(tiny_tokenizer):
    # The decoder drops a real space after an opening quote; at position 0 the prefix
    # space plays the whitespace before the quote (social comments starting '" nên').
    from hamic.tokenization.acceptance import _TokenizersAdapter

    adapter = _TokenizersAdapter(tiny_tokenizer)
    for text in ('" nên mình thấy', 'nói " xin chào', "Hà Nội"):
        assert adapter.restores(adapter.encode(text).ids, text), text


def test_vietnamese_letters_are_never_cut_into_bytes(tiny_spec, tiny_tokenizer):
    from hamic.tokenization.acceptance import _byte_level_inverse
    from hamic.tokenization.trainer import required_characters

    letters = required_characters(tiny_spec)
    assert len(letters) == 134
    vocab = tiny_tokenizer.get_vocab()
    table = {b: u for u, b in _byte_level_inverse().items()}
    for letter in letters:
        assert "".join(table[b] for b in letter.encode("utf-8")) in vocab, letter


def test_frequent_characters_follow_the_share_rule(tiny_spec):
    from collections import Counter

    from hamic.tokenization.trainer import CHARACTER_SHARE, frequent_characters

    common = int(1 / CHARACTER_SHARE)
    counts = Counter({"a": common * 3, "踪": 5, "꧁": 1, chr(0x300): 6, "ｱ": 4})  # gate: 3
    # "ｱ" arrives NFKC-folded; "꧁" is too rare; the combining mark and ASCII are left out.
    assert frequent_characters(tiny_spec, counts) == ["踪", "ア"]  # most frequent first
