"""E6c: PMI n-gram mining, n-gram units and whole-word joining."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hamic.training.config import PretrainConfig
from hamic.training.masking import TokenUnits
from hamic.training.ngram_units import (
    NgramTables,
    NgramUnits,
    TokenClasses,
    WholeWordUnits,
    _compositions,
    mine_pmi_ngrams,
)

# id: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, 4 <mask>, then words.
VOCAB = [
    "",
    "",
    "",
    "",
    "",  # specials
    " học",
    " sinh",
    " viên",  # 5 6 7
    " Hồ",
    " Chí",
    " Minh",  # 8 9 10
    " và",
    " là",
    " có",  # 11 12 13
    ",",
    " tr",
    "í",  # 14 15 16 (16 continues a word)
    " một",
    " hai",
    " ba",
    " bốn",
    " năm",  # 17-21 filler
    " 1844",  # 22 digits
]
SPECIAL = range(5)


def classes() -> TokenClasses:
    return TokenClasses.from_texts(VOCAB)


def corpus(seed: int = 0, documents: int = 400) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    filler = [11, 12, 13, 17, 18, 19, 20, 21]
    docs = []
    for _ in range(documents):
        doc = [0]
        for _ in range(30):
            r = rng.random()
            if r < 0.15:
                doc += [8, 9, 10]  # "Hồ Chí Minh": a cohesive trigram
            elif r < 0.25:
                doc += [5, 6]  # "học sinh"
            elif r < 0.33:
                doc += [6, 7]  # "sinh viên"
            elif r < 0.36:
                doc += [5, 6, 7]  # "học sinh viên" (explained by its parts)
            else:
                doc.append(int(rng.choice(filler)))
            if rng.random() < 0.1:
                doc.append(14)
        doc.append(2)
        docs.append(np.array(doc))
    return docs


def mine(**kwargs) -> NgramTables:
    options = {"min_count": 5, "min_doc_count": 3, "min_pmi": 1.5, "max_n": 3}
    options.update(kwargs)
    return mine_pmi_ngrams(corpus(), len(VOCAB), SPECIAL, classes(), **options)


def keys(tables: NgramTables, n: int) -> set[tuple[int, ...]]:
    return {tuple(tables.decode(n, int(k))) for k in tables.tables[n]}


def test_compositions_cover_every_split() -> None:
    assert _compositions(2) == [(1, 1)]
    assert sorted(_compositions(3)) == [(1, 1, 1), (1, 2), (2, 1)]
    assert len(_compositions(4)) == 7


def test_token_classes() -> None:
    c = classes()
    assert c.word[5]
    assert c.spaced[5]
    assert c.word[16]
    assert not c.spaced[16]
    assert not c.word[14]
    assert not c.word[0]


def test_mining_keeps_cohesive_ngrams_and_rejects_explained_ones() -> None:
    tables = mine()
    assert (5, 6) in keys(tables, 2)
    assert (6, 7) in keys(tables, 2)
    assert (8, 9, 10) in keys(tables, 3)
    # "Hồ Chí" and "Chí Minh" are fragments of "Hồ Chí Minh".
    assert (8, 9) not in keys(tables, 2)
    assert (9, 10) not in keys(tables, 2)
    assert (8, 9) in keys(mine(max_extension_share=1.0), 2)
    # "học sinh viên" is mostly "học sinh" + "viên" or "học" + "sinh viên".
    assert (5, 6, 7) not in keys(tables, 3)
    # Punctuation never joins.
    assert all(14 not in ngram for n in (2, 3) for ngram in keys(tables, n))


def test_min_doc_count_filters_single_document_repeats() -> None:
    rng = np.random.default_rng(3)
    others = [np.array([0, *rng.choice([11, 12, 13, 17, 18], 40), 2]) for _ in range(50)]
    repeated = np.array([0] + [21, 19] * 200 + [2])  # 21 and 19 occur nowhere else
    docs = [repeated, *others]
    options = dict(min_count=5, min_pmi=-10.0, max_n=2, max_extension_share=1.0)
    strict = mine_pmi_ngrams(docs, len(VOCAB), SPECIAL, classes(), min_doc_count=3, **options)
    loose = mine_pmi_ngrams(docs, len(VOCAB), SPECIAL, classes(), min_doc_count=1, **options)
    assert (21, 19) not in keys(strict, 2)
    assert (21, 19) in keys(loose, 2)


def test_digits_never_join_ngrams() -> None:
    docs = [np.array([0, *([21, 22, 11, 12] * 20), 2]) for _ in range(20)]
    tables = mine_pmi_ngrams(
        docs,
        len(VOCAB),
        SPECIAL,
        classes(),
        min_count=2,
        min_doc_count=1,
        min_pmi=-10.0,
        max_n=2,
        max_extension_share=1.0,
    )
    assert all(22 not in ngram for ngram in keys(tables, 2))
    assert (11, 12) in keys(tables, 2)


def test_max_units_keeps_best() -> None:
    assert len(mine(max_units=1)) == 1


def test_tables_round_trip(tmp_path) -> None:
    tables = mine()
    tables.save(tmp_path / "t.npz")
    loaded = NgramTables.load(tmp_path / "t.npz")
    assert loaded.vocab_size == tables.vocab_size
    for n in tables.tables:
        np.testing.assert_array_equal(loaded.tables[n], tables.tables[n])


def units_of(tables: NgramTables, row: list[int], keep: float = 1.0) -> list[int]:
    units = NgramUnits(tables, classes(), SPECIAL, keep=keep)
    return units(torch.tensor([row]), torch.Generator().manual_seed(0))[0].tolist()


def table(**ngrams: list[tuple[int, ...]]) -> NgramTables:
    base = len(VOCAB)
    out = {}
    for name, items in ngrams.items():
        n = int(name[1:])
        packed = []
        for ids in items:
            key = 0
            for i in ids:
                key = key * base + i
            packed.append(key)
        out[n] = np.array(sorted(packed), dtype=np.int64)
    return NgramTables(vocab_size=base, tables=out)


def test_units_longest_match_first_and_no_overlap() -> None:
    t = table(n2=[(5, 6), (6, 7), (9, 10)], n3=[(8, 9, 10)])
    #        <s> học sinh viên , Hồ Chí Minh </s>
    row = [0, 5, 6, 7, 14, 8, 9, 10, 2]
    unit = units_of(t, row)
    assert unit[1] == unit[2]  # "học sinh" (leftmost of the overlapping pairs)
    assert unit[3] != unit[2]  # "viên" alone
    assert unit[5] == unit[6] == unit[7]  # trigram wins over "Chí Minh"
    assert unit[4] not in (unit[3], unit[5])
    assert unit == sorted(unit)
    assert unit[0] == 0


def test_units_keep_continuations_with_their_word() -> None:
    t = table(n2=[(15, 16)])  # never matches: " tr" is not a complete syllable
    row = [0, 15, 16, 11, 2]
    unit = units_of(t, row)
    assert unit[1] == unit[2]
    assert unit[3] != unit[2]


def test_units_keep_probability() -> None:
    t = table(n2=[(5, 6)])
    row = [0] + [5, 6, 11] * 200 + [2]
    units = NgramUnits(t, classes(), SPECIAL, keep=0.5)
    unit = units(torch.tensor([row]), torch.Generator().manual_seed(1))[0]
    joined = sum(int(unit[i] == unit[i + 1]) for i in range(1, len(row) - 1, 3))
    assert 60 < joined < 140


def test_whole_word_wraps_token_units() -> None:
    units = WholeWordUnits(TokenUnits(), classes(), SPECIAL)
    row = torch.tensor([[0, 15, 16, 11, 16, 14, 16, 2]])
    unit = units(row, torch.Generator().manual_seed(0))[0].tolist()
    assert unit[1] == unit[2]  # " tr" + "í"
    assert unit[3] == unit[4]  # " và" + "í" (continuation after a word)
    assert unit[5] != unit[4]
    assert unit[6] != unit[5]  # after punctuation: new unit
    assert len(set(unit)) == 6


def test_config_fields_and_identity() -> None:
    base = dict(
        run_id="x",
        model_preset="p.yaml",
        tokenized="t",
        objective="mlm",
        seq_len=8,
        micro_batch_size=1,
        grad_accum_steps=1,
        total_tokens=64,
        learning_rate=1e-3,
    )
    default = PretrainConfig(**base)
    assert "pmi_max_ngram" not in default.identity()
    e6c = PretrainConfig(**base, masking="pmi_ngram", pmi_max_ngram=2, whole_word_masking=True)
    identity = e6c.identity()
    assert identity["pmi_max_ngram"] == 2
    assert identity["whole_word_masking"] is True
    with pytest.raises(ValueError, match="pmi_max_ngram"):
        PretrainConfig(**base, pmi_max_ngram=4)
