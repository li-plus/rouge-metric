from __future__ import division

import collections
import itertools
import json
import math
import os

import pytest

from rouge_metric import py_rouge
from rouge_metric.py_rouge import isclose, PyRouge


def assert_equal(out_score, ans_score):
    assert isclose(out_score['p'], ans_score['p'])
    assert isclose(out_score['r'], ans_score['r'])
    fscore = py_rouge._f_score(out_score['p'], out_score['r'], 0.5)
    assert isclose(out_score['f'], fscore)


def test_f_score():
    Case = collections.namedtuple('Case', 'alpha p r f')
    cases = [
        Case(alpha=0.5, p=0.75, r=0.6, f=2 / 3),
        Case(alpha=1 / 3, p=0.75, r=0.6, f=9 / 14),
        Case(alpha=1, p=0.75, r=0.6, f=0.75),
        Case(alpha=0, p=0.75, r=0.6, f=0.6)
    ]
    for case in cases:
        assert py_rouge._f_score(case.p, case.r, case.alpha)

    for alpha in [1.2, -0.1]:
        with pytest.raises(ValueError):
            py_rouge._f_score(0.5, 0.5, alpha)


"""ROUGE-L"""


def test_rouge_l_sentence_level():
    Case = collections.namedtuple('Case', 's1 s2 table len str rouge_l')
    cases = [
        Case(s1='I can create robots'.split(),
             s2='I make robots create robots'.split(),
             table=[[0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 2, 2],
                    [0, 1, 1, 2, 2, 3]],
             len=3, str=[(0, 0), (2, 3), (3, 4)],
             rouge_l={'p': 3 / 4, 'r': 3 / 5}),
        Case(s1=[], s2=[], table=[[0]], len=0, str=[], rouge_l={'p': 0, 'r': 0})
    ]
    for case in cases:
        assert py_rouge._lcs_table(case.s1, case.s2) == case.table
        assert py_rouge._lcs_length(case.s1, case.s2) == case.len
        assert py_rouge._lcs_elements(case.s1, case.s2, case.table) == case.str

        ans = case.rouge_l
        out = py_rouge._rouge_l_sentence_level(case.s1, case.s2).to_score(0.5)
        assert_equal(out, ans)


def test_rouge_l_summary_level():
    Case = collections.namedtuple('Case', 'hyp ref rouge_l')
    cases = [
        Case(['1 2 6 7 8'.split(), '1 3 8 9 5'.split()],
             ['1 2 3 4 5'.split()],
             {'p': 4 / 10, 'r': 4 / 5})
    ]
    for case in cases:
        out = py_rouge._rouge_l_summary_level(case.hyp, case.ref).to_score(0.5)
        ans = case.rouge_l
        assert_equal(out, ans)


"""ROUGE-N"""


def test_build_ngrams():
    Case = collections.namedtuple('Case', 's gram1 gram2 gram3 gram4 gram5')
    cases = [
        Case(s='a b c d'.split(),
             gram1=[('a',), ('b',), ('c',), ('d',)],
             gram2=[('a', 'b'), ('b', 'c'), ('c', 'd')],
             gram3=[('a', 'b', 'c'), ('b', 'c', 'd')],
             gram4=[('a', 'b', 'c', 'd')],
             gram5=[])
    ]
    for case in cases:
        for n in range(1, 5):
            out = py_rouge._build_ngrams(case.s, n)
            ans = collections.Counter(getattr(case, 'gram{}'.format(n)))
            assert out == ans


def test_match_count():
    Case = collections.namedtuple('Case', 'hyp ref ngram_matches')
    cases = [
        Case(
            hyp='My name is Jack and I am a high school student'.lower().split(),
            ref='Hello I am a high school teacher and my name is Mike'.lower().split(),
            ngram_matches=[0, 9, 6, 4, 2, 1, 0, 0]),
        Case(hyp=[], ref=[], ngram_matches=[0, 0, 0, 0, 0, 0, 0, 0])
    ]

    for case in cases:
        for n in range(1, 8):
            hyp_ngrams = py_rouge._build_ngrams(case.hyp, n)
            ref_ngrams = py_rouge._build_ngrams(case.ref, n)
            overlap_ngrams = py_rouge._intersect_ngrams(hyp_ngrams, ref_ngrams)
            out = py_rouge._count_ngrams(overlap_ngrams)
            ans = case.ngram_matches[n]
            assert out == ans


def test_rouge_n():
    Case = collections.namedtuple('Case', 'hyp ref rouge_1 rouge_2')
    cases = [
        Case(
            hyp='My name is Jack and I am a high school student'.lower().split(),
            ref='Hello I am a high school teacher and my name is Mike'.lower().split(),
            rouge_1={'p': 9 / 11, 'r': 9 / 12},
            rouge_2={'p': 6 / 10, 'r': 6 / 11}
        )
    ]

    for case in cases:
        for n in range(1, 3):
            out = py_rouge._rouge_n_sentence_level(
                case.hyp, case.ref, n).to_score(0.5)
            ans = getattr(case, 'rouge_{}'.format(n))
            assert_equal(out, ans)

            mid = 5
            out = py_rouge._rouge_n_summary_level(
                [case.hyp[:mid], case.hyp[mid:]],
                [case.ref[:mid], case.ref[mid:]], n).to_score(0.5)
            assert_equal(out, ans)


"""ROUGE-S & ROUGE-SU"""


def test_skip_bigrams():
    Case = collections.namedtuple('Case', 's skip skip0 skip1')
    cases = [
        Case(s='a b c d'.split(),
             skip=[('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'),
                   ('c', 'd')],
             skip0=[('a', 'b'), ('b', 'c'), ('c', 'd')],
             skip1=[('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
             ),
        Case(s=[], skip=[], skip0=[], skip1=[]),
        Case(s=['a'], skip=[], skip0=[], skip1=[])
    ]
    for case in cases:
        assert py_rouge._skip_bigrams(case.s, None) == collections.Counter(
            case.skip)
        assert py_rouge._skip_bigrams(case.s, 0) == collections.Counter(
            case.skip0)
        assert py_rouge._skip_bigrams(case.s, 1) == collections.Counter(
            case.skip1)


def test_rouge_s_sentence_level():
    Case = collections.namedtuple('Case', 'hyp ref rouge_s rouge_su')
    cases = [
        Case(hyp='police killed the gunman'.split(),
             ref='police kill the gunman'.split(),
             rouge_s={'p': 3 / 6, 'r': 3 / 6},
             rouge_su={'p': 5 / 9, 'r': 5 / 9}),
        Case(hyp='police killed the gunman'.split(),
             ref='the gunman kill police'.split(),
             rouge_s={'p': 1 / 6, 'r': 1 / 6},
             rouge_su={'p': 2 / 9, 'r': 2 / 9}),
        Case(hyp='police killed the gunman'.split(),
             ref='the gunman police killed'.split(),
             rouge_s={'p': 2 / 6, 'r': 2 / 6},
             rouge_su={'p': 4 / 9, 'r': 4 / 9}),
        Case(hyp=[], ref=[], rouge_s={'p': 0, 'r': 0},
             rouge_su={'p': 0, 'r': 0}),
        Case(hyp=['a'], ref=['a'], rouge_s={'p': 0, 'r': 0},
             rouge_su={'p': 0, 'r': 0})
    ]
    for case in cases:
        out = py_rouge._rouge_s_sentence_level(
            case.hyp, case.ref, None).to_score(0.5)
        assert_equal(out, case.rouge_s)
        out = py_rouge._rouge_su_sentence_level(
            case.hyp, case.ref, None).to_score(0.5)
        assert_equal(out, case.rouge_su)


def test_rouge_s_summary_level():
    Case = collections.namedtuple('Case', 'hyp ref rouge_s rouge_su')
    cases = [
        Case(hyp=['police killed'.split(), 'the gunman'.split()],
             ref=['police kill'.split(), 'the gunman'.split()],
             rouge_s={'p': 3 / 6, 'r': 3 / 6},
             rouge_su={'p': 5 / 9, 'r': 5 / 9})
    ]
    for case in cases:
        out = py_rouge._rouge_s_summary_level(case.hyp, case.ref,
                                              None).to_score(
            0.5)
        ans = case.rouge_s
        assert_equal(out, ans)

        out = py_rouge._rouge_su_summary_level(
            case.hyp, case.ref, None).to_score(0.5)
        ans = case.rouge_su
        assert_equal(out, ans)


"""ROUGE-W"""


def test_rouge_w_sentence_level():
    Case = collections.namedtuple('Case', 'ref hyp rouge_w')
    cases = [
        Case(ref='a b c d e f g'.split(), hyp='a b c d h i k'.split(),
             rouge_w={'p': 4 / 7, 'r': 4 / 7 / 7}),
        Case(ref='a b c d e f g'.split(), hyp='a h b k c i d'.split(),
             rouge_w={'p': 4 / 7, 'r': 4 / 7 / 7}),
        Case(ref='a h b k c i d'.split(), hyp='a b c d e f g'.split(),
             rouge_w={'p': 2 / 7, 'r': 2 / 7 / 7})
    ]
    for case in cases:
        out = py_rouge._rouge_w_sentence_level(
            case.hyp, case.ref, 2).to_weighted_score(0.5, 2)
        ans = case.rouge_w
        assert_equal(out, ans)

    with pytest.raises(ValueError):
        py_rouge._rouge_w_sentence_level([], [], 0.8)


def test_rouge_w_summary_level():
    Case = collections.namedtuple('Case', 'hyp ref rouge_w')
    cases = [
        Case(hyp=['a b f g h'.split(), 'a c h i e'.split()],
             ref=['a b c d e'.split()],
             rouge_w={'p': math.sqrt(10) / 10, 'r': math.sqrt(10) / 5 ** 2}),
        Case(hyp=['a b c'.split(), 'd e f'.split()],
             ref=['a b c d e f'.split()],
             rouge_w={'p': 1, 'r': 1 / 6}),
        Case(hyp=['a b c'.split(), 'd e f'.split()],
             ref=['a b c d e f'.split(), 'a b c d e f'.split()],
             rouge_w={'p': 1, 'r': 1 / 12})
    ]
    for case in cases:
        out = py_rouge._rouge_w_summary_level(case.hyp, case.ref,
                                              2).to_weighted_score(0.5, 2)
        ans = case.rouge_w
        assert_equal(out, ans)


"""All ROUGE"""


def test_match():
    matches = [py_rouge._Match(1, 1, 2), py_rouge._Match(1, 2, 1)]
    avg = py_rouge._Match(2, 3, 3)
    best = py_rouge._Match(1, 2, 1)
    avg_agg = py_rouge._build_match_aggregator('average')
    best_agg = py_rouge._build_match_aggregator('best')
    assert avg_agg.aggregate(matches) == avg
    assert best_agg.aggregate(matches) == best

    with pytest.raises(ValueError):
        avg_agg.aggregate([])
    with pytest.raises(ValueError):
        best_agg.aggregate([])
    with pytest.raises(ValueError):
        py_rouge._build_match_aggregator('oops')
    with pytest.raises(NotImplementedError):
        py_rouge._MatchAggregator().aggregate([])
    with pytest.raises(NotImplementedError):
        py_rouge._MatchAggregator().aggregate([])
    with pytest.raises(ValueError):
        py_rouge._Match(1, 1, 1) + 1


def test_rouge_aggregator():
    assert py_rouge._build_rouge_aggregator('average', 0.5).aggregate([]) == {}
    with pytest.raises(NotImplementedError):
        py_rouge._RougeAggregator().aggregate([])
    with pytest.raises(ValueError):
        py_rouge._build_rouge_aggregator('oops', 0.5)


def test_rouge_score():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'test_data.json')) as f:
        data = json.load(f)

    max_n = 6
    rouge = PyRouge(max_n, True, True, 1.2, True, True)

    for case in data.values():
        hyp = list(
            itertools.chain.from_iterable(x.split() for x in case['summaries']))
        ref = list(itertools.chain.from_iterable(
            x.split() for x in case['references']))

        scores = rouge.evaluate_tokenized([[hyp]], [[[ref]]])
        for n in range(1, max_n + 1):
            out = scores['rouge-{}'.format(n)]
            ans = py_rouge._rouge_n_sentence_level(hyp, ref, n).to_score(0.5)
            assert_equal(out, ans)

        out = scores['rouge-l']
        ans = py_rouge._rouge_l_sentence_level(hyp, ref).to_score(0.5)
        assert_equal(out, ans)

        out = scores['rouge-w-1.2']
        ans = py_rouge._rouge_w_sentence_level(hyp, ref, 1.2).to_weighted_score(
            0.5, 1.2)
        assert_equal(out, ans)

        out = scores['rouge-s*']
        ans = py_rouge._rouge_s_sentence_level(hyp, ref, None).to_score(0.5)
        assert_equal(out, ans)

        out = scores['rouge-su*']
        ans = py_rouge._rouge_su_sentence_level(hyp, ref, None).to_score(0.5)
        assert_equal(out, ans)

    with pytest.raises(ValueError):
        rouge.evaluate(['a b'], [['a b c'], ['a b c']])
