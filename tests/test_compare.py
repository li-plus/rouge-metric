import json
import os
import re
from typing import Dict, List, Generator, Tuple

from rouge_metric import py_rouge, PyRouge, PerlRouge
from rouge_metric.py_rouge import isclose, ScoreType, RougeType

MAX_N = 6


def assert_close_score(out_score, ans_score, abs_tol=1e-5):
    # type: (ScoreType, ScoreType, float) -> None
    assert isclose(out_score['p'], ans_score['p'], abs_tol=abs_tol)
    assert isclose(out_score['r'], ans_score['r'], abs_tol=abs_tol)
    fscore = py_rouge._f_score(out_score['p'], out_score['r'], 0.5)
    assert isclose(out_score['f'], fscore)


def assert_close_rouge(out, ans, abs_tol=1e-5):
    # type: (RougeType, RougeType, float) -> None
    assert set(out.keys()) == set(ans.keys())
    for key in ans.keys():
        assert_close_score(out[key], ans[key], abs_tol)


def preprocess(text):
    # type: (str) -> str
    return re.sub(r'[^0-9a-z]+', ' ', text.lower()).strip()


def load_data():
    # type: () -> Dict[str, Dict[str, List[str]]]
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'test_data.json')) as f:
        data = json.load(f)
    return data


def load_sentence_pairs():
    # type: () -> Generator[Tuple[str, str]]
    for case in load_data().values():
        for summary, reference in zip(case['summaries'], case['references']):
            hyp = preprocess(summary)
            ref = preprocess(reference)
            yield hyp, ref


def load_summary_pairs():
    # type: () -> Generator[Tuple[str, str]]
    for case in load_data().values():
        hyp = [preprocess(sent) for sent in case['summaries']]
        ref = [preprocess(sent) for sent in case['references']]
        yield '\n'.join(hyp), '\n'.join(ref)


def load_all_summaries():
    # type: () -> Tuple[List[str], List[str]]
    hyps, refs = [], []
    for hyp, ref in load_summary_pairs():
        hyps.append(hyp)
        refs.append(ref)
    return hyps, refs


def load_multi_ref_summary_pairs():
    # type: () -> Generator[Tuple[str, List[str]]]
    hyps, refs = load_all_summaries()
    for idx, hyp in enumerate(hyps):
        multi_ref = refs[idx:idx + 3]
        yield hyp, multi_ref


def load_all_multi_ref_summaries():
    # type: () -> Tuple[List[str], List[List[str]]]
    hyps, refs = [], []
    for hyp, ref in load_multi_ref_summary_pairs():
        hyps.append(hyp)
        refs.append(ref)
    return hyps, refs


def test_compare_sentence():
    for hyp, ref in load_sentence_pairs():
        gt = PerlRouge(MAX_N, True, True, 1.2, True, True, 4,
                       'average').evaluate([hyp], [[ref]])
        hyp = hyp.split()
        ref = ref.split()
        # rouge-n
        for n in range(1, MAX_N + 1):
            ans = gt['rouge-{}'.format(n)]
            out = py_rouge._rouge_n_sentence_level(hyp, ref, n)
            out = out.to_score(0.5)
            assert_close_score(out, ans)
        # rouge-l
        ans = gt['rouge-l']
        out = py_rouge._rouge_l_sentence_level(hyp, ref).to_score(0.5)
        assert_close_score(out, ans)
        # rouge-w
        ans = gt['rouge-w-1.2']
        out = py_rouge._rouge_w_sentence_level(hyp, ref, 1.2)
        out = out.to_weighted_score(0.5, 1.2)
        assert_close_score(out, ans)
        # rouge-su4
        ans = gt['rouge-su4']
        out = py_rouge._rouge_su_sentence_level(hyp, ref, 4).to_score(0.5)
        assert_close_score(out, ans)


def test_compare_summary():
    for hyp, ref in load_summary_pairs():
        gt = PerlRouge(MAX_N, True, True, 1.2, True, True, 4,
                       'average').evaluate([hyp], [[ref]])
        out = PyRouge(MAX_N, True, True, 1.2, True, True, 4,
                      'average').evaluate([hyp], [[ref]])
        assert_close_rouge(out, gt)


def test_compare_multi_ref_summaries():
    for hyp, ref in load_multi_ref_summary_pairs():
        gt = PerlRouge(MAX_N, True, True, 1.2, True, True, 4, 'best').evaluate(
            [hyp], [ref])
        out = PyRouge(MAX_N, True, True, 1.2, True, True, 4, 'best').evaluate(
            [hyp], [ref])
        assert_close_rouge(out, gt)

        gt = PerlRouge(MAX_N, True, True, 1.2, True, True, 4,
                       'average').evaluate([hyp], [ref])
        out = PyRouge(MAX_N, True, True, 1.2, True, True, 4,
                      'average').evaluate([hyp], [ref])
        assert_close_rouge(out, gt)


def test_compare_all_multi_ref_summaries():
    hyp, ref = load_all_multi_ref_summaries()
    avg = PyRouge(MAX_N, True, True, 1.2, True, True, 4,
                  mode='average').evaluate(hyp, ref)
    indiv = PyRouge(MAX_N, True, True, 1.2, True, True, 4,
                    mode='individual').evaluate(hyp, ref)
    for key, avg_score in avg.items():
        assert avg_score['f'] == py_rouge._f_score(
            avg_score['p'], avg_score['r'], 0.5)
        avg_score['p'] *= len(indiv)
        avg_score['r'] *= len(indiv)
        for case in indiv:
            avg_score['p'] -= case[key]['p']
            avg_score['r'] -= case[key]['r']
    for score in avg.values():
        assert isclose(score['p'], 0, abs_tol=1e-9)
        assert isclose(score['r'], 0, abs_tol=1e-9)
