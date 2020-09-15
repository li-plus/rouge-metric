import pytest

from rouge_metric import PerlRouge


def test_perl_rouge():
    hyp = ['a b c d e\nx y z']
    ref = [['b c d e\nr s t']]
    # default
    scores = PerlRouge().evaluate(hyp, ref)
    assert set(scores.keys()) == set(('rouge-1', 'rouge-2', 'rouge-l'))
    # empty
    scores = PerlRouge(rouge_n_max=0, rouge_l=False).evaluate(hyp, ref)
    assert not scores
    # rouge-n
    scores = PerlRouge(rouge_n_max=4, rouge_l=False).evaluate(hyp, ref)
    assert set(scores.keys()) == set('rouge-{}'.format(n) for n in range(1, 5))
    # rouge-l
    scores = PerlRouge(rouge_n_max=0).evaluate(hyp, ref)
    assert set(scores.keys()) == set(('rouge-l',))
    # rouge-w
    scores = PerlRouge(rouge_n_max=0, rouge_l=False, rouge_w=True,
                       rouge_w_weight=1.5).evaluate(hyp, ref)
    # rouge-s
    assert set(scores.keys()) == set(('rouge-w-1.5',))
    scores = PerlRouge(rouge_n_max=0, rouge_l=False,
                       rouge_s=True, skip_gap=4).evaluate(hyp, ref)
    assert set(scores.keys()) == set(('rouge-s4',))
    # rouge-su
    scores = PerlRouge(rouge_n_max=0, rouge_l=False,
                       rouge_su=True, skip_gap=4).evaluate(hyp, ref)
    assert set(scores.keys()) == set(('rouge-su4',))
    # all
    scores = PerlRouge(rouge_w=True, rouge_s=True,
                       rouge_su=True).evaluate(hyp, ref)
    assert set(scores.keys()) == set(('rouge-1', 'rouge-2', 'rouge-l',
                                      'rouge-w-1.2', 'rouge-s*', 'rouge-su*'))
    # exception
    with pytest.raises(ValueError):
        PerlRouge().evaluate(['oops', 'oops'], [['oops']])
    with pytest.raises(ValueError):
        PerlRouge(confidence=101).evaluate(hyp, ref)
    with pytest.raises(ValueError):
        PerlRouge(multi_ref_mode='oops').evaluate(hyp, ref)
    with pytest.raises(ValueError):
        PerlRouge(alpha=1.2).evaluate(hyp, ref)
    with pytest.raises(ValueError):
        PerlRouge(rouge_w=True, rouge_w_weight=0.8).evaluate(hyp, ref)
    with pytest.raises(ValueError):
        PerlRouge(word_limit=100, byte_limit=100).evaluate(hyp, ref)
