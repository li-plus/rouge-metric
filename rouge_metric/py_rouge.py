from __future__ import division

import collections
import itertools
from typing import (List, Dict, Callable, Tuple, Iterable, Set, Counter, Union,
                    Optional)

NGramsType = Counter[Tuple[str]]
ScoreType = Dict[str, float]
RougeType = Dict[str, Dict[str, float]]

try:
    from math import isclose
except ImportError:
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        # type: (float, float, float, float) -> bool
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

"""Precision Recall & F-score"""


def _format_score(fscore, precision, recall):
    # type: (float, float, float) -> Dict[str, float]
    return {'r': recall, 'p': precision, 'f': fscore}


def _f_score(precision, recall, alpha):
    # type: (float, float, float) -> float
    if not 0 <= alpha <= 1:
        raise ValueError(
            'Invalid alpha {}: expected between [0, 1]'.format(alpha))

    if isclose(precision, 0) or isclose(recall, 0):
        return 0.0

    return recall * precision / (alpha * recall + (1 - alpha) * precision)


def _div_or_zero(dividend, divisor):
    # type: (float, float) -> float
    if isclose(divisor, 0):
        return 0.0
    else:
        return dividend / divisor


def _f_p_r_score(match_score, hyp_score, ref_score, alpha):
    # type: (float, float, float, float) -> Dict[str, float]
    precision = _div_or_zero(match_score, hyp_score)
    recall = _div_or_zero(match_score, ref_score)
    fscore = _f_score(precision, recall, alpha)
    return _format_score(fscore, precision, recall)


def _flatten(sentences):
    # type: (List[List[str]]) -> List[str]
    return list(itertools.chain.from_iterable(sentences))


"""Match statistics"""


class _Match(collections.namedtuple('BaseMatch', 'matches hyp_size ref_size')):
    def __add__(self, other):
        # type: (Union[_Match, int]) -> _Match
        if isinstance(other, int) and other == 0:
            return self
        elif isinstance(other, _Match):
            return _Match(self.matches + other.matches,
                          self.hyp_size + other.hyp_size,
                          self.ref_size + other.ref_size)
        else:
            raise ValueError('Unexpected addend {}'.format(other))

    def __radd__(self, other):
        # type: (Union[_Match, int]) -> _Match
        return self.__add__(other)

    def to_score(self, alpha):
        # type: (float) -> Dict[str, float]
        return _f_p_r_score(self.matches, self.hyp_size, self.ref_size, alpha)

    def to_weighted_score(self, alpha, weight):
        # type: (float, float) -> Dict[str, float]
        inv_weight_func = _get_weight_func(weight, inverse=True)
        precision = inv_weight_func(_div_or_zero(self.matches, self.hyp_size))
        recall = inv_weight_func(_div_or_zero(self.matches, self.ref_size))
        fscore = _f_score(precision, recall, alpha)
        return _format_score(fscore, precision, recall)


class _MatchAggregator(object):
    def aggregate(self, matches):
        # type: (Iterable[_Match]) -> _Match
        raise NotImplementedError


class _AverageMatchAggregator(_MatchAggregator):
    def aggregate(self, matches):
        # type: (Iterable[_Match]) -> _Match
        result = sum(matches)
        if result == 0:
            raise ValueError('Average on empty sequence')
        return result


class _BestMatchAggregator(_MatchAggregator):
    def aggregate(self, matches):
        # type: (Iterable[_Match]) -> _Match
        return max(matches, key=lambda x: _div_or_zero(x.matches, x.ref_size))


def _build_match_aggregator(multi_ref_mode):
    # type: (str) -> _MatchAggregator
    if multi_ref_mode == 'average':
        return _AverageMatchAggregator()
    elif multi_ref_mode == 'best':
        return _BestMatchAggregator()
    else:
        raise ValueError(
            'Invalid multi_ref_mode {}: expected (average, best)'.format(
                multi_ref_mode))


"""ROUGE-N scores"""


def _build_ngrams(sent, n):
    # type: (List[str], int) -> NGramsType
    ngrams = collections.Counter()
    for i in range(len(sent) - n + 1):
        ngrams[tuple(sent[i:i + n])] += 1
    return ngrams


def _count_ngrams(ngrams):
    # type: (NGramsType) -> int
    return sum(ngrams.values())


def _intersect_ngrams(hyp_ngrams, ref_ngrams):
    # type: (NGramsType, NGramsType) -> NGramsType
    return hyp_ngrams & ref_ngrams


def _union_ngrams(ngrams, other):
    # type: (NGramsType, NGramsType) -> NGramsType
    return ngrams | other


def _rouge_n_sentence_level(hyp, ref, n):
    # type: (List[str], List[str], int) -> _Match
    hyp_ngrams = _build_ngrams(hyp, n)
    ref_ngrams = _build_ngrams(ref, n)
    match_ngrams = _intersect_ngrams(hyp_ngrams, ref_ngrams)
    return _Match(_count_ngrams(match_ngrams), _count_ngrams(hyp_ngrams),
                  _count_ngrams(ref_ngrams))


def _rouge_n_summary_level(hyps, refs, n):
    # type: (List[List[str]], List[List[str]], int) -> _Match
    return _rouge_n_sentence_level(_flatten(hyps), _flatten(refs), n)


def _rouge_n_multi_ref(hyps, multi_refs, n, multi_ref_mode, alpha):
    # type: (List[List[str]], List[List[List[str]]], int, str, float) -> ScoreType
    agg = _build_match_aggregator(multi_ref_mode)
    match = agg.aggregate(
        _rouge_n_summary_level(hyps, refs, n) for refs in multi_refs)
    return match.to_score(alpha)


"""ROUGE-L scores"""


def _lcs_table(a, b):
    # type: (List[str], List[str]) -> List[List[int]]
    m = len(a)
    n = len(b)
    table = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table


def _lcs_length(a, b):
    # type: (List[str], List[str]) -> int
    table = _lcs_table(a, b)
    return table[-1][-1]


def _lcs_elements(a, b, table):
    # type: (List[str], List[str], List[List[float]]) -> List[Tuple[int, int]]
    s = []
    i = len(a)
    j = len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
            s.append((i, j))
        elif table[i][j] == table[i][j - 1]:
            j -= 1
        else:
            i -= 1
    s.reverse()
    return s


def _lcs_union(hyps, ref):
    # type: (List[List[str]], List[str]) -> Set[int]
    lcs_union = set()
    for hyp in hyps:
        lcs_elem = _lcs_elements(hyp, ref, _lcs_table(hyp, ref))
        lcs_union = lcs_union.union(ref_idx for _, ref_idx in lcs_elem)
    return lcs_union


def _rouge_l_sentence_level(hyp, ref):
    # type: (List[str], List[str]) -> _Match
    return _Match(_lcs_length(hyp, ref), len(hyp), len(ref))


def _rouge_l_summary_level(hyps, refs):
    # type: (List[List[str]], List[List[str]]) -> _Match
    hyp_unigram = _build_ngrams(_flatten(hyps), 1)
    match_size = 0
    for ref in refs:
        lcs_union = _lcs_union(hyps, ref)
        for ref_idx in lcs_union:
            unigram = (ref[ref_idx],)
            if hyp_unigram.get(unigram, 0) > 0:
                hyp_unigram[unigram] -= 1
                match_size += 1
    ref_len = sum(len(ref) for ref in refs)
    hyp_len = sum(len(hyp) for hyp in hyps)
    return _Match(match_size, hyp_len, ref_len)


def _rouge_l_multi_ref(hyps, multi_refs, multi_ref_mode, alpha):
    # type: (List[List[str]], List[List[List[str]]], str, float) -> ScoreType
    agg = _build_match_aggregator(multi_ref_mode)
    match = agg.aggregate(
        _rouge_l_summary_level(hyps, refs) for refs in multi_refs)
    return match.to_score(alpha)


"""ROUGE-W scores"""


def _wlcs_table(a, b, weight):
    # type: (List[str], List[str], Callable[[float], float]) -> List[List[float]]
    m = len(a)
    n = len(b)
    wlen = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
    continuous_matches = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                k = continuous_matches[i - 1][j - 1]
                wlen[i][j] = wlen[i - 1][j - 1] + weight(k + 1) - weight(k)
                continuous_matches[i][j] = k + 1
            else:
                wlen[i][j] = max(wlen[i - 1][j], wlen[i][j - 1])
                continuous_matches[i][j] = 0
    return wlen


def _wlcs_union(hyps, ref, weight_func):
    # type: (List[List[str]], List[str], Callable[[float], float]) -> List[int]
    wlcs_union = set()
    for hyp in hyps:
        wlcs_elem = _lcs_elements(hyp, ref, _wlcs_table(hyp, ref, weight_func))
        wlcs_union = wlcs_union.union(ref_idx for _, ref_idx in wlcs_elem)
    return sorted(wlcs_union)


def _rouge_w_sentence_level(hyp, ref, weight):
    # type: (List[str], List[str], float) -> _Match
    return _rouge_w_summary_level([hyp], [ref], weight)


def _get_weight_func(weight, inverse):
    # type: (float, bool) -> Callable[[float], float]
    if weight < 1:
        raise ValueError('Invalid weight {}: expected >= 1'.format(weight))

    if inverse:
        weight = 1 / weight

    return lambda x: x ** weight


def _rouge_w_summary_level(hyps, refs, weight):
    # type: (List[List[str]], List[List[str]], float) -> _Match
    weight_func = _get_weight_func(weight, inverse=False)

    hyp_flat = _flatten(hyps)
    hyp_unigrams = _build_ngrams(hyp_flat, 1)
    ref_score = weight_func(sum(weight_func(len(ref)) for ref in refs))
    hyp_score = weight_func(sum(len(hyp) for hyp in hyps))
    match_score = 0
    for ref in refs:
        wlcs_union = _wlcs_union(hyps, ref, weight_func)
        consecutive_matches = 0
        for ref_idx in wlcs_union:
            token = (ref[ref_idx],)
            if hyp_unigrams[token] > 0:
                hyp_unigrams[token] -= 1
                consecutive_matches += 1
                if ref_idx == len(ref) - 1 or ref_idx + 1 not in wlcs_union:
                    match_score += weight_func(consecutive_matches)
                    consecutive_matches = 0

    return _Match(match_score, hyp_score, ref_score)


def _rouge_w_multi_ref(hyps, multi_refs, weight, multi_ref_mode, alpha):
    # type: (List[List[str]], List[List[List[str]]], float, str, float) -> ScoreType
    agg = _build_match_aggregator(multi_ref_mode)
    match = agg.aggregate(
        _rouge_w_summary_level(hyps, refs, weight) for refs in multi_refs)
    return match.to_weighted_score(alpha, weight)


"""ROUGE-S scores"""


def _skip_bigrams(sent, skip_gap):
    # type: (List[str], Optional[int]) -> NGramsType
    bigrams = collections.Counter()
    if skip_gap is None or skip_gap < 0:
        skip_gap = len(sent)
    for lo in range(len(sent)):
        for hi in range(lo + 1, min(len(sent), lo + skip_gap + 2)):
            bigrams[(sent[lo], sent[hi])] += 1
    return bigrams


def _rouge_s_or_su(hyp, ref, skip_gap, include_unigram):
    # type: (List[str], List[str], Optional[int], bool) -> _Match
    hyp_skip = _skip_bigrams(hyp, skip_gap)
    ref_skip = _skip_bigrams(ref, skip_gap)
    if include_unigram:
        hyp_skip = _union_ngrams(hyp_skip, _build_ngrams(hyp[:-1], 1))
        ref_skip = _union_ngrams(ref_skip, _build_ngrams(ref[:-1], 1))
    match_skip = _intersect_ngrams(hyp_skip, ref_skip)
    return _Match(_count_ngrams(match_skip), _count_ngrams(hyp_skip),
                  _count_ngrams(ref_skip))


def _rouge_s_sentence_level(hyp, ref, skip_gap):
    # type: (List[str], List[str], Optional[int]) -> _Match
    return _rouge_s_or_su(hyp, ref, skip_gap, False)


def _rouge_s_summary_level(hyps, refs, skip_gap):
    # type: (List[List[str]], List[List[str]], Optional[int]) -> _Match
    return _rouge_s_sentence_level(_flatten(hyps), _flatten(refs), skip_gap)


def _rouge_s_multi_ref(hyps, multi_refs, skip_gap, multi_ref_mode, alpha):
    # type: (List[List[str]], List[List[List[str]]], Optional[int], str, float) -> ScoreType
    agg = _build_match_aggregator(multi_ref_mode)
    match = agg.aggregate(
        _rouge_s_summary_level(hyps, refs, skip_gap) for refs in multi_refs)
    return match.to_score(alpha)


"""ROUGE-SU scores"""


def _rouge_su_sentence_level(hyp, ref, skip_gap):
    # type: (List[str], List[str], Optional[int]) -> _Match
    return _rouge_s_or_su(hyp, ref, skip_gap, True)


def _rouge_su_summary_level(hyps, refs, skip_gap):
    # type: (List[List[str]], List[List[str]], Optional[int]) -> _Match
    return _rouge_su_sentence_level(_flatten(hyps), _flatten(refs), skip_gap)


def _rouge_su_multi_ref(hyps, multi_refs, skip_gap, multi_ref_mode, alpha):
    # type: (List[List[str]], List[List[List[str]]], Optional[int], str, float) -> ScoreType
    agg = _build_match_aggregator(multi_ref_mode)
    match = agg.aggregate(
        _rouge_su_summary_level(hyps, refs, skip_gap) for refs in multi_refs)
    return match.to_score(alpha)


"""All ROUGE scores"""


def _rouge_scores_multi_ref(
        hyp,  # type: List[List[str]]
        multi_ref,  # type: List[List[List[str]]]
        rouge_n,  # type: Union[int, Iterable[int]]
        rouge_l,  # type: bool
        rouge_w,  # type: bool
        rouge_w_weight,  # type: float
        rouge_s,  # type: bool
        rouge_su,  # type: bool
        skip_gap,  # type: Optional[int]
        multi_ref_mode,  # type: str
        alpha,  # type: float
):  # type: (...) -> Dict[str, Dict[str, float]]
    if isinstance(rouge_n, int):
        rouge_n = range(1, 1 + rouge_n)
    skip_suffix = str(skip_gap) if skip_gap and skip_gap >= 0 else '*'

    result = {}
    for n in rouge_n:
        result['rouge-{}'.format(n)] = _rouge_n_multi_ref(
            hyp, multi_ref, n, multi_ref_mode, alpha)
    if rouge_l:
        result['rouge-l'] = _rouge_l_multi_ref(
            hyp, multi_ref, multi_ref_mode, alpha)
    if rouge_w:
        result['rouge-w-{}'.format(rouge_w_weight)] = _rouge_w_multi_ref(
            hyp, multi_ref, rouge_w_weight, multi_ref_mode, alpha)
    if rouge_s:
        result['rouge-s{}'.format(skip_suffix)] = _rouge_s_multi_ref(
            hyp, multi_ref, skip_gap, multi_ref_mode, alpha)
    if rouge_su:
        result['rouge-su{}'.format(skip_suffix)] = _rouge_su_multi_ref(
            hyp, multi_ref, skip_gap, multi_ref_mode, alpha)

    return result


class _RougeAggregator(object):
    def aggregate(self, scores):
        # type: (Iterable[RougeType]) -> Union[List[RougeType], RougeType]
        raise NotImplementedError


class _IndividualRougeAggregator(_RougeAggregator):
    def aggregate(self, scores):
        # type: (Iterable[RougeType]) -> List[RougeType]
        return list(scores)


class _AverageRougeAggregator(_RougeAggregator):
    def __init__(self, alpha):
        self.alpha = alpha

    def aggregate(self, scores):
        # type: (Iterable[RougeType]) -> RougeType
        scores = list(scores)
        if len(scores) == 0:
            return {}

        results = {}
        for key in scores[0].keys():
            results[key] = self.average_score(score[key] for score in scores)

        return results

    def average_score(self, scores):
        # type: (Iterable[ScoreType]) -> ScoreType
        total_p = 0
        total_r = 0
        count = 0
        for score in scores:
            total_p += score['p']
            total_r += score['r']
            count += 1
        precision = _div_or_zero(total_p, count)
        recall = _div_or_zero(total_r, count)
        fscore = _f_score(precision, recall, self.alpha)
        return _format_score(fscore, precision, recall)


def _build_rouge_aggregator(mode, alpha):
    # type: (str, float) -> _RougeAggregator
    if mode == 'individual':
        return _IndividualRougeAggregator()
    if mode == 'average':
        return _AverageRougeAggregator(alpha)

    raise ValueError(
        'Invalid mode {}: expected (individual, average)'.format(mode))


class PyRouge(object):
    """Compute ROUGE scores between multiple hypothesis and reference summaries.

    :param rouge_n: Compute N-gram co-occurrence (ROUGE-N). Given an integer N,
        compute ROUGE-1 to ROUGE-N. Given a list of integers, compute ROUGE-N if
        N is on the list.
    :param rouge_l: If true, compute longest common subsequence (LCS)
        co-occurrence (ROUGE-L).
    :param rouge_w: If true, compute Weighted-LCS (WLCS) co-occurrence
        (ROUGE-W).
    :param rouge_w_weight: The weight w of the weighting function
        :math:`f(x) = x^w` to emphasize consecutive matches in ROUGE-W.
    :param rouge_s: If true, compute skip-bigram co-occurrence (ROUGE-S).
    :param rouge_su: If true, compute skip-bigram with unigram co-occurrence
        (ROUGE-SU).
    :param skip_gap: The maximum gap between two words in skip-bigram.
    :param multi_ref_mode: The method to combine the scores between a
        hypothesis and its multiple references. Choose from {average, best}.
    :param alpha: The balance factor between recall and precision. Favors recall
        if close to 1, precision if close to 0.
    :param mode: The method to combine the scores on multiple documents.
        Choose from {average, individual}.

    Example:
    ::

        >>> from rouge_metric import PyRouge
        >>> hypotheses = ['Police killed the gunman'.lower()]
        >>> references = [['The gunman killed the policeman'.lower()]]
        >>> PyRouge().evaluate(hypotheses, references)
        {
            'rouge-1': {'r': 0.6, 'p': 0.75, 'f': 0.666666667},
            'rouge-2': {'r': 0.5, 'p': 0.666666667, 'f': 0.571428571},
            'rouge-l': {'r': 0.4, 'p': 0.5, 'f': 0.444444444}
        }
        >>> hypotheses = [['Police killed the gunman'.lower().split()]]
        >>> references = [[['The gunman killed the policeman'.lower().split()]]]
        >>> PyRouge().evaluate_tokenized(hypotheses, references)
        {
            'rouge-1': {'r': 0.6, 'p': 0.75, 'f': 0.666666667},
            'rouge-2': {'r': 0.5, 'p': 0.666666667, 'f': 0.571428571},
            'rouge-l': {'r': 0.4, 'p': 0.5, 'f': 0.444444444}
        }
    """

    def __init__(self,
                 rouge_n=(1, 2),  # type: Union[int, Iterable[int]]
                 rouge_l=True,  # type: bool
                 rouge_w=False,  # type: bool
                 rouge_w_weight=1.2,  # type: float
                 rouge_s=False,  # type: bool
                 rouge_su=False,  # type: bool
                 skip_gap=None,  # type: Optional[int]
                 multi_ref_mode='average',  # type: str
                 alpha=0.5,  # type: float
                 mode='average',  # type: str
                 ):
        self.rouge_n = rouge_n
        self.rouge_l = rouge_l
        self.rouge_w = rouge_w
        self.rouge_w_weight = rouge_w_weight
        self.rouge_s = rouge_s
        self.rouge_su = rouge_su
        self.skip_gap = skip_gap
        self.multi_ref_mode = multi_ref_mode
        self.alpha = alpha
        self.mode = mode

    @staticmethod
    def _default_sentencizer(text):
        # type: (str) -> List[str]
        return text.split('\n')

    @staticmethod
    def _default_tokenizer(sent):
        # type: (str) -> List[str]
        return sent.split()

    def evaluate_tokenized(
            self,
            hypotheses,  # type: List[List[List[str]]]
            multi_references,  # type: List[List[List[List[str]]]]
    ):
        # type: (...) -> Union[RougeType, List[RougeType]]
        """Compute ROUGE scores between tokenized hypotheses and references.

        Multiple reference summaries can be specified for a hypothesis summary.
        The input should follow the below format so that we know how to match a
        hypothesis with its references.
        ::

            hypotheses = [
                doc1_hyp_summary,   # Hypothesis summary for document 1
                doc2_hyp_summary,   # Hypothesis summary for document 2
                ...
            ]
            multi_references = [
                [
                    doc1_ref1_summary,  # Reference summary 1 for document 1
                    doc1_ref2_summary,  # Reference summary 2 for document 1
                    ...
                ],
                [
                    doc2_ref1_summary,  # Reference summary 1 for document 2
                    doc2_ref2_summary,  # Reference summary 2 for document 2
                    ...
                ],
            ]

        Note that a summary is represented by a list of sentences, and a
        sentence is represented by a list of tokens. A token is a basic element
        here, represented by a ``str``. i.e.,
        ::

            summary = [
                [sent1_token1, sent1_token2, ...],  # sentence 1
                [sent2_token1, sent2_token2, ...],  # sentence 2
            ]

        :param hypotheses: A list of predicted summaries for multiple documents.
            Each summary contains multiple sentences, and each sentence contains
            multiple tokens.
        :param multi_references: A list of gold standard summaries for multiple
            documents. Each document corresponds to multiple reference
            summaries. Each summary contains multiple sentences, and each
            sentence contains multiple tokens.
        :return: All computed ROUGE scores.
        """
        if len(hypotheses) != len(multi_references):
            raise ValueError('Hypotheses and references must be the same size')

        aggregator = _build_rouge_aggregator(self.mode, self.alpha)

        result = aggregator.aggregate(
            _rouge_scores_multi_ref(
                hyp, multi_ref, self.rouge_n, self.rouge_l, self.rouge_w,
                self.rouge_w_weight, self.rouge_s, self.rouge_su, self.skip_gap,
                self.multi_ref_mode, self.alpha
            ) for hyp, multi_ref in zip(hypotheses, multi_references)
        )
        return result

    def evaluate(self,
                 hypotheses,  # type: List[str]
                 multi_references,  # type: List[List[str]]
                 sentencizer=None,  # type: Optional[Callable[[str], List[str]]]
                 tokenizer=None  # type: Optional[Callable[[str], List[str]]]
                 ):
        # type: (...) -> Union[RougeType, List[RougeType]]
        """Compute ROUGE scores between hypothesis and reference summaries.

        The hypotheses and multi_references should follow the below format.
        ::

            hypotheses = [summary1, summary2, ...]
            multi_references = [
                [summary1_ref1, summary1_ref2, ...],
                [summary2_ref1, summary2_ref2, ...],
                ...
            ]

        A summary here is a ``str`` with multiple lines, separated by ``\\n``.
        Each line represents a sentence.

        :param hypotheses: A list of hypothesis summaries.
        :param multi_references: A double list of reference summaries.
        :param sentencizer: A function to split a paragraph into sentences.
        :param tokenizer: A function to split a sentence into tokens.
        :return: All computed ROUGE scores.
        """
        if sentencizer is None:
            sentencizer = self._default_sentencizer
        if tokenizer is None:
            tokenizer = self._default_tokenizer
        tokenized_hyp = [[tokenizer(sent) for sent in sentencizer(hyp)]
                         for hyp in hypotheses]
        tokenized_multi_ref = [[[tokenizer(sent) for sent in sentencizer(ref)]
                                for ref in multi_ref]
                               for multi_ref in multi_references]
        return self.evaluate_tokenized(tokenized_hyp, tokenized_multi_ref)
