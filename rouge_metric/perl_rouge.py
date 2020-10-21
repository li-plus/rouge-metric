import os
import re
import shutil
import subprocess
import sys
from glob import glob
from tempfile import mkdtemp
from typing import Dict, List, Optional

from rouge_metric import perl_cmd

if sys.version_info < (3,):
    def makedirs(name, mode=0o777, exist_ok=False):
        if not os.path.isdir(name):
            os.makedirs(name, mode)
        else:
            if not exist_ok:
                raise
else:
    from os import makedirs


class PerlRouge(object):
    """A Python wrapper for the ROUGE-1.5.5.pl script.

    :param rouge_n_max: Compute ROUGE-N from N=1 to ``rouge_n_max``
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
    :param alpha: The balance factor between recall and precision. Favors
        recall if close to 1, precision if close to 0.
    :param stemming: If true, stem summaries using Porter stemmer.
    :param remove_stopwords: Remove stopwords in summaries before evaluation.
    :param word_limit: Only use the first n words for evaluation.
    :param byte_limit: Only use the first n bytes for evaluation.
    :param confidence: The confidence level (%) for the confidence interval.
    :param resampling: Number of sampling points in bootstrap resampling.
    :param temp_dir: The directory to hold temporary files.
    :param clean_up: If true, clean up temporary files after evaluation.

    Example:
    ::

        >>> from rouge_metric import PerlRouge
        >>> hypotheses = ['Police killed the gunman']
        >>> references = [['The gunman killed the policeman']]
        >>> PerlRouge().evaluate(hypotheses, references)
        {
            'rouge-1': {
                'r': 0.6, 'r_conf_int': (0.6, 0.6),
                'p': 0.75, 'p_conf_int': (0.75, 0.75),
                'f': 0.66667, 'f_conf_int': (0.66667, 0.66667)
            },
            'rouge-2': {
                'r': 0.5, 'r_conf_int': (0.5, 0.5),
                'p': 0.66667, 'p_conf_int': (0.66667, 0.66667),
                'f': 0.57143, 'f_conf_int': (0.57143, 0.57143)
            },
            'rouge-l': {
                'r': 0.4, 'r_conf_int': (0.4, 0.4),
                'p': 0.5, 'p_conf_int': (0.5, 0.5),
                'f': 0.44444, 'f_conf_int': (0.44444, 0.44444)
            }
        }

    .. warning::

        Only for summaries in English. For non-English summaries, use
        :class:`rouge_metric.evaluate` instead, or convert the tokens to
        integers separated by space before evaluation.
    """

    def __init__(self,
                 rouge_n_max=2,  # type: int
                 rouge_l=True,  # type: bool
                 rouge_w=False,  # type: bool
                 rouge_w_weight=1.2,  # type: float
                 rouge_s=False,  # type: bool
                 rouge_su=False,  # type: bool
                 skip_gap=None,  # type: Optional[int]
                 multi_ref_mode='average',  # type: str
                 alpha=0.5,  # type: float
                 stemming=False,  # type: bool
                 remove_stopwords=False,  # type: bool
                 word_limit=None,  # type: Optional[int]
                 byte_limit=None,  # type: Optional[int]
                 confidence=95,  # type: int
                 resampling=1000,  # type: int
                 temp_dir='./.rouge_metric/',  # type: Optional[str]
                 clean_up=True  # type: bool
                 ):
        perl_cmd.create_wordnet_db()
        self.rouge_n_max = rouge_n_max
        self.rouge_l = rouge_l
        self.rouge_w = rouge_w
        self.rouge_w_weight = rouge_w_weight
        self.rouge_s = rouge_s
        self.rouge_su = rouge_su
        self.skip_gap = skip_gap
        self.multi_ref_mode = multi_ref_mode
        self.alpha = alpha
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self.word_limit = word_limit
        self.byte_limit = byte_limit
        self.confidence = confidence
        self.resampling = resampling
        self.temp_root = temp_dir
        self.clean_up = clean_up

    @staticmethod
    def _write_summaries(summaries, references, hyp_dir, ref_dir):
        # type: (List[str], List[List[str]], str, str) -> None
        makedirs(hyp_dir, exist_ok=True)
        makedirs(ref_dir, exist_ok=True)

        for i, hyp in enumerate(summaries):
            hyp_path = os.path.join(hyp_dir, '{}.txt'.format(i))
            with open(hyp_path, 'w') as f:
                f.write(hyp)

        for i, multi_ref in enumerate(references):
            for j, ref in enumerate(multi_ref):
                ref_path = os.path.join(ref_dir, '{}.{}.txt'.format(i, j))
                with open(ref_path, 'w') as f:
                    f.write(ref)

    @staticmethod
    def _write_config(config_path, hyp_dir, ref_dir):
        # type: (str, str, str) -> None
        xml = '<ROUGE-EVAL version="1.5.5">'
        for n, hyp in enumerate(glob(os.path.join(hyp_dir, '*'))):
            if not os.path.isfile(hyp):
                continue
            hyp_name = os.path.basename(hyp)
            hyp_stem, _ = os.path.splitext(hyp_name)

            hyp_str = '<P ID="{}">{}</P>'.format('A', hyp_name)

            ref_paths = glob(os.path.join(ref_dir, '{}.*'.format(hyp_stem)))
            ref_paths = [ref for ref in ref_paths if os.path.isfile(ref)]
            if not ref_paths:
                raise RuntimeError('Reference not found for {}'.format(hyp))
            ref_str = '\n'.join(
                '<M ID="{}">{}</M>'.format(idx, os.path.basename(ref))
                for idx, ref in enumerate(ref_paths))

            xml += """
<EVAL ID="{eval_id}">
    <MODEL-ROOT>{model_root}</MODEL-ROOT>
    <PEER-ROOT>{peer_root}</PEER-ROOT>
    <INPUT-FORMAT TYPE="SPL">
    </INPUT-FORMAT>
    <PEERS>
        {peers}
    </PEERS>
    <MODELS>
        {models}
    </MODELS>
</EVAL>""".format(eval_id=n + 1, model_root=ref_dir, peer_root=hyp_dir,
                  peers=hyp_str, models=ref_str)
        xml += '</ROUGE-EVAL>'

        with open(config_path, 'w') as f:
            f.write(xml)

    @staticmethod
    def _parse_output(output):
        # type: (str) -> Dict[str, Dict[str, float]]
        result = {}
        # A ROUGE-SU4 Average_P: 1.00000 (95%-conf.int. 1.00000 - 1.00000)
        pattern = (r'A (ROUGE-.+?) Average_([FPR]): ([\d.]+) '
                   r'\(.+?-conf.int. ([\d.]+) - ([\d.]+)\)')
        lines = re.findall(pattern, output)
        for rouge, metric, value, conf_begin, conf_end in lines:
            rouge = rouge.lower()
            metric = metric.lower()
            value = float(value)
            conf_int = (float(conf_begin), float(conf_end))

            result.setdefault(rouge, {})
            result[rouge][metric] = value
            result[rouge]['{}_conf_int'.format(metric)] = conf_int
        return result

    def evaluate(self, hypotheses, multi_references):
        # type: (List[str], List[List[str]]) -> Dict[str, Dict[str, float]]
        """Compute ROUGE scores between hypothesis and reference summaries.

        The hypotheses and multi_references should follow the below format.
        ::

            hypotheses = [summary1, summary2, ...]
            multi_references = [
                [summary1_ref1, summary1_ref2, ...],
                [summary2_ref1, summary2_ref2, ...],
                ...
            ]

        Since the ROUGE-1.5.5.pl script will tokenize sentences before
        evaluation, the summary here is a ``str`` with multiple lines, separated
        by ``\\n``. Each line represents a sentence.

        :param hypotheses: A list of hypothesis summaries.
        :param multi_references: A double list of reference summaries.
        :return: All computed ROUGE scores.
        """
        if len(hypotheses) != len(multi_references):
            raise ValueError('Hypotheses and references must be the same size')
        makedirs(self.temp_root, exist_ok=True)
        temp_dir = mkdtemp(dir=self.temp_root)
        hyp_dir = os.path.join(temp_dir, 'hyp')
        ref_dir = os.path.join(temp_dir, 'ref')
        self._write_summaries(hypotheses, multi_references, hyp_dir, ref_dir)
        result = self.evaluate_from_files(hyp_dir, ref_dir)
        if self.clean_up:
            shutil.rmtree(temp_dir)
        return result

    def evaluate_from_files(self, hypothesis_dir, reference_dir):
        # type: (str, str) -> Dict[str, Dict[str, float]]
        """Compute ROUGE scores from existing files.

        :param hypothesis_dir: The directory containing hypothesis summaries.
            Example hypothesis file names:

            * summary1.txt
            * summary2.txt
            * ...

        :param reference_dir: The directory containing reference summaries.
            To match the hypothesis and reference, the basename of the
            hypothesis file should be the prefix of the corresponding reference
            file name. Example reference file names:

            * summary1.1.txt
            * summary1.2.txt
            * ...
            * summary2.1.txt
            * summary2.2.txt
            * ...

        :return: All computed ROUGE scores.
        """
        makedirs(self.temp_root, exist_ok=True)
        temp_dir = mkdtemp(dir=self.temp_root)
        config_path = os.path.join(temp_dir, 'config.xml')
        self._write_config(config_path, hypothesis_dir, reference_dir)
        cmd = perl_cmd.get_command(
            config_path, self.rouge_n_max, self.rouge_l, self.rouge_w,
            self.rouge_w_weight, self.rouge_s, self.rouge_su, self.skip_gap,
            self.alpha, self.stemming, self.remove_stopwords, self.confidence,
            self.multi_ref_mode, self.word_limit, self.byte_limit,
            self.resampling
        )
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        result = self._parse_output(output.decode('utf-8'))
        if self.clean_up:
            shutil.rmtree(temp_dir)
        return result
