import os
import subprocess
from typing import List, Optional

HERE = os.path.dirname(__file__)
ROUGE_HOME = os.path.join(HERE, 'RELEASE-1.5.5')
ROUGE_EXEC = os.path.join(ROUGE_HOME, 'ROUGE-1.5.5.pl')
ROUGE_DATA_HOME = os.path.join(ROUGE_HOME, 'data')
ROUGE_DB = os.path.join(ROUGE_DATA_HOME, 'WordNet-2.0.exc.db')
ROUGE_WORDNET_DIR = os.path.join(ROUGE_DATA_HOME, 'WordNet-2.0-Exceptions')
ROUGE_BUILD_DB_SCRIPT = os.path.join(ROUGE_WORDNET_DIR, 'buildExeptionDB.pl')
ROUGE_SMART_COMMON_WORDS = os.path.join(ROUGE_DATA_HOME,
                                        'smart_common_words.txt')

CONFIG_FORMATS = ('SEE', 'SPL', 'ISI', 'SIMPLE')

SCORING_FORMULA_MAP = {'average': 'A', 'best': 'B'}

COUNT_SENTENCE = 0
COUNT_TOKEN = 1
COUNT_TOKEN_WITH_RAW_COUNTS = 2
COUNTING_UNITS = (
    COUNT_SENTENCE,
    COUNT_TOKEN,
    COUNT_TOKEN_WITH_RAW_COUNTS,
)

BE_H = 0  # head only scoring (does not applied to Minipar-based BEs).
BE_HM = 1  # head and modifier pair scoring.
BE_HMR = 2  # head, modifier and relation triple scoring.
BE_HM1 = 3  # H and HM scoring (same as HM for Minipar-based BEs).
BE_HMR1 = 4  # HM and HMR scoring (same as HMR for Minipar-based BEs).
BE_HMR2 = 5  # H, HM and HMR scoring (same as HMR for Minipar-based BEs).
BASIC_ELEMENTS = (
    BE_H, BE_HM, BE_HMR,
    BE_HM1, BE_HMR1, BE_HMR2,
)


def create_wordnet_db():
    try:
        subprocess.check_call(['perl', '--version'],
                              stdout=open(os.devnull, 'w'),
                              stderr=subprocess.STDOUT)
    except IOError:
        raise RuntimeError('Perl is not correctly installed on your machine. '
                           'Please make sure its binary is in PATH')

    if not os.path.exists(ROUGE_DB):
        subprocess.call(['perl', ROUGE_BUILD_DB_SCRIPT, ROUGE_WORDNET_DIR,
                         ROUGE_SMART_COMMON_WORDS, ROUGE_DB])


def get_command(
        config_path,  # type: str
        rouge_n_max=None,  # type: Optional[int]
        rouge_l=True,  # type: bool
        rouge_w=False,  # type: bool
        rouge_w_weight=1.2,  # type: float
        rouge_s=False,  # type: bool
        rouge_su=False,  # type: bool
        skip_distance=None,  # type: Optional[int]
        alpha=None,  # type: Optional[float]
        stemming=False,  # type: bool
        remove_stopwords=False,  # type: bool
        confidence=None,  # type: Optional[int]
        scoring_formula=None,  # type: Optional[str]
        word_limit=None,  # type: Optional[int]
        byte_limit=None,  # type: Optional[int]
        resampling_points=None,  # type: Optional[int]
        basic_element=None,  # type: Optional[int]
        print_each_eval=False,  # type: bool
        env=None,  # type: Optional[str]
        counting_unit=None,  # type: Optional[int]
        config_format=None,  # type: Optional[int]
        system_id=None,  # type: Optional[str]
        verbose=False  # type: bool
):  # type: (...) -> List[str]
    """Assemble the command line to invoke the ROUGE-1.5.5.pl perl script.

    :param config_path: The XML configuration files that specifies the path of
        peer and model summaries.
    :param rouge_n_max: Compute ROUGE-N up to `rouge_n_max`. If negative, do not
        compute ROUGE-N.
    :param rouge_l: Whether compute LCS co-occurrence (ROUGE-L).
    :param rouge_w: Whether compute WLCS co-occurrence (ROUGE-W).
    :param rouge_w_weight: Compute ROUGE-W that gives consecutive matches of
        length L in an LCS a weight of 'L^weight' instead of just 'L' as in LCS.
        Typically this is set to 1.2 or other number greater than 1.
    :param rouge_s: Whether compute skip bigram (ROGUE-S) co-occurrence.
    :param rouge_su: Whether compute skip bigram co-occurrence including unigram
        (ROGUE-SU).
    :param skip_distance: The maximum gap between two words (skip bi-gram) in
        ROUGE-S or ROUGE-SU.
    :param alpha: Relative importance of recall and precision. Alpha -> 1 favors
        precision, Alpha -> 0 favors recall.
    :param stemming: Stem both model and system summaries using Porter stemmer.
    :param remove_stopwords: Remove stopwords in model and system summaries
        before computing various statistics.
    :param confidence: Specify CF% (0 <= CF <= 100) confidence interval to
        compute. The default is 95% (i.e. CF=95).
    :param scoring_formula: Method to combine multi-reference results. Choose
        from (average, best).
    :param word_limit: Only use the first n words for evaluation.
    :param byte_limit: Only use the first n bytes for evaluation.
    :param resampling_points: The number of sampling point in bootstrap
        resampling (default is 1000).
    :param basic_element: Compute BE score.

        * H    -> head only scoring (does not applied to Minipar-based BEs).
        * HM   -> head and modifier pair scoring.
        * HMR  -> head, modifier and relation triple scoring.
        * HM1  -> H and HM scoring (same as HM for Minipar-based BEs).
        * HMR1 -> HM and HMR scoring (same as HMR for Minipar-based BEs).
        * HMR2 -> H, HM and HMR scoring (same as HMR for Minipar-based BEs).

    :param print_each_eval: Print per evaluation average score for each system.
    :param env: Specify the directory where the ROUGE data files can be found.
    :param counting_unit: Compute average ROUGE by averaging over the whole test
        corpus instead of sentences (units).

        * 0: use sentence as counting unit
        * 1: use token as counting unit
        * 2: same as 1 but output raw counts instead of precision, recall, \
        and f-measure scores. Useful when computation of the final, precision, \
        recall, and f-measure scores will be conducted later.

    :param config_format: A list of peer-model pair per line in the specified
        format (SEE|SPL|ISI|SIMPLE)
    :param system_id: Specify the system in the config file for evaluation.
        If None, evaluate all systems.
    :param verbose: Print debugging information for diagnostic purpose.
    :return: An executable command line with the given options
    """
    if basic_element is not None and basic_element not in BASIC_ELEMENTS:
        raise ValueError('Invalid basic_element {}: expected {}'.format(
            basic_element, BASIC_ELEMENTS))

    if confidence is not None and not 0 <= confidence <= 100:
        raise ValueError(
            'Invalid confidence {}: expected between [0, 100]'.format(
                confidence))

    if scoring_formula is not None:
        if scoring_formula in SCORING_FORMULA_MAP:
            scoring_formula = SCORING_FORMULA_MAP[scoring_formula]
        else:
            raise ValueError(
                'Invalid scoring_formula {}: expected (average, best)'.format(
                    scoring_formula))

    if counting_unit is not None and counting_unit not in COUNTING_UNITS:
        raise ValueError('Invalid counting_unit {}: expected {}'.format(
            counting_unit, COUNTING_UNITS))

    if config_format is not None and config_format not in CONFIG_FORMATS:
        raise ValueError('Invalid config_format {}: expected {}'.format(
            config_format, CONFIG_FORMATS))

    if rouge_w and not rouge_w_weight >= 1:
        raise ValueError('Invalid rouge_w_weight {}: expected >= 1'.format(
            rouge_w_weight))

    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError('Invalid alpha {}: expected between [0, 1]'.format(
            alpha))

    if word_limit is not None and byte_limit is not None:
        raise ValueError('Cannot specify both word_limit and byte_limit')

    if skip_distance is None:
        skip_distance = -1

    env = env or ROUGE_DATA_HOME

    options = ['perl', ROUGE_EXEC]
    if system_id is None:
        options.append('-a')
    if confidence is not None:
        options.extend(['-c', confidence])
    if print_each_eval:
        options.append('-d')
    options.extend(['-e', env])
    if word_limit is not None:
        options.extend(['-l', word_limit])
    if byte_limit is not None:
        options.extend(['-b', byte_limit])
    if stemming:
        options.append('-m')
    if rouge_n_max is not None:
        options.extend(['-n', rouge_n_max])
    if remove_stopwords:
        options.append('-s')
    if resampling_points is not None:
        options.extend(['-r', resampling_points])
    if rouge_s:
        if rouge_su:
            options.extend(['-2', skip_distance, '-U'])
        else:
            options.extend(['-2', skip_distance])
    else:
        if rouge_su:
            options.extend(['-2', skip_distance, '-u'])
    if basic_element is not None:
        options.extend(['-3', basic_element])
    if rouge_w:
        options.extend(['-w', rouge_w_weight])
    if verbose:
        options.append('-v')
    if not rouge_l:
        options.append('-x')
    if scoring_formula is not None:
        options.extend(['-f', scoring_formula])
    if alpha is not None:
        options.extend(['-p', alpha])
    if counting_unit is not None:
        options.extend(['-t', counting_unit])

    options.append(config_path)

    if system_id is not None:
        options.append(system_id)

    return [str(opt) for opt in options]
