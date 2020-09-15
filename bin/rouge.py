#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
from tempfile import mkdtemp

from rouge_metric import PerlRouge, perl_cmd
from rouge_metric.perl_rouge import makedirs


def main():
    parser = argparse.ArgumentParser(description='Compute ROUGE metrics')

    parser.add_argument('hypothesis', type=str,
                        help='Hypothesis file or directory')
    parser.add_argument('reference', type=str, nargs='+',
                        help='Reference file or directory')

    parser.add_argument('-2', type=int, metavar='SKIP_GAP',
                        help='Compute skip bigram (ROUGE-S) co-occurrence, '
                             'also specify the maximum gap length between two '
                             'words (skip-bigram)')
    parser.add_argument('-u', action='store_true',
                        help='Compute skip bigram as -2 but include unigram, '
                             'i.e. treat unigram as "start-sentence-symbol '
                             'unigram"; -2 has to be specified.')
    parser.add_argument('-U', action='store_true',
                        help='same as -u but also compute regular skip-bigram')
    parser.add_argument('-3', type=str, metavar='BE',
                        choices=['H', 'HM', 'HMR', 'HM1', 'HMR1', 'HMR2'],
                        help='Compute BE score. Currently only SIMPLE BE '
                             'triple format is supported.\n'
                             'H    -> head only scoring (does not applied to '
                             'Minipar-based BEs).\n'
                             'HM   -> head and modifier pair scoring.\n'
                             'HMR  -> head, modifier and relation triple '
                             'scoring.\n'
                             'HM1  -> H and HM scoring (same as HM for '
                             'Minipar-based BEs).\n'
                             'HMR1 -> HM and HMR scoring (same as HMR for '
                             'Minipar-based BEs).\n'
                             'HMR2 -> H, HM and HMR scoring (same as HMR for '
                             'Minipar-based BEs).')
    parser.add_argument('-c', type=int, metavar='CONF_INT',
                        help='Specify CF%% (0 <= CF <= 100) confidence '
                             'interval to compute. The default is 95%% '
                             '(i.e. CF=95).')
    parser.add_argument('-d', action='store_true',
                        help='Print per evaluation average score for each '
                             'system.')
    parser.add_argument('-e', type=str, default=perl_cmd.ROUGE_DATA_HOME,
                        metavar='DATA_HOME',
                        help='Specify where the ROUGE data files can be found')
    parser.add_argument('-f', type=str, choices=['A', 'B'], metavar='FORMULA',
                        help='Select scoring formula: A => average model; '
                             'B => best model')
    parser.add_argument('-b', type=int, metavar='N_BYTES',
                        help='Only use the first n bytes in the system/peer '
                             'summary for the evaluation.')
    parser.add_argument('-l', type=int, metavar='N_WORDS',
                        help='Only use the first n words in the system/peer '
                             'summary for the evaluation.')
    parser.add_argument('-m', action='store_true',
                        help='Stem both model and system summaries using '
                             'Porter stemmer before computing various '
                             'statistics.')
    parser.add_argument('-n', type=int, metavar='MAX_NGRAM',
                        help='Compute ROUGE-N up to max-ngram length.')
    parser.add_argument('-p', type=float, metavar='ALPHA',
                        help='Relative importance of recall and precision '
                             'ROUGE scores. Alpha -> 1 favors precision, '
                             'Alpha -> 0 favors recall.')
    parser.add_argument('-s', action='store_true',
                        help='Remove stopwords in model and system summaries '
                             'before computing various statistics.')
    parser.add_argument('-t', type=int, choices=[0, 1, 2], metavar='UNIT',
                        help='Compute average ROUGE by averaging over the '
                             'whole test corpus instead of sentences (units). '
                             '0: use sentence as counting unit, 1: use token '
                             'as counting unit, 2: same as 1 but output raw '
                             'counts instead of precision, recall, and '
                             'f-measure  scores. 2 is useful when computation '
                             'of the final, precision, recall, and f-measure '
                             'scores will be conducted later.')
    parser.add_argument('-r', type=int, metavar='RESAMPLING',
                        help='Specify the number of sampling point in '
                             'bootstrap resampling (default is 1000).')
    parser.add_argument('-w', type=float, metavar='WEIGHT',
                        help='Compute ROUGE-W that gives consecutive matches '
                             'of length L in an LCS a weight of L^weight '
                             'instead of just L as in LCS. Typically this is '
                             'set to 1.2 or other number greater than 1.')
    parser.add_argument('-v', action='store_true',
                        help='Print debugging information for diagnostic '
                             'purpose.')
    parser.add_argument('-x', action='store_true',
                        help='Do not calculate ROUGE-L.')
    parser.add_argument('-z', type=str, choices=['SEE', 'SPL', 'ISI', 'SIMPLE'],
                        metavar='EVAL_CONFIG',
                        help='ROUGE-eval-config-file is a list of peer-model '
                             'pair per line in the specified format '
                             '(SEE|SPL|ISI|SIMPLE).')
    args = parser.parse_args()

    args = vars(args)
    hyp = args.pop('hypothesis')
    ref = args.pop('reference')

    try:
        if os.path.isfile(hyp):
            for ref_file in ref:
                if not os.path.isfile(ref_file):
                    raise ValueError('When hypothesis is a file, all '
                                     'references must be files.')
        elif os.path.isdir(hyp):
            if not (len(ref) == 1 and os.path.isdir(ref[0])):
                raise ValueError('When hypothesis is a directory, '
                                 'reference must also be a directory')
        else:
            raise ValueError('Hypothesis must be either a file or a directory')
    except ValueError as e:
        print(e)
        exit(1)

    # make temporary directory for config files
    temp_dir = mkdtemp()
    config_path = os.path.join(temp_dir, 'config.xml')

    # assemble command line
    cmd = ['perl', perl_cmd.ROUGE_EXEC, '-a']
    for key, value in args.items():
        if value is not False and value is not None:
            cmd.append('-' + key)
            if value is not True:
                cmd.append(str(value))
    cmd.append(config_path)

    try:
        perl_cmd.create_wordnet_db()
        if os.path.isfile(hyp):
            # create temporary directory
            hyp_dir = os.path.join(temp_dir, 'hyp')
            makedirs(hyp_dir, exist_ok=True)
            ref_dir = os.path.join(temp_dir, 'ref')
            makedirs(ref_dir, exist_ok=True)
            # copy hypothesis file to temp dir
            hyp_base = os.path.basename(hyp)
            shutil.copyfile(hyp, os.path.join(hyp_dir, hyp_base))
            # copy reference files to temp dir
            for ref_file in ref:
                ref_base = os.path.basename(ref_file)
                shutil.copyfile(ref_file, os.path.join(ref_dir, ref_base))
        else:
            hyp_dir = hyp
            ref_dir, = ref
        PerlRouge._write_config(config_path, hyp_dir, ref_dir)
        subprocess.run(cmd)
    except Exception as e:
        print(e)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
