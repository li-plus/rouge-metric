# ROUGE Metric

[![PyPI](https://img.shields.io/pypi/v/rouge-metric)](https://pypi.org/project/rouge-metric/)
[![UnitTest](https://github.com/li-plus/rouge-metric/workflows/UnitTest/badge.svg?branch=master)](https://github.com/li-plus/rouge-metric/actions)
[![codecov](https://codecov.io/gh/li-plus/rouge-metric/branch/master/graph/badge.svg)](https://codecov.io/gh/li-plus/rouge-metric)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/li-plus/rouge-metric/blob/master/LICENSE)

A fast Python implementation of full [ROUGE](https://www.aclweb.org/anthology/W04-1013/) metric for automatic summarization evaluation. A Python wrapper of the official `ROUGE-1.5.5.pl` Perl script is also available.

## Features

For the Perl script wrapper:

+ Easy to install: No need to manually download and configure the Perl scripts. It works as long as Perl is installed.
+ Cross platform: Support Linux, macOS and Windows machines.
+ Elegant CLI and API: A user-friendly API and a command line tool are available

For the Python implementation:

+ Full ROUGE support: Implemented ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S and ROUGE-SU metrics, with multi-reference evaluation support.
+ High speed: Pure Python implementation without invoking another process.
+ Correctness: Produce the same results as `ROUGE-1.5.5.pl` on all ROUGE scores on single document scenarios. The multi-document results might be slightly different, since we do not adopt bootstrap resampling.
+ Flexible and multi-lingual: We only focus on the language-agnostic tokens, and treat a sentence as a list of tokens. The language-aware pre-processing and tokenization are the freedom of user implementation. You may use different method to tokenize different languages, such as `nltk` for English and `jieba` for Chinese.

## Installation

Install a stable version from PyPI.

```sh
pip install rouge-metric
```

Or install the latest version from GitHub.

```sh
pip install git+https://github.com/li-plus/rouge-metric.git@master
```

For Windows users who want to use the `ROUGE-1.5.5.pl` script, please install [Strawberry Perl](http://strawberryperl.com/) and add its binary folder to `PATH`.

## Quick Start

### With Command Line Tool

Basic usage:

```sh
rouge-metric [options] hypothesis reference [reference ...]
```

where the options are almost the same as the `ROUGE-1.5.5.pl` script. Run `rouge-metric -h` for more details.

For single document with single reference, specify two files.

```sh
rouge-metric sample/hypotheses/summary1.txt sample/references/summary1.1.txt -n 2 -w 1.2 -U -2 4
```

For single document with multiple references, specify a hypothesis file and several reference files.

```sh
rouge-metric sample/hypotheses/summary1.txt sample/references/summary1.1.txt sample/references/summary1.2.txt -n 2 -w 1.2 -U -2 4
```

For multiple documents with multiple references, specify two folders.

```sh
rouge-metric sample/hypotheses/ sample/references/ -n 2 -w 1.2 -U -2 4
```

It directly calls the `ROUGE-1.5.5.pl` script and you get the original output.

```
---------------------------------------------
A ROUGE-1 Average_R: 0.51822 (95%-conf.int. 0.42105 - 0.61538)
A ROUGE-1 Average_P: 0.55556 (95%-conf.int. 0.44444 - 0.66667)
A ROUGE-1 Average_F: 0.53622 (95%-conf.int. 0.43243 - 0.64000)
---------------------------------------------
A ROUGE-2 Average_R: 0.19519 (95%-conf.int. 0.11765 - 0.27273)
A ROUGE-2 Average_P: 0.21250 (95%-conf.int. 0.12500 - 0.30000)
A ROUGE-2 Average_F: 0.20346 (95%-conf.int. 0.12121 - 0.28572)
---------------------------------------------
A ROUGE-L Average_R: 0.51822 (95%-conf.int. 0.42105 - 0.61538)
A ROUGE-L Average_P: 0.55556 (95%-conf.int. 0.44444 - 0.66667)
A ROUGE-L Average_F: 0.53622 (95%-conf.int. 0.43243 - 0.64000)
---------------------------------------------
A ROUGE-W-1.2 Average_R: 0.33608 (95%-conf.int. 0.26618 - 0.40599)
A ROUGE-W-1.2 Average_P: 0.47348 (95%-conf.int. 0.38525 - 0.56172)
A ROUGE-W-1.2 Average_F: 0.39308 (95%-conf.int. 0.31483 - 0.47132)
---------------------------------------------
A ROUGE-S4 Average_R: 0.25495 (95%-conf.int. 0.13846 - 0.37143)
A ROUGE-S4 Average_P: 0.29167 (95%-conf.int. 0.15000 - 0.43333)
A ROUGE-S4 Average_F: 0.27200 (95%-conf.int. 0.14400 - 0.40000)
---------------------------------------------
A ROUGE-SU4 Average_R: 0.31495 (95%-conf.int. 0.19512 - 0.43478)
A ROUGE-SU4 Average_P: 0.35527 (95%-conf.int. 0.21053 - 0.50000)
A ROUGE-SU4 Average_F: 0.33382 (95%-conf.int. 0.20253 - 0.46511)
```

### With Perl Script API

Besides the command line tool, you may also use `ROUGE-1.5.5.pl` programmatically. Note that it is only for English corpus. For non-English summaries, use the Python implementation instead, or convert the tokens to integers separated by space before evaluation.

```python
from rouge_metric import PerlRouge

rouge = PerlRouge(rouge_n_max=3, rouge_l=True, rouge_w=True,
    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

# Load summary results and evaluate
hypotheses = [
    'how are you\ni am fine',                       # document 1: hypothesis
    'it is fine today\nwe won the football game',   # document 2: hypothesis
]
references = [[
    'how do you do\nfine thanks',   # document 1: reference 1
    'how old are you\ni am three',  # document 1: reference 2
], [
    'it is sunny today\nlet us go for a walk',  # document 2: reference 1
    'it is a terrible day\nwe lost the game',   # document 2: reference 2
]]

scores = rouge.evaluate(hypotheses, references)
print(scores)
```

The output is like

```
{
    'rouge-1': {
        'r': 0.51822, 'r_conf_int': (0.42105, 0.61538),
        'p': 0.55556, 'p_conf_int': (0.44444, 0.66667),
        'f': 0.53622, 'f_conf_int': (0.43243, 0.64)
    },
    'rouge-2': {...}, 'rouge-3': {...}, 'rouge-l': {...},
    'rouge-w-1.2': {...}, 'rouge-s4': {...}, 'rouge-su4': {...}
}
```

You may also evaluate summaries from existing files.

```python
from rouge_metric import PerlRouge

hypothesis_dir = 'sample/hypotheses'
reference_dir = 'sample/references'
scores = PerlRouge().evaluate_from_files(hypothesis_dir, reference_dir)
print(scores)
```

### With Python Implementation

A fast Python implementation is also available. It has similar API and supports multiple languages.

```python
from rouge_metric import PyRouge

# Load summary results
hypotheses = [
    'how are you\ni am fine',  # document 1: hypothesis
    'it is fine today\nwe won the football game',  # document 2: hypothesis
]
references = [[
    'how do you do\nfine thanks',  # document 1: reference 1
    'how old are you\ni am three',  # document 1: reference 2
], [
    'it is sunny today\nlet us go for a walk',  # document 2: reference 1
    'it is a terrible day\nwe lost the game',  # document 2: reference 2
]]

# Evaluate document-wise ROUGE scores
rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
scores = rouge.evaluate(hypotheses, references)
print(scores)
```

The output is like

```
{
    'rouge-1': {
        'r': 0.5182186234817814,
        'p': 0.5555555555555556,
        'f': 0.5362379555927943
    },
    'rouge-2': {...}, 'rouge-4': {...}, 'rouge-l': {...},
    'rouge-w-1.2': {...}, 'rouge-s4': {...}, 'rouge-su4': {...}
}
```

By default, sentences are separated by `'\n'` and tokens are separated by white space in a document. This tokenization process can be further customized. For example,

```python
from rouge_metric import PyRouge

# Pre-process and tokenize the summaries as you like
hypotheses = [
    ['how are you'.split(), 'i am fine'.split()],                       # document 1: hypothesis
    ['it is fine today'.split(), 'we won the football game'.split()],   # document 2: hypothesis
]
references = [[
    ['how do you do'.split(), 'fine thanks'.split()],   # document 1: reference 1
    ['how old are you'.split(), 'i am three'.split()],  # document 1: reference 2
], [
    ['it is sunny today'.split(), 'let us go for a walk'.split()],  # document 2: reference 1
    ['it is a terrible day'.split(), 'we lost the game'.split()],   # document 2: reference 2
]]

# Evaluate on tokenized documents
rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
scores = rouge.evaluate_tokenized(hypotheses, references)
print(scores)
```

## License

This project is under [MIT License](https://github.com/li-plus/rouge-metric/blob/master/LICENSE).
