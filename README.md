# ROUGE Metric

[![UnitTest](https://github.com/li-plus/rouge-metric/workflows/UnitTest/badge.svg?branch=master)](https://github.com/li-plus/rouge-metric/actions)
[![codecov](https://codecov.io/gh/li-plus/rouge-metric/branch/master/graph/badge.svg)](https://codecov.io/gh/li-plus/rouge-metric)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/li-plus/rouge-metric/blob/master/LICENSE)

A fast Python implementation of full [ROUGE](https://www.aclweb.org/anthology/W04-1013/) metric for automatic summarization evaluation. A Python wrapper of the official `ROUGE-1.5.5.pl` Perl script is also available.

## Features

+ Full ROUGE support: Implemented ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S and ROUGE-SU metrics. Support multiple references for each hypothesis summary.
+ High speed: Pure Python implementation without invoking another process.
+ Correctness: Produce exactly the same results as `ROUGE-1.5.5.pl` on all ROUGE scores on single document scenarios. The results might be slightly different on multi-document, because we directly average the scores across documents while the official script further adopts bootstrap resampling.
+ Flexible and multi-lingual: We only focus on the language-agnostic tokens, and treat a sentence as a list of tokens. The language-aware pre-processing and tokenization are the freedom of user implementation. You may use different method to tokenize different languages, such as `nltk` for English and `jieba` for Chinese.
+ Multi-optional: Besides the Python implementation, we also provide an API to the official Perl script to programmatically evaluate the summary results.

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

### Using Python Implementation

Evaluate the results using pure Python implementation.

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
    'rouge-2': {'r': ..., 'p': ..., 'f': ...},
    'rouge-4': {'r': ..., 'p': ..., 'f': ...},
    'rouge-l': {'r': ..., 'p': ..., 'f': ...},
    'rouge-w-1.2': {'r': ..., 'p': ..., 'f': ...},
    'rouge-s4': {'r': ..., 'p': ..., 'f': ...},
    'rouge-su4': {'r': ..., 'p': ..., 'f': ...}
}
```

By default, sentences are separated by `'\n'` and tokens are separated by white space in a document. This tokenization process can be customized. For example,

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
print(scores)   # The output is the same as above
```

### Using Official Perl Script

Evaluate the results using the `ROUGE-1.5.5.pl` script, which is only for English corpus. For non-English summaries, use the Python implementation instead, or convert the tokens to integers separated by space before evaluation.

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

You can also evaluate summaries from existing files

```python
from rouge_metric import PerlRouge

hypothesis_dir = 'sample/hypotheses'
reference_dir = 'sample/references'
scores = PerlRouge().evaluate_from_files(hypothesis_dir, reference_dir)
print(scores)   # The output is the same as above
```

## License

This project is under [MIT License](https://github.com/li-plus/rouge-metric/blob/master/LICENSE).
