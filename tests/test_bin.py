import subprocess

import pytest


def test_bin():
    # single-document & single-reference
    cmd = 'rouge-metric sample/hypotheses/summary1.txt sample/references/summary1.1.txt -n 2 -w 1.2 -U -2 4'.split()
    subprocess.check_call(cmd)
    # single-document & multi-reference
    cmd = 'rouge-metric sample/hypotheses/summary1.txt sample/references/summary1.1.txt sample/references/summary1.2.txt -n 2 -w 1.2 -U -2 4'.split()
    subprocess.check_call(cmd)
    # multi-document & multi-reference
    cmd = 'rouge-metric sample/hypotheses/ sample/references/ -n 2 -w 1.2 -U -2 4'.split()
    subprocess.check_call(cmd)
    # no options
    cmd = 'rouge-metric sample/hypotheses/ sample/references/'.split()
    subprocess.check_call(cmd)
    # all options
    cmd = 'rouge-metric sample/hypotheses/ sample/references/ -2 2 -u -c 90 -d -f B -b 100 -m -p 0.25 -s -t 1 -r 100 -v -x'.split()
    subprocess.check_call(cmd)

    # error cases
    with pytest.raises(subprocess.CalledProcessError):
        cmd = 'rouge-metric abc abc'.split()
        subprocess.check_call(cmd)
    with pytest.raises(subprocess.CalledProcessError):
        cmd = 'rouge-metric sample/hypotheses/ abc'.split()
        subprocess.check_call(cmd)
    with pytest.raises(subprocess.CalledProcessError):
        cmd = 'rouge-metric sample/ sample/ sample/'.split()
        subprocess.check_call(cmd)
