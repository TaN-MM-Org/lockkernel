"""The tutorials in examples/ run and print what README.md says they print.

Each script is run as a separate process, and its key numbers are parsed
from the output and checked: against the values printed in the README
(to the digits shown), and against what the numbers must be independently
of the run (the closed form of the power-law window, the true exponents
and kernel masses of the synthetic data).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(name):
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not shipped here")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, check=True, env=env).stdout
    return out


def _num(pattern, text):
    m = re.search(pattern, text)
    assert m, pattern
    return [float(g) for g in m.groups()]


def _documented(name, out):
    """The output is the one quoted at the top of the script and in the
    README's Tutorials section."""
    src = (EXAMPLES / name).read_text()
    doc = src.split('"""')[1]
    readme = EXAMPLES.parent / "README.md"
    text = readme.read_text() if readme.exists() else None
    for line in out.splitlines():
        assert "    " + line in doc, line
        if text is not None:
            assert "\n" + line + "\n" in text, line


def test_tutorial_which_kernel():
    out = _run("tutorial_which_kernel.py")
    _documented("tutorial_which_kernel.py", out)
    chi_c, s_chi = _num(r"chi_c = ([\d.]+) \+- ([\d.]+)", out)
    beta, s_beta = _num(r"beta  = ([\d.]+) \+- ([\d.]+)", out)
    m, s_m = _num(r"m = 1/\(p\(0\) chi_c\) = ([\d.]+) \+- ([\d.]+)", out)
    A, s_A, A_pred = _num(r"fitted A = ([\d.]+) \+- ([\d.]+), Kuramoto "
                          r"prediction ([\d.]+)", out)
    # the synthetic data are the Kuramoto branch: beta = 1/2, m = pi/2
    assert abs(beta - 0.5) < 3 * s_beta and abs(beta - 1.0) > 50 * s_beta
    assert abs(m - 1.5708) < 3 * s_m and abs(m - 3.1416) > 20 * s_m
    assert abs(A - A_pred) < 3 * s_A
    assert "kernel tail refused" in out
    assert (chi_c, s_chi) == (0.87949, 0.0088)


def test_tutorial_plan_and_fit():
    out = _run("tutorial_plan_and_fit.py")
    _documented("tutorial_plan_and_fit.py", out)
    eps_max, closed = _num(r"up to eps = ([\d.]+) \(closed form ([\d.]+)\)", out)
    assert eps_max == closed == 0.0101
    n, = _num(r"plan: (\d+) points", out)
    beta, s_beta, p = _num(r"inside the window:  beta = ([\d.]+) \+- ([\d.]+),"
                           r" chi2 = [\d.]+ for \d+ dof, p = ([\d.e-]+)", out)
    s, s_err = _num(r"kernel tail s = ([\d.]+) \+- ([\d.]+)", out)
    wide, s_wide, p_wide = _num(r"eps = 0.5:    beta = ([\d.]+) \+- ([\d.]+),"
                                r" chi2 = [\d.]+ for \d+ dof, p = ([\d.e-]+)", out)
    assert n == 16 and s_beta <= 0.0071
    assert abs(beta - 1) < 3 * s_beta and p > 0.01
    assert abs(s - 2) < 3 * max(s_err, 0.005)
    # beyond the window the fit is biased and the chi-square test says so
    assert abs(wide - 1) > 10 * s_wide and p_wide < 1e-10


def test_tutorial_whole_branch():
    """New in 1.5.0.  The synthetic data are the exact Kuramoto branch on a
    Gaussian line with k = 0.0371 per volt and R unscaled."""
    out = _run("tutorial_whole_branch.py")
    _documented("tutorial_whole_branch.py", out)
    beta, s_beta, p_pl = _num(r"beta = ([\d.]+) \+- ([\d.]+), chi2 = [\d.]+ "
                              r"for 9 dof, p = ([\d.e-]+)", out)
    p_g, = _num(r"Gaussian line: +chi2 = [\d.]+ for 11 dof, p = ([\d.e-]+)", out)
    p_l, = _num(r"Lorentzian line: chi2 = [\d.]+ for 11 dof, p = ([\d.e-]+)", out)
    k, s_k, k_true = _num(r"k = ([\d.]+) \+- ([\d.]+) per volt \(true ([\d.]+)\)",
                          out)
    chi_c, s_c, c_true = _num(r"threshold chi_c = ([\d.]+) \+- ([\d.]+) V "
                              r"\(true ([\d.]+) V\)", out)
    q, s_q, chi_q, s_cq = _num(r"q = ([\d.]+) \+- ([\d.]+), threshold ([\d.]+) "
                               r"\+- ([\d.]+) V", out)
    # the power law over the whole range is biased and rejected
    assert abs(beta - 0.5) > 10 * s_beta and p_pl < 1e-10
    # the right model is kept, the wrong line rejected
    assert p_g > 0.01 and p_l < 1e-10
    # the calibration and the threshold recover the truth
    assert k_true == 0.0371 and abs(k - k_true) < 3 * max(s_k, 1e-6)
    assert abs(chi_c - c_true) < 3 * s_c and abs(chi_q - c_true) < 3 * s_cq
    assert abs(c_true - 18.2658) < 1e-4
    assert abs(q - 1.0) < 3 * s_q
