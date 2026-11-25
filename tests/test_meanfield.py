"""The mean field dynamics against the locking law, and the discretization."""
import numpy as np
import pytest

from lockkernel import lineshapes as L
from lockkernel.cumulant import evolve_meanfield
from lockkernel.ensemble import build_system, class_table

FW = 2 * np.pi
CHINC = 0.5 * FW


def contrast(r, M_core=400, T=150.0, npts=500, freefac=8.0):
    sysm = build_system(L.lorentzian(FW), 1e10, r * CHINC, M_core=M_core,
                        M_tail=24, M_spec=8, core_edge=3 * FW,
                        free_factor=freefac, delta_max=2000.0 * FW)
    t_eval = np.linspace(0.0, T, npts)
    s, _ = evolve_meanfield(sysm, T, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    return float(np.mean(coh[t_eval > 0.5 * T]))


def test_discretisation_keeps_the_whole_population():
    """Nothing is discarded: the populations sum to one."""
    for line in (L.lorentzian(1.0), L.gaussian(1.0), L.student_t(3.0, 0.5)):
        _, n, _, sn = class_table(line, M_core=64, M_tail=14, M_spec=8,
                                  free_beyond=20.0, delta_max=2000.0)
        assert abs(n.sum() + sn.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("Omega", [0.1, 0.5, 2.0])
def test_discretisation_reproduces_the_locking_integral(Omega):
    """The class table must reproduce the integral the locking law is built on.

    This is the quantity the discretization exists to represent, so it is what
    the accuracy of the discretization should be measured on, rather than any
    intermediate such as the density at the center.
    """
    import mpmath as mp
    from lockkernel import kernels as K
    from lockkernel import parametric as P

    mp.mp.dps = 20
    line = L.lorentzian(1.0)
    cons = K.conservative()
    delta, n, s_d, s_n = class_table(line, M_core=800, M_tail=24, M_spec=8,
                                     free_beyond=20.0, delta_max=2000.0)
    W = lambda d: 1.0 / (1.0 + (d / Omega) ** 2)
    got = float(n @ W(delta) + s_n @ W(s_d))
    want = float(P.G_of_Omega(line, cons, Omega) * Omega)
    assert abs(got - want) / want < 2e-3


@pytest.mark.parametrize("r", [1.2, 2.0, 3.0])
def test_dynamics_reaches_the_locking_law(r):
    """The conservative mean field settles on R = 1 - 1/r, not near it."""
    assert abs(contrast(r) - (1.0 - 1.0 / r)) < 5e-3


def test_no_synchronisation_below_threshold():
    """Below threshold the contrast decays away instead of settling."""
    assert contrast(0.6) < 0.05


# ---------------------------------------------------------------------------
# New in 1.6.0: the Voigt line in the dynamics.  References: for the
# conservative kernel the locking integral on a Voigt line is Omega G with
# G = pi erfcx(y) / (sigma sqrt(2 pi)), y = (gamma + Omega)/(sigma sqrt 2)
# (the closed form of README example 12, evaluated here by mpmath), and the
# stationary R of the branch at a coupling is `branch_at_coupling`, which the
# tests of test_branch_tools.py hold to that closed form.
# ---------------------------------------------------------------------------

VOIGT = (1.0, 0.3)          # Gaussian and Lorentzian FWHM, as in example 12


@pytest.mark.parametrize("Omega", [0.1, 0.5, 2.0])
def test_voigt_class_table_reproduces_the_locking_integral(Omega):
    """The class table of a Voigt line against the closed form, with the
    table settings of the Lorentzian test above: within 2e-5 (observed
    5e-8, 4e-7 and 3e-6; the Lorentzian line gives about 1e-5)."""
    import mpmath as mp
    line = L.voigt(*VOIGT)
    delta, n, s_d, s_n = class_table(line, M_core=800, M_tail=24, M_spec=8,
                                     free_beyond=20.0, delta_max=2000.0)
    assert abs(n.sum() + s_n.sum() - 1.0) < 1e-12
    assert np.all(np.abs(s_d) > 20.0) and np.all(np.diff(delta) > 0)
    W = lambda d: 1.0 / (1.0 + (d / Omega) ** 2)
    got = float(n @ W(delta) + s_n @ W(s_d))
    with mp.workdps(30):
        sig = mp.mpf(VOIGT[0]) / (2 * mp.sqrt(2 * mp.log(2)))
        y = (mp.mpf(VOIGT[1]) / 2 + Omega) / (sig * mp.sqrt(2))
        want = float(Omega * mp.pi * mp.exp(y ** 2) * mp.erfc(y)
                     / (sig * mp.sqrt(2 * mp.pi)))
    assert abs(got / want - 1) < 2e-5


def _voigt_contrast(r, T=300.0):
    import mpmath as mp
    from lockkernel import kernels as K, parametric as P
    line = L.voigt(*VOIGT)
    with mp.workdps(15):
        chi_c = float(P.threshold(line, K.conservative()))
    sysm = build_system(line, 1e10, r * chi_c, M_core=400, M_tail=24,
                        M_spec=8, core_edge=3 * line.scale, free_factor=8.0,
                        delta_max=2000.0 * line.scale)
    t_eval = np.linspace(0.0, T, 500)
    s, _ = evolve_meanfield(sysm, T, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    return float(np.mean(coh[t_eval > 0.5 * T])), line, chi_c


@pytest.mark.parametrize("r", [2.0, 3.0])
def test_voigt_dynamics_reaches_the_exact_branch(r):
    """Mean-field dynamics on the Voigt class table settles on the R of the
    exact branch at that coupling, within 5e-3 as for the Lorentzian line
    (observed 5e-4 at r = 2 and 1e-5 at r = 3)."""
    import mpmath as mp
    from lockkernel import kernels as K, parametric as P
    got, line, chi_c = _voigt_contrast(r)
    with mp.workdps(15):
        R = float(P.branch_at_coupling(line, K.conservative(), r * chi_c)[1])
    assert abs(got - R) < 5e-3


def test_voigt_no_synchronization_below_threshold():
    assert _voigt_contrast(0.6)[0] < 0.05


# ---------------------------------------------------------------------------
# New in 1.7.0: the two-peaked line in the dynamics.  References: for the
# conservative kernel G(Omega) on two Gaussian peaks of width w at +-a is
# the Voigt closed form pi [Re w(z+) + Re w(z-)] / (2 w sqrt(2 pi)),
# z+- = (+-a + i Omega)/(w sqrt 2), with the Faddeeva function
# w(z) = exp(-z^2) erfc(-iz) (test_onset.py holds G_of_Omega to it), and the
# stationary R of the branch at a coupling (branch_at_coupling above the
# threshold; on the upper branch of a fold, the root of 1/G(Omega) = chiN
# found by mpmath's root finder above the lower turning point).
# ---------------------------------------------------------------------------

def _bimodal_G(sep, Omega):
    import mpmath as mp
    with mp.workdps(30):
        a, Om = mp.mpf(sep) / 2, mp.mpf(Omega)
        fad = lambda z: mp.exp(-z ** 2) * mp.erfc(-1j * z)
        r2 = mp.sqrt(2)
        return mp.pi * (mp.re(fad((a + 1j * Om) / r2))
                        + mp.re(fad((-a + 1j * Om) / r2))) / (2 * mp.sqrt(2 * mp.pi))


@pytest.mark.parametrize("sep", [1.0, 4.0])
def test_bimodal_class_table_reproduces_the_locking_integral(sep):
    """Settings of the Voigt test above: within 5e-5 (measured 6.8e-7,
    4.3e-6 and 2.2e-5 for sep = 1; 1.5e-6, 5.6e-6 and 1.7e-5 for sep = 4)."""
    line = L.bimodal_gaussian(sep)
    delta, n, s_d, s_n = class_table(line, M_core=800, M_tail=24, M_spec=8,
                                     free_beyond=20.0 * line.scale,
                                     delta_max=2000.0 * line.scale)
    assert abs(n.sum() + s_n.sum() - 1.0) < 1e-12
    assert np.all(np.diff(delta) > 0)
    for Omega in (0.1, 0.5, 2.0):
        W = lambda d: 1.0 / (1.0 + (d / Omega) ** 2)
        got = float(n @ W(delta) + s_n @ W(s_d))
        want = float(Omega * _bimodal_G(sep, Omega))
        assert abs(got / want - 1) < 5e-5


def _bimodal_contrast(sep, chiN, T):
    line = L.bimodal_gaussian(sep)
    sysm = build_system(line, 1e10, chiN, M_core=400, M_tail=24, M_spec=8,
                        core_edge=3 * line.scale, free_factor=8.0,
                        delta_max=2000.0 * line.scale)
    t_eval = np.linspace(0.0, T, 500)
    s, _ = evolve_meanfield(sysm, T, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    return float(np.mean(coh[t_eval > 0.5 * T]))


def test_bimodal_dynamics_reaches_the_exact_branch():
    """Two peaks one width apart (a continuous onset, c > 0): mean-field
    dynamics at twice the threshold settles on the exact branch, within
    5e-3 as for the other lines (measured 3.2e-5)."""
    import mpmath as mp
    from lockkernel import kernels as K, parametric as P
    line = L.bimodal_gaussian(1.0)
    with mp.workdps(15):
        chi_c = float(P.threshold(line, K.conservative()))
        R = float(P.branch_at_coupling(line, K.conservative(), 2 * chi_c)[1])
    assert abs(_bimodal_contrast(1.0, 2 * chi_c, 300.0) - R) < 5e-3


@pytest.mark.parametrize("chiN", [4.5, 7.0])
def test_bimodal_dynamics_on_the_upper_branch_of_a_fold(chiN):
    """Two peaks four widths apart (example 8): the branch folds, with
    hysteresis between chiN = 3.4934 and the threshold 5.8956.  Started
    with every spin in phase, mean-field dynamics at chiN = 4.5, below the
    threshold, stays synchronized on the upper branch (R = 0.7218), as at
    7.0 above it (R = 0.8943): within 5e-3 of the exact branch (measured
    1.6e-3, with a residual oscillation of 0.04 at 4.5, and 3e-5)."""
    import mpmath as mp
    from lockkernel import kernels as K, parametric as P
    line, kern = L.bimodal_gaussian(4.0), K.conservative()
    with mp.workdps(15):
        chi = lambda o: 1 / P.G_of_Omega(line, kern, o)
        O = mp.findroot(lambda o: chi(o) - chiN, (2, 9), solver="anderson")
        # on the upper branch chiN rises with Omega (the lower turning
        # point is at Omega = 1.168, chiN = 3.4934; example 8)
        assert chi(O * 0.999) < chiN < chi(O * 1.001) and O > 1.17
        R = float(O * P.G_of_Omega(line, kern, O))
        # the same point from the Voigt closed form of G
        assert abs(R / float(O * _bimodal_G(4.0, O)) - 1) < 1e-12
    assert abs(_bimodal_contrast(4.0, chiN, 600.0) - R) < 5e-3
