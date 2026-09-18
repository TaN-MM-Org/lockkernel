"""The exponent is set by the kernel tail, and by nothing else."""
import mpmath as mp
import pytest

from lockkernel import kernels as K, lineshapes as L, parametric as P
from lockkernel.kernels import predicted_beta

mp.mp.dps = 25
EXPS = (-3, -4, -5, -6)


@pytest.mark.parametrize("s,tol", [(1.5, 5e-3), (1.8, 1e-3), (2.0, 1e-4),
                                   (2.2, 1e-3), (2.5, 2e-3), (4.0, 1e-4),
                                   (6.0, 1e-6)])
def test_beta_of_s(s, tol):
    """beta(s) = 1/(s-1) below the crossover and 1/2 above it.

    The marginal point s = 3 is excluded: the crossover carries logarithmic
    corrections, so the measured exponent approaches 1/2 only slowly there.
    """
    beta = P.extract_beta(L.gaussian(1.0), K.power_tail(s), EXPS)[-1]
    assert abs(float(beta) - predicted_beta(s)) / predicted_beta(s) < tol


def test_marginal_point_approaches_one_half_from_above():
    """At s = 3 the exponent is 1/2 with logarithmic corrections."""
    beta = float(P.extract_beta(L.gaussian(1.0), K.power_tail(3.0), EXPS)[-1])
    assert 0.5 < beta < 0.56


@pytest.mark.parametrize("line", [L.lorentzian(1.0), L.gaussian(1.0),
                                  L.student_t(3.0, 0.5)])
def test_conservative_kernel_gives_one(line):
    beta = P.extract_beta(line, K.conservative(), EXPS)[-1]
    assert abs(float(beta) - 1.0) < 1e-4


@pytest.mark.parametrize("line", [L.lorentzian(1.0), L.gaussian(1.0),
                                  L.student_t(3.0, 0.5)])
def test_compact_and_fast_kernels_give_one_half(line):
    """Compact support and faster than algebraic decay share the class."""
    for kern in (K.kuramoto(), K.gaussian_kernel()):
        beta = P.extract_beta(line, kern, EXPS)[-1]
        assert abs(float(beta) - 0.5) < 1e-4


def test_exponent_is_independent_of_the_line():
    """Three very different lines give the same exponent at fixed kernel."""
    vals = [float(P.extract_beta(line, K.power_tail(2.5), EXPS)[-1])
            for line in (L.lorentzian(1.0), L.gaussian(1.0), L.student_t(3.0, 0.5))]
    assert max(vals) - min(vals) < 1e-3


@pytest.mark.parametrize("s,tol", [(1.8, 1e-4), (2.0, 1e-5), (2.2, 1e-4)])
def test_general_amplitude(s, tol):
    """The amplitude formula holds along the line, not only at s = 2."""
    for line in (L.gaussian(1.0), L.lorentzian(1.0), L.student_t(3.0, 0.5)):
        kern = K.power_tail(s)
        A = P.amplitude_general(line, kern)
        _, R, eps = P.branch_point(line, kern, mp.mpf(10) ** -7)
        assert abs(A - R / eps ** (1 / (mp.mpf(s) - 1))) / A < tol


def test_general_amplitude_reduces_to_the_s2_form():
    """At s = 2 the general amplitude is pi p(0)^2 / c.

    The two agree to the accuracy of the less accurate one, which is the c
    based form: it evaluates p(0) - p(delta) near the origin, a cancellation,
    where the general form integrates p' instead and has none.  On a Gaussian
    line the general form reproduces pi/2 to one part in 1e12 and the c based
    form to two parts in 1e7, so the tolerance here is set by the latter.
    """
    for line in (L.gaussian(1.0), L.lorentzian(1.0)):
        a1 = P.amplitude_general(line, K.conservative())
        a2 = P.amplitude(line)
        assert abs(a1 - a2) / a2 < 1e-6


def test_tail_integral_diverges_outside_the_window():
    with pytest.raises(ValueError):
        P.tail_integral(L.gaussian(1.0), 3.0)
    with pytest.raises(ValueError):
        P.tail_integral(L.gaussian(1.0), 1.0)


@pytest.mark.parametrize("gamma,base,want", [
    (3.4, "kuramoto", 1.4), (3.8, "kuramoto", 1.8), (4.4, "kuramoto", 2.4),
    (3.4, "conservative", 1.4), (4.4, "conservative", 2.0),
])
def test_coupling_disorder_kernel_tail(gamma, base, want):
    """Averaging over a heavy tailed coupling distribution softens the tail."""
    b = K.kuramoto() if base == "kuramoto" else K.conservative()
    ker = K.heterogeneous(b, gamma)
    assert abs(ker.tail - want) < 1e-9
    v1, v2 = mp.mpf(5000), mp.mpf(50000)
    s_meas = -mp.log(ker.W(v2) / ker.W(v1)) / mp.log(v2 / v1)
    assert abs(float(s_meas) - want) < 1.5e-2


def test_recovers_the_published_network_exponent():
    """beta = 1/(gamma-3) for overdamped oscillators with disordered coupling.

    This is the published scale free result; the framework must reproduce it,
    which is the strongest available check on the kernel-tail formulation.
    """
    ker = K.heterogeneous(K.kuramoto(), 3.8)
    beta = float(P.extract_beta(L.gaussian(1.0), ker, (-3, -4, -5, -6))[-1])
    assert abs(beta - 1.0 / (3.8 - 3.0)) < 5e-3
