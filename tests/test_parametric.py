"""The parametric solution against the cases that are known in closed form."""
import mpmath as mp
import pytest

from lockkernel import kernels as K, lineshapes as L, parametric as P

mp.mp.dps = 30


def test_kernel_masses():
    assert abs(K.conservative().mass() - mp.pi) < mp.mpf("1e-25")
    assert abs(K.kuramoto().mass() - mp.pi / 2) < mp.mpf("1e-25")
    # W_s has area 2 (pi/s) / sin(pi/s).  The tolerance is looser for the
    # family because a tail as slow as |u|^-1.5 is integrated numerically out
    # to infinity, where the two exact kernels are not.
    for s in (1.5, 2.0, 3.0, 4.0):
        want = 2 * (mp.pi / s) / mp.sin(mp.pi / s)
        assert abs(K.power_tail(s).mass() - want) < mp.mpf("1e-14")


def test_threshold_ratio_is_exactly_two():
    """The Kuramoto threshold is twice the conservative one, for every line."""
    for line in (L.lorentzian(1.0), L.gaussian(1.0), L.student_t(3.0, 0.5)):
        ratio = P.threshold(line, K.kuramoto()) / P.threshold(line, K.conservative())
        assert abs(ratio - 2) < mp.mpf("1e-25")


def test_lorentzian_closed_form():
    """chiN = a + Omega and R = 1 - a/chiN for a Lorentzian of half width a."""
    line, cons = L.lorentzian(1.0), K.conservative()
    a = mp.mpf(1) / 2
    for k in range(1, 7):
        Om = mp.mpf(10) ** -k
        chiN, R, _ = P.branch_point(line, cons, Om)
        assert abs(chiN - (a + Om)) < mp.mpf("1e-28")
        assert abs(R - (1 - a / chiN)) < mp.mpf("1e-28")


def test_kuramoto_closed_form():
    """r = sqrt(1 - Kc/K) for the Kuramoto kernel on a Lorentzian line."""
    line, kur = L.lorentzian(1.0), K.kuramoto()
    Kc = P.threshold(line, kur)
    for k in range(1, 5):
        Om = mp.mpf(10) ** -k
        KK, r, _ = P.branch_point(line, kur, Om)
        assert abs(r - mp.sqrt(1 - Kc / KK)) < mp.mpf("1e-26")


@pytest.mark.parametrize("line,want", [
    (L.lorentzian(1.0), mp.mpf(1)),
    (L.gaussian(1.0), mp.pi / 2),
    (L.student_t(3.0, 0.5), mp.mpf(4) / 3),
    (L.box(0.5), mp.pi ** 2 / 4),
])
def test_amplitude_closed_forms(line, want):
    """A = pi p(0)^2 / c takes simple closed values on the standard lines."""
    assert abs(P.amplitude(line) - want) / want < mp.mpf("1e-9")


@pytest.mark.parametrize("line", [L.lorentzian(1.0), L.gaussian(1.0),
                                  L.student_t(3.0, 0.5), L.box(0.5)])
def test_c_integral_matches_slope(line):
    """c from its defining integral equals the slope of the smoothed peak."""
    cons = K.conservative()
    p0 = line.p0()
    g = lambda o: P.G_of_Omega(line, cons, o) / mp.pi
    O1, O2 = mp.mpf("1e-6"), mp.mpf("2e-6")
    slope = 2 * (p0 - g(O1)) / O1 - (p0 - g(O2)) / O2
    c = P.c_coefficient(line)
    assert abs(slope - c) / abs(c) < mp.mpf("1e-9")


def test_amplitude_predicts_the_branch():
    """R / eps on the branch converges to the predicted amplitude."""
    for line in (L.lorentzian(1.0), L.gaussian(1.0), L.student_t(3.0, 0.5)):
        _, R, eps = P.branch_point(line, K.conservative(), mp.mpf(10) ** -6)
        assert abs(R / eps - P.amplitude(line)) / P.amplitude(line) < mp.mpf("1e-5")
