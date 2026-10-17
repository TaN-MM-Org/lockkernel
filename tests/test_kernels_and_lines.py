"""New in 1.4.0: the coupling-averaged kernels against closed forms, the
Voigt line, and fixes found in the second audit.

References are independent of the package's quadratures:

* Averaged kernels.  With a = (gamma-2)/eta, the normalized average of a
  base kernel W over P(k) ~ k^-gamma (k >= 1, field k^eta) is
  int_0^1 t^(gamma-3) W(v t^eta) dt / int_0^1 t^(gamma-3) dt (substitute
  t = 1/k).  For W = 1/(1+|x|^s0) expanding the geometric series gives
  2F1(1, b; 1+b; -v^s0) with b = (gamma-2)/(eta s0); for the Gaussian
  kernel W = exp(-x^2) it is b v^(-2b) gamma_inc(b, v^2) with
  b = (gamma-2)/(2 eta).  Both are evaluated by mpmath's hypergeometric
  and incomplete gamma functions at 40 digits.  The mass is
  m0 k_min^eta (gamma-2)/(gamma-2-eta) (integrate over v first).
* Voigt line.  DLMF 7.2.3 and 7.19.3 (https://dlmf.nist.gov/7.2,
  https://dlmf.nist.gov/7.19): the convolution of a Gaussian and a
  Lorentzian is Re w(z)/(sigma sqrt(2 pi)), the form SciPy's
  voigt_profile documents.  It is held here to a 120-digit evaluation, to
  SciPy (an independent double-precision implementation), to the
  convolution integral itself and to its Lorentzian tail.  For the
  conservative kernel G(Omega) is pi times the Lorentzian smoothing of
  the line at the center, which is the Voigt line with gamma + Omega at
  zero detuning: G = pi erfcx(y) / (sigma sqrt(2 pi)),
  y = (gamma + Omega)/(sigma sqrt 2), with erfcx(y) = exp(y^2) erfc(y) and
  erfcx'(y) = 2 y erfcx(y) - 2/sqrt(pi).
"""
import mpmath as mp
import numpy as np
import pytest

from lockkernel import kernels as K, lineshapes as L, parametric as P
from lockkernel.lineshapes import LineShape


# ---------------------------------------------------------------------------
# Coupling-averaged kernels
# ---------------------------------------------------------------------------

def _ref_rational(v, g, eta, s0):
    with mp.workdps(40):
        b = (mp.mpf(g) - 2) / (mp.mpf(eta) * s0)
        return mp.hyp2f1(1, b, b + 1, -mp.mpf(v) ** s0)


def _ref_gauss(v, g, eta):
    with mp.workdps(40):
        b = (mp.mpf(g) - 2) / (2 * mp.mpf(eta))
        v = mp.mpf(v)
        return b * v ** (-2 * b) * mp.gammainc(b, 0, v ** 2)


VS = ("0.3", "1", "3", "100", "1e4", "1e6", "1e9", "1e12")


@pytest.mark.parametrize("base,g,eta,s0", [
    ("conservative", 3.4, 1.0, 2), ("conservative", 3.8, 1.0, 2),
    ("conservative", 4.4, 1.0, 2), ("conservative", 2.6, 0.5, 2),
    ("conservative", 5.0, 2.0, 2), ("power2.5", 4.0, 1.0, 2.5),
    ("gaussian", 3.4, 1.0, None), ("gaussian", 2.5, 0.25, None),
])
def test_averaged_kernel_closed_forms(base, g, eta, s0):
    """1.3.0 integrated the weight on a linear scale out to infinity: for
    gamma = 2.6, eta = 0.5 (conservative) W was 2.9 % low at v = 1e9; for
    the Gaussian kernel with gamma = 2.5, eta = 0.25, 2.5 % low at
    v = 1e4 and 9.5 % from 1e6; for gamma = 4.4, 0.4 % at v = 1e9."""
    b = {"conservative": K.conservative(), "power2.5": K.power_tail(2.5),
         "gaussian": K.gaussian_kernel()}[base]
    ker = K.heterogeneous(b, g, eta)
    for v in VS:
        want = _ref_gauss(v, g, eta) if s0 is None else _ref_rational(v, g, eta, s0)
        assert abs(ker.W(mp.mpf(v)) / want - 1) < mp.mpf("1e-13"), v
    assert ker.W(0) == 1


@pytest.mark.parametrize("k_min", [2.0, 0.5])
def test_averaged_kuramoto_with_k_min(k_min):
    """The Kuramoto average with k_min != 1, against the weight integral
    itself, int_kmin^inf k^(1-g) sqrt(1 - v^2/k^2) dk / int_kmin^inf
    k^(1-g) dk, by quadrature at 30 digits split at the support edge
    k = v and in log(k).  1.3.0 took it on a linear scale: 2.8e-4 off at
    v = 1e9 for gamma = 3.8, k_min = 2."""
    g = mp.mpf("3.8")
    ker = K.heterogeneous(K.kuramoto(), 3.8, k_min=k_min)
    assert ker.breaks == (float(k_min),)
    for v in ("0.3", "1.9", "2.1", "50", "1e6", "1e9"):
        v = mp.mpf(v)
        with mp.workdps(30):
            km = mp.mpf(k_min)
            lo = max(km, v)
            f = lambda w: mp.exp((2 - g) * w) * mp.sqrt(1 - (v * mp.exp(-w)) ** 2)
            num = mp.quad(f, [mp.log(lo), mp.log(lo) + 1, mp.log(lo) + 5,
                              mp.log(lo) + 40])
            num += mp.quad(lambda w: mp.exp((2 - g) * w),
                           [mp.log(lo) + 40, mp.inf])
            want = num / (km ** (2 - g) / (g - 2))
        assert abs(ker.W(v) / want - 1) < mp.mpf("1e-13"), v


@pytest.mark.parametrize("base,g,eta,k_min", [
    ("kuramoto", 3.4, 1.0, 2.0), ("kuramoto", 3.8, 1.0, 0.5),
    ("conservative", 4.4, 1.0, 1.0),
])
def test_averaged_kernel_mass(base, g, eta, k_min):
    """m = m0 k_min^eta (gamma-2)/(gamma-2-eta).  1.3.0: 1.4e-6 and
    2.8e-6 too large for the Kuramoto average with k_min = 2 (gamma 3.4,
    3.8; the kink at v = k_min fell inside a quadrature segment), and,
    by hand, 1.8e-4 too small for the conservative average with
    gamma = 2.6, eta = 0.5 (107 s; now 7 s, not run here)."""
    b, m0 = {"kuramoto": (K.kuramoto(), mp.pi / 2),
             "conservative": (K.conservative(), mp.pi)}[base]
    ker = K.heterogeneous(b, g, eta, k_min=k_min)
    G, e = mp.mpf(g), mp.mpf(eta)
    want = m0 * mp.mpf(k_min) ** e * (G - 2) / (G - 2 - e)
    assert abs(ker.mass() / want - 1) < mp.mpf("1e-13")


def test_amplitude_general_accepts_an_exact_power_tail_with_k_min():
    """The Kuramoto average with k_min = 2 has an exact power-law tail
    above v = 2, W = C v^-s with C = (gamma-2) B(s/2, 3/2)/2 2^s.  1.3.0
    refused it ("not converged": W v^s was 2.389384 at 1e6 and 2.390062
    at 1e9).  Now the default C equals the closed form and gives the same
    amplitude to 1e-12."""
    g = mp.mpf("3.8")
    ker = K.heterogeneous(K.kuramoto(), 3.8, k_min=2.0)
    s = mp.mpf(ker.tail)
    C = (g - 2) * mp.beta(s / 2, mp.mpf(3) / 2) / 2 * 2 ** s
    for v in ("1e6", "1e9"):
        v = mp.mpf(v)
        assert abs(ker.W(v) * v ** s / C - 1) < mp.mpf("1e-13")
    line = L.gaussian(1.0)
    A = P.amplitude_general(line, ker)
    assert abs(A / P.amplitude_general(line, ker, C=C) - 1) < mp.mpf("1e-12")


def test_kernel_without_breaks_is_unchanged():
    """`breaks` is empty for the shipped kernels and for the k_min = 1
    Kuramoto average, so their integrals are the ones of 1.3.0."""
    for ker in (K.conservative(), K.kuramoto(), K.power_tail(2.5),
                K.gaussian_kernel(), K.heterogeneous(K.kuramoto(), 3.8)):
        assert ker.breaks == ()


# ---------------------------------------------------------------------------
# Line shapes
# ---------------------------------------------------------------------------

def test_negative_separation_is_the_same_line():
    """bimodal_gaussian(-4) is the same density as bimodal_gaussian(4).
    1.3.0 gave it the integration scale 1 + (-2) = -1: G(0.1) came out
    0.390375 instead of 0.195187, c = -0.0878293 instead of -0.0891192,
    and branch_point with the Kuramoto kernel raised TypeError."""
    pos, neg = L.bimodal_gaussian(4.0), L.bimodal_gaussian(-4.0)
    assert neg.scale == pos.scale == 3.0
    Om = mp.mpf("0.1")
    assert P.G_of_Omega(neg, K.conservative(), Om) == \
        P.G_of_Omega(pos, K.conservative(), Om)
    assert P.c_coefficient(neg) == P.c_coefficient(pos)
    assert P.branch_point(neg, K.kuramoto(), mp.mpf("0.01")) == \
        P.branch_point(pos, K.kuramoto(), mp.mpf("0.01"))


def test_line_shape_scale_must_be_positive():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="scale"):
            LineShape("bad", lambda d: mp.mpf(0), bad)


def _voigt_ref(d, fg, fl, dps=120):
    with mp.workdps(dps):
        sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
        z = (mp.mpf(d) + 1j * mp.mpf(fl) / 2) / (sig * mp.sqrt(2))
        return mp.re(mp.exp(-z ** 2) * mp.erfc(-1j * z)) / (sig * mp.sqrt(2 * mp.pi))


@pytest.mark.parametrize("fg,fl", [(1.0, 0.2), (0.01, 1.0), (1.0, 1e-4),
                                   (1000.0, 3.0)])
def test_voigt_density(fg, fl):
    """Against 120 digits out to 1e30 widths (2 log10|z| digits cancel in
    Re w(z) far out: without guard digits the density was 44 % off at
    1e8 widths), the Lorentzian tail gamma/(pi d^2) far out, and SciPy's
    voigt_profile."""
    from scipy.special import voigt_profile

    line = L.voigt(fg, fl)
    w = fg + fl
    sig = fg / (2 * np.sqrt(2 * np.log(2)))
    for d in ("0", "0.1", "0.7", "1", "3", "10", "100", "1e4", "1e8",
              "1e12", "1e20", "1e30"):
        d = mp.mpf(d) * w
        got = line.pdf(d)
        assert abs(got / _voigt_ref(d, fg, fl) - 1) < mp.mpf("1e-14")
        if d < 1e6 * w:
            assert abs(float(got) / voigt_profile(float(d), sig, fl / 2) - 1) < 1e-12
    for d in ("1e10", "1e20"):
        d = mp.mpf(d) * w
        assert abs(line.pdf(d) / (mp.mpf(fl) / 2 / mp.pi / d ** 2) - 1) < mp.mpf("1e-13")


def test_voigt_is_the_convolution():
    """Direct convolution of the Gaussian and the Lorentzian at 30 digits,
    and the normalization int p = 1."""
    fg, fl = 1.0, 0.4
    line = L.voigt(fg, fl)
    with mp.workdps(30):
        sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
        g = mp.mpf(fl) / 2
        for d in ("0", "0.8", "5"):
            d = mp.mpf(d)
            f = lambda x: (mp.exp(-x ** 2 / (2 * sig ** 2)) / (sig * mp.sqrt(2 * mp.pi))
                           * g / mp.pi / ((d - x) ** 2 + g ** 2))
            conv = mp.quad(f, [-mp.inf, -10, d - 1, d, d + 1, 10, mp.inf])
            assert abs(line.pdf(d) / conv - 1) < mp.mpf("1e-13")
    total = 2 * mp.quad(line.pdf, [0, 1, 10, 100, mp.inf])
    assert abs(total - 1) < mp.mpf("1e-12")


def _voigt_G(Om, fg, fl):
    """Closed form G(Omega) for the conservative kernel (module docstring)."""
    sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
    y = (mp.mpf(fl) / 2 + Om) / (sig * mp.sqrt(2))
    return mp.pi * mp.exp(y ** 2) * mp.erfc(y) / (sig * mp.sqrt(2 * mp.pi))


@pytest.mark.parametrize("fg,fl", [(1.0, 0.3)])
def test_voigt_conservative_branch_closed_form(fg, fl):
    """G, eps and the threshold on the branch, and the amplitude
    A = pi p(0)^2/c with c = -G'(0)/pi = (1/sqrt(pi) - y0 erfcx(y0)) /
    (sigma^2 sqrt(pi)), y0 = gamma/(sigma sqrt 2).  c > 0 for every
    width ratio (erfcx(y) < 1/(sqrt(pi) y)), so the onset is continuous."""
    line, kern = L.voigt(fg, fl), K.conservative()
    with mp.workdps(40):
        G0 = _voigt_G(0, fg, fl)
    assert abs(P.threshold(line, kern) * G0 - 1) < mp.mpf("1e-13")
    for Om in ("1e-8", "0.5"):
        Om = mp.mpf(Om)
        chiN, R, eps = P.branch_point(line, kern, Om)
        with mp.workdps(40):
            G = _voigt_G(Om, fg, fl)
            eps_x = G0 / G - 1
        assert abs(chiN * G - 1) < mp.mpf("1e-13")
        assert abs(R / (Om * G) - 1) < mp.mpf("1e-13")
        assert abs(eps / eps_x - 1) < mp.mpf("1e-12")
    with mp.workdps(40):
        sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
        y0 = mp.mpf(fl) / 2 / (sig * mp.sqrt(2))
        ex = mp.exp(y0 ** 2) * mp.erfc(y0)
        c = (1 / mp.sqrt(mp.pi) - y0 * ex) / (sig ** 2 * mp.sqrt(mp.pi))
        p0 = ex / (sig * mp.sqrt(2 * mp.pi))
        A = mp.pi * p0 ** 2 / c
        # the derivative of erfcx used above, checked numerically
        d = mp.diff(lambda t: mp.exp(t ** 2) * mp.erfc(t), y0)
        assert abs(d - (2 * y0 * ex - 2 / mp.sqrt(mp.pi))) < mp.mpf("1e-30")
    assert abs(P.c_coefficient(line) / c - 1) < mp.mpf("1e-12")
    assert abs(P.amplitude(line) / A - 1) < mp.mpf("1e-12")


def test_voigt_curvature_amplitude():
    """Kuramoto kernel on a Voigt line: A = G0^(3/2) sqrt(2/(-p''(0) M2))
    with G0 = p(0) pi/2, M2 = pi/8 and, from w''(z) = (4z^2-2) w(z) -
    4iz/sqrt(pi) at z = i y0, p''(0) = [4 y0/sqrt(pi) - (4 y0^2 + 2)
    erfcx(y0)] / (2 sigma^2 sigma sqrt(2 pi)).  And the branch at
    Omega = 1e-6 against it."""
    fg, fl = 1.0, 0.3
    line = L.voigt(fg, fl)
    with mp.workdps(40):
        sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
        y0 = mp.mpf(fl) / 2 / (sig * mp.sqrt(2))
        ex = mp.exp(y0 ** 2) * mp.erfc(y0)
        p0 = ex / (sig * mp.sqrt(2 * mp.pi))
        pp = (4 * y0 / mp.sqrt(mp.pi) - (4 * y0 ** 2 + 2) * ex) \
            / (2 * sig ** 2) / (sig * mp.sqrt(2 * mp.pi))
        G0 = p0 * mp.pi / 2
        A = G0 * mp.sqrt(2 * G0 / (-pp * mp.pi / 8))
    got = P.amplitude_curvature(line, K.kuramoto())
    assert abs(got / A - 1) < mp.mpf("1e-12")
    with mp.workdps(20):
        _, R, eps = P.branch_point(line, K.kuramoto(), mp.mpf("1e-6"))
        assert abs(R / mp.sqrt(eps) / A - 1) < mp.mpf("1e-10")


def test_voigt_refusals():
    with pytest.raises(ValueError, match="fwhm_gauss"):
        L.voigt(0.0, 1.0)
    with pytest.raises(ValueError, match="fwhm_lorentz"):
        L.voigt(1.0, -1.0)
    with pytest.raises(NotImplementedError):
        L.voigt(1.0, 1.0).cdf(0.0)


# ---------------------------------------------------------------------------
# Inputs refused since 1.4.0
# ---------------------------------------------------------------------------

def test_negative_or_infinite_omega_is_refused():
    """1.3.0: G_of_Omega(lorentzian(), conservative(), -0.1) returned
    3.3333 (twice G(0.1) = 1.6667); branch_point raised TypeError for a
    negative Omega and ZeroDivisionError for an infinite one."""
    line, kern = L.lorentzian(1.0), K.conservative()
    for bad in (-0.1, mp.inf, mp.nan):
        with pytest.raises(ValueError, match="Omega"):
            P.G_of_Omega(line, kern, bad)
        with pytest.raises(ValueError, match="Omega"):
            P.branch_point(line, kern, bad)
        with pytest.raises(ValueError, match="Omega"):
            P.H_of_Omega(line, kern, bad)
    assert P.G_of_Omega(line, kern, 0) == line.p0() * kern.mass()


# ---------------------------------------------------------------------------
# Discretization and the cumulant diagnostics
# ---------------------------------------------------------------------------

def test_class_table_free_edge_beyond_delta_max():
    """With the free edge beyond delta_max (build_system puts it there
    for chiN above about 125 line widths), 1.3.0 counted the population
    between delta_max and the free edge twice: for free_beyond = 5000,
    delta_max = 1000 the free classes held 2 sf(~1000) instead of
    2 sf(5000), the raw total was 1.000197, and the locking integral at
    Omega = 0.5 was 2.1e-4 off (1.4e-5 now; 8.5e-6 with the edge
    inside)."""
    from lockkernel.ensemble import class_table

    line = L.lorentzian(1.0)
    d, n, sd, sn = class_table(line, M_core=800, M_tail=24, M_spec=8,
                               free_beyond=5000.0, delta_max=1000.0)
    assert np.all(np.abs(sd) >= 5000.0)
    assert abs(sn.sum() / (2 * line.frozen.sf(5000.0)) - 1) < 1e-9
    for Om in (0.5, 2.0):
        got = n @ (1 / (1 + (d / Om) ** 2)) + sn @ (1 / (1 + (sd / Om) ** 2))
        want = float(P.G_of_Omega(line, K.conservative(), Om) * Om)
        assert abs(got / want - 1) < 5e-5


def test_physicality_and_valid_window():
    """A coherent spin state along +x has variance 0 along x and N/4 across
    it, and Bloch vectors of length 1: physicality returns (0, 1).
    valid_window stops at the first state that leaves the physical set."""
    from lockkernel.cumulant import (State, System, coherent_state_x,
                                     physicality, valid_window)

    sysm = System(delta=np.linspace(-1, 1, 5), n=np.full(5, 40.0), chiN=1.0)
    st = coherent_state_x(sysm)
    mev, vmax = physicality(st, sysm)
    assert abs(mev) < 1e-12 and abs(vmax - 1) < 1e-15
    bad = State(st.s * 1.2, st.z, st.P, st.Q, st.R, st.Z)   # |v| = 1.2
    assert physicality(bad, sysm)[1] > 1.1
    assert valid_window([st, st, bad, st], sysm) == 1
    assert valid_window([bad, st], sysm) == -1
