"""New in 1.2.0: the reduced coupling without cancellation, the order of the
onset on a two-peaked line, the fold of the branch, the amplitude of the
beta = 1/2 class, and lines that are flat at the centre.

Every reference here is computed independently of the package's quadrature:
closed forms of the branch (Lorentzian line, box line through a Gauss
hypergeometric function, Gaussian mixtures through the Faddeeva function
w(z) = exp(-z^2) erfc(-iz), whose real part is the Voigt profile), Dawson's
function for the coefficient c, and closed forms of kernel moments.
"""
import mpmath as mp
import pytest

from lockkernel import kernels as K, lineshapes as L, parametric as P
from lockkernel.lineshapes import LineShape


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def _faddeeva(z):
    return mp.exp(-z ** 2) * mp.erfc(-1j * z)


def _mixture_pdf(comps):
    """Symmetric Gaussian mixture: (weight, centre, sigma); a centre a > 0
    stands for two half-weight peaks at +a and -a."""
    terms = []
    for w, a, s in comps:
        c, k = w / (s * mp.sqrt(2 * mp.pi)), 1 / (2 * s ** 2)
        terms += [(c, a, k)] if a == 0 else [(c / 2, a, k), (c / 2, -a, k)]

    def pdf(d):
        d = mp.mpf(d)
        return mp.fsum(c * mp.exp(-k * (d - a) ** 2) for c, a, k in terms)
    return pdf


def _G_voigt(O, comps):
    """G(Omega) for the conservative kernel on a Gaussian mixture.

    G = int p(delta) Omega/(delta^2 + Omega^2) d delta is pi times the
    Lorentzian smoothing of the line at the centre, which for each Gaussian
    is a Voigt profile, Re w((a + i Omega)/(sigma sqrt 2))/(sigma sqrt(2 pi)).
    """
    tot = mp.mpf(0)
    for w, a, s in comps:
        z = (a + 1j * mp.mpf(O)) / (s * mp.sqrt(2))
        tot += w * mp.re(_faddeeva(z)) / (s * mp.sqrt(2 * mp.pi))
    return mp.pi * tot


def _dG_voigt(O, comps):
    """dG/dOmega from w'(z) = -2 z w(z) + 2i/sqrt(pi)."""
    tot = mp.mpf(0)
    for w, a, s in comps:
        z = (a + 1j * mp.mpf(O)) / (s * mp.sqrt(2))
        dw = -2 * z * _faddeeva(z) + 2j / mp.sqrt(mp.pi)
        tot += w * mp.re(dw * 1j / (s * mp.sqrt(2))) / (s * mp.sqrt(2 * mp.pi))
    return mp.pi * tot


def _fold_reference(comps, got):
    """Turning points and jump from the Voigt closed form, by root finding
    started at the package's answers."""
    G0 = _G_voigt(0, comps)
    O_lo = mp.findroot(lambda o: _dG_voigt(o, comps), got["Omega_lo"])
    if got["Omega_hi"] > 0:
        O_hi = mp.findroot(lambda o: _dG_voigt(o, comps), got["Omega_hi"])
        G_hi = _G_voigt(O_hi, comps)
    else:
        O_hi, G_hi = mp.mpf(0), G0
    O_j = mp.findroot(lambda o: _G_voigt(o, comps) - G_hi, got["Omega_jump"])
    return dict(chiN_lo=1 / _G_voigt(O_lo, comps), chiN_hi=1 / G_hi,
                R_low=O_hi * G_hi, R_jump=O_j * _G_voigt(O_j, comps),
                R_high_at_lo=O_lo * _G_voigt(O_lo, comps),
                Omega_lo=O_lo, Omega_hi=O_hi, Omega_jump=O_j)


def _dawson(x):
    return mp.sqrt(mp.pi) / 2 * mp.exp(-x ** 2) * mp.erfi(x)


BIMODAL4 = [(mp.mpf(1), mp.mpf(2), mp.mpf(1))]          # bimodal_gaussian(4.0)
TRIMODAL = [(mp.mpf("0.1"), mp.mpf(0), mp.mpf("0.1")),
            (mp.mpf("0.9"), mp.mpf(4), mp.mpf("0.3"))]


# ---------------------------------------------------------------------------
# The reduced coupling keeps the working precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dps,exps,tol", [
    (15, (6, 9, 12, 14), "1e-13"),
    (30, (9, 14, 20), "1e-28"),
])
def test_eps_keeps_working_precision(dps, exps, tol):
    """Conservative kernel, Lorentzian of half width a: chiN = a + Omega,
    so eps = Omega/a exactly.  1.1.1 lost log10(1/eps) digits here (at 15
    digits: 2e-4 relative at Omega = 1e-9, wrong by a factor 220 at 1e-12),
    and G itself was 4e-10 off at Omega = 1e-12."""
    a = mp.mpf(1) / 2
    with mp.workdps(dps):
        for e in exps:
            Om = mp.mpf(10) ** -e
            chiN, R, eps = P.branch_point(L.lorentzian(1.0), K.conservative(), Om)
            assert abs(eps / (Om / a) - 1) < mp.mpf(tol)
            assert abs(chiN / (a + Om) - 1) < mp.mpf(tol)
            assert abs(R / (Om / (a + Om)) - 1) < mp.mpf(tol)


@pytest.mark.parametrize("s", [1.5, 2.5, 4.0, 6.0])
def test_box_line_branch_closed_form(s):
    """Box line of half width 1: G(Omega) = int_0^X du/(1+u^s) with
    X = 1/Omega, which is X 2F1(1, 1/s; 1+1/s; -X^s), and G(0) is the half
    mass (pi/s)/sin(pi/s).  eps and G to 1e-13 at 15 digits, down to
    eps ~ 1e-26 (s = 6)."""
    line, kern = L.box(1.0), K.power_tail(s)
    for e in (1, 3, 5):
        Om = mp.mpf(10) ** -e
        _, R, eps = P.branch_point(line, kern, Om)
        G = P.G_of_Omega(line, kern, Om)
        with mp.workdps(60):
            X, S = 1 / Om, mp.mpf(s)
            Gx = X * mp.hyp2f1(1, 1 / S, 1 + 1 / S, -X ** S)
            eps_x = ((mp.pi / S) / mp.sin(mp.pi / S) - Gx) / Gx
        assert abs(G / Gx - 1) < mp.mpf("1e-13")
        assert abs(eps / eps_x - 1) < mp.mpf("1e-13")


# ---------------------------------------------------------------------------
# Lines that are flat at the centre, and branches below threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [4.0, 6.0])
def test_flat_top_line_gives_one_over_s_minus_one_beyond_three(s):
    """On the box line G(0) - G(Omega) = p(0) int_{|u|>1/Omega} W, so
    beta = 1/(s-1) for every s > 1: 1/3 and 1/5 here, where the rule for
    a rounded line (predicted_beta) says 1/2."""
    beta = P.extract_beta(L.box(1.0), K.power_tail(s), [-2, -3, -4, -5])[-1]
    assert abs(beta - 1 / (mp.mpf(s) - 1)) < mp.mpf("1e-6")
    assert K.predicted_beta(s) == 0.5


def test_extract_beta_refuses_a_branch_at_or_below_threshold():
    """1.1.1 returned slopes 1.0016447, 1.0001645, 1.0000165 (exponents
    -3 .. -6, 20 digits) for the backward branch of the two-peaked line (eps < 0) and raised ZeroDivisionError on the flat
    one (box line, Kuramoto kernel: G(Omega) = G(0) exactly)."""
    with pytest.raises(ValueError, match="bends back"):
        P.extract_beta(L.bimodal_gaussian(4.0), K.conservative(), [-3, -4])
    with pytest.raises(ValueError, match="flat"):
        P.extract_beta(L.box(1.0), K.kuramoto(), [-3, -4])
    _, _, eps = P.branch_point(L.box(1.0), K.kuramoto(), mp.mpf("1e-3"))
    assert eps == 0


# ---------------------------------------------------------------------------
# Two-peaked line: the order of the onset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sep", [1.0, 2.0, 4.0])
def test_bimodal_c_against_dawson(sep):
    """For two unit Gaussians at +-a the Voigt form gives
    c = (1 - 2 x D(x))/pi, x = a/sqrt(2), with Dawson's function D."""
    x = mp.mpf(sep) / 2 / mp.sqrt(2)
    want = (1 - 2 * x * _dawson(x)) / mp.pi
    assert abs(P.c_coefficient(L.bimodal_gaussian(sep)) - want) < mp.mpf("1e-13")


def test_bimodal_order_switches_at_the_dawson_maximum():
    """c changes sign where 1 - 2 x D(x) = D'(x) = 0, the maximum of
    Dawson's function: separation 2 sqrt(2) x*, about 2.61386 widths."""
    xs = mp.findroot(lambda x: 1 - 2 * x * _dawson(x), 0.9)
    sep_c = 2 * mp.sqrt(2) * xs
    assert abs(sep_c - mp.mpf("2.6138594554")) < mp.mpf("1e-9")
    assert P.c_coefficient(L.bimodal_gaussian(float(sep_c) - 1e-3)) > 0
    assert P.c_coefficient(L.bimodal_gaussian(float(sep_c) + 1e-3)) < 0


def test_bimodal_branch_against_voigt():
    line = L.bimodal_gaussian(4.0)
    for O in ("1e-3", "0.1", "1", "3"):
        G = P.G_of_Omega(line, K.conservative(), mp.mpf(O))
        assert abs(G / _G_voigt(mp.mpf(O), BIMODAL4) - 1) < mp.mpf("1e-13")


# ---------------------------------------------------------------------------
# The fold of the branch
# ---------------------------------------------------------------------------

def test_fold_leaving_the_threshold_backwards():
    """Two peaks 4 widths apart (c < 0): the branch leaves the threshold
    backwards.  Every returned number against the Voigt closed form; the
    end of the low branch is the threshold itself (1.1.1 returned the
    first grid point, chiN_hi = 5.89464 instead of 5.89561)."""
    got = P.fold_interval(L.bimodal_gaussian(4.0), n=30, lo=-2, hi=1)
    assert got["Omega_hi"] == 0 and got["R_low"] == 0
    with mp.workdps(30):
        ref = _fold_reference(BIMODAL4, got)
    assert got["chiN_hi"] == P.threshold(L.bimodal_gaussian(4.0), K.conservative())
    for k, v in ref.items():
        assert abs(got[k] - v) <= mp.mpf("1e-12") * abs(v), k


def test_fold_s_shaped_branch():
    """A narrow central peak (continuous onset) with two strong side peaks
    at +-4: the branch rises, then folds between Omega ~ 1.8 and 2.8.
    Both turning points are interior."""
    line = LineShape("trimodal", _mixture_pdf(TRIMODAL), 4.3)
    got = P.fold_interval(line, n=20, lo=-1, hi=0.8)
    assert got["Omega_hi"] > 1
    with mp.workdps(30):
        ref = _fold_reference(TRIMODAL, got)
    for k, v in ref.items():
        assert abs(got[k] - v) <= mp.mpf("1e-12") * abs(v), k
    assert got["chiN_lo"] < got["chiN_hi"]


def test_no_fold_on_a_monotonic_branch():
    assert P.fold_interval(L.bimodal_gaussian(1.0), n=12, lo=-2, hi=1) is None


# ---------------------------------------------------------------------------
# The amplitude of the beta = 1/2 class
# ---------------------------------------------------------------------------

def test_second_moment_closed_forms():
    """int u^2 W du: pi/8 (Kuramoto), sqrt(pi)/2 (Gaussian kernel), and
    2 (pi/s)/sin(3 pi/s) for 1/(1+|u|^s), s > 3."""
    assert abs(K.kuramoto().second_moment() - mp.pi / 8) < mp.mpf("1e-14")
    assert abs(K.gaussian_kernel().second_moment() - mp.sqrt(mp.pi) / 2) < mp.mpf("1e-14")
    for s in (4.0, 6.0):
        want = 2 * (mp.pi / s) / mp.sin(3 * mp.pi / s)
        assert abs(K.power_tail(s).second_moment() / want - 1) < mp.mpf("1e-12")
    for kern in (K.conservative(), K.power_tail(3.0)):
        with pytest.raises(ValueError, match="diverges"):
            kern.second_moment()


def test_amplitude_curvature_closed_forms():
    """Kuramoto kernel on a Lorentzian line: R = sqrt(1 - chiN_c/chiN), so
    R/sqrt(eps) -> 1.  On a Gaussian line of width sigma, p''(0) =
    -p(0)/sigma^2 and the width cancels: A = sqrt(pi) for the Kuramoto
    kernel and sqrt(2) for the Gaussian kernel, at any width."""
    A = P.amplitude_curvature(L.lorentzian(1.0), K.kuramoto())
    assert abs(A - 1) < mp.mpf("1e-12")
    for fwhm in (1.0, 7.0):
        A = P.amplitude_curvature(L.gaussian(fwhm), K.kuramoto())
        assert abs(A / mp.sqrt(mp.pi) - 1) < mp.mpf("1e-12")
        A = P.amplitude_curvature(L.gaussian(fwhm), K.gaussian_kernel())
        assert abs(A / mp.sqrt(2) - 1) < mp.mpf("1e-12")


@pytest.mark.parametrize("line,kern,tol", [
    (L.gaussian(1.0), K.kuramoto(), "1e-10"),
    (L.student_t(3.0, 0.5), K.gaussian_kernel(), "1e-10"),
    (L.gaussian(1.0), K.power_tail(6.0), "1e-10"),
    (L.gaussian(1.0), K.power_tail(4.0), "2e-6"),
])
def test_amplitude_curvature_predicts_the_branch(line, kern, tol):
    """R/sqrt(eps) at Omega = 1e-6 against the formula.  The correction is
    of relative order Omega^2, but Omega^(s-3) = 1e-6 for s = 4."""
    with mp.workdps(20):
        A = P.amplitude_curvature(line, kern)
        _, R, eps = P.branch_point(line, kern, mp.mpf("1e-6"))
        assert abs(R / mp.sqrt(eps) / A - 1) < mp.mpf(tol)


def test_amplitude_curvature_refusals():
    with pytest.raises(ValueError, match="rounded maximum"):
        P.amplitude_curvature(L.box(1.0), K.kuramoto())            # flat top
    with pytest.raises(ValueError, match="rounded maximum"):
        P.amplitude_curvature(L.bimodal_gaussian(4.0), K.kuramoto())  # dip
    with pytest.raises(ValueError, match="diverges"):
        P.amplitude_curvature(L.gaussian(1.0), K.conservative())   # s = 2


# ---------------------------------------------------------------------------
# New in 1.3.0: slow algebraic tails and the edges of the window 1 < s < 3.
# References: int_0^inf du/(1+u^s) = (pi/s)/sin(pi/s) and
# int_0^inf u^2 du/(1+u^s) = (pi/s)/sin(3 pi/s) (Beta-function integrals),
# and closed forms of I_s derived in the docstrings below.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dps,tol", [(15, "1e-14"), (30, "1e-28")])
def test_mass_of_slow_tails(dps, tol):
    """1.2.0 integrated the tail directly: power_tail(1.1).mass() was
    1.2 % (15 digits) and 3.7e-4 (30 digits) too small, and so the
    threshold too large."""
    with mp.workdps(dps):
        for s in ("1.05", "1.1", "1.25", "1.5"):
            s = mp.mpf(s)
            want = 2 * (mp.pi / s) / mp.sin(mp.pi / s)
            assert abs(K.power_tail(s).mass() / want - 1) < mp.mpf(tol)


def test_threshold_of_a_slow_tail_matches_the_branch():
    """Lorentzian line of FWHM 1 (p(0) = 2/pi), W = 1/(1+|u|^1.1): the
    threshold is s sin(pi/s)/4, and chiN/chiN_c - 1 on the branch equals
    eps from the separate difference integral.  1.2.0 at 15 digits:
    threshold 0.0784174 instead of 0.0774765, and chiN/chiN_c - 1 =
    1.11102 against eps = 1.13666 at Omega = 1e-3."""
    with mp.workdps(15):
        s = mp.mpf("1.1")
        line, kern = L.lorentzian(1.0), K.power_tail(s)
        chi_c = P.threshold(line, kern)
        assert abs(chi_c / (s * mp.sin(mp.pi / s) / 4) - 1) < mp.mpf("1e-13")
        chiN, _, eps = P.branch_point(line, kern, mp.mpf("1e-3"))
        assert abs((chiN / chi_c - 1) / eps - 1) < mp.mpf("1e-10")


@pytest.mark.parametrize("gamma", [3.4, 3.8, 4.4])
def test_heterogeneous_kuramoto_mass_closed_form(gamma):
    """Averaging the Kuramoto kernel (mass pi/2) over P(k) ~ k^-gamma,
    k >= 1, with field k^eta: int W(v/k^eta) dv = (pi/2) k^eta, so
    m = (pi/2) (gamma-2)/(gamma-2-eta) for gamma > 2 + eta (eta = 1 here).
    1.2.0 gave 1.4e-8 relative error at 15 digits for gamma = 3.4."""
    with mp.workdps(15):
        g = mp.mpf(gamma)
        want = mp.pi / 2 * (g - 2) / (g - 3)
        got = K.heterogeneous(K.kuramoto(), gamma).mass()
        assert abs(got / want - 1) < mp.mpf("1e-13")


def test_second_moment_of_slow_tails():
    """s just above 3: u^2 W falls like u^-(s-2).  1.2.0 at 15 digits:
    1.2 % too small for s = 3.1."""
    with mp.workdps(15):
        for s in ("3.1", "3.25", "3.5"):
            s = mp.mpf(s)
            want = 2 * (mp.pi / s) / mp.sin(3 * mp.pi / s)
            got = K.power_tail(s).second_moment()
            assert abs(got / want - 1) < mp.mpf("1e-13")


def _I_lorentzian(fwhm, s):
    """I_s on a Lorentzian of half width g: p(0) - p(d) = d^2/(pi g (d^2+g^2)),
    and int_0^inf x^(a-1)/(1+x^2) dx = pi/(2 sin(pi a/2)) with a = 3-s,
    so I_s = g^-s / sin(pi (3-s)/2)."""
    g = mp.mpf(fwhm) / 2
    return g ** -s / mp.sin(mp.pi * (3 - s) / 2)


def _I_gaussian(fwhm, s):
    """I_s on a Gaussian of standard deviation sig: with y = d^2/(2 sig^2)
    and int_0^inf (1-e^-y) y^(-a-1) dy = Gamma(1-a)/a, a = (s-1)/2,
    I_s = 2 p(0) sig^(1-s) 2^(-(s+1)/2) Gamma((3-s)/2) 2/(s-1)."""
    sig = mp.mpf(fwhm) / (2 * mp.sqrt(2 * mp.log(2)))
    p0 = 1 / (sig * mp.sqrt(2 * mp.pi))
    return (2 * p0 * sig ** (1 - s) * mp.mpf(2) ** (-(s + 1) / 2)
            * mp.gamma((3 - s) / 2) * 2 / (s - 1))


@pytest.mark.parametrize("fwhm", [1.0, 1000.0, 0.001])
def test_tail_integral_closed_forms(fwhm):
    """1.2.0 integrated the singularity d^(2-s) at the origin directly: on
    the Lorentzian line of FWHM 1 at 15 digits I_s was 1.4 % too small at
    s = 2.9 and 12 % at s = 2.95 (4.5e-10 at s = 2.5)."""
    with mp.workdps(15):
        for s in ("1.1", "1.5", "2", "2.5", "2.9", "2.95", "2.99"):
            s = mp.mpf(s)
            got = P.tail_integral(L.lorentzian(fwhm), s)
            assert abs(got / _I_lorentzian(fwhm, s) - 1) < mp.mpf("1e-13")
            got = P.tail_integral(L.gaussian(fwhm), s)
            assert abs(got / _I_gaussian(fwhm, s) - 1) < mp.mpf("1e-13")


def test_tail_integral_at_30_digits():
    with mp.workdps(30):
        for s in ("1.1", "2.5", "2.95"):
            s = mp.mpf(s)
            got = P.tail_integral(L.gaussian(1.0), s)
            assert abs(got / _I_gaussian(1.0, s) - 1) < mp.mpf("1e-27")


def test_general_amplitude_tail_amplitude_check():
    """The default C of amplitude_general is W(v) v^s at v = 1e6.

    Conservative kernel averaged over P(k) ~ k^-3.8 (eta = 1): the tail is
    C v^-1.8 with C = (gamma-2) int_0^inf q^(3-gamma)/(1+q^2) dq =
    (gamma-2) pi / (2 sin(pi (4-gamma)/2)), approached only with a relative
    correction of order v^-0.2; at v = 1e6 the estimate is 6 % low (1.2.0
    used it silently, giving A 8.3 % high on a Gaussian line).  It is now
    refused.  The Kuramoto average has an exact power-law tail above
    v = 1, C = (gamma-2) B(s/2, 3/2) / 2, and is accepted."""
    with mp.workdps(15):
        g = mp.mpf(3.8)
        a = 4 - g
        C_closed = (g - 2) * mp.pi / (2 * mp.sin(mp.pi * a / 2))
        # the integral itself, by quadrature in variables that remove the
        # endpoint singularity (q = t^(1/a) on [0, 1], q = 1/p beyond)
        near = mp.quad(lambda t: 1 / (a * (1 + t ** (2 / a))), [0, 1])
        far = mp.quad(lambda p: p ** (1 - a) / (1 + p ** 2), [0, 1])
        assert abs((g - 2) * (near + far) / C_closed - 1) < mp.mpf("1e-13")
        het = K.heterogeneous(K.conservative(), 3.8)
        v = mp.mpf(10) ** 6
        assert abs(het.W(v) * v ** het.tail / C_closed - 1) > 0.05
        with pytest.raises(ValueError, match="not converged"):
            P.amplitude_general(L.gaussian(1.0), het)

        het_k = K.heterogeneous(K.kuramoto(), 3.8)
        s = mp.mpf(het_k.tail)
        C_k = (g - 2) * mp.beta(s / 2, mp.mpf(3) / 2) / 2
        line = L.gaussian(1.0)
        A_default = P.amplitude_general(line, het_k)
        A_closed = P.amplitude_general(line, het_k, C=C_k)
        assert abs(A_default / A_closed - 1) < mp.mpf("1e-12")


@pytest.mark.parametrize("s", ["1.1", "1.5", "2.5", "2.9"])
def test_general_amplitude_closed_form_on_a_lorentzian(s):
    """A_s = p(0) m [p(0) m / (C I_s)]^(1/(s-1)) with every piece in closed
    form for W = 1/(1+|u|^s) (C = 1) on a Lorentzian line of half width 1/2:
    p(0) = 2/pi, m = 2 (pi/s)/sin(pi/s), I_s from _I_lorentzian.  The power
    1/(s-1) amplifies errors: on a Gaussian line at 15 digits, 1.2.0 gave
    8.4219 instead of 9.6179 for s = 1.1 (from the mass) and 0.45574
    instead of 0.45258 for s = 2.9 (from I_s)."""
    with mp.workdps(15):
        s = mp.mpf(s)
        p0m = 2 / mp.pi * 2 * (mp.pi / s) / mp.sin(mp.pi / s)
        want = p0m * (p0m / _I_lorentzian(1.0, s)) ** (1 / (s - 1))
        line, kern = L.lorentzian(1.0), K.power_tail(s)
        assert abs(P.amplitude_general(line, kern, C=1) / want - 1) < mp.mpf("1e-12")
        # the default C = W(1e6) 1e6^s = 1/(1 + 1e6^-s) is 1e-6^s low
        assert abs(P.amplitude_general(line, kern) / want - 1) < mp.mpf("1e-5")


def test_general_amplitude_refusals():
    with pytest.raises(ValueError, match="algebraic tail"):
        P.amplitude_general(L.gaussian(1.0), K.kuramoto())
    with pytest.raises(ValueError, match="algebraic tail"):
        P.amplitude_general(L.gaussian(1.0), K.power_tail(4.0))
    with pytest.raises(ValueError, match="positive"):
        P.amplitude_general(L.gaussian(1.0), K.power_tail(2.5), C=-1)


# ---------------------------------------------------------------------------
# New in 1.5.0: I_s and the general amplitude on lines with compact support
# ---------------------------------------------------------------------------

def _semicircle(w):
    """Semicircle line p = 2 sqrt(w^2 - d^2) / (pi w^2) on |d| < w: compact
    support with a rounded top, p''(0) = -2/(pi w^3)."""
    w = mp.mpf(w)

    def pdf(d):
        d = mp.mpf(d)
        return 2 * mp.sqrt(w ** 2 - d ** 2) / (mp.pi * w ** 2) if abs(d) < w \
            else mp.mpf(0)

    return LineShape("semicircle", pdf, float(w), compact_support=float(w))


@pytest.mark.parametrize("w", [0.001, 1.0, 1000.0])
def test_tail_integral_on_compact_lines(w):
    """Closed forms of I_s = 2 int_0^inf [p(0) - p(d)] d^-s dd.

    Box of half width w: p(0) - p = 0 inside, p(0) = 1/(2w) outside, so
    I_s = w^-s / (s-1).  Semicircle of radius w: with y = (d/w)^2 the inner
    part is (2/(pi w^s)) int_0^1 y^(a-1) [1 - (1-y)^(1/2)] dy, a = (1-s)/2,
    which is 1/a - B(a, 3/2) by analytic continuation of the Beta integral
    (both sides analytic for a > -1); 1/a cancels the outer part
    2 p(0) w^(1-s)/(s-1), leaving I_s = -(2/pi) B((1-s)/2, 3/2) w^-s.
    1.4.0 integrated -p' by parts, which misses the jump of the box at its
    edge: I_s = 0 for the box at every s (closed form 2 at s = 1.5 and
    w = 1), and amplitude_general raised ZeroDivisionError."""
    with mp.workdps(15):
        for s in ("1.1", "1.5", "2", "2.5", "2.9", "2.99"):
            s = mp.mpf(s)
            got = P.tail_integral(L.box(w), s)
            assert abs(got / (mp.mpf(w) ** -s / (s - 1)) - 1) < mp.mpf("1e-13")
            got = P.tail_integral(_semicircle(w), s)
            with mp.workdps(40):
                want = -(2 / mp.pi) * mp.beta((1 - s) / 2, mp.mpf(3) / 2) \
                    * mp.mpf(w) ** -s
            assert abs(got / want - 1) < mp.mpf("1e-13")
        # at s = 2, I_s = pi c, with c from its own (direct) integral
        for line in (L.box(w), _semicircle(w)):
            c = P.c_coefficient(line)
            assert abs(P.tail_integral(line, 2) / (mp.pi * c) - 1) < mp.mpf("1e-13")


@pytest.mark.parametrize("s", ["1.5", "2.5"])
def test_general_amplitude_on_the_box_line(s):
    """Box of half width 1 with W = 1/(1+|u|^s): every piece in closed form,
    p(0) = 1/2, m = 2 (pi/s)/sin(pi/s), C = 1, I_s = 1/(s-1); and the limit
    R / eps^(1/(s-1)) of the closed-form branch at Omega = 1e-40 and 60
    digits, where the correction (s/(s-1)) eps is below 1e-18.  There
    D = G(0) - G = int_X^inf du/(1+u^s) = X^(1-s) 2F1(1, b; 1+b; -X^-s)/(s-1),
    X = 1/Omega, b = (s-1)/s (termwise integration of the geometric series),
    eps = D/(G(0) - D) and R = Omega (G(0) - D)."""
    with mp.workdps(15):
        A = P.amplitude_general(L.box(1.0), K.power_tail(s), C=1)
        S = mp.mpf(s)
        G0 = (mp.pi / S) / mp.sin(mp.pi / S)
        want = G0 * (G0 * (S - 1)) ** (1 / (S - 1))
        assert abs(A / want - 1) < mp.mpf("1e-13")
    with mp.workdps(60):
        S = mp.mpf(s)
        G0 = (mp.pi / S) / mp.sin(mp.pi / S)
        Om = mp.mpf(10) ** -40
        X, b = 1 / Om, (S - 1) / S
        D = X ** (1 - S) * mp.hyp2f1(1, b, 1 + b, -X ** -S) / (S - 1)
        lim = Om * (G0 - D) / (D / (G0 - D)) ** (1 / (S - 1))
    assert abs(A / lim - 1) < mp.mpf("1e-13")


def test_input_checks_of_lines_and_kernels():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            L.lorentzian(bad)
        with pytest.raises(ValueError, match="positive"):
            L.gaussian(bad)
        with pytest.raises(ValueError, match="positive"):
            L.box(bad)
    with pytest.raises(ValueError, match="positive"):
        L.student_t(-3.0)
    with pytest.raises(ValueError, match="positive"):
        L.bimodal_gaussian(4.0, width=0.0)
    with pytest.raises(ValueError, match="finite"):
        K.power_tail(float("nan"))
    with pytest.raises(ValueError, match="positive"):
        K.heterogeneous(K.kuramoto(), 3.8, eta=0.0)
    with pytest.raises(ValueError, match="s > 1"):
        K.predicted_beta(0.5)
    assert K.predicted_beta(None) == 0.5
    assert K.predicted_beta(float("inf")) == 0.5
    assert K.predicted_beta(2.5) == 1 / 1.5
