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
