"""New in 1.4.0: the branch at a given coupling, the exact local exponent and
the power-law window, against closed forms; and the speed-ups of
branch_point, sweep and points_for_beta held bit for bit to
the 1.3.0 code paths, which are kept here as the reference.

Closed forms used:

* Conservative kernel, Lorentzian line of half width a: chiN = a + Omega,
  eps = Omega/a, R = eps/(1+eps), so d log R/d log eps = 1/(1+eps).
* Kuramoto kernel, Lorentzian line: R = sqrt(1 - chiN_c/chiN) =
  sqrt(eps/(1+eps)), so d log R/d log eps = (1/2)/(1+eps).
* Conservative kernel, Voigt line: G(Omega) = pi erfcx(y)/(sigma sqrt(2 pi)),
  y = (gamma + Omega)/(sigma sqrt 2) (see test_kernels_and_lines.py), and
  with R = Omega G, eps = G(0)/G - 1 the local exponent is
  -(G + Omega G') (G(0) - G) / (Omega G' G(0)), G' from
  erfcx'(y) = 2 y erfcx(y) - 2/sqrt(pi).
"""
import mpmath as mp
import pytest

from lockkernel import kernels as K, lineshapes as L, parametric as P


def _voigt_G_and_dG(Om, fg, fl):
    sig = mp.mpf(fg) / (2 * mp.sqrt(2 * mp.log(2)))
    r = sig * mp.sqrt(2)
    y = (mp.mpf(fl) / 2 + Om) / r
    ex = mp.exp(y ** 2) * mp.erfc(y)
    c = mp.pi / (sig * mp.sqrt(2 * mp.pi))
    return c * ex, c * (2 * y * ex - 2 / mp.sqrt(mp.pi)) / r


# ---------------------------------------------------------------------------
# local_exponent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kern,b0", [(K.conservative(), 1), (K.kuramoto(), 0.5)])
def test_local_exponent_lorentzian(kern, b0):
    line = L.lorentzian(1.0)
    for Om in ("1e-9", "1e-4", "0.1", "2"):
        eps, b = P.local_exponent(line, kern, mp.mpf(Om))
        assert abs(b / (b0 / (1 + eps)) - 1) < mp.mpf("1e-13")
    # and eps itself: Omega/a for the conservative kernel
    eps, _ = P.local_exponent(line, K.conservative(), mp.mpf("1e-9"))
    assert abs(eps / mp.mpf("2e-9") - 1) < mp.mpf("1e-13")


def test_local_exponent_voigt():
    fg, fl = 1.0, 0.3
    line = L.voigt(fg, fl)
    for Om in ("1e-6", "1"):
        Om = mp.mpf(Om)
        eps, b = P.local_exponent(line, K.conservative(), Om)
        with mp.workdps(40):
            G0, _ = _voigt_G_and_dG(0, fg, fl)
            G, Gp = _voigt_G_and_dG(Om, fg, fl)
            want = -(G + Om * Gp) * (G0 - G) / (Om * Gp * G0)
            eps_x = G0 / G - 1
        assert abs(b / want - 1) < mp.mpf("1e-12")
        assert abs(eps / eps_x - 1) < mp.mpf("1e-12")


def test_local_exponent_against_the_branch():
    """Gaussian line, W = 1/(1+|u|^2.5): the local exponent against a
    central difference of log R over log eps along the branch, at 30
    digits (Omega (1 +- 1e-8)), and its approach to 2/3."""
    line, kern = L.gaussian(1.0), K.power_tail(2.5)
    with mp.workdps(30):
        for Om in ("1e-5", "1e-2"):
            Om = mp.mpf(Om)
            _, b = P.local_exponent(line, kern, Om)
            h = mp.mpf("1e-8")
            _, R1, e1 = P.branch_point(line, kern, Om * (1 - h))
            _, R2, e2 = P.branch_point(line, kern, Om * (1 + h))
            fd = mp.log(R2 / R1) / mp.log(e2 / e1)
            assert abs(b / fd - 1) < mp.mpf("1e-12")
    _, b = P.local_exponent(line, kern, mp.mpf("1e-7"))
    assert abs(b / (mp.mpf(2) / 3) - 1) < 2e-3


def test_local_exponent_refusals():
    with pytest.raises(ValueError, match="compact support"):
        P.local_exponent(L.box(1.0), K.power_tail(2.5), mp.mpf("1e-3"))
    with pytest.raises(ValueError, match="does not rise"):     # bends back
        P.local_exponent(L.bimodal_gaussian(4.0), K.conservative(), mp.mpf("1e-2"))
    with pytest.raises(ValueError, match="Omega"):
        P.local_exponent(L.gaussian(1.0), K.conservative(), 0)


# ---------------------------------------------------------------------------
# branch_at_coupling / branch_at_eps
# ---------------------------------------------------------------------------

def test_branch_at_coupling_lorentzian():
    line = L.lorentzian(1.0)
    a = mp.mpf(1) / 2
    for chi in ("0.5000001", "0.6", "3"):
        chi = mp.mpf(chi)
        chiN, R, eps, Om = P.branch_at_coupling(line, K.conservative(), chi)
        assert abs(chiN / chi - 1) < mp.mpf("1e-13")
        assert abs(Om / (chi - a) - 1) < mp.mpf("1e-7")   # chi - a cancels
        assert abs(R / (Om / (a + Om)) - 1) < mp.mpf("1e-13")
        chiN, R, eps, Om = P.branch_at_coupling(line, K.kuramoto(), 2 * chi)
        assert abs(chiN / (2 * chi) - 1) < mp.mpf("1e-13")
        assert abs(R / mp.sqrt(eps / (1 + eps)) - 1) < mp.mpf("1e-13")
    # eps given directly keeps the precision right at the threshold
    for e in ("1e-12", "1e-6", "0.3"):
        chiN, R, eps, Om = P.branch_at_eps(line, K.conservative(), mp.mpf(e))
        assert abs(eps / mp.mpf(e) - 1) < mp.mpf("1e-13")
        assert abs(Om / (a * mp.mpf(e)) - 1) < mp.mpf("1e-13")


def test_branch_at_eps_voigt_closed_form():
    """Invert the closed form G(Omega) by mpmath's root finder at 40
    digits and compare."""
    fg, fl = 1.0, 0.3
    line = L.voigt(fg, fl)
    for e in ("1e-3",):
        chiN, R, eps, Om = P.branch_at_eps(line, K.conservative(), mp.mpf(e))
        with mp.workdps(40):
            G0, _ = _voigt_G_and_dG(0, fg, fl)
            Ox = mp.findroot(lambda o: G0 / _voigt_G_and_dG(o, fg, fl)[0] - 1
                             - mp.mpf(e), Om)
            Gx = _voigt_G_and_dG(Ox, fg, fl)[0]
        assert abs(Om / Ox - 1) < mp.mpf("1e-12")
        assert abs(R / (Ox * Gx) - 1) < mp.mpf("1e-12")
        assert abs(chiN * Gx - 1) < mp.mpf("1e-13")


def test_branch_at_eps_round_trip():
    """Gaussian line with W = 1/(1+|u|^2.5): branch_point at Omega, then
    back from its eps."""
    line, kern = L.gaussian(1.0), K.power_tail(2.5)
    for Om in ("1e-6", "0.05"):
        Om = mp.mpf(Om)
        chi0, R0, e0 = P.branch_point(line, kern, Om)
        chiN, R, eps, Om2 = P.branch_at_eps(line, kern, e0)
        assert abs(Om2 / Om - 1) < mp.mpf("1e-12")
        assert abs(R / R0 - 1) < mp.mpf("1e-12")


def test_branch_at_refusals():
    line = L.lorentzian(1.0)
    with pytest.raises(ValueError, match="not above the threshold"):
        P.branch_at_coupling(line, K.conservative(), 0.5)
    with pytest.raises(ValueError, match="positive"):
        P.branch_at_eps(line, K.conservative(), -1e-3)
    with pytest.raises(ValueError, match="bends back|flat"):
        P.branch_at_eps(L.bimodal_gaussian(4.0), K.conservative(), 1e-3)
    with pytest.raises(ValueError, match="bends back|flat"):
        P.branch_at_eps(L.box(1.0), K.kuramoto(), 1e-3)


# ---------------------------------------------------------------------------
# eps_window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kern,tols", [(K.conservative(), (0.01,)),
                                       (K.kuramoto(), (0.01, 0.05))])
def test_eps_window_closed_form(kern, tols):
    """beta_local = beta/(1+eps) leaves the band rel_tol at
    eps = rel_tol/(1 - rel_tol)."""
    for tol in tols:
        eps_max, _ = P.eps_window(L.lorentzian(1.0), kern, tol, lo=-4)
        assert abs(eps_max / (tol / (1 - tol)) - 1) < 1e-8


def test_eps_window_refusals():
    with pytest.raises(ValueError, match="already"):
        P.eps_window(L.lorentzian(1.0), K.conservative(), 0.01, beta=0.5)
    with pytest.raises(ValueError, match="rel_tol"):
        P.eps_window(L.lorentzian(1.0), K.conservative(), 1.5)


# ---------------------------------------------------------------------------
# Speed-ups: bit-identical to the 1.3.0 paths
# ---------------------------------------------------------------------------

def _old_G0_minus_G(line, kernel, Omega):
    """1.3.0's _G0_minus_G: both passes evaluate the integrand anew."""
    Omega = mp.mpf(Omega)
    if Omega == 0:
        return mp.mpf(0)
    ratio = Omega / mp.mpf(line.scale)
    guard = 10
    if ratio < 1:
        guard += int(mp.ceil(-2 * mp.log10(ratio)))
    with mp.extradps(guard):
        p0 = line.p0()

    def f(u):
        with mp.extradps(guard):
            d = p0 - line.pdf(Omega * u)
        return (+d) * kernel.W(u)

    pts = P._half_points(line, kernel, Omega)
    val = P._even_quad(f, *pts)[0]
    if val == 0:
        return val
    size = abs(val)
    return size * P._even_quad(lambda u: f(u) / size, *pts)[0]


@pytest.fixture
def old_paths(monkeypatch):
    """Switch parametric to the 1.3.0 code paths: no remembered kernel
    values, and the old difference integral."""
    def use():
        monkeypatch.setattr(P, "_memo_kernel", lambda k: k)
        monkeypatch.setattr(P, "_G0_minus_G", _old_G0_minus_G)
    return use


CASES = [(L.lorentzian(1.0), K.conservative()),
         (L.gaussian(1.0), K.power_tail(2.5)),
         (L.student_t(3.0, 0.5), K.kuramoto()),
         (L.box(1.0), K.power_tail(4.0)),
         (L.gaussian(1.0), K.heterogeneous(K.kuramoto(), 3.8))]


def test_branch_point_and_sweep_bit_identical(old_paths):
    exps = (-2, -9)
    new = [(P.branch_point(l, k, mp.mpf("0.1")), P.sweep(l, k, exps))
           for l, k in CASES]
    old_paths()
    old = [(P.branch_point(l, k, mp.mpf("0.1")), P.sweep(l, k, exps))
           for l, k in CASES]
    assert new == old


def test_points_for_beta_same_as_counting():
    """1.3.0 counted n up from 3; the bisection must give the same n."""
    from lockkernel import beta_relative_sigma, points_for_beta

    def counted(tgt, dec, slog):
        n = 3
        while beta_relative_sigma(n, dec, slog) > tgt:
            n += 1
        return n, beta_relative_sigma(n, dec, slog)

    for tgt, dec, slog in [(0.01, 2.0, 0.05), (0.3, 1.0, 0.1), (1.0, 3.0, 0.01),
                           (1e-3, 2.0, 0.05), (2.3e-3, 0.7, 0.02),
                           (0.004, 5.0, 0.2)]:
        assert points_for_beta(tgt, dec, slog) == counted(tgt, dec, slog)
    with pytest.raises(ValueError, match="10\\^7"):
        points_for_beta(1e-6, 2.0, 0.05)
