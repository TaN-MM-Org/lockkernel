"""New in 1.5.0: the exact branch as a checked table (branch_table), and
fits of measured data with it (fit_exact_branch).

References, all independent of the package's integrals:

* Lorentzian line, Kuramoto kernel: R = sqrt(eps/(1+eps)) (README example 7),
  so d log R / d log eps = (1/2)/(1+eps); threshold 2a = FWHM.
* Gaussian line of standard deviation sigma, Kuramoto kernel:
  G(Omega) = p(0) int_{-1}^{1} exp(-Omega^2 u^2/(2 sigma^2)) sqrt(1-u^2) du
  = p(0) (pi/2) exp(-z) [I0(z) + I1(z)], z = Omega^2/(4 sigma^2).  Derivation:
  u = sin(phi), then sin^2 = (1 - cos 2phi)/2, cos^2 = (1 + cos 2phi)/2 and
  theta = 2 phi turn the integral into (1/4) exp(-z) int_{-pi}^{pi}
  exp(z cos theta) (1 + cos theta) d theta, and I_n(z) = (1/pi) int_0^pi
  exp(z cos theta) cos(n theta) d theta (the integral representation of the
  modified Bessel function of integer order).  The test also checks the
  resulting formula against 40-digit mpmath quadrature of the integral.  Then eps = G(0)/G - 1,
  R = Omega G, G(0) = p(0) pi/2.
* For the fits: an independent least-squares fit (scipy) of the same data
  with the closed-form model, and the scatter of seeded fits.
"""
import mpmath as mp
import numpy as np
import pytest
from scipy.optimize import least_squares

from lockkernel import fit_branch, fit_exact_branch
from lockkernel import kernels as K, lineshapes as L, parametric as P

REF = "synthetic: closed-form branch with seeded noise"


@pytest.fixture(scope="module")
def kur_table():
    """Lorentzian line (FWHM 1), Kuramoto kernel, eps 1e-4 .. 1."""
    with mp.workdps(15):
        return P.branch_table(L.lorentzian(1.0), K.kuramoto(), 1e-4, 1.0,
                              rel_tol=1e-7)


def _kur_data(seed, k=2.5e-3, q=1.0, noise=0.01, n=12, eps_hi=0.5):
    """Recorded couplings chi = chiN / k and R = q sqrt(eps/(1+eps)) with
    seeded log-normal noise, on the Lorentzian-Kuramoto branch (chi_c = 1)."""
    rng = np.random.default_rng(seed)
    eps = np.geomspace(1e-3, eps_hi, n)
    r = q * np.sqrt(eps / (1 + eps)) * np.exp(rng.normal(0.0, noise, n))
    return (1 + eps) / k, r


# ---------------------------------------------------------------------------
# branch_table
# ---------------------------------------------------------------------------

def test_branch_table_lorentzian_closed_form(kur_table):
    """Values everywhere and slopes at the nodes against the closed form.

    The cubic Hermite error term is h^4 f''''(xi) / 384 at most, which the
    midpoint check samples where it is largest; between check points it
    can exceed the checked value only through the change of f'''' across
    one interval, so 2 rel_tol is allowed (observed: 0.96 rel_tol)."""
    tab = kur_table
    assert tab.max_error <= tab.rel_tol == 1e-7
    assert abs(tab.chi_c - 1.0) < 1e-14
    assert abs(tab.eps_min / 1e-4 - 1) < 1e-12 and abs(tab.eps_max - 1) < 1e-12
    e = np.geomspace(tab.eps_min, tab.eps_max, 3001)
    want = 0.5 * np.log(e / (1 + e))
    lr, slope = tab.log_R_and_slope(e)
    assert np.max(np.abs(lr - want)) < 2 * tab.rel_tol
    assert np.max(np.abs(np.log(tab.R(e)) - want)) < 2 * tab.rel_tol
    xn = np.array(tab.x)
    en = np.exp(xn)
    assert np.max(np.abs(np.array(tab.y) - 0.5 * np.log(en / (1 + en)))) < 1e-13
    assert np.max(np.abs(np.array(tab.slope) / (0.5 / (1 + en)) - 1)) < 1e-12


def _gauss_kuramoto(Om, fwhm):
    """(eps, R) of the Kuramoto branch on a Gaussian line, Bessel form."""
    sig = mp.mpf(fwhm) / (2 * mp.sqrt(2 * mp.log(2)))
    p0 = 1 / (sig * mp.sqrt(2 * mp.pi))
    z = Om ** 2 / (4 * sig ** 2)
    G = p0 * mp.pi / 2 * mp.exp(-z) * (mp.besseli(0, z) + mp.besseli(1, z))
    return p0 * mp.pi / 2 / G - 1, Om * G, G, p0, sig


def test_branch_table_gaussian_kuramoto_closed_form():
    """A second line: the Bessel form of the branch (checked here against
    40-digit quadrature of its integral), at 25 values of Omega spanning the
    table, on a Gaussian of FWHM 2.5 (the table depends only on the line's
    shape: scaling the line by a maps Omega to a Omega and leaves the curve
    R(eps) unchanged)."""
    with mp.workdps(15):
        tab = P.branch_table(L.gaussian(2.5), K.kuramoto(), 1e-3, 0.5,
                             rel_tol=1e-7)
    assert tab.max_error <= 1e-7
    with mp.workdps(40):
        _, _, G, p0, sig = _gauss_kuramoto(mp.mpf("0.7"), 2.5)
        Gq = p0 * mp.quad(lambda u: mp.exp(-(mp.mpf("0.7") * u) ** 2
                                           / (2 * sig ** 2))
                          * mp.sqrt(1 - u ** 2), [-1, 0, 1])
        assert abs(G / Gq - 1) < mp.mpf("1e-35")
        pts = []
        for Om in np.geomspace(0.15, 3.2, 25):
            e, R, _, _, _ = _gauss_kuramoto(mp.mpf(Om), 2.5)
            if tab.eps_min <= e <= tab.eps_max:
                pts.append((float(e), float(R)))
    assert len(pts) >= 20
    e, R = np.array(pts).T
    assert np.max(np.abs(np.log(tab.R(e)) - np.log(R))) < 2e-7


def test_branch_table_refusals(kur_table):
    with pytest.raises(ValueError, match="compact support"):
        P.branch_table(L.box(1.0), K.power_tail(2.5), 1e-3, 0.1)
    with pytest.raises(ValueError, match="bends back|flat"):
        P.branch_table(L.bimodal_gaussian(4.0), K.conservative(), 1e-3, 0.1)
    with pytest.raises(ValueError, match="eps_min < eps_max"):
        P.branch_table(L.lorentzian(1.0), K.kuramoto(), 0.1, 1e-3)
    with pytest.raises(ValueError, match="eps_min < eps_max"):
        P.branch_table(L.lorentzian(1.0), K.kuramoto(), 0.0, 1e-3)
    with pytest.raises(ValueError, match="rel_tol"):
        P.branch_table(L.lorentzian(1.0), K.kuramoto(), 1e-3, 0.1, rel_tol=0.1)
    with pytest.raises(ValueError, match="step"):
        P.branch_table(L.lorentzian(1.0), K.kuramoto(), 1e-3, 0.1, step=0)
    with pytest.raises(ValueError, match="wider range"):
        kur_table.R(2.0)
    with pytest.raises(ValueError, match="wider range"):
        kur_table.R([1e-3, 1e-5])


# ---------------------------------------------------------------------------
# fit_exact_branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fit_q,q", [(False, 1.0), (True, 0.9)])
def test_fit_matches_a_closed_form_fit(kur_table, fit_q, q):
    """The same data fitted by scipy with the closed form R = q sqrt(eps/(1+
    eps)), eps = k chi - 1: k, q, their error bars and chi2 agree.

    The table differs from the closed form by at most d = 2e-7 in log R
    (test above), which moves each weighted residual by d/sigma_log = 2e-5:
    the parameters by at most about sqrt(n) d/sigma_log = 7e-5 of their
    error bars, and chi2 by at most 2 sqrt(n chi2) d/sigma_log + n
    (d/sigma_log)^2 (below 1e-3 here).  The tolerances are these bounds."""
    chi, r = _kur_data(3, q=q)
    f = fit_exact_branch(chi, r, REF, kur_table, sigma_r=0.01 * r,
                         fit_r_scale=fit_q)

    def res(p):
        e = np.exp(p[0]) * chi - 1
        return ((p[1] if fit_q else 0.0) + 0.5 * np.log(e / (1 + e))
                - np.log(r)) / 0.01

    o = least_squares(res, [np.log(2.5e-3)] + ([0.0] if fit_q else []),
                      x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    cov = np.linalg.inv(o.jac.T @ o.jac)
    sig = np.sqrt(np.diag(cov))
    d = 2e-7 / 0.01
    assert abs(np.log(f.chi_scale) - o.x[0]) < np.sqrt(12) * d * sig[0]
    assert abs(f.sigma_chi_scale / f.chi_scale / sig[0] - 1) < 1e-4
    assert abs(f.chi_c - 1 / f.chi_scale) < 1e-12 * f.chi_c
    chi2 = 2 * o.cost
    assert abs(f.chi2 - chi2) < 2 * np.sqrt(12 * chi2) * d + 12 * d ** 2
    assert f.dof == 12 - (2 if fit_q else 1)
    if fit_q:
        assert abs(np.log(f.r_scale) - o.x[1]) < np.sqrt(12) * d * sig[1]
        assert abs(f.sigma_r_scale / f.r_scale / sig[1] - 1) < 1e-4
    else:
        assert f.r_scale == 1.0 and f.sigma_r_scale == 0.0


def test_fit_error_bars_against_seeded_fits(kur_table):
    """400 seeded data sets (12 points, eps 1e-3 .. 0.5, 1 % noise), k and q
    fitted: the scatter of log k and log q matches the median reported
    error bar within 12 % (the sampling error of a standard deviation from
    400 fits is about 3.5 %), the means lie within 4 standard errors of the
    truth, and chi2 averages to dof within 4 standard errors."""
    lk, lq, sk, sq, c2 = [], [], [], [], []
    for seed in range(400):
        chi, r = _kur_data(1000 + seed, q=0.8)
        f = fit_exact_branch(chi, r, REF, kur_table, sigma_r=0.01 * r,
                             fit_r_scale=True)
        lk.append(np.log(f.chi_scale))
        lq.append(np.log(f.r_scale))
        sk.append(f.sigma_chi_scale / f.chi_scale)
        sq.append(f.sigma_r_scale / f.r_scale)
        c2.append(f.chi2)
    lk, lq = np.array(lk), np.array(lq)
    for v, s, truth in ((lk, sk, np.log(2.5e-3)), (lq, sq, np.log(0.8))):
        assert abs(np.std(v) / np.median(s) - 1) < 0.12
        assert abs(np.mean(v) - truth) < 4 * np.std(v) / np.sqrt(v.size)
    assert abs(np.mean(c2) - 10) < 4 * np.sqrt(2 * 10 / 400)


def test_wrong_model_is_rejected(kur_table):
    """Kuramoto data over eps 1e-3 .. 0.5: the conservative kernel's exact
    branch (on the same line, default rel_tol) is rejected, while the power
    law of fit_branch on the same points is biased and also fails its
    chi-square test, which is why the whole range needs the exact branch."""
    with mp.workdps(15):
        wrong = P.branch_table(L.lorentzian(1.0), K.conservative(), 1e-4, 1.0)
    chi, r = _kur_data(8)
    good = fit_exact_branch(chi, r, REF, kur_table, sigma_r=0.01 * r)
    bad = fit_exact_branch(chi, r, REF, wrong, sigma_r=0.01 * r)
    assert good.p_value > 0.01 and bad.p_value < 1e-10
    assert abs(good.chi_c - 400.0) < 4 * good.sigma_chi_c
    pl = fit_branch(chi, r, REF, sigma_r=0.01 * r)
    assert pl.p_value < 1e-10 and abs(pl.beta - 0.5) > 10 * pl.sigma_beta


def test_fit_without_sigma_r_is_residual_scaled(kur_table):
    """Without sigma_r the error bars are those of the fit with sigma_r,
    scaled by sqrt(chi2/dof), and there is no chi2 test."""
    chi, r = _kur_data(4)
    a = fit_exact_branch(chi, r, REF, kur_table, sigma_r=0.01 * r)
    b = fit_exact_branch(chi, r, REF, kur_table)
    assert b.chi2 is None and b.p_value is None
    assert abs(b.chi_scale / a.chi_scale - 1) < 1e-9
    scale = np.sqrt(a.chi2 / a.dof)
    assert abs(b.sigma_chi_scale / (a.sigma_chi_scale * scale) - 1) < 1e-6


def test_fit_exact_branch_refusals(kur_table):
    chi, r = _kur_data(5)
    with pytest.raises(ValueError, match="reference"):
        fit_exact_branch(chi, r, "x", kur_table)
    with pytest.raises(ValueError, match=">= 5"):
        fit_exact_branch(chi[:4], r[:4], REF, kur_table)
    with pytest.raises(ValueError, match="increasing"):
        fit_exact_branch(chi[::-1], r, REF, kur_table)
    with pytest.raises(ValueError, match="> 0"):
        fit_exact_branch(chi, -r, REF, kur_table)
    with pytest.raises(ValueError, match="sigma_r"):
        fit_exact_branch(chi, r, REF, kur_table, sigma_r=0 * r)
    # a table too coarse for 1e-6 errors of R
    with pytest.raises(ValueError, match="rel_tol"):
        fit_exact_branch(chi, r, REF, kur_table, sigma_r=1e-6 * r)
    # couplings spanning more than the table holds
    eps = np.geomspace(1e-3, 3.0, 12)
    with pytest.raises(ValueError, match="wider eps range"):
        fit_exact_branch((1 + eps) / 2.5e-3, np.sqrt(eps / (1 + eps)), REF,
                         kur_table)
    # data whose best fit needs eps below the table's start
    eps = np.geomspace(1e-6, 0.5, 12)
    with pytest.raises(ValueError, match="edge of the table"):
        fit_exact_branch((1 + eps) / 2.5e-3, np.sqrt(eps / (1 + eps)), REF,
                         kur_table, sigma_r=0.01 * np.sqrt(eps / (1 + eps)))
