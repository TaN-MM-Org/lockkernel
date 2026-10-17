"""Fit a MEASURED synchronization branch: threshold, exponent, kernel.

The theory half of this package computes the branch exactly from a
known line shape and a known locking kernel.  This module is the lab
half: given measured (coupling, order parameter) points near an onset
-- from an experiment or from a simulation whose kernel is NOT known
-- it fits the near-threshold law

    R = A (chiN / chiN_c - 1)^beta,        chiN > chiN_c,

for the threshold chiN_c, the exponent beta and the amplitude A
together, with error bars, and translates beta back into the locking
kernel's tail exponent through the exact dictionary of
`lockkernel.parametric`:

    beta = 1/(s-1)  for 1 < s < 3,        beta = 1/2  for s >= 3,

so s = 1 + 1/beta wherever beta > 1/2, while beta consistent with 1/2
identifies only the CLASS s >= 3 (compact support or fast decay) --
`kernel_tail_from_beta` refuses to name a single s there, because the
data cannot single one out.

House rules, as everywhere in this organization: measured data carry
a mandatory `reference` (instrument or simulation provenance), the
fit REFUSES -- with the reason -- rather than returning numbers it
cannot defend (too few points, couplings not increasing, less than a
decade of reduced coupling, a threshold estimate that collides with
the data, a fit that does not converge), and every closed form in
this module is held in the tests against the exact parametric branch
of `lockkernel.parametric` and against seeded Monte Carlo -- the
theory half checks the lab half.

Two lab-side additions (1.3.0).  `fit_branch(..., sigma_chi=...)` takes
the uncertainty of each measured COUPLING into account, which near the
threshold dominates: R ~ (chi - chi_c)^beta is steep there, so a small
error in chi moves log R by beta sigma_chi / (chi - chi_c).  It is
propagated to first order (the effective-variance method: each point's
variance of log R gains that term, and the fit is iterated until the
weights settle) and refused where first order is not enough.  And
`with_calibration` adds the uncertainty of an overall calibration SCALE of
the couplings or of R to the error bars: such a scale moves chi_c or A
and, exactly, nothing else (eps = chi/chi_c - 1 does not change when every
chi and chi_c are scaled together), so beta is immune to both.  With
`sigma_r` given, every fit also reports chi^2, its degrees of freedom and
the p-value, a check of whether the data are consistent with the power law
at the stated noise.

`points_for_beta` plans the measurement before it is taken: the
closed-form error bar of a log-log slope over a given range of
reduced coupling, inverted exactly for the number of points a target
error bar costs.  `plan_fit` and `points_for_fit` do the same for the
three-parameter fit `fit_branch` actually performs, where the
threshold is fitted too.

Honest limits: the power law is the LEADING near-threshold behavior;
data taken far above threshold bend away from it and bias beta -- fit
what is near the onset (the eps_max refusal guards the worst of it,
and the residual diagnostics are returned so the bend is visible).
Error bars are the standard asymptotic (J^T J)^-1 covariance of the
weighted fit: honest when the model holds and the noise estimate is
right, and stated as such.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import least_squares

__all__ = ["BranchFit", "fit_branch", "kernel_tail_from_beta",
           "beta_relative_sigma", "points_for_beta", "FitPlan", "plan_fit",
           "points_for_fit", "with_calibration"]

# search limits of the exponent in `fit_branch`
_BETA_MIN, _BETA_MAX = 0.05, 20.0
# largest sigma_chi / (chi - chi_c) for which `fit_branch` propagates the
# coupling uncertainty to first order.  The tests check the error bars
# against seeded fits at 0.1; by hand, 1000 seeded fits at 0.1, 0.2 and 0.3
# gave error bars within 6 % of the scatter, for coupling noise added to
# the true or to the recorded coupling.
_SIGMA_CHI_MAX_RATIO = 0.25


@dataclass(frozen=True)
class BranchFit:
    """The fitted near-threshold law, with provenance.

    chi_c, beta, amplitude and their 1-sigma errors; n_points; eps_range,
    the (smallest, largest) fitted eps = chi/chi_c - 1; residual_rms_log,
    the root-mean-square misfit in log R; reference, where the data come
    from.

    Added in 1.3.0 (all with defaults, so existing code is unaffected):
    dof = n_points - 3; chi2 and p_value, the weighted sum of squared
    residuals and the probability of a chi2 at least that large if the
    model and the stated errors are right (None when sigma_r was not given,
    because the scatter of the points then sets the error scale and there
    is nothing to test); sigma_chi_used, True when coupling errors were
    propagated; calibration_rel, the relative calibration-scale errors
    (of chi, of R) folded in by `with_calibration`.
    """
    chi_c: float
    beta: float
    amplitude: float
    sigma_chi_c: float
    sigma_beta: float
    sigma_amplitude: float
    n_points: int
    eps_range: tuple
    residual_rms_log: float
    reference: str
    dof: int = 0
    chi2: Optional[float] = None
    p_value: Optional[float] = None
    sigma_chi_used: bool = False
    calibration_rel: Tuple[float, float] = (0.0, 0.0)


def _check_reference(reference: str) -> None:
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError("measured branch data require a real "
                         "reference (instrument or simulation "
                         "provenance, >= 8 characters); this package "
                         "does not fit uncited data")


def fit_branch(chi: ArrayLike, r: ArrayLike, reference: str,
               sigma_r: Optional[ArrayLike] = None, min_decade: float = 1.0,
               sigma_chi: Optional[ArrayLike] = None) -> BranchFit:
    """Fit R = A (chi/chi_c - 1)^beta to measured points.

    chi : couplings chiN, strictly increasing, all above threshold.
    r : measured order parameters at those couplings, all > 0.
    reference : where the data come from (mandatory, >= 8 chars).
    sigma_r : absolute 1-sigma errors of r (same shape); when given,
        the covariance is exact for that noise; when omitted, the
        residual scatter sets the scale and the error bars say so by
        construction (residual-scaled).
    min_decade : the minimum decades of reduced coupling
        eps = chi/chi_c - 1 the fitted points must span (default one
        decade); a narrower fit cannot separate the exponent from the
        amplitude and is refused.
    sigma_chi : absolute 1-sigma errors of the couplings (same shape as
        chi, >= 0; new in 1.3.0).  Requires sigma_r.  Near the threshold
        an error in chi moves log R by beta sigma_chi / (chi - chi_c), so
        each point's variance of log R becomes
            (sigma_r / r)^2 + (beta sigma_chi / (chi - chi_c))^2
        at the fitted chi_c and beta, and the fit is repeated with these
        weights until they settle (the effective-variance method, first
        order in sigma_chi).  Refused when sigma_chi exceeds 0.25 (chi -
        chi_c) at any point, where first order is not enough: drop the
        points closest to the threshold or measure the coupling better.

    Returns a `BranchFit`.  With sigma_r it also carries chi2, dof =
    n - 3 and p_value (the chi-square probability of a misfit at least
    this large if the power law and the stated errors are right; a very
    small p_value says the data are not consistent with the law at that
    noise, often because the range reaches too far above the threshold).

    Refusals: fewer than 6 points, couplings not finite, positive and
    strictly increasing, non-positive R, a converged threshold
    indistinguishable from the smallest coupling (the data do not
    reach below the bend), a span of eps under `min_decade` decades,
    a fit that does not converge, an exponent that runs into the
    search limits 0.05 or 20 (the data are not a power law of this
    kind; before 1.3.0 the limit itself was returned as the fitted
    beta, with an error bar), and the sigma_chi cases above.
    """
    _check_reference(reference)
    x = np.asarray(chi, dtype=float)
    y = np.asarray(r, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("chi and r must be one-dimensional sequences")
    if x.size < 6 or x.size != y.size:
        raise ValueError("need >= 6 (chi, R) points of equal length: "
                         "three parameters and error bars cannot come "
                         "from fewer")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("couplings must be finite and strictly "
                         "increasing")
    if x[0] <= 0.0:
        raise ValueError("couplings must be positive: the threshold "
                         "chiN_c = 1/(p(0) m) is positive and the fitted "
                         "points lie above it")
    if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("order parameters must be finite and > 0: "
                         "points below threshold (R = 0) carry no "
                         "exponent information in this law; drop them")
    if not (np.isfinite(float(min_decade)) and float(min_decade) >= 0.0):
        raise ValueError("min_decade must be finite and >= 0")
    if sigma_r is not None:
        sy = np.asarray(sigma_r, dtype=float)
        if sy.shape != y.shape or np.any(~(sy > 0.0)) \
                or np.any(~np.isfinite(sy)):
            raise ValueError("sigma_r must be finite and positive on the "
                             "same grid as r")
        w_log = sy / y          # d(log R) = dR / R
    else:
        w_log = np.ones_like(y)
    if sigma_chi is not None:
        if sigma_r is None:
            raise ValueError(
                "sigma_chi needs sigma_r: the coupling errors are added "
                "to the stated errors of R, which must then be given")
        sx = np.asarray(sigma_chi, dtype=float)
        if sx.shape != x.shape or np.any(~np.isfinite(sx)) \
                or np.any(sx < 0.0):
            raise ValueError("sigma_chi must be finite and >= 0 on the "
                             "same grid as chi")
        w_r = w_log

    ly = np.log(y)
    chi_min = x[0]

    def solve(w, p_start):
        def resid(p):
            chi_c, beta, la = p
            eps = x / chi_c - 1.0
            return (la + beta * np.log(eps) - ly) / w

        def jac(p):
            # analytic derivatives of resid.  The default finite-difference
            # Jacobian of least_squares steps chi_c by about 1e-8 relative,
            # which is not small next to eps = 1e-7: before 1.2.0 the
            # reported sigma_chi_c came out up to 17 % low that close to
            # threshold (and sigma_beta about 2 % low).
            chi_c, beta, la = p
            eps = x / chi_c - 1.0
            return np.stack([-beta * x / (chi_c ** 2 * eps), np.log(eps),
                             np.ones_like(x)], axis=1) / w[:, None]

        out = least_squares(resid, p_start, jac=jac,
                            bounds=([1e-300, _BETA_MIN, -np.inf],
                                    [chi_min * (1 - 1e-9), _BETA_MAX,
                                     np.inf]))
        if not out.success:
            raise ValueError(f"the branch fit did not converge: "
                             f"{out.message}")
        return out

    # start: threshold a little under the first point, slope from the
    # outermost pair
    eps0 = x / (0.95 * chi_min) - 1.0
    b0 = (ly[-1] - ly[0]) / np.log(eps0[-1] / eps0[0])
    b0 = float(np.clip(b0, 0.1, 10.0))
    p0 = np.array([0.95 * chi_min, b0,
                   ly[-1] - b0 * np.log(eps0[-1])])
    sol = solve(w_log, p0)
    if sigma_chi is not None and np.any(sx > 0.0):
        # effective variance: re-weight with the fitted chi_c and beta
        # until the weights settle
        for _ in range(100):
            chi_c, beta, _la = sol.x
            if chi_c >= chi_min * (1.0 - 1e-8):
                break               # refused below
            w_new = np.sqrt(w_r ** 2 + (beta * sx / (x - chi_c)) ** 2)
            if np.max(np.abs(w_new / w_log - 1.0)) < 1e-10:
                break
            w_log = w_new
            sol = solve(w_log, sol.x)
        else:
            raise ValueError("the weights of the coupling errors did not "
                             "settle in 100 refits")
    chi_c, beta, la = sol.x
    if chi_c >= chi_min * (1.0 - 1e-8):
        raise ValueError(
            "the fitted threshold is indistinguishable from the "
            "smallest measured coupling: the data do not resolve the "
            "onset. Measure closer to threshold (smaller R), or "
            "supply points at lower coupling")
    if chi_c <= chi_min * 1e-12:
        raise ValueError(
            "the fitted threshold ran to zero: the data do not show an "
            "onset at a positive coupling")
    if beta <= _BETA_MIN * (1 + 1e-6) or beta >= _BETA_MAX * (1 - 1e-6):
        raise ValueError(
            f"the fitted exponent ran into the search limit "
            f"(beta = {beta:.4g}; the fit searches {_BETA_MIN:g} < beta "
            f"< {_BETA_MAX:g}): the data are not described by "
            "R = A (chi/chi_c - 1)^beta. An exponent near 0 means R "
            "hardly changes with chi (a jump at the threshold, or points "
            "far above it)")
    if sigma_chi is not None:
        ratio = sx / (x - chi_c)
        if np.max(ratio) > _SIGMA_CHI_MAX_RATIO:
            k = int(np.argmax(ratio))
            raise ValueError(
                f"sigma_chi is {ratio[k]:.2f} of chi - chi_c at chi = "
                f"{x[k]:.6g} (limit {_SIGMA_CHI_MAX_RATIO}): the coupling "
                "error is not small next to the distance from the "
                "threshold, and first-order propagation does not hold. "
                "Drop the points closest to the threshold or measure the "
                "coupling more precisely")
    eps = x / chi_c - 1.0
    decades = np.log10(eps[-1] / eps[0])
    if decades < float(min_decade):
        raise ValueError(
            f"the fitted points span only {decades:.2f} decades of "
            f"reduced coupling eps (minimum {float(min_decade):.2f}): "
            "exponent and amplitude are not separable on so short a "
            "lever arm. Extend the coupling range")

    # asymptotic covariance of the weighted fit
    jac = sol.jac
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        raise ValueError("the fit covariance is singular: the data "
                         "do not constrain all three parameters "
                         "independently")
    dof = x.size - 3
    res = sol.fun
    chi2 = float(res @ res)
    if sigma_r is None and dof > 0:
        cov = cov * chi2 / dof   # residual-scaled
    sig = np.sqrt(np.diag(cov))
    return BranchFit(
        chi_c=float(chi_c), beta=float(beta),
        amplitude=float(np.exp(la)),
        sigma_chi_c=float(sig[0]), sigma_beta=float(sig[1]),
        sigma_amplitude=float(np.exp(la) * sig[2]),
        n_points=int(x.size),
        eps_range=(float(eps[0]), float(eps[-1])),
        residual_rms_log=float(np.sqrt(np.mean((res * w_log) ** 2))),
        reference=reference,
        dof=int(dof),
        chi2=chi2 if sigma_r is not None else None,
        p_value=(float(stats.chi2.sf(chi2, dof)) if sigma_r is not None
                 else None),
        sigma_chi_used=sigma_chi is not None)


def with_calibration(fit: BranchFit, chi_scale_rel: float = 0.0,
                     r_scale_rel: float = 0.0) -> BranchFit:
    """Add calibration-scale uncertainties to a fit's error bars.

    chi_scale_rel : relative 1-sigma uncertainty of an overall scale
        factor common to all couplings (for example the conversion of a
        pump power or a field into chiN).
    r_scale_rel : the same for the order parameter R (for example a
        detection efficiency or a contrast normalization).

    Scaling every coupling by k leaves eps = chi/chi_c - 1 unchanged when
    chi_c scales by k as well, so the fitted chi_c scales by k and beta
    and A do not change at all; scaling every R by k scales A by k and
    changes nothing else.  These are exact properties of the fitted law
    (checked in the tests by refitting scaled data).  So the calibration
    uncertainties are added in quadrature,

        sigma_chi_c -> sqrt(sigma_chi_c^2 + (chi_c chi_scale_rel)^2),
        sigma_A     -> sqrt(sigma_A^2 + (A r_scale_rel)^2),

    and sigma_beta is returned unchanged: the exponent is immune to both
    scale calibrations.  Returns a new `BranchFit` that records the
    values in `calibration_rel`.  Refuses negative or non-finite values,
    and a fit that already has a calibration folded in (it would be
    counted twice).
    """
    if not isinstance(fit, BranchFit):
        raise TypeError("with_calibration needs a BranchFit from fit_branch")
    cx, cr = float(chi_scale_rel), float(r_scale_rel)
    if not (np.isfinite(cx) and np.isfinite(cr) and cx >= 0.0
            and cr >= 0.0):
        raise ValueError("chi_scale_rel and r_scale_rel must be finite "
                         "and >= 0")
    if tuple(fit.calibration_rel) != (0.0, 0.0):
        raise ValueError(
            f"this fit already includes calibration errors "
            f"{fit.calibration_rel}; apply with_calibration once, to the "
            "fit returned by fit_branch")
    return dataclasses.replace(
        fit,
        sigma_chi_c=float(np.hypot(fit.sigma_chi_c, fit.chi_c * cx)),
        sigma_amplitude=float(np.hypot(fit.sigma_amplitude,
                                       fit.amplitude * cr)),
        calibration_rel=(cx, cr))


def kernel_tail_from_beta(beta: float, sigma_beta: float = 0.0,
                          n_sigma: float = 2.0) -> Tuple[float, float]:
    """The kernel tail exponent s from a measured beta: s = 1 + 1/beta.

    Error propagation: sigma_s = sigma_beta / beta^2.

    Refuses to name a single s when beta is consistent with 1/2
    within `n_sigma` error bars: every kernel with s >= 3 (compact
    support, Gaussian, any decay faster than |u|^-3) produces
    beta = 1/2, so the measurement identifies only the class.  Also
    refuses beta significantly BELOW 1/2, which no locking kernel
    produces on a line with a rounded maximum at its centre -- usually
    a sign the fitted range left the near-threshold regime.  (A line
    that is flat at its centre, such as a box, does give
    beta = 1/(s-1) < 1/2 for a tail s > 3; this function assumes a
    rounded line and does not cover that case.)
    """
    b = float(beta)
    sb = abs(float(sigma_beta))
    if not (np.isfinite(b) and b > 0.0):
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(sb):
        raise ValueError("sigma_beta must be finite")
    if not (np.isfinite(float(n_sigma)) and float(n_sigma) >= 0.0):
        raise ValueError("n_sigma must be finite and >= 0")
    if b + n_sigma * sb < 0.5:
        raise ValueError(
            f"beta = {b:.4g} +- {sb:.4g} lies significantly below "
            "1/2, which no locking kernel produces in the dictionary "
            "beta = 1/(s-1) (1 < s < 3), 1/2 (s >= 3) of a line with "
            "a rounded centre. The fitted range has likely left the "
            "near-threshold regime; refit closer to the onset")
    if b - n_sigma * sb <= 0.5:
        raise ValueError(
            f"beta = {b:.4g} +- {sb:.4g} is consistent with 1/2, "
            "which identifies only the CLASS s >= 3 (compact support "
            "or decay faster than |u|^-3); no single tail exponent "
            "can be named from it")
    s = 1.0 + 1.0 / b
    if s <= 1.0 or s >= 3.0:
        raise ValueError(f"implied s = {s:.4g} outside the algebraic "
                         "window 1 < s < 3")
    return s, sb / b ** 2


def _count(n_points, what: str) -> int:
    """An integer count: refuses a non-integral value instead of
    truncating it (before 1.3.0, 12.9 points were silently taken as 12)."""
    try:
        n = int(n_points)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what} must be an integer, got {n_points!r}")
    if n != n_points:
        raise ValueError(f"{what} must be an integer, got {n_points!r}")
    return n


def beta_relative_sigma(n_points: int, decades: float,
                        sigma_log: float) -> float:
    """Closed-form 1-sigma error of a log-log slope.

    n_points : measured couplings, log-uniform over the eps range.
    decades : decades of reduced coupling spanned.
    sigma_log : per-point scatter of log R (for small relative noise,
        sigma_log ~ sigma_R / R).

    For n equally spaced abscissas spanning L = decades * ln 10,

        sigma_beta = sigma_log / sqrt( L^2 n (n+1) / (12 (n-1)) ),

    the exact variance of the least-squares slope; the tests hold it
    against seeded simulation.  This is the KNOWN-threshold error --
    fitting chi_c together with beta (as `fit_branch` does) is
    strictly harder; `plan_fit` gives that error bar.
    """
    n = _count(n_points, "n_points")
    if n < 3:
        raise ValueError("need >= 3 points for a slope with an error "
                         "bar")
    d = float(decades)
    s = float(sigma_log)
    if not (np.isfinite(d) and np.isfinite(s) and d > 0.0 and s > 0.0):
        raise ValueError("decades and sigma_log must be finite and "
                         "positive")
    ell = d * np.log(10.0)
    sxx = ell ** 2 * n * (n + 1) / (12.0 * (n - 1))
    return float(s / np.sqrt(sxx))


def points_for_beta(target_sigma_beta: float, decades: float,
                    sigma_log: float) -> Tuple[int, float]:
    """Points needed for a target error bar on beta.

    Exact inversion of `beta_relative_sigma`: returns (n, achieved)
    with achieved <= target and n-1 failing it (asserted two-sided in
    the tests).  Refuses a target that would need more than 10^7 points.

    The error bar falls strictly as n grows (n (n+1)/(n-1) increases for
    n >= 3), so since 1.4.0 the smallest n is found by bisection instead
    of counting up from 3: the same n (checked in the tests against the
    old count), found in about 2 log2(n) steps instead of n; a refusal
    at 10^7 points now takes microseconds instead of about a minute.
    """
    tgt = float(target_sigma_beta)
    if not (np.isfinite(tgt) and tgt > 0.0):
        raise ValueError("target_sigma_beta must be positive")

    def sb(n):
        return beta_relative_sigma(n, decades, sigma_log)

    lo, cap = 3, 10 ** 7
    if sb(lo) <= tgt:
        return lo, sb(lo)
    hi = 2 * lo
    while sb(hi) > tgt:         # doubling, then bisection
        if hi == cap:
            raise ValueError(
                "over 10^7 points would be needed: the target is out "
                "of reach at this noise and range; widen the range "
                "or reduce the noise")
        lo, hi = hi, min(2 * hi, cap)
    while hi - lo > 1:          # sb(lo) > tgt >= sb(hi)
        mid = (lo + hi) // 2
        if sb(mid) > tgt:
            lo = mid
        else:
            hi = mid
    return hi, sb(hi)


@dataclass(frozen=True)
class FitPlan:
    """Error bars a planned measurement will give (see `plan_fit`).

    sigma_beta is absolute; sigma_chi_c_rel and sigma_amplitude_rel are
    relative (sigma_chi_c / chi_c and sigma_A / A).
    """
    n_points: int
    eps_range: tuple
    beta: float
    sigma_log: float
    fit_threshold: bool
    sigma_beta: float
    sigma_chi_c_rel: float
    sigma_amplitude_rel: float


def plan_fit(n_points: int, eps_min: float, eps_max: float, beta: float,
             sigma_log: float, fit_threshold: bool = True) -> FitPlan:
    """The error bars `fit_branch` will report, before measuring.

    n_points : couplings to measure, spaced evenly in log(eps) from
        eps_min to eps_max, where eps = chi/chi_c - 1 (both ends
        included).  In practice, a first rough estimate of chi_c places
        them.
    beta : the exponent you expect.  It enters only the threshold
        column below, as a factor; so sigma_beta and sigma_amplitude_rel
        do not depend on beta at all, and sigma_chi_c_rel is
        proportional to 1/beta.
    sigma_log : per-point scatter of log R (sigma_R / R for small noise).
    fit_threshold : True (default) for the three-parameter fit that
        `fit_branch` does; False for a threshold known in advance.

    The fitted model is log R = log A + beta log(chi/chi_c - 1).  Its
    derivatives at each point, with respect to (log chi_c, beta,
    log A), are (-beta (1+eps)/eps, log eps, 1); stacked into a matrix
    J, the asymptotic covariance is sigma_log^2 (J^T J)^-1.  This is
    the same covariance `fit_branch` reports when it is given
    sigma_r = sigma_log * r and the model holds, and it is exact to
    first order in the noise.  With fit_threshold=False the first
    column is dropped and sigma_beta equals `beta_relative_sigma`.  The
    error bars do not depend on chi_c or A.

    Fitting the threshold as well makes sigma_beta larger than the
    known-threshold value, by a factor that depends on the span of eps
    and not on beta: for 12 points about 2.0 over two decades,
    1.6 over three and 1.3 over five (for example 0.0021 instead of
    0.0013 over eps = 1e-5 .. 1e-2 with sigma_log = 0.01).

    The tests hold it against the closed form for a known threshold,
    against the error bars `fit_branch` reports on noiseless data, and
    against the scatter of 300 seeded simulated fits.  Refuses fewer
    than 6 points for a fitted threshold (fit_branch needs 6) or 3 for
    a known one, a non-positive or reversed eps range, and a
    non-positive beta or sigma_log.
    """
    n = _count(n_points, "n_points")
    need = 6 if fit_threshold else 3
    if n < need:
        raise ValueError(f"need >= {need} points "
                         f"({'fitted' if fit_threshold else 'known'} "
                         "threshold)")
    lo, hi = float(eps_min), float(eps_max)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("need 0 < eps_min < eps_max, both finite")
    b, s = float(beta), float(sigma_log)
    if not (np.isfinite(b) and b > 0.0 and np.isfinite(s) and s > 0.0):
        raise ValueError("beta and sigma_log must be finite and positive")
    eps = np.geomspace(lo, hi, n)
    cols = [np.log(eps), np.ones(n)]
    if fit_threshold:
        cols = [-b * (1.0 + eps) / eps] + cols
    jac = np.stack(cols, axis=1)
    # column scaling and QR keep (J^T J)^-1 accurate when the threshold
    # column is many orders of magnitude larger than the others
    norms = np.linalg.norm(jac, axis=0)
    rmat = np.linalg.qr(jac / norms, mode="r")
    if np.linalg.cond(rmat) > 1e12:
        raise ValueError("the planned points do not constrain all "
                         "parameters independently")
    rinv = np.linalg.inv(rmat)
    cov = s ** 2 * (rinv @ rinv.T) / np.outer(norms, norms)
    sig = np.sqrt(np.diag(cov))
    if fit_threshold:
        s_chi, s_beta, s_amp = sig
    else:
        s_chi, (s_beta, s_amp) = 0.0, sig
    return FitPlan(n_points=n, eps_range=(lo, hi), beta=b, sigma_log=s,
                   fit_threshold=bool(fit_threshold),
                   sigma_beta=float(s_beta), sigma_chi_c_rel=float(s_chi),
                   sigma_amplitude_rel=float(s_amp))


def points_for_fit(target_sigma_beta: float, eps_min: float, eps_max: float,
                   beta: float, sigma_log: float,
                   fit_threshold: bool = True) -> Tuple[int, FitPlan]:
    """Points needed for a target error bar on beta, threshold fitted.

    Like `points_for_beta`, but for the fit `fit_branch` actually does
    (see `plan_fit`).  Returns (n, FitPlan) with plan.sigma_beta <=
    target and n-1 points failing it.  The count is searched by doubling
    and bisection, which assumes the error bar falls as points are added
    over a fixed range (checked in the tests for n = 6 .. 400); the
    result is then confirmed against n-1.  Refuses a target that would
    need more than 10^7 points.
    """
    tgt = float(target_sigma_beta)
    if not (np.isfinite(tgt) and tgt > 0.0):
        raise ValueError("target_sigma_beta must be positive")

    def sb(n):
        return plan_fit(n, eps_min, eps_max, beta, sigma_log,
                        fit_threshold).sigma_beta

    lo = 6 if fit_threshold else 3
    if sb(lo) <= tgt:
        return lo, plan_fit(lo, eps_min, eps_max, beta, sigma_log,
                            fit_threshold)
    hi = lo
    while sb(hi) > tgt:
        lo, hi = hi, 2 * hi
        if hi > 10 ** 7:
            raise ValueError(
                "over 10^7 points would be needed: the target is out "
                "of reach at this noise and range; widen the range "
                "or reduce the noise")
    while hi - lo > 1:      # sb(lo) > tgt >= sb(hi)
        mid = (lo + hi) // 2
        if sb(mid) > tgt:
            lo = mid
        else:
            hi = mid
    return hi, plan_fit(hi, eps_min, eps_max, beta, sigma_log,
                        fit_threshold)
