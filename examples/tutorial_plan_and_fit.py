"""Tutorial: plan an onset measurement, take it, and read off the kernel tail.

Conservative spins (kernel W = 1/(1+u^2), tail s = 2, beta = 1) on a
Lorentzian line of FWHM 1, in one frequency unit.  Three steps:

  1. Plan.  `eps_window` says how far above the threshold the branch is
     still a power law (local exponent within 1 % of beta), and
     `points_for_fit` how many couplings, spread evenly in log(eps) from
     1e-4 up to that limit, give an error bar of 0.007 on beta with 2 %
     noise on R.
  2. Measure.  Synthetic data: the exact branch at those couplings
     (`branch_at_coupling`) with seeded 2 % noise on R.
  3. Fit and interpret: beta with its error bar and the chi-square check,
     then the kernel tail s = 1 + 1/beta.  For contrast, the same number
     of points spread up to eps = 0.5, beyond the window: the fit is
     biased and the chi-square test flags it.

All numbers are illustrative.  Runs in about 10 to 20 s.  Output
(lockkernel 1.4.0):

    plan: power law within 1 % up to eps = 0.0101 (closed form 0.0101)
    plan: 16 points for beta +- 0.007 (planned error bar 0.00700)
    fit inside the window:  beta = 1.0012 +- 0.0070, chi2 = 5.9 for 13 dof, p = 0.95
    kernel tail s = 2.00 +- 0.01 (conservative spins: s = 2)
    fit up to eps = 0.5:    beta = 0.9554 +- 0.0027, chi2 = 180.7 for 13 dof, p = 1.2e-31
"""
import mpmath as mp
import numpy as np

from lockkernel import fit_branch, kernel_tail_from_beta, points_for_fit
from lockkernel.kernels import conservative
from lockkernel.lineshapes import lorentzian
from lockkernel.parametric import branch_at_coupling, eps_window, threshold

mp.mp.dps = 15
line, kern = lorentzian(1.0), conservative()
NOISE = 0.02

# ---- 1. plan --------------------------------------------------------------
eps_max, _ = eps_window(line, kern, rel_tol=0.01, lo=-4)
print(f"plan: power law within 1 % up to eps = {float(eps_max):.4f} "
      f"(closed form {0.01 / 0.99:.4f})")
n, plan = points_for_fit(0.007, 1e-4, float(eps_max), beta=1.0, sigma_log=NOISE)
print(f"plan: {n} points for beta +- 0.007 (planned error bar {plan.sigma_beta:.5f})")

# ---- 2. measure (synthetic) -----------------------------------------------
chi_c = float(threshold(line, kern))
rng = np.random.default_rng(7)


def measure(eps_top):
    chi = chi_c * (1 + np.geomspace(1e-4, eps_top, n))
    r = np.array([float(branch_at_coupling(line, kern, c)[1]) for c in chi])
    return chi, r * np.exp(rng.normal(0.0, NOISE, n))


# ---- 3. fit and interpret ---------------------------------------------------
chi, r = measure(float(eps_max))
fit = fit_branch(chi, r, "synthetic: exact conservative branch, 2 % noise",
                 sigma_r=NOISE * r)
print(f"fit inside the window:  beta = {fit.beta:.4f} +- {fit.sigma_beta:.4f}, "
      f"chi2 = {fit.chi2:.1f} for {fit.dof} dof, p = {fit.p_value:.2g}")
s, s_err = kernel_tail_from_beta(fit.beta, fit.sigma_beta)
print(f"kernel tail s = {s:.2f} +- {s_err:.2f} (conservative spins: s = 2)")

chi, r = measure(0.5)
wide = fit_branch(chi, r, "synthetic: exact conservative branch, 2 % noise",
                  sigma_r=NOISE * r)
print(f"fit up to eps = 0.5:    beta = {wide.beta:.4f} +- {wide.sigma_beta:.4f}, "
      f"chi2 = {wide.chi2:.1f} for {wide.dof} dof, p = {wide.p_value:.2g}")
