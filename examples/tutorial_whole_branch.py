"""Tutorial: fit the whole measured branch, test the model, calibrate the coupling.

Overdamped phase oscillators (Kuramoto kernel) whose natural frequencies
have a Gaussian spread of FWHM 1 (one frequency unit, measured separately).
The coupling is set by a drive voltage, and the factor k that turns volts
into chiN is not known.  Synthetic measurement: R at 12 voltages, from
just above the onset (eps = 0.002) to far above it (eps = 0.5), with 1 %
seeded noise on R.

  1. The power law R = A eps^beta (`fit_branch`) holds only near the
     threshold.  Fitted to all 12 points it returns a biased exponent, and
     its chi-square test rejects it.
  2. `branch_table` tabulates the exact branch of a model (a line and a
     kernel), and `fit_exact_branch` fits it to all the points with k
     free.  Two candidate lines, Gaussian and Lorentzian, with the
     Kuramoto kernel: the chi-square test keeps one and rejects the other.
     R(eps) depends only on the shape of the line, not on its width, so
     the test does not need the width.
  3. The accepted model gives k, the threshold in volts, and, with the R
     scale free as well, a check of the R calibration.  Only k and the
     threshold's conversion to chiN use the measured width.

All numbers are illustrative.  Runs in about 5 s (CPU time).  Output
(lockkernel 1.5.0):

    power law, all 12 points: beta = 0.4115 +- 0.0024, chi2 = 276.4 for 9 dof, p = 2.6e-54
    exact branch, Kuramoto kernel, Gaussian line:   chi2 = 9.9 for 11 dof, p = 0.54
    exact branch, Kuramoto kernel, Lorentzian line: chi2 = 16065.0 for 11 dof, p = 0
    coupling scale k = 0.037101 +- 0.000001 per volt (true 0.0371)
    threshold chi_c = 18.2653 +- 0.0005 V (true 18.2658 V)
    with the R scale free: q = 1.0005 +- 0.0035, threshold 18.2654 +- 0.0006 V
"""
import mpmath as mp
import numpy as np

from lockkernel import fit_branch, fit_exact_branch
from lockkernel.kernels import kuramoto
from lockkernel.lineshapes import gaussian, lorentzian
from lockkernel.parametric import branch_point, branch_table, threshold

mp.mp.dps = 15
K_TRUE = 0.0371            # chiN per volt, unknown to the analysis
NOISE = 0.01

# ---- synthetic measurement: the exact Kuramoto branch on a Gaussian line --
line = gaussian(1.0)
pts = [branch_point(line, kuramoto(), mp.mpf(o)) for o in np.geomspace(0.05, 0.9, 12)]
volts = np.array([float(p[0]) for p in pts]) / K_TRUE
rng = np.random.default_rng(11)
r = np.array([float(p[1]) for p in pts]) * np.exp(rng.normal(0.0, NOISE, 12))
ref = "synthetic: Kuramoto branch on a Gaussian line, 1 % noise"

# ---- 1. the power law over the whole range --------------------------------
pl = fit_branch(volts, r, ref, sigma_r=NOISE * r)
print(f"power law, all 12 points: beta = {pl.beta:.4f} +- {pl.sigma_beta:.4f}, "
      f"chi2 = {pl.chi2:.1f} for {pl.dof} dof, p = {pl.p_value:.2g}")

# ---- 2. the exact branch of two models, coupling scale free ----------------
tables, fits = {}, {}
for name, cand in (("Gaussian", gaussian(1.0)), ("Lorentzian", lorentzian(1.0))):
    tables[name] = branch_table(cand, kuramoto(), 1e-4, 1.0)
    fits[name] = f = fit_exact_branch(volts, r, ref, tables[name], sigma_r=NOISE * r)
    print(f"exact branch, Kuramoto kernel, {name + ' line:':16s} chi2 = {f.chi2:.1f} "
          f"for {f.dof} dof, p = {f.p_value:.2g}")

# ---- 3. the accepted model: calibration and threshold ----------------------
f = fits["Gaussian"]
chi_c_true = float(threshold(line, kuramoto())) / K_TRUE
print(f"coupling scale k = {f.chi_scale:.6f} +- {f.sigma_chi_scale:.6f} per volt "
      f"(true {K_TRUE})")
print(f"threshold chi_c = {f.chi_c:.4f} +- {f.sigma_chi_c:.4f} V (true {chi_c_true:.4f} V)")
g = fit_exact_branch(volts, r, ref, tables["Gaussian"], sigma_r=NOISE * r,
                     fit_r_scale=True)
print(f"with the R scale free: q = {g.r_scale:.4f} +- {g.sigma_r_scale:.4f}, "
      f"threshold {g.chi_c:.4f} +- {g.sigma_chi_c:.4f} V")
