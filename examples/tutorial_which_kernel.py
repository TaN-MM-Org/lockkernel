"""Tutorial: is the onset of my ensemble conservative or overdamped?

A lab question, end to end.  An ensemble has a measured inhomogeneous line
(a Voigt line: Gaussian FWHM 1.00 +- 0.02, Lorentzian FWHM 0.30 +- 0.02,
in one frequency unit), and its synchronization branch R(chi) has been
measured just above the onset.  Two models are on the table:

  * conservative spins, kernel W = 1/(1+u^2): beta = 1, kernel mass pi;
  * overdamped phase oscillators (Kuramoto), W = sqrt(1-u^2): beta = 1/2,
    kernel mass pi/2.

The data give two independent answers: the exponent, and the threshold,
which with the measured line gives the kernel mass m = 1/(p(0) chi_c).
A third check is the amplitude A that the Kuramoto model predicts.

The "measurement" here is synthetic: 12 couplings from 1e-4 to 1e-2 above
the threshold of the Kuramoto model on that line, the order parameter at
each from `branch_at_coupling` (the exact branch), with seeded noise: 1 %
on R, a coupling error of 5e-6 on each recorded coupling, and a 1 %
uncertainty of the overall coupling scale.  All numbers are illustrative.

Runs in about 5 to 10 s.  Output (lockkernel 1.4.0):

    fitted threshold chi_c = 0.879490 +- 0.0088 (with 1 % scale calibration)
    fitted exponent  beta  = 0.4957 +- 0.0048, chi2 = 4.3 for 9 dof, p = 0.89
    exponent: Kuramoto (1/2) is 0.9 sigma off, conservative (1) is 104 sigma off
    kernel tail refused: beta = 0.4957 +- 0.004827 is consistent with 1/2, which identifies only the CLASS s >= 3
    line center p(0) = 0.72386 +- 0.016
    kernel mass m = 1/(p(0) chi_c) = 1.571 +- 0.038
    mass: Kuramoto (pi/2 = 1.571) is 0.00 sigma off, conservative (pi = 3.142) is 41 sigma off
    amplitude: fitted A = 1.5224 +- 0.0421, Kuramoto prediction 1.5676 (1.1 sigma)
"""
import mpmath as mp
import numpy as np

from lockkernel import fit_branch, kernel_tail_from_beta, with_calibration
from lockkernel.kernels import kuramoto
from lockkernel.lineshapes import voigt
from lockkernel.parametric import (amplitude_curvature, branch_at_coupling,
                                   threshold)

mp.mp.dps = 15
FG, FL, SIG_FG, SIG_FL = 1.00, 0.30, 0.02, 0.02
line = voigt(FG, FL)

# ---- the synthetic measurement (the truth: Kuramoto kernel) --------------
rng = np.random.default_rng(2026)
chi_c_true = float(threshold(line, kuramoto()))
chi_set = chi_c_true * (1 + np.geomspace(1e-4, 1e-2, 12))
sigma_chi = np.full(chi_set.size, 5e-6)
chi_true = chi_set + rng.normal(0.0, 1.0, chi_set.size) * sigma_chi
r = np.array([float(branch_at_coupling(line, kuramoto(), c)[1]) for c in chi_true])
r = r * np.exp(rng.normal(0.0, 0.01, r.size))

# ---- fit the onset --------------------------------------------------------
fit = fit_branch(chi_set, r, "synthetic: exact Kuramoto branch, seeded noise",
                 sigma_r=0.01 * r, sigma_chi=sigma_chi)
fit = with_calibration(fit, chi_scale_rel=0.01)
print(f"fitted threshold chi_c = {fit.chi_c:.6f} +- {fit.sigma_chi_c:.2g} "
      "(with 1 % scale calibration)")
print(f"fitted exponent  beta  = {fit.beta:.4f} +- {fit.sigma_beta:.4f}, "
      f"chi2 = {fit.chi2:.1f} for {fit.dof} dof, p = {fit.p_value:.2g}")

# ---- answer 1: the exponent ------------------------------------------------
pull_k = abs(fit.beta - 0.5) / fit.sigma_beta
pull_c = abs(fit.beta - 1.0) / fit.sigma_beta
print(f"exponent: Kuramoto (1/2) is {pull_k:.1f} sigma off, "
      f"conservative (1) is {pull_c:.0f} sigma off")
try:
    kernel_tail_from_beta(fit.beta, fit.sigma_beta)
except ValueError as err:
    print("kernel tail refused:", str(err).split(" (")[0])

# ---- answer 2: the threshold and the measured line ------------------------
# p(0) of the Voigt line and its error from the two width errors (central
# differences, added in quadrature)
p0 = float(line.p0())
dp_dfg = float((voigt(FG + 1e-4, FL).p0() - voigt(FG - 1e-4, FL).p0()) / 2e-4)
dp_dfl = float((voigt(FG, FL + 1e-4).p0() - voigt(FG, FL - 1e-4).p0()) / 2e-4)
sig_p0 = float(np.hypot(dp_dfg * SIG_FG, dp_dfl * SIG_FL))
m = 1.0 / (p0 * fit.chi_c)
sig_m = m * float(np.hypot(sig_p0 / p0, fit.sigma_chi_c / fit.chi_c))
print(f"line center p(0) = {p0:.5f} +- {sig_p0:.2g}")
print(f"kernel mass m = 1/(p(0) chi_c) = {m:.3f} +- {sig_m:.3f}")
print(f"mass: Kuramoto (pi/2 = {np.pi / 2:.3f}) is "
      f"{abs(m - np.pi / 2) / sig_m:.2f} sigma off, conservative "
      f"(pi = {np.pi:.3f}) is {abs(m - np.pi) / sig_m:.0f} sigma off")

# ---- answer 3: the amplitude the Kuramoto model predicts -------------------
A_pred = float(amplitude_curvature(line, kuramoto()))
print(f"amplitude: fitted A = {fit.amplitude:.4f} +- {fit.sigma_amplitude:.4f}, "
      f"Kuramoto prediction {A_pred:.4f} "
      f"({abs(fit.amplitude - A_pred) / fit.sigma_amplitude:.1f} sigma)")
