# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, high-precision quadrature against
independent closed forms, or seeded simulation; the release notes on
GitHub carry the full anchor lists.

## v1.2.0 - 2026-09-30

### Added

- `parametric.amplitude_curvature(line, kernel)`: the amplitude `A` in
  `R = A eps^(1/2)` for the class `beta = 1/2` (compact support such as
  Kuramoto, faster than algebraic decay, or a tail s > 3, on a line
  with p''(0) < 0): `A = G0^(3/2) sqrt(2 / (-p''(0) M_2))`. Until now
  the package gave the amplitude only for 1 < s < 3. Tests: 1 for the
  Kuramoto kernel on a Lorentzian line (the closed form
  `R = sqrt(1 - chiN_c/chiN)`), `sqrt(pi)` and `sqrt(2)` on Gaussian
  lines of two widths, to 1e-12; `R/sqrt(eps)` on the exact branch at
  Omega = 1e-6 to a relative 1e-10 (Kuramoto/Gaussian line, Gaussian
  kernel/Student-t line, s = 6/Gaussian line) and 2e-6 (s = 4, where
  the correction is of order Omega^(s-3)). Refuses p''(0) >= 0 and
  tails s <= 3.
- `kernels.Kernel.second_moment()`: `M_2 = int u^2 W du`. Tests: pi/8
  (Kuramoto), sqrt(pi)/2 (Gaussian kernel), 2 (pi/s)/sin(3 pi/s) for
  s = 4 and 6, to 1e-12; refuses s <= 3.
- `measured.plan_fit` and `measured.points_for_fit` (also at the top
  level, with the result type `FitPlan`): the error bars `fit_branch`
  will report for a planned measurement, with the threshold fitted as
  `fit_branch` does it, and the number of points a target error bar
  on beta needs. `beta_relative_sigma` / `points_for_beta` assume a
  known threshold; fitting it too makes the error bar about 2.0 times
  larger over two decades of eps, 1.6 over three and 1.3 over five (12
  points), independently of beta (beta only scales the threshold
  column of the Jacobian, so sigma_beta and sigma_A do not depend on it
  and sigma_chi_c scales as 1/beta; tested). Tests: with the threshold known it equals
  `beta_relative_sigma` to 1e-12; it equals the covariance built from
  mpmath numerical derivatives at 30 digits to 1e-6; over 300 seeded
  noisy fits the scatter of chi_c, beta and A matches it within 12 %;
  `points_for_fit` is checked two-sided, and the error bar is checked
  to fall with each added point from 6 to 400.
- Tests for `fold_interval` and `bimodal_gaussian`, which had none:
  every number `fold_interval` returns is held to the Voigt-profile
  closed form of the branch (Faddeeva function), turning points
  located by root finding on its derivative, to a relative 1e-12, for a
  branch leaving the threshold backwards (two peaks 4 widths apart) and
  for an S-shaped one (a three-peak line); `c_coefficient` of the
  two-peaked line is held to `(1 - 2x D(x))/pi` (Dawson's function) to
  1e-13, and its sign to flip at separation 2.61386 widths, the maximum
  of Dawson's function.
- Tests that the box (flat-topped) line gives `beta = 1/(s-1)` also for
  s = 4 and 6 (1/3 and 1/5, within 1e-6), and that its branch matches
  the closed form `G = X 2F1(1, 1/s; 1+1/s; -X^s)` to 1e-13.

### Fixed

- `branch_point` (and so `sweep`, `extract_beta`) computed
  `eps = chiN/chiN_c - 1` and lost about log10(1/eps) digits. It now
  integrates `G(0) - G(Omega)` directly, with guard digits for the
  difference `p(0) - p(Omega u)` and a second pass scaled to the size
  of the result, and keeps the working precision. At 15 digits on the
  Lorentzian line with the conservative kernel (where eps = 2 Omega
  exactly) the relative error of eps was 2.5e-10, 2.2e-4 and 220 at
  Omega = 1e-6, 1e-9 and 1e-12; it is now below 1e-13 down to 1e-14
  (tested), and below 1e-28 at 30 digits down to 1e-20 (tested).
- `G_of_Omega` integrated the segment from u = 10 to u = scale/Omega
  on a linear scale; over many decades the quadrature could not resolve
  its start. At 15 digits on the Lorentzian line G was 4.4e-13
  (relative) off at Omega = 1e-9 and 4.4e-10 at 1e-12. The part u >= 1
  is now integrated in log(u), and the tail beyond 10 scale/Omega is
  mapped onto (0, 1] (mpmath's own treatment of [X, inf) with X large
  is good only to about 1e-9 relative at 15 digits). The integrand is
  now integrated over u >= 0 and doubled, using the symmetry of line
  and kernel that the package already assumes.
- `extract_beta` returned a plausible slope for a branch that bends
  back (for `bimodal_gaussian(4.0)` with the conservative kernel, where
  the onset is first order and eps < 0, exponents [-3, -4, -5, -6] at
  20 digits gave 1.0016447, 1.0001645, 1.0000165) and raised
  ZeroDivisionError on a flat branch (box line with the Kuramoto kernel,
  where eps = 0 exactly). It now raises ValueError saying which.
- `fit_branch` used the finite-difference Jacobian of
  `scipy.optimize.least_squares`, whose step in chi_c (about 1e-8
  relative) is not small next to eps near the threshold. With points
  down to eps = 1e-7 it reported sigma_chi_c about 17 % and sigma_beta
  about 2 % too small (noiseless test data; in 300 seeded noisy fits the
  scatter of chi_c was 1.28 times the median reported error bar, now
  1.06). It now passes the analytic Jacobian.
- `fold_interval`, when the branch leaves the threshold backwards,
  returned the first grid point instead of the threshold as the end of
  the low branch, and a turning point it failed to refine was silently
  replaced by a grid point (now it raises RuntimeError). Turning points
  are now found from an integral for dG/dOmega instead of numerically
  differentiating a quadrature, which makes it about 20 times faster
  (79 s -> 3.5 s for `bimodal_gaussian(4.0)` with n = 60 at 15 digits).
  The result dict gains `Omega_jump`.

### Behaviour changes

- README example 4 (`fit_branch` on 14 noisy points reaching eps =
  1.6e-7): sigma_chi_c 3.504e-9 -> 3.848e-9; beta 0.6732185 ->
  0.6732500; sigma_beta 0.0027493 -> 0.0027902; A 0.65222 -> 0.65241;
  sigma_A 0.018766 -> 0.018977. The new point has the lower
  least-squares cost (2.259876 against 2.259998).
- `fold_interval(bimodal_gaussian(4.0), n=60)` at 15 digits: chiN_hi
  5.89464088 -> 5.89561378 (the threshold), Omega_hi 1e-4 -> 0, R_low
  1.696e-5 -> 0, R_jump 0.84799683 -> 0.84805059.
- `extract_beta` raises ValueError where it used to return a number
  or raise ZeroDivisionError (see Fixed); `fold_interval` raises
  ValueError for a fold on a compact-support line and RuntimeError for
  an unbracketed turning point.
- `branch_point`/`G_of_Omega` values change in the digits that were
  wrong before. On a box line with s = 4 at 15 digits, the last
  `extract_beta` entry for Omega = 1e-2 .. 1e-5 was 0.3194, and is now
  0.33333333 (1/3); with s = 6 it raised ZeroDivisionError.
- All 65 earlier tests pass unchanged, and README examples 1, 2, 3, 6
  and 9 (the old 7) print the same as in 1.1.1.

### Changed

- Documentation now states that `beta = 1/2` for s >= 3 assumes a line
  with a rounded top (p''(0) < 0): on a flat-topped line, such as
  `box`, a tail gives beta = 1/(s-1) for every s > 1 and a compact
  kernel gives a jump. `predicted_beta` and `kernel_tail_from_beta`
  keep their behaviour; their docstrings, and the refusal message for a
  beta below 1/2, now say so.
- README: new examples 7 (amplitude when beta = 1/2), 8 (hysteresis
  loop of a first-order onset) and an extended example 5 (planning with
  the threshold fitted); the old example 7 is now 9. Limits updated.
  97 tests.
- `parametric.__all__` now also lists `tail_integral`,
  `amplitude_general` and `amplitude_curvature`.

## v1.1.1 - 2026-09-22

### Fixed

- `parametric.c_coefficient`, and therefore `parametric.amplitude`,
  lost about 40 digits to the cancellation p(delta) - p(0) near the
  lower end of its integral. At mpmath's default precision (15 digits)
  `amplitude(lorentzian())` returned about 5e-4 instead of 1, and
  `amplitude(gaussian())` about 3e-4 instead of pi/2; at 20 digits the
  Lorentzian value was off by about 0.16 %, and a Lorentzian of width
  1000 was off by about 1e-9 even at 30 digits. The tests ran at 30
  digits on unit-width lines, where the error is about 1e-12, so they
  did not see it. The integral now runs with 25 extra digits and its
  lower end is placed at 1e-20 line widths instead of 1e-20 absolute.
  Anyone who called `c_coefficient` or `amplitude` below about 30
  digits should re-run those numbers. Skipping the first 1e-20 line
  widths still limits both to about 20 significant digits, however
  high the precision is set (as in 1.1.0 for unit-width lines); this
  is now stated in the docstring and the README.

### Tests

- New `test_amplitude_at_default_precision`: at 15 digits, the
  amplitude of the Lorentzian (widths 1 and 1000) and Gaussian lines
  matches 1 and pi/2 to a relative 1e-12. It fails on 1.1.0.
- CI now runs Python 3.10 too (the classifiers claimed 3.9-3.14, but
  3.10 was missing from the matrix), and a new `oldest-dependencies`
  job runs the suite on Python 3.10 with NumPy 1.22.0, SciPy 1.8.0 and
  mpmath 1.2.1 (PyPI has no mpmath 1.2.0), the lowest versions
  pyproject.toml allows. 65 tests.
- The source distribution now ships `tests/conftest.py` (new
  `MANIFEST.in`). The 1.1.0 sdist left it out, and without it the
  test files overwrite each other's mpmath precision: run in one go
  from the 1.1.0 sdist (leaving out the six slowest tests), 11 tests
  failed. The repository at v1.1.0, which CI runs, had the file.

### Changed

- README rewritten in plainer language, with worked examples whose
  printed output is checked, a list of every refusal, and the test
  tolerances as the tests actually assert them. Corrections to the
  1.1.0 README: the coefficient c is checked to a relative 1e-9, not
  1e-11; only the Lorentzian closed form is checked to 1e-28 over six
  decades (the Kuramoto one to 1e-26 over four); `class_exact` is not
  covered by any test; the claim that exponents come out "at tens of
  digits" is not supported by any test (they are checked to a relative
  1e-6 to 5e-3, depending on s); the general amplitude is checked at
  s = 1.8, 2.0 and 2.2, not along the whole line; `points_for_beta(0.01)`
  targets an absolute error of 0.01 on beta, not 1 %; and the example
  passed `extract_beta` exponents in the order that makes its last
  entry the one farthest from the onset.

### Corrections to earlier entries

- v1.1.0 says the noisy fit lands "within its own reported error
  bars"; the test allows four error bars. It also says "Python
  3.9-3.14 CI", but 3.10 was not in the CI matrix.
- The note at the top ("Every claim added in any release is pinned by
  a test") does not hold for `fold_interval`, `bimodal_gaussian`,
  `class_exact`, `physicality` and `valid_window`, which have no test.

## v1.1.0 - 2026-09-18

First release under this organization: the maintained distribution
of the locking-kernel-universality reference implementation
(Tanvir-Mahmud-Mahim/locking-kernel-universality v1.0.7, concept DOI
10.5281/zenodo.22696369), whose scripts, archived run records and
figures remain with the study.

- Core modules (`lineshapes`, `kernels`, `parametric`, `cumulant`,
  `ensemble`, `exact`) carried over unchanged, with their 56 tests.
- NEW `measured`: the lab half. `fit_branch` fits a measured
  synchronization branch for threshold, exponent and amplitude
  together, with asymptotic error bars, residual diagnostics,
  mandatory data provenance and refusals (too few points, couplings
  out of order, under a decade of reduced coupling, an unresolved
  threshold, non-convergence); `kernel_tail_from_beta` inverts the
  exact dictionary beta = 1/(s-1) (1 < s < 3), 1/2 (s >= 3) with
  error propagation, refusing to name a single tail where only the
  s >= 3 class is identified; `beta_relative_sigma` /
  `points_for_beta` give the closed-form error bar of a log-log
  slope and its exact inversion for measurement planning.
- Anchors for the new module: the lab half is checked BY the theory
  half -- `fit_branch` recovers the threshold and exponent of
  branches generated by the exact parametric solution (conservative
  beta = 1; tail s = 2.5, beta = 2/3), noiselessly and under seeded
  noise within its own reported error bars; the planning closed form
  matches seeded simulation; the inversion is two-sided.
- Packaging: matplotlib removed from the dependencies (the core
  never imports it -- asserted by a test); Python 3.9-3.14 CI;
  Trusted-Publishing release workflow.

## v1.0.7 and earlier

See the research repository:
https://github.com/Tanvir-Mahmud-Mahim/locking-kernel-universality
