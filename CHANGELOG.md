# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, high-precision quadrature against
independent closed forms, or seeded simulation; the release notes on
GitHub carry the full anchor lists.

## 1.5.0 (2026-10-18)

### Added

- `parametric.branch_table(line, kernel, eps_min, eps_max, rel_tol=1e-5,
  step=0.5)` and its result `BranchTable`: the exact branch `R(eps)` of
  one model as a checked table, evaluated in floating point (`R(eps)`,
  `log_R_and_slope(eps)`). Between nodes it is the cubic Hermite
  interpolant of `log R` in `log eps` through the exact values and the
  exact local exponents; each interval is checked against an exact
  branch point at its middle (in `log Omega`) and halved until they
  differ by at most `rel_tol`, and `max_error` reports the largest
  difference found. Tests: Lorentzian line, Kuramoto kernel, against
  `R = sqrt(eps/(1+eps))` at 3001 points from `eps = 1e-4` to 1 (within
  2 `rel_tol`, `rel_tol = 1e-7`; observed 0.96 `rel_tol`), node slopes
  against `(1/2)/(1+eps)` (1e-12); Gaussian line, Kuramoto kernel,
  against `G = p(0) (pi/2) e^-z (I0(z) + I1(z))`, `z = Omega^2/(4
  sigma^2)`, derived in the test from the integral representation of
  the modified Bessel functions and checked there against 40-digit
  quadrature (2e-7 at 25 points). Refuses lines with
  compact support, folded or flat branches, bad ranges and tolerances,
  and evaluation outside its range.
- `measured.fit_exact_branch(chi, r, reference, table, sigma_r=None,
  fit_r_scale=False)` (also at the top level) and its result
  `ExactBranchFit`: fits the exact branch of a model to measured points
  over their whole range, not only where the power law holds. The free
  parameter is the coupling scale `k` (`chiN = k chi`, for couplings
  recorded as a power, a field or a voltage), optionally also an `R`
  scale `q`; it returns them with error bars, the threshold in the
  recorded units, and `chi2`, `dof` and `p_value`, which test the line
  and kernel over all the points. Tests: an independent SciPy fit of
  the same data with the closed-form branch (parameters and chi2 within
  bounds derived from the table error, error bars to 1e-4); 400 seeded
  fits (scatter of `log k` and `log q` against the median error bar:
  observed ratios 1.03 and 0.99, required within 12 %; mean chi2 10.03
  for 10 dof); Kuramoto data over `eps = 1e-3 .. 0.5` reject the
  conservative kernel's branch (`p < 1e-10`) and accept their own,
  while `fit_branch`'s power law on the same points is biased (beta
  0.465 +- 0.002 against 1/2) and rejected (`p = 4e-20`). Refuses fewer
  than 5 points, couplings spanning more than the table holds, a best
  fit at the table's edge, and a table whose error exceeds 1 % of the
  relative errors of `R`.
- Tests for `tail_integral` on lines with compact support (box and a
  semicircle line, closed forms derived in the test), for `full_exact`
  against the 1.4.0 code, and for the new functions: 194 tests (from
  172).

### Fixed

- `tail_integral`, and so `amplitude_general`, on a line with compact
  support: the integration by parts of `-p'` missed the jump of the
  `box` line at its edge. Before -> after, box of half width 1, 15
  digits: `I_s` 0 -> 2 at `s = 1.5` (closed form `1/(s-1)`; 0 at every
  `s` before), and `amplitude_general(box(1.0), power_tail(2.5))`
  `ZeroDivisionError` -> 2.08481122566036 (the closed form, to 1e-13).
  Such a line is now integrated in the direct form, inside the support
  with guard digits and outside in closed form. Every earlier release
  had this error. Tests: `test_tail_integral_on_compact_lines`,
  `test_general_amplitude_on_the_box_line`.
- `points_for_fit` refused every target that needs between 6291457 and
  10^7 points ("over 10^7 points would be needed"): its doubling went
  from 6291456 straight past 10^7. It now stops at 10^7, as
  `points_for_beta` does. Before -> after:
  `points_for_fit(1.3297e-5, 1e-4, 1e-2, beta=1, sigma_log=0.05,
  fit_threshold=False)` refused -> 8000001 points (closed form 8000000;
  `plan_fit` agrees with it to about 1e-7 at that size). Test:
  `test_points_for_fit_reaches_the_cap` (with the error bar in closed
  form, so that millions of points cost nothing).

### Changed

- `full_exact` refuses detunings that are not a non-empty
  one-dimensional list of finite numbers, and a non-finite `chiN` or
  time.
- All 172 earlier tests pass (the version check now expects 1.5.0), and
  README examples 1 to 12 print the same as with 1.4.0.

### Performance

- `full_exact` builds `J+` as a sparse matrix from the bits of the basis
  index, the exchange term as `chi1 J+ J-` (one product instead of `N^2`
  dense products) and the covariance from the vectors `J_a psi`:
  8 emitters 1.60 s -> 0.104 s (15x) and 10 emitters 133 s ->
  10.3 s (13x), medians of 5 and 3 runs on the same (busy) machine.
  The test suite's slowest test (`full_exact` at 10 emitters against the
  cumulant solver) went from about 97 s to about 6 s. Equality is
  checked against the 1.4.0 code, kept in the test, with tolerances
  derived from rounding: `R` to 1e-11 and `xi^2` to a relative
  `1e-11/R^2` (`test_full_exact_same_as_1_4_0`; observed differences
  below 1e-14 in `R`).

### Tutorials

- `examples/tutorial_whole_branch.py`: synthetic data over `eps = 0.002
  .. 0.5`, where the power law fails; `branch_table` and
  `fit_exact_branch` keep the right line (Gaussian, `p = 0.54`) and
  reject the wrong one (Lorentzian), and calibrate the coupling scale,
  the threshold in volts and the `R` scale. It prints the output quoted
  in the README, which `tests/test_tutorials.py` checks along with the
  truth of its synthetic data.

## 1.4.0 (2026-10-17)

### Added

- `lineshapes.voigt(fwhm_gauss, fwhm_lorentz)`: the Voigt line, a
  Gaussian convolved with a Lorentzian, `p = Re w(z)/(sigma sqrt(2 pi))`
  with the Faddeeva function `w(z) = exp(-z^2) erfc(-iz)` (DLMF 7.2.3,
  7.19.3), evaluated with `2 log10|z|` guard digits (without them the
  density was 44 % off at 1e8 widths). Tests: a 120-digit evaluation
  out to 1e30 widths (1e-14), SciPy's `voigt_profile` (1e-12), the
  convolution integral (1e-13), normalization and the Lorentzian tail;
  with the conservative kernel the branch, threshold, `c` and amplitude
  against the closed form `G = pi erfcx((gamma + Omega)/(sigma sqrt 2))
  / (sigma sqrt(2 pi))` (1e-12); with the Kuramoto kernel
  `amplitude_curvature` against its closed form (1e-12).
- `parametric.branch_at_coupling(line, kernel, chiN)` and
  `branch_at_eps(line, kernel, eps)`: the branch point
  `(chiN, R, eps, Omega)` at a coupling the lab sets, by a safeguarded
  secant iteration on log eps against log Omega. Tests: closed forms on
  the Lorentzian line (conservative and Kuramoto kernels, 1e-13), the
  Voigt closed form inverted by mpmath's root finder at 40 digits
  (1e-12), round trips through `branch_point` (1e-12). Refuses couplings
  at or below the threshold and branches that bend back or are flat.
- `parametric.local_exponent(line, kernel, Omega)`: `eps` and the exact
  local exponent `d log R / d log eps = -(G + Omega G') D /
  (Omega G' (G + D))`, from `dG/dOmega` (no finite difference along the
  branch). Tests: `1/(1+eps)` and `(1/2)/(1+eps)` on the Lorentzian line
  (1e-13), the Voigt closed form (1e-12), a 30-digit central difference
  along the branch (1e-12).
- `parametric.eps_window(line, kernel, rel_tol=0.01)`: how far above the
  threshold the local exponent stays within `rel_tol` of `beta`, for
  choosing the range of a fit. Test: `rel_tol/(1 - rel_tol)` for the
  conservative and Kuramoto kernels on the Lorentzian line (1e-8).
- `kernels.Kernel.breaks` (default empty): kinks of `W`, used as
  quadrature break points; the Kuramoto average with `k_min != 1` has
  one at `v = k_min^eta`.
- Tests for `physicality` and `valid_window`, which had none.
- README example 12 and a Tutorials section; 172 tests (from 128).

### Fixed

- `heterogeneous` took the weight integral on a linear scale out to
  infinity and lost accuracy far out in the tail (and, for the
  Kuramoto base with `k_min != 1`, at the kink `v = k_min`). It is now
  reduced, for every base kernel, to `a X^-a int_0^X x^(a-1) W(x) dx`
  (`a = (gamma-2)/eta`, `X = |v|/k_min^eta`), one quadrature scaled by
  the size of the tail (the Kuramoto base keeps its closed form, now for
  any `k_min`). Before -> after, relative to closed forms at 15 digits:
  conservative base, `gamma = 2.6`, `eta = 0.5`: `W(1e9)` -2.9e-2 ->
  2e-16, mass 18.84621 -> 18.84956 (closed form 18.84956; it took
  107 s, now 7 s); Gaussian base, `gamma = 2.5`, `eta = 0.25`: `W(1e4)`
  -2.5e-2 -> -2e-15, `W(1e6)` -9.5e-2 -> 9e-16; conservative base,
  `gamma = 4.4`: `W(1e9)` +4.1e-3 -> 2e-15; Kuramoto base, `k_min = 2`,
  `gamma = 3.8`: `W(1e9)` +2.8e-4 -> 0, mass +2.8e-6 -> 2e-16, and
  `amplitude_general` no longer refuses it ("not converged": 2.389384
  at 1e6 against 2.390062 at 1e9; the closed form is 2.389392). Tests:
  `test_averaged_kernel_closed_forms`, `test_averaged_kuramoto_with_k_min`,
  `test_averaged_kernel_mass`,
  `test_amplitude_general_accepts_an_exact_power_tail_with_k_min`.
- `bimodal_gaussian` with a negative separation got the integration
  scale `width + sep/2`, zero or negative for `sep <= -2 width`: for
  `sep = -4`, `G(0.1)` 0.390375 -> 0.195187, `c` -0.0878293 ->
  -0.0891192, and `branch_point` with the Kuramoto kernel raised
  TypeError. The scale is now `width + |sep|/2` (unchanged for
  `sep >= 0`). Test: `test_negative_separation_is_the_same_line`.
- `G_of_Omega` with a negative `Omega` returned twice `G(|Omega|)`
  (3.3333 instead of 1.6667 at -0.1 on the Lorentzian line), and
  `branch_point` raised TypeError (negative) or ZeroDivisionError
  (infinite). They now refuse. Test:
  `test_negative_or_infinite_omega_is_refused`.
- `class_table` with the free edge at or beyond `delta_max` (as
  `build_system` places it for `chiN` above about 125 line widths)
  reversed the range of the free bins, and the last one counted the
  population between `delta_max` and the edge a second time: for the
  edge at 5000 and `delta_max` 1000 the raw total was 1.000197 and the
  locking integral at `Omega = 0.5` was off by 2.1e-4 -> 1.4e-5. The
  free bins now span one decade beyond the edge in that case. Test:
  `test_class_table_free_edge_beyond_delta_max`.

### Changed

- `LineShape` refuses a scale that is not finite and positive (the
  integrals put their break points at multiples of it).
- `G_of_Omega`, `H_of_Omega` and `branch_point` refuse a negative or
  non-finite `Omega`.
- Type hints on `branch_point`, `sweep` and the new functions.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged. The source distribution now
  includes `examples/`.
- All 128 earlier tests pass (the version check now expects 1.4.0), and
  README examples 1 to 11 print the same as with 1.3.0.

### Performance

- `branch_point` and `sweep` (so `extract_beta`) remember each kernel
  value `W(u)` within a call, and `G(0) - G(Omega)` reuses its first
  pass in its second: `extract_beta` of example 6 (averaged Kuramoto
  kernel, 25 digits) 4.94 s -> 2.01 s (2.5x) and of example 1 (`s =
  2.5`) 2.44 s -> 1.51 s (1.6x), medians of 5 runs on the same machine;
  single `branch_point` calls 1.0x to 2x. Bit-identical: the test runs
  the 1.3.0 code path (kept in the test) on five line-kernel pairs and
  compares every number exactly (`test_branch_point_and_sweep_bit_identical`).
- `points_for_beta` finds the number of points by doubling and
  bisection instead of counting up from 3: the same result (checked
  against the old count, `test_points_for_beta_same_as_counting`);
  15716 points 0.068 s -> 0.00009 s, and a refusal (over 10^7 points)
  53 s -> 0.00007 s.

### Tutorials

- `examples/tutorial_which_kernel.py`: conservative or overdamped? From
  a synthetic measured onset on a Voigt line with width errors, coupling
  errors and calibration, the exponent, the kernel mass from the
  threshold and the predicted amplitude decide between the two models.
- `examples/tutorial_plan_and_fit.py`: plan the range with `eps_window`
  and the number of points with `points_for_fit`, measure with
  `branch_at_coupling`, fit, and read off the kernel tail; points beyond
  the window give a biased exponent that the chi-square test flags.
- Both print the output quoted in the README, which
  `tests/test_tutorials.py` checks along with the truth of their
  synthetic data.

## 1.3.0 (2026-10-10)

### Added

- `fit_branch(..., sigma_chi=...)`: errors of the measured couplings.
  Near the threshold an error in chi moves log R by
  `beta sigma_chi / (chi - chi_c)`; this is added to each point's
  variance at the fitted chi_c and beta, and the fit is repeated until
  the weights settle (effective variance, first order in sigma_chi).
  Requires `sigma_r`; refused where sigma_chi exceeds 0.25 (chi -
  chi_c). Tests: with coupling noise of 0.1 (chi - chi_c) at the first
  point, the reported error bars of chi_c, beta and A match the scatter
  of 300 seeded fits within 12 %, while without `sigma_chi` sigma_beta
  is more than 15 % too small (about 23 %); `sigma_chi = 0` reproduces
  the plain fit exactly. By hand, 1000 seeded fits at 0.1, 0.2 and 0.3
  held within 6 %, for noise on the true or on the recorded coupling.
- Goodness of fit: with `sigma_r`, `BranchFit` now carries `chi2`,
  `dof` and `p_value` (chi-square tail probability), `None` without
  `sigma_r`. Tests: chi2 recomputed from the returned parameters;
  p-values of 300 seeded fits under the model pass a Kolmogorov-Smirnov
  test for uniformity and chi2 averages to dof; on the exact branch
  `R = eps/(1+eps)` (conservative kernel, Lorentzian line) with 1 %
  noise, p > 0.05 for points up to eps = 0.01 and p < 1e-10 (with beta
  more than 10 error bars from 1) up to eps = 0.5.
- `measured.with_calibration(fit, chi_scale_rel, r_scale_rel)` (also at
  the top level): adds the relative uncertainty of an overall coupling
  scale to sigma_chi_c and of an overall R scale to sigma_A, in
  quadrature; beta is exactly immune to both. Tests: refitting data
  with the couplings scaled by 1.07 scales chi_c by 1.07 to 1e-9 and
  leaves beta and A unchanged to 1e-7; scaling R scales only A.
- `BranchFit` gains `dof`, `chi2`, `p_value`, `sigma_chi_used` and
  `calibration_rel`, all with defaults.
- Tests for `class_exact`, which had none: it matches `full_exact` to
  1e-10 in R and 1e-9 in xi^2.
- README examples 10 (consistency of data with the power law) and 11
  (coupling errors and calibration).

### Fixed

- `Kernel.mass()` and `Kernel.second_moment()` integrated slow
  algebraic tails directly to infinity. At 15 digits `power_tail(1.1)`
  had mass 20.0312 instead of 20.2745 (3.7e-4 relative error at 30
  digits), so `threshold` on a Lorentzian line of FWHM 1 was 0.0784174
  instead of 0.0774765; `second_moment` of `power_tail(3.1)` was
  19.7910 instead of 20.0343; the averaged Kuramoto kernel with
  gamma = 3.4 had a mass 1.4e-8 off. The tail is now mapped onto (0, 1]
  as in `G_of_Omega`; all match their closed forms to 1e-13 or better
  at 15 digits (the `power_tail` masses to 1e-14, and to 1e-28 at 30
  digits). Tests: `test_mass_of_slow_tails`,
  `test_threshold_of_a_slow_tail_matches_the_branch`,
  `test_heterogeneous_kuramoto_mass_closed_form`,
  `test_second_moment_of_slow_tails`.
- `tail_integral` integrated the singularity delta^(2-s) at the center
  directly: on a Lorentzian line of FWHM 1 at 15 digits I_s was 47.063
  instead of 47.715 at s = 2.9, 86.97 instead of 98.49 at s = 2.95, and
  7.9999999964 instead of 8 at s = 2.5. It now substitutes
  delta = scale t^(1/(3-s)), takes p' with guard digits (and as
  p''(0) delta very close to the center), and integrates in units of the
  line width; it matches closed forms on Lorentzian and Gaussian lines of
  widths 0.001, 1 and 1000 to 1e-13 for s from 1.1 to 2.99. Test:
  `test_tail_integral_closed_forms`, `test_tail_integral_at_30_digits`.
- `amplitude_general`, through the two fixes above, on a Gaussian line
  at 15 digits: 8.4219 -> 9.6179 for s = 1.1 and 0.45574 -> 0.45258 for
  s = 2.9. Its default tail amplitude C (W(v) v^s at v = 1e6) was used
  unchecked: for `heterogeneous(conservative(), 3.8)` it is 6.2 % below
  the asymptotic value (8.5819 against the closed form 9.1498), which
  made A 8.3 % high (0.60510 against 0.55853). It is now compared with
  the value at v = 1e9 and refused if they differ by more than 1e-6.
  Tests: `test_general_amplitude_closed_form_on_a_lorentzian`,
  `test_general_amplitude_tail_amplitude_check`.
- `fit_branch` returned the limit of its search as the exponent: for
  noiseless R = 0.8 eps^0.04 it reported beta = 0.0500 +- 0.0040, for
  eps^25 beta = 20 +- 1.5. It now refuses. Test:
  `test_fit_refuses_an_exponent_at_the_search_limit`.
- `class_exact` skipped a time earlier than the previous one and
  returned the later state: times [1.0, 0.2] on the system of README
  example 9 gave R = 0.677579 for both; at t = 0.2 it is 0.984316. It
  now propagates in either direction. Test:
  `test_class_exact_times_in_any_order`.

### Changed

- Input checks with clear messages: line shapes refuse widths that are
  not finite and positive (`lorentzian(-1)` used to return a negative
  density); `power_tail` refuses a non-finite s; `predicted_beta`
  refuses s <= 1 (it returned -2.0 for s = 0.5); `heterogeneous`
  refuses non-positive eta or k_min; `fit_branch` refuses non-positive
  couplings (before, SciPy's message about bounds), arrays that are not
  one-dimensional, non-finite sigma_r and a negative or non-finite
  min_decade; `kernel_tail_from_beta`
  refuses a non-finite sigma_beta (it returned a NaN error) and a
  negative n_sigma; `beta_relative_sigma` and `plan_fit` refuse a
  number of points that is not a whole number (12.9 was taken as 12)
  and non-finite inputs; `amplitude_general` refuses a kernel without a
  tail in 1 < s < 3 with a ValueError (it raised TypeError) and a
  non-positive C; `class_exact` refuses populations that are not whole
  numbers (they were truncated) and mismatched lengths.
- Type hints on the public functions of `measured`, and on
  `predicted_beta` and the `exact` functions.
- README: the lab half's new features, the new checks and refusals,
  corrections, limits, and 128 tests (from 97). Examples 1 to 9 print
  the same as with 1.2.0. All 97 earlier tests pass unchanged.

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
