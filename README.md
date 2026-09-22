# lockkernel

[![PyPI](https://img.shields.io/pypi/v/lockkernel.svg)](https://pypi.org/project/lockkernel/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22829483-blue)](https://doi.org/10.5281/zenodo.22829483) [![tests](https://github.com/TaN-MM-Org/lockkernel/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/lockkernel/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`lockkernel` is a Python package about **synchronization**: many
oscillators, each with a slightly different natural frequency, start to
move in step once the coupling between them is strong enough. Below a
certain coupling (the **threshold**) nothing happens; above it, a
collective motion appears and grows. The package answers the questions
one asks about that onset:

- At what coupling does synchronization start?
- How fast does it grow above that point? Near the threshold the growth
  follows a power law, and the power is called the **exponent**
  (`beta`).
- Is the onset smooth, or does the system jump and show hysteresis
  (a first-order onset)?
- Given a **measured** onset, from an experiment or a simulation, what
  are the threshold and the exponent, with error bars? What does the
  exponent say about the system? How many measurements does a target
  error bar need?

It has two halves. The **theory half** computes the onset exactly from
two inputs: the spread of natural frequencies, and the **locking
kernel**, a function that says how strongly an oscillator follows the
group as a function of how far its own frequency is from the group's.
The collective motion must be the one the oscillators themselves
produce (the "self-consistency condition"); one substitution turns
that condition into an exact formula (a "parametric solution") valid
for any frequency spread and any kernel. The exponent depends only on how fast the kernel falls off
far from the centre (its **tail**). The **lab half** (`lockkernel.measured`)
works the other way round: it fits measured points for the threshold
and the exponent, reads the kernel's tail back off the exponent, and
plans how many points a target error bar costs. When the data cannot
support the number asked for, it stops with an error message that says
why, rather than returning a number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install and conventions](#install-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Detuning** `delta` -- how far one oscillator's natural frequency
  is from the centre of the group.
- **Line shape** `p(delta)` -- the distribution of detunings across
  the group (the name comes from spectroscopy, where it is the shape
  of a spectral line). The package ships a Lorentzian, a Gaussian, a
  Student-t (heavy tails), a box (flat) and a two-peaked Gaussian, and
  you can build your own `LineShape`.
- **Coupling** `chiN` -- the coupling per oscillator times the number
  of oscillators. The **threshold** `chiN_c` is the coupling where
  synchronization starts.
- **Order parameter** `R` -- how synchronized the group is: 0 for no
  common motion, 1 for perfect step. It is zero below the threshold.
- **Reduced coupling** `eps = chiN/chiN_c - 1` -- the relative
  distance above the threshold. Near the onset `R ~ A eps^beta`, with
  **amplitude** `A` and **exponent** `beta`.
- **Locking bandwidth** `Omega = chiN * R` -- the range of detunings
  over which an oscillator follows the group.
- **Locking kernel** `W(u)` -- the time-averaged share of an
  oscillator that lines up with the group, as a function of its
  detuning in units of the locking bandwidth, `u = delta/Omega`.
  `W(0) = 1` (an oscillator at the centre follows perfectly) and `W`
  is even. Its **mass** `m` is its area, `m = integral W(u) du`. Its
  **tail exponent** `s` describes how fast it falls far out:
  `W ~ |u|^-s`. The package's kernels are:
  - *conservative* `W = 1/(1+u^2)`, for spins with no damping
    (`s = 2`, mass `pi`);
  - *Kuramoto* `W = sqrt(1-u^2)` for `|u| < 1` and 0 outside, for
    overdamped phase oscillators, whose motion is dominated by friction
    (the classic Kuramoto model; no tail at all, mass `pi/2`);
  - the test family `W = 1/(1+|u|^s)`, and a Gaussian kernel;
  - a kernel averaged over a broad, power-law spread of coupling
    strengths between oscillators (`heterogeneous`).
- **The rule that links them** (from the `parametric` module's
  derivation, and tested below): the threshold is
  `chiN_c = 1/(p(0) m)`, and the exponent is
  `beta = 1/(s-1)` for `1 < s < 3` and `beta = 1/2` for `s >= 3` or
  for a kernel with no algebraic tail. So a measured `beta` above 1/2
  names the tail, `s = 1 + 1/beta`, while `beta = 1/2` only says
  "`s >= 3`" and cannot name one value.
- **Order of the onset** -- for the conservative kernel, the sign of
  one number `c` (an integral over the line shape) decides it:
  `c > 0` gives a smooth (continuous) onset, `c < 0` a jump with
  hysteresis (first order).
- **Cumulant dynamics**, **Wineland parameter** -- terms from the
  quantum-spin half of the package; see
  [What is in the package](#what-is-in-the-package).

## Install and conventions

```
pip install lockkernel
```

It needs Python 3.9 or newer, NumPy 1.22 or newer, SciPy 1.8 or newer
and mpmath 1.2 or newer, and nothing else.

- **Units.** Detunings, line widths, couplings `chiN` and the locking
  bandwidth `Omega` share one frequency unit of your choice (the line
  width is a natural one). In the dynamics modules, time is in the
  inverse of that unit (the Hamiltonian is written with hbar = 1).
- **Precision.** The theory half (`parametric`, `kernels`,
  `lineshapes`) works in mpmath's arbitrary precision and returns
  mpmath numbers. Set the working precision with `mpmath.mp.dps`
  (decimal digits). The lab half and the dynamics work in ordinary
  floating point with NumPy.
- **Line shapes** are normalised probability densities; `lorentzian`
  and `gaussian` take a full width at half maximum, `box` a half
  width.

## Examples

Each example below runs as written, and the output shown is what it
printed with lockkernel 1.1.1. Line widths, couplings and noise levels
are illustrative values, not taken from any experiment.

### 1. Threshold and exponent from the theory

```python
import mpmath as mp
from lockkernel.lineshapes import lorentzian, gaussian
from lockkernel.kernels import conservative, kuramoto, power_tail, predicted_beta
from lockkernel.parametric import threshold, extract_beta

mp.mp.dps = 25    # working precision of mpmath, in decimal digits

line = lorentzian(fwhm=1.0)
print("threshold, conservative kernel:", mp.nstr(threshold(line, conservative()), 15))
print("threshold, Kuramoto kernel:    ", mp.nstr(threshold(line, kuramoto()), 15))

# Local slopes d log R / d log eps at Omega = 1e-3 ... 1e-6 (closest to onset last).
slopes = extract_beta(gaussian(1.0), power_tail(2.5), [-3, -4, -5, -6])
print("local exponents:", [mp.nstr(b, 8) for b in slopes])
print("predicted 1/(s-1):", predicted_beta(2.5))
```

```
threshold, conservative kernel: 0.5
threshold, Kuramoto kernel:     1.0
local exponents: ['0.67152587', '0.6681861', '0.66714445']
predicted 1/(s-1): 0.6666666666666666
```

The threshold is `1/(p(0) m)`: the kernel enters only through its
mass, so the Kuramoto threshold (mass `pi/2`) is twice the
conservative one (mass `pi`) on every line. `extract_beta` steps the
exact branch towards the onset and returns the local slope between
neighbouring points; the list settling down is the evidence that the
exponent has been reached. List the exponents of `Omega` so that the
one closest to the onset comes last, because the last entry is the
one to quote. Here it is within 0.1 % of `2/3`.

### 2. The exact branch against a closed form

```python
import mpmath as mp
from lockkernel.lineshapes import lorentzian
from lockkernel.kernels import conservative
from lockkernel.parametric import branch_point

mp.mp.dps = 30
line, kern = lorentzian(fwhm=1.0), conservative()
a = mp.mpf(1) / 2                        # half width of this line
for Omega in ["1e-2", "1e-4", "1e-6"]:
    chiN, R, eps = branch_point(line, kern, mp.mpf(Omega))
    gap = abs(R - (1 - a / chiN))
    print(f"Omega={Omega}: chiN={mp.nstr(chiN, 10)}  R={mp.nstr(R, 10)}  "
          f"eps={mp.nstr(eps, 6)}  matches 1 - a/chiN to 1e-28: {gap < 1e-28}")
```

```
Omega=1e-2: chiN=0.51  R=0.01960784314  eps=0.02  matches 1 - a/chiN to 1e-28: True
Omega=1e-4: chiN=0.5001  R=0.000199960008  eps=0.0002  matches 1 - a/chiN to 1e-28: True
Omega=1e-6: chiN=0.500001  R=1.999996e-6  eps=2.0e-6  matches 1 - a/chiN to 1e-28: True
```

The branch is traced by choosing the locking bandwidth `Omega` and
reading off the coupling and the order parameter. No equation is
solved, so nothing is lost to cancellation near the threshold. For
conservative spins on a Lorentzian line the answer is known in closed
form (`chiN = a + Omega`, `R = 1 - a/chiN`), and the package matches
it to 1e-28 at 30 digits.

### 3. Amplitude, and whether the onset is smooth

```python
import mpmath as mp
from lockkernel.lineshapes import lorentzian, gaussian, bimodal_gaussian
from lockkernel.parametric import amplitude, c_coefficient

for line in (lorentzian(1.0), gaussian(1.0)):
    print(f"{line.name:10s} A = {mp.nstr(amplitude(line), 12)}")
print("pi/2       =", mp.nstr(mp.pi / 2, 12))

# Two Gaussian peaks (width 1) at +-sep/2: the sign of c decides the order.
for sep in (1.0, 4.0):
    c = c_coefficient(bimodal_gaussian(sep))
    kind = "continuous onset" if c > 0 else "first order (hysteretic) onset"
    print(f"sep = {sep}: c = {mp.nstr(c, 6)} -> {kind}")
```

```
lorentzian A = 1.0
gaussian   A = 1.57079632679
pi/2       = 1.57079632679
sep = 1.0: c = 0.245044 -> continuous onset
sep = 4.0: c = -0.0891192 -> first order (hysteretic) onset
```

For conservative spins the exponent is 1, so near the onset
`R = A eps`, with `A = pi p(0)^2 / c`. This example runs at mpmath's
default 15 digits; before version 1.1.1 `amplitude` and
`c_coefficient` needed about 30 digits to be right (see
[Corrections](#corrections-in-earlier-versions)). The two-peaked line
is not covered by the tests; the sign rule it illustrates is the one
derived in the `parametric` module.

### 4. Fit a measured branch

```python
import numpy as np
from lockkernel import fit_branch, kernel_tail_from_beta
from lockkernel.lineshapes import lorentzian
from lockkernel.kernels import power_tail
from lockkernel.parametric import sweep, threshold

# Stand-in "measurement": 14 points of the exact branch for a kernel with
# tail s = 2.5 (so beta = 2/3), with 2 % seeded noise on R.
line, kern = lorentzian(1.0), power_tail(2.5)
pts = sweep(line, kern, np.linspace(-5.0, -2.4, 14))
chi = np.array([float(p[0]) for p in pts])
r = np.array([float(p[1]) for p in pts])
rng = np.random.default_rng(1)
r_meas = r * np.exp(rng.normal(0.0, 0.02, r.size))

fit = fit_branch(chi, r_meas, reference="illustrative: exact branch + 2 % noise",
                 sigma_r=0.02 * r_meas)
print(f"threshold chi_c = {fit.chi_c:.5f} +- {fit.sigma_chi_c:.1e}"
      f"   (exact {float(threshold(line, kern)):.5f})")
print(f"exponent  beta  = {fit.beta:.3f} +- {fit.sigma_beta:.3f}   (exact 0.667)")
print(f"amplitude A     = {fit.amplitude:.3f} +- {fit.sigma_amplitude:.3f}")
print(f"eps range {fit.eps_range[0]:.2e} .. {fit.eps_range[1]:.2e}, "
      f"rms log residual {fit.residual_rms_log:.3f}")

s, s_err = kernel_tail_from_beta(fit.beta, fit.sigma_beta)
print(f"kernel tail s = {s:.2f} +- {s_err:.2f}   (true 2.5)")
```

```
threshold chi_c = 0.59441 +- 3.5e-09   (exact 0.59441)
exponent  beta  = 0.673 +- 0.003   (exact 0.667)
amplitude A     = 0.652 +- 0.019
eps range 1.56e-07 .. 1.09e-03, rms log residual 0.011
kernel tail s = 2.49 +- 0.01   (true 2.5)
```

`fit_branch` fits `R = A (chi/chi_c - 1)^beta` for the threshold,
exponent and amplitude together. With your own data, pass your
couplings and order parameters, and a `reference` that says where
they come from (it is required). The error bars are the standard
asymptotic ones of a least-squares fit: they are right when the power
law holds over the fitted range and `sigma_r` is right. Without
`sigma_r`, the scatter of the points sets them. Here the fitted
exponent is 2.4 of its own error bars from the exact 2/3; the tests
allow four (see [How the results are checked](#how-the-results-are-checked)).
`kernel_tail_from_beta` turns the exponent into the kernel's tail
exponent `s = 1 + 1/beta`, with error `sigma_beta / beta^2`.

### 5. Plan the measurement, and a refusal

```python
from lockkernel import beta_relative_sigma, points_for_beta, kernel_tail_from_beta

# How many points, spread evenly in log(eps) over 2 decades, with 5 %
# scatter in R, for an error bar of 0.01 on beta (threshold known)?
n, achieved = points_for_beta(0.01, decades=2.0, sigma_log=0.05)
print(f"points needed: {n} (error bar {achieved:.5f}); "
      f"with {n - 1}: {beta_relative_sigma(n - 1, 2.0, 0.05):.5f}")

# A fitted beta = 0.52 +- 0.02 cannot name a kernel tail:
try:
    kernel_tail_from_beta(0.52, 0.02)
except ValueError as err:
    print("refused:", err)
```

```
points needed: 12 (error bar 0.00999); with 11: 0.01035
refused: beta = 0.52 +- 0.02 is consistent with 1/2, which identifies only the CLASS s >= 3 (compact support or decay faster than |u|^-3); no single tail exponent can be named from it
```

`beta_relative_sigma` gives the error bar of a straight-line slope on a
log-log plot. Despite its name, the result is the absolute error of
`beta`, not a relative one. It assumes the threshold is known, while
`fit_branch` also fits the threshold, which is harder, so plan with
some margin. `points_for_beta` finds the smallest number of points
that meets the target.

### 6. A spread of coupling strengths changes the exponent

```python
import mpmath as mp
from lockkernel.lineshapes import gaussian
from lockkernel.kernels import kuramoto, heterogeneous, predicted_beta
from lockkernel.parametric import extract_beta

mp.mp.dps = 25
# Kuramoto oscillators whose coupling weights k follow P(k) ~ k^-3.8 (k >= 1).
ker = heterogeneous(kuramoto(), degree_exponent=3.8)
print("tail exponent s of the averaged kernel:", round(ker.tail, 12))
print("predicted beta:", round(predicted_beta(ker.tail), 12))
beta = extract_beta(gaussian(1.0), ker, [-3, -4, -5, -6])[-1]
print("beta from the branch:", mp.nstr(beta, 6))
```

```
tail exponent s of the averaged kernel: 1.8
predicted beta: 1.25
beta from the branch: 1.24989
```

When oscillators couple with different strengths `k`, drawn from a
power law `P(k) ~ k^-gamma`, the kernel that decides the onset is an
average over them. A strongly coupled oscillator stays locked far out
in detuning, so the averaged kernel gets a tail even when each
oscillator's own kernel has none: `s = min(s0, (gamma-2)/eta)`, with
`s0` the tail of the single-oscillator kernel (infinite for Kuramoto).
Here `s = 1.8`, so `beta = 1/(s-1) = 1.25`, which is `1/(gamma-3)`.

### 7. The quantum spin model against exact diagonalisation

```python
import numpy as np
from lockkernel.cumulant import System, coherent_state_x, evolve, coherence, wineland_xi2
from lockkernel.exact import full_exact

# 8 emitters, two at each of four detunings, all starting along +x.
deltas = np.array([-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0])
chiN, times = 1.5, np.array([0.05, 0.2, 1.0])

u, count = np.unique(deltas, return_counts=True)
sysm = System(delta=u, n=count.astype(float), chiN=chiN)
states = evolve(coherent_state_x(sysm), sysm, times[-1], t_eval=times,
                rtol=1e-11, atol=1e-14)
R_ex, xi_ex = full_exact(deltas, chiN, times)
for t, st, r, xi in zip(times, states, R_ex, xi_ex):
    print(f"t={t:4.2f}  R: cumulant {coherence(st, sysm):.6f} exact {r:.6f}   "
          f"xi^2: cumulant {wineland_xi2(st, sysm):.4f} exact {xi:.4f}")
```

```
t=0.05  R: cumulant 0.999012 exact 0.999012   xi^2: cumulant 0.9374 exact 0.9374
t=0.20  R: cumulant 0.984309 exact 0.984316   xi^2: cumulant 0.7818 exact 0.7824
t=1.00  R: cumulant 0.674467 exact 0.677579   xi^2: cumulant 0.5328 exact 0.5771
```

The dynamics modules treat the conservative case as a quantum model:
two-level emitters (spin one-half) with equal all-to-all exchange
coupling and spread detunings, with no damping. The **cumulant**
solver follows the average of each emitter group plus the pair
correlations between emitters, and drops higher correlations. That
makes it fast for large numbers of emitters, but it is an
approximation. For small groups `full_exact` solves the full quantum
problem by exact diagonalisation (computing the quantum states of the
whole group directly, which is only affordable for a few emitters).
The two agree at short times and drift apart later, as a truncation
should. `xi^2` is the **Wineland spin-squeezing parameter**:
1 for uncorrelated spins, below 1 when the spins are entangled in a way
that improves phase measurements.

## What is in the package

The top level imports the `measured` names below and the submodules
`lineshapes`, `kernels`, `parametric` and `measured`. Import
`lockkernel.cumulant`, `lockkernel.ensemble` and `lockkernel.exact`
directly.

**Line shapes and kernels** (`lockkernel.lineshapes`, `lockkernel.kernels`)

- `LineShape` -- a symmetric density with its value at the centre
  (`p0()`), and, where available, a floating-point `cdf` and `ppf`.
  Makers: `lorentzian`, `gaussian`, `student_t`, `box`,
  `bimodal_gaussian`; `LINESHAPES` maps names to them.
- `Kernel` -- an even kernel with `W(0) = 1`, its tail exponent, its
  support, and `mass()`. Makers: `conservative`, `kuramoto`,
  `power_tail(s)`, `gaussian_kernel`; `KERNELS` maps names to them.
- `heterogeneous(base, degree_exponent, eta=1, k_min=1)` -- the kernel
  averaged over a power-law spread of coupling strengths (example 6).
- `predicted_beta(s)` -- the exponent the rule gives for tail `s`
  (`None` meaning no algebraic tail).

**The exact solution** (`lockkernel.parametric`, mpmath precision)

- `G_of_Omega`, `H_of_Omega` -- the two integrals the solution is built
  on: `G(Omega) = integral p(Omega u) W(u) du` and
  `H = Omega G`, so that `chiN = 1/G` and `R = H`.
- `threshold`, `branch_point`, `sweep`, `extract_beta` -- the
  threshold, one point `(chiN, R, eps)` of the branch, the branch at
  `Omega = 10^e` for a list of `e`, and the local exponents along it.
- `c_coefficient`, `amplitude` -- for the conservative kernel, the
  number `c` whose sign sets the order of the onset, and the amplitude
  `A = pi p(0)^2 / c`.
- `tail_integral`, `amplitude_general` -- the amplitude for a kernel
  with tail exponent `1 < s < 3`.
- `fold_interval` -- samples the branch and, if it folds back, returns
  the coupling range of the hysteresis and the jump in `R`
  (`None` if the branch does not fold). Not covered by the tests.

**Fitting measured data** (`lockkernel.measured`, also at the top level)

- `fit_branch(chi, r, reference, sigma_r=None, min_decade=1.0)` --
  returns a `BranchFit` with `chi_c`, `beta`, `amplitude`, their
  errors, `n_points`, `eps_range`, `residual_rms_log` (the
  root-mean-square misfit in `log R`) and `reference`.
- `kernel_tail_from_beta(beta, sigma_beta=0, n_sigma=2)` -- the tail
  exponent `s` and its error.
- `beta_relative_sigma(n_points, decades, sigma_log)`,
  `points_for_beta(target_sigma_beta, decades, sigma_log)` -- the
  error bar of `beta` for a planned measurement, and the number of
  points for a target error bar.

**Quantum spin dynamics** (`lockkernel.cumulant`, `lockkernel.ensemble`,
`lockkernel.exact`)

- `System`, `State`, `coherent_state_x`, `evolve` -- groups ("classes")
  of emitters with equal detuning, the cumulant state, the start with
  all spins along +x, and the time evolution. Emitters detuned far
  outside the locking range can be kept as free spins that precess
  exactly without being integrated, so the line's tails are not cut.
- `evolve_meanfield` -- the same with all correlations dropped.
- `coherence` (the order parameter `R`), `wineland_xi2`,
  `collective_moments` -- what is read from a state;
  `physicality` and `valid_window` flag when the truncated state stops
  being physical.
- `class_table`, `build_system` -- turn a line shape into a finite
  set of detuning classes that keeps the whole population.
- `symmetric_exact` (all detunings equal, any number of emitters),
  `full_exact` (full quantum problem, up to about twelve emitters) and
  `class_exact` (emitters grouped by detuning, a few tens) -- exact
  references. `class_exact` is not covered by the tests.

Each function's docstring (`help(lockkernel.fit_branch)`, for example)
gives its inputs and conventions.

## When it refuses, and why

`lockkernel` raises an error instead of guessing when:

- measured data come without a `reference` of at least 8 characters
  saying where they come from;
- fewer than 6 points are given, or `chi` and `r` differ in length
  (three parameters with error bars need more);
- the couplings are not finite and strictly increasing, or an order
  parameter is not finite and positive (points below the threshold,
  `R = 0`, carry no exponent information; drop them);
- `sigma_r` is not positive or not on the same grid as `r`;
- the fit does not converge, or its covariance is singular (the data
  cannot pin down the three parameters separately);
- the fitted threshold is indistinguishable from the smallest
  coupling: the data do not reach the onset;
- the fitted points span less than `min_decade` decades of `eps`
  (default one): exponent and amplitude cannot be told apart on so
  short a range;
- a fitted `beta` is consistent with 1/2 (only the class `s >= 3` is
  identified), clearly below 1/2 (no kernel gives that; the fit has
  probably left the near-threshold range), or not positive;
- a planned measurement has fewer than 3 points, non-positive range or
  scatter, a non-positive target, or would need more than 10^7 points;
- `power_tail(s)` is asked for `s <= 1` (the kernel mass would be
  infinite), or `heterogeneous` for `gamma <= 2` or a resulting tail
  `s <= 1`;
- `tail_integral` is asked for `s` outside `1 < s < 3`, where it
  diverges;
- a line shape without a floating-point distribution
  (`bimodal_gaussian`) is asked for `cdf`, `ppf` or a class table;
- the solver of the differential equations fails in `evolve` or
  `evolve_meanfield`.

## How the results are checked

65 automated tests run on every push and pull request, on Python 3.9
to 3.14, and once more on Python 3.10 with the oldest NumPy (1.22.0),
SciPy (1.8.0) and mpmath (1.2.1) the package allows. The numerical
checks compare the package with something independent of it: a closed
form, an exact identity, a second calculation done a different way,
exact diagonalisation, or seeded simulation. The rest check that the
refusals fire, plus one check of the version number and that importing
the package does not import matplotlib. The main checks:

**The exact solution** (30 digits unless stated)

- Kernel masses: `pi` (conservative) and `pi/2` (Kuramoto) to 1e-25;
  `power_tail(s)` for `s` = 1.5, 2, 3, 4 against
  `2 (pi/s) / sin(pi/s)` to 1e-14.
- The Kuramoto threshold is twice the conservative one to 1e-25, on
  three lines.
- Conservative kernel, Lorentzian line: the branch matches
  `chiN = a + Omega`, `R = 1 - a/chiN` to 1e-28 for `Omega` from 1e-1
  to 1e-6. Kuramoto kernel, Lorentzian line: `R = sqrt(1 - chiN_c/chiN)`
  to 1e-26 for `Omega` from 1e-1 to 1e-4.
- The amplitude takes the closed values 1, `pi/2`, 4/3 and `pi^2/4` on
  Lorentzian, Gaussian, Student-t and box lines, to a relative 1e-9,
  and (new in 1.1.1) 1 and `pi/2` to a relative 1e-12 at mpmath's
  default 15 digits (Lorentzian lines of width 1 and 1000, and a
  Gaussian).
- `c` from its integral equals the slope of `G(Omega)/pi` at
  `Omega` = 1e-6 and 2e-6 to a relative 1e-9, on four lines; `R/eps` at `Omega = 1e-6` is
  within a relative 1e-5 of the amplitude.

**The exponent rule** (25 digits, Gaussian line, `Omega` down to 1e-6)

- `beta = 1/(s-1)` or 1/2 for the kernel family with `s` = 1.5, 1.8,
  2.0, 2.2, 2.5, 4 and 6, to a relative 5e-3, 1e-3, 1e-4, 1e-3, 2e-3,
  1e-4 and 1e-6 respectively. At the borderline `s = 3` the local
  exponent lies between 0.5 and 0.56.
- The conservative kernel gives 1, and the Kuramoto and Gaussian
  kernels give 1/2, within 1e-4 on three lines; for `s = 2.5` three
  lines agree within 1e-3.
- The general amplitude matches the branch at `Omega = 1e-7` for
  `s` = 1.8, 2.0 and 2.2 (relative 1e-4, 1e-5, 1e-4) on three lines,
  and equals the `s = 2` form within a relative 1e-6.
- The averaged kernel of example 6 has the tail `(gamma-2)/eta` or
  `s0` in five cases (within 1.5e-2 from its measured slope), and
  gives `beta = 1/(gamma-3)` within 5e-3 for `gamma = 3.8`.

**Fitting measured data** (the theory half checks the lab half)

- On exact branches with no noise, `fit_branch` recovers `beta = 1`
  (conservative) and `beta = 2/3` (`s = 2.5`) within 0.02, and the
  threshold within a relative 1e-3; `s` comes back within 0.1 of 2.5.
- With 2 % seeded noise, the fitted `beta` is within four of its own
  error bars of 1, between 0.9 and 1.1, with an error bar below 0.05.
- `kernel_tail_from_beta(1.0, 0.02)` returns `s = 2` and error 0.02 to
  1e-12.
- The planning formula matches 6000 seeded simulated fits within 5 %,
  and `points_for_beta` returns the smallest `n` that meets the target
  (checked on both sides).

**Dynamics and discretisation**

- Cumulant solver, all detunings equal, against the exact result:
  `xi^2` within 0.2, 0.015 and 0.0015 for 50, 200 and 1000 emitters
  (and `R` within a quarter of that), with the error falling with `N`
  (more than 50 times smaller at 1000 than at 50).
- Cumulant solver, 8 and 10 emitters at several detunings, against
  full diagonalisation: `R` within 1e-7 and `xi^2` within 1e-4 at
  `t = 0.05`; at `t = 0.2` the difference is larger but below 1e-2.
- The equations keep the exact symmetries of the pair correlations to
  a relative 1e-12, and an uncorrelated start gives `xi^2 = 1` and
  `R = 1` to 1e-12.
- The class table keeps the whole population (sums to 1 within 1e-12)
  and reproduces the locking integral within 0.2 %.
- Mean-field dynamics on a Lorentzian line settles at `R = 1 - 1/r`
  within 5e-3 for couplings `r` = 1.2, 2 and 3 times the threshold,
  and stays below 0.05 at 0.6 times the threshold.

Not covered by tests: `fold_interval`, `bimodal_gaussian`,
`class_exact`, `physicality` and `valid_window`.

## Corrections in earlier versions

**1.1.1 fixed the amplitude at ordinary precision.**
`c_coefficient` (and so `amplitude`) lost about 40 digits to a
cancellation near the centre of the line. At mpmath's default 15
digits, `amplitude(lorentzian())` returned about 0.0005 instead of 1;
at 20 digits it was off by about 0.16 %; a Lorentzian 1000 wide lost
digits even at 30. The tests ran at 30 digits with a unit-width line
and did not see it. The integral now runs with 25 extra digits, and a
test at 15 digits was added. If you computed amplitudes or `c` below
30 digits with 1.1.0 or earlier, please re-run them.

1.1.1 also corrects this README: several tolerances it quoted were not
the ones the tests use, it described `class_exact` as tested, and the
CI did not run Python 3.10. The full history is in
[CHANGELOG.md](CHANGELOG.md).

## Limits

- The power law `R ~ eps^beta` is the leading behaviour near the
  onset. Data taken far above it bend away and bias `beta`. The
  one-decade rule and the residuals guard the fit, not your choice of
  range.
- The fit's error bars are the standard asymptotic ones; they are
  right when the power law holds and the noise estimate is right.
- `extract_beta` converges slowly near `s = 3`, where the exponent
  carries logarithmic corrections.
- In the undamped spin model the truncated cumulant equations become
  unstable at long times; check `physicality` / `valid_window` before
  trusting a long run. The time-averaged kernel is an assumption,
  checked against the full dynamics (the mean-field tests above), not
  a theorem.
- `c_coefficient` and `amplitude` are good to about 20 significant
  digits at most, however high `mp.dps` is set: the integral leaves
  out the first 1e-20 line widths next to the centre, a piece of
  relative size about 1e-20.
- `fold_interval` is slow (each branch point is a high-precision
  integral, and it takes 400 by default) and is not covered by the
  tests.
- `build_system`'s docstring refers to a convergence check in
  `scripts/vlasov_check.py`; that script belongs to the research
  repository and is not part of this package.

## Where it comes from

This package is the maintained distribution of the reference
implementation for the locking-kernel universality study
([Tanvir-Mahmud-Mahim/locking-kernel-universality](https://github.com/Tanvir-Mahmud-Mahim/locking-kernel-universality),
concept DOI
[10.5281/zenodo.22696369](https://doi.org/10.5281/zenodo.22696369)),
whose scripts, archived run records and figures remain with the
study. The core modules were carried over unchanged in v1.1.0 (1.1.1
changes only `c_coefficient`); the `measured` module and the
packaging are new here. Copyright as in [NOTICE](NOTICE).

## Citing, support and license

If `lockkernel` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22829483](https://doi.org/10.5281/zenodo.22829483),
which always resolves to the latest release; every release is archived
on Zenodo. [CITATION.cff](CITATION.cff) has the details.

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches the physics arrives with a
test, and a claim arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/lockkernel/issues).
Usage questions are welcome alongside bug reports; a docstring that
left a unit or a convention unclear is treated as a documentation
bug, not user error.

Licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
