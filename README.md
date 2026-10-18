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
far from the center (its **tail**). The **lab half** (`lockkernel.measured`)
works the other way round: it fits measured points for the threshold
and the exponent, reads the kernel's tail back off the exponent, and
plans how many points a target error bar costs. It also tests whether
the data are consistent with the power law at their stated noise, and
carries the uncertainties of the measured couplings and of calibration
scales into the error bars. Since 1.4.0 it also answers the planning
questions in theory terms: what `R` a model gives at the coupling you
set, and how far above the threshold the onset is still a power law.
Since 1.5.0 it can also fit a measured onset over its whole range with
the exact branch of a model, not only the power law near the threshold.
That tests the model and calibrates the coupling scale.
When the data cannot
support the number asked for, it stops with an error message that says
why, rather than returning a number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install and conventions](#install-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [Tutorials](#tutorials) (three worked lab analyses, from data to result)
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
  Voigt line (a Gaussian convolved with a Lorentzian, new in 1.4.0), a
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
  "`s >= 3`" and cannot name one value. The value 1/2 needs a line
  with a rounded top at its centre (`p''(0) < 0`, true for every
  shipped line except `box`). On a line that is flat at the centre,
  such as `box`, a tail gives `beta = 1/(s-1)` for every `s > 1`
  (so below 1/2 once `s > 3`), and a kernel with no tail gives a jump
  instead of a power law (see [Limits](#limits)).
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
  (decimal digits). The reduced coupling `eps` keeps that precision
  however close to the threshold it is (since 1.2.0), so the working
  precision does not need to exceed the number of decades you step
  towards the onset. The lab half and the dynamics work in ordinary
  floating point with NumPy.
- **Line shapes** are normalized probability densities; `lorentzian`
  and `gaussian` take a full width at half maximum, `box` a half
  width, and `voigt` the full widths of its Gaussian and Lorentzian
  parts.

## Examples

Each example below runs as written, and the output shown is what it
printed with lockkernel 1.5.0 (examples 1 to 12 print the same as with
1.4.0). Line widths, couplings and noise levels are illustrative
values, not taken from any experiment.

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
[Corrections](#corrections-in-earlier-versions)). For two unit-width
Gaussian peaks at `+-a`, `c = (1 - 2x D(x))/pi` with `x = a/sqrt(2)`
and `D` Dawson's function, and the tests hold `c_coefficient` to this
formula. So the onset turns first order when the peaks are more than
2.61386 widths apart, where Dawson's function has its maximum.
Example 8 follows the first-order case through its hysteresis loop.

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
threshold chi_c = 0.59441 +- 3.8e-09   (exact 0.59441)
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
`sigma_r`, the scatter of the points sets them. Since 1.2.0 they are
built from exact derivatives; 1.1.1 printed `3.5e-09` for the
threshold here (see [Corrections](#corrections-in-earlier-versions)). Here the fitted
exponent is 2.4 of its own error bars from the exact 2/3; the tests
allow four (see [How the results are checked](#how-the-results-are-checked)).
`kernel_tail_from_beta` turns the exponent into the kernel's tail
exponent `s = 1 + 1/beta`, with error `sigma_beta / beta^2`.

### 5. Plan the measurement, and a refusal

```python
from lockkernel import (beta_relative_sigma, points_for_beta, kernel_tail_from_beta,
                        plan_fit, points_for_fit)

# How many points, spread evenly in log(eps) over 2 decades, with 5 %
# scatter in R, for an error bar of 0.01 on beta (threshold known)?
n, achieved = points_for_beta(0.01, decades=2.0, sigma_log=0.05)
print(f"points needed: {n} (error bar {achieved:.5f}); "
      f"with {n - 1}: {beta_relative_sigma(n - 1, 2.0, 0.05):.5f}")

# The same range (eps = 1e-4 .. 1e-2), but with the threshold fitted too,
# as fit_branch does it:
plan = plan_fit(n, 1e-4, 1e-2, beta=2/3, sigma_log=0.05)
print(f"{n} points, threshold fitted: beta +- {plan.sigma_beta:.4f}, "
      f"chi_c +- {plan.sigma_chi_c_rel:.1e} (relative)")
n_fit, plan = points_for_fit(0.01, 1e-4, 1e-2, beta=2/3, sigma_log=0.05)
print(f"points needed with the threshold fitted: {n_fit} "
      f"(error bar {plan.sigma_beta:.5f})")

# A fitted beta = 0.52 +- 0.02 cannot name a kernel tail:
try:
    kernel_tail_from_beta(0.52, 0.02)
except ValueError as err:
    print("refused:", err)
```

```
points needed: 12 (error bar 0.00999); with 11: 0.01035
12 points, threshold fitted: beta +- 0.0197, chi_c +- 1.4e-05 (relative)
points needed with the threshold fitted: 57 (error bar 0.00995)
refused: beta = 0.52 +- 0.02 is consistent with 1/2, which identifies only the CLASS s >= 3 (compact support or decay faster than |u|^-3); no single tail exponent can be named from it
```

`beta_relative_sigma` gives the error bar of a straight-line slope on a
log-log plot. Despite its name, the result is the absolute error of
`beta`, not a relative one. It assumes the threshold is known, and
`points_for_beta` finds the smallest number of points that meets the
target on that assumption.

`fit_branch` fits the threshold as well, and that costs precision:
the same 12 points give an error bar about twice as large. `plan_fit`
gives the error bars `fit_branch` will report (for `beta`, and
relative ones for `chi_c` and `A`), before any data are taken, from
the planned range of `eps` and the expected `beta`. `points_for_fit`
turns that into a number of points: 57 instead of 12 here. How much
fitting the threshold costs depends on how many decades the points
span: for 12 points the error bar grows about 2.0 times over two
decades, 1.6 over three, and 1.3 over five. These are the standard
asymptotic error bars, held in the tests against 300 seeded simulated
fits (within 12 %).

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

### 7. The amplitude when beta = 1/2

```python
import mpmath as mp
from lockkernel.lineshapes import lorentzian, gaussian
from lockkernel.kernels import kuramoto, gaussian_kernel
from lockkernel.parametric import amplitude_curvature, branch_point

mp.mp.dps = 20
for line, kern in [(lorentzian(1.0), kuramoto()), (gaussian(1.0), kuramoto()),
                   (gaussian(1.0), gaussian_kernel())]:
    A = amplitude_curvature(line, kern)
    _, R, eps = branch_point(line, kern, mp.mpf("1e-6"))
    print(f"{kern.name:9s} kernel, {line.name:10s} line: A = {mp.nstr(A, 12)}, "
          f"R/sqrt(eps) at Omega = 1e-6: {mp.nstr(R / mp.sqrt(eps), 12)}")
print("sqrt(pi) =", mp.nstr(mp.sqrt(mp.pi), 12))
```

```
kuramoto  kernel, lorentzian line: A = 1.0, R/sqrt(eps) at Omega = 1e-6: 0.999999999999
kuramoto  kernel, gaussian   line: A = 1.77245385091, R/sqrt(eps) at Omega = 1e-6: 1.7724538509
gaussian  kernel, gaussian   line: A = 1.41421356237, R/sqrt(eps) at Omega = 1e-6: 1.41421356237
sqrt(pi) = 1.77245385091
```

For a kernel with no tail, or a tail `s > 3`, the onset is
`R = A eps^(1/2)`. The first correction to the self-consistency then
comes from the curvature of the line at its centre, `p''(0)`, and
`amplitude_curvature` returns
`A = G0^(3/2) sqrt(2 / (-p''(0) M2))`, where `G0 = p(0) m` and
`M2 = integral u^2 W(u) du` is the kernel's second moment
(`Kernel.second_moment()`). For the Kuramoto kernel on a Lorentzian
line this is 1, the closed form `R = sqrt(1 - chiN_c/chiN)`. On a
Gaussian line the width cancels, which leaves `sqrt(pi)` (Kuramoto
kernel) and `sqrt(2)` (Gaussian kernel) at any width. The branch
approaches these values with corrections of relative size
`Omega^2`, or `Omega^(s-3)` for a tail `3 < s < 5`.

### 8. A first-order onset: the hysteresis loop

```python
import mpmath as mp
from lockkernel.lineshapes import bimodal_gaussian
from lockkernel.kernels import conservative
from lockkernel.parametric import fold_interval, threshold, extract_beta

line = bimodal_gaussian(4.0)      # two Gaussian peaks (width 1) at -2 and +2
f = fold_interval(line, n=30, lo=-2, hi=1)
print("threshold             :", mp.nstr(threshold(line, conservative()), 10))
print("hysteresis from chiN =", mp.nstr(f["chiN_lo"], 10), "to", mp.nstr(f["chiN_hi"], 10))
print("R jumps from 0 to      :", mp.nstr(f["R_jump"], 10))
print("R where the high branch ends:", mp.nstr(f["R_high_at_lo"], 10))
try:
    extract_beta(line, conservative(), [-3, -4, -5])
except ValueError as err:
    print("refused:", str(err)[:72], "...")
```

```
threshold             : 5.89561378
hysteresis from chiN = 3.493404803 to 5.89561378
R jumps from 0 to      : 0.8480505911
R where the high branch ends: 0.3342759883
refused: eps = -0.0016464 <= 0 at Omega = 10^-3: the branch is at or below the th ...
```

With the peaks 4 widths apart `c < 0` (example 3). The branch leaves
the threshold backwards, then turns round at `chiN = 3.4934`. In the
usual reading of such a fold (the package computes where the
solutions are, not whether they are stable), raising the coupling
keeps the unsynchronized state up to the threshold, where `R` jumps to
0.848; lowering it again keeps the synchronized state down to
`chiN = 3.4934` (where `R = 0.334`) before it collapses. `fold_interval` samples the branch on `n` points
between `Omega = 10^lo` and `10^hi` and then finds the turning points
and the jump exactly. The tests check every number it returns against
an independent closed form (the Voigt profile). There is no exponent
on such a branch, and `extract_beta` says so instead of returning
one.

### 9. The quantum spin model against exact diagonalisation

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

### 10. Are the data consistent with the power law?

```python
import numpy as np
from lockkernel import fit_branch

# Conservative spins on a Lorentzian line of FWHM 1: chi_c = 1/2 and,
# exactly, R = eps/(1+eps). 14 points from eps = 1e-4, 1 % seeded noise.
rng = np.random.default_rng(4)
for eps_max in (0.01, 0.1, 0.5):
    eps = np.geomspace(1e-4, eps_max, 14)
    r = eps / (1 + eps) * np.exp(rng.normal(0.0, 0.01, eps.size))
    fit = fit_branch(0.5 * (1 + eps), r, sigma_r=0.01 * r,
                     reference="illustrative: closed form + 1 % noise")
    print(f"eps up to {eps_max:4}: beta = {fit.beta:.4f} +- {fit.sigma_beta:.4f}, "
          f"chi2 = {fit.chi2:5.1f} for {fit.dof} dof, p = {fit.p_value:.2g}")
```

```
eps up to 0.01: beta = 1.0001 +- 0.0037, chi2 =  11.1 for 11 dof, p = 0.44
eps up to  0.1: beta = 0.9833 +- 0.0019, chi2 =  48.3 for 11 dof, p = 1.3e-06
eps up to  0.5: beta = 0.9546 +- 0.0014, chi2 = 584.2 for 11 dof, p = 3.4e-118
```

The true exponent is 1. When the points reach far above the threshold,
the branch bends away from the power law, and the fit returns a biased
exponent with a small error bar: 0.9833 +- 0.0019 is 9 error bars from
1. The error bar alone does not show this. The chi-square test does.
When `sigma_r` is given, `fit_branch` also returns `chi2` (the weighted
sum of squared misfits in `log R`), its degrees of freedom `dof`
(points minus 3) and `p_value`, the chance of a misfit at least this
large if the law and the stated errors were right. A very small
`p_value` means one of the two is wrong: drop the points farthest from
the threshold, or check `sigma_r`. Without `sigma_r` the scatter of the
points sets the error scale, so there is nothing to test, and `chi2`
and `p_value` are `None`.

### 11. Errors in the couplings, and calibration

```python
import numpy as np
from lockkernel import fit_branch, with_calibration

# Stand-in measurement: R = 0.65 eps^(2/3) above chi_c = 0.5, at 14
# recorded couplings. Each true coupling differs from the recorded one by
# a seeded error of 5e-6 (a tenth of the distance to the threshold at the
# first point), and R carries 1 % seeded noise.
rng = np.random.default_rng(2)
chi = 0.5 * (1 + np.geomspace(1e-4, 1e-1, 14))
sigma_chi = np.full(chi.size, 5e-6)
chi_true = chi + rng.normal(0.0, 1.0, chi.size) * sigma_chi
r = 0.65 * (chi_true / 0.5 - 1) ** (2 / 3) * np.exp(rng.normal(0.0, 0.01, chi.size))

ref = "illustrative: power law + seeded noise"
plain = fit_branch(chi, r, ref, sigma_r=0.01 * r)
full = fit_branch(chi, r, ref, sigma_r=0.01 * r, sigma_chi=sigma_chi)
for name, f in (("errors of R only:", plain), ("errors of R and chi:", full)):
    print(f"{name:20s} beta = {f.beta:.4f} +- {f.sigma_beta:.4f}, "
          f"chi_c = {f.chi_c:.7f} +- {f.sigma_chi_c:.1e}, p = {f.p_value:.2g}")

# The couplings share a 2 % calibration uncertainty, R a 5 % one:
cal = with_calibration(full, chi_scale_rel=0.02, r_scale_rel=0.05)
print(f"with calibration:    chi_c +- {cal.sigma_chi_c:.1e}, "
      f"A = {cal.amplitude:.3f} +- {cal.sigma_amplitude:.3f}, "
      f"beta +- {cal.sigma_beta:.4f} (unchanged)")
```

```
errors of R only:    beta = 0.6698 +- 0.0020, chi_c = 0.4999987 +- 1.2e-06, p = 0.00051
errors of R and chi: beta = 0.6658 +- 0.0026, chi_c = 0.5000034 +- 3.7e-06, p = 0.15
with calibration:    chi_c +- 1.0e-02, A = 0.649 +- 0.033, beta +- 0.0026 (unchanged)
```

Near the threshold `R` rises steeply with the coupling, so a small
error in the coupling moves `R` a lot: by `beta sigma_chi / (chi -
chi_c)` in `log R`. Pass the coupling errors as `sigma_chi` and
`fit_branch` adds this term to each point's error (to first order, and
repeats the fit until the weights settle). Here, without it, the error
bars are too small and the chi-square test says so (`p = 0.00051`).
With it the fit is consistent (`p = 0.15`) and the exponent lands within
its error bar of 2/3. `fit_branch` refuses `sigma_chi` larger than a
quarter of `chi - chi_c` at any point, where first order is not enough.

`with_calibration` adds the uncertainty of an overall scale factor: one
common to all couplings (for example the conversion of a laser power
into `chiN`), and one common to all `R` (for example a detection
efficiency). Scaling every coupling scales the fitted threshold by the
same factor and leaves `beta` and `A` exactly as they were; scaling
`R` scales only `A`. So the two calibrations add to the error bars of
`chi_c` and `A` in quadrature, and the exponent is immune to both. The
tests check both statements by refitting scaled data.

### 12. A Voigt line, the branch at a set coupling, and the power-law range

```python
import mpmath as mp
from lockkernel.lineshapes import voigt
from lockkernel.kernels import conservative, kuramoto
from lockkernel.parametric import (threshold, amplitude, branch_at_coupling,
                                   local_exponent, eps_window)

line = voigt(fwhm_gauss=1.0, fwhm_lorentz=0.3)   # Gaussian and Lorentzian FWHM
print("threshold, conservative:", mp.nstr(threshold(line, conservative()), 10))
print("threshold, Kuramoto:    ", mp.nstr(threshold(line, kuramoto()), 10))
print("amplitude A, conservative:", mp.nstr(amplitude(line), 10))

# What R does the Kuramoto model give at 1 % above its threshold?
chi = 1.01 * threshold(line, kuramoto())
chiN, R, eps, Omega = branch_at_coupling(line, kuramoto(), chi)
print(f"at chiN = {mp.nstr(chiN, 8)}: R = {mp.nstr(R, 8)}, eps = {mp.nstr(eps, 8)}")
print("local exponent there:", mp.nstr(local_exponent(line, kuramoto(), Omega)[1], 8))

# How far above the threshold is the branch a power law (slope within 1 %)?
eps_max, _ = eps_window(line, kuramoto(), rel_tol=0.01)
print("power law within 1 % up to eps =", mp.nstr(eps_max, 4))
```

```
threshold, conservative: 0.4397423632
threshold, Kuramoto:     0.8794847264
amplitude A, conservative: 1.415383889
at chiN = 0.88827957: R = 0.15528283, eps = 0.01
local exponent there: 0.49058262
power law within 1 % up to eps = 0.005279
```

Real inhomogeneous lines are often Voigt lines: a Gaussian spread of
frequencies, each with its own Lorentzian width. `voigt` takes the two
full widths at half maximum. For the conservative kernel its branch is
known in closed form (`G` is the same Voigt function at the Lorentzian
half width plus `Omega`), and the tests hold the package to it.

The theory half gives the branch as a function of the locking bandwidth
`Omega`. In the lab you set the coupling instead.
`branch_at_coupling(line, kernel, chiN)` finds the branch point at a
given coupling and returns `(chiN, R, eps, Omega)`; `branch_at_eps` does
the same for a given `eps`, which keeps full precision right at the
threshold. Both assume a smooth (continuous) onset and refuse
couplings at or below the threshold.

`local_exponent(line, kernel, Omega)` returns `eps` and the exact local
slope `d log R / d log eps` at that point, computed from `dG/dOmega`
(no finite differences). Near the onset it tends to `beta`; farther up
the branch bends. `eps_window` finds where the slope first differs from
`beta` by `rel_tol`. Points below that `eps` see a local slope within
`rel_tol` of `beta`, so a power-law fit there is biased by less than
that. Here the Kuramoto branch is within 1 % of `beta = 1/2` only up to
`eps = 0.0053`. Choose the measured range from it, then the number of
points from `plan_fit` (tutorial 2 does both).

## Tutorials

Three worked analyses live in [`examples/`](examples). Each makes a
seeded synthetic measurement with the package's exact branch, analyzes
it the way you would analyze real data, and prints the result with its
uncertainty. Each runs in under about 20 s, and a test checks that
it prints the output below.

### Tutorial 1: conservative or overdamped?

[`examples/tutorial_which_kernel.py`](examples/tutorial_which_kernel.py).
An ensemble has a measured Voigt line (Gaussian FWHM 1.00 +- 0.02,
Lorentzian FWHM 0.30 +- 0.02) and a measured onset: 12 couplings from
`1e-4` to `1e-2` above the threshold, 1 % noise on `R`, coupling errors
and a 1 % coupling-scale calibration. Is it made of conservative spins
(`beta = 1`, kernel mass `pi`) or overdamped phase oscillators
(Kuramoto, `beta = 1/2`, mass `pi/2`)? The fit gives two independent
answers: the exponent, and the kernel mass `m = 1/(p(0) chi_c)` from
the threshold and the measured line. The amplitude that the Kuramoto
model predicts (`amplitude_curvature`) is a third check.

```
fitted threshold chi_c = 0.879490 +- 0.0088 (with 1 % scale calibration)
fitted exponent  beta  = 0.4957 +- 0.0048, chi2 = 4.3 for 9 dof, p = 0.89
exponent: Kuramoto (1/2) is 0.9 sigma off, conservative (1) is 104 sigma off
kernel tail refused: beta = 0.4957 +- 0.004827 is consistent with 1/2, which identifies only the CLASS s >= 3
line center p(0) = 0.72386 +- 0.016
kernel mass m = 1/(p(0) chi_c) = 1.571 +- 0.038
mass: Kuramoto (pi/2 = 1.571) is 0.00 sigma off, conservative (pi = 3.142) is 41 sigma off
amplitude: fitted A = 1.5224 +- 0.0421, Kuramoto prediction 1.5676 (1.1 sigma)
```

### Tutorial 2: plan, measure, read off the kernel tail

[`examples/tutorial_plan_and_fit.py`](examples/tutorial_plan_and_fit.py).
Conservative spins on a Lorentzian line. `eps_window` gives the range of
`eps` where the branch is a power law within 1 %, and `points_for_fit`
the number of couplings for an error bar of 0.007 on `beta` at 2 %
noise. The exact branch at those couplings (`branch_at_coupling`), with
seeded noise, is fitted, and the kernel tail `s = 1 + 1/beta` is read
off. For contrast, the same number of points spread up to `eps = 0.5`
gives a biased exponent, and the chi-square test flags it.

```
plan: power law within 1 % up to eps = 0.0101 (closed form 0.0101)
plan: 16 points for beta +- 0.007 (planned error bar 0.00700)
fit inside the window:  beta = 1.0012 +- 0.0070, chi2 = 5.9 for 13 dof, p = 0.95
kernel tail s = 2.00 +- 0.01 (conservative spins: s = 2)
fit up to eps = 0.5:    beta = 0.9554 +- 0.0027, chi2 = 180.7 for 13 dof, p = 1.2e-31
```

### Tutorial 3: the whole branch, a model test and the coupling calibration

[`examples/tutorial_whole_branch.py`](examples/tutorial_whole_branch.py)
(new in 1.5.0). Overdamped oscillators (Kuramoto kernel) with a
Gaussian spread of natural frequencies. The coupling is recorded as a
drive voltage, and the factor `k` that turns volts into `chiN` is not
known. The synthetic data are 12 points from `eps = 0.002` to
`eps = 0.5`, with 1 % noise on `R`. The power law of `fit_branch`
holds only near the threshold, so over this range it gives a biased
exponent, and its chi-square test rejects it. `branch_table` and
`fit_exact_branch` fit the exact branch of a model to all the points,
with `k` free. Of two candidate lines, the Gaussian one fits and the
Lorentzian one is rejected (`p = 0` means a probability below the
smallest floating-point number). The accepted model then gives `k`, the
threshold in volts and, with the `R` scale free as well, a check of
the `R` calibration.

```
power law, all 12 points: beta = 0.4115 +- 0.0024, chi2 = 276.4 for 9 dof, p = 2.6e-54
exact branch, Kuramoto kernel, Gaussian line:   chi2 = 9.9 for 11 dof, p = 0.54
exact branch, Kuramoto kernel, Lorentzian line: chi2 = 16065.0 for 11 dof, p = 0
coupling scale k = 0.037101 +- 0.000001 per volt (true 0.0371)
threshold chi_c = 18.2653 +- 0.0005 V (true 18.2658 V)
with the R scale free: q = 1.0005 +- 0.0035, threshold 18.2654 +- 0.0006 V
```

The shape of `R(eps)` depends only on the shape of the line, not on its
width, so the model test does not need the width. The width enters
only `k` (through the model's threshold `1/(p(0) m)`); here it is taken
as measured separately. `fit_exact_branch` does not take coupling
errors (`sigma_chi`); see [Limits](#limits).

## What is in the package

The top level imports the `measured` names below and the submodules
`lineshapes`, `kernels`, `parametric` and `measured`. Import
`lockkernel.cumulant`, `lockkernel.ensemble` and `lockkernel.exact`
directly.

**Line shapes and kernels** (`lockkernel.lineshapes`, `lockkernel.kernels`)

- `LineShape` -- a symmetric density with its value at the centre
  (`p0()`), and, where available, a floating-point `cdf` and `ppf`.
  Makers: `lorentzian`, `gaussian`, `student_t`, `box`,
  `bimodal_gaussian` and (new in 1.4.0) `voigt(fwhm_gauss,
  fwhm_lorentz)`; `LINESHAPES` maps names to them.
- `Kernel` -- an even kernel with `W(0) = 1`, its tail exponent, its
  support, `mass()` and `second_moment()` (the integral of
  `u^2 W(u)`, finite only for compact support, fast decay or `s > 3`).
  New in 1.4.0: `breaks`, points where `W` has a kink, which the
  integrals use as break points (empty for the shipped kernels).
  Makers: `conservative`, `kuramoto`,
  `power_tail(s)`, `gaussian_kernel`; `KERNELS` maps names to them.
- `heterogeneous(base, degree_exponent, eta=1, k_min=1)` -- the kernel
  averaged over a power-law spread of coupling strengths (example 6).
  Since 1.4.0 any base kernel is reduced to one average over the base
  kernel, `a X^-a integral_0^X x^(a-1) W(x) dx` with `a = (gamma-2)/eta`
  and `X = |v|/k_min^eta`, which stays accurate far out in the tail.
- `predicted_beta(s)` -- the exponent the rule gives for tail `s`
  (`None` meaning no algebraic tail), for a line with a rounded top.

**The exact solution** (`lockkernel.parametric`, mpmath precision)

- `G_of_Omega`, `H_of_Omega` -- the two integrals the solution is built
  on: `G(Omega) = integral p(Omega u) W(u) du` and
  `H = Omega G`, so that `chiN = 1/G` and `R = H`.
- `threshold`, `branch_point`, `sweep`, `extract_beta` -- the
  threshold, one point `(chiN, R, eps)` of the branch, the branch at
  `Omega = 10^e` for a list of `e`, and the local exponents along it.
  `eps` is computed from `G(0) - G(Omega)` directly, so it keeps the
  working precision however small it is.
- `c_coefficient`, `amplitude` -- for the conservative kernel, the
  number `c` whose sign sets the order of the onset, and the amplitude
  `A = pi p(0)^2 / c`.
- `tail_integral`, `amplitude_general` -- the amplitude for a kernel
  with tail exponent `1 < s < 3` (since 1.5.0 also on a line with
  compact support, such as `box`).
- `amplitude_curvature` -- the amplitude when `beta = 1/2` (no tail,
  or a tail `s > 3`, on a line with a rounded top; example 7).
- `fold_interval` -- samples the branch and, if it folds back, returns
  the coupling range of the hysteresis, the turning points and the
  jump in `R` (`None` if the branch does not fold; example 8).
- New in 1.4.0 (example 12): `branch_at_coupling(line, kernel, chiN)`
  and `branch_at_eps(line, kernel, eps)` -- the branch point
  `(chiN, R, eps, Omega)` at a given coupling or reduced coupling;
  `local_exponent(line, kernel, Omega)` -- `eps` and the exact local
  slope `d log R / d log eps`; `eps_window(line, kernel, rel_tol=0.01)`
  -- the largest `eps` (and its `Omega`) up to which that slope stays
  within `rel_tol` of `beta`.
- New in 1.5.0 (tutorial 3): `branch_table(line, kernel, eps_min,
  eps_max, rel_tol=1e-5)` -- the exact branch `R(eps)` from `eps_min` to
  `eps_max` as a checked table (a `BranchTable`), evaluated in ordinary
  floating point with `R(eps)` and `log_R_and_slope(eps)`. Between nodes
  it is a cubic Hermite interpolant of `log R` in `log eps`, built from
  the exact values and exact local exponents. Each interval is checked
  against an exact branch point at its middle and halved until they
  differ by at most `rel_tol` in `log R`; `max_error` reports the
  largest difference found.

**Fitting measured data** (`lockkernel.measured`, also at the top level)

- `fit_branch(chi, r, reference, sigma_r=None, min_decade=1.0,
  sigma_chi=None)` -- returns a `BranchFit` with `chi_c`, `beta`,
  `amplitude`, their errors, `n_points`, `eps_range`,
  `residual_rms_log` (the root-mean-square misfit in `log R`) and
  `reference`. Since 1.3.0 it also has `dof`, `chi2` and `p_value`
  (the last two `None` without `sigma_r`; example 10),
  `sigma_chi_used` and `calibration_rel`. `sigma_chi` gives the errors
  of the couplings (example 11).
- `with_calibration(fit, chi_scale_rel=0, r_scale_rel=0)` -- the same
  fit with the relative uncertainties of an overall coupling scale and
  an overall `R` scale added to the error bars of `chi_c` and `A`
  (example 11).
- New in 1.5.0 (tutorial 3): `fit_exact_branch(chi, r, reference,
  table, sigma_r=None, fit_r_scale=False)` -- fits the exact branch of a
  model (a `BranchTable`) to measured points over their whole range.
  The free parameter is the coupling scale `k` (`chiN = k chi`), and
  with `fit_r_scale=True` also an `R` scale `q`. It returns an
  `ExactBranchFit` with `chi_scale`, `r_scale`, the threshold `chi_c` in
  the recorded units, their errors, `chi2`, `dof` and `p_value` (the
  last two `None` without `sigma_r`), `eps_range`, `residual_rms_log`
  and `table_error`.
- `kernel_tail_from_beta(beta, sigma_beta=0, n_sigma=2)` -- the tail
  exponent `s` and its error.
- `beta_relative_sigma(n_points, decades, sigma_log)`,
  `points_for_beta(target_sigma_beta, decades, sigma_log)` -- the
  error bar of `beta` for a planned measurement with the threshold
  known, and the number of points for a target error bar.
- `plan_fit(n_points, eps_min, eps_max, beta, sigma_log,
  fit_threshold=True)`, `points_for_fit(target_sigma_beta, eps_min,
  eps_max, beta, sigma_log)` -- the same for the fit `fit_branch`
  actually does, with the threshold fitted: `plan_fit` returns a
  `FitPlan` with `sigma_beta`, `sigma_chi_c_rel` and
  `sigma_amplitude_rel` (example 5).

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
  `full_exact` (full quantum problem, up to about twelve emitters; since
  1.5.0 it builds sparse operators, and 10 emitters take about 10 s
  instead of about 130 s) and
  `class_exact` (emitters grouped by detuning, a few tens) -- exact
  references. `class_exact` is tested against `full_exact` since 1.3.0.

Each function's docstring (`help(lockkernel.fit_branch)`, for example)
gives its inputs and conventions.

## When it refuses, and why

`lockkernel` raises an error instead of guessing when:

- measured data come without a `reference` of at least 8 characters
  saying where they come from;
- fewer than 6 points are given, or `chi` and `r` differ in length
  (three parameters with error bars need more);
- the couplings are not finite, positive and strictly increasing, or
  an order parameter is not finite and positive (points below the
  threshold, `R = 0`, carry no exponent information; drop them);
- `sigma_r` is not finite and positive, or not on the same grid as `r`;
- the fit does not converge, or its covariance is singular (the data
  cannot pin down the three parameters separately);
- the fitted exponent runs into the limits of the search, 0.05 or 20
  (new in 1.3.0; the data are not a power law of this kind, for
  example because `R` hardly changes, as at a jump);
- `sigma_chi` is given without `sigma_r`, is negative, or is larger
  than a quarter of `chi - chi_c` at some point (new in 1.3.0);
- `with_calibration` gets a negative uncertainty, or a fit that
  already includes one;
- the fitted threshold is indistinguishable from the smallest
  coupling: the data do not reach the onset;
- the fitted points span less than `min_decade` decades of `eps`
  (default one): exponent and amplitude cannot be told apart on so
  short a range;
- a fitted `beta` is consistent with 1/2 (only the class `s >= 3` is
  identified), clearly below 1/2 (no kernel gives that; the fit has
  probably left the near-threshold range), or not positive;
- a planned measurement has fewer than 3 points (6 with the threshold
  fitted, as `fit_branch` needs), a number of points that is not a
  whole number, a non-positive or reversed range, a non-positive
  scatter or `beta`, a non-positive target, or would need more than
  10^7 points;
- `extract_beta` meets a point with `eps <= 0`: the branch is at or
  below the threshold there, because it bends back (a first-order
  onset, example 8) or is flat (`R` jumps at the threshold), and there
  is no exponent to measure;
- `amplitude_curvature` is asked for a line with `p''(0) >= 0` (a flat
  top or a dip at the centre) or a kernel with tail `s <= 3`, and
  `second_moment` for a kernel with tail `s <= 3` (it diverges);
- `fold_interval` finds a fold on a line with compact support (`box`),
  which it cannot refine, or cannot bracket a turning point on its
  grid;
- `power_tail(s)` or `predicted_beta(s)` is asked for `s <= 1` (the
  kernel mass would be infinite), or `heterogeneous` for `gamma <= 2`,
  a non-positive `eta` or `k_min`, or a resulting tail `s <= 1`;
- a line shape is given a width that is not finite and positive;
- `tail_integral` is asked for `s` outside `1 < s < 3`, where it
  diverges;
- (new in 1.4.0) `G_of_Omega`, `H_of_Omega`, `branch_point` or
  `local_exponent` get a negative or non-finite `Omega` (1.3.0 returned
  twice the right `G` for a negative one); a `LineShape` is built with a
  scale that is not finite and positive; `voigt` gets a width that is
  not finite and positive;
- (new in 1.4.0) `branch_at_coupling` gets a coupling at or below the
  threshold, or `branch_at_eps` an `eps <= 0`; either meets a branch
  that bends back or is flat (a first-order or jump onset; see
  `fold_interval`) or does not converge;
- (new in 1.4.0) `local_exponent` or `eps_window` gets a line with
  compact support (it is not differentiable), or a point where the
  branch does not rise; `eps_window` finds the local exponent outside
  the band already at the start of its search, or gets `rel_tol`
  outside `(0, 1)`;
- (new in 1.5.0) `branch_table` gets a line with compact support, a
  range that is not `0 < eps_min < eps_max`, a `rel_tol` outside
  `(0, 1e-3]` or a `step` outside `(0, 2]` decades, meets a branch that
  folds or is flat, or cannot reach `rel_tol` after halving an interval
  30 times; a `BranchTable` is asked for `R` outside its range (it does
  not extrapolate);
- (new in 1.5.0) `fit_exact_branch` gets fewer than 5 points, couplings
  or order parameters that fail the checks of `fit_branch`, couplings
  that span a larger ratio than the table can hold, a best fit at the
  edge of the table's range (build the table wider), a fit that does not
  converge, or a table whose `max_error` is more than 1 % of the
  smallest relative error of `R` (of the rms misfit without `sigma_r`),
  where the table could bias the fit;
- (new in 1.5.0) `full_exact` gets detunings that are not a non-empty
  one-dimensional list of finite numbers, or a non-finite `chiN` or
  time;
- `amplitude_general` is asked for a kernel without a tail in
  `1 < s < 3`, or (new in 1.3.0) its default tail amplitude `C`, read
  off the kernel at `|u| = 10^6`, differs by more than a relative 1e-6
  from the value at `10^9` (pass `C` yourself then);
- `class_exact` gets class populations that are not whole numbers;
- a line shape without a floating-point distribution
  (`bimodal_gaussian`, `voigt`) is asked for `cdf`, `ppf` or a class
  table;
- the solver of the differential equations fails in `evolve` or
  `evolve_meanfield`.

## How the results are checked

194 automated tests run on every push and pull request, on Python 3.9
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

**New in 1.2.0** (15 digits unless stated; the references are closed
forms evaluated separately from the package's integrals)

- `eps`, `chiN` and `R` on the Lorentzian line match `eps = Omega/a`,
  `chiN = a + Omega`, `R = Omega/(a + Omega)` to a relative 1e-13 for
  `Omega` from 1e-6 to 1e-14, and to 1e-28 at 30 digits down to
  `Omega` = 1e-20.
- Box line, kernels `1/(1+|u|^s)` with `s` = 1.5, 2.5, 4 and 6: `G`
  and `eps` match the closed form `G = X 2F1(1, 1/s; 1+1/s; -X^s)`,
  `X = 1/Omega`, to a relative 1e-13 at `Omega` = 1e-1, 1e-3 and 1e-5
  (down to `eps` of about 1e-26). On this flat-topped line
  `extract_beta` gives `1/(s-1)` = 1/3 and 1/5 for `s` = 4 and 6, within
  1e-6.
- Two-peaked line: `c_coefficient` matches `(1 - 2x D(x))/pi` to 1e-13
  for separations 1, 2 and 4; the sign of `c` flips across the
  separation 2.61386 (the maximum of Dawson's function, located by root
  finding); `G` matches the Voigt closed form to a relative 1e-13.
- `fold_interval`: every number it returns (couplings, order
  parameters, `Omega` of the turning points and of the jump) matches
  the Voigt closed form, with turning points found by root finding on
  its derivative, to a relative 1e-12, for a branch that leaves the
  threshold backwards (two peaks 4 widths apart) and for an S-shaped
  branch (a three-peak line); it returns `None` for a monotonic branch.
- `extract_beta` refuses the backward branch and the flat one (box
  line, Kuramoto kernel, where `eps` is exactly 0).
- `second_moment` matches `pi/8` (Kuramoto), `sqrt(pi)/2` (Gaussian
  kernel) and `2 (pi/s)/sin(3 pi/s)` (`s` = 4, 6) to 1e-12.
  `amplitude_curvature` is 1 for the Kuramoto kernel on a Lorentzian
  line and `sqrt(pi)`, `sqrt(2)` on Gaussian lines of two widths, to
  1e-12; at 20 digits it matches `R/sqrt(eps)` at `Omega = 1e-6` to a
  relative 1e-10 for three kernel-line pairs, and to 2e-6 for `s = 4`.

**New in 1.3.0** (15 digits unless stated; every reference is a closed
form or a calculation done another way)

- Kernel masses for slow tails, `power_tail(s)` with `s` = 1.05, 1.1,
  1.25 and 1.5, against `2 (pi/s)/sin(pi/s)` to a relative 1e-14, and
  1e-28 at 30 digits. The averaged Kuramoto kernel of example 6 with
  `gamma` = 3.4, 3.8 and 4.4 against its mass
  `(pi/2)(gamma-2)/(gamma-3)` to 1e-13. `second_moment` for `s` = 3.1,
  3.25 and 3.5 against `2 (pi/s)/sin(3 pi/s)` to 1e-13.
- For `s = 1.1` on a Lorentzian line the threshold equals
  `s sin(pi/s)/4` to 1e-13, and `chiN/chiN_c - 1` on the branch equals
  `eps` from its separate integral to 1e-10.
- `tail_integral` against the closed forms on Lorentzian and Gaussian
  lines of widths 0.001, 1 and 1000, for `s` from 1.1 to 2.99, to a
  relative 1e-13 (and 1e-27 at 30 digits). `amplitude_general` against
  its closed form on the Lorentzian line for `s` = 1.1, 1.5, 2.5 and
  2.9, to 1e-12.
- `amplitude_general` refuses the conservative kernel averaged over
  coupling strengths, whose tail amplitude (a closed form, also checked
  by quadrature) is not reached at `|u| = 10^6`, and accepts the
  averaged Kuramoto kernel, whose tail is an exact power law (its
  default `C` matches the closed form `(gamma-2) B(s/2, 3/2)/2` to give
  the same amplitude to 1e-12).
- `class_exact` matches `full_exact` (8 emitters, two sets of classes)
  to 1e-10 in `R` and 1e-9 in `xi^2`, with the times in any order.
- `fit_branch`: `chi2` matches the sum recomputed from the returned
  parameters; over 300 seeded fits under the model the p-values pass a
  Kolmogorov-Smirnov test for uniformity and `chi2` averages to `dof`
  within four standard errors; on the exact conservative branch with
  1 % noise, points up to `eps = 0.01` give `p > 0.05` and `beta`
  within 3 error bars of 1, points up to `eps = 0.5` give
  `p < 1e-10` and `beta` more than 10 error bars off.
- With coupling noise of a tenth of `chi - chi_c` at the first point,
  the error bars of `chi_c`, `beta` and `A` with `sigma_chi` match the
  scatter of 300 seeded fits within 12 %, while without it `sigma_beta`
  is more than 15 % too small. `sigma_chi = 0` gives exactly the plain
  fit. A fit refuses data generated with `beta` = 0.02, 0.04 and 25.
- Refitting data with all couplings scaled by 1.07 scales `chi_c` by
  1.07 to 1e-9 and leaves `beta` and `A` unchanged to 1e-7; scaling `R`
  scales only `A`. `with_calibration` adds in quadrature to 1e-12.

**New in 1.4.0** (15 digits unless stated; every reference is a closed
form, a special function evaluated by mpmath at 40 digits or more, an
independent implementation, or the 1.3.0 code path)

- Averaged kernels (`heterogeneous`) against closed forms for the
  conservative, `1/(1+|u|^2.5)` and Gaussian base kernels: a Gauss
  hypergeometric function `2F1(1, b; 1+b; -v^s0)`, `b = (gamma-2)/(eta s0)`,
  and an incomplete gamma function, for eight pairs of `gamma` and
  `eta` and `v` from 0.3 to 1e12, to a relative 1e-13. The Kuramoto
  average with `k_min` = 2 and 0.5 against the weight integral itself
  (30-digit quadrature) to 1e-13; masses against
  `m0 k_min^eta (gamma-2)/(gamma-2-eta)` to 1e-13; with `k_min = 2`,
  `amplitude_general` now accepts the exact power-law tail and its
  default `C` equals the closed form.
- Voigt line: the density against a 120-digit evaluation out to 1e30
  line widths (1e-14), SciPy's `voigt_profile` (1e-12), the
  convolution integral of the Gaussian and the Lorentzian (1e-13), its
  normalization (1e-12) and its Lorentzian tail (1e-13), for four width
  ratios. With the conservative kernel: threshold, `chiN`, `R` and
  `eps` on the branch against the closed form
  `G = pi erfcx(y)/(sigma sqrt(2 pi))`, `y = (gamma + Omega)/(sigma sqrt 2)`,
  to 1e-12 or better; `c` and the amplitude against their closed forms
  (1e-12). With the Kuramoto kernel, `amplitude_curvature` against its
  closed form (1e-12) and against `R/sqrt(eps)` on the branch (1e-10).
- `local_exponent` against `1/(1+eps)` and `(1/2)/(1+eps)` (Lorentzian
  line, conservative and Kuramoto kernels) to 1e-13, against the Voigt
  closed form to 1e-12, and against a central difference along the
  branch at 30 digits (Gaussian line, `s = 2.5`) to 1e-12.
- `branch_at_coupling` and `branch_at_eps` against `chiN = a + Omega`
  and `R = sqrt(eps/(1+eps))` on the Lorentzian line (1e-13), against
  the Voigt closed form inverted by mpmath's root finder at 40 digits
  (1e-12), and round trips through `branch_point` (1e-12).
  `eps_window` against `rel_tol/(1 - rel_tol)` (1e-8).
- `branch_point` and `sweep` give bit-identical results to the 1.3.0
  code path (kept in the test) on five line-kernel pairs, and
  `points_for_beta` the same counts as 1.3.0's search.
- `bimodal_gaussian(-4)` gives exactly the results of
  `bimodal_gaussian(4)`; the class table with the free edge beyond
  `delta_max` holds `2 sf(free edge)` in its free classes (1e-9) and
  reproduces the locking integral within 5e-5; `physicality` and
  `valid_window` on a coherent state (covariance eigenvalue 0, Bloch
  length 1) and on a state outside the Bloch sphere.
- The two tutorials run and print what this README shows, and their
  results agree with the truth of their synthetic data.

**New in 1.5.0** (15 digits unless stated; every reference is a closed
form derived in the test, an independent SciPy fit, seeded simulation,
or the 1.4.0 code path)

- `tail_integral` on lines with compact support: the box line against
  `w^-s/(s-1)`, and a semicircle line against
  `-(2/pi) B((1-s)/2, 3/2) w^-s` (a Beta integral continued
  analytically), for widths 0.001, 1 and 1000 and `s` from 1.1 to 2.99,
  to a relative 1e-13; at `s = 2` both equal `pi c` from the direct
  integral of `c_coefficient`. `amplitude_general` on the box line with
  `s` = 1.5 and 2.5 against its closed form, and against the limit
  `R/eps^(1/(s-1))` of the closed-form branch at `Omega = 1e-40` (60
  digits), to 1e-13.
- `branch_table` on the Lorentzian line with the Kuramoto kernel against
  `R = sqrt(eps/(1+eps))` at 3001 points from `eps = 1e-4` to 1, within
  twice its `rel_tol` of 1e-7; its node values to 1e-13 and node slopes
  against `(1/2)/(1+eps)` to 1e-12. On a Gaussian line with the Kuramoto
  kernel, against `G = p(0) (pi/2) e^-z (I0(z) + I1(z))`,
  `z = Omega^2/(4 sigma^2)` (derived in the test from the integral
  representation of the modified Bessel functions, and checked there
  against 40-digit quadrature), at 25 points to 2e-7.
- `fit_exact_branch` against an independent SciPy fit of the same data
  with the closed-form branch: the coupling scale, the `R` scale and
  chi2 agree within bounds derived from the table's error, and the
  error bars to a relative 1e-4. Over 400 seeded fits, the scatter of
  `log k` and `log q` matches the median error bar within 12 %
  (observed: 3 % and 1 %), the means are unbiased and `chi2` averages
  to `dof`. Kuramoto data from `eps = 1e-3` to 0.5 reject the
  conservative kernel's branch (`p < 1e-10`) and accept their own
  (`p > 0.01`), while `fit_branch`'s power law on the same points is
  rejected and more than 10 error bars off. Without `sigma_r` the error
  bars are the residual-scaled ones.
- `points_for_fit` reaches its cap: targets that need 6291457, 8000000
  and 10^7 points get those counts, with the error bar in closed form
  (see [Corrections](#corrections-in-earlier-versions)).
- `full_exact` against the 1.4.0 code (kept in the test), for four sets
  of up to 8 emitters and times up to 11: `R` to 1e-11 and `xi^2` to a
  relative `1e-11/R^2` (`xi^2` is ill-conditioned where `R` is small).
- The three tutorials run and print what this README shows. In
  tutorial 3 the right model is kept and the wrong line rejected, and
  `k`, the threshold and the `R` scale come out within three error
  bars of their true values.

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
- (New in 1.2.0) On noiseless data reaching `eps = 1e-7`, the error
  bars `fit_branch` reports, and those `plan_fit` predicts, equal the
  ones built from mpmath's numerical derivatives at 30 digits to a
  relative 1e-6. With the threshold known, `plan_fit` equals
  `beta_relative_sigma` to 1e-12. Over 300 seeded noisy fits (1 %
  scatter), the scatter of `chi_c`, `beta` and `A` matches `plan_fit`,
  and the median reported error bar matches the scatter, within 12 %.
  `points_for_fit` is checked on both sides, and the error bar is
  checked to fall with every added point from 6 to 400. `sigma_beta`
  and `sigma_A` from `plan_fit` do not depend on `beta` and
  `sigma_chi_c` scales as `1/beta` (to 1e-12).

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

## Corrections in earlier versions

**1.5.0 fixed two errors.** Numbers are at mpmath's default 15 digits.

- `tail_integral`, and so `amplitude_general`, was wrong on a line with
  compact support. It integrates `-p'` by parts, which misses the jump
  of the `box` line at its edge: on a box of half width 1, `I_s` came
  out 0 at every `s` (the closed form is `1/(s-1)`, so 2 at `s = 1.5`),
  and `amplitude_general(box(1.0), power_tail(2.5))` raised
  `ZeroDivisionError` (the closed form is 2.08481122566036). Such a
  line is now integrated in the direct form, with the part outside the
  support in closed form. Every earlier release had this error.
- `points_for_fit` refused targets that need between 6291457 and 10^7
  points, saying that over 10^7 would be needed: its search doubled the
  number of points from 6291456 straight past 10^7. For example,
  `points_for_fit(1.3297e-5, 1e-4, 1e-2, beta=1, sigma_log=0.05,
  fit_threshold=False)` was refused and now returns 8000001 points
  (the closed form of `beta_relative_sigma` gives 8000000; at 8 million
  points `plan_fit` agrees with it to about 1e-7). Searching that far
  takes about 100 s.

All 172 earlier tests pass (one changed only in the version number it
checks), and examples 1 to 12 print the same as with 1.4.0.

**1.4.0 fixed four silent errors.** Numbers are at mpmath's default 15
digits.

- `heterogeneous` integrated the weight over the coupling strengths on a
  linear scale out to infinity, which lost accuracy far out in the
  kernel's tail. Against the closed forms, the conservative kernel
  averaged with `gamma = 2.6`, `eta = 0.5` was 2.9 % low at `v = 1e9`,
  and its mass was 18.84621 instead of 18.84956 (so its threshold 1.8e-4
  high; the mass took 107 s, now 7 s). The Gaussian kernel averaged
  with `gamma = 2.5`, `eta = 0.25` was 2.5 % low at `v = 1e4` and 9.5 %
  from `1e6` on. The Kuramoto average with `k_min = 2`, `gamma = 3.8`
  was 2.8e-4 off at `v = 1e9`, so `amplitude_general` refused it as
  "not converged", and its mass was 2.8e-6 high. All now match the
  closed forms to about 1e-15. The `k_min = 1` Kuramoto average of
  example 6 was already exact and is unchanged.
- `bimodal_gaussian` with a negative separation (the same line as the
  positive one) got a negative integration scale: for `sep = -4`,
  `G(0.1)` was 0.390375 instead of 0.195187, `c` was -0.0878293 instead
  of -0.0891192, and `branch_point` with the Kuramoto kernel raised
  `TypeError`.
- `G_of_Omega` with a negative `Omega` returned twice the value at
  `|Omega|` (3.3333 instead of 1.6667 at -0.1 on the Lorentzian line).
  A negative or non-finite `Omega` is now refused.
- `class_table` (and `build_system`, for couplings above about 125 line
  widths) counted part of the population twice when the edge of the
  free classes lay beyond `delta_max`: with the edge at 5000 and
  `delta_max` at 1000 the raw total was 1.000197, and the locking
  integral was 2.1e-4 off instead of 1.4e-5.

All 128 earlier tests pass (one changed only in the version number it
checks), and examples 1 to 11 print the same as with 1.3.0.

**1.3.0 fixed five silent errors.** Numbers are at mpmath's default 15
digits unless stated.

- `Kernel.mass()` integrated a slow tail directly out to infinity. For
  `power_tail(1.1)` the mass was 20.0312 instead of 20.2745 (1.2 % low;
  3.7e-4 at 30 digits), so the threshold on a Lorentzian line of FWHM 1
  was 0.0784174 instead of 0.0774765, and did not match the branch's own
  `eps`. `second_moment()` had the same problem just above `s = 3`
  (19.7910 instead of 20.0343 for `s = 3.1`). The tail is now mapped
  onto a finite interval, as `G(Omega)` already was, and both match
  their closed forms to the working precision.
- `tail_integral` integrated a singularity `delta^(2-s)` at the center
  directly. On a Lorentzian line of FWHM 1, `I_s` was 47.063 instead
  of 47.715 at `s = 2.9` and 86.97 instead of 98.49 at `s = 2.95`. A
  change of variables now removes the singularity.
- Through these two, `amplitude_general` was wrong at the edges of
  `1 < s < 3`, where the power `1/(s-1)` amplifies errors: on a
  Gaussian line 8.4219 instead of 9.6179 for `s = 1.1`, and 0.45574
  instead of 0.45258 for `s = 2.9`. Its default tail amplitude was also
  used without a check: for the conservative kernel averaged over
  coupling strengths (`gamma = 3.8`) it was 6.2 % low, and the
  amplitude 8.3 % high. It now refuses such a kernel unless `C` is
  given.
- `fit_branch` returned the limit of its search as the exponent when
  the best fit lay beyond it: for `R = 0.8 eps^0.04` it reported
  `beta = 0.0500 +- 0.0040`, and for `eps^25`, `beta = 20 +- 1.5`. It
  now refuses.
- `class_exact` skipped a time earlier than the one before it and
  returned the later state: for times `[1.0, 0.2]` (the system of
  example 9) both entries gave `R = 0.677579`; the right value at
  `t = 0.2` is 0.984316. It now accepts times in any order.

All earlier tests pass unchanged, and examples 1 to 9 print the same
as with 1.2.0.

**1.2.0 fixed four silent inaccuracies.** Numbers below are at
mpmath's default 15 digits unless stated.

- `branch_point` formed `eps` as `chiN/chiN_c - 1` and lost about
  `log10(1/eps)` digits: on the Lorentzian line `eps` was 2e-4
  (relative) off at `Omega = 1e-9` and wrong by a factor of about 220 at
  `1e-12`. `G_of_Omega` itself was 4e-10 off at `Omega = 1e-12`,
  because a segment spanning many decades was integrated on a linear
  scale. On the box line with `s = 4`, `extract_beta` quoted 0.3194
  instead of 1/3, and with `s = 6` it raised `ZeroDivisionError`. Both
  quantities now keep the working precision (see the new checks above).
  The earlier tests and examples ran at 20 to 30 digits with `eps` no
  smaller than about 1e-12, where the loss did not show: all of them
  still pass unchanged, and README examples 1, 2, 3, 6 and 9 print the
  same as before.
- `extract_beta` returned slopes close to 1 (1.0016, 1.00016, 1.000016
  for `Omega` = 1e-3 down to 1e-6, at 20 digits) for the backward
  branch of the two-peaked line (a first-order onset, `eps < 0`), and
  raised `ZeroDivisionError` on a flat branch. It now refuses both.
- `fit_branch` took its derivatives by finite differences, which near
  the threshold are not small steps: with points down to `eps = 1e-7`
  it reported `sigma_chi_c` about 17 % and `sigma_beta` about 2 % too
  small, and on example 4's data the fit stopped marginally short of
  the least-squares minimum. It now uses exact derivatives. In example 4 the threshold
  error bar goes from 3.5e-9 to 3.8e-9 (beta 0.67322 -> 0.67325, its
  error bar 0.00275 -> 0.00279).
- `fold_interval`, for a branch that leaves the threshold backwards,
  returned the first grid point instead of the threshold as the end of
  the low branch (5.89464 instead of 5.89561 in example 8, and so a
  jump to 0.84800 instead of 0.84805). A turning point it could not
  refine was silently replaced by a grid point; it now raises. It is
  also much faster: on the line of example 8 with `n = 60` and the
  default range it took 79 s in 1.1.1 and takes 3.5 s now (17 s at the
  default `n = 400`).

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
- The rule `beta = 1/2` for `s >= 3` (and `predicted_beta`,
  `kernel_tail_from_beta`, `amplitude_curvature`) assumes a line with a
  rounded top, `p''(0) < 0`. On a line that is flat at the centre, as
  `box` is, a tail gives `beta = 1/(s-1)` for every `s > 1`, and a
  kernel without a tail gives a jump. A measured `beta` below 1/2 is
  refused by `kernel_tail_from_beta`, although a flat-topped line can
  produce one.
- `plan_fit` gives the asymptotic error bars, exact to first order in
  the noise. They were checked against simulated fits at 1 % scatter;
  at much larger scatter the real fit can do worse.
- In the undamped spin model the truncated cumulant equations become
  unstable at long times; check `physicality` / `valid_window` before
  trusting a long run. The time-averaged kernel is an assumption,
  checked against the full dynamics (the mean-field tests above), not
  a theorem.
- `c_coefficient` and `amplitude` are good to about 20 significant
  digits at most, however high `mp.dps` is set: the integral leaves
  out the first 1e-20 line widths next to the centre, a piece of
  relative size about 1e-20.
- `fold_interval` finds a fold only if its grid shows it: structure
  below `Omega = 10^lo` or narrower than the grid spacing is missed, and
  with several folds only the first and last turning points are used.
  It needs a differentiable line. It costs `n` plus about 40
  high-precision integrals (about 17 s at the default `n = 400` on the
  two-peaked line).
- `fit_branch` propagates coupling errors `sigma_chi` to first order.
  The tests check this at a tenth of `chi - chi_c`, and hand checks
  with 1000 seeded fits held to within 6 % up to three tenths. It is
  refused above a quarter.
- `with_calibration` treats calibration as one scale factor for all
  couplings and one for all `R`. An offset (a zero error) is not a
  scale: it changes `eps` and so the exponent, and is not covered.
- The default tail amplitude of `amplitude_general` is read off the
  kernel at `|u| = 10^6`; for `power_tail(s)` it is low by a relative
  `10^(-6 s)`, which the power `1/(s-1)` multiplies (2.5e-6 in `A` for
  `s = 1.1`). Pass `C` for more digits.
- `voigt` has no floating-point distribution, so it cannot be turned
  into a class table for the dynamics modules. Each density value
  takes extra digits far out in the tail, which makes integrals over a
  Voigt line several times slower than over a Gaussian one.
- `branch_at_coupling` and `branch_at_eps` assume a continuous onset,
  where `eps` rises with `Omega`; on a folded branch they refuse when
  they meet `eps <= 0`, but where a branch folds above the threshold (an
  S shape) they return one of the solutions. Check with
  `fold_interval` first if the line has several peaks.
- `local_exponent` and `eps_window` need a differentiable line, and
  `eps_window` assumes the local exponent leaves the band once and
  for all as `eps` grows (true for the closed forms tested). It follows
  the branch in half decades of `Omega`, so a band left and re-entered
  within half a decade is not seen.
- `branch_table` checks its interpolant at one point inside each
  interval, where the error of a cubic Hermite interpolant is largest,
  and assumes the error elsewhere in the interval is not much larger
  (the tests allow twice `rel_tol` against closed forms and find about
  once). It needs a differentiable line and a continuous onset over the
  whole range. Its cost grows as `rel_tol` shrinks, roughly as
  `rel_tol^(-1/4)` nodes, each a few high-precision integrals.
- `fit_exact_branch` fits only the coupling scale (and, if asked, the
  `R` scale): the shape of the line and the kernel are fixed by the
  table, so a rejected model can be wrong in either. It does not take
  coupling errors (`sigma_chi`). Its error bars are the asymptotic ones,
  checked against seeded fits at 1 % noise. Build the table over a
  wider range of `eps` than the data cover, since a best fit at its edge
  is refused.
- Far above the line width, `G(Omega)` integrates the line's part of
  the kernel range on a linear scale across many decades and loses
  digits: on the Lorentzian line at 15 digits it is 1.5e-14 (relative)
  off for `Omega` from 100 to 1e6 line widths, 3e-13 at 1e9 and 3e-10
  at 1e12 (at 30 digits, 1e-28 at 1e9 and 2e-25 at 1e12). Near the
  threshold, where the onset is, it keeps the working precision.
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
changes only `c_coefficient`; 1.2.0 changes how `G`, `eps` and the
fold are computed, and adds `amplitude_curvature` and
`Kernel.second_moment`; 1.3.0 changes how the kernel mass, the second
moment and `tail_integral` are integrated, and how `class_exact` steps
through time; 1.4.0 changes how `heterogeneous` averages the kernel,
fixes the scale of `bimodal_gaussian` and the free classes of
`class_table`, and adds `voigt` and the branch tools; 1.5.0 changes how
`tail_integral` treats a line with compact support and how
`full_exact` builds its operators, and adds `branch_table`); the
`measured` module and the packaging are new here. Copyright as in [NOTICE](NOTICE).

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
