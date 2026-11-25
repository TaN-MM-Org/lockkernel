"""Locking kernels.

A locking kernel W(u) is the time averaged projection of an oscillator on the
mean field, as a function of the detuning measured in units of the locking
bandwidth, u = delta / Omega.  It is normalized so that a resonant oscillator
follows the field perfectly, W(0) = 1, and it is even.

Two kernels matter physically.

* Conservative spins.  An off resonant spin precesses about a tilted axis and
  keeps a nonzero time averaged projection on the mean field for every
  detuning.  The projection is

      W(u) = 1 / (1 + u^2),

  an ALGEBRAIC kernel with tail exponent s = 2.

* Overdamped phase oscillators, the Kuramoto case.  An oscillator that cannot
  be entrained drifts, and its time averaged contribution vanishes exactly.
  The kernel is COMPACTLY SUPPORTED,

      W(u) = sqrt(1 - u^2) for |u| < 1 and 0 otherwise.

The one parameter family

      W_s(u) = 1 / (1 + |u|^s),   s > 1,

interpolates between them in the sense that matters here, namely the tail, and
is what the universality line is tested against.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import mpmath as mp

__all__ = ["Kernel", "conservative", "kuramoto", "power_tail",
           "gaussian_kernel", "KERNELS"]


@dataclasses.dataclass(frozen=True)
class Kernel:
    """An even locking kernel with W(0) = 1.

    Attributes
    ----------
    name : str
    W : callable
        The kernel, on `mpmath` scalars.
    tail : float or None
        Exponent s of the algebraic tail W ~ |u|^-s, or None for compact
        support or faster than algebraic decay.
    support : float or None
        Half width of the support if compact, otherwise None.
    breaks : tuple of float
        Points u > 0, other than the support edge, where W is not smooth
        (a kink), given to the quadratures as break points (new in 1.4.0;
        empty by default, which leaves every integral as before).  The
        coupling average of a compactly supported kernel has one, see
        `heterogeneous`.
    tail_amplitude : callable or None
        For a kernel with an algebraic tail, a function of no arguments
        that returns the exact tail amplitude C in W ~ C |u|^-s, at the
        current mpmath precision (new in 1.7.0; None when it is not
        known).  `lockkernel.parametric.amplitude_general` uses it when
        no C is passed.  It is 1 for `conservative` and `power_tail`, and
        `heterogeneous` gives it in closed form or as one integral.
    """

    name: str
    W: Callable
    tail: float = None
    support: float = None
    breaks: tuple = ()
    tail_amplitude: Optional[Callable] = None

    def __call__(self, u):
        return self.W(u)

    def mass(self) -> mp.mpf:
        """m = int W(u) du, which fixes the threshold coupling.

        For an algebraic tail W ~ |u|^-s the part beyond u = 1000 is mapped
        onto (0, 1] so that the slow tail becomes a smooth integrand (see
        `_algebraic_quad`).  Before 1.3.0 it was integrated directly, which
        for `power_tail(1.1)` left the mass 1.2 % too small at 15 digits.
        """
        if self.support is not None:
            return 2 * mp.quad(self.W, _with_breaks([0, self.support],
                                                    self.breaks))
        if self.tail is not None:
            return 2 * _algebraic_quad(self.W, self.tail, breaks=self.breaks)
        # faster than algebraic decay: the decades resolved explicitly
        return 2 * mp.quad(self.W, _with_breaks([0, 1, 10, 100, 1000, mp.inf],
                                                self.breaks))

    def second_moment(self) -> mp.mpf:
        """M_2 = int u^2 W(u) du.

        It sets the curvature correction G(0) - G(Omega) ~ Omega^2 and so
        the amplitude of the beta = 1/2 class (see
        `lockkernel.parametric.amplitude_curvature`).  It is finite only for
        compact support, faster than algebraic decay, or a tail s > 3;
        a kernel with tail s <= 3 is refused.  For a tail the integrand
        u^2 W(u) falls like |u|^-(s-2), which is slow just above s = 3; it
        is integrated as in `mass` (before 1.3.0, `power_tail(3.1)` came out
        1.2 % too small at 15 digits).
        """
        f = lambda u: u ** 2 * self.W(u)
        if self.support is not None:
            return 2 * mp.quad(f, _with_breaks([0, self.support], self.breaks))
        if self.tail is not None and self.tail <= 3:
            raise ValueError(
                f"the second moment of a kernel with tail s = {self.tail:g} "
                "diverges (it converges only for s > 3)")
        if self.tail is not None:
            return 2 * _algebraic_quad(f, self.tail - 2, breaks=self.breaks)
        return 2 * mp.quad(f, _with_breaks([0, 1, 10, 100, 1000, mp.inf],
                                           self.breaks))


def _with_breaks(pts, breaks):
    """The break points `pts` (increasing, from 0) with `breaks` added
    where they fall strictly inside; `pts` itself when there are none."""
    extra = [mp.mpf(b) for b in breaks if pts[0] < b < pts[-1]]
    if not extra:
        return pts
    return sorted(set(mp.mpf(p) for p in pts) | set(extra))


def _algebraic_quad(f, s, X=1000, breaks=()):
    """int_0^inf f(u) du for an integrand with an algebraic tail f ~ u^-s.

    Up to u = 1 the integral is taken directly, from 1 to X in v = log(u)
    (break points at 10 and 100), and the tail [X, inf) is mapped onto
    (0, 1] by u = X t^(-k) with k = 1/(s-1), under which a tail u^-s
    becomes a constant at t -> 0.  This is the substitution
    `lockkernel.parametric` uses for G(Omega).  mpmath's own treatment of
    [X, inf) converges badly when s is close to 1.  `breaks` are extra
    break points (kinks of f); the tail starts beyond the largest one.
    """
    s = mp.mpf(s)
    X = mp.mpf(X)
    if breaks:
        X = max(X, 10 * max(mp.mpf(b) for b in breaks))
    k = 1 / (s - 1)
    v0 = mp.quad(f, _with_breaks([0, 1], breaks))
    logs = [0, mp.log(10), mp.log(100), mp.log(X)]
    logs = _with_breaks(logs, [mp.log(b) for b in breaks if b > 1])
    v1 = mp.quad(lambda w: f(mp.exp(w)) * mp.exp(w), logs)
    v2 = mp.quad(lambda t: f(X * t ** -k) * k * X * t ** (-k - 1), [0, 1])
    return v0 + v1 + v2


def conservative() -> Kernel:
    """The conservative spin kernel, W = 1/(1+u^2), mass pi."""
    return Kernel("conservative", lambda u: 1 / (1 + mp.mpf(u) ** 2), tail=2.0,
                  tail_amplitude=lambda: mp.mpf(1))


def kuramoto() -> Kernel:
    """The Kuramoto kernel, W = sqrt(1-u^2) on |u| < 1, mass pi/2."""

    def W(u):
        u = mp.mpf(u)
        return mp.sqrt(1 - u ** 2) if abs(u) < 1 else mp.mpf(0)

    return Kernel("kuramoto", W, tail=None, support=1.0)


def power_tail(s: float) -> Kernel:
    """The test family W_s(u) = 1/(1+|u|^s).  Requires s > 1 for a finite mass."""
    s = mp.mpf(s)
    if not mp.isfinite(s):
        raise ValueError(f"power_tail needs a finite s, got {s}")
    if s <= 1:
        raise ValueError("power_tail needs s > 1 for the kernel mass to converge")

    def W(u):
        return 1 / (1 + abs(mp.mpf(u)) ** s)

    return Kernel(f"power{float(s):g}", W, tail=float(s),
                  tail_amplitude=lambda: mp.mpf(1))


def gaussian_kernel() -> Kernel:
    """A Gaussian kernel, faster than any algebraic decay.  Control case."""
    return Kernel("gaussian", lambda u: mp.e ** (-mp.mpf(u) ** 2), tail=None)


KERNELS = {
    "conservative": conservative,
    "kuramoto": kuramoto,
    "power_tail": power_tail,
    "gaussian": gaussian_kernel,
}


def predicted_beta(s: Optional[float], center_order: float = 2.0) -> float:
    """The order parameter exponent predicted for a kernel of tail exponent s.

    beta = 1/(s-1) for 1 < s < 3, and beta = 1/2 for s >= 3 or for a kernel
    with compact support or faster than algebraic decay (s is None).  The
    crossover at s = 3 is where the second moment of the kernel stops
    converging, so that the leading correction to the self consistency changes
    from being set by the tail to being set by the curvature of the line.

    The value 1/2 assumes the line has a rounded maximum at its center,
    p''(0) < 0 (every shipped line except `box`, and `bimodal_gaussian`
    with its peaks two or more widths apart, sep >= 2 width, which has a
    dip or, at sep = 2 width, a flat top).

    center_order (new in 1.7.0) covers other line tops.  It is the power n
    in p(0) - p(delta) ~ c |delta|^n at the center (c > 0): 2 for a rounded
    top (the default, and the rule above), 4 for `bimodal_gaussian` with
    its peaks exactly two widths apart (p(0) - p(delta) = p(0) delta^4 /
    (12 width^4) to leading order), and math.inf for a line that is flat
    on an interval around its center, such as `box`.  The same argument
    as for n = 2 gives

        beta = 1/(s-1)  for 1 < s < n+1,     beta = 1/n  for s >= n+1,

    because G(0) - G(Omega) = int [p(0) - p(Omega u)] W(u) du is led by
    the kernel's tail, C I_s Omega^(s-1), when the integral I_s of
    `lockkernel.parametric.tail_integral` converges at the center
    (s < n+1), and otherwise by the center of the line, c M_n Omega^n with
    M_n = int |u|^n W(u) du.  On a flat top a tail gives beta = 1/(s-1)
    for every s > 1, and a kernel without a tail gives no power law at all
    (the order parameter jumps at the threshold), which is refused.  The
    tests check n = 1, 4 and inf against the exact branch.

    Refuses s <= 1 (and a non-finite s other than +inf): no kernel with a
    finite mass has such a tail.  Refuses a center_order that is not
    positive (or is NaN).
    """
    n = _center_order(center_order)
    if s is not None:
        s = float(s)
        if s != s or s <= 1.0:
            raise ValueError(f"tail exponent s = {s:g}: a kernel with a "
                             "finite mass needs s > 1 (or None for no "
                             "algebraic tail)")
    if s is None or s == math.inf:
        if n == math.inf:
            raise ValueError(
                "on a line that is flat around its center a kernel without "
                "an algebraic tail gives no power law: the order parameter "
                "jumps at the threshold")
        return 1.0 / n
    if s >= n + 1:
        return 1.0 / n
    return 1.0 / (s - 1.0)


def _center_order(center_order) -> float:
    """center_order as a float > 0 (math.inf allowed); refuses the rest."""
    try:
        n = float(center_order)
    except (TypeError, ValueError):
        raise ValueError(f"center_order must be a number > 0 (or math.inf "
                         f"for a flat top), got {center_order!r}")
    if not n > 0:
        raise ValueError(f"center_order must be > 0 (or math.inf for a flat "
                         f"top), got {center_order!r}")
    return n


def heterogeneous(base: Kernel, degree_exponent: float, eta: float = 1.0,
                  k_min: float = 1.0) -> Kernel:
    """The kernel produced by a broad distribution of coupling strengths.

    When the oscillators do not all couple with the same strength, the kernel
    that appears in the self consistency is not the single oscillator response
    but an average of it.  Let oscillator i carry weight k_i drawn from
    P(k) = (gamma-1) k_min^(gamma-1) k^-gamma above k_min, let the field it
    feels be proportional to k_i^eta, and let its contribution to the
    collective field carry the weight k_i.  Then

        Wt(v) = (1 / <k>) int P(k) k W(v / k^eta) dk,

    which is again a function of v alone, so the parametric solution applies
    unchanged with this kernel in place of W.

    The tail of the average is not the tail of W.  A weight far above the
    typical one gives that oscillator a locking bandwidth far above the
    typical one, so it stays locked at detunings where a typical oscillator
    has long since been released, and the heavy tail of P is inherited by the
    kernel.  For a base kernel of tail exponent s0, including s0 = infinity for
    compact support,

        s = min(s0, (gamma - 2) / eta).

    The two cases that matter are the overdamped kernel, where s0 is infinite
    so that the coupling distribution alone sets the tail, and the
    conservative kernel, where s0 = 2 caps it.  `eta` covers couplings that
    themselves depend on the weight, J k^(eta-1), for which the field on an
    oscillator scales as k^eta.

    THE REDUCTION (since 1.4.0 for every base kernel).  Substituting
    x = v / k^eta in the weight integral and dividing by its value at v = 0
    gives, with a = (gamma - 2)/eta and X = |v| / k_min^eta,

        Wt(v) = a X^-a int_0^X x^(a-1) W(x) dx,

    a weighted average of the base kernel over [0, X].  So k_min only
    rescales the detuning, and the tail is visible directly: the integral
    converges as X grows when a < s0 (tail X^-a), and grows like
    X^(a-s0) when a > s0 (tail X^-s0).  For the Kuramoto kernel the
    integral is a Beta function or a Gauss hypergeometric function in
    closed form (below).  For any other base it is one quadrature,
    taken on [0, 1] directly and above x = 1 in log(x), scaled by
    X^tail so that the integrand stays of order one however far out the
    kernel is evaluated.  Before 1.4.0 the k integral was taken on a
    linear scale out to infinity, which lost accuracy for large |v|:
    against the closed forms (tests), `heterogeneous(conservative(),
    2.6, eta=0.5)` was 2.9 % low at v = 1e9 and its mass 1.8e-4 low;
    `heterogeneous(gaussian_kernel(), 2.5, eta=0.25)` was 2.5 % low at
    v = 1e4; and `heterogeneous(kuramoto(), 3.8, k_min=2)` was 2.8e-4
    off at v = 1e9, so `amplitude_general` refused it as unconverged
    although its tail is an exact power law.

    The mass is m = m0 k_min^eta (gamma-2)/(gamma-2-eta), with m0 the
    mass of the base kernel (integrate the weight integral over v first);
    the tests hold `Kernel.mass()` of the averaged kernel to it.
    """
    g = mp.mpf(degree_exponent)
    eta = mp.mpf(eta)
    kmin = mp.mpf(k_min)
    if not (mp.isfinite(g) and mp.isfinite(eta) and mp.isfinite(kmin)):
        raise ValueError("degree_exponent, eta and k_min must be finite")
    if not (eta > 0 and kmin > 0):
        raise ValueError(f"eta and k_min must be positive (got eta = {eta}, "
                         f"k_min = {kmin})")
    if g <= 2:
        raise ValueError("the mean weight diverges unless gamma > 2")
    s0 = mp.mpf(base.tail) if base.tail is not None else mp.inf
    tail = min(s0, (g - 2) / eta)
    if tail <= 1:
        raise ValueError("the kernel mass diverges unless min(s0, (gamma-2)/eta) > 1")

    if base.name == "kuramoto":
        # W(x) = sqrt(1-x^2) on |x| < 1.  Only weights with k^eta > v are
        # locked at detuning v, and substituting w = k^(2 eta) reduces the
        # weight integral to a Beta function above v = 1 and to a Gauss
        # hypergeometric below it, both exact:
        #   a = (2-gamma)/(2 eta) < 0,
        #   v >= 1:  Wt = (gamma-1) v^(2a) B(-a, 3/2) / (2 eta),
        #   v <  1:  Wt = (gamma-1) 2F1(-1/2, -a; 1-a; v^2) / (2 eta (-a)).
        # The first line shows directly that the tail is an exact power law
        # with exponent (gamma-2)/eta.
        a = (2 - g) / (2 * eta)

        # (These are for k_min = 1; any other k_min rescales v, see above.)
        def Wt(v):
            v = abs(mp.mpf(v))
            if v >= 1:
                return (g - 1) * v ** (2 * a) * mp.beta(-a, mp.mpf(3) / 2) / (2 * eta)
            return (g - 1) * mp.hyp2f1(-mp.mpf(1) / 2, -a, 1 - a, v ** 2) \
                / (2 * eta * (-a))

        W0 = Wt(0)
        if kmin == 1:
            W = lambda u: Wt(u) / W0
        else:
            W = lambda u: Wt(mp.mpf(u) / kmin ** eta) / W0
        # above v = 1 the average is exactly C v^-s with, from the Beta
        # function above, C = (s/2) B(s/2, 3/2), s = (gamma-2)/eta; k_min
        # rescales v, so C gains the factor k_min^(eta s)
        sk = (g - 2) / eta
        C = lambda: sk / 2 * mp.beta(sk / 2, mp.mpf(3) / 2) * kmin ** (eta * sk)
    else:
        W = _coupling_average(base, (g - 2) / eta, tail, kmin ** eta)
        C = _average_tail_amplitude(base, (g - 2) / eta, kmin ** eta)
    # a compactly supported base leaves a kink where its support edge
    # meets the smallest weight, at v = support k_min^eta
    breaks = ()
    if base.support is not None:
        kink = mp.mpf(base.support) * kmin ** eta
        if kink != 1:           # u = 1 is a break point everywhere already
            breaks = (float(kink),)
    return Kernel(f"het_{base.name}_g{float(g):g}_e{float(eta):g}",
                  W, tail=float(tail), breaks=breaks, tail_amplitude=C)


def _average_tail_amplitude(base: Kernel, a, v_unit):
    """The tail amplitude C of the coupling average of `base` (new in 1.7.0).

    With Wt(v) = a X^-a int_0^X x^(a-1) W(x) dx and X = |v| / v_unit (see
    `heterogeneous`):

    * a < s0 (the coupling spread sets the tail, s = a): the integral
      converges as X grows, so Wt ~ C |v|^-a with
      C = a v_unit^a int_0^inf x^(a-1) W(x) dx;
    * a > s0 (the base kernel's tail W ~ C0 x^-s0 sets it, s = s0): the
      integral grows like C0 X^(a-s0) / (a-s0), so C = a C0 v_unit^s0 /
      (a - s0), which needs the base's own C0;
    * a = s0: Wt ~ a C0 log(X) X^-a, not a pure power, so there is no C.

    Returns a function of no arguments (evaluated at the precision in force
    when it is called), or None when C is not known.  The approach to the
    tail is slow when a and s0 are close (the relative correction is of
    order X^-|a - s0|), which is why the value read off the kernel at
    |v| = 10^6 is not good enough there.
    """
    a = mp.mpf(a)
    v_unit = mp.mpf(v_unit)
    s0 = mp.mpf(base.tail) if base.tail is not None else mp.inf
    if a < s0:
        def C():
            f = lambda x: x ** (a - 1) * base.W(x)
            if base.support is not None:
                J = mp.quad(f, _with_breaks([0, base.support], base.breaks))
            elif base.tail is not None:
                # f ~ C0 x^-(s0 - a + 1), an algebraic tail with exponent > 1
                J = _algebraic_quad(f, s0 - a + 1, breaks=base.breaks)
            else:
                J = mp.quad(f, _with_breaks([0, 1, 10, 100, 1000, mp.inf],
                                            base.breaks))
            return a * v_unit ** a * J
        return C
    if a > s0 and base.tail_amplitude is not None:
        return lambda: a * base.tail_amplitude() * v_unit ** s0 / (a - s0)
    return None


def _coupling_average(base: Kernel, a, tail, v_unit):
    """v -> a X^-a int_0^X x^(a-1) W(x) dx with X = |v| / v_unit.

    The weighted average of the base kernel that `heterogeneous` reduces
    the weight integral to.  For X <= 1 it is a int_0^1 y^(a-1) W(X y) dy,
    of order one.  For X > 1 the part x <= 1 is a fixed integral J0 and
    the part above is taken in w = log(x), both multiplied by X^tail (the
    size of the result) so that mpmath's quadrature, which stops on an
    absolute error estimate, keeps the relative precision.
    """
    W = base.W
    sup = mp.mpf(base.support) if base.support is not None else None
    a = mp.mpf(a)
    tail = mp.mpf(tail)
    v_unit = mp.mpf(v_unit)
    j0_cache = {}

    def J0():
        # int_0^min(1, sup) x^(a-1) W(x) dx, once per working precision
        key = mp.mp.prec
        if key not in j0_cache:
            top = mp.mpf(1) if sup is None else min(mp.mpf(1), sup)
            j0_cache[key] = mp.quad(lambda x: x ** (a - 1) * W(x), [0, top])
        return j0_cache[key]

    def Wt(v):
        X = abs(mp.mpf(v)) / v_unit
        if X == 0:
            return mp.mpf(1)
        if X <= 1:
            top = mp.mpf(1) if sup is None else min(mp.mpf(1), sup / X)
            return a * mp.quad(lambda y: y ** (a - 1) * W(X * y), [0, top])
        lX = mp.log(X)
        scaled = a * mp.exp((tail - a) * lX) * J0()
        top = X if sup is None else min(X, sup)
        if top > 1:
            lt = mp.log(top)
            pts = [mp.mpf(0)] + [p for p in (mp.log(10), mp.log(100))
                                 if p < lt] + [lt]
            scaled += a * mp.quad(
                lambda w: mp.exp(a * w + (tail - a) * lX) * W(mp.exp(w)), pts)
        return scaled * mp.exp(-tail * lX)

    return Wt
