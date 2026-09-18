"""Locking kernels.

A locking kernel W(u) is the time averaged projection of an oscillator on the
mean field, as a function of the detuning measured in units of the locking
bandwidth, u = delta / Omega.  It is normalised so that a resonant oscillator
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
from typing import Callable

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
    """

    name: str
    W: Callable
    tail: float = None
    support: float = None

    def __call__(self, u):
        return self.W(u)

    def mass(self) -> mp.mpf:
        """m = int W(u) du, which fixes the threshold coupling."""
        if self.support is not None:
            return 2 * mp.quad(self.W, [0, self.support])
        # A slowly decaying tail needs the decades resolved explicitly.
        return 2 * mp.quad(self.W, [0, 1, 10, 100, 1000, mp.inf])


def conservative() -> Kernel:
    """The conservative spin kernel, W = 1/(1+u^2), mass pi."""
    return Kernel("conservative", lambda u: 1 / (1 + mp.mpf(u) ** 2), tail=2.0)


def kuramoto() -> Kernel:
    """The Kuramoto kernel, W = sqrt(1-u^2) on |u| < 1, mass pi/2."""

    def W(u):
        u = mp.mpf(u)
        return mp.sqrt(1 - u ** 2) if abs(u) < 1 else mp.mpf(0)

    return Kernel("kuramoto", W, tail=None, support=1.0)


def power_tail(s: float) -> Kernel:
    """The test family W_s(u) = 1/(1+|u|^s).  Requires s > 1 for a finite mass."""
    s = mp.mpf(s)
    if s <= 1:
        raise ValueError("power_tail needs s > 1 for the kernel mass to converge")

    def W(u):
        return 1 / (1 + abs(mp.mpf(u)) ** s)

    return Kernel(f"power{float(s):g}", W, tail=float(s))


def gaussian_kernel() -> Kernel:
    """A Gaussian kernel, faster than any algebraic decay.  Control case."""
    return Kernel("gaussian", lambda u: mp.e ** (-mp.mpf(u) ** 2), tail=None)


KERNELS = {
    "conservative": conservative,
    "kuramoto": kuramoto,
    "power_tail": power_tail,
    "gaussian": gaussian_kernel,
}


def predicted_beta(s) -> float:
    """The order parameter exponent predicted for a kernel of tail exponent s.

    beta = 1/(s-1) for 1 < s < 3, and beta = 1/2 for s >= 3 or for a kernel
    with compact support or faster than algebraic decay (s is None).  The
    crossover at s = 3 is where the second moment of the kernel stops
    converging, so that the leading correction to the self consistency changes
    from being set by the tail to being set by the curvature of the line.
    """
    if s is None or s >= 3:
        return 0.5
    return 1.0 / (float(s) - 1.0)


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

    For the two base kernels the weight integral is reduced analytically to a
    single smooth quadrature, which is what makes the exponent reachable.
    """
    g = mp.mpf(degree_exponent)
    eta = mp.mpf(eta)
    kmin = mp.mpf(k_min)
    if g <= 2:
        raise ValueError("the mean weight diverges unless gamma > 2")
    s0 = mp.mpf(base.tail) if base.tail is not None else mp.inf
    tail = min(s0, (g - 2) / eta)
    if tail <= 1:
        raise ValueError("the kernel mass diverges unless min(s0, (gamma-2)/eta) > 1")

    if base.name == "kuramoto" and kmin == 1:
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

        def Wt(v):
            v = abs(mp.mpf(v))
            if v >= 1:
                return (g - 1) * v ** (2 * a) * mp.beta(-a, mp.mpf(3) / 2) / (2 * eta)
            return (g - 1) * mp.hyp2f1(-mp.mpf(1) / 2, -a, 1 - a, v ** 2) \
                / (2 * eta * (-a))
    elif base.support is not None:
        sup = mp.mpf(base.support)

        def Wt(v):
            v = abs(mp.mpf(v))
            if v == 0:
                return (g - 1) * kmin / (g - 2)
            edge = mp.power(v / sup, 1 / eta)
            lo = max(kmin, edge)
            f = lambda k: k ** (1 - g) * base.W(v / k ** eta)
            return (g - 1) * kmin ** (g - 1) * mp.quad(
                f, [lo, 2 * lo, 10 * lo, 100 * lo, mp.inf])
    else:
        def Wt(v):
            v = abs(mp.mpf(v))
            f = lambda k: k ** (1 - g) * base.W(v / k ** eta)
            scale = mp.power(max(v, mp.mpf("1e-40")), 1 / eta)
            pts = sorted({kmin, max(kmin, scale), max(kmin, 10 * scale),
                          max(kmin, 100 * scale)})
            return (g - 1) * kmin ** (g - 1) * mp.quad(list(pts) and f,
                                                       list(pts) + [mp.inf])

    W0 = Wt(0)
    return Kernel(f"het_{base.name}_g{float(g):g}_e{float(eta):g}",
                  lambda u: Wt(u) / W0, tail=float(tail))
