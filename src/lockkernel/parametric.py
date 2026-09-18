"""The exact parametric solution of the locking self consistency.

THE SELF CONSISTENCY.  Write R for the order parameter, chiN for the coupling
per spin times the spin number, and Omega = chiN R for the locking bandwidth,
the width in detuning over which an oscillator follows the mean field.  With a
locking kernel W the stationary condition is

    R = int p(delta) W(delta / Omega) d delta.

THE RECASTING.  Substituting delta = Omega u turns the right hand side into a
function of Omega alone,

    H(Omega) = Omega int p(Omega u) W(u) du,

so that R = H(Omega), and since R = Omega / chiN,

    chiN(Omega) = Omega / H(Omega),      R(Omega) = H(Omega).

This is an EXACT PARAMETRIC SOLUTION of the transition for any line shape and
any kernel.  Sweeping Omega upward from zero traces the branch, with no root
finding and no cancellation near threshold, which is what makes high precision
exponent extraction possible.  Letting Omega go to zero gives the threshold

    chiN_c = 1 / (p(0) m),      m = int W(u) du,

for every kernel: the kernel enters the threshold only through its mass.

THE MECHANISM.  Everything about the onset is contained in how H leaves its
linear behaviour.  Define

    G(Omega) = H(Omega) / Omega = int p(Omega u) W(u) du,

so G(0) = p(0) m.  If the kernel has an algebraic tail W ~ |u|^-s with
1 < s < 3, the integral samples the whole line and

    G(0) - G(Omega) ~ const * Omega^(s-1),

because the tail of the kernel meets the tail of the line.  If instead the
kernel has compact support, or decays faster than any power, the leading
correction comes from the local curvature of the line and is quadratic,

    G(0) - G(Omega) = -(1/2) p''(0) M_2 Omega^2 + O(Omega^4),
    M_2 = int u^2 W(u) du.

Since eps = chiN/chiN_c - 1 = G(0)/G(Omega) - 1 and R = Omega G(Omega), the
order parameter follows eps to the power beta with

    beta = 1/(s-1)   for 1 < s < 3,        beta = 1/2   for s >= 3.

The crossover at s = 3 is exactly where the second moment M_2 of the kernel
stops converging, so that the tail contribution and the curvature contribution
exchange dominance.

THE CONSERVATIVE CASE.  For conservative spins W(u) = 1/(1+u^2), whence
s = 2, m = pi and beta = 1.  Then H is a Lorentzian smoothing of the line,

    H(Omega) = pi Omega (p * L_Omega)(0),

with L_Omega the normalised Lorentzian of half width Omega, and the linear
coefficient can be written in closed form,

    G(Omega) = pi [p(0) - c Omega + o(Omega)],
    c = -(1/pi) int [p(delta) - p(0)] / delta^2 d delta,

giving R = A eps with A = pi p(0)^2 / c.  The SIGN of c decides the order of
the transition: c > 0 gives a monotonic branch and a continuous onset, c < 0
gives a fold in the branch and therefore a hysteretic, first order onset.
"""
from __future__ import annotations

import mpmath as mp

from .kernels import Kernel, conservative
from .lineshapes import LineShape

__all__ = ["G_of_Omega", "H_of_Omega", "branch_point", "threshold",
           "c_coefficient", "amplitude", "sweep", "extract_beta",
           "fold_interval"]

INF = mp.inf


def _breakpoints(line: LineShape, kernel: Kernel, Omega):
    """Quadrature break points in u for int p(Omega u) W(u) du.

    The integrand has structure on two scales: the kernel varies on u ~ 1 and
    the line varies on u ~ scale/Omega.  Both are given to the quadrature.
    """
    s = mp.mpf(line.scale) / Omega
    if kernel.support is not None:
        cut = mp.mpf(kernel.support)
        pts = sorted({mp.mpf(0), min(s, cut), cut})
        return [-p for p in reversed(pts)] + list(pts)[1:]
    pts = sorted({mp.mpf(0), mp.mpf(1), s, 10 * s, mp.mpf(10)})
    return [-INF] + [-p for p in reversed(pts) if p > 0] + list(pts) + [INF]


def G_of_Omega(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """G(Omega) = int p(Omega u) W(u) du.  Finite and positive at Omega = 0."""
    Omega = mp.mpf(Omega)
    if Omega == 0:
        return line.p0() * kernel.mass()
    f = lambda u: line.pdf(Omega * u) * kernel.W(u)
    return mp.quad(f, _breakpoints(line, kernel, Omega))


def H_of_Omega(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """H(Omega) = Omega G(Omega).  Equals the order parameter on the branch."""
    return mp.mpf(Omega) * G_of_Omega(line, kernel, Omega)


def threshold(line: LineShape, kernel: Kernel) -> mp.mpf:
    """chiN_c = 1 / (p(0) m).  The kernel enters only through its mass."""
    return 1 / (line.p0() * kernel.mass())


def branch_point(line: LineShape, kernel: Kernel, Omega):
    """One point of the branch: (chiN, R, eps) at locking bandwidth Omega."""
    Omega = mp.mpf(Omega)
    G = G_of_Omega(line, kernel, Omega)
    chiN = 1 / G
    R = Omega * G
    eps = chiN / threshold(line, kernel) - 1
    return chiN, R, eps


def sweep(line: LineShape, kernel: Kernel, exponents):
    """The branch at Omega = 10^e for each e in `exponents`."""
    return [branch_point(line, kernel, mp.mpf(10) ** mp.mpf(e)) for e in exponents]


def extract_beta(line: LineShape, kernel: Kernel, exponents):
    """Successive local slopes d log R / d log eps along the branch.

    Returned as a list with one fewer entry than `exponents`.  Convergence of
    the list is the evidence that the exponent has been reached; the last
    entry is quoted as the measured exponent.
    """
    pts = sweep(line, kernel, exponents)
    out = []
    for i in range(len(pts) - 1):
        e0, r0 = pts[i][2], pts[i][1]
        e1, r1 = pts[i + 1][2], pts[i + 1][1]
        out.append(mp.log(r1 / r0) / mp.log(e1 / e0))
    return out


def c_coefficient(line: LineShape) -> mp.mpf:
    """c = -(1/pi) int [p(delta) - p(0)] / delta^2 d delta.

    Defined for the conservative kernel only, where it is both the linear
    coefficient of the smoothed peak and the criterion for the order of the
    transition.  The integrand has a REMOVABLE singularity at the origin, its
    limit being p''(0)/2, so the quadrature must not evaluate it there; the
    range is split away from zero and the integrand is even.
    """
    p0 = line.p0()
    sc = mp.mpf(line.scale)

    def f(d):
        if d == 0:
            return mp.diff(line.pdf, mp.mpf(0), 2) / 2
        return (line.pdf(d) - p0) / d ** 2

    val = 2 * mp.quad(f, [mp.mpf(10) ** -20, sc * mp.mpf("1e-3"), sc,
                          10 * sc, 100 * sc, INF])
    return -val / mp.pi


def amplitude(line: LineShape) -> mp.mpf:
    """A = pi p(0)^2 / c, the slope of R against eps for conservative spins."""
    return mp.pi * line.p0() ** 2 / c_coefficient(line)


def tail_integral(line: LineShape, s) -> mp.mpf:
    """I_s = int [p(0) - p(delta)] |delta|^-s d delta.

    This is the object that carries the whole onset for a kernel of tail
    exponent s, and the regime 1 < s < 3 of the exponent formula is exactly
    the regime in which it converges: at s = 3 it diverges at the origin, so
    the local curvature takes over and the exponent returns to 1/2, and at
    s = 1 it diverges at infinity, together with the mass of the kernel, so
    the threshold collapses.  At s = 2 it equals pi times the coefficient c.

    Integrating by parts,

        I_s = (2 / (s-1)) int_0^inf delta^(1-s) [-p'(delta)] d delta,

    which is the form used here: it removes the subtraction of two nearly
    equal numbers near the origin, where p(0) - p(delta) is a cancellation of
    order delta^2 that costs as many digits as the quadrature comes close.
    """
    s = mp.mpf(s)
    if s <= 1 or s >= 3:
        raise ValueError("I_s converges only for 1 < s < 3")
    sc = mp.mpf(line.scale)
    f = lambda d: d ** (1 - s) * (-mp.diff(line.pdf, d))
    return (2 / (s - 1)) * (mp.quad(f, [0, sc])
                            + mp.quad(f, [sc, 10 * sc, 100 * sc, INF]))


def amplitude_general(line: LineShape, kernel: Kernel, C=None) -> mp.mpf:
    """A_s in R = A_s eps^(1/(s-1)), for a kernel of tail W ~ C |u|^-s.

        A_s = p(0) m [ p(0) m / (C I_s) ]^(1/(s-1)),

    with m the kernel mass and I_s the integral above.  It reduces to
    pi p(0)^2 / c at s = 2, where the exponent is one and the branch is
    linear.  `C` defaults to the tail amplitude measured from the kernel
    itself, which is one for the family 1/(1+|u|^s).
    """
    s = mp.mpf(kernel.tail)
    m = kernel.mass()
    if C is None:
        v = mp.mpf(10) ** 6
        C = kernel.W(v) * v ** s
    p0m = line.p0() * m
    return p0m * (p0m / (mp.mpf(C) * tail_integral(line, s))) ** (1 / (s - 1))


def fold_interval(line: LineShape, kernel: Kernel = None, n: int = 400,
                  lo: float = -4.0, hi: float = 1.5):
    """Locate a fold in the branch, if there is one.

    The branch chiN(Omega) is sampled on a logarithmic grid.  Where it is not
    monotonic there are two turning points, and between the couplings they
    define the stationary condition has three solutions, so the onset is
    hysteretic.  Each turning point is refined by solving d chiN / d Omega = 0
    rather than being read off the grid, and the size of the jump is measured
    at the upper turning point, where the low branch ceases to exist and the
    order parameter must move to the highest solution at the same coupling.

    Returns None when the branch is monotonic.
    """
    kernel = kernel or conservative()
    Om = [mp.mpf(10) ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    chi = [branch_point(line, kernel, o)[0] for o in Om]
    down = [i for i in range(len(chi) - 1) if chi[i + 1] < chi[i]]
    if not down:
        return None
    i0, i1 = down[0], down[-1] + 1

    dchi = lambda o: mp.diff(lambda x: branch_point(line, kernel, x)[0], o)
    def refine(ia, ib):
        try:
            return mp.findroot(dchi, (Om[ia], Om[ib]), solver="bisect",
                               tol=mp.mpf("1e-16"))
        except Exception:
            return Om[ia]

    O_hi = refine(max(i0 - 1, 0), min(i0 + 1, n - 1))   # end of the low branch
    O_lo = refine(max(i1 - 1, 0), min(i1 + 1, n - 1))   # end of the high branch
    chiN_hi, R_low, _ = branch_point(line, kernel, O_hi)
    chiN_lo, R_high_at_lo, _ = branch_point(line, kernel, O_lo)

    # the state the system jumps to: the largest Omega with chiN(Omega) = chiN_hi
    f = lambda o: branch_point(line, kernel, o)[0] - chiN_hi
    O_jump = None
    for k in range(i1, n - 1):
        if f(Om[k]) * f(Om[k + 1]) <= 0:
            O_jump = mp.findroot(f, (Om[k], Om[k + 1]), solver="bisect",
                                 tol=mp.mpf("1e-14"))
    R_jump = branch_point(line, kernel, O_jump)[1] if O_jump else None
    return dict(chiN_lo=chiN_lo, chiN_hi=chiN_hi,
                R_low=R_low, R_jump=R_jump, R_high_at_lo=R_high_at_lo,
                Omega_lo=O_lo, Omega_hi=O_hi)
