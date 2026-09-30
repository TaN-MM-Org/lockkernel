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

The value 1/2 needs a curvature term, p''(0) < 0.  On a line that is flat at
its centre (the `box` line) G(0) - G(Omega) = p(0) int_{|u| > w/Omega} W(u) du
exactly, for Omega below the half width w, so a tail gives beta = 1/(s-1) for
every s > 1 (below 1/2 once s > 3), and a compactly supported kernel gives
G(Omega) = G(0): no power law, the order parameter jumps at the threshold.
For p''(0) > 0 (a dip at the centre) the branch bends back.

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
           "fold_interval", "tail_integral", "amplitude_general",
           "amplitude_curvature"]

INF = mp.inf


def _half_points(line: LineShape, kernel: Kernel, Omega):
    """Break points on u >= 0 for an even integrand, the start X of the tail
    that `_even_quad` maps onto (0, 1] (None for compact support), and the
    power k of that map.

    The integrand has structure on two scales: the kernel varies on u ~ 1
    and the line on u ~ scale/Omega.  Both are given to the quadrature.
    """
    s = mp.mpf(line.scale) / Omega
    if kernel.support is not None:
        cut = mp.mpf(kernel.support)
        return sorted({mp.mpf(0), min(s, cut), cut}), None, 1
    pos = sorted({mp.mpf(0), mp.mpf(1), mp.mpf(10), s, 10 * s})
    k = 1 / (mp.mpf(kernel.tail) - 1) if kernel.tail is not None else 1
    return pos, pos[-1], k


def _even_quad(f, pos, X, k=1):
    """(2 int_0^inf f(u) du, error estimate) for an even integrand f.

    Three parts.  Up to u = 1 the integral is taken directly.  From u = 1
    to X it is taken in v = log(u), because the integrand there falls or
    rises like a power of u across many decades: on a linear scale the
    quadrature cannot resolve the start of a segment such as
    [10, 10^12], and before 1.2.0 this left G(Omega) about 4e-10 off at
    Omega = 1e-12 and mpmath's default 15 digits.  The tail [X, inf) is
    mapped onto (0, 1] by u = X t^(-k); with k = 1/(s-1) a tail
    f ~ u^(-s) becomes a constant at t -> 0.  (mpmath's own treatment of
    [X, inf) with X large is good only to about 1e-9 relative at 15
    digits.)  `pos` lists the break points on u >= 0; for compact support
    (X None) they are all integrated directly.
    """
    if X is None:
        return tuple(2 * x for x in mp.quad(f, pos, error=True))
    lin = [p for p in pos if p <= 1]
    logs = [mp.log(p) for p in pos if p >= 1]
    v, e = mp.quad(f, lin, error=True)
    v1, e1 = mp.quad(lambda w: f(mp.exp(w)) * mp.exp(w), logs, error=True)
    v2, e2 = mp.quad(lambda t: f(X * t ** -k) * k * X * t ** (-k - 1),
                     [0, 1], error=True)
    return 2 * (v + v1 + v2), 2 * (e + e1 + e2)


def G_of_Omega(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """G(Omega) = int p(Omega u) W(u) du.  Finite and positive at Omega = 0."""
    Omega = mp.mpf(Omega)
    if Omega == 0:
        return line.p0() * kernel.mass()
    f = lambda u: line.pdf(Omega * u) * kernel.W(u)
    return _even_quad(f, *_half_points(line, kernel, Omega))[0]


def H_of_Omega(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """H(Omega) = Omega G(Omega).  Equals the order parameter on the branch."""
    return mp.mpf(Omega) * G_of_Omega(line, kernel, Omega)


def threshold(line: LineShape, kernel: Kernel) -> mp.mpf:
    """chiN_c = 1 / (p(0) m).  The kernel enters only through its mass."""
    return 1 / (line.p0() * kernel.mass())


def _G0_minus_G(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """D(Omega) = G(0) - G(Omega) = int [p(0) - p(Omega u)] W(u) du.

    The reduced coupling is eps = G(0)/G(Omega) - 1 = D / G.  Forming it
    as chiN/chiN_c - 1 subtracts two nearly equal numbers and loses
    about log10(1/eps) digits; integrating the difference directly does
    not.  Two details keep the digits.  p(0) - p(Omega u) itself cancels
    near the centre of the line, by about 2 log10(scale/Omega) digits, so
    it is formed with that many guard digits.  And mpmath's quadrature
    stops on an ABSOLUTE error estimate, which a small D meets too early,
    so the integral is done twice, the second time with the integrand
    divided by the size the first pass found.
    """
    Omega = mp.mpf(Omega)
    if Omega == 0:
        return mp.mpf(0)
    ratio = Omega / mp.mpf(line.scale)
    guard = 10
    if ratio < 1:
        guard += int(mp.ceil(-2 * mp.log10(ratio)))
    with mp.extradps(guard):
        p0 = line.p0()

    def f(u):
        with mp.extradps(guard):
            d = p0 - line.pdf(Omega * u)
        return (+d) * kernel.W(u)

    pts = _half_points(line, kernel, Omega)
    val = _even_quad(f, *pts)[0]
    if val == 0:
        return val
    size = abs(val)
    return size * _even_quad(lambda u: f(u) / size, *pts)[0]


def branch_point(line: LineShape, kernel: Kernel, Omega):
    """One point of the branch: (chiN, R, eps) at locking bandwidth Omega.

    chiN = 1/G(Omega), R = Omega G(Omega), and the reduced coupling
    eps = chiN/chiN_c - 1 is computed as [G(0) - G(Omega)] / G(Omega) from
    a direct integral of the difference, so it keeps the working precision
    however small it is (since 1.2.0; before, it lost about log10(1/eps)
    digits to cancellation).
    """
    Omega = mp.mpf(Omega)
    G = G_of_Omega(line, kernel, Omega)
    chiN = 1 / G
    R = Omega * G
    eps = _G0_minus_G(line, kernel, Omega) / G
    return chiN, R, eps


def sweep(line: LineShape, kernel: Kernel, exponents):
    """The branch at Omega = 10^e for each e in `exponents`."""
    return [branch_point(line, kernel, mp.mpf(10) ** mp.mpf(e)) for e in exponents]


def extract_beta(line: LineShape, kernel: Kernel, exponents):
    """Successive local slopes d log R / d log eps along the branch.

    Returned as a list with one fewer entry than `exponents`.  Convergence of
    the list is the evidence that the exponent has been reached; the last
    entry is quoted as the measured exponent.

    Refuses (ValueError) when a point has eps <= 0, that is, a coupling at
    or below the threshold: the branch then bends back (a first order,
    hysteretic onset, see `fold_interval`) or is flat (the order parameter
    jumps at the threshold), and there is no power law to measure.  Before
    1.2.0 a backward branch returned a plausible looking slope and a flat
    one raised ZeroDivisionError.
    """
    pts = sweep(line, kernel, exponents)
    for e, p in zip(exponents, pts):
        if not p[2] > 0:
            raise ValueError(
                f"eps = {mp.nstr(p[2], 5)} <= 0 at Omega = 10^{e}: the branch "
                "is at or below the threshold there, so the onset is not a "
                "power law R ~ eps^beta. A negative eps means the branch "
                "bends back (first order, hysteretic onset; see "
                "fold_interval); eps = 0 means it is flat and R jumps at "
                "the threshold")
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

    Near the lower end of the range, p(delta) - p(0) is a cancellation that
    loses about 40 digits, so the quadrature runs with 25 digits more than
    the working precision and the lower end is placed at 1e-20 line widths.
    Without this the result was wrong at mpmath's default precision (15
    digits): `amplitude(lorentzian())` came out near 5e-4 instead of 1.
    Skipping the first 1e-20 line widths leaves out a piece of relative
    size about 1e-20, so the result is good to about 20 significant digits
    at most, however high the working precision is set.
    """
    with mp.extradps(25):
        p0 = line.p0()
        sc = mp.mpf(line.scale)

        def f(d):
            if d == 0:
                return mp.diff(line.pdf, mp.mpf(0), 2) / 2
            return (line.pdf(d) - p0) / d ** 2

        val = 2 * mp.quad(f, [sc * mp.mpf(10) ** -20, sc * mp.mpf("1e-3"), sc,
                              10 * sc, 100 * sc, INF])
        out = -val / mp.pi
    return +out


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


def amplitude_curvature(line: LineShape, kernel: Kernel) -> mp.mpf:
    """A in R = A eps^(1/2), for the class where beta = 1/2.

    That class is every kernel with compact support (Kuramoto), faster
    than algebraic decay (Gaussian), or a tail s > 3, on a line with a
    rounded maximum at its centre, p''(0) < 0.  There the first
    correction to G(0) = p(0) m is set by the curvature of the line,

        G(Omega) = G(0) + (1/2) p''(0) M_2 Omega^2 + ...,
        M_2 = int u^2 W(u) du,

    and with eps = G(0)/G - 1 and R = Omega G this gives

        A = G(0)^(3/2) sqrt( 2 / (-p''(0) M_2) ),     G(0) = p(0) m.

    For the Kuramoto kernel on a Lorentzian line A = 1, which is the
    closed form R = sqrt(1 - chiN_c/chiN) near its onset.  The next
    correction to R / sqrt(eps) is of relative order Omega^2, or
    Omega^(s-3) for a tail 3 < s < 5, so it is approached slowly just
    above s = 3.

    Refuses a kernel with tail s <= 3 (M_2 diverges; use `amplitude` or
    `amplitude_general`) and a line with p''(0) >= 0: at p''(0) = 0 (a
    flat top, such as `box`) there is no curvature term, and at
    p''(0) > 0 (a dip at the centre) the branch bends back and the onset
    is first order.  p''(0) is taken by mpmath's numerical
    differentiation.
    """
    M2 = kernel.second_moment()
    pp = mp.diff(line.pdf, mp.mpf(0), 2)
    if not pp < 0:
        raise ValueError(
            f"p''(0) = {mp.nstr(pp, 5)} is not negative: the line has no "
            "rounded maximum at its centre, so the onset is not of the "
            "form R = A eps^(1/2) (flat top: no curvature term; dip: the "
            "branch bends back and the onset is first order)")
    G0 = line.p0() * kernel.mass()
    return G0 * mp.sqrt(2 * G0 / (-pp * M2))


def _dG_dOmega(line: LineShape, kernel: Kernel, Omega) -> mp.mpf:
    """dG/dOmega = int u p'(Omega u) W(u) du, for a differentiable line."""
    Omega = mp.mpf(Omega)
    f = lambda u: u * mp.diff(line.pdf, Omega * u) * kernel.W(u)
    return _even_quad(f, *_half_points(line, kernel, Omega))[0]


def fold_interval(line: LineShape, kernel: Kernel = None, n: int = 400,
                  lo: float = -4.0, hi: float = 1.5):
    """Locate a fold in the branch, if there is one.

    G(Omega), and so chiN = 1/G, is sampled at n points with Omega from
    10^lo to 10^hi, evenly spaced in log(Omega).  Where chiN(Omega) is not
    monotonic the branch folds: between the two turning points the
    stationary condition has three solutions, and the onset is
    hysteretic.  Each turning point is refined by solving
    dG/dOmega = int u p'(Omega u) W(u) du = 0, so the line must be
    differentiable (a fold on a line with compact support is refused).
    The jump is found at the upper turning point, where the low branch
    ends and R must move to the highest solution at the same coupling.

    Two shapes are handled.  If chiN falls from the start of the grid
    with chiN(10^lo) below the threshold, the branch leaves the threshold
    backwards (a subcritical onset, as for conservative spins with c < 0),
    and the low branch ends at the threshold itself: Omega_hi = 0,
    chiN_hi = chiN_c, R_low = 0, all exact.  Otherwise it rises first and
    folds later (an S shape) and both turning points are interior.
    Structure below 10^lo or finer than the grid is not resolved; with
    several folds, the first and last turning points are used.

    Returns None when the branch is monotonic on the grid, otherwise a
    dict with chiN_lo, chiN_hi (the coupling range of the hysteresis),
    R_low (R at the end of the low branch), R_jump (R after the jump, or
    None if the high branch does not reach chiN_hi below 10^hi),
    R_high_at_lo (R where the high branch ends), Omega_lo, Omega_hi and
    Omega_jump.

    Cost: n + about 40 quadratures (about 0.1 s each at 15 digits on a
    two-peaked Gaussian line).  Changed in 1.2.0: the turning points are
    refined from the integral for dG/dOmega instead of differentiating the
    quadrature numerically (roughly 20 times faster), a refinement that
    fails now raises instead of silently returning a grid point, and a
    subcritical onset returns the exact threshold as chiN_hi instead of
    the first grid point.
    """
    kernel = kernel or conservative()
    Om = [mp.mpf(10) ** (lo + (hi - lo) * mp.mpf(i) / (n - 1)) for i in range(n)]
    G = [G_of_Omega(line, kernel, o) for o in Om]
    # chiN = 1/G falls where G rises; ignore changes at the rounding level
    tiny = mp.mpf(2) ** (20 - mp.mp.prec)
    up = [i for i in range(n - 1) if G[i + 1] > G[i] * (1 + tiny)]
    if not up:
        return None
    if line.compact_support is not None:
        raise ValueError(
            f"{line.name} has compact support (it is not differentiable at "
            "its edge), and the turning points are refined from p'; "
            "fold_interval does not handle this line")
    i0, i1 = up[0], up[-1] + 1
    G0 = line.p0() * kernel.mass()
    seen = {}

    def dG(o):            # each value costs a quadrature; keep them
        if o not in seen:
            seen[o] = _dG_dOmega(line, kernel, o)
        return seen[o]

    def turning(ia, ib):
        for _ in range(3):
            a, b = Om[ia], Om[ib]
            if dG(a) * dG(b) <= 0:
                return mp.findroot(dG, (a, b), solver="anderson")
            ia, ib = max(ia - 1, 0), min(ib + 1, n - 1)
        raise RuntimeError(
            "a turning point of the branch was not bracketed on the grid; "
            "raise n, or lower lo if it lies below 10^lo")

    if i0 == 0 and G[0] > G0:
        O_hi, G_hi = mp.mpf(0), G0          # leaves the threshold backwards
    else:
        O_hi = turning(max(i0 - 1, 0), min(i0 + 1, n - 1))
        G_hi = G_of_Omega(line, kernel, O_hi)
    O_lo = turning(max(i1 - 1, 0), min(i1 + 1, n - 1))
    G_lo = G_of_Omega(line, kernel, O_lo)

    # the state the system jumps to: the largest Omega with G(Omega) = G_hi
    O_jump = None
    for k in range(i1, n - 1):
        if (G[k] - G_hi) * (G[k + 1] - G_hi) <= 0:
            O_jump = mp.findroot(lambda o: G_of_Omega(line, kernel, o) - G_hi,
                                 (Om[k], Om[k + 1]), solver="anderson")
    R_jump = O_jump * G_of_Omega(line, kernel, O_jump) if O_jump else None
    return dict(chiN_lo=1 / G_lo, chiN_hi=1 / G_hi,
                R_low=O_hi * G_hi, R_jump=R_jump, R_high_at_lo=O_lo * G_lo,
                Omega_lo=O_lo, Omega_hi=O_hi, Omega_jump=O_jump)
