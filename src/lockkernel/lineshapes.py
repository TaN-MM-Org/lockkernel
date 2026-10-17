"""Inhomogeneous line shapes p(delta).

Each line shape is returned as a `LineShape`: a callable density together with
the pieces the rest of the package needs, namely the density at the centre,
a cumulative distribution and its inverse for building quadrature classes, and
a nominal width used to set integration scales.

Densities are written for `mpmath` scalars so that the same object can be used
both in the arbitrary precision parametric solver and, through `numpy`
broadcasting of the float path, in the finite precision dynamics.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import mpmath as mp
import numpy as np
from scipy import stats

__all__ = ["LineShape", "lorentzian", "gaussian", "student_t", "box",
           "bimodal_gaussian", "voigt", "LINESHAPES"]


@dataclasses.dataclass(frozen=True)
class LineShape:
    """A symmetric probability density on the detuning axis.

    Attributes
    ----------
    name : str
        Short identifier used in output files.
    pdf : callable
        Density, accepting an `mpmath` scalar and returning one.
    scale : float
        A nominal width, used only to choose integration break points.
    frozen : scipy.stats distribution or None
        Float precision distribution supplying `cdf` and `ppf`.  Line shapes
        that are only used analytically may leave this as None.
    compact_support : float or None
        Half width of the support when the density has compact support.
    """

    name: str
    pdf: Callable
    scale: float
    frozen: object = None
    compact_support: float = None

    def __post_init__(self):
        # the integrals place their break points at multiples of the scale;
        # a zero or negative one silently gives wrong results (since 1.4.0
        # refused)
        _positive(self.scale, "LineShape scale")

    def __call__(self, delta):
        return self.pdf(delta)

    def p0(self):
        """Density at the line centre."""
        return self.pdf(mp.mpf(0))

    def cdf(self, x):
        if self.frozen is None:
            raise NotImplementedError(f"{self.name} has no float cdf")
        return self.frozen.cdf(x)

    def ppf(self, q):
        if self.frozen is None:
            raise NotImplementedError(f"{self.name} has no float ppf")
        return self.frozen.ppf(q)


def _positive(value, what: str) -> None:
    v = mp.mpf(value)
    if not (mp.isfinite(v) and v > 0):
        raise ValueError(f"{what} must be finite and positive, got {value!r}")


def lorentzian(fwhm: float = 1.0) -> LineShape:
    """Lorentzian of the given full width at half maximum."""
    _positive(fwhm, "fwhm")
    g = mp.mpf(fwhm) / 2  # half width at half maximum

    def pdf(d):
        d = mp.mpf(d)
        return g / mp.pi / (d ** 2 + g ** 2)

    return LineShape("lorentzian", pdf, float(fwhm),
                     stats.cauchy(loc=0.0, scale=float(g)))


def gaussian(fwhm: float = 1.0) -> LineShape:
    """Gaussian of the given full width at half maximum."""
    _positive(fwhm, "fwhm")
    sig = mp.mpf(fwhm) / (2 * mp.sqrt(2 * mp.log(2)))

    def pdf(d):
        d = mp.mpf(d)
        return mp.e ** (-d ** 2 / (2 * sig ** 2)) / (sig * mp.sqrt(2 * mp.pi))

    return LineShape("gaussian", pdf, float(fwhm),
                     stats.norm(loc=0.0, scale=float(sig)))


def student_t(nu: float = 3.0, scale: float = 1.0) -> LineShape:
    """Student t density, a heavy tailed line with p(d) ~ |d|^-(nu+1)."""
    _positive(nu, "nu")
    _positive(scale, "scale")
    nu = mp.mpf(nu)
    sc = mp.mpf(scale)
    norm = mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2) * sc)

    def pdf(d):
        d = mp.mpf(d) / sc
        return norm * (1 + d ** 2 / nu) ** (-(nu + 1) / 2)

    return LineShape(f"student_t{float(nu):g}", pdf, float(scale),
                     stats.t(df=float(nu), scale=float(scale)))


def box(halfwidth: float = 1.0) -> LineShape:
    """Uniform line on [-w, w].  Discontinuous, kept as a stress test."""
    _positive(halfwidth, "halfwidth")
    w = mp.mpf(halfwidth)

    def pdf(d):
        return 1 / (2 * w) if abs(mp.mpf(d)) <= w else mp.mpf(0)

    return LineShape("box", pdf, float(halfwidth),
                     stats.uniform(loc=-float(w), scale=2 * float(w)),
                     compact_support=float(halfwidth))


def bimodal_gaussian(sep: float, width: float = 1.0) -> LineShape:
    """Two equal Gaussians of standard deviation `width` at +/- sep/2.

    The ratio sep/width controls the order of the transition; see
    `lockkernel.parametric.c_coefficient`.  A negative `sep` gives the same
    line as its absolute value.  (Before 1.4.0 its integration scale,
    width + sep/2, came out zero or negative for sep <= -2 width, and the
    integrals went wrong: for sep = -4, G(0.1) was twice the value for
    sep = 4 and `branch_point` with the Kuramoto kernel raised TypeError.)
    """
    _positive(width, "width")
    if not mp.isfinite(mp.mpf(sep)):
        raise ValueError(f"sep must be finite, got {sep!r}")
    a = mp.mpf(sep) / 2
    s = mp.mpf(width)

    def pdf(d):
        d = mp.mpf(d)
        z = 1 / (2 * s * mp.sqrt(2 * mp.pi))
        return z * (mp.e ** (-(d - a) ** 2 / (2 * s ** 2))
                    + mp.e ** (-(d + a) ** 2 / (2 * s ** 2)))

    return LineShape(f"bimodal{float(sep) / float(width):g}", pdf,
                     float(width) + abs(float(a)))


def voigt(fwhm_gauss: float = 1.0, fwhm_lorentz: float = 1.0) -> LineShape:
    """Voigt line: a Gaussian of FWHM `fwhm_gauss` convolved with a
    Lorentzian of FWHM `fwhm_lorentz` (new in 1.4.0).

    The usual shape of an inhomogeneous line that also has a Lorentzian
    part.  With sigma = fwhm_gauss / (2 sqrt(2 ln 2)) and gamma =
    fwhm_lorentz / 2 (the Lorentzian half width),

        p(delta) = Re w(z) / (sigma sqrt(2 pi)),
        z = (delta + i gamma) / (sigma sqrt 2),

    with the Faddeeva function w(z) = exp(-z^2) erfc(-i z) (DLMF 7.2.3;
    the Voigt functions are its real and imaginary parts, DLMF 7.19).
    Far out, p ~ gamma / (pi delta^2), the Lorentzian tail.  Evaluating
    Re w(z) there cancels about 2 log10|z| digits, so it is done with that
    many guard digits; the tests hold the density to a 120-digit
    evaluation, to SciPy's `voigt_profile`, to the convolution integral
    and to the Lorentzian tail.

    For the conservative kernel the branch is known in closed form,
    because the Lorentzian smoothing that makes G(Omega) turns the line
    into another Voigt line with gamma + Omega:

        G(Omega) = pi erfcx((gamma + Omega) / (sigma sqrt 2)) / (sigma sqrt(2 pi)),

    which the tests use.  Both widths must be finite and positive (use
    `gaussian` or `lorentzian` for the limits).  There is no floating-
    point distribution (`frozen` is None), so the line cannot be turned
    into a class table for the dynamics modules.
    """
    _positive(fwhm_gauss, "fwhm_gauss")
    _positive(fwhm_lorentz, "fwhm_lorentz")
    fg, fl = mp.mpf(fwhm_gauss), mp.mpf(fwhm_lorentz)

    def pdf(d):
        d = mp.mpf(d)
        # |Re z| = |d| / (sigma sqrt 2) = 2 sqrt(ln 2) |d| / fwhm_gauss
        guard = 10 + int(2 * mp.log10(1 + 2 * abs(d) / fg))
        with mp.extradps(guard):
            sig = fg / (2 * mp.sqrt(2 * mp.log(2)))
            r2 = sig * mp.sqrt(2)
            z = mp.mpc(d, fl / 2) / r2
            v = mp.re(mp.exp(-z ** 2) * mp.erfc(-1j * z)) \
                / (sig * mp.sqrt(2 * mp.pi))
        return +v

    return LineShape(f"voigt{float(fwhm_gauss):g}_{float(fwhm_lorentz):g}",
                     pdf, float(fwhm_gauss) + float(fwhm_lorentz))


LINESHAPES = {
    "lorentzian": lorentzian,
    "gaussian": gaussian,
    "student_t": student_t,
    "box": box,
    "bimodal_gaussian": bimodal_gaussian,
    "voigt": voigt,
}


def float_pdf(line: LineShape):
    """A vectorised float precision version of a line shape density."""
    return np.vectorize(lambda x: float(line.pdf(mp.mpf(float(x)))))
