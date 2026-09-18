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
           "bimodal_gaussian", "LINESHAPES"]


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


def lorentzian(fwhm: float = 1.0) -> LineShape:
    """Lorentzian of the given full width at half maximum."""
    g = mp.mpf(fwhm) / 2  # half width at half maximum

    def pdf(d):
        d = mp.mpf(d)
        return g / mp.pi / (d ** 2 + g ** 2)

    return LineShape("lorentzian", pdf, float(fwhm),
                     stats.cauchy(loc=0.0, scale=float(g)))


def gaussian(fwhm: float = 1.0) -> LineShape:
    """Gaussian of the given full width at half maximum."""
    sig = mp.mpf(fwhm) / (2 * mp.sqrt(2 * mp.log(2)))

    def pdf(d):
        d = mp.mpf(d)
        return mp.e ** (-d ** 2 / (2 * sig ** 2)) / (sig * mp.sqrt(2 * mp.pi))

    return LineShape("gaussian", pdf, float(fwhm),
                     stats.norm(loc=0.0, scale=float(sig)))


def student_t(nu: float = 3.0, scale: float = 1.0) -> LineShape:
    """Student t density, a heavy tailed line with p(d) ~ |d|^-(nu+1)."""
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
    w = mp.mpf(halfwidth)

    def pdf(d):
        return 1 / (2 * w) if abs(mp.mpf(d)) <= w else mp.mpf(0)

    return LineShape("box", pdf, float(halfwidth),
                     stats.uniform(loc=-float(w), scale=2 * float(w)),
                     compact_support=float(halfwidth))


def bimodal_gaussian(sep: float, width: float = 1.0) -> LineShape:
    """Two equal Gaussians of standard deviation `width` at +/- sep/2.

    The ratio sep/width controls the order of the transition; see
    `lockkernel.parametric.c_coefficient`.
    """
    a = mp.mpf(sep) / 2
    s = mp.mpf(width)

    def pdf(d):
        d = mp.mpf(d)
        z = 1 / (2 * s * mp.sqrt(2 * mp.pi))
        return z * (mp.e ** (-(d - a) ** 2 / (2 * s ** 2))
                    + mp.e ** (-(d + a) ** 2 / (2 * s ** 2)))

    return LineShape(f"bimodal{float(sep) / float(width):g}", pdf,
                     float(width) + float(a))


LINESHAPES = {
    "lorentzian": lorentzian,
    "gaussian": gaussian,
    "student_t": student_t,
    "box": box,
    "bimodal_gaussian": bimodal_gaussian,
}


def float_pdf(line: LineShape):
    """A vectorised float precision version of a line shape density."""
    return np.vectorize(lambda x: float(line.pdf(mp.mpf(float(x)))))
