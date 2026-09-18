"""Discretisation of a line into detuning classes.

The dynamics is solved class by class, so a continuous line has to be replaced
by a finite set of detunings carrying the right populations.  Two features of
the problem dictate the construction.

* The line can have ALGEBRAIC TAILS.  A uniform grid, or a grid truncated at a
  few widths, changes the density at the centre once it is renormalised, and
  the threshold depends on that density directly.  The construction below
  therefore keeps the entire population: nothing is discarded.

* The far tail is STIFF and IRRELEVANT.  An emitter detuned by a thousand
  linewidths precesses a thousand times faster than anything else in the
  problem, which forces the integrator to tiny steps, while its interaction
  with the mean field is smaller by (chiN/delta)^2.  Such emitters are
  therefore removed from the differential equation and propagated exactly as
  free spins, with their population fully retained.

The three regions are

    core      |delta| < core_edge          M_core equal probability bins
    tail      core_edge < |delta| < free   M_tail logarithmic bins
    free      |delta| > free               M_spec logarithmic bins, exact

with the node of each bin placed at its median, so that the first moment of
the population within the bin is represented rather than its edge.
"""
from __future__ import annotations

import numpy as np

from .cumulant import System
from .lineshapes import LineShape

__all__ = ["build_system", "class_table"]


def _log_bins(dist, lo, hi, M, lump_last):
    """Logarithmic bins on [lo, hi] of the positive half line."""
    edges = np.geomspace(lo, hi, M + 1)
    nodes, mass = [], []
    for k in range(M):
        a, b = edges[k], edges[k + 1]
        m = float(1.0 - dist.cdf(a)) if (lump_last and k == M - 1) \
            else float(dist.cdf(b) - dist.cdf(a))
        if not np.isfinite(m) or m <= 1e-14:
            continue
        node = float(dist.ppf(dist.cdf(a) + 0.5 * m))
        if not np.isfinite(node):
            node = float(np.sqrt(a * b))
        nodes.append(node)
        mass.append(m)
    return np.array(nodes), np.array(mass)


def class_table(line: LineShape, M_core: int = 64, M_tail: int = 14,
                M_spec: int = 8, core_edge: float = None,
                free_beyond: float = None, delta_max: float = None):
    """Return (delta, n, spec_delta, spec_n) with populations summing to one."""
    dist = line.frozen
    if dist is None:
        raise ValueError(f"{line.name} has no float distribution to discretise")
    scale = float(line.scale)
    core_edge = 3.0 * scale if core_edge is None else float(core_edge)
    delta_max = 1000.0 * scale if delta_max is None else float(delta_max)
    tail_end = delta_max if free_beyond is None else max(float(free_beyond),
                                                         core_edge * 1.001)

    c_lo, c_hi = float(dist.cdf(-core_edge)), float(dist.cdf(core_edge))
    q = c_lo + (c_hi - c_lo) * (np.arange(M_core) + 0.5) / M_core
    d_core = np.asarray(dist.ppf(q), float)
    n_core = np.full(M_core, (c_hi - c_lo) / M_core)

    lump = free_beyond is None
    d_t, n_t = _log_bins(dist, core_edge, tail_end, M_tail, lump)
    delta = np.concatenate([-d_t[::-1], d_core, d_t])
    n = np.concatenate([n_t[::-1], n_core, n_t])

    if free_beyond is None:
        s_d, s_n = np.zeros(0), np.zeros(0)
    else:
        d_s, n_s = _log_bins(dist, tail_end, delta_max, M_spec, True)
        s_d = np.concatenate([-d_s[::-1], d_s])
        s_n = np.concatenate([n_s[::-1], n_s])

    total = n.sum() + s_n.sum()
    return delta, n / total, s_d, s_n / total


def build_system(line: LineShape, N: float, chiN: float, M_core: int = 64,
                 M_tail: int = 14, M_spec: int = 8, core_edge: float = None,
                 free_factor: float = 8.0, delta_max: float = None) -> System:
    """Build a `System` for a line, an emitter number and a coupling.

    The boundary of the free region is placed at `free_factor` times the
    larger of the coupling and twice the line width, so that it always sits
    well outside the locking bandwidth.  Convergence in `free_factor` is
    checked in `scripts/vlasov_check.py`.
    """
    scale = float(line.scale)
    free = free_factor * max(abs(float(chiN)), 2.0 * scale)
    delta, n, s_d, s_n = class_table(line, M_core, M_tail, M_spec,
                                     core_edge, free, delta_max)
    return System(delta=delta, n=n * N, chiN=float(chiN),
                  spec_delta=s_d, spec_n=s_n * N)
