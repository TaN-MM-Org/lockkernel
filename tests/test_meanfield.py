"""The mean field dynamics against the locking law, and the discretisation."""
import numpy as np
import pytest

from lockkernel import lineshapes as L
from lockkernel.cumulant import evolve_meanfield
from lockkernel.ensemble import build_system, class_table

FW = 2 * np.pi
CHINC = 0.5 * FW


def contrast(r, M_core=400, T=150.0, npts=500, freefac=8.0):
    sysm = build_system(L.lorentzian(FW), 1e10, r * CHINC, M_core=M_core,
                        M_tail=24, M_spec=8, core_edge=3 * FW,
                        free_factor=freefac, delta_max=2000.0 * FW)
    t_eval = np.linspace(0.0, T, npts)
    s, _ = evolve_meanfield(sysm, T, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    return float(np.mean(coh[t_eval > 0.5 * T]))


def test_discretisation_keeps_the_whole_population():
    """Nothing is discarded: the populations sum to one."""
    for line in (L.lorentzian(1.0), L.gaussian(1.0), L.student_t(3.0, 0.5)):
        _, n, _, sn = class_table(line, M_core=64, M_tail=14, M_spec=8,
                                  free_beyond=20.0, delta_max=2000.0)
        assert abs(n.sum() + sn.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("Omega", [0.1, 0.5, 2.0])
def test_discretisation_reproduces_the_locking_integral(Omega):
    """The class table must reproduce the integral the locking law is built on.

    This is the quantity the discretisation exists to represent, so it is what
    the accuracy of the discretisation should be measured on, rather than any
    intermediate such as the density at the centre.
    """
    import mpmath as mp
    from lockkernel import kernels as K
    from lockkernel import parametric as P

    mp.mp.dps = 20
    line = L.lorentzian(1.0)
    cons = K.conservative()
    delta, n, s_d, s_n = class_table(line, M_core=800, M_tail=24, M_spec=8,
                                     free_beyond=20.0, delta_max=2000.0)
    W = lambda d: 1.0 / (1.0 + (d / Omega) ** 2)
    got = float(n @ W(delta) + s_n @ W(s_d))
    want = float(P.G_of_Omega(line, cons, Omega) * Omega)
    assert abs(got - want) / want < 2e-3


@pytest.mark.parametrize("r", [1.2, 2.0, 3.0])
def test_dynamics_reaches_the_locking_law(r):
    """The conservative mean field settles on R = 1 - 1/r, not near it."""
    assert abs(contrast(r) - (1.0 - 1.0 / r)) < 5e-3


def test_no_synchronisation_below_threshold():
    """Below threshold the contrast decays away instead of settling."""
    assert contrast(0.6) < 0.05
