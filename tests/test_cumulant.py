"""The cumulant solver against exact references.

Two independent checks.  On a degenerate line the model is one axis twisting
on the maximal spin manifold, which is exact and cheap at any emitter number,
so the cumulant error can be watched falling as 1/N.  On an inhomogeneous line
the reference is exact diagonalisation of the full Hamiltonian for a small
ensemble, which tests every term of the hierarchy: a transcription error in
any one of them would appear at leading order in time.
"""
import numpy as np
import pytest

from lockkernel.cumulant import (System, coherence, coherent_state_x, evolve,
                                 wineland_xi2)
from lockkernel.exact import full_exact, symmetric_exact


def _cumulant(sysm, times, rtol=1e-11):
    states = evolve(coherent_state_x(sysm), sysm, times[-1], t_eval=times,
                    rtol=rtol, atol=1e-14)
    return (np.array([coherence(s, sysm) for s in states]),
            np.array([wineland_xi2(s, sysm) for s in states]))


@pytest.mark.parametrize("N,tol_xi", [(50, 2.0e-1), (200, 1.5e-2), (1000, 1.5e-3)])
def test_degenerate_line_against_exact_twisting(N, tol_xi):
    """The error against the exact result falls like 1/N."""
    chiN = 2.0
    times = np.linspace(0.0, 3.0, 7)[1:]
    sysm = System(delta=np.zeros(1), n=np.array([float(N)]), chiN=chiN)
    co_c, xi_c = _cumulant(sysm, times, rtol=1e-10)
    co_e, xi_e = symmetric_exact(N, chiN, times)
    assert np.max(np.abs(xi_c - xi_e)) < tol_xi
    assert np.max(np.abs(co_c - co_e)) < tol_xi / 4


def test_cumulant_error_decreases_with_N():
    chiN = 2.0
    times = np.linspace(0.0, 3.0, 7)[1:]
    errs = []
    for N in (50, 200, 1000):
        sysm = System(delta=np.zeros(1), n=np.array([float(N)]), chiN=chiN)
        _, xi_c = _cumulant(sysm, times, rtol=1e-10)
        _, xi_e = symmetric_exact(N, chiN, times)
        errs.append(np.max(np.abs(xi_c - xi_e)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < errs[0] / 50


@pytest.mark.parametrize("deltas", [
    [-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0],
    [-1.2, -1.2, -0.4, -0.4, -0.4, 0.4, 0.4, 0.4, 1.2, 1.2],
])
def test_inhomogeneous_line_against_full_diagonalisation(deltas):
    """Short time agreement to eight digits with the full Hamiltonian.

    The cumulant is a truncation, so the difference grows with time; what is
    tested is that it vanishes with the correct order in time, which is what
    fixes every term of the hierarchy.
    """
    deltas = np.array(deltas, float)
    chiN = 1.5
    times = np.array([0.05, 0.1, 0.2])
    u, cnt = np.unique(deltas, return_counts=True)
    sysm = System(delta=u, n=cnt.astype(float), chiN=chiN)
    co_c, xi_c = _cumulant(sysm, times)
    co_e, xi_e = full_exact(deltas, chiN, times)
    assert abs(co_c[0] - co_e[0]) < 1e-7
    assert abs(xi_c[0] - xi_e[0]) < 1e-4
    # the discrepancy must grow, not sit at a constant offset
    assert abs(xi_c[2] - xi_e[2]) > abs(xi_c[0] - xi_e[0])
    assert abs(xi_c[2] - xi_e[2]) < 1e-2


def test_coherent_state_has_unit_wineland_parameter():
    sysm = System(delta=np.linspace(-1, 1, 8), n=np.full(8, 100.0), chiN=1.0)
    assert abs(wineland_xi2(coherent_state_x(sysm), sysm) - 1.0) < 1e-12
    assert abs(coherence(coherent_state_x(sysm), sysm) - 1.0) < 1e-12


def test_free_classes_do_not_change_the_initial_state():
    """Adding far detuned classes leaves the initial values unchanged."""
    sysm = System(delta=np.zeros(1), n=np.array([90.0]), chiN=1.0,
                  spec_delta=np.array([-50.0, 50.0]), spec_n=np.array([5.0, 5.0]))
    assert abs(wineland_xi2(coherent_state_x(sysm), sysm) - 1.0) < 1e-12
    assert abs(coherence(coherent_state_x(sysm), sysm) - 1.0) < 1e-12


def test_structural_symmetries_of_the_equations():
    """The hierarchy must preserve the symmetry of each correlation block.

    P is Hermitian, Q and Z are symmetric under exchange of the two emitters,
    and Z is real, because each is a moment of commuting operators on distinct
    emitters.  These identities are exact, they hold term by term, and they are
    what a transposed index inside any of the partner sums destroys, so they
    catch transcription errors that agreement with a truncated reference at
    short time does not.
    """
    from lockkernel.cumulant import State, _rhs

    rng = np.random.default_rng(1)
    M, N, chiN = 5, 800.0, 3.0
    sysm = System(delta=np.array([-2.0, -0.7, 0.0, 0.7, 2.0]),
                  n=np.full(M, N / M), chiN=chiN)

    def rnd(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    A = 1e-3 * rnd((M, M))
    B = 1e-3 * rnd((M, M))
    C = 1e-3 * rnd((M, M))
    st = State(s=0.4 * rnd(M), z=0.3 * rng.normal(size=M).astype(complex),
               P=A + A.conj().T, Q=B + B.T, R=1e-3 * rnd((M, M)),
               Z=(C + C.T).real.astype(complex))
    d = State.unpack(_rhs(0.0, st.pack(), sysm.delta, sysm.n, sysm.chi1), M)
    scale = max(np.abs(d.Z).max(), np.abs(d.P).max(), np.abs(d.Q).max())
    assert np.abs(d.P - d.P.conj().T).max() < 1e-12 * scale
    assert np.abs(d.Q - d.Q.T).max() < 1e-12 * scale
    assert np.abs(d.Z - d.Z.T).max() < 1e-12 * scale
    assert np.abs(d.Z.imag).max() < 1e-12 * scale
