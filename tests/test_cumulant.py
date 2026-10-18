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


# ---------------------------------------------------------------------------
# New in 1.3.0: class_exact, which had no test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deltas", [
    [-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0],
    [-1.2, -0.4, -0.4, -0.4, 0.4, 0.9, 0.9, 1.2],      # unequal classes
])
def test_class_exact_against_full_diagonalisation(deltas):
    """Emitters grouped by detuning stay in their classes' maximal spin
    manifolds, so class_exact must reproduce the full 2^N calculation."""
    from lockkernel.exact import class_exact

    deltas = np.array(deltas, float)
    times = np.array([0.05, 0.2, 1.0, 3.0])
    u, cnt = np.unique(deltas, return_counts=True)
    co_c, xi_c = class_exact(u, cnt, 1.5, times)
    co_e, xi_e = full_exact(deltas, 1.5, times)
    assert np.max(np.abs(co_c - co_e)) < 1e-10
    assert np.max(np.abs(xi_c - xi_e)) < 1e-9


def test_class_exact_times_in_any_order():
    """1.2.0 skipped a time earlier than the one before it and returned the
    later state: for times [1.0, 0.2] both entries were R = 0.677579, where
    full diagonalization gives 0.984316 at t = 0.2."""
    from lockkernel.exact import class_exact

    deltas = np.array([-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0])
    u, cnt = np.unique(deltas, return_counts=True)
    times = np.array([1.0, 0.2, 0.0, 0.5, 0.5])
    co_c, xi_c = class_exact(u, cnt, 1.5, times)
    co_e, xi_e = full_exact(deltas, 1.5, times)
    assert np.max(np.abs(co_c - co_e)) < 1e-10
    assert np.max(np.abs(xi_c - xi_e)) < 1e-9
    assert abs(co_c[2] - 1.0) < 1e-12           # t = 0 is the start


def test_class_exact_refusals():
    from lockkernel.exact import class_exact

    with pytest.raises(ValueError, match="whole numbers"):
        class_exact([0.0, 1.0], [2.5, 2.0], 1.0, [0.1])
    with pytest.raises(ValueError, match="equal length"):
        class_exact([0.0, 1.0], [2, 2, 2], 1.0, [0.1])


# ---------------------------------------------------------------------------
# New in 1.5.0: the faster full_exact against the 1.4.0 code, kept here
# ---------------------------------------------------------------------------

def _full_exact_1_4_0(deltas, chiN, times):
    """full_exact as in 1.4.0: dense Kronecker products, the exchange term
    as the N^2 products sigma+_j sigma-_k, the covariance from J_a J_b."""
    from lockkernel.exact import _wineland_from_moments
    deltas = np.asarray(deltas, float)
    N = len(deltas)
    dim = 2 ** N
    chi1 = chiN / N
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    sp = np.array([[0, 1], [0, 0]], complex)
    sm = sp.T.copy()

    def op(single, j):
        out = np.array([[1.0 + 0j]])
        for i in range(N):
            out = np.kron(out, single if i == j else np.eye(2))
        return out

    SP = [op(sp, j) for j in range(N)]
    SM = [op(sm, j) for j in range(N)]
    SZ = [op(sz, j) for j in range(N)]
    SX = [op(sx, j) for j in range(N)]
    SY = [op(sy, j) for j in range(N)]
    H = np.zeros((dim, dim), complex)
    for j in range(N):
        H += 0.5 * deltas[j] * SZ[j]
        for k in range(N):
            H += chi1 * (SP[j] @ SM[k])
    Js = [0.5 * sum(SX), 0.5 * sum(SY), 0.5 * sum(SZ)]
    w, V = np.linalg.eigh(H)
    psi0 = np.ones(dim, complex) / np.sqrt(dim)
    c0 = V.conj().T @ psi0
    contrast, xi2 = [], []
    for t in np.atleast_1d(times):
        psi = V @ (np.exp(-1j * w * t) * c0)
        mean = np.array([np.vdot(psi, A @ psi).real for A in Js])
        Cov = np.empty((3, 3))
        for a in range(3):
            for b in range(3):
                Cov[a, b] = 0.5 * np.vdot(
                    psi, (Js[a] @ Js[b] + Js[b] @ Js[a]) @ psi).real
        Cov -= np.outer(mean, mean)
        contrast.append(2.0 * np.hypot(mean[0], mean[1]) / N)
        xi2.append(_wineland_from_moments(mean, Cov, N))
    return np.array(contrast), np.array(xi2)


@pytest.mark.parametrize("deltas,chiN", [
    ([0.0], 1.0),
    ([-0.5, 0.7], 2.0),
    ([-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0], 1.5),
    ([-1.3, -0.2, 0.0, 0.1, 0.4, 0.9, 1.7, 2.2], 0.8),
])
def test_full_exact_same_as_1_4_0(deltas, chiN):
    """The same contrast R and Wineland parameter xi^2 as the 1.4.0 code.

    The two build the same Hamiltonian in a different order (chi1 J+ J- in
    one product against N^2 summed products), so H differs by rounding,
    of order N^2 * 1e-16 relative.  The propagated state then differs by
    about (eigensolver backward error ~ dim * 1e-16) * t ||H||, below
    1e-12 for these sizes and times, and so does R (an O(1) contrast):
    tolerance 1e-11.  xi^2 = N v_min / |J|^2 with |J| = N R / 2 is
    ill-conditioned when R is small (here R falls to 0.026 at t = 3, where
    xi^2 = 1297): its relative error is about 2 dR/R + dv/v, so the
    tolerance is 1e-11 xi^2 / R^2 (observed: below 3e-13 relative)."""
    times = np.array([0.0, 0.05, 0.7, 3.0, 11.0])
    R_new, xi_new = full_exact(deltas, chiN, times)
    R_old, xi_old = _full_exact_1_4_0(deltas, chiN, times)
    assert np.max(np.abs(R_new - R_old)) < 1e-11
    assert np.all(np.abs(xi_new - xi_old) < 1e-11 * xi_old / R_old ** 2)
    assert abs(R_new[0] - 1.0) < 1e-12          # t = 0: all along +x


def test_full_exact_refusals():
    with pytest.raises(ValueError, match="finite detunings"):
        full_exact([], 1.0, [0.1])
    with pytest.raises(ValueError, match="finite detunings"):
        full_exact([0.0, float("nan")], 1.0, [0.1])
    with pytest.raises(ValueError, match="finite"):
        full_exact([0.0, 1.0], float("inf"), [0.1])
