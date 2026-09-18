"""Second order cumulant dynamics of the conservative model.

MODEL.  Spin one half emitters with uniform all to all exchange and an
inhomogeneous distribution of detunings,

    H = chi1 sum_{j,k} sigma+_j sigma-_k + sum_j (delta_j / 2) sigma z_j,
    chi1 = chiN / N.

There is no damping, no drive and no thermal bath: the evolution is unitary.
This is the quantum model whose mean field limit obeys the locking law of
`lockkernel.parametric`.

VARIABLES.  Emitters are grouped into classes of equal detuning.  Within a
class every emitter is equivalent, so the state is carried by the class means

    s_m = <sigma+_a>,        z_m = <sigma z_a>,        a in class m,

and by the CONNECTED pair correlations of two DISTINCT emitters a in class m
and b in class n,

    P_mn = <sigma+_a sigma-_b> - s_m conj(s_n),
    Q_mn = <sigma+_a sigma+_b> - s_m s_n,
    R_mn = <sigma z_a sigma+_b> - z_m s_n,
    Z_mn = <sigma z_a sigma z_b> - z_m z_n.

Connected correlations are of order 1/N while the raw moments are of order
one, so evolving them directly avoids the cancellation that makes raw moments
useless beyond N of order 1e8.  The hierarchy is closed at third order by the
Gaussian rule <XYB>_c = <X><YB>_c + <Y><XB>_c.

FAR DETUNED EMITTERS.  Emitters whose detuning is far outside the locking
bandwidth are propagated exactly as free spins rather than integrated.  Their
population is retained in full, so the line is never truncated; only their
interaction, whose relative size is (chiN/delta)^2, is dropped.  This removes
the stiffness of the far tail from the differential equation.

VALIDATION.  `tests/test_cumulant.py` checks this solver against exact
diagonalisation of the full Hamiltonian for small emitter numbers with several
distinct detunings, and against the closed form of one axis twisting on
resonance.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["System", "State", "coherent_state_x", "evolve", "evolve_meanfield",
           "wineland_xi2", "coherence", "collective_moments"]


@dataclasses.dataclass
class System:
    """Classes, populations and coupling.

    Attributes
    ----------
    delta : (M,) float
        Detuning of each interacting class.
    n : (M,) float
        Population of each interacting class.
    chiN : float
        Coupling per emitter times the total emitter number.
    spec_delta, spec_n : (K,) float
        Detuning and population of the free, far detuned classes.
    """

    delta: np.ndarray
    n: np.ndarray
    chiN: float
    spec_delta: np.ndarray = None
    spec_n: np.ndarray = None

    def __post_init__(self):
        self.delta = np.asarray(self.delta, float)
        self.n = np.asarray(self.n, float)
        if self.spec_delta is None:
            self.spec_delta = np.zeros(0)
            self.spec_n = np.zeros(0)
        self.spec_delta = np.asarray(self.spec_delta, float)
        self.spec_n = np.asarray(self.spec_n, float)

    @property
    def M(self) -> int:
        return len(self.delta)

    @property
    def K(self) -> int:
        return len(self.spec_delta)

    @property
    def N(self) -> float:
        """Total emitter number, interacting plus free."""
        return float(self.n.sum() + self.spec_n.sum())

    @property
    def chi1(self) -> float:
        return float(self.chiN) / self.N


@dataclasses.dataclass
class State:
    """Class means and connected pair correlations."""

    s: np.ndarray
    z: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    vs: np.ndarray = None  # (K,3) Bloch vectors of the free classes

    def __post_init__(self):
        if self.vs is None:
            self.vs = np.zeros((0, 3))

    @property
    def M(self):
        return len(self.s)

    def pack(self):
        return np.concatenate([self.s, self.z.astype(complex),
                               self.P.ravel(), self.Q.ravel(),
                               self.R.ravel(), self.Z.ravel()])

    @classmethod
    def unpack(cls, y, M, vs=None):
        b = y[2 * M:].reshape(4, M, M)
        return cls(s=y[:M], z=y[M:2 * M], P=b[0], Q=b[1], R=b[2], Z=b[3], vs=vs)


def coherent_state_x(sys: System) -> State:
    """All emitters along +x: a coherent spin state, the standard start."""
    M, K = sys.M, sys.K
    zero = np.zeros((M, M), complex)
    return State(np.full(M, 0.5, complex), np.zeros(M, complex),
                 zero.copy(), zero.copy(), zero.copy(), zero.copy(),
                 np.tile(np.array([1.0, 0.0, 0.0]), (K, 1)))


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def _rhs(t, y, delta, n, chi1):
    """Right hand side in connected variables.

    Every sum over partners excludes the emitters that already carry a free
    index, which is what the `_ex` helper subtracts.
    """
    M = len(delta)
    st = State.unpack(y, M)
    s, z, P, Q, R, Z = st.s, st.z, st.P, st.Q, st.R, st.Z
    c = 1j * chi1
    cc = -1j * chi1
    A = 1j * (delta + chi1)          # free precession of sigma-
    Ac = np.conj(A)
    sc = np.conj(s)
    Rc = np.conj(R)
    ns = n @ s                       # sum_p n_p s_p

    # ---- first moments -----------------------------------------------------
    S_zp = (R @ n - np.diag(R)) + z * (ns - s)
    ds = -Ac * s + cc * S_zp
    S_pm = (P @ n - np.diag(P)) + s * np.conj(ns - s)
    dz = -4.0 * np.real(c * S_pm)

    # ---- shorthands --------------------------------------------------------
    zm, zn = z[:, None], z[None, :]
    sm, sn = s[:, None], s[None, :]
    scm, scn = sc[:, None], sc[None, :]

    def _ex(full, at_m, at_n):
        """sum_p n_p T(p) with the terms p = m and p = n removed."""
        return full - at_m - at_n

    # ---- P -----------------------------------------------------------------
    S1 = _ex(zm * (n @ P)[None, :] + Rc * ns,
             zm * P + Rc * sm,
             zm * np.diag(P)[None, :] + Rc * sn)
    S2 = _ex(zn * (P @ n)[:, None] + R.T * np.conj(ns),
             zn * np.diag(P)[:, None] + R.T * scm,
             zn * P + R.T * scn)
    src_a = 0.5 * (zm + zm * zn + Z) - R * scn - zm * (sn * scn)
    src_b = 0.5 * (zn + zm * zn + Z) - sm * np.conj(R.T) - zn * (sm * scm)
    dP = (-(Ac[:, None] + A[None, :]) * P
          + cc * (src_a + S1)
          + c * (src_b + S2))

    # ---- Q -----------------------------------------------------------------
    S3 = _ex(zm * (n @ Q)[None, :] + R * ns,
             zm * Q + R * sm,
             zm * np.diag(Q)[None, :] + R * sn)
    S4 = _ex(zn * (Q @ n)[:, None] + R.T * ns,
             zn * np.diag(Q)[:, None] + R.T * sm,
             zn * Q + R.T * sn)
    dQ = (-(Ac[:, None] + Ac[None, :]) * Q
          + cc * (-(R + zm * sn) * sn + S3)
          + cc * (-sm * (R.T + zn * sm) + S4))

    # ---- R -----------------------------------------------------------------
    S5 = _ex(c * (sm * (P @ n)[None, :] + Q * np.conj(ns))
             + cc * (P.T * ns + scm * (n @ Q)[None, :]),
             c * (sm * P.T + Q * scm) + cc * (P.T * sm + scm * Q),
             c * (sm * np.diag(P)[None, :] + Q * scn)
             + cc * (P.T * sn + scm * np.diag(Q)[None, :]))
    S7 = _ex(zn * (R @ n)[:, None] + Z * ns,
             zn * np.diag(R)[:, None] + Z * sm,
             zn * R + Z * sn)
    P_mn = P + sm * scn
    P_nm = P.T + sn * scm
    src_za = c * (0.5 * (sm - R.T - sm * zn) - P_mn * sn) + cc * (-P_nm * sn)
    dR = (-2.0 * src_za
          - 2.0 * S5
          - Ac[None, :] * R
          + cc * (1.0 - zm) * (R.T + sm * zn)
          + cc * S7)

    # ---- Z -----------------------------------------------------------------
    S8 = _ex(c * (sm * (Rc @ n)[None, :] + R.T * np.conj(ns))
             + cc * (Rc.T * ns + scm * (R @ n)[None, :]),
             c * (sm * Rc.T + R.T * scm) + cc * (Rc.T * sm + scm * R.T),
             c * (sm * np.diag(Rc)[None, :] + R.T * scn)
             + cc * (Rc.T * sn + scm * np.diag(R)[None, :]))
    S9 = _ex(c * (sn * (Rc @ n)[:, None] + R * np.conj(ns))
             + cc * (Rc * ns + scn * (R @ n)[:, None]),
             c * (sn * np.diag(Rc)[:, None] + R * scm)
             + cc * (Rc * sm + scn * np.diag(R)[:, None]),
             c * (sn * Rc + R * scn) + cc * (Rc * sn + scn * R))
    src_zz_a = c * P_mn * (1.0 - zn) - cc * P_nm * (1.0 + zn)
    src_zz_b = -c * P_nm * (1.0 + zm) + cc * P_mn * (1.0 - zm)
    dZ = (-2.0 * (src_zz_a + src_zz_b) - 2.0 * S8 - 2.0 * S9)

    return State(ds, dz, dP, dQ, dR, dZ).pack()


def _rhs_meanfield(t, y, delta, n, chi1):
    """Mean field limit: the same equations with all correlations set to zero."""
    M = len(delta)
    s, z = y[:M], y[M:]
    c = 1j * chi1
    A = 1j * (delta + chi1)
    ns = n @ s
    ds = -np.conj(A) * s + np.conj(c) * z * (ns - s)
    dz = -4.0 * np.real(c * s * np.conj(ns - s))
    return np.concatenate([ds, dz])


def _free(vs, spec_delta, tau):
    """Exact free precession of the far detuned classes."""
    if vs.shape[0] == 0:
        return vs
    ph = np.asarray(spec_delta) * tau
    x, y, z = vs[:, 0], vs[:, 1], vs[:, 2]
    return np.stack([x * np.cos(ph) - y * np.sin(ph),
                     y * np.cos(ph) + x * np.sin(ph), z], axis=1)


def evolve(st: State, sys: System, t: float, t_eval=None, rtol=1e-8,
           atol=None, method="DOP853"):
    """Evolve the cumulant state.  Returns a State, or a list at `t_eval`."""
    if atol is None:
        atol = 1e-10 / max(sys.N, 1.0)
    args = (sys.delta, sys.n, sys.chi1)
    sol = solve_ivp(_rhs, (0.0, t), st.pack(), args=args, t_eval=t_eval,
                    rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    if t_eval is None:
        return State.unpack(sol.y[:, -1], sys.M, _free(st.vs, sys.spec_delta, t))
    return [State.unpack(sol.y[:, k], sys.M,
                         _free(st.vs, sys.spec_delta, sol.t[k]))
            for k in range(sol.y.shape[1])]


def evolve_meanfield(sys: System, t: float, t_eval=None, rtol=1e-8, atol=1e-11,
                     method="DOP853"):
    """Mean field evolution from the coherent spin state along +x.

    Returns (s, z) arrays with a column per requested time.
    """
    y0 = np.concatenate([np.full(sys.M, 0.5, complex), np.zeros(sys.M, complex)])
    sol = solve_ivp(_rhs_meanfield, (0.0, t), y0, args=(sys.delta, sys.n, sys.chi1),
                    t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:sys.M, :], sol.y[sys.M:, :]


# ---------------------------------------------------------------------------
# Collective observables
# ---------------------------------------------------------------------------

def _cartesian(st: State):
    """Bloch vectors and the connected correlation tensor in Cartesian form."""
    s, z, P, Q, R, Z = st.s, st.z, st.P, st.Q, st.R, st.Z
    v = np.stack([2 * s.real, 2 * s.imag, z.real], axis=1)
    M = st.M
    C = np.empty((M, M, 3, 3), complex)
    Qc, PT = np.conj(Q), P.T
    C[:, :, 0, 0] = Q + P + PT + Qc
    C[:, :, 0, 1] = -1j * (Q - P + PT - Qc)
    C[:, :, 1, 0] = -1j * (Q + P - PT - Qc)
    C[:, :, 1, 1] = -(Q - P - PT + Qc)
    C[:, :, 0, 2] = 2 * R.T.real
    C[:, :, 1, 2] = 2 * R.T.imag
    C[:, :, 2, 0] = 2 * R.real
    C[:, :, 2, 1] = 2 * R.imag
    C[:, :, 2, 2] = Z
    return v, C


def collective_moments(st: State, sys: System):
    """Mean and symmetrised covariance of J = (1/2) sum_a sigma_a."""
    n = sys.n
    v, C = _cartesian(st)
    J = 0.5 * (n @ v)
    pair = np.einsum("m,n,mnab->ab", n, n, C) - np.einsum("m,mmab->ab", n, C)
    same = np.sum(n) * np.eye(3) - np.einsum("m,ma,mb->ab", n, v, v)
    Cov = 0.25 * (pair.real + same)
    S1 = float(n.sum())
    if sys.K:
        sn = sys.spec_n
        J = J + 0.5 * (sn @ st.vs)
        Cov = Cov + 0.25 * (np.sum(sn) * np.eye(3)
                            - np.einsum("k,ka,kb->ab", sn, st.vs, st.vs))
        S1 += float(sn.sum())
    return J, 0.5 * (Cov + Cov.T), S1


def _transverse(J, Cov):
    """Smallest and largest variance in the plane perpendicular to J."""
    Jn = np.linalg.norm(J)
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.stack([e1, e2], axis=1)
    vals = np.linalg.eigvalsh(B.T @ Cov @ B)
    return vals[0], vals[1], Jn


def wineland_xi2(st: State, sys: System):
    """Wineland spin squeezing parameter xi_R^2 = N var_min / |<J>|^2.

    Equal to one for a coherent spin state.  A value below one certifies
    metrological entanglement and a phase sensitivity beyond the standard
    quantum limit.
    """
    J, Cov, S1 = collective_moments(st, sys)
    if np.linalg.norm(J) == 0:
        return np.inf
    vmin, _, Jn = _transverse(J, Cov)
    return float(vmin * S1 / Jn ** 2)


def coherence(st: State, sys: System) -> float:
    """Transverse contrast 2 |<J_perp>| / N, the order parameter R."""
    J, _, S1 = collective_moments(st, sys)
    return float(2.0 * np.hypot(J[0], J[1]) / S1)


def physicality(st: State, sys: System):
    """Diagnostics that say whether the truncated state is still physical.

    A second order cumulant expansion is not guaranteed to stay inside the set
    of physical states, and in an undamped model the truncated hierarchy is
    secularly unstable: the connected correlations, which should stay of order
    1/N, eventually grow and the reconstructed covariance loses positivity.
    Two exact properties of any true state are therefore monitored.

    * The collective covariance must be positive semidefinite.
    * Every class Bloch vector must lie inside the unit sphere.

    Returns (min_eigenvalue_of_covariance, max_bloch_length).  The state is
    treated as physical while the first is not negative and the second does
    not exceed one.
    """
    _, Cov, _ = collective_moments(st, sys)
    v, _ = _cartesian(st)
    return (float(np.linalg.eigvalsh(Cov)[0]),
            float(np.max(np.linalg.norm(v.real, axis=1))))


def valid_window(states, sys: System, bloch_tol=1e-6):
    """Index of the last state that is still physical, walking forward in time.

    Once the truncation has failed it does not recover, so the window is the
    leading run of physical states rather than the set of all of them.
    """
    last = -1
    for k, st in enumerate(states):
        mev, vmax = physicality(st, sys)
        if mev < 0.0 or vmax > 1.0 + bloch_tol:
            break
        last = k
    return last
