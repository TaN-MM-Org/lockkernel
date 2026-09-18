"""Exact references used to validate the cumulant solver.

Two independent references are provided.

DEGENERATE LINE.  When every detuning is equal the Hamiltonian reduces, on the
maximal spin manifold that the coherent spin state occupies, to

    H = chi1 S+ S- = chi1 (S^2 - S_z^2 + S_z),

which is diagonal in the S_z basis.  The evolution of the coherent spin state
is therefore exact and cheap for any emitter number: it is one axis twisting
in disguise.  `symmetric_exact` uses this.

INHOMOGENEOUS LINE.  With several distinct detunings the maximal spin manifold
is no longer closed, and the reference is exact diagonalisation of the full
Hamiltonian in the 2^N dimensional Hilbert space.  `full_exact` does this for
small emitter numbers, which is enough to test the structure of the cumulant
equations term by term.
"""
from __future__ import annotations

import numpy as np

__all__ = ["symmetric_exact", "full_exact"]


def _wineland_from_moments(J, Cov, N):
    Jn = np.linalg.norm(J)
    if Jn == 0:
        return np.inf
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.stack([e1, e2], axis=1)
    vals = np.linalg.eigvalsh(B.T @ Cov @ B)
    return float(vals[0] * N / Jn ** 2)


def symmetric_exact(N: int, chiN: float, times):
    """Exact contrast and Wineland parameter for a degenerate line.

    The state stays in the maximal spin manifold, of dimension N+1, and the
    Hamiltonian is diagonal there, so this is exact to machine precision at
    any emitter number.

    Returns (contrast, xi2), each an array over `times`.
    """
    S = N / 2.0
    m = np.arange(-S, S + 1)                      # S_z eigenvalues
    chi1 = chiN / N
    E = chi1 * (S * (S + 1) - m ** 2 + m)         # eigenvalues of chi1 S+ S-

    # Coherent spin state along +x in the S_z basis: binomial amplitudes.
    k = np.arange(N + 1)
    logc = 0.5 * (np.array([_log_binom(N, kk) for kk in k]) - N * np.log(2.0))
    amp = np.exp(logc)                            # real and positive

    sp = np.sqrt(S * (S + 1) - m[:-1] * (m[:-1] + 1))   # <m+1|S+|m>
    contrast, xi2 = [], []
    for t in np.atleast_1d(times):
        psi = amp * np.exp(-1j * E * t)
        # <S+> = sum_m conj(psi_{m+1}) sp_m psi_m
        Sp = np.sum(np.conj(psi[1:]) * sp * psi[:-1])
        Sz = np.sum(np.abs(psi) ** 2 * m)
        Sz2 = np.sum(np.abs(psi) ** 2 * m ** 2)
        # <S+ S+>
        spp = sp[:-1] * sp[1:]
        Spp = np.sum(np.conj(psi[2:]) * spp * psi[:-2])
        # <S+ S-> = <S^2 - S_z^2 + S_z>
        Spm = S * (S + 1) - Sz2 + Sz
        # <S_z S+ + S+ S_z>
        SzSp = np.sum(np.conj(psi[1:]) * m[1:] * sp * psi[:-1])
        SpSz = np.sum(np.conj(psi[1:]) * sp * m[:-1] * psi[:-1])

        Sx = Sp.real
        Sy = Sp.imag
        Sxx = 0.5 * (Spp.real + Spm.real)          # (S+^2 + S-^2 + S+S- + S-S+)/4
        Syy = 0.5 * (-Spp.real + Spm.real)
        Sxy = 0.5 * Spp.imag                       # symmetrised
        Sxz = 0.5 * (SzSp + SpSz).real
        Syz = 0.5 * (SzSp + SpSz).imag
        J = np.array([Sx, Sy, Sz.real])
        Cov = np.array([[Sxx - Sx ** 2, Sxy - Sx * Sy, Sxz - Sx * Sz.real],
                        [Sxy - Sx * Sy, Syy - Sy ** 2, Syz - Sy * Sz.real],
                        [Sxz - Sx * Sz.real, Syz - Sy * Sz.real, Sz2 - Sz.real ** 2]])
        Cov = 0.5 * (Cov + Cov.T).real
        contrast.append(2.0 * np.hypot(Sx, Sy) / N)
        xi2.append(_wineland_from_moments(J, Cov, N))
    return np.array(contrast), np.array(xi2)


def _log_binom(n, k):
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def full_exact(deltas, chiN: float, times):
    """Exact diagonalisation of the full Hamiltonian for a small ensemble.

    `deltas` lists the detuning of every individual emitter, so the emitter
    number is len(deltas) and the Hilbert space has 2^N dimensions.  Practical
    up to about twelve emitters.

    Returns (contrast, xi2), each an array over `times`.
    """
    deltas = np.asarray(deltas, float)
    N = len(deltas)
    dim = 2 ** N
    chi1 = chiN / N

    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    sp = np.array([[0, 1], [0, 0]], complex)      # |up><down|
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

    Jx = 0.5 * sum(SX)
    Jy = 0.5 * sum(SY)
    Jz = 0.5 * sum(SZ)

    w, V = np.linalg.eigh(H)
    psi0 = np.ones(dim, complex) / np.sqrt(dim)   # all emitters along +x
    c0 = V.conj().T @ psi0

    contrast, xi2 = [], []
    Js = [Jx, Jy, Jz]
    for t in np.atleast_1d(times):
        psi = V @ (np.exp(-1j * w * t) * c0)
        mean = np.array([np.vdot(psi, A @ psi).real for A in Js])
        Cov = np.empty((3, 3))
        for a in range(3):
            for b in range(3):
                Cov[a, b] = 0.5 * np.vdot(psi, (Js[a] @ Js[b] + Js[b] @ Js[a]) @ psi).real
        Cov -= np.outer(mean, mean)
        contrast.append(2.0 * np.hypot(mean[0], mean[1]) / N)
        xi2.append(_wineland_from_moments(mean, Cov, N))
    return np.array(contrast), np.array(xi2)


def class_exact(deltas, populations, chiN: float, times, tol=1e-9):
    """Exact dynamics of an inhomogeneous ensemble in the collective spin basis.

    Emitters within a class are exchange symmetric and start in a symmetric
    state, so each class stays in its maximal spin manifold of dimension
    n_j + 1.  The Hilbert space is therefore the product of those manifolds,
    of dimension prod(n_j + 1) rather than 2^N, which puts exact dynamics of a
    genuinely inhomogeneous ensemble within reach for emitter numbers of a few
    tens.  This is the reference that fixes how far the cumulant expansion can
    be trusted at finite N.

    Returns (contrast, xi2), each an array over `times`.
    """
    import numpy as _np
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply

    deltas = _np.asarray(deltas, float)
    pops = _np.asarray(populations, int)
    M = len(pops)
    N = int(pops.sum())
    chi1 = chiN / N
    dims = pops + 1

    def spin_ops(n):
        """S+, Sz for a spin of length n/2, in the S_z basis, as sparse."""
        S = n / 2.0
        m = _np.arange(-S, S + 1)
        d = n + 1
        up = _np.sqrt(S * (S + 1) - m[:-1] * (m[:-1] + 1))
        Sp = sparse.diags(up, offsets=-1, shape=(d, d), format="csr")  # |m+1><m|
        Sz = sparse.diags(m, format="csr")
        return Sp, Sz

    def embed(op, j):
        out = sparse.identity(1, format="csr")
        for i in range(M):
            out = sparse.kron(out, op if i == j else sparse.identity(dims[i]),
                              format="csr")
        return out

    SP, SZ = [], []
    for j in range(M):
        sp, sz = spin_ops(pops[j])
        SP.append(embed(sp, j))
        SZ.append(embed(sz, j))

    Jp = sum(SP)
    Jz = sum(SZ)
    H = chi1 * (Jp @ Jp.getH()) + sum(deltas[j] * SZ[j] for j in range(M))
    H = ((H + H.getH()) * 0.5).tocsr()

    Jx = 0.5 * (Jp + Jp.getH())
    Jy = -0.5j * (Jp - Jp.getH())
    Js = [Jx.tocsr(), Jy.tocsr(), Jz.tocsr()]

    # coherent spin state along +x: a product of x polarised class states
    psi = _np.ones(1, complex)
    for j in range(M):
        n = pops[j]
        k = _np.arange(n + 1)
        amp = _np.exp(0.5 * (_np.array([_log_binom(n, kk) for kk in k])
                             - n * _np.log(2.0)))
        psi = _np.kron(psi, amp)
    psi /= _np.linalg.norm(psi)

    contrast, xi2 = [], []
    prev = 0.0
    state = psi
    for t in _np.atleast_1d(times):
        if t > prev:
            state = expm_multiply(-1j * (t - prev) * H, state)
            prev = t
        mean = _np.array([_np.vdot(state, A @ state).real for A in Js])
        Cov = _np.empty((3, 3))
        for a in range(3):
            va = Js[a] @ state
            for b in range(a, 3):
                vb = Js[b] @ state
                Cov[a, b] = Cov[b, a] = _np.vdot(va, vb).real
        Cov = 0.5 * (Cov + Cov.T) - _np.outer(mean, mean)
        contrast.append(2.0 * _np.hypot(mean[0], mean[1]) / N)
        xi2.append(_wineland_from_moments(mean, Cov, N))
    return _np.array(contrast), _np.array(xi2)
