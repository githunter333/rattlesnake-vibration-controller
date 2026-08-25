"""
global_cva_frf.py -- global/batch (linear, non-local) CVA subspace
realization + FRF extraction for the 6-drive/12-response/8-control frame
system. Companion: validate_global_cva.py.

VALIDATED against the system's known ground-truth FRF (frf_frame6x12_H.npz):
with lags=40, rank=66 on 2 s of broadband data it recovers 33/33 true modes
to within 2%, all poles stable (max|eig| = 0.994), FRF relative error 0.098
over 100-1000 Hz. Against classical Welch/H1 at matched record length:

    duration   CVA rel_err   H1 rel_err
      0.25 s      0.423        0.729
      0.50 s      0.228        0.556
      1.00 s      0.141        0.197
      2.00 s      0.098        0.126
      4.00 s      0.080        0.101

i.e. CVA reaches a given accuracy on roughly 2-4x less data than H1 in the
short-record regime -- which is the point: a usable FRF without waiting to
average many blocks.

An earlier attempt at this (rank from rank(Spp), past/future shifted by one
sample, no D term) failed badly -- unstable poles up to |eig|=30, FRF error
20-50x. The two structural fixes that made it work:

1. NON-OVERLAPPING past/future for the canonical-correlation step, but the
   state sequence for the A,B regression built by applying the SAME
   projection J to past rows ONE SAMPLE apart.

   Why: cvaonestep's rank comes from rank(Spp) over a ~960-row NEAREST-
   NEIGHBOR subset. Locally the trajectory lies on a low-dimensional
   manifold, so Spp really is near-rank-deficient and the SVD tolerance
   recovers the true local order. Globally, over all ~10k sequential rows,
   the past matrix spans the whole state space -- Spp is legitimately
   high-rank and the tolerance never finds the true order. So in a global
   method the order must come from the CANONICAL CORRELATIONS instead --
   and those only carry information if past and future do NOT overlap
   (with block=1 they are all ~1.0, which is what the first attempt saw).

   But non-overlapping blocks must NOT be used to define the state
   transition directly (that would make A advance by `lags` samples). The
   classical construction: x(tau) = J @ P[tau], x(tau+1) = J @ P[tau+1],
   where P is the past Hankel matrix -- consecutive ROWS of P are one
   sample apart, so A is a genuine one-sample transition.

2. DIRECT FEEDTHROUGH D. These are ACCELERATION outputs; the true system
   has D_accel = phi_response @ M^-1 @ phi_excitation.T != 0 (acceleration
   responds instantaneously to force). H = C(zI-A)^-1 B with no +D
   structurally cannot match, especially at high frequency.

Index convention (delaybuild: row r of the Hankel matrix covers samples
[r, r+L-1]):
    past row i     -> samples [i,     i+L-1],  state time tau = i+L
    future row i   -> samples [i+L, i+2L-1]  = Yh[i+L]      (adjacent, no overlap)
    x(tau)   = J @ [Yh[i],   Uh[i]]
    x(tau+1) = J @ [Yh[i+1], Uh[i+1]]
    u(tau), y(tau) = u[:, i+L], y[:, i+L]
"""
import numpy as np
from scipy.linalg import svd as scipy_svd


def delaybuild(data, nch, lags):
    """data: (nch, nsamples). Returns (nsamples-lags+1, nch*lags), lag-major."""
    nsamples = data.shape[1]
    nrows = nsamples - lags + 1
    delayed = np.zeros((nrows, nch * lags))
    for ch in range(nch):
        for lag in range(lags):
            delayed[:, lag * nch + ch] = data[ch, lag:lag + nrows]
    return delayed


def _whiten(M, tol):
    """SVD whitening: returns W with W @ M @ W.T ~ I on the retained subspace."""
    U, s, Vt = scipy_svd(M, full_matrices=False)
    rank = int(np.sum((s / s[0]) > tol)) if (len(s) and s[0] > 0) else 0
    inv_sqrt = np.zeros(len(s))
    inv_sqrt[:rank] = 1.0 / np.sqrt(s[:rank])
    return inv_sqrt[:, None] * Vt, rank


def global_cva_v2(y, u, lags, tol=1e-8, rank=None, include_D=True):
    """
    y: (nout, nsamples) response time history
    u: (nin, nsamples) drive time history
    lags: L, depth of BOTH the past and the (adjacent, non-overlapping) future block
    tol: whitening rank tolerance (loose here on purpose -- the ORDER is
         chosen from the canonical correlations, not from rank(Spp))
    rank: system order. If None, chosen from the canonical-correlation
          spectrum via a largest-relative-drop heuristic.

    Returns dict with A, B, C, D, rank, cc (canonical correlations), etc.
    """
    nout, nsamples = y.shape
    nin = u.shape[0]
    L = lags

    Yh = delaybuild(y, nout, L)
    Uh = delaybuild(u, nin, L)
    nrows = Yh.shape[0]

    N = nrows - L - 1
    if N < 10:
        raise ValueError(f"not enough samples: N={N}")

    P = np.hstack([Yh[:N], Uh[:N]])        # past: outputs AND inputs
    F = Yh[L:L + N]                         # future outputs, adjacent (non-overlapping)

    # --- canonical correlations between past and future ---
    Spp = P.T @ P / N
    Sff = F.T @ F / N
    Spf = P.T @ F / N
    Spp += 1e-12 * np.trace(Spp) / Spp.shape[0] * np.eye(Spp.shape[0])
    Sff += 1e-12 * np.trace(Sff) / Sff.shape[0] * np.eye(Sff.shape[0])

    Wp, rank_p = _whiten(Spp, tol)
    Wf, rank_f = _whiten(Sff, tol)

    G = Wp @ Spf @ Wf.T
    Ug, cc, Vgt = scipy_svd(G, full_matrices=False)   # cc = canonical correlations

    if rank is None:
        # order = largest relative drop in the canonical-correlation spectrum
        cc_pos = cc[cc > 1e-12]
        n_max = min(len(cc_pos) - 1, 2 * nout * L)
        ratios = cc_pos[:n_max] / np.maximum(cc_pos[1:n_max + 1], 1e-30)
        rank = int(np.argmax(ratios)) + 1
    rnk = int(rank)

    # projection from past onto the state
    J = (Ug[:, :rnk].T) @ Wp          # (rnk, ncols_P)

    Pnext = np.hstack([Yh[1:N + 1], Uh[1:N + 1]])
    Xk = P @ J.T                       # x(tau),   tau = i+L
    Xkp1 = Pnext @ J.T                 # x(tau+1)

    tau = np.arange(L, L + N)
    u_k = u[:, tau].T                  # (N, nin)
    y_k = y[:, tau].T                  # (N, nout)

    # --- A, B:  x(tau+1) = A x(tau) + B u(tau) ---
    Z = np.hstack([Xk, u_k])
    AB, *_ = np.linalg.lstsq(Z, Xkp1, rcond=None)
    AB = AB.T
    A = AB[:, :rnk]
    B = AB[:, rnk:]

    # --- C, D:  y(tau) = C x(tau) + D u(tau)  (D matters: accel feedthrough) ---
    if include_D:
        CD, *_ = np.linalg.lstsq(Z, y_k, rcond=None)
        CD = CD.T
        C = CD[:, :rnk]
        D = CD[:, rnk:]
    else:
        C, *_ = np.linalg.lstsq(Xk, y_k, rcond=None)
        C = C.T
        D = np.zeros((nout, nin))

    return dict(A=A, B=B, C=C, D=D, rank=rnk, cc=cc,
                rank_p=rank_p, rank_f=rank_f, Xk=Xk, y_k=y_k, u_k=u_k)


def frf_from_ss(A, B, C, D, freqs_hz, dt):
    """H(f) = C (zI - A)^-1 B + D,  z = exp(i*2*pi*f*dt)."""
    n = A.shape[0]
    I = np.eye(n)
    H = np.zeros((len(freqs_hz), C.shape[0], B.shape[1]), dtype=complex)
    for i, f in enumerate(freqs_hz):
        z = np.exp(1j * 2 * np.pi * f * dt)
        H[i] = C @ np.linalg.solve(z * I - A, B) + D
    return H


def modes_from_A(A, dt, f_lo=20.0, f_hi=None, zeta_max=0.5):
    """Discrete A -> (freq_hz, zeta) for physically plausible, stable poles."""
    ev = np.linalg.eigvals(A)
    s = np.log(ev.astype(complex)) / dt
    freq = np.abs(np.imag(s)) / (2 * np.pi)
    zeta = -np.real(s) / np.maximum(np.abs(s), 1e-30)
    if f_hi is None:
        f_hi = 0.5 / dt
    good = (np.imag(ev) > 0) & (freq > f_lo) & (freq < f_hi) & \
           (zeta > 0) & (zeta < zeta_max) & (np.abs(ev) <= 1.0 + 1e-9)
    order = np.argsort(freq[good])
    return freq[good][order], zeta[good][order], np.abs(ev).max()
