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

lags=40, rank=66 ARE VALIDATED DEFAULTS, not just the hand-picked values
they started as (see globalcva/sweep_rank_lags.py and the session notes
doc in examples/sixdrive12resp/results/ for the full sweep):
  - rank is a SHARP THRESHOLD, not a tunable knob: rank<=58 collapses (FRF
    rel_err 0.5-0.8, modes go missing, and the identified poles come back
    UNDER-damped -- a qualitatively different failure than the bias below).
    rank=66 snaps to good performance and is essentially flat all the way
    to rank=120, so 66 (=2x33 true modes) is not an arbitrary choice.
  - lags=40 is near-optimal, not just convenient: best error is around
    lags=30 (a wash vs 40), but past ~60 it degrades steadily -- by
    lags=100 FRF rel_err roughly DOUBLES. More embedding depth shrinks the
    number of independent non-overlapping past/future blocks available to
    the canonical-correlation order-selection step at fixed record length,
    so bigger lags is not free.
  - the rank=None auto heuristic (largest relative drop in the
    canonical-correlation spectrum) picks 62-65 across 0.5-4s of data --
    always a little under the true 66, but close enough that its FRF error
    matches the forced-rank=66 case within ~1-3%. Safe as an unattended
    fallback, but it sits close to (not comfortably inside) the rank<=58
    collapse threshold above, so treat it as validated-but-not-slack.

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
                rank_p=rank_p, rank_f=rank_f, Xk=Xk, Xkp1=Xkp1, y_k=y_k, u_k=u_k)


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


def _steady_state_kalman_gain(A, C, Q, R, S):
    """Solve the filter-form discrete algebraic Riccati equation for the
    steady-state a-priori error covariance P and Kalman gain K, given
        x(k+1) = A x(k) + w(k),   y(k) = C x(k) + e(k)
        cov([w; e]) = [[Q, S], [S.T, R]]

    scipy only ships the CONTROL-form solve_discrete_are:
        X = a.T X a - X - (a.T X b + s)(r + b.T X b)^-1(b.T X a + s.T) + q
    The filter Riccati is the dual of that with (a,b) <- (A.T, C.T):
        P = A P A.T + Q - (A P C.T + S)(C P C.T + R)^-1 (A P C.T + S).T
    which is exactly what calling solve_discrete_are(A.T, C.T, Q, R, s=S)
    returns (substitute a=A.T so a.conj().T=A and expand to check).

    Returns (K, P) with K in PREDICTOR form, matching
        x(k+1|k) = A x(k|k-1) + B u(k) + K (y(k) - C x(k|k-1) - D u(k))
    """
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A.T, C.T, Q, R, s=S)
    innovation_cov = C @ P @ C.T + R
    K = (A @ P @ C.T + S) @ np.linalg.inv(innovation_cov)
    return K, P


def global_cva_innovations(y, u, lags, tol=1e-8, rank=None, include_D=True,
                            refine_iters=1):
    """
    Combined deterministic-stochastic ("innovations-form") extension of
    global_cva_v2:
        x(k+1) = A x(k) + B u(k) + K e(k)
        y(k)   = C x(k) + D u(k) + e(k)

    global_cva_v2 fits a pure output-error (deterministic-only) model --
    all of y not explained by u through the state gets absorbed into the
    A,B,C,D least-squares residual, with no separate noise model. That
    costs nothing on noise-free data, but a noise-injection sweep
    (globalcva/noise_injection_sweep.py) showed it costs real accuracy
    under additive measurement noise: at >=1s of data, even 2% RMS sensor
    noise is enough for classical H1 to match or beat CVA, and the
    "2-4x less data than H1" headline result is a noise-free-only finding.

    IMPORTANT SCOPING NOTE: this does NOT re-derive the state extraction
    step (the oblique-projection theory underlying textbook N4SID's proof
    of consistency under noise). It keeps global_cva_v2's Xk = J @ P
    exactly as validated, and instead uses a subspace-then-Kalman-
    refinement scheme, which is a standard, legitimate two-stage
    identification pattern:
      1. Run global_cva_v2 once for an initial A,B,C,D and state sequence.
      2. Compute that fit's residuals -- w_k (process) and e_k (output,
         i.e. the innovation) -- and their sample covariances Q,R,S.
      3. Solve the filter-form DARE for the steady-state Kalman gain K.
      4. Run the resulting steady-state Kalman filter FORWARD through the
         actual (noisy) data to get a filtered state x_hat(k): a
         minimum-variance estimate given the model, which down-weights
         noisy measurements via K rather than treating every raw sample
         equally the way the static projection Xk=J@P does.
      5. Re-fit A,B,C,D on x_hat instead of Xk, and repeat 2-4 for
         refine_iters rounds.
    Note that merely computing K without refinement (refine_iters=0)
    changes nothing about the deterministic transfer function C(zI-A)^-1B
    + D -- K only describes the noise-to-state path. The refinement loop
    is what can actually change the FRF estimate.

    Parameters mirror global_cva_v2, plus:
      refine_iters : number of Kalman-refit rounds (0 = report K/Q/R/S
                     for the plain v2 fit without changing A,B,C,D at all).

    Returns everything global_cva_v2 returns, plus:
      K, Q, R, S, P   : Kalman gain / noise covariances / Riccati solution
                        (from the FINAL iteration)
      innovations     : final-iteration e_k sequence, (N-1, nout)
      x_hat           : final-iteration filtered state sequence, (N, rank)
      refine_history  : list of per-iteration dicts (max|eig| of the
                        refit A each round), so a caller can see whether
                        refinement is converging rather than trusting the
                        last iteration blindly
    """
    base = global_cva_v2(y, u, lags=lags, tol=tol, rank=rank, include_D=include_D)
    A, B, C, D = base['A'], base['B'], base['C'], base['D']
    Xk, Xkp1, u_k, y_k = base['Xk'], base['Xkp1'], base['u_k'], base['y_k']
    N, rnk = Xk.shape
    nin = u_k.shape[1]
    nout = y_k.shape[1]

    def residual_covariances(Xk_, Xkp1_, u_k_, y_k_, A_, B_, C_, D_):
        w = Xkp1_ - Xk_ @ A_.T - u_k_ @ B_.T
        e = y_k_ - Xk_ @ C_.T - u_k_ @ D_.T
        WE = np.hstack([w, e])
        WE = WE - WE.mean(axis=0, keepdims=True)
        n = WE.shape[0]
        cov = (WE.T @ WE) / n
        Q_ = cov[:rnk, :rnk]
        S_ = cov[:rnk, rnk:]
        R_ = cov[rnk:, rnk:]
        # tiny regularization: residual covariances from a finite fit
        # aren't guaranteed PD, and the DARE solver needs them to be
        Q_ = Q_ + 1e-10 * np.trace(Q_) / rnk * np.eye(rnk)
        R_ = R_ + 1e-10 * np.trace(R_) / nout * np.eye(nout)
        return Q_, R_, S_, w, e

    Q, R, S, w, e = residual_covariances(Xk, Xkp1, u_k, y_k, A, B, C, D)
    K, P = _steady_state_kalman_gain(A, C, Q, R, S)

    x_hat = Xk
    refine_history = []
    for it in range(refine_iters):
        # steady-state predictor-form Kalman filter, run forward through
        # the actual data; warm-started from the CVA state estimate so
        # the filter transient dies out fast instead of starting at zero
        xf = np.zeros((N, rnk))
        xf[0] = Xk[0]
        for k in range(N - 1):
            innov = y_k[k] - C @ xf[k] - D @ u_k[k]
            xf[k + 1] = A @ xf[k] + B @ u_k[k] + K @ innov
        x_hat = xf

        Z = np.hstack([x_hat[:-1], u_k[:-1]])
        AB, *_ = np.linalg.lstsq(Z, x_hat[1:], rcond=None)
        AB = AB.T
        A_new, B_new = AB[:, :rnk], AB[:, rnk:]
        if include_D:
            CD, *_ = np.linalg.lstsq(Z, y_k[:-1], rcond=None)
            CD = CD.T
            C_new, D_new = CD[:, :rnk], CD[:, rnk:]
        else:
            C_fit, *_ = np.linalg.lstsq(x_hat[:-1], y_k[:-1], rcond=None)
            C_new, D_new = C_fit.T, np.zeros((nout, nin))

        A, B, C, D = A_new, B_new, C_new, D_new
        Q, R, S, w, e = residual_covariances(x_hat[:-1], x_hat[1:], u_k[:-1], y_k[:-1],
                                              A, B, C, D)
        try:
            K, P = _steady_state_kalman_gain(A, C, Q, R, S)
        except Exception as exc:
            refine_history.append(dict(iter=it, ok=False, error=str(exc)))
            break
        refine_history.append(dict(iter=it, ok=True,
                                    max_eig=float(np.abs(np.linalg.eigvals(A)).max())))

    out = dict(base)
    out.update(A=A, B=B, C=C, D=D, K=K, Q=Q, R=R, S=S, P=P,
                innovations=e, x_hat=x_hat, refine_history=refine_history)
    return out
