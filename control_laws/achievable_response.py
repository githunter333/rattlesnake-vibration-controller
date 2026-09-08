"""
achievable_response.py

The "pseudo control law": given an FRF, what is the BEST response auto-spectrum
any control law could possibly produce?  This does not drive hardware.  It
answers, per frequency line, the question every control-law comparison needs a
denominator for:

    "How much of my response error is the control law's fault, and how much is
     simply not reachable with this structure and these shakers?"

WHY THIS IS A REAL QUESTION
---------------------------
With N drives and M control responses, the commanded drive CPSD X must be a
valid (positive-semidefinite) cross-spectral matrix, and the response is
Y = H X H^H.  Only diag(Y) is specified.  The set of achievable response
auto-spectra is therefore

    D = { diag(H X H^H) : X >= 0 }

which is the image of the PSD cone under a linear map -- a convex cone in R^M.
When M > N (here 8 responses from 6 drives) that cone is a PROPER subset of the
positive orthant, so a perfectly reasonable-looking flat specification can be
partly or wholly unreachable, at some frequency lines and not others.  No
control law, however clever, gets outside D.  Chasing a target outside it is
exactly how an integrating law winds up against its own anti-windup ceiling.

WHAT IS COMPUTED
----------------
Per frequency line, over X >= 0:

  best_rms_db      min sqrt(mean_m (10 log10(a_m / y_m))^2)      -- best achievable
                                                                    RMS dB error
  best_minimax_db  min max_m |10 log10(a_m / y_m)|               -- best achievable
                                                                    PEAK dB error;
                                                                    the honest
                                                                    "+/- X dB" floor
  achieved         the a_m = diag(H X H^H) that attains it
  drive_trace      trace(X) required to attain it -- feasibility against real
                   shaker limits, not just a mathematical floor
  exactly_achievable  whether the target is hit to within `exact_tol_db`

Errors are measured in dB, NOT as a linear least-squares residual.  A linear
residual (what optimal_diagonal_control minimizes) implicitly weights the
loudest channels hardest; on a spec with any dynamic range across channels that
is not the quantity the test is judged by.

METHOD
------
The underlying problem is convex in X, but an SDP solver is a heavy dependency
(cvxpy may not be present in a given Rattlesnake environment).  Instead X is
factored as X = L L^H, which makes X >= 0 structural and the parametrization
unconstrained.  This is the Burer-Monteiro factorization; for a problem with
this few constraints (M = 8) a factor rank of N = 6 is comfortably above the
sqrt(2M) ~ 4 threshold at which spurious local minima generically disappear, so
the factored problem is expected to recover the global optimum.  That is not
taken on faith.  Measured on real, badly conditioned FRF data (condition
numbers to 8e4), the best-of-restarts value was identical to 0.000 dB across
five independent random seeds, and best-of-8 restarts equalled best-of-2
exactly -- the minimum is recovered reliably.  Individual starting points do
NOT all land there: on ill-conditioned lines the pseudoinverse start can
converge several dB worse than a random one.  `restart_spread_db` reports
that spread BETWEEN starts.  It is a measure of how rugged the landscape is,
not of whether the reported minimum is trustworthy; re-running with a
different `seed` is the test for that.

Only numpy and scipy are required.

TIMING (measured, 8 responses x 6 drives, per frequency line solved)
  restarts=2 nfev=400  minimax on   ~125 ms/line   -> ~2 min for a 901-line band
  restarts=2 nfev=300  minimax off   ~68 ms/line   -> ~1 min
  restarts=2 nfev=1200 minimax off  ~266 ms/line   -> ~4 min
Cost is per line and lines are independent, so it scales linearly and
parallelises trivially.  The defaults target the ~2 minute mark.  Against a
fully converged reference (restarts=3, nfev=4000) the default settings were
0.02 dB worse at the median and 0.5 dB worse at the very worst line, so the
expensive settings buy little; raise `max_nfev` only if you care about the
last few tenths on the hardest lines.

USAGE
-----
    import numpy as np
    from control_laws.achievable_response import achievable_diagonal, evaluate_drive

    res = achievable_diagonal(H, y_target)          # H (F,M,N), y_target (F,M)
    print(res['best_rms_db'][in_band].mean())

    # score an existing law's drive against that floor
    ref = evaluate_drive(H, X_from_some_law, y_target)
    headroom = ref['rms_db'] - res['best_rms_db']   # dB left on the table
"""

import numpy as np
from scipy.optimize import least_squares, minimize

__all__ = ['achievable_diagonal', 'evaluate_drive']

_DB = 10.0 / np.log(10.0)          # natural-log error -> dB
_TINY = 1e-300


# ----------------------------------------------------------------------
# packing helpers: complex L (N,K) <-> real parameter vector
# ----------------------------------------------------------------------
def _unpack(v, N, K):
    h = N * K
    return (v[:h] + 1j * v[h:]).reshape(N, K)


def _pack(L):
    return np.concatenate([L.real.ravel(), L.imag.ravel()])


def _diag_response(Hb, L):
    """a_m = h_m L L^H h_m^H = ||h_m L||^2, plus the row products needed by the
    gradient.  Hb is (M,N), L is (N,K)."""
    R = Hb @ L                                     # (M,K)
    a = np.maximum(np.sum(R.real**2 + R.imag**2, axis=1), _TINY)
    return a, R


def _wirtinger_terms(Hb, R, a):
    """d(a_m)/dL_bar = conj(h_m) outer r_m, scaled by 1/a_m for the log error.
    Returns (M,N,K)."""
    return np.einsum('mi,mk->mik', Hb.conj(), R) / a[:, None, None]


# ----------------------------------------------------------------------
# objective 1: least squares on the dB (log) error  ->  best RMS
# ----------------------------------------------------------------------
def _resid_jac(v, Hb, y_log, N, K):
    L = _unpack(v, N, K)
    a, R = _diag_response(Hb, L)
    d = np.log(a) - y_log
    G = 2.0 * _wirtinger_terms(Hb, R, a)           # (M,N,K)
    M = d.size
    J = np.concatenate([G.real.reshape(M, -1), G.imag.reshape(M, -1)], axis=1)
    return d, J


def _resid(v, Hb, y_log, N, K):
    return _resid_jac(v, Hb, y_log, N, K)[0]


def _jac(v, Hb, y_log, N, K):
    return _resid_jac(v, Hb, y_log, N, K)[1]


# ----------------------------------------------------------------------
# objective 2: smooth minimax (log-sum-exp) -> best peak error
# ----------------------------------------------------------------------
def _softmax_obj(v, Hb, y_log, N, K, beta):
    L = _unpack(v, N, K)
    a, R = _diag_response(Hb, L)
    d = np.log(a) - y_log
    z = beta * d
    mx = np.max(np.abs(z))
    ep = np.exp(z - mx)
    em = np.exp(-z - mx)
    S = np.sum(ep + em)
    f = (mx + np.log(S)) / beta
    w = (ep - em) / S                              # dF/dd_m
    G = 2.0 * np.einsum('m,mik->ik', w, _wirtinger_terms(Hb, R, a))
    return f, np.concatenate([G.real.ravel(), G.imag.ravel()])


# ----------------------------------------------------------------------
# starting points
# ----------------------------------------------------------------------
def _sqrtm_psd(X):
    w, V = np.linalg.eigh(X)
    return V * np.sqrt(np.clip(w, 0.0, None))[None, :]


def _init_pinv(Hb, y, rcond):
    """Ordinary pseudoinverse synthesis -- the solution the existing laws use.
    Exact whenever M <= N and H has full row rank."""
    Hp = np.linalg.pinv(Hb, rcond)
    X = Hp @ np.diag(y).astype(complex) @ Hp.conj().T
    X = 0.5 * (X + X.conj().T)
    return _sqrtm_psd(X)


def _init_random(N, K, scale, rng):
    return scale * (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2 * K)


# ----------------------------------------------------------------------
# single-bin solve
# ----------------------------------------------------------------------
def _solve_bin(Hb, y, K, n_restarts, rng, rcond, do_minimax,
               max_nfev, betas):
    """Best achievable diag(H X H^H) for one frequency line.

    Returns (best_L, rms_db, minimax_db, spread_db)."""
    N = Hb.shape[1]
    y_log = np.log(np.maximum(y, _TINY))
    scale = np.sqrt(max(np.mean(y), _TINY)) / max(np.linalg.norm(Hb), _TINY)

    starts = []
    try:
        L0 = _init_pinv(Hb, y, rcond)
        if np.all(np.isfinite(L0)):
            starts.append(L0)
    except np.linalg.LinAlgError:
        pass
    while len(starts) < max(1, n_restarts):
        starts.append(_init_random(N, K, scale, rng))

    best = None
    rms_values = []
    for L0 in starts:
        try:
            # 'trf', not 'lm': there are 2*N*K parameters against only M
            # residuals, and Levenberg-Marquardt refuses m < n.  The excess
            # parameters are pure gauge freedom (L and L U give the same X for
            # any unitary U), which trf handles without complaint.
            sol = least_squares(_resid, _pack(L0), jac=_jac, method='trf',
                                args=(Hb, y_log, N, K), max_nfev=max_nfev)
        except Exception:
            continue
        d = _resid(sol.x, Hb, y_log, N, K)
        rms = np.sqrt(np.mean(d**2)) * _DB
        rms_values.append(rms)
        if best is None or rms < best[0]:
            best = (rms, sol.x)

    if best is None:                                # every start failed
        return None, np.nan, np.nan, np.nan

    rms_db, v = best
    spread = (max(rms_values) - min(rms_values)) if len(rms_values) > 1 else 0.0

    minimax_db = np.max(np.abs(_resid(v, Hb, y_log, N, K))) * _DB
    if do_minimax:
        vm = v.copy()
        for beta in betas:
            try:
                out = minimize(_softmax_obj, vm, jac=True, method='L-BFGS-B',
                               args=(Hb, y_log, N, K, beta),
                               options={'maxiter': 200})
                cand = np.max(np.abs(_resid(out.x, Hb, y_log, N, K))) * _DB
                if np.isfinite(cand) and cand < minimax_db:
                    minimax_db = cand
                    vm = out.x
            except Exception:
                break
        v_minimax = vm
    else:
        v_minimax = v

    return (v, v_minimax), rms_db, minimax_db, spread


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def achievable_diagonal(transfer_function, y_target, line_indices=None,
                        rank=None, n_restarts=2, seed=0, rcond=1e-8,
                        restrict_rcond=None,
                        compute_minimax=True, exact_tol_db=0.01,
                        max_nfev=400, betas=(10.0, 40.0, 160.0),
                        return_drives=False, progress=None):
    """Best achievable response auto-spectra, per frequency line.

    Parameters
    ----------
    transfer_function : (F, M, N) complex
        FRF: M control responses, N drives.
    y_target : (F, M) real
        Target auto-spectra (the specification diagonal).  Lines whose target
        is all <= 0 or non-finite are skipped and reported as NaN.
    line_indices : array of int, optional
        Restrict the computation to these lines (e.g. every 5th line for a
        fast look).  Default: every line with a positive target.
    rank : int, optional
        Factor rank K in X = L L^H.  Default N (full rank, no restriction).
        Setting K < N answers a different question: the best achievable using
        drive of rank at most K.
    n_restarts : int
        Starting points per bin.  The first is always the pseudoinverse
        solution; the rest are random.  Two is enough in practice: best-of-8
        was measured identical to best-of-2 on real data.  `restart_spread_db`
        reports how far apart the individual starts landed, which on
        ill-conditioned lines can be several dB -- that reflects a rugged
        landscape, NOT an unreliable minimum.  To test the minimum itself,
        re-run with a different `seed` and compare.
    rcond : float
        Conditioning cutoff for the pseudoinverse STARTING POINT only.  It does
        NOT constrain the answer -- see `restrict_rcond` for that.
    restrict_rcond : float, optional
        Confine the drive to the well-conditioned subspace, the way a control
        law using pinv(H, rcond) implicitly does.  Only the right singular
        directions of H with singular value >= restrict_rcond * sigma_max are
        allowed to carry drive; the rest are forbidden outright.  This is a
        genuine CONSTRAINT on the answer, unlike `rcond` above.

        Use it to separate two things that are otherwise conflated when a
        pinv-based law underperforms: how much accuracy is lost by discarding
        near-singular drive directions (a deliberate, safety-motivated choice
        -- those directions cost enormous drive for almost no response), versus
        how much the law is simply leaving on the table.  Running with
        restrict_rcond=1e-3 gives the floor that a law truncating at 1e-3 is
        actually entitled to; the gap to the unrestricted floor is the price
        of the truncation itself.

        Implemented exactly, not by penalty: with V_r the retained right
        singular vectors, X = V_r X_r V_r^H, so the problem is re-solved in the
        reduced drive space and lifted back.  trace(X) is unchanged by the
        lift since V_r has orthonormal columns.  Default None = unrestricted.
    exact_tol_db : float
        Threshold below which a line is reported as exactly achievable.

    Returns
    -------
    dict with per-line arrays (length F, NaN off-band):
        best_rms_db, best_minimax_db, achieved (F,M), drive_trace,
        exactly_achievable (bool), restart_spread_db, solved (bool)
        and, if return_drives, drive_cpsd (F,N,N).
    """
    H = np.asarray(transfer_function)
    y = np.asarray(y_target, dtype=float)
    if H.ndim != 3:
        raise ValueError('transfer_function must be (F, M, N)')
    F, M, N = H.shape
    if y.shape != (F, M):
        raise ValueError(f'y_target must be (F, M) = {(F, M)}, got {y.shape}')
    K = N if rank is None else int(rank)

    valid = np.all(np.isfinite(y), axis=1) & (np.max(y, axis=1) > 0) \
        & np.all(np.isfinite(H), axis=(1, 2))
    if line_indices is None:
        idx = np.flatnonzero(valid)
    else:
        idx = np.asarray(line_indices, dtype=int)
        idx = idx[valid[idx]]

    out = {
        'best_rms_db': np.full(F, np.nan),
        'best_minimax_db': np.full(F, np.nan),
        'achieved': np.full((F, M), np.nan),
        'drive_trace': np.full(F, np.nan),
        'exactly_achievable': np.zeros(F, dtype=bool),
        'restart_spread_db': np.full(F, np.nan),
        'solved': np.zeros(F, dtype=bool),
        'retained_rank': np.zeros(F, dtype=int),
        'n_lines_solved': 0,
        'rank': K,
    }
    if return_drives:
        out['drive_cpsd'] = np.full((F, N, N), np.nan, dtype=complex)

    rng = np.random.default_rng(seed)
    for count, f in enumerate(idx):
        yf = np.maximum(y[f], _TINY)
        Hb = H[f]
        if restrict_rcond is not None and restrict_rcond > 0:
            # Confine the drive to the retained right-singular subspace.
            # diag(H V_r X_r V_r^H H^H) == diag((H V_r) X_r (H V_r)^H), so the
            # restricted problem IS the unrestricted problem on H V_r.
            sv, Vh = np.linalg.svd(Hb)[1:]
            r_keep = int(np.count_nonzero(sv >= restrict_rcond*sv[0])) if sv[0] > 0 else 0
            if r_keep == 0:
                continue
            Vr = Vh.conj().T[:, :r_keep]
            Hs = Hb @ Vr
        else:
            Vr, r_keep, Hs = None, N, Hb
        Ks = min(K, Hs.shape[1])
        vv, rms_db, mm_db, spread = _solve_bin(
            Hs, yf, Ks, n_restarts, rng, rcond, compute_minimax, max_nfev, betas)
        if vv is None:
            continue
        v_rms, v_mm = vv
        L = _unpack(v_rms, Hs.shape[1], Ks)
        a, _ = _diag_response(Hs, L)
        Xr = L @ L.conj().T
        X = Vr @ Xr @ Vr.conj().T if Vr is not None else Xr
        out['retained_rank'][f] = r_keep
        out['best_rms_db'][f] = rms_db
        out['best_minimax_db'][f] = mm_db
        out['achieved'][f] = a
        out['drive_trace'][f] = np.real(np.trace(X))
        out['exactly_achievable'][f] = mm_db <= exact_tol_db
        out['restart_spread_db'][f] = spread
        out['solved'][f] = True
        if return_drives:
            out['drive_cpsd'][f] = X
        if progress and (count % progress == 0):
            print(f'  ...{count}/{idx.size} lines', flush=True)

    out['n_lines_solved'] = int(out['solved'].sum())
    return out


def evaluate_drive(transfer_function, drive_cpsd, y_target):
    """Score an existing law's drive CPSD against the same target, so it can be
    compared directly with `achievable_diagonal`'s floor.

    Returns dict with per-line rms_db, minimax_db, achieved, drive_trace.
    """
    H = np.asarray(transfer_function)
    X = np.asarray(drive_cpsd)
    y = np.asarray(y_target, dtype=float)
    F, M, N = H.shape
    Y = np.einsum('fmn,fnk,flk->fml', H, X, H.conj())
    a = np.maximum(np.real(np.einsum('fmm->fm', Y)), _TINY)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = 10.0 * np.log10(a / np.maximum(y, _TINY))
    bad = ~np.all(np.isfinite(y), axis=1) | (np.max(y, axis=1) <= 0)
    e[bad] = np.nan
    return {
        'rms_db': np.sqrt(np.mean(e**2, axis=1)),
        'minimax_db': np.max(np.abs(e), axis=1),
        'achieved': a,
        'drive_trace': np.real(np.einsum('fnn->f', X)),
        'error_db': e,
    }
