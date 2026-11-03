"""
optimal_diagonal_control_fast.py

Speed-optimized variant of optimal_diagonal_control: same buzz baseline and
three-step budgeted scheduler (inherited unchanged from optimal_diagonal_
control), but each bin's SDP is replaced by an unconstrained Burer-Monteiro
factored solve (X = L L^H, L is N x r) whenever it's SAFE to do so --
falling back to the base class's real, coherence-capped SDP otherwise.

WHY THIS EXISTS: the SDP's cost is dominated by N (drive count), scaling
roughly ~N^4 with the PSD-cone/coherence-constraint machinery. Benchmarked
at a 12-drive/30-response scale: SDP ~105-119 ms/bin vs. the factored solve
~5.4-5.7 ms/bin (~18-20x faster), matching SDP accuracy closely (mean diff
-0.02 dB, max 0.32 dB on real system H) -- but ONLY when unconstrained.
Left unconstrained, the factored solve was measured to drive pairwise drive
coherence to EXACTLY 1.0 in ~100% of bins at a 12-drive/30-response scale --
perfectly rank-deficient drives, which would make a live H1/H2 estimator's
reference CPSD matrix singular. That's fine if nothing is re-estimating H
live, but unsafe the moment it is.

At 6 drives the same effect was NOT seen in a saved run (median pairwise drive coherence
about 0.4-0.5, none at exactly 1.0), so treat "unconstrained drive coherence reaches 1.0"
as established at 12 drives only.  A caution for judging a system identification on this
rig: use the median and 5th percentile of the multiple coherence, or per-channel minima
compared like against like -- a bare minimum over all channels and lines is dominated by
channel 7 at the band edge and says nothing about run quality.
(Evidence: notes/optimal_diagonal_history.md#fast-coherence-claim)

THE SAFETY RULE (this is the important part): rather than trying to detect
Rattlesnake's "Update Transfer Function During Control" checkbox directly
(control()'s signature doesn't expose it), this watches the actual
safety-relevant property instead -- has H MOVED since the last call, by
more than frf_update_threshold in relative Frobenius norm? If it has not,
nothing is meaningfully re-estimating it right now and the fast
unconstrained path is safe. The moment it has, every solve for the rest of
that call uses the real coherence-capped SDP instead, via
super()._solve_one_bin(). This degrades gracefully: a run that starts
static and has TF-update enabled mid-test downgrades to the safe path as
soon as the plant model actually moves, rather than trusting a stale
assumption from init time. See _h_moved().

(The gate was once "any bitwise difference in H", which with the FRF update on sent EVERY
bin to the SDP and made this law identical to the base class.  The relative-change
threshold fixes that; frf_update_threshold = 0 restores the any-change behaviour.
See notes/optimal_diagonal_history.md#fast-safety-rule.)

PARAMETERS: the named form ("name: value" per line, see optimal_diagonal_control)
takes every parameter of both laws, this one's included (bm_rank, the svd_*
group, cap_baseline, db_coherence_guard, warm_start, gate_reference).  The
complete list, with field numbers and defaults, is generated from the tables:
control_laws/optimal_diagonal_parameters.md.  The notes below are for the
comma-separated form (1-based field numbers):

extra_parameters: same values as optimal_diagonal_control -- the 6th is
this subclass's own, the 7th is parsed by the base class, so ONE string
works for both laws:
    "reg,frf_update_threshold,max_bins_per_update,error_threshold_db,max_drive_coherence,bm_rank,startup_test_level_cap_db"
    bm_rank - factorization rank r for the fast solve (default 4; the SDP's
              own constrained solutions rarely need rank >3-4 for 99% of
              their energy, and the unconstrained solve needs even less)
    startup_test_level_cap_db - ceiling on the FIRST control command only,
              in dB re the specification trace (default -9.0). Handled
              entirely in the base class; see its docstring, including the
              caveat that the ceiling keys on "first command" so it does not
              bind after cycle 1.
response_trim_gain and its three companions (fields 11-14) are inherited
              unchanged from the base class, which owns the whole
              response-error trim: this subclass overrides only the per-bin
              solve, and the trim acts on the TARGET handed to that solve.
              Off by default, as in the base class.
max_drive_coherence still applies to the SDP fallback path exactly as in
the base class; it has no effect on the fast path (which is, by design,
unconstrained).

PER-LINE SINGULAR-VECTOR SELECTION (fields 16-18, added 2026-10-24)
------------------------------------------------------------------
The fast path keeps, at each line, the k strongest singular directions of
that line's H and solves only in them.  Keeping the right singular vectors
V_k for the drive keeps the response in the matching LEFT singular vectors
U_k, because H V_k = U_k S_k -- one truncation selects both.  What differs
between the modes is how the cut for each line is chosen:

  svd_select = 0  (default, unchanged)  one threshold for the whole band:
        drive_rcond > 0  ->  drive_rcond x the band's median structural
                             spread (_effective_drive_rcond), then each line
                             keeps sigma >= that x its own sigma_max
        drive_rcond < 0  ->  an absolute ratio |drive_rcond|, e.g. -1e-3
        drive_rcond = 0  ->  no restriction
  svd_select = 1  each line chooses from ITS OWN spectrum: the deepest gap
        of at least svd_gap_min between adjacent ratios sigma_k/sigma_1
        (among ratios >= svd_floor) marks that line's structural floor -- the
        deepest, not the largest, because near a resonance the largest gap is
        sigma_1/sigma_2.  The line's reference is the smallest direction above
        that gap, and the cut is
        drive_rcond x that reference -- the same meaning drive_rcond has in
        the band mode, now per line (1.0 keeps everything above the gap; the
        rig's tuned 2.0 also drops the weakest structural direction when it
        lies within 2x of the gap).  A line with no gap that large keeps all
        directions above svd_floor.  drive_rcond < 0 or 0 behave exactly as
        in the band mode.

    svd_select    field 16, default 0
    svd_gap_min   field 17, default 3.0   (ratio between adjacent sigmas)
    svd_floor     field 18, default 1e-6  (sigma/sigma_1 below this is never
                                           kept, and never counted as a gap)
    svd_weak_max  field 19, default 0.05  (a gap counts as a structural floor
                                           only if the direction below it has
                                           sigma/sigma_1 at or under this)
    svd_min_keep  field 20, default 0     FEWEST directions any line keeps, in
                                          EITHER mode: 0 = auto, the band's
                                          structural rank minus one (rank from
                                          the median spectrum's largest gap; 5
                                          on the 6-drive frame, so at least 4);
                                          n >= 1 = that number; -1 = N - 1.
                                          1 restores the pre-2026-10-24
                                          behaviour exactly.

    WHY THE FLOOR, and why it is ON by default unlike the other new fields:
    every cut is relative to the line's own sigma_1, and next to a strong
    resonance every other direction looks weak against it, so either mode
    can keep one or two directions there and abandon the channels that mode
    does not reach (the sweep plant: band mode and per-line mode both did).
    The floor only RAISES k on such lines; on a line that already keeps at
    least the floor nothing changes.

Empty fields keep their position and mean "use the default".  Per-line
selection on, everything else default:
    ,,,,,,,,,,,,,,,1
A full default string with per-line selection on:
    1e-6,0.5,20,1.0,0.95,4,-9,0,2,2.0,0,3,0.5,0.5,0.5,1,3,1e-6,0.05,0
Field 21, max_solve_ms_per_update, is the base class's per-call time budget
(see optimal_diagonal_control); it caps the SDP fallback's calls too.

FIELD 25 (added 2026-10-25)
    gate_reference      field 25, default 1  what the fast/SDP movement gate
                                             compares the FRF with: 1 = the
                                             previous call's FRF (per-cycle
                                             movement, what its threshold was
                                             set from); 0 = each line's FRF at
                                             its last solve (pre-2026-10-25)

FIELDS 22-24 (added 2026-10-25)
    cap_baseline        field 22, default 0  1 = apply max_drive_coherence to
                                             the buzz baseline that unsolved
                                             lines sit on (see _buzz_solve_all)
    db_coherence_guard  field 23, default 1  with the dB objective (field 8 =
                                             1), a fast solution that breaks
                                             max_drive_coherence is re-solved
                                             with the SDP for that line; 2 =
                                             shrink its cross terms instead
                                             (cheaper, less accurate); 0 = off
                                             (pre-2026-10-25)
    warm_start          field 24, default 1  start each fast solve from the
                                             line's current drive when that
                                             scores better than the buzz seed;
                                             0 = always the buzz seed (pre-
                                             2026-10-25)
Fields 23 and 24 default ON, so a string with 22 or fewer fields no longer
reproduces a pre-2026-10-25 run exactly.  For that, append fields 23-25 as 0
(empty fields in between are fine): e.g. ',,,,,,,,,,,,,,,,,,,,,,0,0,0' -- field
25 is the movement-gate reference below.
"""

import os
import importlib.util
import numpy as np
import scipy.optimize as _scipy_optimize  # module import, not `from ... import minimize` --
                                            # a bare top-level `minimize` name would itself get
                                            # picked up as a spurious "control law" candidate by
                                            # Rattlesnake's loose function-detection heuristic
                                            # (>=12 parameters, which minimize's signature has)

# Rattlesnake loads this file standalone, by path (see load_sibling in the base file), so
# the base law is loaded by path too -- the one load that cannot use load_sibling, since
# the base file is what provides it.
_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimal_diagonal_control.py")
_spec = importlib.util.spec_from_file_location("optimal_diagonal_control_base", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)
optimal_diagonal_control = _base_module.optimal_diagonal_control
# The drive-coherence cap used by the closed-form laws (shrink the offending cross terms,
# re-project onto the PSD cone).
_cap_drive_coherence = _base_module.load_sibling(
    'control_laws.py', '_rattlesnake_control_laws_helpers_fast')._cap_drive_coherence

# This law's own parameters.  The Param class, the converters and the parser
# (optimal_diagonal_control._read_parameters) are the base module's, so one
# string -- legacy or named -- serves both laws; the base law's parameters are
# read by the base class, these here.
Param = _base_module.Param
_as_int, _magnitude = _base_module._as_int, _base_module._magnitude

_FAST_PARAMS = [
    Param('bm_rank', 4, _as_int, 5,
          "factorization rank of the fast solve; the SDP's constrained solutions rarely need rank above 3-4"),
    Param('svd_select', 0, _as_int, 15,
          '0 = one singular-value cut for the whole band, 1 = each line picks its own'),
    Param('svd_gap_min', 3.0, lambda s: max(1.0, float(s)), 16,
          "per-line selection: smallest ratio between adjacent sigmas that marks a line's structural floor"),
    Param('svd_floor', 1e-6, _magnitude, 17,
          'sigma/sigma_1 below this is never kept and never counted as a gap'),
    Param('svd_weak_max', 0.05, _magnitude, 18,
          'a gap counts as a structural floor only if the direction below it has sigma/sigma_1 at or under this'),
    Param('svd_min_keep', 0, _as_int, 19,
          'fewest directions any line keeps (0 = auto: structural rank - 1; -1 = N - 1; 1 = pre-2026-10-24 behaviour)'),
    Param('cap_baseline', 0, _as_int, 21,
          '1 = apply max_drive_coherence to the buzz baseline that unsolved lines sit on'),
    Param('db_coherence_guard', 1, _as_int, 22,
          'with the dB objective, a fast solution that breaks the coherence cap: 1 = re-solve with the SDP, 2 = shrink its cross terms, 0 = off'),
    Param('warm_start', 1, _as_int, 23,
          "1 = start each fast solve from the line's current drive when that scores better than the buzz seed"),
    Param('gate_reference', 1, _as_int, 24,
          "what the fast/SDP movement gate compares the FRF with: 1 = the previous call's, 0 = each line's at its last solve"),
]



# 10/ln(10), squared: converts a natural-log ratio to dB and back.
_C_DB = 10.0/np.log(10.0)
_C_DB2 = _C_DB**2
# Floor under yhat and y inside the dB objective, so a bin passing through
# zero response gives a large finite penalty rather than an overflow.
_DB_FLOOR = 1e-300
# Used only if the singular-value spectrum cannot be measured at all.
_FALLBACK_SPREAD = 0.0152


class optimal_diagonal_control_fast(optimal_diagonal_control):
    OWN_PARAMS = _FAST_PARAMS

    def __init__(self, *args, **kwargs):
        # This subclass's own state is set up BEFORE super().__init__(): the base
        # constructor calls _initialize() whenever it is handed a non-None transfer_function,
        # and that runs _refine_batch() -> _solve_one_bin(), which is overridden here and
        # reads self.bm_rank / self.n_fast_solves / self._h_changed_this_call.  Set after the
        # super() call, a law constructed with the FRF in hand raises AttributeError.
        # (Rattlesnake itself constructs laws before system ID, with transfer_function=None.)
        self._H_prev_call = None
        extra_parameters = kwargs.get('extra_parameters', args[3] if len(args) > 3 else '')
        self._read_parameters(extra_parameters, optimal_diagonal_control_fast.OWN_PARAMS)
        # per-call record of how many directions each fast solve kept
        self._k_counts = {}

        self.n_fast_solves = 0
        self.n_safe_solves = 0
        self._h_changed_this_call = False

        print(f"[optimal_diagonal_control_fast] bm_rank={self.bm_rank} "
              f"svd_select={self.svd_select} ({'per line' if self.svd_select == 1 else 'band'}) "
              f"svd_gap_min={self.svd_gap_min:g} svd_floor={self.svd_floor:g} "
              f"svd_weak_max={self.svd_weak_max:g} "
              f"svd_min_keep={self.svd_min_keep if self.svd_min_keep != 0 else 'auto'} "
              f"cap_baseline={self.cap_baseline} db_coherence_guard={self.db_coherence_guard} "
              f"warm_start={self.warm_start} "
              f"gate_reference={'previous call' if self.gate_reference else 'per-line last solve'}",
              flush=True)

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def _refine_batch(self, transfer_function):
        # Same H_changed check the base class's _refine_batch computes
        # internally (before self.H_cache gets mutated by any bin solves
        # this call) -- duplicated here (not read back out of the base
        # class, which doesn't expose it) so _solve_one_bin knows, this
        # call, whether it's safe to use the fast unconstrained path.
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self._h_changed_this_call = self._h_moved(H_clean)
        self._H_prev_call = H_clean.copy()
        self._k_counts = {}
        self._n_raised = 0
        super()._refine_batch(transfer_function)
        if self._k_counts and (self._n_calls <= 5 or self._n_calls % 25 == 0):
            dist = ', '.join(f'k={k}: {n}' for k, n in sorted(self._k_counts.items()))
            print(f"[optimal_diagonal_control_fast] directions kept per solved line "
                  f"({'per-line gap' if self.svd_select == 1 else 'band threshold'}, "
                  f"floor {self._min_keep(self.N or 0)}): {dist}; "
                  f"{self._n_raised} line(s) raised to the floor", flush=True)

    def _h_moved(self, H_clean):
        """Has the FRF moved enough to make the unconstrained fast path
        unsafe?

        (This used to be `not np.array_equal(H_clean, self.H_cache)` -- ANY difference --
        which with the FRF update on sent every bin to the SDP; see
        notes/optimal_diagonal_history.md#fast-safety-rule.)

        The threshold is frf_update_threshold, the SAME relative-change
        metric the base class already uses to decide which refined bins are
        stale enough to re-solve (optimal_diagonal_control.py's `drifted`
        selection).  So the fast path survives ordinary averaging jitter and
        a genuinely moving plant still demotes every bin to the coherence-
        capped SDP, which is what the gate exists for.  Setting
        frf_update_threshold to 0 restores the old any-change behaviour.
        """
        if self.H_cache is None:
            return True
        # REFERENCE (field 25).  The gate measures how much the FRF moved SINCE THE LAST CALL;
        # its threshold was set from per-cycle movement.  Compared against H_cache, which holds
        # each line's FRF at that line's LAST SOLVE, it would measure accumulated drift instead,
        # and once that passed the threshold every later call would go to the SDP.  So: the
        # previous call's FRF (H_cache on the first call).  gate_reference = 0 restores the old
        # reference.  Both are logged.  History: notes/optimal_diagonal_history.md#fast-h-moved
        H_ref = (self._H_prev_call if self.gate_reference and self._H_prev_call is not None
                 and self._H_prev_call.shape == H_clean.shape else self.H_cache)
        # IN-BAND ONLY.  Over the whole array this metric is dominated by out-of-band bins,
        # where there is no excitation, the live estimate is noise and the relative change per
        # cycle is enormous; the gate would respond to that noise, never to the plant.  (Same
        # rule as the singular-value spread in _effective_drive_rcond.)  History: notes/optimal_diagonal_history.md#in-band
        def _rel(Href):
            Hc, Hn = Href, H_clean
            if (Hc.shape[0] == self.y_diag_target.shape[0]
                    and Hn.shape[0] == self.y_diag_target.shape[0]):
                in_band = self.y_diag_target.max(axis=1) > 0
                if in_band.any():
                    Hc, Hn = Hc[in_band], Hn[in_band]
            baseline = np.linalg.norm(Hc.reshape(-1))
            if not np.isfinite(baseline) or baseline <= 0:
                return np.inf
            delta = np.linalg.norm((Hn - Hc).reshape(-1))
            return delta/baseline if np.isfinite(delta) else np.inf
        rel = _rel(H_ref)
        if not np.isfinite(rel):
            return True
        rel_since_solve = rel if H_ref is self.H_cache else _rel(self.H_cache)
        moved = rel > self.frf_update_threshold

        # Record what the FRF actually does, whether or not it trips the gate: the demotion
        # threshold should be set from the per-cycle movement this rig really produces, which
        # cannot be recovered afterwards (only the FINAL FRF is saved, so a run file shows
        # cumulative drift, not the movement the gate tests).
        self._h_move_log = getattr(self, '_h_move_log', [])
        self._h_move_log.append(float(rel))
        n = len(self._h_move_log)
        if moved or n <= 5 or n % 25 == 0:
            a = np.asarray(self._h_move_log)
            print(f"[optimal_diagonal_control_fast] FRF moved {rel:.4f} "
                  f"since the {'previous call' if H_ref is not self.H_cache else 'per-line last solve'} "
                  f"({rel_since_solve:.4f} since each line's last solve; "
                  f"threshold {self.frf_update_threshold:g}, "
                  f"{'SDP' if moved else 'fast'} path) -- over {n} calls: "
                  f"median {np.median(a):.4f}, 90th {np.percentile(a, 90):.4f}, "
                  f"max {a.max():.4f}, demoted {100*np.mean(a > self.frf_update_threshold):.0f}%; "
                  # PER-BIN counters, not per-call: the demotion rate above counts CALLS; these
                  # count the solves that actually took each path, which is what "was the fast
                  # law used?" really asks.
                  f"solves: {self.n_fast_solves} fast / {self.n_safe_solves} SDP "
                  f"({100*self.n_fast_solves/max(1, self.n_fast_solves+self.n_safe_solves):.0f}% fast)",
                  flush=True)
        return moved

    # ------------------------------------------------------------------
    # Unconstrained Burer-Monteiro factored solve: X = L L^H, L is N x r.
    # Same objective as the base class's SDP minus the coherence-cap
    # constraint: || diag(H X H^H) - y ||^2 + reg * ||X||_F^2.
    # Gradient (Wirtinger, verified against finite differences): for
    # g(X) = objective as a function of X directly, grad_L = 2*(grad_X g)@L.
    # ------------------------------------------------------------------
    @staticmethod
    def _pack(L):
        return np.concatenate([L.real.ravel(), L.imag.ravel()])

    @staticmethod
    def _unpack(v, N, r):
        n = N * r
        return v[:n].reshape(N, r) + 1j * v[n:].reshape(N, r)

    def _bm_obj_and_grad(self, v, H, y_target, N, r):
        """Objective and gradient for the factored solve.

        error_domain 'linear' (the DEFAULT, set in the base class) minimizes
            sum_m (yhat_m - y_m)^2.
        error_domain 'db' (opt in with field 8 = 1) minimizes
            sum_m [10 log10(yhat_m / y_m)]^2
        directly.

        WHY 'db' EXISTS.  The linear objective caps what abandoning a channel can cost at
        that channel's own target squared -- drive its response to zero and the penalty stops
        growing -- so on a plant whose rows have unequal gain the solver sells the hardest
        channels to buy the easy ones, and is correct to, given what it was asked to
        minimize.  The dB objective has no such ceiling, so it spreads the error across every
        channel instead.  (Measurements: notes/optimal_diagonal_history.md#fast-db-objective.)

        The dB objective is NOT convex, which is why it lives here and not in
        the base class's SDP; the base class reaches the same objective by
        iteratively reweighted least squares instead, keeping every subproblem
        convex and its coherence-cap constraint valid.  The two agree on what
        they are minimizing but will not agree to the last decimal.
        """
        L = self._unpack(v, N, r)
        B = H @ L
        Ydiag = np.sum(np.abs(B) ** 2, axis=1)
        X = L @ L.conj().T
        f_reg = self.reg * np.sum(np.abs(X) ** 2)
        g_reg = 2 * self.reg * X

        if self.error_domain == 'db':
            # f = sum (C ln(yhat/y))^2, C = 10/ln10.  df/dyhat = 2 C^2
            # ln(yhat/y) / yhat.  Guarded: a bin can pass through yhat = 0.
            Yc = np.maximum(Ydiag, _DB_FLOOR)
            yc = np.maximum(y_target, _DB_FLOOR)
            logr = np.log(Yc / yc)
            f_fit = _C_DB2 * np.sum(logr ** 2)
            dfdy = 2.0 * _C_DB2 * logr / Yc
            dfdy = np.where(y_target > 0, dfdy, 0.0)
        else:
            resid = Ydiag - y_target
            f_fit = np.sum(resid ** 2)
            dfdy = 2.0 * resid

        f = f_fit + f_reg
        gX = (H.conj().T @ (dfdy[:, None] * H)) + g_reg
        gL = 2 * (gX @ L)
        grad = np.concatenate([gL.real.ravel(), gL.imag.ravel()])
        return f, grad

    def _buzz_seeded_L(self, H, y_target, r, rcond=1e-12):
        """Rank-r warm start: eigendecompose the pinv/buzz diagonal-only
        closed-form solution for this bin, keep the top-r eigenpairs."""
        N = H.shape[1]
        Hpinv = np.linalg.pinv(H, rcond=rcond)
        Xbuzz = Hpinv @ np.diag(y_target).astype(complex) @ Hpinv.conj().T
        Xbuzz = (Xbuzz + Xbuzz.conj().T) / 2
        w, V = np.linalg.eigh(Xbuzz)
        order = np.argsort(-w)[:r]
        w_top = np.clip(w[order], 0, None)
        return (V[:, order] * np.sqrt(w_top)[None, :]).astype(complex)

    def _warm_L(self, X, r):
        """Rank-r factor of a drive CPSD (top-r eigenpairs), or None."""
        if X is None or not np.all(np.isfinite(X)):
            return None
        Xh = (X + X.conj().T)/2
        w, V = np.linalg.eigh(Xh)
        if not np.any(w > 0):
            return None
        order = np.argsort(-w)[:r]
        return (V[:, order]*np.sqrt(np.clip(w[order], 0, None))[None, :]).astype(complex)

    def _bm_solve(self, H, y_target, X_warm=None):
        M, N = H.shape
        r = min(self.bm_rank, N)
        L0 = self._buzz_seeded_L(H, y_target, r)
        # WARM START (field 24, 0 = off).  Start from this line's CURRENT drive when that scores
        # better than the buzz seed: restarting every re-solve from the buzz eigenvectors throws
        # away the line's existing solution, is slower, and is free to land in a different local
        # minimum of the non-convex factored problem from one cycle to the next.  Taking the
        # better of the two starts means a stale or poor current drive can never make it worse.
        if self.warm_start and X_warm is not None:
            Lw = self._warm_L(X_warm, r)
            if Lw is not None and Lw.shape == L0.shape:
                f_buzz, _ = self._bm_obj_and_grad(self._pack(L0), H, y_target, N, r)
                f_warm, _ = self._bm_obj_and_grad(self._pack(Lw), H, y_target, N, r)
                if np.isfinite(f_warm) and f_warm < f_buzz:
                    L0 = Lw
                    self.n_warm_starts = getattr(self, 'n_warm_starts', 0) + 1
        v0 = self._pack(L0)
        # Tolerances matter a lot here: on REAL (not synthetic-random) FRF data, ~70% of bins hit
        # a 200-iteration cap at tight tolerances without visibly improving dB accuracy.  maxiter=50,
        # ftol=1e-9, gtol=1e-7 cuts the time ~5.7x for only +0.4 dB mean / +2.7 dB worst case versus
        # the SDP's own accuracy on the same bins -- a real speedup (~3.6x vs the SDP mean on real
        # data), much more modest than synthetic-random-H benchmarks suggest.
        res = _scipy_optimize.minimize(self._bm_obj_and_grad, v0, args=(H, y_target, N, r), jac=True,
                                        method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7})
        L = self._unpack(res.x, N, r)
        return L @ L.conj().T

    # ------------------------------------------------------------------
    def _effective_drive_rcond(self, H):
        """The subspace threshold actually used, as a MULTIPLE of this plant's
        own singular-value spread.

        drive_rcond > 0 is read as a multiple of the smallest STRUCTURAL
        singular-value ratio; drive_rcond < 0 is an absolute ratio, |value|,
        for reproducing an old run or pinning a value by hand.

        WHY A MULTIPLE.  An absolute ratio has to be re-tuned for every test article, because it
        is a threshold on a quantity that is a property of the structure.  A multiple of the
        smallest structural direction tracks the plant automatically; 2.0x is the default (the
        rig sweep put the optimum at 3.0e-2 on the 6-drive frame, whose smallest structural
        direction sits at sigma_5/sigma_1 = 0.0152).

        FINDING THE STRUCTURAL FLOOR.  This 8-response/6-drive frame is rank 5:
        its measured sigma_k/sigma_1 medians run 1.000, 0.319, 0.084, 0.032,
        0.015, 0.0017, and that last value is the identification noise floor,
        not a direction the plant has.  Scaling off sigma_min would therefore
        track how good the system ID was rather than how the structure is
        conditioned.  The floor is found instead by the largest gap in the
        spectrum -- here 8.9x between sigma_5 and sigma_6, against 2.1-3.8x
        everywhere else -- and the reference is the smallest direction above
        that gap.

        Computed once per FRF from the median over in-band bins, so a single
        noisy line cannot move it.
        """
        if self.drive_rcond < 0:
            return -self.drive_rcond          # absolute override
        if self.drive_rcond == 0:
            return 0.0
        # Only cache once the FULL band is in hand.  H_cache is populated by _initialize before
        # _refine_batch runs, so the control loop always has it -- but _solve_one_bin called
        # directly (a harness, a unit test) would otherwise compute the reference from the ONE
        # bin it was handed and cache that answer forever.
        if self.H_cache is None:
            return float(np.clip(self.drive_rcond*_FALLBACK_SPREAD, 1e-4, 0.5))
        key = self.H_cache.shape
        if getattr(self, '_rcond_cache', None) is not None and self._rcond_key == key:
            return self._rcond_cache
        Hb = self.H_cache
        if Hb.ndim == 2:
            Hb = Hb[np.newaxis]
        # IN-BAND ONLY.  Out-of-band lines are not controlled and are
        # conditioned quite differently; including them moved the reference
        # from 0.0152 to 0.1435 on this frame, a factor of 9, which is the
        # difference between the tuned 3.0e-2 and a useless 0.29.
        if Hb.shape[0] == self.y_diag_target.shape[0]:
            in_band = self.y_diag_target.max(axis=1) > 0
            if in_band.any():
                Hb = Hb[in_band]
        try:
            sv = np.linalg.svd(np.nan_to_num(Hb), compute_uv=False)
        except np.linalg.LinAlgError:
            return self.drive_rcond*_FALLBACK_SPREAD
        ratios = sv/np.maximum(sv[:, :1], 1e-300)
        med = np.median(ratios, axis=0)                      # (N,)
        med = med[np.isfinite(med) & (med > 0)]
        if med.size < 2:
            ref = _FALLBACK_SPREAD
        else:
            gaps = med[:-1]/np.maximum(med[1:], 1e-300)
            ref = float(med[int(np.argmax(gaps))])           # last direction above the gap
        eff = float(np.clip(self.drive_rcond*ref, 1e-4, 0.5))
        self._rcond_cache = eff; self._rcond_key = key
        print(f"[optimal_diagonal_control_fast] drive_rcond {self.drive_rcond:g} x "
              f"plant spread {ref:.4f} -> effective {eff:.4g}", flush=True)
        return eff

    def _restricted_basis(self, H):
        """Right-singular directions of H carrying sigma >= drive_rcond*sigma_max,
        as an (N, k) matrix -- or None for no restriction.

        WHY THE FACTORED PATH NEEDS THIS AND THE SDP PATH DOES NOT.  The dB objective removes
        the ceiling on what missing a channel costs, which is the point, but it also means the
        solver will spend unlimited drive to chase a channel the plant barely reaches: a nearly
        rank-one drive pushed down a near-singular direction.  The base class's SDP does not
        need this because its coherence-cap constraint already restrains it; the factored path
        is unconstrained by design and has nothing else holding it.

        reg is NOT the right lever: it penalizes every direction equally, so raising it gives
        back the accuracy while still leaving excess mean and worst-bin drive.  Restricting the
        SUBSPACE is what the reachability floor itself does (restrict_rcond), and it targets the
        actual problem.  Set drive_rcond to 0 to disable the restriction.  (The measurements
        behind the default: notes/optimal_diagonal_history.md#fast-drive-rcond.)
        """
        # APPLIES IN BOTH ERROR DOMAINS.  Without the restriction the factored path in its linear
        # mode has nothing constraining it at all -- no coherence cap, which it cannot express, and
        # no subspace restriction -- and commands a drive whose response depends on inter-drive
        # cancellation the rig cannot physically realize.  drive_rcond bounds that sensitivity
        # directly, and does it better than the coherence cap, which only halves it.  Set
        # drive_rcond to 0 to disable.  The measurement: notes/optimal_diagonal_history.md#fast-drive-rcond
        per_line = self.svd_select == 1 and self.drive_rcond > 0
        rc = None if per_line else self._effective_drive_rcond(H)
        if not per_line and not (rc > 0):
            return None
        try:
            _, sv, Vh = np.linalg.svd(H, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] <= 0:
            return None
        if per_line:
            k = self._per_line_k(sv)
        else:
            k = int(np.sum(sv >= rc*sv[0]))
        # FLOOR ON THE COUNT.  Every cut is relative to the line's
        # own sigma_1, and next to a strong resonance every other direction
        # looks weak against it, so either selection mode can keep one or two
        # directions there and abandon the channels the resonant mode does
        # not reach.  Never keep fewer than _min_keep.
        k_floor = min(self._min_keep(H.shape[1]), sv.size)
        if k < k_floor:
            k = k_floor
            self._n_raised = getattr(self, '_n_raised', 0) + 1
        self._k_counts[k] = self._k_counts.get(k, 0) + 1
        if k < 1:
            k = 1
        if k >= H.shape[1]:
            return None          # nothing excluded, skip the change of basis
        return Vh[:k].conj().T

    def _band_structural_rank(self):
        """Directions above the largest gap in the band's MEDIAN in-band
        singular-value ratios -- the plant's structural rank, measured the
        same way _effective_drive_rcond finds its reference.  Full rank when
        the median spectrum has no gap of at least svd_gap_min.  Cached per
        FRF shape; None until the FRF is in hand."""
        Hb = self.H_cache
        if Hb is None:
            return None
        key = Hb.shape
        if getattr(self, '_struct_rank_key', None) == key:
            return self._struct_rank
        if Hb.shape[0] == self.y_diag_target.shape[0]:
            in_band = self.y_diag_target.max(axis=1) > 0
            if in_band.any():
                Hb = Hb[in_band]
        try:
            sv = np.linalg.svd(np.nan_to_num(Hb), compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        med = np.median(sv/np.maximum(sv[:, :1], 1e-300), axis=0)
        med = med[np.isfinite(med) & (med > 0)]
        n = int(med.size)
        if n >= 2:
            gaps = med[:-1]/np.maximum(med[1:], 1e-300)
            i = int(np.argmax(gaps))
            if gaps[i] >= self.svd_gap_min:
                n = i + 1
        self._struct_rank, self._struct_rank_key = n, key
        print(f"[optimal_diagonal_control_fast] band structural rank {n} of "
              f"{Hb.shape[2]} drives (median sigma ratios "
              f"{np.array2string(med, precision=4)}); fewest directions kept "
              f"per line = {self._min_keep(Hb.shape[2])}", flush=True)
        return n

    def _min_keep(self, N):
        """Fewest singular directions any line may keep.
           svd_min_keep >= 1 : that number
           svd_min_keep == 0 : auto = band structural rank - 1 (at least 1)
           svd_min_keep < 0  : N + svd_min_keep, e.g. -1 -> N - 1"""
        if not N:
            return 1
        if self.svd_min_keep >= 1:
            return int(min(self.svd_min_keep, N))
        if self.svd_min_keep < 0:
            return int(max(1, N + self.svd_min_keep))
        r = self._band_structural_rank()
        return int(max(1, (r if r is not None else N) - 1))

    @staticmethod
    def _max_drive_coherence_of(X):
        """Largest |X_ij| / sqrt(X_ii X_jj) over drive pairs -- the same
        (magnitude) measure max_drive_coherence caps on the SDP path."""
        d = np.real(np.diag(X))
        den = np.sqrt(np.maximum(np.outer(d, d), 1e-300))
        c = np.abs(X)/den
        np.fill_diagonal(c, 0.0)
        return float(np.max(c))

    def _buzz_solve_all(self, H_clean, sysid_response_cpsd):
        """The instant baseline every line starts on (base class), optionally
        coherence-capped (field 22, default 0 = off).

        Lines the scheduler has not yet solved keep this baseline, and the base
        class leaves it uncapped on purpose: it takes its cross terms from the
        system-ID response, which had independent drives.  But after the
        pseudoinverse those lines can still carry drive coherence near 1 --
        a comparison read a MAXIMUM of 0.98-1.00 for the SDP law
        itself, from exactly these lines.  cap_baseline = 1 applies
        max_drive_coherence to the baseline too.  Also applies to the
        deadlock escape, which rebuilds the baseline."""
        out = super()._buzz_solve_all(H_clean, sysid_response_cpsd)
        if self.cap_baseline and self.max_drive_coherence < 1.0:
            out = _cap_drive_coherence(out, self.max_drive_coherence)
        return out

    def _per_line_k(self, sv):
        """Directions to keep at one line, chosen from that line's own
        singular values (svd_select = 1).  See PER-LINE SINGULAR-VECTOR
        SELECTION in the module docstring."""
        ratios = sv/sv[0]
        valid = ratios >= max(self.svd_floor, 0.0)
        r = ratios[valid]
        if r.size < 2:
            return max(1, int(r.size))
        gaps = r[:-1]/np.maximum(r[1:], 1e-300)
        # The DEEPEST qualifying gap, not the largest.  Near a resonance one mode dominates and
        # the largest gap is sigma_1/sigma_2 (20x or more), which would keep a single direction
        # and abandon every other channel.  The structural floor is at the bottom of the
        # spectrum.  ...and only where the direction BELOW the gap is genuinely weak
        # (sigma/sigma_1 <= svd_weak_max): without this, a line whose only qualifying gap is a
        # resonance's sigma_1/sigma_2 still keeps one direction.
        big = np.where((gaps >= self.svd_gap_min) & (r[1:] <= self.svd_weak_max))[0]
        if big.size == 0:
            return int(r.size)                  # no structural floor: keep all valid
        i = int(big[-1])
        ref = r[i]                              # smallest direction above the gap
        thr = float(np.clip(self.drive_rcond*ref, 0.0, 0.5))
        return max(1, int(np.sum(ratios >= thr)))

    def _solve_one_bin(self, H, y_target, X_warm=None):
        # X_warm is this line's current drive.  The SDP fallback uses it for
        # its IRLS weights; the fast path uses it as a starting point (fix 5).
        if not self._h_changed_this_call:
            # H is unchanged since the last call -- nothing is re-
            # estimating it live right now, so the fast unconstrained
            # solve is safe even though it drives coherence toward 1.0.
            self.n_fast_solves += 1
            try:
                # Solve in the well-conditioned drive subspace and map back,
                # so the dB objective cannot buy accuracy with drive down a
                # near-singular direction -- see _restricted_basis.  The
                # coherence cap does not apply on this path either way, so
                # nothing is lost by working in a rotated basis here.
                Vk = self._restricted_basis(H)
                if Vk is None:
                    X = self._bm_solve(H, y_target, X_warm=X_warm)
                else:
                    Xw = None if X_warm is None else Vk.conj().T @ X_warm @ Vk
                    Xk = self._bm_solve(H @ Vk, y_target, X_warm=Xw)
                    X = Vk @ Xk @ Vk.conj().T
                # dB-OBJECTIVE COHERENCE GUARD (field 23).  With the dB objective the unconstrained
                # factored solve drives the median pairwise drive coherence (gamma^2) to 0.99-1.00 at
                # 6 drives -- fully dependent drives, the failure this file's top docstring warns
                # about; the linear objective does not.
                #   1 (default): if this line's solution breaks max_drive_coherence, re-solve THIS
                #     LINE with the SDP, which holds the cap as a constraint.
                #   2: shrink the offending cross terms after the solve (cheap, but it costs
                #     accuracy: capping afterwards breaks the inter-drive cancellation the solve
                #     relied on).
                #   0: off.
                if (self.db_coherence_guard and self.error_domain == 'db'
                        and self.max_drive_coherence < 1.0
                        and self._max_drive_coherence_of(X) > self.max_drive_coherence + 1e-6):
                    if self.db_coherence_guard == 2:
                        self.n_db_guard_caps = getattr(self, 'n_db_guard_caps', 0) + 1
                        return _cap_drive_coherence(X[None], self.max_drive_coherence)[0]
                    self.n_db_guard_sdp = getattr(self, 'n_db_guard_sdp', 0) + 1
                    self.n_safe_solves += 1
                    return super()._solve_one_bin(H, y_target, X_warm=X_warm)
                return X
            except Exception as e:
                print(f"[optimal_diagonal_control_fast] BM solve raised {type(e).__name__}: {e} "
                      f"-- falling back to SDP for this bin", flush=True)
        self.n_safe_solves += 1
        return super()._solve_one_bin(H, y_target, X_warm=X_warm)

    # ------------------------------------------------------------------
    # Explicit pass-through overrides -- required, not decorative.
    # Rattlesnake's control-law loader (components/random_vibration_sys_id_
    # environment.py) only recognizes a class as a valid class-style control
    # law if 'system_id_update' and 'control' are in the class's OWN
    # __dict__, not just inherited from a base class. Without these, this
    # class is invisible to the "Control Python Function" dropdown even
    # though it fully implements both via inheritance.
    def system_id_update(self, *args, **kwargs):
        return super().system_id_update(*args, **kwargs)

    def control(self, *args, **kwargs):
        return super().control(*args, **kwargs)
