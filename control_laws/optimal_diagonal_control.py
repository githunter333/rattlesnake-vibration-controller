"""
optimal_diagonal_control.py

Rattlesnake control law that targets ONLY the diagonal (auto-spectra) of the
response specification, choosing the drive CPSD's cross terms (coherence,
phase) to minimize diagonal error rather than fixing them from an
independent-drive survey.

STRATEGY (this is the important part): solving the full per-bin SDP for
every frequency line up front is expensive (~20-30 ms/bin -- 5-10+ seconds
across a typical band). Instead:

  1. INSTANT BASELINE: every bin gets Rattlesnake's own "buzz control"
     solution first -- a plain closed-form H+ synthesis using the spec's
     target diagonal with cross terms (coherence/phase) taken from a
     measured buzz/survey CPSD if available (falls back to diagonal-only,
     i.e. zero cross terms, if not). This is the exact logic from the
     built-in buzz_control_class.match_coherence_phase, reused here. No
     optimization, just linear algebra -- effectively instant even across
     hundreds of bins.

  2. TARGETED SDP REFINEMENT: predicted diagonal error (buzz X vs. target,
     through the actual H) is computed for every not-yet-refined bin. The
     `max_bins_per_update` worst-error bins (above `error_threshold_db`)
     get the full SDP treatment this call, replacing their buzz solution
     with the optimal one. Bins already close enough under buzz are left
     alone -- no SDP budget wasted on bins that don't need it.

  3. PROGRESSIVE: this repeats on every system_id_update()/control() call,
     so the worst-offending frequency lines get optimized first, and the
     control law keeps chipping away at the remainder (a handful of bins
     per call) until the whole band is either SDP-refined or already good
     enough under buzz. Control can start on iteration 1 with a full-
     spectrum (buzz-quality) result rather than waiting for one large
     up-front solve.

  4. FRF drift: once a bin IS SDP-refined, if the live FRF (if Rattlesnake's
     "Update Transfer Function During Control" is enabled) drifts past
     `frf_update_threshold` for that bin, it gets re-solved with priority
     over any remaining not-yet-refined bins -- protects already-optimized
     bins from going stale under a moving resonance. Capped at half of
     `max_bins_per_update` so this can never fully starve step 2.

  5. STALE-ERROR RE-TRIAGE: step 4's drift check is a proxy (has H moved a
     lot?), not the real question (is this bin's accuracy bad now?) -- a bin
     can drift by less than `frf_update_threshold` while still degrading
     enough to matter, and once every bin is SDP-refined (n_deferred == 0),
     step 2 never runs again, so step 4's proxy becomes the ONLY route back
     to re-optimization. Without this step, such a bin is orphaned on its
     stale solution forever, no matter how long the test runs. Step 5 closes
     that gap: with whatever budget remains after steps 4 and 2, it
     re-evaluates ALREADY-refined bins' predicted error against the CURRENT
     H (not H-drift magnitude) and re-solves the worst-error ones, same
     worst-first triage as step 2 but applied to the refined population too.
     This only ever spends budget step 2 wasn't using, so it changes nothing
     about a fresh start's convergence (step 2 is already using the full
     budget on new bins there) -- it only matters once coverage is complete.

Formulation for a single bin's SDP:
    Y(f) = H(f) X(f) H(f)^H
    minimize_X   || diag(Y) - y_diag_target ||^2 + reg * ||X||_F^2
    subject to   X ⪰ 0
                 |X_ij| <= max_drive_coherence * sqrt(X_ii * X_jj)   for all i != j
The coherence cap only applies to SDP-refined bins (this formulation's decision
variable IS the drive CPSD). The buzz baseline is left uncapped on purpose --
it already draws its cross terms from an independent-drive survey, so it has
no dependent-drive problem to begin with. Both constraints are convex (the
coherence one is |affine| <= concave, i.e. convex <= concave), no manifold
optimization. H can be any M x N shape.

Interface: class-style control law (matches buzz_control_class in
control_laws.py).

extra_parameters (string): comma-separated
    "reg,frf_update_threshold,max_bins_per_update,error_threshold_db,max_drive_coherence"
    reg                   - Tikhonov-style weight on ||X||_F (default 1e-6)
    frf_update_threshold  - relative Frobenius-norm change (0-1) that triggers
                             re-solving an already-refined bin (default 0.05)
    max_bins_per_update   - hard cap on SDP solves per call, split between
                             drifted-refined bins (priority) and worst-error
                             not-yet-refined bins (default 20)
    error_threshold_db    - don't bother SDP-refining a bin whose buzz
                             solution is already within this much of target
                             (default 1.0 dB)
    max_drive_coherence   - hard cap (0-1) on pairwise coherence between any
                             two SDP-solved drive channels, so the optimizer
                             can't converge on totally dependent drives
                             (default 0.95; 1.0 disables the cap)
    bm_rank               - read only by optimal_diagonal_control_fast; this
                             class parses past it so a single parameter
                             string works for both (default 4 there)
    startup_test_level_cap_db
                          - ceiling on the VERY FIRST control command, in dB
                             relative to the specification's own trace
                             (default -9.0; 0 dB = full spec-match). Added
                             2026-09-17 to match every law in
                             control_laws.py. The clamp is on the predicted
                             response, trace(H X H^H), not on the drive, and
                             it is applied to the RETURNED command only --
                             the stored solution self.output_cpsd is left
                             unscaled so the SDP refinement continues from
                             the real solve.
                             CAVEAT: this law is OPEN LOOP, so the ceiling
                             binds on the first command only; cycle 2 returns
                             the uncapped solution. It guards the one command
                             issued before anyone has seen the rig respond,
                             not the level of the run.
A bare single value (no comma) is accepted too, read as `reg`.
"""

import numpy as np

# Rattlesnake loads a control-law file with importlib.spec_from_file_location,
# i.e. as a standalone module with NO package context, so a relative import
# raises ImportError and the law cannot be loaded at all.  (Same applies when
# optimal_diagonal_control_fast.py loads THIS file that way.)  Fall back to an
# explicit path import without touching sys.path -- putting control_laws/ on
# the path would shadow the package of the same name for everything else.
try:
    from .control_laws import _apply_startup_level_cap_from_trace
except ImportError:
    import importlib.util as _ilu
    import os as _os
    _sib = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         'control_laws.py')
    _hspec = _ilu.spec_from_file_location('_rattlesnake_control_laws_helpers',
                                          _sib)
    _hmod = _ilu.module_from_spec(_hspec)
    _hspec.loader.exec_module(_hmod)
    _apply_startup_level_cap_from_trace = _hmod._apply_startup_level_cap_from_trace

try:
    import cvxpy as cp
except ImportError as e:
    raise ImportError(
        "optimal_diagonal_control requires cvxpy. Install it in the "
        "rattlesnake conda env with:\n"
        "    pip install cvxpy --break-system-packages\n"
        "or\n"
        "    conda install -c conda-forge cvxpy"
    ) from e


class optimal_diagonal_control:
    def __init__(self,
                 specification: np.ndarray,       # Specifications
                 warning_levels: np.ndarray,       # Warning levels
                 abort_levels: np.ndarray,         # Abort Levels
                 extra_parameters: str,            # Extra parameters for the control law
                 transfer_function: np.ndarray = None,
                 noise_response_cpsd: np.ndarray = None,
                 noise_reference_cpsd: np.ndarray = None,
                 sysid_response_cpsd: np.ndarray = None,
                 sysid_reference_cpsd: np.ndarray = None,
                 multiple_coherence: np.ndarray = None,
                 frames=None,
                 total_frames=None,
                 # ACCEPTED AND NEVER READ. This law is OPEN LOOP: it refines
                 # each bin against its own PREDICTED response,
                 # diag(H X H^H), not against the measured response. Nothing
                 # in this class or optimal_diagonal_control_fast reads
                 # last_response_cpsd or last_output_cpsd -- verified
                 # 2026-09-17, they appear only in signatures. The only thing
                 # that changes between calls is H.
                 last_response_cpsd: np.ndarray = None,
                 last_output_cpsd: np.ndarray = None,
                 ):
        y_diag_target = np.real(np.einsum('fmm->fm', specification))
        self.y_diag_target = np.nan_to_num(y_diag_target, nan=0.0, posinf=0.0, neginf=0.0)
        self.F, self.M = self.y_diag_target.shape

        self.reg = 1e-6
        self.frf_update_threshold = 0.05
        self.max_bins_per_update = 20
        self.error_threshold_db = 1.0
        self.max_drive_coherence = 0.95
        self.startup_test_level_cap_db = -9.0
        # 'linear' is the DEFAULT and reproduces runs 03-06 exactly.  'db' is
        # the 2026-09-17 objective change -- better pooled rms, but it spreads
        # error across every channel instead of sacrificing the two the plant
        # cannot reach, which is not what this rig wants (see the 8th
        # extra_parameters field).  Opt in with an 8th value of 1.
        self.error_domain = 'linear'
        self.n_irls_passes = 2
        # Drive-subspace restriction for the FACTORED path only -- see
        # optimal_diagonal_control_fast._solve_one_bin.  Parsed here so one
        # parameter string serves both classes.
        #
        # NOW A MULTIPLE of the plant's own singular-value spread, not an
        # absolute ratio -- see optimal_diagonal_control_fast.
        # _effective_drive_rcond.  Runs 22-25 swept the absolute value on the
        # rig at 0 / 1e-2 / 3e-2 / 5e-2 and 3e-2 was a genuine minimum; this
        # frame's smallest structural direction is sigma_5/sigma_1 = 0.0152,
        # so 3e-2 is 2.0x that.  Expressed as a multiple it tracks a
        # differently conditioned article without re-tuning.  Negative means
        # an absolute ratio, |value|.
        self.drive_rcond = 2.0
        if extra_parameters:
            try:
                parts = [p.strip() for p in str(extra_parameters).split(',') if p.strip() != '']
                if len(parts) >= 1: self.reg = float(parts[0])
                if len(parts) >= 2: self.frf_update_threshold = float(parts[1])
                if len(parts) >= 3: self.max_bins_per_update = int(float(parts[2]))
                if len(parts) >= 4: self.error_threshold_db = float(parts[3])
                if len(parts) >= 5: self.max_drive_coherence = float(parts[4])
                # parts[5] is bm_rank, read by optimal_diagonal_control_fast
                if len(parts) >= 7: self.startup_test_level_cap_db = float(parts[6])
                if len(parts) >= 8: self.error_domain = (
                    'linear' if float(parts[7]) == 0 else 'db')
                if len(parts) >= 9: self.n_irls_passes = max(0, int(float(parts[8])))
                if len(parts) >= 10: self.drive_rcond = float(parts[9])
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        self.output_cpsd = None    # (F, N, N) current best drive CPSD per bin
        # Console throttling -- see _should_log.  Once every bin is solved and
        # the FRF is frozen, _refine_batch does no work but still ran two print
        # statements per control cycle, which on run 14 buried the terminal in
        # hundreds of identical lines and read as a hung process.  (It was not
        # hung: newly_refined=0, n_deferred=0, cum_sdp_refinements frozen.)
        self._idle_repeats = 0
        self._log_every_when_idle = 50
        self.H_cache = None        # (F, M, N) FRF each bin's current solution was derived from
        self.sdp_refined = None    # (F,) bool -- True once a bin has been through the SDP
        self.err_db_cache = None   # (F,) achieved dB error each bin had the last time it was solved --
                                    # lets step 3 detect "got worse since I last touched this bin" instead
                                    # of "isn't perfect" (many bins, e.g. near a structural null, can never
                                    # get under error_threshold_db no matter how many times they're re-solved)
        self.N = None
        self._sdp_prob = None        # cached parametrized cp.Problem, built once (see _build_sdp_problem)
        self._sdp_X = None
        self._sdp_W_params = None
        self._sdp_y_param = None
        self._sdp_shape = None       # (M, N) the cached problem was built for
        self.n_frf_updates = 0       # diagnostic: drifted-refined-bin re-solves
        self.n_sdp_refinements = 0   # diagnostic: total worst-error refinements performed
        self.n_deferred = 0          # diagnostic: never-refined bins still above threshold, awaiting SDP budget
        self.n_stale_refinements = 0 # diagnostic: already-refined bins re-solved due to stale predicted error
        self.n_stale_deferred = 0    # diagnostic: stale already-refined bins still above threshold, awaiting budget
        self.n_solver_failures = 0   # diagnostic: SDP solves that fell back to pinv
        self._initialized = False
        self._n_calls = 0            # diagnostic: total system_id_update()/control() calls

        print(f"[optimal_diagonal_control] __init__: F={self.F} M={self.M} "
              f"reg={self.reg:g} frf_update_threshold={self.frf_update_threshold:g} "
              f"max_bins_per_update={self.max_bins_per_update} "
              f"error_threshold_db={self.error_threshold_db:g} "
              f"max_drive_coherence={self.max_drive_coherence:g} "
              f"transfer_function={'given' if transfer_function is not None else 'None'} "
              f"sysid_response_cpsd={'given' if sysid_response_cpsd is not None else 'None'}",
              flush=True)

        if transfer_function is not None:
            self._initialize(transfer_function, sysid_response_cpsd)

    # ------------------------------------------------------------------
    # Fast "buzz" baseline (Rattlesnake's own match_coherence_phase logic)
    # ------------------------------------------------------------------
    def _cpsd_coherence(self, cpsd):
        """cpsd: (M,M) complex for one bin -> (M,M) real coherence matrix."""
        num = np.abs(cpsd) ** 2
        d = np.real(np.diagonal(cpsd))
        den = np.outer(d, d)
        den = np.where(den == 0.0, 1.0, den)
        return num / den

    def _match_coherence_phase(self, target_diag, cpsd_to_match):
        """target_diag: (M,) real target autospectra for this bin.
        cpsd_to_match: (M,M) measured CPSD (e.g. sysid_response_cpsd[f]) to
        pull coherence/phase from. Returns (M,M) complex modified spec with
        the target diagonal but the measured cross-term structure."""
        coh = np.clip(self._cpsd_coherence(cpsd_to_match), 0.0, 1.0)
        phs = np.angle(cpsd_to_match)
        asd_outer = np.outer(target_diag, target_diag)
        magnitude = np.sqrt(np.clip(coh * asd_outer, 0.0, None))
        return magnitude * np.exp(1j * phs)

    def _buzz_solve_all(self, H_clean, sysid_response_cpsd):
        """Instant closed-form baseline for every bin."""
        F, M, N = H_clean.shape
        output = np.zeros((F, N, N), dtype=complex)
        sysid_clean = None
        if sysid_response_cpsd is not None:
            sysid_clean = np.nan_to_num(sysid_response_cpsd, nan=0.0, posinf=0.0, neginf=0.0)
        for f in range(F):
            Hpinv = np.linalg.pinv(H_clean[f], rcond=1e-12)
            if sysid_clean is not None:
                modified_spec = self._match_coherence_phase(self.y_diag_target[f], sysid_clean[f])
            else:
                modified_spec = np.diag(self.y_diag_target[f]).astype(complex)
            output[f] = Hpinv @ modified_spec @ Hpinv.conj().T
        return output

    # ------------------------------------------------------------------
    # SDP refinement for a single bin
    # ------------------------------------------------------------------
    def _build_sdp_problem(self, M, N):
        """Build the per-bin SDP ONCE (shape is fixed for the life of a run)
        and reuse it across every bin via cp.Parameter, instead of paying
        DCP canonicalization on every solve. H can't be a Parameter directly
        -- diag(H X H^H) is bilinear in H (H appears on both sides of X),
        which isn't DPP-affine-in-parameter. Instead each response channel
        m's diagonal term is reformulated as a linear functional of X:
            diag(H X H^H)[m] = h_m X h_m^H = sum_kl H[m,k] conj(H[m,l]) X[k,l]
                              = real(sum(W_m .* X)),  W_m = outer(h_m, conj(h_m))
        so W_m (computed in plain numpy per bin, negligible cost) is what's
        fed in as a Parameter -- now a straightforward "coefficient .* variable"
        pattern, which DPP allows."""
        X = cp.Variable((N, N), hermitian=True)
        W_params = [cp.Parameter((N, N), complex=True) for _ in range(M)]
        y_target_param = cp.Parameter(M)
        diagY = cp.hstack([
            cp.real(cp.sum(cp.multiply(W_params[m], X))) for m in range(M)
        ])
        # The objective's FORM is unchanged.  IRLS weighting is applied by
        # scaling the values fed into the existing parameters, not by adding a
        # weight Parameter: setting W_m <- w_m * outer(h_m, conj(h_m)) makes
        # diagY[m] equal w_m * yhat_m, and y_target_param <- w * y then gives
        # sum_m w_m^2 (yhat_m - y_m)^2 exactly.  A separate weight Parameter
        # multiplying this expression is parameter-times-parameter and cvxpy
        # rejects it as non-DPP (tried 2026-09-17, "SDP problem is not
        # DPP-compliant").
        objective = cp.Minimize(
            cp.sum_squares(diagY - y_target_param) + self.reg * cp.sum_squares(cp.abs(X))
        )
        constraints = [X >> 0]
        if self.max_drive_coherence < 1.0:
            # |X_ij| <= max_drive_coherence * sqrt(X_ii * X_jj) for every drive
            # pair -- keeps the SDP from converging on totally dependent
            # (coherence ~1) drive channels. X >> 0 already implies coherence
            # <= 1 for free (Cauchy-Schwarz); this just tightens that bound.
            for i in range(N):
                for j in range(i + 1, N):
                    constraints.append(
                        cp.abs(X[i, j]) <= self.max_drive_coherence
                        * cp.geo_mean(cp.hstack([cp.real(X[i, i]), cp.real(X[j, j])]))
                    )
        prob = cp.Problem(objective, constraints)
        assert prob.is_dcp(dpp=True), "SDP problem is not DPP-compliant"
        return prob, X, W_params, y_target_param

    def _irls_weights(self, H, X, y_target):
        """Relative-error weights for the next IRLS pass: w_m = y_m / yhat_m.

        Minimizing sum_m (yhat_m - y_m)^2 in LINEAR units caps what abandoning
        a channel can cost you at that channel's own target squared, so the
        solver sells the hardest channels to buy the easy ones.  In dB the
        penalty has no such ceiling.  Weighting each residual by 1/yhat_m is
        the Gauss-Newton step for least squares on log(yhat/y), so iterating
        solve -> reweight -> solve converges on the dB objective while every
        subproblem stays a convex SDP -- which is what keeps the coherence-cap
        constraint, and the whole "demote to the safe path" design, intact.

        Normalized by y_m rather than left as 1/yhat_m so that w = 1 exactly
        when a channel is on target.  That keeps the weighted problem on the
        same scale as the unweighted one, so self.reg means what it meant
        before and does not have to be re-tuned.
        """
        yhat = np.real(np.einsum('mn,nk,mk->m', H, X, H.conj()))
        with np.errstate(divide='ignore', invalid='ignore'):
            w = y_target/yhat
        bad = (~np.isfinite(w)) | (w <= 0) | (yhat <= 0) | (y_target <= 0)
        w = np.where(bad, 1.0, w)
        # A channel the current iterate has abandoned by orders of magnitude
        # would otherwise get an enormous weight and make the next subproblem
        # about nothing else.  Clip to a 20 dB span either way.
        return np.clip(w, 0.1, 10.0)

    def _solve_one_bin(self, H, y_target, X_warm=None):
        M, N = H.shape
        if self._sdp_prob is None or self._sdp_shape != (M, N):
            self._sdp_prob, self._sdp_X, self._sdp_W_params, self._sdp_y_param = \
                self._build_sdp_problem(M, N)
            self._sdp_shape = (M, N)

        W_unweighted = [np.outer(H[m, :], H[m, :].conj()) for m in range(M)]

        # Pass 0 is unweighted unless a previous solution for this bin is in
        # hand, in which case its weights are already informative and the
        # first pass is a real IRLS step rather than a wasted one.
        if self.error_domain == 'db' and X_warm is not None:
            w = self._irls_weights(H, X_warm, y_target)
        else:
            w = np.ones(M)
        n_passes = 1 + (self.n_irls_passes if self.error_domain == 'db' else 0)

        Xf = None
        for _ in range(n_passes):
            for m in range(M):
                self._sdp_W_params[m].value = w[m]*W_unweighted[m]
            self._sdp_y_param.value = w*y_target
            try:
                self._sdp_prob.solve(solver=cp.CLARABEL, warm_start=True)
                X_try = self._sdp_X.value
                if X_try is None:
                    print(f"[optimal_diagonal_control] SDP solve returned no value, "
                          f"status={self._sdp_prob.status!r} -- falling back to pinv", flush=True)
                    break
            except Exception as e:
                print(f"[optimal_diagonal_control] SDP solve raised {type(e).__name__}: {e} "
                      f"-- falling back to pinv", flush=True)
                break
            Xf = X_try
            if self.error_domain != 'db':
                break
            w = self._irls_weights(H, Xf, y_target)
        if Xf is None:
            self.n_solver_failures += 1
            Hpinv = np.linalg.pinv(H, rcond=1e-15)
            Yspec = np.diag(y_target).astype(complex)
            Xf = Hpinv @ Yspec @ Hpinv.conj().T
        return Xf

    def _err_db(self, H_clean, indices):
        """Per-bin max-over-channel dB diagonal error for the given bin
        indices, using the CURRENT output_cpsd -- shared by steps 2 and 3."""
        Y = np.einsum('fmn,fnk,flk->fml', H_clean[indices], self.output_cpsd[indices],
                       H_clean[indices].conj())
        achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
        target = np.maximum(self.y_diag_target[indices], 1e-30)
        return np.max(np.abs(10 * np.log10(achieved / target)), axis=1)

    # ------------------------------------------------------------------
    def _initialize(self, transfer_function, sysid_response_cpsd=None):
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self.N = H_clean.shape[2]
        print(f"[optimal_diagonal_control] _initialize: H shape={H_clean.shape} "
              f"(F,M,N), sysid_response_cpsd={'given' if sysid_response_cpsd is not None else 'None'}, "
              f"H any-nan-in-input={bool(np.any(~np.isfinite(transfer_function)))}", flush=True)
        self.output_cpsd = self._buzz_solve_all(H_clean, sysid_response_cpsd)
        self.H_cache = H_clean.copy()
        self.H_initial = H_clean.copy()  # diagnostic: fixed baseline to measure cumulative drift against
        self.sdp_refined = np.zeros(self.F, dtype=bool)
        self.err_db_cache = np.full(self.F, np.inf)  # unset until a bin is actually SDP-solved
        self._initialized = True
        self._refine_batch(H_clean)  # spend the first batch of SDP budget immediately

    def _refine_batch(self, transfer_function):
        """
        Spend up to max_bins_per_update SDP solves this call:
          1) previously-refined bins whose FRF has drifted (priority --
             protects already-optimized bins from going stale), capped at
             half the budget
          2) not-yet-refined bins with the largest predicted diagonal error
             (skip anything already under error_threshold_db -- buzz is
             good enough there)
          3) whatever budget remains: already-refined bins whose error has
             gotten worse than it was at their own last solve (not judged
             against the absolute error_threshold_db, since some bins can
             never get under that no matter how many times they're
             re-solved) -- closes the gap where step 2 can never reconsider
             a bin once every bin has been refined at least once
        """
        self._n_calls += 1
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        H_changed = self.H_cache is None or not np.array_equal(H_clean, self.H_cache)
        budget = self.max_bins_per_update

        # --- Step 1: re-solve drifted, previously-refined bins ---
        # Capped at half the budget so persistent (or noise-driven false-
        # positive) drift can never fully starve step 2's new-bin coverage --
        # without this cap, small run-to-run FRF jitter that keeps tripping
        # frf_update_threshold on already-refined bins can claim the entire
        # budget every call, forever, leaving n_deferred stuck.
        drift_budget = max(1, self.max_bins_per_update // 2)
        refined_idx = np.where(self.sdp_refined)[0]
        if refined_idx.size > 0:
            sub_new = H_clean[refined_idx]
            sub_old = self.H_cache[refined_idx]
            num = np.linalg.norm((sub_new - sub_old).reshape(refined_idx.size, -1), axis=1)
            den = np.linalg.norm(sub_old.reshape(refined_idx.size, -1), axis=1) + 1e-30
            since_last_ratio = num / den
            drifted = refined_idx[since_last_ratio > self.frf_update_threshold]
            # diagnostic: drift relative to the very first H seen, not just since
            # this bin's last solve -- reveals whether H is oscillating around a
            # fixed baseline (bounded) or walking away from it (runaway/unbounded)
            sub_init = self.H_initial[refined_idx]
            num0 = np.linalg.norm((sub_new - sub_init).reshape(refined_idx.size, -1), axis=1)
            den0 = np.linalg.norm(sub_init.reshape(refined_idx.size, -1), axis=1) + 1e-30
            since_init_ratio = num0 / den0
            if self._idle_repeats < 3:
              print(f"[optimal_diagonal_control] H drift on refined bins: "
                  f"since_last_solve[median={np.median(since_last_ratio):.4f}, max={np.max(since_last_ratio):.4f}], "
                  f"since_initial[median={np.median(since_init_ratio):.4f}, max={np.max(since_init_ratio):.4f}]",
                  flush=True)
        else:
            drifted = np.array([], dtype=int)

        n_fix = min(drifted.size, drift_budget, budget)
        fixed_this_call = drifted[:n_fix]
        for f in fixed_this_call:
            self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self.y_diag_target[f],
                                                     X_warm=self.output_cpsd[f])
            self.H_cache[f] = H_clean[f]
        if n_fix > 0:
            self.err_db_cache[fixed_this_call] = self._err_db(H_clean, fixed_this_call)
            self.n_frf_updates += int(n_fix)
        budget -= n_fix

        # --- Step 2: worst-error not-yet-refined bins get remaining budget ---
        not_refined = np.where(~self.sdp_refined)[0]
        self.n_deferred = 0
        n_refine = 0
        if not_refined.size > 0:
            err_db = self._err_db(H_clean, not_refined)

            above = not_refined[err_db > self.error_threshold_db]
            above_err = err_db[err_db > self.error_threshold_db]
            order = above[np.argsort(-above_err)]

            n_refine = min(order.size, budget)
            self.n_deferred = int(order.size - n_refine)
            for f in order[:n_refine]:
                self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self.y_diag_target[f],
                                                     X_warm=self.output_cpsd[f])
                self.sdp_refined[f] = True
                self.H_cache[f] = H_clean[f]
            if n_refine > 0:
                self.err_db_cache[order[:n_refine]] = self._err_db(H_clean, order[:n_refine])
            self.n_sdp_refinements += int(n_refine)
            budget -= n_refine

        # --- Step 3: stale-error re-triage on ALREADY-refined bins, using
        # whatever budget steps 1 and 2 didn't spend. Step 1 only catches
        # bins whose H moved a lot (>frf_update_threshold); a bin can drift
        # less than that and still be hurting accuracy, and once every bin
        # is refined (not_refined empty forever) step 2 can never reconsider
        # it again -- this closes that gap by re-evaluating refined bins'
        # predicted error against the CURRENT H, worst-first, same as step 2.
        # Staleness is judged against each bin's OWN error at its last solve
        # (err_db_cache), not the absolute error_threshold_db -- many bins
        # (e.g. near a genuine structural null) can never get under that
        # absolute threshold no matter how many times they're re-solved, and
        # comparing to an absolute floor would have this step burn its whole
        # budget forever re-solving bins that were never going to improve,
        # instead of the bins that actually got worse since a real FRF change.
        n_stale = 0
        self.n_stale_deferred = 0
        refined_idx = np.where(self.sdp_refined)[0]
        if n_fix > 0:
            refined_idx = refined_idx[~np.isin(refined_idx, fixed_this_call)]
        if budget > 0 and refined_idx.size > 0:
            err_db_r = self._err_db(H_clean, refined_idx)
            degraded = err_db_r > self.err_db_cache[refined_idx] + self.error_threshold_db

            stale = refined_idx[degraded]
            stale_err = err_db_r[degraded]
            stale_order = stale[np.argsort(-stale_err)]

            n_stale = min(stale_order.size, budget)
            self.n_stale_deferred = int(stale_order.size - n_stale)
            for f in stale_order[:n_stale]:
                self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self.y_diag_target[f],
                                                     X_warm=self.output_cpsd[f])
                self.H_cache[f] = H_clean[f]
            if n_stale > 0:
                self.err_db_cache[stale_order[:n_stale]] = self._err_db(H_clean, stale_order[:n_stale])
            self.n_stale_refinements += int(n_stale)

        # Self-consistency check: what does the control law's OWN H estimate
        # and current solution predict the achieved diagonal error to be?
        # If this stays near 0 dB while Rattlesnake's live measured Response
        # Error panel stays high, the gap is a model mismatch between the
        # system-ID H used here and the true live system -- not a bug in the
        # SDP/scheduling logic, which would be self-consistent by construction.
        in_band = np.any(self.y_diag_target > 0, axis=1)
        if np.any(in_band):
            Y_all = np.einsum('fmn,fnk,flk->fml', H_clean[in_band], self.output_cpsd[in_band],
                               H_clean[in_band].conj())
            achieved_all = np.maximum(np.real(np.einsum('fmm->fm', Y_all)), 1e-30)
            target_all = np.maximum(self.y_diag_target[in_band], 1e-30)
            err_db_all = 10 * np.log10(achieved_all / target_all)
            self_rms_per_channel = np.sqrt(np.mean(err_db_all ** 2, axis=0))
        else:
            self_rms_per_channel = np.zeros(self.M)

        # A call that solved nothing, deferred nothing and saw no FRF movement
        # has nothing to report that the previous line did not already say.
        # Log the first few, then one in every _log_every_when_idle, so the
        # loop stays visibly alive without flooding.  Any real activity resets
        # the counter and restores full logging immediately.
        idle = (not H_changed and n_fix == 0 and n_refine == 0 and n_stale == 0
                and self.n_deferred == 0 and self.n_stale_deferred == 0)
        if idle:
            self._idle_repeats += 1
        else:
            if self._idle_repeats > 0:
                print(f"[optimal_diagonal_control] ...{self._idle_repeats} idle "
                      f"call(s) suppressed; work resumed", flush=True)
            self._idle_repeats = 0
        if idle and self._idle_repeats == 4:
            print(f"[optimal_diagonal_control] every bin solved and the FRF is "
                  f"static -- nothing left to do. Logging one call in "
                  f"{self._log_every_when_idle} from here. THIS IS NOT A HANG.",
                  flush=True)
        if (idle and self._idle_repeats > 3
                and self._idle_repeats % self._log_every_when_idle != 0):
            return

        print(f"[optimal_diagonal_control] _refine_batch call #{self._n_calls}: "
              f"H_changed_since_last={H_changed}, drifted_resolved={n_fix}, "
              f"newly_refined={n_refine}, n_deferred={self.n_deferred}, "
              f"stale_resolved={n_stale}, n_stale_deferred={self.n_stale_deferred}, "
              f"cum_sdp_refinements={self.n_sdp_refinements}, cum_frf_updates={self.n_frf_updates}, "
              f"cum_stale_refinements={self.n_stale_refinements}, "
              f"cum_solver_failures={self.n_solver_failures}, n_refined_total={int(np.sum(self.sdp_refined))}/{self.F}, "
              f"self_predicted_rms_db_per_channel={np.array2string(self_rms_per_channel, precision=2)}",
              flush=True)

    # ------------------------------------------------------------------
    def system_id_update(self,
                          transfer_function: np.ndarray = None,
                          noise_response_cpsd: np.ndarray = None,
                          noise_reference_cpsd: np.ndarray = None,
                          sysid_response_cpsd: np.ndarray = None,
                          sysid_reference_cpsd: np.ndarray = None,
                          multiple_coherence: np.ndarray = None,
                          frames=None,
                          total_frames=None,
                          ):
        if transfer_function is None:
            return
        if not self._initialized:
            self._initialize(transfer_function, sysid_response_cpsd)
        else:
            self._refine_batch(transfer_function)

    def control(self,
                transfer_function: np.ndarray = None,
                multiple_coherence: np.ndarray = None,
                frames=None,
                total_frames=None,
                last_response_cpsd: np.ndarray = None,
                last_output_cpsd: np.ndarray = None) -> np.ndarray:
        if not self._initialized:
            self._initialize(transfer_function, None)
            return self._startup_capped(self.output_cpsd, transfer_function,
                                        last_output_cpsd)
        if transfer_function is not None:
            self._refine_batch(transfer_function)
        return self._startup_capped(self.output_cpsd, transfer_function,
                                    last_output_cpsd)

    def _startup_capped(self, output, transfer_function, last_output_cpsd):
        """Ceiling on the first command only -- see
        startup_test_level_cap_db in this module's docstring.  Scales a COPY
        on the way out; self.output_cpsd keeps the unscaled solve so the SDP
        refinement is not dragged down with it.  Must be the last thing done
        to the returned command (the drive-coherence cap is already inside
        the solve here, as an SDP constraint, so nothing runs after this)."""
        if last_output_cpsd is not None or output is None:
            return output
        H = transfer_function if transfer_function is not None else self.H_cache
        if H is None:
            return output
        return _apply_startup_level_cap_from_trace(
            output, np.sum(self.y_diag_target, axis=1), H,
            self.startup_test_level_cap_db)