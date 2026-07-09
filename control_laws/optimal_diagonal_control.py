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
     bins from going stale under a moving resonance.

Formulation for a single bin's SDP:
    Y(f) = H(f) X(f) H(f)^H
    minimize_X   || diag(Y) - y_diag_target ||^2 + reg * ||X||_F^2
    subject to   X ⪰ 0
Convex, no manifold optimization. H can be any M x N shape.

Interface: class-style control law (matches buzz_control_class in
control_laws.py).

extra_parameters (string): comma-separated
    "reg,frf_update_threshold,max_bins_per_update,error_threshold_db"
    reg                   - Tikhonov-style weight on ||X||_F (default 1e-6)
    frf_update_threshold  - relative Frobenius-norm change (0-1) that triggers
                             re-solving an already-refined bin (default 0.05)
    max_bins_per_update   - hard cap on SDP solves per call, split between
                             drifted-refined bins (priority) and worst-error
                             not-yet-refined bins (default 20)
    error_threshold_db    - don't bother SDP-refining a bin whose buzz
                             solution is already within this much of target
                             (default 1.0 dB)
A bare single value (no comma) is accepted too, read as `reg`.
"""

import numpy as np

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
        if extra_parameters:
            try:
                parts = [p.strip() for p in str(extra_parameters).split(',') if p.strip() != '']
                if len(parts) >= 1: self.reg = float(parts[0])
                if len(parts) >= 2: self.frf_update_threshold = float(parts[1])
                if len(parts) >= 3: self.max_bins_per_update = int(float(parts[2]))
                if len(parts) >= 4: self.error_threshold_db = float(parts[3])
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        self.output_cpsd = None    # (F, N, N) current best drive CPSD per bin
        self.H_cache = None        # (F, M, N) FRF each bin's current solution was derived from
        self.sdp_refined = None    # (F,) bool -- True once a bin has been through the SDP
        self.N = None
        self.n_frf_updates = 0       # diagnostic: drifted-refined-bin re-solves
        self.n_sdp_refinements = 0   # diagnostic: total worst-error refinements performed
        self.n_deferred = 0          # diagnostic: bins still above threshold, awaiting SDP budget
        self._initialized = False

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
    def _solve_one_bin(self, H, y_target):
        N = H.shape[1]
        X = cp.Variable((N, N), hermitian=True)
        Y = H @ X @ H.conj().T
        diagY = cp.real(cp.diag(Y))
        objective = cp.Minimize(
            cp.sum_squares(diagY - y_target) + self.reg * cp.sum_squares(cp.abs(X))
        )
        prob = cp.Problem(objective, [X >> 0])
        try:
            prob.solve(solver=cp.SCS)
            Xf = X.value
        except (cp.error.SolverError, TypeError):
            Xf = None
        if Xf is None:
            Hpinv = np.linalg.pinv(H, rcond=1e-15)
            Yspec = np.diag(y_target).astype(complex)
            Xf = Hpinv @ Yspec @ Hpinv.conj().T
        return Xf

    # ------------------------------------------------------------------
    def _initialize(self, transfer_function, sysid_response_cpsd=None):
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self.N = H_clean.shape[2]
        self.output_cpsd = self._buzz_solve_all(H_clean, sysid_response_cpsd)
        self.H_cache = H_clean.copy()
        self.sdp_refined = np.zeros(self.F, dtype=bool)
        self._initialized = True
        self._refine_batch(H_clean)  # spend the first batch of SDP budget immediately

    def _refine_batch(self, transfer_function):
        """
        Spend up to max_bins_per_update SDP solves this call:
          1) previously-refined bins whose FRF has drifted (priority --
             protects already-optimized bins from going stale)
          2) remaining budget on not-yet-refined bins with the largest
             predicted diagonal error (skip anything already under
             error_threshold_db -- buzz is good enough there)
        """
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        budget = self.max_bins_per_update

        # --- Step 1: re-solve drifted, previously-refined bins ---
        refined_idx = np.where(self.sdp_refined)[0]
        if refined_idx.size > 0:
            sub_new = H_clean[refined_idx]
            sub_old = self.H_cache[refined_idx]
            num = np.linalg.norm((sub_new - sub_old).reshape(refined_idx.size, -1), axis=1)
            den = np.linalg.norm(sub_old.reshape(refined_idx.size, -1), axis=1) + 1e-30
            drifted = refined_idx[(num / den) > self.frf_update_threshold]
        else:
            drifted = np.array([], dtype=int)

        n_fix = min(drifted.size, budget)
        for f in drifted[:n_fix]:
            self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self.y_diag_target[f])
            self.H_cache[f] = H_clean[f]
        if n_fix > 0:
            self.n_frf_updates += int(n_fix)
        budget -= n_fix

        # --- Step 2: worst-error not-yet-refined bins get remaining budget ---
        not_refined = np.where(~self.sdp_refined)[0]
        self.n_deferred = 0
        if not_refined.size > 0:
            Y = np.einsum('fmn,fnk,flk->fml',
                           H_clean[not_refined], self.output_cpsd[not_refined],
                           H_clean[not_refined].conj())
            achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
            target = np.maximum(self.y_diag_target[not_refined], 1e-30)
            err_db = np.max(np.abs(10 * np.log10(achieved / target)), axis=1)

            above = not_refined[err_db > self.error_threshold_db]
            above_err = err_db[err_db > self.error_threshold_db]
            order = above[np.argsort(-above_err)]

            n_refine = min(order.size, budget)
            self.n_deferred = int(order.size - n_refine)
            for f in order[:n_refine]:
                self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self.y_diag_target[f])
                self.sdp_refined[f] = True
                self.H_cache[f] = H_clean[f]
            self.n_sdp_refinements += int(n_refine)

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
            return self.output_cpsd
        if transfer_function is not None:
            self._refine_batch(transfer_function)
        return self.output_cpsd