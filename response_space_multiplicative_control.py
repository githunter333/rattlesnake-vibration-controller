"""
response_space_multiplicative_control.py

Rattlesnake control law implementing closed-loop response-space feedback
control, now with adaptive cross-term (coherence/phase) structure -- not
just diagonal magnitude correction.

WHY THIS CHANGED FROM THE ORIGINAL VERSION: the original design tracked a
response-space Cholesky factor W_resp, initialized from a purely diagonal
target, and updated by scaling entire rows by a scalar gain each iteration.
That update rule can PROVABLY never introduce off-diagonal content -- scaling
a row of a diagonal matrix stays diagonal, forever, no matter how many
iterations run (verified directly: off-diagonal energy stays exactly 0.0
under that rule). That capped this law's achievable quality well below
optimal_diagonal_control's SDP, which explicitly searches over cross-term
structure, and made it fragile (needing heavy alpha damping) whenever the
real plant had meaningful cross-coupling the diagonal-only target couldn't
represent.

NEW DESIGN: track two things per frequency bin instead of one factor:
  - current_diag(f)   : (M,) running per-DOF target magnitude (same
                        multiplicative update as before)
  - rho(f)            : (M,M) running COMPLEX correlation-coefficient
                        estimate (diag = 1, off-diag = coherence & phase
                        between response DOFs), blended each iteration
                        toward whatever correlation structure is actually
                        being measured in last_response_cpsd:
                            rho_meas_ij = CPSD_ij / sqrt(CPSD_ii * CPSD_jj)
                            rho <- (1-beta)*rho + beta*rho_meas
Reconstructing in complex-correlation space (rather than blending magnitude
and phase separately) avoids phase-wraparound issues.

Each control() call:
    Yss(f) = outer(sqrt(current_diag(f)), sqrt(current_diag(f))) * rho(f)
    Yss(f) <- PSD-projected (Hermitian-symmetrized, negative eigenvalues
              clipped to 0) -- a naive correlation-based reconstruction
              isn't guaranteed PSD, so this is a hard safety step, not
              optional
    X(f)   = H+(f) Yss(f) H+(f)^H

rho starts at the identity (uncorrelated, i.e. behaves exactly like the old
diagonal-only law) and only develops structure as real measured correlation
comes in -- so this strictly extends the old behavior rather than replacing
it; with beta=0 it reduces to the original law exactly.

FRF drift tracking (from the previous version) is unchanged: a single
Tikhonov-regularized H+ is derived once, and re-derived only for frequency
bins where a live-updating FRF ("Update Transfer Function During Control")
has drifted past frf_update_threshold.

extra_parameters (string): comma-separated
    "alpha,alpha_reg,clip_db,frf_update_threshold,beta"
    alpha                 - diagonal update damping, 0-1 (default 0.5)
    alpha_reg             - Tikhonov regularization weight for H+ (default 0.01)
    clip_db               - max per-iteration diagonal correction in dB (default 6.0)
    frf_update_threshold  - relative Frobenius-norm FRF change (0-1) that
                             triggers re-deriving H+ for a bin (default 0.05)
    beta                  - cross-term learning rate, 0-1 (default 0.2).
                             beta=0 disables cross-term adaptation entirely
                             (reduces to the original diagonal-only law).
                             Higher beta trusts each new measurement more
                             and adapts faster, at the cost of more noise
                             sensitivity in the learned correlation structure.
Any missing/unparsable values fall back to the defaults above.
"""

import numpy as np


class response_space_multiplicative_control:
    def __init__(self,
                 specification: np.ndarray,
                 warning_levels: np.ndarray,
                 abort_levels: np.ndarray,
                 extra_parameters: str,
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
        # Defaults
        self.alpha = 0.05
        self.alpha_reg = 0.01
        self.clip_db = 6.0
        self.frf_update_threshold = 0.05
        self.beta = 0.2
        if extra_parameters:
            try:
                parts = [p.strip() for p in extra_parameters.split(',') if p.strip() != '']
                if len(parts) >= 1: self.alpha = float(parts[0])
                if len(parts) >= 2: self.alpha_reg = float(parts[1])
                if len(parts) >= 3: self.clip_db = float(parts[2])
                if len(parts) >= 4: self.frf_update_threshold = float(parts[3])
                if len(parts) >= 5: self.beta = float(parts[4])
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        # Only the diagonal of the spec drives the target magnitude; off-
        # diagonal in `specification` is ignored -- cross terms are LEARNED
        # from measured response correlation instead, not fixed up front.
        # Frequency lines outside the controlled band are legitimately NaN
        # (or 0) in Rattlesnake's specification_cpsd_matrix.
        spec_diag = np.real(np.einsum('fmm->fm', specification))
        self.spec_diag = np.nan_to_num(spec_diag, nan=0.0, posinf=0.0, neginf=0.0)
        self.F, self.M = self.spec_diag.shape

        self.H_pinv = None          # cached (F, N, M) Tikhonov pseudoinverse
        self.H_cache = None         # (F, M, N) FRF last used to build H_pinv, per bin
        self.current_diag = None    # (F, M) running per-DOF target magnitude
        self.rho = None             # (F, M, M) running complex correlation estimate
        self.N = None
        self.n_frf_updates = 0      # diagnostic: total H+ bin re-derivations

        if transfer_function is not None:
            self._init_from_frf(transfer_function)

    # ------------------------------------------------------------------
    def _pinv_one_bin(self, Hf):
        """Tikhonov-regularized pseudoinverse for a single frequency bin."""
        N = Hf.shape[1]
        HtH = Hf.conj().T @ Hf
        reg = self.alpha_reg * np.real(np.trace(HtH)) / N
        return np.linalg.solve(HtH + reg * np.eye(N), Hf.conj().T)

    def _psd_project(self, Yss):
        """Hermitian-symmetrize and clip negative eigenvalues to 0. A
        correlation-based reconstruction (magnitude/phase swapped onto a
        different diagonal) isn't automatically PSD, so this is a required
        safety step before Yss can be used, not an optional cleanup."""
        Yss = (Yss + Yss.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(Yss)
        eigvals = np.clip(eigvals, 0.0, None)
        return (eigvecs * eigvals) @ eigvecs.conj().T

    def _init_from_frf(self, transfer_function):
        """First-time setup: derive H+ for every bin and initialize the
        target state (diag = spec, rho = identity -- uncorrelated start,
        identical behavior to the old law until measurements inform it)."""
        self.N = transfer_function.shape[2]
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self.H_pinv = np.zeros((self.F, self.N, self.M), dtype=complex)
        for f in range(self.F):
            self.H_pinv[f] = self._pinv_one_bin(H_clean[f])
        self.H_cache = H_clean.copy()

        self.current_diag = np.maximum(self.spec_diag, 0.0).copy()
        self.rho = np.tile(np.eye(self.M, dtype=complex), (self.F, 1, 1))

    def _update_h_pinv_if_changed(self, transfer_function):
        """Re-derive H+ ONLY for frequency bins where the incoming transfer
        function has drifted more than frf_update_threshold. Does not touch
        current_diag or rho -- only changes how the current target maps to
        drives."""
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        num = np.linalg.norm((H_clean - self.H_cache).reshape(self.F, -1), axis=1)
        den = np.linalg.norm(self.H_cache.reshape(self.F, -1), axis=1) + 1e-30
        rel_change = num / den
        changed = np.where(rel_change > self.frf_update_threshold)[0]
        if changed.size > 0:
            for f in changed:
                self.H_pinv[f] = self._pinv_one_bin(H_clean[f])
            self.H_cache[changed] = H_clean[changed]
            self.n_frf_updates += int(changed.size)
        return changed

    def _current_output(self):
        F, N, M = self.F, self.N, self.M
        output = np.zeros((F, N, N), dtype=complex)
        sqrt_diag = np.sqrt(self.current_diag)  # (F, M)
        for f in range(F):
            outer_mag = np.outer(sqrt_diag[f], sqrt_diag[f])
            Yss = self._psd_project(outer_mag * self.rho[f] + 1e-12 * np.eye(M))
            Hp = self.H_pinv[f]
            output[f] = Hp @ Yss @ Hp.conj().T
        return output

    def _update_correlation(self, last_response_cpsd):
        """Blend the running correlation estimate toward whatever is
        actually being measured in the response, in complex-correlation
        space (avoids phase-wraparound issues from blending angles
        separately)."""
        if self.beta <= 0.0:
            return
        meas = np.nan_to_num(last_response_cpsd, nan=0.0, posinf=0.0, neginf=0.0)
        diag = np.real(np.einsum('fmm->fm', meas))
        diag_safe = np.maximum(diag, 1e-30)
        denom = np.sqrt(diag_safe[:, :, None] * diag_safe[:, None, :])
        rho_meas = meas / denom
        # Safety clip: measurement noise can push |rho| slightly past 1
        mag = np.abs(rho_meas)
        over = mag > 1.0
        rho_meas = np.where(over, rho_meas / np.maximum(mag, 1e-30), rho_meas)
        self.rho = (1 - self.beta) * self.rho + self.beta * rho_meas
        # Force exact self-correlation on the diagonal regardless of blend noise
        idx = np.arange(self.M)
        self.rho[:, idx, idx] = 1.0 + 0.0j

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
        if transfer_function is not None:
            if self.H_pinv is None:
                self._init_from_frf(transfer_function)
            else:
                self._update_h_pinv_if_changed(transfer_function)

    def control(self,
                transfer_function: np.ndarray = None,
                multiple_coherence: np.ndarray = None,
                frames=None,
                total_frames=None,
                last_response_cpsd: np.ndarray = None,
                last_output_cpsd: np.ndarray = None) -> np.ndarray:
        if self.H_pinv is None:
            self._init_from_frf(transfer_function)
        elif transfer_function is not None:
            # Only relevant if Rattlesnake's "Update Transfer Function During
            # Control" is enabled -- otherwise this is a no-op cost-wise.
            self._update_h_pinv_if_changed(transfer_function)

        if last_response_cpsd is not None:
            achieved_diag = np.real(np.einsum('fmm->fm', last_response_cpsd))
            achieved_diag = np.maximum(achieved_diag, 1e-30)
            E = self.spec_diag / achieved_diag
            clip = 10 ** (self.clip_db / 10.0)
            E = np.clip(E, 1.0 / clip, clip)
            self.current_diag = (self.alpha * E + (1 - self.alpha)) * self.current_diag

            self._update_correlation(last_response_cpsd)

        return self._current_output()

    # ------------------------------------------------------------------
    def update_target(self, new_specification: np.ndarray):
        """
        Call this when the test level changes. Updates the target diagonal
        only -- current_diag and the learned correlation structure (rho)
        are left alone, so the existing closed loop converges toward the
        new level using whatever cross-term structure it's already learned,
        rather than throwing that learning away on a level step.
        """
        spec_diag = np.real(np.einsum('fmm->fm', new_specification))
        self.spec_diag = np.nan_to_num(spec_diag, nan=0.0, posinf=0.0, neginf=0.0)