"""
response_space_multiplicative_control.py

Rattlesnake control law implementing the response-space multiplicative
update validated earlier (stage6_closed_loop_control.py), ported to the
actual Rattlesnake class-based control law interface.

Key property: a Tikhonov-regularized pseudoinverse H+ is derived from the
system-ID transfer function. Every control iteration thereafter adapts
primarily from the measured response error -- H is NOT re-inverted on
every call. But if Rattlesnake is set to feed a live-updating FRF during
control ("Update Transfer Function During Control" checkbox, fed by
components/spectral_processing.py), this law now detects per-frequency-line
drift in that incoming FRF and re-derives H+ only for the bins that moved
-- e.g. a resonance shifting under increasing test level. This does NOT
reset the response-space Cholesky factor W_resp, so the closed-loop-learned
cross-term structure and convergence progress survive an FRF update; only
the drive-space mapping for the affected bins changes.

Algorithm (per frequency line f), each control() call:
    1. Current target-in-progress:  Yss(f) = W(f) W(f)^H
    2. Output CPSD:                 X(f)   = H+(f) Yss(f) H+(f)^H
    3. (Rattlesnake applies X, measures the real response, calls control()
       again with last_response_cpsd = what was actually measured, and a
       possibly-updated transfer_function if live FRF updating is enabled)
    4. Error ratio (per response DOF i):
           E_i(f) = spec_diag_i(f) / achieved_diag_i(f)
       clipped to +/- clip_db to avoid violent single-step corrections
    5. Update:  W_i(f) <- sqrt(alpha*E_i(f) + (1-alpha)) * W_i(f)
    6. FRF drift check: for any bin f where the incoming transfer_function
       has moved more than frf_update_threshold (relative Frobenius norm)
       from the H last used to build H+(f), re-derive H+(f) for that bin
       only. Cheap (a single NxN linear solve per changed bin).

Because the correction rule only ever touches measured response diagonals,
a pure test-level change still needs no special-casing: update
self.spec_diag (or just re-run with a new `specification`), and the next
few control() calls converge to the new level via the same multiplicative
loop. A resonance shift is different -- that changes H itself, which is
what the drift check above is for.

extra_parameters (string): comma-separated
    "alpha,alpha_reg,clip_db,frf_update_threshold"
    alpha                 - update damping, 0-1 (default 0.5)
    alpha_reg             - Tikhonov regularization weight for H+ (default 0.01)
    clip_db               - max per-iteration correction in dB (default 6.0)
    frf_update_threshold  - relative Frobenius-norm change (0-1) in a bin's H
                             that triggers re-deriving H+ for that bin (default 0.05)
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
        self.alpha = 0.15
        self.alpha_reg = 0.01
        self.clip_db = 6.0
        self.frf_update_threshold = 0.05
        if extra_parameters:
            try:
                parts = [p.strip() for p in extra_parameters.split(',') if p.strip() != '']
                if len(parts) >= 1: self.alpha = float(parts[0])
                if len(parts) >= 2: self.alpha_reg = float(parts[1])
                if len(parts) >= 3: self.clip_db = float(parts[2])
                if len(parts) >= 4: self.frf_update_threshold = float(parts[3])
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        # Only the diagonal of the spec drives this law; off-diagonal in
        # `specification` is ignored (same convention as optimal_diagonal_control).
        # Frequency lines outside the controlled band are legitimately NaN (or 0)
        # in Rattlesnake's specification_cpsd_matrix -- treat both as "no target,
        # drive this bin toward zero" rather than letting NaN propagate.
        spec_diag = np.real(np.einsum('fmm->fm', specification))
        self.spec_diag = np.nan_to_num(spec_diag, nan=0.0, posinf=0.0, neginf=0.0)
        self.F, self.M = self.spec_diag.shape

        self.H_pinv = None    # cached (F, N, M) Tikhonov pseudoinverse
        self.H_cache = None   # (F, M, N) FRF last used to build H_pinv, per bin
        self.W_resp = None    # response-space Cholesky factor, (F, M, M)
        self.N = None
        self.n_frf_updates = 0   # diagnostic: total bin re-derivations performed

        if transfer_function is not None:
            self._init_from_frf(transfer_function)

    # ------------------------------------------------------------------
    def _pinv_one_bin(self, Hf):
        """Tikhonov-regularized pseudoinverse for a single frequency bin."""
        N = Hf.shape[1]
        HtH = Hf.conj().T @ Hf
        reg = self.alpha_reg * np.real(np.trace(HtH)) / N
        return np.linalg.solve(HtH + reg * np.eye(N), Hf.conj().T)

    def _init_from_frf(self, transfer_function):
        """First-time setup: derive H+ for every bin and initialize W_resp."""
        self.N = transfer_function.shape[2]
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self.H_pinv = np.zeros((self.F, self.N, self.M), dtype=complex)
        for f in range(self.F):
            self.H_pinv[f] = self._pinv_one_bin(H_clean[f])
        self.H_cache = H_clean.copy()

        # Start W_resp as the Cholesky factor of a diagonal-only target
        # (off-diagonal terms will emerge naturally as the update runs)
        self.W_resp = np.zeros((self.F, self.M, self.M), dtype=complex)
        for f in range(self.F):
            Yss0 = np.diag(np.maximum(self.spec_diag[f], 0.0)).astype(complex)
            self.W_resp[f] = np.linalg.cholesky(Yss0 + 1e-12 * np.eye(self.M))

    def _update_h_pinv_if_changed(self, transfer_function):
        """
        Re-derive H+ ONLY for frequency bins where the incoming transfer
        function has drifted more than frf_update_threshold from the H it
        was last built from. W_resp is untouched -- this only changes how
        the current response-space target gets mapped to drives.
        """
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
        F, N = self.F, self.N
        output = np.zeros((F, N, N), dtype=complex)
        for f in range(F):
            Wf = self.W_resp[f]
            Yss = Wf @ Wf.conj().T
            Hp = self.H_pinv[f]
            output[f] = Hp @ Yss @ Hp.conj().T
        return output

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
            # Control" is enabled -- otherwise this transfer_function is the
            # same static sysid FRF every call and nothing will trip the
            # threshold, so this is a no-op cost-wise when the feature is off.
            self._update_h_pinv_if_changed(transfer_function)

        if last_response_cpsd is not None:
            achieved_diag = np.real(np.einsum('fmm->fm', last_response_cpsd))
            achieved_diag = np.maximum(achieved_diag, 1e-30)
            E = self.spec_diag / achieved_diag
            clip = 10 ** (self.clip_db / 10.0)
            E = np.clip(E, 1.0 / clip, clip)
            gain = np.sqrt(self.alpha * E + (1 - self.alpha))
            self.W_resp = self.W_resp * gain[:, :, np.newaxis]

        return self._current_output()

    # ------------------------------------------------------------------
    def update_target(self, new_specification: np.ndarray):
        """
        Call this when the test level changes. Updates the target diagonal
        without touching H+ or the current W_resp -- the existing
        multiplicative loop just converges toward the new level over the
        next several control() iterations, same as any other level step.
        """
        spec_diag = np.real(np.einsum('fmm->fm', new_specification))
        self.spec_diag = np.nan_to_num(spec_diag, nan=0.0, posinf=0.0, neginf=0.0)
