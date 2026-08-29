"""
octave_band_switching_control.py

Rattlesnake control law that behaves EXACTLY like optimal_diagonal_control
(same buzz baseline, same budgeted per-bin SDP scheduler, same coherence
cap -- it subclasses it and changes nothing about that machinery) until the
current test level reaches a user-configured threshold, at which point it
switches its control DECISIONS from per-FFT-line to 1/6-octave-band
granularity: every narrowband line inside a band is driven by ONE shared
solve for that band, instead of its own independent solve.

WHY: see control_laws/cva_frf_integration_design_2026-08-26.md, section 6,
for the full design discussion and the empirical prior art this is based on
(spectral_analysis/fractional_octave.py, siso_octave_control/code/
build_siso_octave_control_demo.py -- a SISO closed-loop demo that showed
octave-only control decisions converge to the same octave-band target as
full narrowband control, at the cost of within-band narrowband ripple that
narrowband control can cancel and octave-only control cannot). This file is
the MIMO generalization of that idea, wired into optimal_diagonal_control's
existing scheduler so switching to octave mode also means far fewer bins
to solve per call (~1/fraction-of-an-octave-worth of lines collapse into
one solve), which is the point: faster full-coverage convergence right
when a short full-level run can least afford the per-line SDP's slow
progressive coverage.

DRIVE SYNTHESIS AND RESPONSE MEASUREMENT ARE NOT CHANGED. This control law
still receives and returns full-narrowband arrays (F FFT lines) -- exactly
like optimal_diagonal_control. In octave mode it just derives a single
representative solve per band and broadcasts that band's drive CPSD flat
across every narrowband line inside it. No change to acquisition, output
generation, or spectral-processing frame sizing is needed anywhere in
Rattlesnake for this control law to work.

KNOWN LIMITATION -- band-averaged H is an approximation. The mathematically
correct way to aggregate an H1 estimate across lines is to band-average the
numerator and denominator cross/auto spectra SEPARATELY and then divide
(see fractional_octave.octave_band_frf's docstring) -- but this control
law is only ever given the already-divided H (transfer_function) at update
time, never the raw per-line Sxx/Sxy that produced it (Rattlesnake's
control-law interface doesn't pass those on every call). So the band H
used here is a plain complex mean of H across the band's narrowband lines.
This is a reasonable approximation away from a resonance/antiresonance,
and exactly the effect the SISO demo already characterized as "leftover
narrowband ripple" -- expected, not a bug.

KNOWN GAP -- there is no live "current test level" signal in Rattlesnake's
control-law interface today. system_id_update()/control() never receive
test level; RandomVibrationDataAnalysisProcess.run_control() doesn't track
it either (ADJUST_TEST_LEVEL only reaches data_collector and
signal_generation, never the control-law process). This control law
exposes set_test_level_db() for exactly this purpose, and defaults to
narrowband (the safe choice -- full per-line control authority) until
something calls it. See cva_frf_integration_design_2026-08-26.md section 7
for the small, optional, additive patch that would let Rattlesnake's core
call this hook automatically on every ADJUST_TEST_LEVEL/start_control --
not applied here, since that's a change to core files, not a new control
law. Until that's wired in (or wired in some other way -- see below),
switch_test_level_db can also be exercised manually/for testing by calling
set_test_level_db() directly, e.g. from a wrapping script, notebook, or a
different control law variant that IS given a level source.

extra_parameters (string): comma-separated, the same 5 values as
optimal_diagonal_control, PLUS 5 more, ALL required if any of the extra
5 are used (matches optimal_diagonal_control_fast's convention -- and
inherits the base class's parsing, which drops blank comma fields rather
than preserving their position, so no positions can be skipped):

    "reg,frf_update_threshold,max_bins_per_update,error_threshold_db,
     max_drive_coherence,frequency_spacing,switch_level_db,
     octave_fraction,fmin,fmax"

    reg, frf_update_threshold, max_bins_per_update, error_threshold_db,
    max_drive_coherence  - identical to optimal_diagonal_control (see that
                            file's docstring for defaults/meaning)
    frequency_spacing    - Hz per FFT line (sample_rate / samples_per_frame
                            for this environment). REQUIRED for octave mode
                            to ever engage -- the control law is not given
                            an explicit frequency vector by Rattlesnake, so
                            this is how it reconstructs one. Left at its
                            default (None / not given), octave mode never
                            engages and this behaves exactly like
                            optimal_diagonal_control regardless of test
                            level. No default -- must be supplied.
    switch_level_db       - test level (dB) at/above which octave-band
                             mode engages. Default 0.0.
    octave_fraction       - fractional-octave denominator, e.g. 6 for
                             1/6 octave, 3 for 1/3 octave. Default 6.
    fmin, fmax            - Hz range to band. Default: auto, spanning the
                             lowest through highest frequency line with a
                             nonzero diagonal specification target.
"""

import os
import importlib.util
import numpy as np

# Rattlesnake loads "Control Python Script" files standalone via
# importlib.util.spec_from_file_location (see components/utilities.py),
# not as part of the control_laws package -- a `from .optimal_diagonal_
# control import ...` relative import fails there with "attempted
# relative import with no known parent package". Load the sibling file
# the same way Rattlesnake itself loads control scripts instead (same
# pattern as optimal_diagonal_control_fast.py).
_this_dir = os.path.dirname(os.path.abspath(__file__))
_base_path = os.path.join(_this_dir, "optimal_diagonal_control.py")
_spec = importlib.util.spec_from_file_location("optimal_diagonal_control_base", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)
optimal_diagonal_control = _base_module.optimal_diagonal_control


# ---------------------------------------------------------------------
# Vendored fractional-octave band-edge math (pure numpy, no cross-repo
# import -- kept self-contained for the same reason the loader above is:
# this file has to work when Rattlesnake loads it standalone by path, not
# as part of any package, so it can't rely on spectral_analysis/ being
# importable). Identical formula to
# spectral_analysis/fractional_octave.py:octave_band_frequencies (ANSI
# S1.11 / IEC 61260 base-2 system).
# ---------------------------------------------------------------------
def _octave_band_frequencies(fmin, fmax, fraction, base=2.0, ref=1000.0):
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Require 0 < fmin < fmax")
    i_lo = int(np.floor(fraction * np.log(fmin / ref) / np.log(base)))
    i_hi = int(np.ceil(fraction * np.log(fmax / ref) / np.log(base)))
    indices = np.arange(i_lo, i_hi + 1)
    centers = ref * base ** (indices / fraction)
    edge_factor = base ** (1.0 / (2 * fraction))
    lower = centers / edge_factor
    upper = centers * edge_factor
    keep = (centers >= fmin) & (centers <= fmax)
    return centers[keep], lower[keep], upper[keep]


class octave_band_switching_control(optimal_diagonal_control):
    def __init__(self, *args, **kwargs):
        # These have to exist BEFORE super().__init__() runs: the base
        # class's __init__ calls self._initialize(...) -> self._refine_
        # batch(...) synchronously if a transfer_function is already
        # given, and _refine_batch is overridden below to reference these
        # attributes. Safe defaults here mean that first call (if it
        # happens) behaves as plain narrowband optimal_diagonal_control,
        # which is correct -- extra_parameters hasn't been parsed yet.
        self.frequency_spacing = None
        self.switch_level_db = 0.0
        self.octave_fraction = 6
        self.octave_fmin = None
        self.octave_fmax = None
        self.current_test_level_db = None
        self._band_of_line = None
        self._lines_of_band = None
        self._n_bands = 0
        self._octave_mode_active = False
        self._oct_output_cpsd = None
        self._oct_H_cache = None
        self._oct_H_initial = None
        self._oct_sdp_refined = None
        self._oct_err_db_cache = None
        self._oct_y_target = None
        self.n_octave_switches = 0
        self.n_octave_band_solves = 0

        super().__init__(*args, **kwargs)

        extra_parameters = kwargs.get('extra_parameters', args[3] if len(args) > 3 else '')
        if extra_parameters:
            try:
                parts = [p.strip() for p in str(extra_parameters).split(',') if p.strip() != '']
                if len(parts) >= 6: self.frequency_spacing = float(parts[5])
                if len(parts) >= 7: self.switch_level_db = float(parts[6])
                if len(parts) >= 8: self.octave_fraction = int(float(parts[7]))
                if len(parts) >= 9: self.octave_fmin = float(parts[8])
                if len(parts) >= 10: self.octave_fmax = float(parts[9])
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        if self.frequency_spacing:
            self._build_bands()

        print(f"[octave_band_switching_control] __init__: frequency_spacing={self.frequency_spacing} "
              f"switch_level_db={self.switch_level_db:g} octave_fraction=1/{self.octave_fraction} "
              f"fmin={self.octave_fmin} fmax={self.octave_fmax} "
              f"(octave mode {'CAN' if self.frequency_spacing else 'CANNOT'} engage -- "
              f"{'ready' if self.frequency_spacing else 'frequency_spacing not supplied'})",
              flush=True)

    # ------------------------------------------------------------------
    # Live test-level input. Rattlesnake's core doesn't call this today
    # (see module docstring) -- exposed so it CAN be wired in, and so this
    # can be exercised directly (tests, notebooks, a wrapping script).
    # ------------------------------------------------------------------
    def set_test_level_db(self, level_db):
        self.current_test_level_db = None if level_db is None else float(level_db)

    # ------------------------------------------------------------------
    def _build_bands(self):
        if self.F is None:
            return
        freqs = self.frequency_spacing * np.arange(self.F)
        fmin, fmax = self.octave_fmin, self.octave_fmax
        if fmin is None or fmax is None:
            in_band_lines = np.where(np.any(self.y_diag_target > 0, axis=1))[0]
            if in_band_lines.size == 0:
                print("[octave_band_switching_control] _build_bands: no in-band spec lines "
                      "yet -- deferring band construction until specification is populated",
                      flush=True)
                return
            if fmin is None:
                fmin = max(freqs[in_band_lines[0]], self.frequency_spacing)
            if fmax is None:
                fmax = freqs[in_band_lines[-1]]
        centers, lower, upper = _octave_band_frequencies(fmin, fmax, self.octave_fraction)
        band_of_line = np.full(self.F, -1, dtype=int)
        for b, (lo, hi) in enumerate(zip(lower, upper)):
            band_of_line[(freqs >= lo) & (freqs < hi)] = b
        n_unassigned_in_band = int(np.sum((band_of_line == -1)
                                           & np.any(self.y_diag_target > 0, axis=1)))
        self._band_centers, self._band_lower, self._band_upper = centers, lower, upper
        self._band_of_line = band_of_line
        self._lines_of_band = [np.where(band_of_line == b)[0] for b in range(len(centers))]
        self._n_bands = len(centers)
        print(f"[octave_band_switching_control] _build_bands: {self._n_bands} bands over "
              f"{fmin:.2f}-{fmax:.2f} Hz (1/{self.octave_fraction} octave), "
              f"{self.F} narrowband lines, {n_unassigned_in_band} in-band lines unassigned "
              f"(outside [fmin,fmax) -- these stay on narrowband control even in octave mode)",
              flush=True)

    def _init_octave_state(self, H_clean):
        """(Re)seed per-band scheduler state from scratch with a diagonal/
        buzz baseline, exactly like optimal_diagonal_control._initialize
        does for the narrowband case. Called every time octave mode is
        (re-)entered -- see _on_mode_switch -- so it never carries over
        stale per-band state from a previous excursion into octave mode."""
        n_bands = self._n_bands
        self._oct_y_target = np.stack([
            self.y_diag_target[lines].mean(axis=0) if lines.size > 0 else np.zeros(self.M)
            for lines in self._lines_of_band
        ])
        H_band0 = self._band_average_H(H_clean)
        output = np.zeros((n_bands, self.N, self.N), dtype=complex)
        for b in range(n_bands):
            Hpinv = np.linalg.pinv(H_band0[b], rcond=1e-12)
            output[b] = Hpinv @ np.diag(self._oct_y_target[b]).astype(complex) @ Hpinv.conj().T
        self._oct_output_cpsd = output
        self._oct_H_cache = H_band0.copy()
        self._oct_H_initial = H_band0.copy()
        self._oct_sdp_refined = np.zeros(n_bands, dtype=bool)
        self._oct_err_db_cache = np.full(n_bands, np.inf)
        print(f"[octave_band_switching_control] _init_octave_state: seeded {n_bands} bands "
              f"with diagonal/buzz baseline", flush=True)

    def _band_average_H(self, H_clean):
        """Plain complex mean of H across each band's narrowband lines.
        See the module docstring's KNOWN LIMITATION note -- this is an
        approximation of the mathematically-correct Sxy/Sxx band-average,
        which isn't reconstructable from what this control law is given
        at update time."""
        return np.stack([
            H_clean[lines].mean(axis=0) if lines.size > 0
            else np.zeros((self.M, self.N), dtype=complex)
            for lines in self._lines_of_band
        ])

    def _on_mode_switch(self, entering_octave, H_clean):
        self.n_octave_switches += 1
        if entering_octave:
            print(f"[octave_band_switching_control] SWITCHING to octave-band control "
                  f"(test level {self.current_test_level_db:g} dB >= "
                  f"switch_level_db {self.switch_level_db:g} dB), switch #{self.n_octave_switches}",
                  flush=True)
            self._init_octave_state(H_clean)
        else:
            print(f"[octave_band_switching_control] SWITCHING to narrowband control "
                  f"(test level {self.current_test_level_db} dB < "
                  f"switch_level_db {self.switch_level_db:g} dB), switch #{self.n_octave_switches}",
                  flush=True)
            # Every line that was under octave-broadcast control has an
            # output_cpsd value that is NOT a real per-line SDP solution
            # (it's the coarse band value) -- mark those lines as not-
            # SDP-refined so the narrowband scheduler's step 2 picks them
            # back up on worst-error-first priority, instead of leaving
            # them mistakenly flagged "already refined" (which would
            # freeze them on the coarse octave value indefinitely).
            if self._band_of_line is not None:
                touched = np.where(self._band_of_line >= 0)[0]
                self.sdp_refined[touched] = False

    # ------------------------------------------------------------------
    # Per-band scheduler -- same 3-step structure as optimal_diagonal_
    # control._refine_batch (drifted-refined bins, then worst-error
    # not-yet-refined bins, then stale-error re-triage), operating on
    # n_bands "bins" instead of F narrowband lines. See that method's
    # docstring for the rationale behind each step; not repeated here.
    # ------------------------------------------------------------------
    def _refine_batch_octave(self, H_clean):
        H_band = self._band_average_H(H_clean)
        budget = self.max_bins_per_update

        drift_budget = max(1, self.max_bins_per_update // 2)
        refined_idx = np.where(self._oct_sdp_refined)[0]
        if refined_idx.size > 0:
            num = np.linalg.norm((H_band[refined_idx] - self._oct_H_cache[refined_idx])
                                  .reshape(refined_idx.size, -1), axis=1)
            den = np.linalg.norm(self._oct_H_cache[refined_idx].reshape(refined_idx.size, -1),
                                  axis=1) + 1e-30
            drifted = refined_idx[(num / den) > self.frf_update_threshold]
        else:
            drifted = np.array([], dtype=int)
        n_fix = min(drifted.size, drift_budget, budget)
        fixed_this_call = drifted[:n_fix]
        for b in fixed_this_call:
            self._oct_output_cpsd[b] = self._solve_one_bin(H_band[b], self._oct_y_target[b])
            self._oct_H_cache[b] = H_band[b]
        if n_fix > 0:
            self._oct_err_db_cache[fixed_this_call] = self._err_db_oct(H_band, fixed_this_call)
        budget -= n_fix

        not_refined = np.where(~self._oct_sdp_refined)[0]
        n_deferred = 0
        n_refine = 0
        if not_refined.size > 0:
            err_db = self._err_db_oct(H_band, not_refined)
            above = not_refined[err_db > self.error_threshold_db]
            above_err = err_db[err_db > self.error_threshold_db]
            order = above[np.argsort(-above_err)]
            n_refine = min(order.size, budget)
            n_deferred = int(order.size - n_refine)
            for b in order[:n_refine]:
                self._oct_output_cpsd[b] = self._solve_one_bin(H_band[b], self._oct_y_target[b])
                self._oct_sdp_refined[b] = True
                self._oct_H_cache[b] = H_band[b]
            if n_refine > 0:
                self._oct_err_db_cache[order[:n_refine]] = self._err_db_oct(H_band, order[:n_refine])
            budget -= n_refine

        n_stale = 0
        refined_idx = np.where(self._oct_sdp_refined)[0]
        if n_fix > 0:
            refined_idx = refined_idx[~np.isin(refined_idx, fixed_this_call)]
        if budget > 0 and refined_idx.size > 0:
            err_db_r = self._err_db_oct(H_band, refined_idx)
            degraded = err_db_r > self._oct_err_db_cache[refined_idx] + self.error_threshold_db
            stale = refined_idx[degraded]
            stale_order = stale[np.argsort(-err_db_r[degraded])]
            n_stale = min(stale_order.size, budget)
            for b in stale_order[:n_stale]:
                self._oct_output_cpsd[b] = self._solve_one_bin(H_band[b], self._oct_y_target[b])
                self._oct_H_cache[b] = H_band[b]
            if n_stale > 0:
                self._oct_err_db_cache[stale_order[:n_stale]] = self._err_db_oct(H_band, stale_order[:n_stale])

        self.n_octave_band_solves += int(n_fix + n_refine + n_stale)

        # Broadcast: every narrowband line in a band gets that band's
        # drive CPSD, unchanged. Lines outside [fmin,fmax) (band_of_line
        # == -1) are left exactly as they were -- see _build_bands.
        for b, lines in enumerate(self._lines_of_band):
            if lines.size > 0:
                self.output_cpsd[lines] = self._oct_output_cpsd[b]
        self.H_cache = H_clean.copy()

        print(f"[octave_band_switching_control] _refine_batch_octave: {self._n_bands} bands, "
              f"drift_resolved={n_fix}, newly_refined={n_refine}, n_deferred={n_deferred}, "
              f"stale_resolved={n_stale}, n_band_refined_total={int(np.sum(self._oct_sdp_refined))}/"
              f"{self._n_bands}, cum_band_solves={self.n_octave_band_solves}",
              flush=True)

    def _err_db_oct(self, H_band_clean, indices):
        Y = np.einsum('fmn,fnk,flk->fml', H_band_clean[indices], self._oct_output_cpsd[indices],
                       H_band_clean[indices].conj())
        achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
        target = np.maximum(self._oct_y_target[indices], 1e-30)
        return np.max(np.abs(10 * np.log10(achieved / target)), axis=1)

    # ------------------------------------------------------------------
    def _refine_batch(self, transfer_function):
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)

        should_use_octave = (
            self.frequency_spacing is not None
            and self._band_of_line is not None
            and self.current_test_level_db is not None
            and self.current_test_level_db >= self.switch_level_db
        )
        if should_use_octave != self._octave_mode_active:
            self._octave_mode_active = should_use_octave
            self._on_mode_switch(should_use_octave, H_clean)

        if should_use_octave:
            self._refine_batch_octave(H_clean)
        else:
            super()._refine_batch(transfer_function)

    # ------------------------------------------------------------------
    # Explicit pass-through overrides -- required, not decorative. See
    # optimal_diagonal_control_fast.py's identical note: Rattlesnake's
    # control-law loader only recognizes a class as valid if
    # 'system_id_update' and 'control' are in the class's OWN __dict__,
    # not just inherited from a base class.
    # ------------------------------------------------------------------
    def system_id_update(self, *args, **kwargs):
        return super().system_id_update(*args, **kwargs)

    def control(self, *args, **kwargs):
        return super().control(*args, **kwargs)
