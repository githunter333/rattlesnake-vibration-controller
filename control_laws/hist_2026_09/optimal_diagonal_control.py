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
                             CAVEAT: the ceiling binds on the FIRST command
                             only -- it keys on last_output_cpsd being None,
                             which is true exactly once -- so cycle 2 returns
                             the uncapped solution. It guards the one command
                             issued before anyone has seen the rig respond,
                             not the level of the run.
response_trim_gain - fraction of the measured response error folded into
              the solver's target each control cycle (field 11, default 0.0
              = OFF).  See "What kind of loop this is" below.
response_trim_limit_db - total authority of the trim, +/- dB (field 12,
              default 3.0).  Hard clamp on the accumulated correction.
response_trim_step_db - most the trim may move in one cycle, dB (field 13,
              default 0.5).
response_trim_deadband_db - measured errors smaller than this are left alone
              (field 14, default 0.5), so the trim does not chase the
              frame-to-frame scatter of a short-average CPSD.
response_trim_gate_db - how close the solver must have come to its own target
              before the error left over is treated as model error rather
              than as the solver's own business (field 15, default 0.5).
              Deliberately NOT error_threshold_db, which is the scheduler's
              "worth refining" bar and answers a different question -- see
              the comment in __init__ and what run 35 measured.
A bare single value (no comma) is accepted too, read as `reg`.

WHAT KIND OF LOOP THIS IS
-------------------------
Two different things can be called "closed loop" here and this law has
historically been mislabelled on both counts.  Reports before 2026-09-17
called it a feedback law; the correction made then over-shot and called it
open loop, which understates it just as badly.  Precisely:

  CLOSED on the identified plant model.  control() and system_id_update()
  both hand the LIVE transfer function to _refine_batch every cycle.  Bins
  whose H has moved more than frf_update_threshold since their own last
  solve are re-solved (step 1), and bins whose predicted error has degraded
  against the CURRENT H are re-solved worst-first (step 3) even when their H
  moved less than the gate.  Step 3 is the one that matters for a plant that
  softens with level: the drift can sit under the gate and still be caught,
  because the error is always recomputed with the new H.  This is what runs
  29/31/32 demonstrated -- 17.03 / 18.44 / 16.22 V^2 with the FRF update on,
  against 153 / 485 / 245 V^2 with it off.

  OPEN on the response error, unless response_trim_gain is set.  The solve
  minimizes the PREDICTED response diag(H X H^H) against the target.  With
  the gain at 0 (the default) last_response_cpsd is accepted and never read,
  so any error the FRF does not explain -- bias in the H1 estimate,
  extraneous input, drive clipping, nonlinear cross terms that never appear
  in a linear FRF -- is invisible to the law and sits there uncorrected.
  _update_response_trim closes that half: see its docstring for why it
  integrates measured-vs-SPECIFICATION error but gates on
  predicted-vs-TARGET agreement, and why that distinction is what keeps it
  from winding up at bins the plant simply cannot reach.

The honest one-line description is an indirect adaptive (self-tuning)
regulator: fast model update, optional slow error trim.
"""

import numpy as np

# ---------------------------------------------------------------------------
# LIVE-FRF SAFETY LIMITS.  Added 2026-09-19 after run 30, where the live H1
# estimate walked 22,000x away from the system identification (reported by
# this class's own since_initial diagnostic as median 39834, max 500217) and
# the law solved against it, commanding 1.8e5 V.  A healthy run sits well
# under 1: run 28 logged since_initial median 0.1752, max 0.8920 over its
# whole length.  10.0 therefore has an order of magnitude of headroom over
# anything ever measured on a working run and still catches run 30 at its
# 55th call.
_FRF_DIVERGENCE_LIMIT = 10.0

# DEADLOCK ESCAPE.  Run 30's end state: corrupted H -> near-zero commanded
# drive (0.0015 V) -> no excitation -> the live FRF stops moving -> nothing
# exceeds frf_update_threshold -> no bins scheduled -> the law holds a dead
# solution forever.  70+ consecutive calls with every scheduler field zero
# and a bit-identical self-prediction.  It cannot recover on its own, because
# recovering requires excitation the law is no longer commanding.  These
# thresholds only fire in that absorbing state: 20 dB below the target trace
# is far outside anything a working run reaches (runs 19-29 all land within
# +/-3 dB), and 10 consecutive calls rules out a transient.
_DEADLOCK_ESCAPE_DB = 20.0
_DEADLOCK_ESCAPE_CALLS = 10

# RESPONSE-ERROR TRIM FREEZE.  components/data_collector.py divides every
# acquired frame by the CURRENT test level, so everything this law sees --
# FRF, last_response_cpsd, last_drive_cpsd -- is referred to FULL level and
# the trim ratio is level-consistent in steady state.  It is NOT consistent
# across a level change: a frame that spans the ramp is normalized by the
# wrong number.  Rattlesnake calls set_test_level_db on every change, so the
# trim sits out this many control cycles afterwards.
#
# RAISED FROM 3 TO 25, 2026-09-25, after run 52.  Three was sized for a single
# level change.  A level RAMP is many changes -- run 52 logged 17 of them
# getting from 0 dB to -18 dB -- and the freeze expires between steps, so the
# trim spent the whole ramp acting on transient error.  25 matches the ~25
# control-cycle time constant of the CPSD exponential average (coefficient
# 0.04), which is how long a measured frame takes to stop carrying the old
# level.  Each change re-arms the counter, so a ramp is ONE long freeze and
# the trim resumes only once the level has actually been still that long.
_TRIM_FREEZE_CALLS = 25

# RESPONSE-ERROR TRIM LEAK.  Per cycle, in-band bins that do NOT act -- gated
# out by the solver-agreement test, or inside the deadband -- decay toward
# zero by this much instead of holding.
#
# WHY, and this is the defect run 52 exposed.  The gate is evaluated BEFORE
# the error term, and gated bins previously got step_db = 0, i.e. they kept
# whatever correction they already carried, permanently.  That blocks harmful
# accumulation and corrective UNWINDING equally: a bin that picks up a
# spurious trim while the loop is still converging, and whose solver residual
# then exceeds the gate, is frozen at that value for the rest of the run.
# Run 52 ended with the trim spread across the full +/-3 dB clamp, median
# -1.51 dB, after 261 updates, on a plant whose true model gap at that level
# is 0.012 dB -- and it cost 30% more drive (11.47 V rms vs 8.84 V) to
# deliver the same response as trim-off.  Integrator windup behind a
# conditional gate.
#
# A REAL model error re-drives itself every cycle at up to
# response_trim_step_db, so a leak an order of magnitude smaller is invisible
# to it in steady state.  A spurious one bleeds off in ~limit/leak cycles
# (3.0/0.05 = 60).  This makes the trim self-healing rather than dependent on
# never having made a mistake.
_TRIM_LEAK_DB = 0.05

# ...but ONLY after this many CONSECUTIVE gate failures on the same bin.
#
# Second correction, same session.  Leaking on a single gate failure was also
# wrong: max_bins_per_update is 20 against 901 in-band bins, so after the trim
# moves a target it takes ~45 cycles before that bin is re-solved, and until
# then its cached prediction is stale and the gate fails for a reason that has
# nothing to do with reachability.  Leaking immediately therefore bled away
# corrections that were merely WAITING THEIR TURN -- verify_response_trim.py
# TEST 2 settled at -1.05 dB against a real -2.00 dB bias.
#
# 50 cycles is comfortably longer than one full 901/20 = 45-cycle refinement
# sweep, so a bin that has failed the gate this many times in a row has had at
# least one chance to be re-solved and still cannot reach its target.  That is
# the population that wound up in run 52.
_TRIM_LEAK_AFTER = 50

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
                 # NEVER READ HERE, and read in control() only when
                 # response_trim_gain > 0.  __init__ runs before the rig has
                 # responded to anything, so there is nothing to read.  With
                 # the trim off the solve is driven entirely by H: each bin
                 # is refined against its own PREDICTED response,
                 # diag(H X H^H), and the only thing that changes between
                 # calls is H.  With the trim on, control() folds the
                 # measured response error into the target -- see
                 # _update_response_trim and "WHAT KIND OF LOOP THIS IS" in
                 # the module docstring.
                 last_response_cpsd: np.ndarray = None,
                 last_output_cpsd: np.ndarray = None,
                 ):
        y_diag_target = np.real(np.einsum('fmm->fm', specification))
        self.y_diag_target = np.nan_to_num(y_diag_target, nan=0.0, posinf=0.0, neginf=0.0)
        self.F, self.M = self.y_diag_target.shape

        self.reg = 1e-6
        # 0.5, raised from 0.05 on 2026-09-19 once runs 29 and 31 measured
        # what this gate actually sees.  0.05 was set against the broken
        # _h_moved metric, which included out-of-band bins and read median
        # 4.37; corrected and in-band the per-cycle movement is median 0.196,
        # 90th 0.232, with a thin tail to 3.5.  At 0.05 essentially every
        # call demoted the fast law to the SDP (99% on run 28) AND every
        # previously-refined bin counted as drifted, so half the refinement
        # budget went to re-solving old bins forever (cum_frf_updates 10 per
        # call, 5570 and climbing).  0.5 sits at about the 95th percentile of
        # the measured distribution: 5% demotion, budget freed.
        #
        # Changing this default does not alter any archived run: every
        # profile_01..26 sets field 2 explicitly (verified cell by cell).
        self.frf_update_threshold = 0.5
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
        # RESPONSE-ERROR TRIM -- fields 11-14, OFF by default.  Every
        # behaviour change in this file ships defaulted off (commit
        # 45d417a3); this one is no exception, so an archived profile that
        # names ten fields or fewer reproduces exactly.
        self.response_trim_gain = 0.0
        self.response_trim_limit_db = 3.0
        self.response_trim_step_db = 0.5
        self.response_trim_deadband_db = 0.5
        # The solver-agreement gate has its OWN threshold, and the separation
        # is not cosmetic.  It first reused error_threshold_db, on the
        # reasoning that the scheduler's "worth refining" bar and "the solver
        # reached what it aimed at" were the same question.  Run 35 says they
        # are not.  Reconstructing this law's own per-bin prediction from
        # that run's saved FRF and drive and comparing it with the measured
        # response: model mismatch is 0.060 dB rms across all 7208 in-band
        # channel-bins, median +0.008 -- the FRF-update loop leaves
        # essentially nothing for an error trim to correct on this plant.
        # Yet at a 1.0 dB gate, 43.6% of gated channel-bins sat outside the
        # 0.5 dB deadband, with error-vs-spec 0.541 dB rms against a model
        # mismatch of 0.041 dB rms.  Thirteen to one: the trim would have
        # spent the entire run integrating the SOLVER's own sub-threshold
        # shortfall -- second-guessing the optimizer that is already trading
        # these channels against each other -- and calling it model error.
        # At 0.5 dB only 1.6% of gated channel-bins clear the deadband, so on
        # a healthy rig the trim stays quiet.  A real bias still wakes it:
        # the shortfall is predicted-vs-target, computed entirely from H and
        # X, so a plant that does not match H cannot change it.  The gate
        # stays open exactly as wide while the error grows.
        self.response_trim_gate_db = 0.5
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
                if len(parts) >= 11: self.response_trim_gain = max(0.0, float(parts[10]))
                if len(parts) >= 12: self.response_trim_limit_db = abs(float(parts[11]))
                if len(parts) >= 13: self.response_trim_step_db = abs(float(parts[12]))
                if len(parts) >= 14: self.response_trim_deadband_db = abs(float(parts[13]))
                if len(parts) >= 15: self.response_trim_gate_db = abs(float(parts[14]))
            except ValueError:
                pass  # keep defaults if the string doesn't parse

        self.output_cpsd = None    # (F, N, N) current best drive CPSD per bin
        # Response-error trim state.  y_trim stays None while the trim is
        # off, and _eff_target then returns the specification slice itself,
        # so the solve is bit-identical to the pre-trim code.
        self.y_trim = None            # (F, M) multiplicative correction on the target
        self._p_commanded = None      # (F, M) predicted diagonal of the command actually issued
        self._target_commanded = None # (F, M) target that command was solved against
        self._trim_frozen_calls = 0
        self._trim_gate_fail = None   # (F, M) consecutive gate failures per bin
        self._test_level_db = None
        self._H_last_used = None
        self.n_trim_updates = 0
        # Console throttling -- see _should_log.  Once every bin is solved and
        # the FRF is frozen, _refine_batch does no work but still ran two print
        # statements per control cycle, which on run 14 buried the terminal in
        # hundreds of identical lines and read as a hung process.  (It was not
        # hung: newly_refined=0, n_deferred=0, cum_sdp_refinements frozen.)
        self._idle_repeats = 0
        # Live-FRF guard state.  Set here as well as in _initialize so a
        # harness that reaches _refine_batch without going through
        # _initialize cannot raise AttributeError -- the same latent bug
        # the fast subclass hit on 2026-09-17 with n_fast_solves.
        self._frf_rejected = 0
        self._last_frf_rejected = 0
        self._starved_calls = 0
        self._n_escapes = 0
        self._sysid_response_cpsd = None
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
        target = np.maximum(self._eff_target(indices), 1e-30)
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
        self.H_initial = H_clean.copy()  # diagnostic AND, since 2026-09-19, the
                                         # fallback the divergence guard reverts to
        self._sysid_response_cpsd = (None if sysid_response_cpsd is None
                                     else np.asarray(sysid_response_cpsd).copy())
        self._frf_rejected = 0            # bins currently held at H_initial
        self._last_frf_rejected = 0
        self._starved_calls = 0           # consecutive calls in the dead state
        self._n_escapes = 0
        self.sdp_refined = np.zeros(self.F, dtype=bool)
        self.err_db_cache = np.full(self.F, np.inf)  # unset until a bin is actually SDP-solved
        self._initialized = True
        self._refine_batch(H_clean)  # spend the first batch of SDP budget immediately

    # ------------------------------------------------------------------
    def _reject_diverged_frf(self, H_clean):
        """GUARD 1 -- refuse a live FRF that is no longer a plant model.

        Rattlesnake re-publishes an incrementally-averaged FRF every cycle
        when "Update Transfer Function During Control" is on.  That estimate
        is formed from the drive the control law itself commands, so a bad
        solve feeds a bad estimate which feeds a worse solve.  Run 30 closed
        that loop: since_initial reached median 39834 and max 500217 -- the
        live H was four to five orders of magnitude away from the system
        identification -- and the law dutifully solved against it and asked
        for 1.8e5 V.

        Per BIN, not globally, so one wild line cannot condemn the whole
        array and the law keeps controlling on the bins that are still sane.
        A rejected bin reverts to its system-ID value for this call; it is
        not frozen, and recovers by itself the moment the live estimate comes
        back inside the limit.

        This does NOT fix whatever made the estimate diverge.  It stops the
        law amplifying it, and says so in the log.
        """
        H_init = getattr(self, 'H_initial', None)
        if H_init is None or H_init.shape != H_clean.shape:
            return H_clean
        limit = getattr(self, 'frf_divergence_limit', _FRF_DIVERGENCE_LIMIT)
        if not np.isfinite(limit) or limit <= 0:
            return H_clean

        n = H_clean.shape[0]
        num = np.linalg.norm((H_clean - H_init).reshape(n, -1), axis=1)
        den = np.linalg.norm(H_init.reshape(n, -1), axis=1)
        live = den > 0

        # IN-BAND ONLY, and this was got wrong the first time.  Run 31 fired
        # this guard on 986-1161 bins of 2049 on essentially every call while
        # controlling perfectly well (converged 2.13 V, 17.70 V^2, and the
        # refined-bin drift never left median 0.07).  2049 - 901 in-band =
        # 1148 out-of-band bins: no drive energy there, so the live estimate
        # is noise and its relative change is unbounded.  The guard was
        # reporting noise.  Harmless -- those bins are not controlled -- but
        # it buried the log and made the threshold untestable.
        #
        # THIRD TIME this same mistake has been made in this file and its
        # subclass: the singular-value spread in _effective_drive_rcond, the
        # movement metric in _h_moved, and now this.  Any statistic taken
        # over the FRF array on this rig must be masked to in-band bins.
        in_band = self.y_diag_target.max(axis=1) > 0
        if in_band.shape[0] == n and in_band.any():
            live &= in_band

        ratio = np.zeros(n)
        ratio[live] = num[live]/den[live]
        bad = live & (ratio > limit)
        n_bad = int(bad.sum())
        n_scored = int(live.sum())

        if n_bad:
            H_clean = H_clean.copy()
            H_clean[bad] = H_init[bad]
            # Log on entry and whenever the count changes by more than a few
            # bins, not every call -- this fires on the cycle that matters.
            if abs(n_bad - self._last_frf_rejected) > 2 or self._last_frf_rejected == 0:
                print(f"[optimal_diagonal_control] LIVE FRF REJECTED on {n_bad} of "
                      f"{n_scored} in-band bins (relative change from the system ID above "
                      f"{limit:g}; worst {ratio.max():.1f}). Those bins are using "
                      f"the system-ID FRF this call. The live estimate is "
                      f"diverging -- check excitation and drive level.", flush=True)
                self._last_frf_rejected = n_bad
        elif self._last_frf_rejected:
            print(f"[optimal_diagonal_control] live FRF back inside the "
                  f"divergence limit on every bin; using it again.", flush=True)
            self._last_frf_rejected = 0
        self._frf_rejected = n_bad
        return H_clean

    # ------------------------------------------------------------------
    def _escape_deadlock_if_starved(self, H_clean, worked):
        """GUARD 2 -- break out of the zero-drive absorbing state.

        The failure this exists for (run 30, calls 109-182): the commanded
        drive collapses to near zero, so the plant is barely excited, so the
        live FRF stops moving, so no bin trips frf_update_threshold, so the
        scheduler has nothing to do, so the law keeps commanding the same
        dead solution.  Every scheduler field zero and a bit-identical
        self-prediction for seventy consecutive calls.  Nothing in the loop
        can restart it, because restarting requires excitation the law is no
        longer asking for.

        Detection is deliberately narrow: the predicted response trace has to
        be _DEADLOCK_ESCAPE_DB below the specification trace AND no bin can
        have been scheduled, for _DEADLOCK_ESCAPE_CALLS consecutive calls.  A
        working run sits within a few dB of target, so this cannot fire on
        one.

        Recovery throws away the corrupted state and starts over from the
        system identification -- the one plant model known to be good.
        """
        in_band = self.y_diag_target.max(axis=1) > 0
        if not in_band.any():
            return
        Y = np.einsum('fmn,fnk,flk->fml', H_clean[in_band],
                      self.output_cpsd[in_band], H_clean[in_band].conj())
        achieved = float(np.sum(np.maximum(np.real(np.einsum('fmm->fm', Y)), 0.0)))
        target = float(np.sum(self.y_diag_target[in_band]))
        if target <= 0:
            return
        shortfall_db = 10.0*np.log10(max(achieved, 1e-300)/target)

        if worked or shortfall_db > -_DEADLOCK_ESCAPE_DB:
            self._starved_calls = 0
            return

        self._starved_calls += 1
        if self._starved_calls < _DEADLOCK_ESCAPE_CALLS:
            return

        self._n_escapes += 1
        print(f"[optimal_diagonal_control] DEADLOCK ESCAPE #{self._n_escapes}: "
              f"predicted response has sat {shortfall_db:.1f} dB below "
              f"specification with nothing scheduled for "
              f"{self._starved_calls} calls. Re-solving from the system-ID "
              f"FRF and discarding the live one.", flush=True)
        self.output_cpsd = self._buzz_solve_all(self.H_initial,
                                                self._sysid_response_cpsd)
        self.H_cache = self.H_initial.copy()
        self.sdp_refined[:] = False
        self.err_db_cache[:] = np.inf
        self._starved_calls = 0
        self._idle_repeats = 0

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
        H_clean = self._reject_diverged_frf(H_clean)
        self._H_last_used = H_clean
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
            self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self._eff_target(f),
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
                self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self._eff_target(f),
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
                self.output_cpsd[f] = self._solve_one_bin(H_clean[f], self._eff_target(f),
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
            target_all = np.maximum(self._eff_target(in_band), 1e-30)
            err_db_all = 10 * np.log10(achieved_all / target_all)
            self_rms_per_channel = np.sqrt(np.mean(err_db_all ** 2, axis=0))
        else:
            self_rms_per_channel = np.zeros(self.M)

        # A call that solved nothing, deferred nothing and saw no FRF movement
        # has nothing to report that the previous line did not already say.
        # Log the first few, then one in every _log_every_when_idle, so the
        # loop stays visibly alive without flooding.  Any real activity resets
        # the counter and restores full logging immediately.
        self._escape_deadlock_if_starved(
            H_clean, worked=(n_fix > 0 or n_refine > 0 or n_stale > 0))

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
              f"self_predicted_rms_db_per_channel={np.array2string(self_rms_per_channel, precision=2)}"
              f"{self._trim_log_fragment()}",
              flush=True)

    # ------------------------------------------------------------------
    # THE RESPONSE-ERROR TRIM -- the second loop.  Added 2026-09-21.
    # ------------------------------------------------------------------
    def _eff_target(self, indices):
        """The target the solver actually aims at.

        The specification diagonal, times the response-error trim when it is
        enabled.  With response_trim_gain == 0 self.y_trim is None and this
        returns the specification slice itself -- the same object the
        pre-trim code passed -- so every solve is bit-identical.
        """
        if self.y_trim is None:
            return self.y_diag_target[indices]
        return self.y_diag_target[indices] * self.y_trim[indices]

    def _trim_log_fragment(self):
        if self.y_trim is None:
            return ""
        in_band = self.y_diag_target.max(axis=1) > 0
        t_db = 10*np.log10(np.maximum(self.y_trim[in_band], 1e-30))
        return (f", trim_db[median={np.median(t_db):+.2f}, "
                f"min={t_db.min():+.2f}, max={t_db.max():+.2f}], "
                f"cum_trim_updates={self.n_trim_updates}")

    def set_test_level_db(self, test_level_db):
        """Rattlesnake calls this on every test-level change (see
        components/random_vibration_sys_id_data_analysis.py:149).  Nothing
        else in this law depends on level -- the data collector already
        refers everything to full level -- but the trim does: a frame
        acquired across the ramp is divided by the wrong number, so its
        diagonal cannot be compared with a prediction cached before the
        change.  Sit out _TRIM_FREEZE_CALLS cycles and drop the stale
        pairing.
        """
        prev = self._test_level_db
        self._test_level_db = test_level_db
        if self.response_trim_gain > 0 and prev is not None and test_level_db != prev:
            self._trim_frozen_calls = _TRIM_FREEZE_CALLS
            self._p_commanded = None
            self._target_commanded = None
            print(f"[optimal_diagonal_control] test level {prev:g} -> "
                  f"{test_level_db:g} dB; response-error trim frozen for "
                  f"{_TRIM_FREEZE_CALLS} cycles (level-ramp frames are not "
                  f"comparable). Accumulated trim is kept, and leaks toward 0 "
                  f"once the trim resumes.", flush=True)

    def _cache_commanded_prediction(self, output, transfer_function):
        """Record what THIS command is predicted to produce, so the next
        cycle's measurement can be compared against the right thing.

        It has to be the command as RETURNED -- after the startup cap --
        because that is what the rig is about to be driven with, and it has
        to be paired with the target that command was solved against, which
        is not necessarily the target in force by the time the measurement
        comes back.
        """
        if self.response_trim_gain <= 0 or output is None:
            self._p_commanded = None
            self._target_commanded = None
            return
        H = self._H_last_used if self._H_last_used is not None else transfer_function
        if H is None:
            self._p_commanded = None
            self._target_commanded = None
            return
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        if H.shape[0] != output.shape[0] or H.shape[1] != self.M:
            self._p_commanded = None
            self._target_commanded = None
            return
        Y = np.einsum('fmn,fnk,flk->fml', H, output, H.conj())
        self._p_commanded = np.maximum(np.real(np.einsum('fmm->fm', Y)), 0.0)
        self._target_commanded = (self.y_diag_target.copy() if self.y_trim is None
                                  else self.y_diag_target * self.y_trim)

    def _update_response_trim(self, last_response_cpsd):
        """Close the loop on the RESPONSE ERROR.

        WHAT IT CORRECTS.  Everything above this method is driven by H.  If
        the plant delivers something other than diag(H X H^H) for reasons H
        does not capture -- bias in the H1 estimate, extraneous input, drive
        clipping, a nonlinearity that never shows up in a linear FRF -- the
        law cannot see it, because it never looks at the measured response.
        There is no integral action, so that offset simply stands.  This
        method is the integrator.

        WHAT IT INTEGRATES, AND WHY THAT PARTICULAR SIGNAL.  Two candidate
        error signals, and the difference matters:

          measured vs PREDICTED, 10log10(z/p).  For a constant multiplicative
          plant bias b this reads 10log10(b) forever, no matter what the trim
          does -- shrinking the target shrinks prediction and measurement
          together.  It is a model-mismatch READOUT, not an error: it never
          goes to zero, so an integrator driven by it just runs to its clamp
          and stops there.  Worked through on paper 2026-09-21 before any of
          this was written; the clamp happening to sit at the right value is
          a coincidence, not a design.

          measured vs SPECIFICATION, 10log10(z/S).  This is the thing the
          test is actually judged on, it is what Rattlesnake's own Response
          Error panel shows, and it drives to zero: trim_db converges
          geometrically to -10log10(b), at which point the measurement sits
          on the specification.  This is what is used.

        WHY IT STILL NEEDS A GATE.  Integrating measured-vs-specification on
        its own winds up at every bin the plant cannot reach.  This frame is
        rank 5 with 6 drives; at a structural null the law lands short of
        specification no matter what is asked of it, the error never clears,
        and the trim would climb to its clamp demanding drive that buys
        nothing.  So the trim is applied only where the solver reached what
        it aimed at -- |10log10(p/target)| <= response_trim_gate_db, both
        sides cached from the cycle that issued the command.  A shortfall the
        SOLVER already predicted is an achievability limit and is none of
        this loop's business; only the part the solver thought it had and
        did not get is model error.

        THE GATE HAS ITS OWN THRESHOLD, not error_threshold_db, and run 35 is
        why -- see the comment on response_trim_gate_db in __init__.  Briefly:
        a gate loose enough to serve as the scheduler's "worth refining" bar
        is loose enough to let the solver's own sub-threshold shortfall
        through, and on a rig whose model mismatch measures 0.06 dB rms that
        shortfall is thirteen times larger than the thing this loop exists to
        correct.

        SPEED.  The per-channel correction is realized by the ordinary
        scheduler: a changed target shows up in _err_db against the current
        H, and step 3 picks those bins up worst-first.  That runs at the
        refinement budget's pace (max_bins_per_update per cycle), which is
        slow across 901 in-band bins.  The COMMON part of each bin's change
        is therefore applied directly to the stored drive CPSD instead:
        scaling X by a positive scalar scales diag(H X H^H) by exactly that
        scalar, so the overall level correction -- the dominant term, and
        the one an operator notices -- is exact and immediate, and only the
        channel-to-channel shape waits for a re-solve.  Scaling by a real
        positive scalar leaves drive coherence and rank untouched, so it
        cannot walk around max_drive_coherence or drive_rcond.

        IN-BAND ONLY.  Out of band there is no drive energy, the measured
        diagonal is noise, and the ratio is unbounded.  This is the fourth
        statistic in this file that has had to learn that lesson.
        """
        if self.response_trim_gain <= 0 or last_response_cpsd is None:
            return
        if self._trim_frozen_calls > 0:
            self._trim_frozen_calls -= 1
            self._p_commanded = None
            self._target_commanded = None
            return
        if self._p_commanded is None or self._target_commanded is None:
            return

        z = np.real(np.einsum('fmm->fm', last_response_cpsd))
        if z.shape != self.y_diag_target.shape:
            return
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        p = self._p_commanded
        tc = self._target_commanded
        spec = self.y_diag_target
        if p.shape != z.shape or tc.shape != z.shape:
            return

        in_band = spec.max(axis=1) > 0
        ok = (np.broadcast_to(in_band[:, None], z.shape)
              & (z > 0) & (p > 0) & (tc > 0) & (spec > 0))
        n_scored = int(ok.sum())
        if n_scored == 0:
            return

        # GATE: did the solver get what it asked for on this bin/channel?
        shortfall_db = np.zeros_like(z)
        shortfall_db[ok] = 10*np.log10(p[ok]/tc[ok])
        ok &= np.abs(shortfall_db) <= self.response_trim_gate_db
        n_gated = int(ok.sum())
        gate_ok = ok.copy()      # passed the solver-agreement gate; see the leak below

        # The error the test is judged on.
        err_db = np.zeros_like(z)
        err_db[ok] = 10*np.log10(z[ok]/spec[ok])
        ok &= np.abs(err_db) > self.response_trim_deadband_db
        n_moving = int(ok.sum())

        if self.y_trim is None:
            self.y_trim = np.ones_like(spec)
        trim_db = 10*np.log10(np.maximum(self.y_trim, 1e-30))

        # Nothing to act on AND nothing accumulated to bleed off: return
        # exactly as the pre-leak code did.
        if n_moving == 0 and not np.any(np.abs(trim_db) > 0.0):
            self._p_commanded = None
            self._target_commanded = None
            return

        step_db = np.clip(-self.response_trim_gain*err_db,
                          -self.response_trim_step_db, self.response_trim_step_db)
        step_db[~ok] = 0.0

        # LEAK (see _TRIM_LEAK_DB).  ONLY bins that failed the SOLVER-AGREEMENT
        # GATE decay toward zero.  NOT bins that passed the gate and merely sat
        # inside the deadband.
        #
        # That distinction is the whole fix, and the first version of it was
        # wrong: leaking deadband bins too made the steady state sit at the
        # deadband EDGE rather than at zero, because the trim unwound until the
        # error grew back past 0.5 dB and then re-corrected -- a limit cycle.
        # verify_response_trim.py TEST 2 caught it: a real -2.00 dB bias
        # settled at -0.83 dB with 0.83 dB of residual error.
        #
        # The justification is simply what each mask means.  A bin inside the
        # deadband has a TRUSTWORTHY measurement saying the correction is
        # right, so it holds.  A bin that failed the gate has NO trustworthy
        # measurement at all -- the solver did not reach its own target, so the
        # residual cannot be attributed to the model -- and a correction held
        # there is unjustified by construction.  Those are the bins that wound
        # up in run 52, and those are the ones that bleed off.
        in_band_full = np.broadcast_to(in_band[:, None], spec.shape)
        if self._trim_gate_fail is None or self._trim_gate_fail.shape != spec.shape:
            self._trim_gate_fail = np.zeros(spec.shape, dtype=np.int32)
        self._trim_gate_fail[gate_ok] = 0
        self._trim_gate_fail[(~gate_ok) & in_band_full] += 1
        idle = ((self._trim_gate_fail >= _TRIM_LEAK_AFTER)
                & in_band_full & (np.abs(trim_db) > 0.0))
        n_leaked = int(idle.sum())
        if n_leaked:
            step_db[idle] = -(np.sign(trim_db[idle])
                              * np.minimum(np.abs(trim_db[idle]), _TRIM_LEAK_DB))

        new_trim_db = np.clip(trim_db + step_db,
                              -self.response_trim_limit_db, self.response_trim_limit_db)
        applied_db = new_trim_db - trim_db
        moved_mask = applied_db != 0.0
        self.y_trim = 10.0**(new_trim_db/10.0)
        self.n_trim_updates += 1

        # Realize the common part of each bin's change immediately (see
        # SPEED above).  Mean over the channels that actually moved.
        n_bins_scaled = 0
        if self.output_cpsd is not None and self.output_cpsd.shape[0] == spec.shape[0]:
            # Follow moved_mask, not ok -- leaked bins changed too, and the
            # in-place rescale must stay consistent with the new target.
            cnt = moved_mask.sum(axis=1)
            rows = np.where(cnt > 0)[0]
            if rows.size:
                g_db = applied_db[rows].sum(axis=1)/cnt[rows]
                g = 10.0**(g_db/10.0)
                self.output_cpsd[rows] *= g[:, None, None]
                n_bins_scaled = int(rows.size)

        if self.n_trim_updates <= 5 or self.n_trim_updates % 20 == 0:
            moved = applied_db[moved_mask]
            at_clamp = int(np.sum(np.abs(new_trim_db[in_band]) >=
                                  self.response_trim_limit_db - 1e-9))
            print(f"[optimal_diagonal_control] response-error trim #{self.n_trim_updates}: "
                  f"{n_scored} in-band channel-bins scored, {n_gated} passed the "
                  f"{self.response_trim_gate_db:g} dB solver-agreement gate, "
                  f"{n_moving} outside the "
                  f"{self.response_trim_deadband_db:g} dB deadband; "
                  f"{n_leaked} leaking back toward 0; "
                  f"applied median {(np.median(moved) if moved.size else 0.0):+.3f} dB "
                  f"(max |{(np.abs(moved).max() if moved.size else 0.0):.3f}|), "
                  f"{n_bins_scaled} bins rescaled "
                  f"in place, {at_clamp} channel-bins at the "
                  f"+/-{self.response_trim_limit_db:g} dB clamp.", flush=True)

        self._p_commanded = None
        self._target_commanded = None

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
            out = self._startup_capped(self.output_cpsd, transfer_function,
                                       last_output_cpsd)
            self._cache_commanded_prediction(out, transfer_function)
            return out
        # The trim runs BEFORE the refinement, so every bin the scheduler
        # touches this call is solved against the corrected target rather
        # than against one the measurement has already contradicted.
        self._update_response_trim(last_response_cpsd)
        if transfer_function is not None:
            self._refine_batch(transfer_function)
        out = self._startup_capped(self.output_cpsd, transfer_function,
                                   last_output_cpsd)
        self._cache_commanded_prediction(out, transfer_function)
        return out

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