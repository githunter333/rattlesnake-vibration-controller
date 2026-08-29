# CVA-Innovations Live FRF Integration — Design Notes (2026-08-26)

Status: **design discussion only, nothing implemented yet.** This captures
the architecture investigation and design decisions from a session
following on from the offline `globalcva/` research (see
`examples/sixdrive12resp/results/global_cva_session_notes_2026-08-25.txt`
for the CVA-innovations method itself: validated `lags=40, rank=66`
defaults, the innovations-form extension `x(k+1)=Ax+Bu+Ke(k)`,
`y(k)=Cx+Du+e(k)`, and its noise-robustness / damping-bias / close-mode-pair
results).

That prior work was all offline research on synthetic data. This note is
about what it would take to bring the innovations-form CVA estimator into
the *live* Rattlesnake control loop as an alternative to H1, and — the
harder problem — how to hand a live sequence of CVA-estimated FRFs to a
control law that expects a reasonably stable FRF.

---

## 1. Origin question

Five original points, paraphrased:

1. Make FRF computation pluggable (H1 vs CVA-innovations); if CVA is used,
   want a *fast* FRF update to enhance control. How does this interact
   with exponential averaging?
2. Fast control update using `Sxx = Hpinv @ Syy @ Hpinv^H` from the
   fast-updated FRF.
3. Is `optimal_diagonal_control` (SDP-based, working well now) suited to
   very short full-level runs?
4. Fast FRF updating is wanted specifically when test level ramps to 0 dB.
5. Interest in switching H1's frequency resolution by an on-the-fly
   integer factor (2, 4, 8, ...) at levels like -3 or 0 dB, to speed up
   FRF computation. How hard?

---

## 2. Current live architecture (as found in the codebase)

Pipeline, three synchronized subprocesses:

- **`components/data_collector.py`** (`RandomDataCollector.acquire`) —
  acquires raw time frames, applies the configured window (Hann/etc.) via
  `self.response_window` / `self.reference_window`, then does
  `rfft(...)` and pushes `(response_fft, reference_fft)` onto
  `data_out_queues`. **Raw, un-windowed time-domain data never leaves this
  process today** — only the windowed FFT frames do.
- **`components/spectral_processing.py`** (`SpectralProcessingProcess`) —
  the single place FRF gets computed. Accumulates `response_fft`/
  `reference_fft` frames into spectral matrices, either
  `AveragingTypes.LINEAR` (rolling FIFO of `averages` frames) or
  `AveragingTypes.EXPONENTIAL` (`coef * new + (1-coef) * old`, applied
  directly to the spectral matrices). Computes FRF via the `Estimator`
  enum (`H1/H2/H3/HV`) off those matrices, plus coherence, then pushes
  `(frames, frequencies, frf, coherence, response_cpsd, reference_cpsd,
  frf_condition)` out.
- **`components/random_vibration_sys_id_data_analysis.py`**
  (`RandomVibrationDataAnalysisProcess.run_control`) — pulls the latest
  spectral-processing output each cycle, sets
  `control_frf = self.control_frf if update_tf_during_control else
  self.sysid_frf`, and calls the loaded control law's `.control(control_frf,
  control_coherence, frames, frames_in_cpsd, last_response_cpsd,
  last_drive_cpsd)`. This is already exactly the "swap FRF source and feed
  it to control" toggle point 1 needs.

Key existing mechanisms relevant to this design:

- **`control_laws/control_laws.py:pseudoinverse_control`** already
  implements `tf_pinv @ specification @ tf_pinv.conj().T` — i.e. point 2's
  `Sxx = Hpinv @ Syy @ Hpinv^H` map already exists and is also the
  buzz-baseline / warm-start closed form inside `optimal_diagonal_control`.
- **`control_laws/optimal_diagonal_control.py`** — SDP-based, coherence-
  capped, with a budgeted three-step scheduler
  (`optimal_diagonal_control_flow.md`): Step 1 re-solves bins where
  `‖ΔH‖/‖H‖ > frf_update_threshold` (default 0.05), capped at half the
  per-call budget specifically to stop *noise-driven FRF jitter* from
  starving Step 2's progressive first-time coverage. `max_bins_per_update`
  defaults to 20 — full-spectrum SDP coverage can take many calls, which
  matters directly for point 3 (short full-level runs may never reach full
  coverage).
- **`control_laws/optimal_diagonal_control_fast.py`** — ~18-20x faster
  per-bin solve (Burer-Monteiro unconstrained factorization), but *only*
  engages when the incoming `H` is bit-for-bit identical to the previous
  call (`np.array_equal`, tracked as `_h_changed_this_call`). Any H that
  changes every cycle permanently falls off this fast path.
- **Test-level transitions do not clear spectral-processing buffers.**
  `adjust_test_level()` / `start_control()` only set `skip_frames` in the
  collector (to drop ramp-transient frames) — there's no
  `CLEAR_SPECTRAL_PROCESSING` call on a level change. So after ramping to
  0 dB, the spectral matrices (and hence H1's FRF) still contain a mix of
  pre-ramp and post-ramp frames until enough new data displaces/decays the
  old (FIFO depth for linear averaging, `(1-coef)^n` decay for
  exponential) — a real bias source on any amplitude-dependent
  (nonlinear) structure, and the direct motivation for point 4.
- **Frequency resolution (point 5) is not an isolated parameter.**
  `RandomVibrationMetadata.samples_per_frame` simultaneously determines
  `fft_lines`, `frequency_spacing`, and `samples_per_output` (COLA block
  size for the *output* signal generator), and
  `SpectralProcessingProcess`'s buffers are pre-shaped by
  `num_frequency_lines`. Changing resolution on the fly means a
  synchronized re-init of acquisition frame size, COLA overlap-add state
  in signal generation (risk of an output discontinuity/glitch), and
  spectral-processing buffers (equivalent to a `CLEAR_SPECTRAL_PROCESSING`
  — same convergence-restart cost as the level-change problem above). Not
  a single-parameter tweak; a coordinated resync of three subprocesses at
  a chosen boundary.

---

## 3. The core new problem: handling a *sequence* of CVA-estimated FRFs

Raised directly: with `optimal_diagonal_control`, we want a "pretty
consistent, if updated" FRF — but naively feeding a freshly re-identified
CVA FRF into `control_frf` every cycle breaks that.

**Why you can't average CVA's A/B/C/D across successive fits.** Each CVA
run on a fresh data window picks its own state basis (canonical variates
come out of that window's own Hankel-matrix SVD/whitening) — state `x_k`
in one fit and `x_k` in the next aren't expressed in the same coordinates,
and rank/order selection can legitimately pick a different number of
modes window to window. Averaging `A_1` and `A_2` directly is meaningless
in the same way averaging two independently-derived modal models' raw
matrices (without first aligning mode shapes) is meaningless.

**What IS safe to combine, in order of preference:**

1. **The evaluated FRF curve**, `H(f) = C(jωI-A)⁻¹B + D`. Basis-independent
   — an ordinary complex array, same object H1 already produces. This is
   where any smoothing belongs, and it means the existing
   `exponential_averaging_coefficient` concept can be reused conceptually
   without new plumbing.
2. **Pole-level (modal) parameters**, via mode-*tracking* (nearest
   frequency / MAC-style shape correlation), not averaging. Basis-
   independent (`modes_from_A` already extracts these), but requires
   matching modes between windows since order can vary. Buys a
   stabilization-diagram-style consistency check: a pole that's stable
   across 2-3 consecutive windows is real; one that appears once and
   vanishes is spurious SVD noise. Useful as a trust/quality signal more
   than as the primary path to `control_frf`.
3. **Raw data itself** — the real analog of what H1's "averaging" does.
   H1 doesn't fuse independent FRF point-estimates; it accumulates raw
   spectral content and runs one estimator over the total. The CVA
   equivalent is a rolling/sliding window over raw time samples,
   recomputed periodically, treating the newest fit as *the* estimate
   rather than something to merge post-hoc with prior fits.

**Why the mismatch specifically breaks `optimal_diagonal_control`:** H1's
averaging naturally produces a low-jitter, slowly-converging FRF (variance
shrinks as data accumulates), so Step 1's drift check rarely fires once
control is steady. An independently-refit CVA estimate every window does
NOT have shrinking variance just because time passed — Step 1 will read
ordinary CVA estimation noise as "drift," burn its half-budget
revalidating bins that didn't really change, and starve Step 2's
progressive coverage. Separately, `optimal_diagonal_control_fast`'s
bit-equality gate means a "recompute and replace every cycle" CVA feed
permanently falls off the fast path — paying full SDP cost every bin,
every call, exactly when speed was the point.

### Resolution: treat CVA FRF updates as *gated, published events* — not a continuous replace

- Hold `control_frf` literally constant (same array) between updates.
- Only publish a new candidate when it is **both**:
  - **well-supported** — use the innovations-form's own residual/
    covariance output (`Q, R, S, K`, innovations energy) as a
    confidence/quality signal, CVA's analog of H1's coherence;
  - **meaningfully different** — gate on `‖ΔH_candidate‖/‖H_published‖`
    against a deadband tied to (a fraction of) `optimal_diagonal_control`'s
    own `frf_update_threshold`, so a publish only ever corresponds to a
    change large enough that Step 1's drift re-solve *should* fire anyway.
- High-overlap sliding windows (successive raw-data windows sharing most
  of their samples) reduce window-to-window estimation variance
  mechanically, shrinking spurious deltas before the gate even has to
  reject them — but the gate, not the overlap, is what actually delivers
  "consistent, if updated."

This also restores `optimal_diagonal_control_fast`'s speed benefit almost
for free: between real publish events `control_frf` is bit-identical, so
the ~20x unconstrained BM path engages; the instant a genuine change
publishes (e.g. post-ramp settling at 0 dB), `H` changes and the control
law correctly falls back to the coherence-capped SDP for that transition
— exactly when the more careful, constrained answer is wanted.

---

## 4. Two-piece breakdown for future implementation

Not coded yet — this is the agreed decomposition, with the interface
contract between the pieces nailed down so each side can be built (and
tested) independently.

**Piece 1 — CVA-innovations insertion (estimation side).** Produce a
*candidate* FRF + confidence signal on a live cadence:
  - a raw, un-windowed time-domain tap out of `data_collector` (the
    existing FFT-frame queue is windowed and unusable for CVA's
    Hankel/state-transition structure — needs a new parallel output or a
    branch upstream of the window multiply in `acquire()`);
  - rolling/sliding-window buffering of that raw data;
  - `global_cva_innovations` run per window at the validated `lags=40`
    settings;
  - evaluate the resulting realization onto the frequency grid for
    `candidate_H(f)`, plus the innovations-based confidence metric.
  - This piece's job ends at emitting `(candidate_H, confidence_metric)`
    — it does not decide whether to publish.

**Piece 2 — control-loop "has FRF changed" logic (decision side).**
Consumes the `(candidate_H, confidence_metric)` stream (from Piece 1, or
in principle from any future estimator) and decides whether to publish:
  - deadband check against `frf_update_threshold`;
  - confidence gate from the innovations residual;
  - emits `control_frf` that only changes on a genuine publish event and
    is otherwise bit-identical to the previous cycle.
  - This is the part that actually needs to sit close to (or in front of)
    `RandomVibrationDataAnalysisProcess`'s `control_frf` assignment.

**The seam:** `(candidate_H, confidence_metric) -> [Piece 2 gate] ->
control_frf`. This contract is deliberately estimator-agnostic and lets
Piece 2 be validated against synthetic candidate sequences with known
jitter/step characteristics before Piece 1 is wired into live acquisition
at all.

---

## 5. Open items / not yet decided

- Exact window length / overlap fraction for the CVA sliding window in
  live use (offline research used 1-2s fixed blocks; live cadence and
  `frame_time` constraints haven't been reconciled with that yet).
- Exact form of the innovations-based confidence metric (raw residual
  energy vs. a normalized/whitened version; per-bin vs. scalar).
- How points 4 and 5 (0 dB-triggered fast update, on-the-fly resolution
  switching) relate to Piece 2's gating — both share the same
  underlying need for a "coordinated resync at a triggerable boundary"
  primitive, noted but not designed.
- Where exactly the raw-time tap should live in `data_collector.py`
  (new parallel `data_out_queue`, or a flag-gated branch) — not decided.

---

## 6. Alternative/complement to point 5 — 1/6-octave control at high test level

Revisiting point 5 (on-the-fly FRF resolution switching) with a different
mechanism: instead of changing the underlying FFT frequency resolution
(which requires a synchronized resync of `data_collector`,
`signal_generation`, and `spectral_processing` — see section 2 above),
switch the *control law's* decision granularity from per-FFT-line to
1/6-octave-band, triggered the same way as the 0 dB transition in
section 3. Narrowband acquisition, drive synthesis, and response
measurement/reporting all stay exactly as they are; only how the control
law aggregates frequency-domain data before solving, and broadcasts the
result back out, changes.

### Existing prior art (already validated offline, SISO)

- **`spectral_analysis/fractional_octave.py`** — standalone,
  Rattlesnake-independent utilities: `octave_band_frequencies` (ANSI
  S1.11/IEC 61260 base-2 band edges, e.g. `fraction=6` for 1/6 octave),
  `narrowband_cross_spectra`, `octave_band_psd`, `octave_band_frf`,
  `narrowband_coherence`, `octave_band_coherence`. All SISO — band
  averaging is energy-preserving averaging of an ordinary narrowband
  Welch/CSD estimate into log-spaced bands, not a constant-Q filter bank.
- **`siso_octave_control/code/build_siso_octave_control_demo.py`**
  (results in `siso_octave_control/results/`) — closed-loop SISO random
  control on the six-drive/twelve-response frame system (node 1 drive ->
  node 18 response), 1/6-octave bands over 100-1000 Hz (46 bands). Drive
  synthesis and response measurement stay full narrowband throughout;
  only the control decision is coarsened. Two scenarios compared: initial
  drive estimate seeded from full narrowband H1 (683 gains) vs. from
  1/6-octave-averaged H1 only (46 gains, flat within each band). **Both
  converge to the same octave-band target level** (RMS band error well
  under 2%) via the same iterative octave-band correction loop
  (`target/achieved` per band, broadcast flat across that band's
  narrowband lines). The only difference: octave-only-H init leaves
  visible leftover *narrowband ripple within each band*, since it never
  had within-band H shape available to cancel individual
  resonances/antiresonances — narrowband-H init does cancel that shape.

### Why this is an easier lift than section 2's resolution-switching

No change to `samples_per_frame`, `fft_lines`, `frequency_spacing`,
COLA output blocking, or any spectral-processing buffer shape — the
narrowband FRF and response keep flowing through `spectral_processing.py`
unmodified. This is purely a re-chunking of what the control law does
with narrowband data before solving, plus broadcasting its per-band
output back out flat across that band's narrowband lines (the
`expand_octave_to_narrowband` pattern already implemented in the SISO
demo). No synchronized multi-subprocess resync, no COLA discontinuity
risk.

### What the MIMO generalization needs

`fractional_octave.py`'s band-averaging is currently scalar
(`Sxx`/`Sxy` per line). Generalizing to MIMO is mechanical, not new
research: apply the same energy-preserving band-average to the
matrix-valued `reference_spectral_matrix` and
`response_reference_spectral_matrix` already computed per-line in
`spectral_processing.py` (average over the lines inside each band, same
einsum accumulation pattern, before the `Gxf @ Gffpinv` H1 divide), and
the same treatment for `specification_cpsd_matrix` and the achieved
response CPSD before computing dB error.

Downstream, `optimal_diagonal_control`'s budgeted scheduler
(Step 1/2/3, coherence cap, buzz warm-start) doesn't care whether "a bin"
is an FFT line or an octave band — feed it one `H_band`/`y_target_band`
pair per band (~46 instead of hundreds of lines) and the existing
machinery carries over unchanged. `max_bins_per_update` reaches full
coverage in far fewer calls at band granularity. This reframes point 3:
a short full-level run that can't reach full per-line SDP coverage in
time might reach full *octave-band* coverage easily — octave mode could
be a better answer to "short full-level runs" than treating
`optimal_diagonal_control` as poorly suited to them.

### The real trade-off (not just a speed optimization)

Octave-only control gives up the ability to cancel within-band narrowband
structure (individual resonances/antiresonances) — confirmed directly by
the SISO demo's leftover-ripple result. Triggering this specifically at
0 dB trades control authority (fine per-line shape) for solve
speed/coverage, right at the moment you're at the highest, most
demanding level. Whether that trade is acceptable likely depends on how
densely modes are packed within a given 1/6-octave band — a system with
widely-spaced modes loses little; this system (from the CVA close-pair
work) has several closely-spaced mode pairs where within-band shape
matters more.

There's also a transition-consistency concern that parallels section 3's
FRF-changed-logic problem: switching mode mid-test flattens the drive's
within-band shape abruptly at the level transition — the same kind of
gated/blended-transition thinking (rather than a hard instantaneous mode
switch) would likely apply here too, though not designed yet.

### Where this would live

A band-vs-line mode flag feeding the control law's per-bin loop (e.g. a
wrapper/variant of `optimal_diagonal_control`'s `_refine_batch`), plus the
matrix-valued band-averaging step ahead of it. No changes needed to
`data_collector`, `signal_generation`, or `spectral_processing` frame
sizing. Not implemented — this section is a design/feasibility record
only, same status as the rest of this document.

---

## 7. Scope estimate — adding CVA-innovations as a live FRF method

Breaking down what's already done vs. genuinely new work, since the gap is
uneven.

### Already done (offline research, `globalcva/`)

The core estimator — `global_cva_innovations` in
`globalcva/global_cva_frf.py` — exists and is validated: `lags=40,
rank=66` defaults confirmed via sweep, noise-robustness confirmed
(`noise_injection_compare_innovations.py`), damping-bias reduction
confirmed, mode-level accuracy checked against ground truth
(`check_damping_bias_innovations.py`). It already returns everything a
live estimator needs — `A, B, C, D` plus `Q, R, S, K, P, innovations`.
`frf_from_ss` already evaluates a realization onto an arbitrary frequency
grid. The research risk is retired; what's left is integration engineering.

### New work, roughly foundational-to-incidental

1. **Raw-time-tap (the biggest structural gap).**
   `data_collector.acquire()` windows and `rfft()`s every frame before
   anything leaves that process (see section 2) — CVA needs raw,
   un-windowed, causally-ordered time samples, which exist nowhere
   downstream of acquisition today. Needs a second output path out of
   `data_collector` (or a flag-gated branch), tapped *before* the
   `response_window`/`reference_window` multiply, buffered across many
   acquisition frames to the block length CVA needs (offline validation
   used 1s @ 5120 Hz = 5120 samples; must be reconciled against live
   `frame_time`). Real new plumbing, not a config change — closest
   existing analog is the buffering `SpectralProcessingProcess` already
   does, just on raw data instead of FFT frames.

2. **Live wrapper around `global_cva_innovations`.** New subprocess or a
   new branch inside spectral processing: rolls the raw buffer, calls the
   fit, evaluates `frf_from_ss` on the live frequency grid
   (`frequency_spacing * arange(fft_lines)`), emits output shaped like the
   existing `(frames, frequencies, frf, coherence, response_cpsd,
   reference_cpsd, frf_condition)` tuple so downstream code doesn't need
   to know the estimator changed. Mechanical but nontrivial glue.

3. **Coherence-analog.** Downstream code expects `coherence` — used for
   GUI display and, more importantly,
   `optimal_diagonal_control`'s buzz-baseline cross-term construction
   (`_match_coherence_phase`). CVA doesn't produce coherence natively, but
   the innovations covariance gives a legitimate model-based substitute
   (predicted signal power vs. innovations/noise power per bin) — a new
   derivation, not a lookup.

4. **`Estimator` enum + selection plumbing.** Small, low-risk: add
   `Estimator.CVA_INNOVATIONS` alongside `H1/H2/H3/HV` in
   `spectral_processing.py`; add a branch in
   `RandomVibrationEnvironment.get_spectral_processing_metadata()`'s
   if/elif chain (currently maps the UI string `'H1'/'H2'/'H3'/'Hv'` to
   the enum); add the option to the `estimatorComboBox` UI widget and the
   xlsx-template parsing (`worksheet.cell(20,2)`, `'System ID
   Estimator:'`). Per-environment dropdown, defaults to H1 today — adding
   a choice is additive, no regression risk to existing H1/H2/H3/HV
   behavior.

5. **Publish/gating logic (section 3's design).** Determines whether CVA
   is genuinely *usable* in `optimal_diagonal_control`, not just
   structurally present — without it, every fresh window's estimation
   noise gets read by the control law as drift. Treat as required for real
   deployment, though separable enough to prototype/test independently of
   item 1 (per the seam already defined: it only consumes a
   `(candidate_H, confidence_metric)` stream).

### Not yet validated, matters before trusting this on a real test

- **Channel-count scaling.** `lags=40, rank=66` validated on the
  6-drive/12-response example only. Hankel matrix size scales with
  `(drives+responses) x lags` — a production system with more channels
  needs its own rank/lags sweep; the validated defaults don't
  automatically transfer.
- **Real-hardware behavior.** All validation so far is synthetic-model-
  driven (known ground truth, clean generative white-Gaussian noise
  assumption in the noise-injection sweep). Real sensor noise, actuator
  nonlinearity, and rig dynamics aren't guaranteed to match — worth a
  real-data check before this is anywhere near a control loop.

### Reassuring point: no hard real-time race

CVA's fit doesn't need to complete before the next control cycle. The
existing async, queue-based subprocess pattern (and the gating logic
specifically) means CVA can run on its own, slower cadence, and the
control loop just keeps using the last-published FRF until a new one
clears the gate — exactly like `run_control` already does with H1 today.

### Net read

Estimator math: done. Plumbing (raw tap, live wrapper, enum/UI wiring):
real but mechanical, low regression risk to existing functionality.
Gating logic + a fresh rank/lags/real-data validation pass: the two
pieces that turn "technically works" into "trustworthy in a control
loop." Not implemented — this section is a scope estimate only, same
status as the rest of this document.

## 8. Core wiring for live test-level -> octave-band switching (implemented 2026-08-27)

`octave_band_switching_control` (section 6) needed a live dB test-level
signal to decide when to switch modes, but Rattlesnake's control-law
interface never carried one — `system_id_update()`/`control()` only ever
see FRF/CPSD data. This section closes that gap with a small, targeted
core-code change. Implemented directly on the device repo (not yet
committed).

**Unit note:** by the time test level reaches `RandomVibrationEnvironment`,
it has already been converted from dB to a linear scale factor by the UI
(`db2scale(...)` in `random_vibration_sys_id_environment.py`, both in
`RandomVibrationUI.start_control` and `.change_control_test_level`). The
core patch converts back to dB with the existing, exact inverse
`scale2db = lambda scale: 20*np.log10(scale)` (`components/utilities.py`)
rather than threading a second dB-valued payload through the queues.

**`components/random_vibration_sys_id_data_analysis.py`:**
- New `RandomVibrationDataAnalysisCommands.SET_TEST_LEVEL_DB = 4` (the old
  commented-out placeholder `# SHUTDOWN_ACHIEVED = 4` shifted to `= 5`,
  it was never active).
- Mapped in `__init__` to a new handler `set_test_level_db(self, data)`,
  and a new `self.current_test_level_db = None` attribute.
- Handler stores `self.current_test_level_db = data`, then — only for
  class-style control laws (`control_python_function_type == 2`) that
  define `set_test_level_db` (checked with `hasattr`, so this is a no-op
  for every existing control law) — calls
  `self.control_function.set_test_level_db(data)`.
- `initialize_sysid_parameters` (class-style construction branch) syncs
  `self.current_test_level_db`, if already known, into a freshly
  constructed control law immediately after building it. Belt-and-
  suspenders only: within one process, the command queue is strictly
  FIFO and `INITIALIZE_PARAMETERS` is always queued and fully processed
  before `SET_TEST_LEVEL_DB`/`RUN_CONTROL` in the observed call sites, so
  this path shouldn't be reachable today — kept for safety against future
  reordering.

**`components/random_vibration_sys_id_environment.py`:**
- Imports `scale2db` alongside the existing `db2scale`.
- `RandomVibrationEnvironment.start_control(self, data)`: posts
  `SET_TEST_LEVEL_DB, scale2db(data)` to `data_analysis_command_queue`
  right before the existing `RUN_CONTROL` post, so the control law knows
  the level before its very first `control()` call.
- `RandomVibrationEnvironment.adjust_test_level(self, data)`: posts the
  same command whenever the user changes test level live during a run
  (alongside the existing posts to `signal_generation_command_queue` and
  `collector_command_queue`).

**Why this is safe for every other control law:** the new command is
additive — nothing subscribes to it unless it defines
`set_test_level_db`, and the `hasattr` guard means `optimal_diagonal_control`,
`optimal_diagonal_control_fast`, generator-style, and function-style
control laws are all completely unaffected. No existing command, queue,
or method signature was changed.

**Verification performed:** both edited files parse cleanly
(`ast.parse`) and the diffs are minimal, targeted string replacements
reviewed line-by-line. A true runtime import of `components.*` could not
be exercised on the device bridge's sandboxed VM (it lacks the PySide6/Qt
GUI shared libraries the `components` package pulls in at import time,
and has no sudo/network to install them) — same environment limitation
noted earlier in this doc for cvxpy. **Not yet tested against a live
Rattlesnake run.**

**Still open / suggested next steps:**
- Run this for real against the sixdrive12resp example (per section 6's
  `extra_parameters` string) and confirm `octave_band_switching_control`
  actually flips modes when the test level selector crosses the
  configured `switch_level_db`.
- Watch the log output (`self.log(...)` calls already in
  `octave_band_switching_control._on_mode_switch`) during that run to
  confirm the mode-switch bookkeeping fires at the right moments.
- No commit made yet for any of this — commit only if/when asked.

## 9. Live validation on sixdrive12resp (2026-08-28/29)

The core wiring from section 8 and the `octave_band_switching_control`
law from section 6 were run for real against the sixdrive12resp example,
closing out the "not yet tested against a live Rattlesnake run" item.

**Setup.** `examples/sixdrive12resp/results/sdynpy_frame6x12_profile.xlsx`
was updated (backup kept as `sdynpy_frame6x12_profile_backup_before_octave.xlsx`)
so its Control Python Script/Function/Parameters point at
`octave_band_switching_control.py`, function `octave_band_switching_control`,
parameters `1e-6,0.05,20,1.0,0.95,1.0,0.0,6` (base-class params unchanged;
`frequency_spacing=1.0` Hz, `switch_level_db=0.0`, `octave_fraction=6`).
Launched via `make launch-rattlesnake` (bare `RANDOM` environment, not a
loaded Combined profile — the user re-entered these three fields by hand
in the GUI to match, confirmed correct before starting control).

**Switch confirmed working, both directions, on a live run.** With
control running and the GUI's test level raised from -3 dB to 0 dB,
`gui_debug.log` showed:
```
[octave_band_switching_control] SWITCHING to octave-band control (test level 0 dB >= switch_level_db 0 dB), switch #1
```
followed by clean `_refine_batch_octave` calls (all 20 bands refined,
`cum_band_solves` climbing steadily, zero solver failures). Dropping the
level back to -3 dB (which ramps through -1 dB en route, per the
`Test Level Ramp Time` of 2 s) produced:
```
[octave_band_switching_control] SWITCHING to narrowband control (test level -0.9999999999999997 dB < switch_level_db 0 dB), switch #2
```
with `n_refined_total` correctly resetting and climbing back up from a
low count afterward — confirming `_on_mode_switch`'s `sdp_refined[touched]
= False` reset (section 6/the control law's own docstring) actually forces
a genuine re-solve of the previously band-broadcast lines rather than
leaving them on stale coarse values. No tracebacks, no repeated
switch-flapping (only switch #1 and #2 across the whole session), no
`cum_solver_failures` anywhere in the log. This confirms the full chain
end-to-end: UI test-level selector -> `db2scale`/`scale2db` -> the new
`SET_TEST_LEVEL_DB` command -> `RandomVibrationDataAnalysisProcess.set_test_level_db()`
-> `octave_band_switching_control.set_test_level_db()` -> the mode-switch
decision itself.

**Error cost of octave-band control, measured.** Comparing the GUI's
Response Error (dB) at 0 dB in octave mode against the SDP solver's own
self-predicted RMS error per channel in narrowband mode immediately
before the switch:

| Channel | Narrowband (self-predicted, dB) | Octave (measured, dB) |
|---|---|---|
| 1 | 5.8 | 7.1 |
| 2 | 1.9 | 4.2 |
| 3 | 2.7 | 4.9 |
| 4 | 2.3 | 4.2 |
| 5 | 2.1 | 4.3 |
| 6 | 2.3 | 4.6 |
| 7 | 12.1 | 12.7 |
| 8 | 9.4 | 9.2 |

Channels 1-6: octave mode runs ~1.5-2.5 dB worse — a real but modest and
bounded cost, consistent with the expected within-band-ripple trade-off.
Channels 7-8: essentially unchanged (channel 8 even marginally better in
octave mode). Those two channels were already the hardest to control at
full per-line resolution, so their large error looks like an inherent
property of this system/spec (an under-actuated response direction, most
likely) rather than something the octave switch causes. Caveat: the
narrowband column is the solver's own self-predicted RMS at one point in
time, not the same measured-over-time quantity as the GUI's Response
Error column, so the comparison is directional, not exact.

**New diagnostic tool: `plot_octave_band_overlay.py`.** Added at
`examples/sixdrive12resp/code/plot_octave_band_overlay.py`. Answers "how
do you visualize the sixth-octave control level on the normal narrowband
frequency axis" — reads a `.nc4` saved via the Run tab's "Save Current
Spectral Data" button (which conveniently embeds both the measured
response/drive CPSDs AND the specification actually used for that run, so
no separate spec file or axis-order bookkeeping is needed), band-averages
the response (or drive) diagonal PSD into 1/6-octave bands using
`spectral_analysis/fractional_octave.py`'s existing `octave_band_frequencies`/
`octave_band_psd` (the validated energy-preserving averaging from the SISO
octave demo — NOT the plain-complex-mean-of-H approximation
`octave_band_switching_control` uses internally for speed), and broadcasts
each band's level back across its narrowband lines to draw a "staircase"
overlaid on the raw narrowband response and the specification.

Validated against three saved `.nc4` files: an old `optimal_diagonal_control`
run (smoke test, script mechanics only) and a fresh pair saved directly
from this session's live run — one captured in narrowband mode, one in
octave mode. The two plots make the trade-off from the table above
visible directly: the narrowband-mode plot shows the response hugging
spec tightly across all 8 channels; the octave-mode plot shows visibly
more ripple and, on channel 7 particularly, a real rolloff above ~750 Hz
that the band-level control can't reach (matching that channel's 12.7 dB
error). The band-averaged staircase itself generally tracks spec better
than the raw narrowband ripple in octave mode — confirming the control
law is doing its job at the resolution it's given, while also making
concrete exactly how much resolution is being given up.

**Net read.** The end-to-end feature — core Rattlesnake wiring plus the
octave-band control law — works correctly on a real run of the
sixdrive12resp system: switches fire on the right side of the threshold,
in both directions, with correct state bookkeeping and no errors. The
accuracy cost of running at 1/6-octave resolution instead of full
per-line is real, measured, and modest for most channels (~1.5-2.5 dB),
with the two already-difficult channels unaffected. No further core code
changes are needed for the switching mechanism itself; remaining open
items are analysis, not plumbing (e.g. quantifying how much of the
per-channel gap is the control law's plain-complex-mean band-averaging
approximation specifically, versus genuinely uncancellable within-band
dynamics — `plot_octave_band_overlay.py`'s more rigorous band-averaging
could be used for that comparison directly).
