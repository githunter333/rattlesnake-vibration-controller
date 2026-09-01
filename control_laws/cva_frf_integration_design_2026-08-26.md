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

3. **Coherence-analog — diagnostic only, not a blocker (corrected
   2026-08-29, see section 10).** Originally scoped as "needed by
   `optimal_diagonal_control`'s buzz-baseline construction" — that was
   wrong. Read every control law's actual computation
   (`control_laws.py`'s `pseudoinverse_control`/`buzz_control`/
   `buzz_control_class`, `optimal_diagonal_control.py`,
   `octave_band_switching_control.py`): all of them accept a
   `multiple_coherence` parameter but none reference it anywhere in the
   math. The buzz baseline's cross-term matching
   (`_match_coherence_phase`) recomputes its own simple bivariate
   coherence directly from the raw `sysid_response_cpsd` — a different,
   locally-derived quantity, not the passed-in `multiple_coherence`
   array. `multiple_coherence` is purely diagnostic in the Python path:
   it feeds the GUI's Multiple Coherence plot
   (`abstract_sysid_environment.py`) and gets saved to the nc4 as
   `frf_coherence`. One exception: `matlab_interface.py` does forward
   `self.multiple_coherence` into MATLAB as a real argument, so a custom
   MATLAB-side control law could legitimately consume it — worth a check
   there specifically, given the MATLAB->Python transition underway.
   Net: CVA can go live for real control without ever solving this;
   coherence-analog is now a nice-to-have GUI item, not a dependency.
   See section 10 for the empirical investigation into what a valid
   analog would actually look like, since the naive approach turned out
   not to be as simple as first assumed.

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

## 10. Coherence-analog investigation for CVA (2026-08-29)

Follow-up to section 7 item 3, after that item's premise turned out to be
wrong (see the correction inline in item 3 above): coherence isn't
load-bearing for any Python control law, so this is diagnostic/GUI work,
not a blocker. Investigated anyway since a GUI-facing quantity that lies
to the operator is still a real problem, and the question turned out to
have a genuinely useful answer.

Script: `globalcva/coherence_analog_investigation.py`. Uses the same
synthetic 6-drive/8-response system as the rest of `globalcva/`
(`build_true_system.build_system()`), injects known-fraction additive
white measurement noise (0-30% of per-channel RMS), and grades three
candidate coherence-analogs at each noise level against each other and
against the exact TRUE system transfer function evaluated through the
same measurement pipeline.

**Candidate (a) — algebraic substitution.** Plug a candidate `H(f)` into
Rattlesnake's own multiple-coherence formula (`components/
spectral_processing.py`), using the same measured `Guu`/`Gyy`:
`coherence_i(f) = Re[(H @ Guu @ H^H)_ii] / Gyy_ii`. **Key finding:** this
is NOT bounded in [0,1] for any `H` other than the data's own H1 fit —
confirmed empirically by substituting the exact, noise-free TRUE system
`H` and watching it swing up to ~1.7 and down toward 0 with the same
spiky character as CVA's substituted `H`. This isn't a CVA weakness; it's
structural. The formula's boundedness is a special property of H1 being
the per-bin least-squares-optimal fit to *this* `Guu`/`Gyu` (Cauchy-
Schwarz forces the prediction/residual cross-term to vanish); any other
`H` — even the physically correct one — leaves that cross-term nonzero
and unbounded. Candidate (a) is not usable as-is; substituting CVA's H
this way inherits the same defect the true system's H would.

**Candidate (b) — innovations-based decomposition.** Use
`global_cva_innovations`'s own fitted Kalman gain/noise covariances to
split the model's own predicted PSD into signal-path and noise-path terms
(both PSD-additive, so provably bounded in [0,1] by construction — no
optimality assumption needed here). Empirically, with the validated
defaults (`lags=40, rank=66, refine_iters=1`), it collapses to
near-zero (0.00-0.15) almost immediately, even at zero added measurement
noise. It's dominated by unmodeled dynamics / finite subspace rank being
absorbed into the fitted noise covariance, not by actual measurement
noise — i.e. it measures model-order adequacy, not signal-vs-noise
fraction. Would need substantially higher model order and/or a way to
separate truncation residual from real noise before reading like
coherence.

**Candidate (c) — explained-variance / residual-based (from discussion
2026-08-29: "how much of the total response is explained by the FRF
times the input?").** Instead of the algebraic Gxf/pinv(Gff)/Gxf^H
identity, compute the model's prediction directly, block by block, and
difference it against the actual measured response:
```
Yhat_block(f) = H(f) @ U_block(f)         (per Welch block, any H)
e_block(f)    = Y_block(f) - Yhat_block(f)
coherence_c_i(f) = 1 - mean_blocks|e_block_i(f)|^2 / mean_blocks|Y_block_i(f)|^2
```
Same Welch segmentation for numerator and denominator, so the overall PSD
normalization constant cancels in the ratio — no need to match scipy's
exact scaling. Because `mean|e|^2 >= 0` always, this is **upper-bounded
at 1 by construction for ANY H**, with no optimality assumption required
— unlike candidate (a). It can go slightly negative (model actively worse
than predicting zero at that bin/channel — observed down to about -0.29
at 20% noise), which is a legitimate, informative result (like a
negative-R² regression diagnostic), not numerical breakage; clip to 0 in
practice.

**Empirical result: candidate (c) works.** Plugged in for CVA, its
max never exceeded ~0.998 across the full 0-30% noise sweep (vs.
candidate (a)'s CVA substitution reaching ~1.5+), and its curve tracks
candidate (c) computed for the exact TRUE system closely at every noise
level — both bounded, both showing the same resonance dips and the same
high-frequency rolloff onset around 800-1000 Hz as noise increases. This
is a materially different, better-behaved result than candidate (a): it
directly answers the literal question ("how much of the measured
response does H@u explain") without inheriting the boundedness failure
that afflicts the algebraic substitution approach, CVA-specific or not.
See `examples/sixdrive12resp/results/coherence_analog_investigation.png`
for the full comparison plot (candidates a/b/c across all 6 noise
levels).

**If a coherence-analog is ever built for real, candidate (c) is the one
to use.** Practical notes for that implementation: it needs the actual
raw input/output time blocks (windowed, FFT'd) that CVA's raw-time-tap
would already provide per section 7 item 1 — no separate new data path
beyond what CVA needs anyway. It needs `H(f)` evaluated on the same
frequency grid as the Welch blocks (already available via
`frf_from_ss`). Cost is one extra `einsum` and a residual difference per
update — cheap relative to the fit itself. Still: per the corrected item
3, this is worth doing for GUI/operator trust, not because any control
law's math needs it.

---

## 11. Live validation of items 1/2/4 on sixdrive12resp (2026-08-29)

Confirms the staged-rollout plumbing from section 7 (raw-time-tap, live
CVA wrapper, `Estimator` enum/UI selection) actually works end-to-end on
real hardware-in-the-loop GUI operation, not just offline/synthetic
checks. Driven collaboratively: the user ran the actual Rattlesnake GUI
(Qt-dependent, can't be exercised from this session's headless bridge
VM), I read `Rattlesnake.log`/`gui_debug.log` through the device bridge
to diagnose failures between attempts.

### What was implemented (recap, matches section 7 items 1/2/4)

- **`components/data_collector.py`** — raw-time-tap. `CollectorMetadata`
  gained `raw_tap_enabled`; `DataCollectorProcess` gained
  `raw_data_out_queues`; `acquire()` now copies each acquired frame,
  slices/transforms it into response/reference channels, normalizes by
  `test_level`, and pushes `(raw_response, raw_reference)` onto every
  raw queue right after the frame is pulled off `data_in_queue` — before
  the existing window/rfft path touches it.
- **`components/spectral_processing.py`** — live CVA wrapper.
  `Estimator.CVA_INNOVATIONS` added; `SpectralProcessingProcess` gained
  a raw-data input queue, rolling response/reference buffers, and
  `_run_cva_processing()`, which fits `global_cva_innovations` at
  validated defaults (`lags=40, rank=66, refine_iters=1`), evaluates
  `frf_from_ss` on the live frequency grid, computes the candidate-(c)
  explained-variance coherence-analog from section 10
  (`_cva_explained_variance_coherence`), and emits the same
  `(frames, frequencies, frf, coherence, response_cpsd, reference_cpsd,
  frf_condition)` shape H1 already produces so nothing downstream needed
  to change.
- **`components/random_vibration_sys_id_environment.py` /
  `components/abstract_sysid_environment.py`** — `Estimator` selection
  wiring for both the control-phase and system-ID-phase
  `CollectorMetadata`/`SpectralProcessingMetadata` builders (these are
  two genuinely separate construction paths, see item below), plus
  `system_identification.ui`'s `estimatorComboBox` gaining a "CVA"
  entry.

### Bugs found and fixed via live debugging

1. **`pseudoinverse_control` recommendation was wrong (my mistake, not a
   CVA defect).** Recommended it to isolate CVA testing from
   `optimal_diagonal_control`'s gating dependencies, without checking the
   control-law loading convention first. The loader
   (`initialize_sysid_parameters`) calls `getattr(module,function_name)(...)`
   immediately at parameter-init time with `transfer_function=None`.
   Class-type laws (`control_python_function_type == 2`) — including
   `optimal_diagonal_control`, `octave_band_switching_control`,
   `buzz_control_class` — have a cheap `__init__` that just stores config
   and defer real computation to a separate `.control()` call, so this is
   safe for them. `pseudoinverse_control` is plain-function-style and
   executes its full computation immediately, so it crashed instantly on
   `np.linalg.pinv(None, rcond)`. Fixed by switching to
   `buzz_control_class` instead, verified compatible by reading its
   `__init__` first this time.

2. **Missing `raw_tap_enabled` for the system-ID phase.** The first
   implementation pass only set `raw_tap_enabled` in
   `get_data_collector_metadata()` (`random_vibration_sys_id_environment.py`,
   the CONTROL-phase `CollectorMetadata` builder), missing the entirely
   separate `get_sysid_data_collector_metadata()`
   (`abstract_sysid_environment.py`, the system-ID/buzz-phase builder).
   Symptom: system ID started but the estimation arrow never appeared —
   confirmed via `Rattlesnake.log` showing `CVA buffer filling:
   0/10240 samples` repeating forever with zero progress, since the raw
   tap was never actually enabled during that phase. Fixed by adding the
   same `raw_tap_enabled = self.environment_parameters.sysid_estimator
   == 'CVA'` kwarg to the system-ID builder. Confirmed by grep that these
   are the only two `CollectorMetadata(` construction sites relevant to
   CVA (a third, in `modal_environment.py`, is unrelated/out of scope).

3. **Noise-floor phase hung indefinitely / "Start" stayed grayed out —
   two compounding bugs in `_run_cva_processing`.**
   - On a failed CVA fit — structurally *expected* during the zero-drive
     noise-floor phase, since CVA's Hankel/covariance matrices are
     exactly singular with no persistent excitation on the reference
     channels (unlike H1's `pinv`, which degrades gracefully there
     instead of raising) — the method logged the failure and rescheduled
     without ever calling `self.data_out_queue.put(...)`. Since
     `AbstractSysIDAnalysisProcess.run_sysid_noise()`'s exact-equality
     completion check (`sysid_noise_averages == self.frames`) only
     advances `self.frames` when `flush_queue(self.data_in_queue)`
     returns something, a fit that never emits anything hangs the phase
     forever.
   - Independently, even a successful fit reported `frames` as a raw
     sample count (`self._cva_response_buffer.shape[-1]`, e.g. 10240)
     rather than a value that could ever exactly equal the small phase
     target (`sysid_noise_averages`/`sysid_averages`, e.g. 20) — this
     alone would have caused the identical hang once bug (a) was fixed.

   Symptom: confirmed via `Rattlesnake.log` showing several minutes
   stuck in `RUN_NOISE` at Test Level 0.0 with `CVA fit failed
   (LinAlgError('Singular matrix'))` repeating and `self.frames` never
   advancing off 0.

   Fix: `_run_cva_processing` now always emits on `data_out_queue` once
   the raw buffer is full and the refit-throttle interval has elapsed,
   *regardless* of fit success — using new `_cva_last_frf`/
   `_cva_last_coherence`/`_cva_last_condition` state (updated on
   success, reused as the emitted value on failure, `None` before the
   first-ever success — harmless, since `run_sysid_noise()` doesn't
   consume the frf/coherence payload itself). The diagonal-CPSD display
   computation now always runs too, since it doesn't depend on fit
   success. `frames` is now reported as `params.averages` — the exact
   target value the phase-completion checks in
   `abstract_sysid_data_analysis.py` expect — instead of a raw sample
   count.

### Live results

Two consecutive live system-ID runs with the CVA estimator selected on
`sixdrive12resp`, both completing cleanly: Progress reached "20 of 20"
with zero errors/tracebacks in `Rattlesnake.log`, and the Transfer
Function display showed a real, physically plausible multi-mode
FRF (confirmed by the user directly against the GUI plot: "Noise floor
looked normal, system id ran, we got an FRF display"; the second run was
run specifically "to confirm repeatability" and matched). Both the
fit-fails-gracefully branch (noise-floor phase, zero drive) and the
fit-succeeds branch (system-ID phase, real excitation) were each
independently exercised live, on both runs, without issue.

### Net status after this section

Items 1, 2, and 4 from section 7 are implemented and live-validated on
one bench system. Item 5 (gating/publish logic) is addressed next, in
section 12. Item 3 (coherence-analog) is implemented as part of item 2's
wrapper, using candidate (c) from section 10. Still open, unchanged from
section 7's "not yet validated" list: channel-count-specific
rank/lags scaling beyond the validated 6-drive/12-response case, and
broader real-hardware behavior beyond this one bench system.

---

## 12. Item 5 implementation — gated publish logic (2026-08-30)

Implements section 3's resolution and section 4's Piece 2, now that
Piece 1 (section 11) is live-validated. Inserted directly into
`components/random_vibration_sys_id_data_analysis.py`'s
`RandomVibrationDataAnalysisProcess.run_control()`, at the exact point
identified as the "naive continuous replace" the design doc's section 3
diagnosed: the block that previously unpacked `spectral_data[-1]`
straight into `self.control_frf`/`self.control_coherence`/
`self.control_frf_condition` every single cycle, unconditionally.

### What changed

The incoming tuple now unpacks into local `candidate_frf`/
`candidate_coherence`/`candidate_frf_condition` instead of assigning
straight to `self.control_frf` etc. (`self.frames`,
`self.last_response_cpsd`, and `self.last_drive_cpsd` are still updated
unconditionally every cycle, same as before — abort/warning checks and
the frames-in-cpsd bookkeeping must always see live data; only the
*FRF/coherence/condition* triple is gated).

When `self.parameters.sysid_estimator == 'CVA'` **and** a `control_frf`
has already been published once (i.e. not the first cycle), the
candidate is only adopted — `self.control_frf = candidate_frf` etc. — if
both:

- **well-supported**: `np.nanmedian(np.real(candidate_coherence)) >=
  cva_publish_confidence_threshold` (default `0.5`). If
  `candidate_coherence` is `None` (coherence-analog unavailable or
  disabled that cycle) or its median is non-finite (e.g. an all-NaN
  array), the confidence check is treated as unavailable rather than
  failed — `well_supported = True` — so a missing diagnostic never blocks
  a real update, matching section 7 item 3's "diagnostic only, not a
  blocker" framing and mirroring the fit-failure fallback already added
  to `_run_cva_processing` in section 11.
- **meaningfully different**: whole-array relative Frobenius-norm change
  `‖candidate_frf - self.control_frf‖ / ‖self.control_frf‖` exceeds a
  deadband, `cva_publish_deadband_fraction * frf_update_threshold`
  (defaults `0.5 * 0.05 = 0.025`). `frf_update_threshold` is read live
  off the loaded control law via `getattr(self.control_function,
  'frf_update_threshold', 0.05)` — when the loaded law is
  `optimal_diagonal_control`(_fast), this picks up its actual configured
  value directly (ties the two thresholds together with no duplicated
  parsing, per section 3's "tied to a fraction of
  `optimal_diagonal_control`'s own `frf_update_threshold`"); for any
  other control-law type (generator/plain-function laws, which don't
  have this attribute), it falls back to `optimal_diagonal_control`'s
  own default of `0.05`.

If either check fails, `self.control_frf`/`self.control_coherence`/
`self.control_frf_condition` are left untouched — the same array
objects as the previous cycle, bit-identical, restoring
`optimal_diagonal_control_fast`'s bit-equality fast path between real
publish events exactly as section 3 intended. The very first candidate
after control starts (`self.control_frf is None`) always publishes
unconditionally regardless of confidence/deadband, so control can never
stall waiting on a gate at startup. Non-CVA estimators (`H1`/`H2`/`H3`/
`Hv`) are unaffected — `self.parameters.sysid_estimator` is a dynamic
attribute already present on `RandomVibrationMetadata` (set by
`update_sysid_metadata()` from the System ID tab's estimator combo box,
travels through the IPC pickle unchanged), so this check adds no new
plumbing; when it isn't `'CVA'`, the code takes the same unconditional
`self.control_frf = candidate_frf` path as before the change — their
already-validated incrementally-averaged behavior is untouched.

Two new instance attributes on `RandomVibrationDataAnalysisProcess`
(`cva_publish_confidence_threshold = 0.5`, `cva_publish_deadband_fraction
= 0.5`) hold these defaults. Not yet exposed via xlsx/UI — hardcoded for
now, same status as the CVA fit parameters (`cva_lags`/`cva_rank`/etc.)
noted as an open item in section 5.

### Verification

Not live-tested yet (needs a real CVA control run, not just system ID,
to exercise `run_control()`) — offline-only so far, same constraint as
the rest of this document's implementation work (headless bridge VM
can't import Qt). Verified: `python3 -m py_compile` on the patched file;
a standalone scratch script (`/tmp/cva_work/test_item5_gate.py`,
reimplementing the gate's math verbatim, not committed) exercising the
exact six cases the design called for — bootstrap always publishes;
small jitter around a fixed H at high confidence never publishes
(`control_frf` stays the literal same object through 30 cycles of
jitter well under the deadband); a genuine large step change at high
confidence publishes; the same step change at low confidence does not;
a `None` coherence still gates on magnitude alone (publishes on the big
step, holds on a tiny one); an all-NaN coherence array is treated as
"unavailable" rather than a hard rejection. All six passed.

### Net status

Sections 7's five-item scope list (raw tap, live wrapper, coherence-
analog, enum/UI selection, gating/publish logic) is now fully
implemented and offline-verified. What remains before this is
trustworthy for a real test, per section 7's "not yet validated" list
and this section's own gap: a live control-phase run exercising the
gate against real CVA refits (not just system ID), channel-count-
specific `rank`/`lags` validation beyond the 6-drive/12-response bench
system, and real-hardware (vs. synthetic-model) behavior more broadly.

---

## 13. Live control-run incident and bootstrap fix (2026-08-30)

First live control-phase test of item 5 (`optimal_diagonal_control`,
`update_tf_during_control` on, CVA estimator), run immediately after
section 12's offline-only implementation. System ID (twice, matching
section 11) and the control-prediction preview both looked correct
beforehand. The control run itself did not go well, and exposed a real
gap in the gating design — caught and fixed live, documented here before
further testing.

### What happened

Response error spiked to 100+ dB and stayed ragged/oscillating for
roughly the first 1000 control cycles before gradually decaying back
toward the noise floor over the remainder of the run (user-provided
screenshot of the live GUI plot). No automatic abort fired
(`allow_automatic_aborts` path never triggered — confirmed via
`Rattlesnake.log`, no `Aborting due to channel indices` line). User
stopped the environment manually; output RMS confirmed at zero
afterward.

### Root cause

The gate itself worked exactly as designed: every candidate FRF after
the first was correctly rejected — confidence (candidate-c
explained-variance coherence-analog median) stayed near zero
(~1e-4 to ~1e-6, need >=0.5) and relative Frobenius-norm change against
the published FRF stayed enormous (140-560%) for the entire run,
logged via the `CVA publish gate held:` lines added in section 12.

The bug was upstream of the gate: `run_control()`'s startup path (before
this section's fix) left `self.control_frf = None` and relied on the
gate's own `self.control_frf is not None` bootstrap exception —
*the first live candidate is always published unconditionally, with no
confidence check, specifically so control can't stall waiting on a
gate at startup* (section 12's stated rationale). Given every candidate
after it was low-confidence garbage, the first one almost certainly was
too, and it got a free pass. Nothing after it could ever displace it
(everything was rejected), so control ran the *entire* test off one
unvetted closed-loop CVA fit.

**Why live CVA refits under control were consistently low-confidence
and mutually inconsistent (best available diagnosis, not yet
confirmed against raw data):** system ID's excitation is a dedicated,
persistently-exciting broadband buzz signal — good conditions for a
subspace method, matching the two clean, repeatable fits in section 11.
During control, the "drive" is `optimal_diagonal_control`'s own output,
computed *from* the measured response error — a closed loop, where the
input depends on the very output being measured. Applying an
open-loop-oriented subspace method (CVA, as currently implemented) to
closed-loop data directly is a known failure mode in system
identification: the input is no longer close to white/persistently
exciting the way buzz was, and the fit can become biased or unstable
window-to-window because the plant and the controller's own reaction
aren't cleanly separable in the data. Likely self-reinforcing: the
initial unvetted, poor-gain FRF drove a wildly wrong correction (visible
as the 100+dB ragged error), which made the drive signal for the next
~2s fit window even less stationary/white, degrading the *next* fit too
— consistent with confidence and relative-change never settling for the
whole run rather than converging. Not yet confirmed directly (e.g. by
checking the raw excitation's spectral flatness or the Hankel matrix
condition number during a live window) — inferred from fit timing,
confidence values, and RMS output/error behavior in the log.

### Fix

`run_control()`'s startup block now seeds `self.control_frf` (and
`self.control_coherence`/`self.control_frf_condition`) from the
already-validated `self.sysid_frf` (etc.) when `sysid_estimator ==
'CVA'`, instead of leaving them `None`. This removes the special
unconditional-bootstrap exception entirely: the exact same
well-supported + meaningfully-different gate from section 12 now
governs the very first live candidate too, exactly like every candidate
after it. If `sysid_frf` is somehow unavailable, `control_frf` stays
`None` and the gate's existing fallback reverts to the old
unconditional-publish behavior rather than deadlocking control.

Practical consequence, stated plainly: given CVA's current closed-loop
fit quality, live candidates essentially never clear the confidence
gate, so control now effectively runs off the static `sysid_frf` for
the whole test — the same outcome `update_tf_during_control=False`
would give directly. Until closed-loop fit quality is
understood/improved, recommend actually setting
`update_tf_during_control=False` for CVA runs rather than relying on the
gate to produce that outcome as a side effect.

**Related, pre-existing bug found while diagnosing this (not introduced
by CVA work, not fixed here):** `self.sysid_coherence` — read
throughout `random_vibration_sys_id_data_analysis.py` — is never
actually populated. The inherited
`AbstractSysIDAnalysisProcess.run_sysid_transfer_function()` (in
`abstract_sysid_data_analysis.py`, shared across environments) sets a
differently-named attribute, `self.coherence`, instead. So
`self.sysid_coherence` is always `None` at runtime for
`RandomVibrationDataAnalysisProcess`, for every estimator, not just
CVA — meaning `update_tf_during_control=False` runs already silently
pass `coherence=None` into the control law's baseline. Not fixed in the
shared base class here (risks other environments —
`transient_sys_id_environment.py` legitimately uses its own
`self.coherence`); the new seed reads `getattr(self,'coherence',None)`
instead to get the real value. Worth a real fix separately, scoped to
`RandomVibrationDataAnalysisProcess` specifically rather than the shared
base class.

### Verification

Offline only so far (`/tmp/cva_work/test_item5_bootstrap_fix.py`, not
committed): replayed the old code's behavior against a synthetic
"garbage first candidate" shaped like the incident (confirms it would
be adopted unconditionally, matching what actually happened); replayed
the new seeded behavior against 40 candidates shaped like the observed
confidence/relative-change numbers (confirms `control_frf` stays
bit-identical to the seeded `sysid_frf` throughout, 0/40 published);
confirmed a genuinely good, well-supported first candidate still gets
adopted under the new logic. `python3 -m py_compile` clean on the
patched file.

### Net status

Not yet re-tested live — this fix has not been run against real
hardware yet. Before the next live control run: re-confirm with
`update_tf_during_control=False` first (the now-recommended safe
setting) to validate baseline `optimal_diagonal_control` behavior off a
static CVA-derived FRF without the closed-loop refit question in play
at all; only revisit live-updating control_frf during control once the
closed-loop fit-quality question above has real evidence behind it
(e.g. checking excitation persistence/Hankel conditioning during an
actual control window), not just this section's inference.

---

## 14. Proposal — independent probing/dither injection to enable live CVA updates during control (2026-08-30, not implemented)

Follows directly from section 13's finding: live CVA refits under closed
loop are consistently low-confidence and mutually inconsistent, most
likely because the drive is computed *from* the measured response error
rather than being an independent excitation, which biases/destabilizes
subspace identification (CVA assumes the input is not a function of the
disturbance it's trying to separate from the plant). Section 13's
practical recommendation was to run with `update_tf_during_control=False`
until this is addressed. This section proposes the actual fix. Proposal
only — no code written, nothing tested, not scoped for the next live run.

### The fix: an independent broadband dither added to the drive CPSD

Standard technique in closed-loop system identification: superimpose a
small, broadband perturbation on the drive that is *not* derived from
the response error, so a genuinely exogenous component exists in the
input that the response can be honestly correlated against. This
directly restores the identifiability condition CVA (like most subspace
methods) implicitly assumes — that the input is independent of the
disturbance/noise process — without which no amount of averaging or
longer fit windows fixes the bias.

### Where it fits Rattlesnake's existing architecture

Drives are not synthesized sample-by-sample from the control law
directly. `optimal_diagonal_control` (via `RandomVibrationDataAnalysisProcess.run_control()`)
produces a target `output_cpsd` each cycle; `CPSDSignalGenerator`
(`components/signal_generation.py`) draws a fresh random-phase
time-domain realization from that CPSD every frame
(`update_parameters(cpsd_matrix)` / `generate_frame()`). This is a clean
injection point: add a small, independent broadband term to
`output_cpsd` — e.g. a diagonal (or lightly cross-correlated) floor
CPSD — before it reaches the signal generator, superimposed on whatever
the control law computes for spec tracking. Because
`CPSDSignalGenerator` already draws an independent random-phase
realization from the target CPSD each frame, adding this floor term
requires no new signal-generation machinery — just augmenting the CPSD
matrix that's already being built and queued in `run_control()`
(`self.data_out_queue.put([output_cpsd])`), most naturally right after
the control law returns `output_cpsd` and before it's queued.

### Why channel rotation (probe one drive at a time) is not the right shape for CVA specifically

Considered directly, since it was the original framing of the question.
Rotating which channel receives independent excitation, cycling through
the drive set window to window, does not suit CVA the way it might suit
some other identification techniques: CVA fits one joint multi-input
state-space model per window, and needs persistent, ideally-uncorrelated
excitation on *all* reference channels *simultaneously* within that
window to identify a good MIMO realization (the validated
`rank=66, lags=40` defaults are fitting joint structure across all 6
drives at once, section 7/11). Probing one channel per window while the
others sit closed-loop-only would starve the fit of joint cross-channel
content rather than fix it, unless the fit window were made
substantially longer than a full rotation cycle to accumulate coverage
across all channels — adding real complexity for a technique that
doesn't structurally need it. A small **simultaneous** dither on all
channels, active every window, is the more direct fix for CVA as
currently implemented. (Rotation-style probing is a legitimate technique
for other identification approaches — e.g. sequential/relay methods —
just not this one, with this estimator, as implemented.)

### Open design questions (none decided yet)

- **Amplitude/energy budget.** Needs real headroom analysis against
  spec/warning/abort margins before any number is chosen — not
  guesswork, especially given section 13's incident. Likely candidate
  starting point: scale relative to the already-validated buzz level
  from the system-ID phase, rather than inventing a new amplitude
  convention from scratch.
- **Fixed floor vs. confidence-scaled.** Could dither at a constant
  small level throughout control, or scale it up while CVA's own
  confidence is low (more probing when the model is least trusted) and
  taper it down once a candidate has actually cleared the gate a few
  times. The latter is more elegant but adds a feedback loop of its
  own (dither level depends on confidence, confidence depends on fit
  quality, fit quality depends partly on dither level) that would need
  its own stability reasoning before trusting it live.
- **Diagonal-only vs. structured cross-terms.** A pure diagonal floor is
  simplest and already breaks the closed-loop correlation on each
  channel; whether a specific cross-channel structure in the dither
  CPSD (rather than independent per-channel) would identify faster or
  more accurately hasn't been analyzed.
- **Where exactly `output_cpsd` gets augmented** — inside
  `run_control()` right after the control law call (estimator-agnostic,
  same place item 5's gate reads `self.control_function`), or as a
  parameter the CVA-aware path of `optimal_diagonal_control` itself
  adds internally. The former keeps it decoupled from any specific
  control law (consistent with Piece 1/Piece 2's estimator-agnostic
  seam in section 4); the latter could let the control law reason about
  dither vs. spec-tracking tradeoffs directly. Leaning toward the
  former for the same separation-of-concerns reasons as the gate itself,
  but not decided.
- **Interaction with the section 13 fix.** Even with dither restoring
  identifiability, the section 13 gate (confidence + deadband) should
  stay in place regardless — dither makes good candidates *possible*
  again, it doesn't make every candidate trustworthy on its own.

### Net status

Not implemented, not scheduled. The recommended safe default from
section 13 (`update_tf_during_control=False` for CVA) stands until this
is designed further and validated — first offline (synthetic closed-loop
simulation with and without dither, checking whether CVA's fit quality
and confidence actually recover), then live, following the same
staged-rollout pattern as the rest of this document.

---

## 15. Live control-run finding — CVA-derived FRF accuracy gap, not just FRF-update instability (2026-08-30/31)

Second live control-phase test, this time with `update_tf_during_control=False`
(the safe baseline recommended in section 13) — `optimal_diagonal_control`
running off the static, already-validated CVA `sysid_frf` with no live
refitting in the loop at all. Confirms section 13's fix works as intended
(no ragged 100+ dB instability like the first run) but surfaces a
different, more fundamental problem: the CVA-derived FRF itself does not
appear accurate enough to drive `optimal_diagonal_control` to the same
tracking performance H1 achieves, independent of any live-update
question.

### What the run showed

No repeat of section 13's incident — confirmed via
`optimal_diagonal_control`'s own diagnostic log
(`H_changed_since_last=False` on every single `_refine_batch` call,
hundreds of consecutive calls) that the transfer function was
bit-identical throughout, exactly as `update_tf_during_control=False`
should give. No automatic abort, drive RMS settled to a low, stable
range (~3-4 per channel) within a few minutes and stayed there.

But tracking error stayed high and stable — user-reported live GUI
panel: **7.66, 5.96, 6.87, 5.84, 7.50, 7.57, 12.39, 9.96 dB** across the
8 response channels, essentially unchanged for several minutes.
Refinement had stalled: `n_refined_total=901/2561` bins, `n_deferred=0`,
`newly_refined=0` for 200+ consecutive calls — the control law's own
Step 2 logic (refine worst-error not-yet-refined bins) found nothing
left exceeding its `error_threshold_db=1.0` threshold, i.e. it believes
it is already done.

### Root cause — the control law's own model doesn't match reality on most channels

`optimal_diagonal_control`'s log also reports its own internal
model-based error prediction each call
(`self_predicted_rms_db_per_channel`, computed by simulating what
response its own H model + solved drive CPSD *should* produce):
`[5.21, 0.77, 1.66, 0.92, 0.43, 0.25, 11.58, 9.58]`. Compared directly
against the actual measured error above:

| Channel | Self-predicted (dB) | Actual measured (dB) |
|---|---|---|
| 1 | 5.21 | 7.66 |
| 2 | 0.77 | 5.96 |
| 3 | 1.66 | 6.87 |
| 4 | 0.92 | 5.84 |
| 5 | 0.43 | 7.50 |
| 6 | 0.25 | 7.57 |
| 7 | 11.58 | 12.39 |
| 8 | 9.58 | 9.96 |

Channels 7 and 8 are consistent — the model correctly predicts they're
hard, and they are (likely a genuine actuation limit, e.g. the
`max_drive_coherence=0.95` cap biting on a direction those channels
need; a separate, model-independent issue). Channels 1-6 are not
consistent — the model believes it has already refined those to under
2 dB (several under 1 dB), while actual measured error sits at 5.8-7.6
dB. That mismatch is why refinement stalled: the SDP solve is
correctly optimal *with respect to the model it was given*, and that
model — the static CVA-derived `sysid_frf` — does not match the real
plant closely enough on 5 of 8 channels for the resulting "optimal"
drive to actually work.

**This reframes the whole CVA-integration effort's current status.**
Section 11's two system-ID runs confirmed the CVA fit *looks* right —
correct general shape, resonances in sane places, repeatable between
runs. This run is effectively the first real test of a stricter bar —
*accurate enough to drive a model-based inverse-design control law* —
and it fails that bar on most channels, even with the FRF held
perfectly static (ruling out every live-refit/gating concern from
sections 12/13 as the cause here). "Physically plausible" and
"accurate enough for `optimal_diagonal_control`" are evidently different
standards, and only the first has actually been validated so far.

### Recommended next step

Directly compare the CVA-derived `sysid_frf` against an H1-derived FRF
computed from the same hardware/conditions, bin by bin (magnitude and
phase, not just visual shape), rather than continuing to infer accuracy
indirectly from downstream control performance. This is the natural,
overdue accuracy check section 7's "not yet validated" list gestured at
but didn't specifically call for. Until that comparison exists and CVA's
accuracy gap against H1 is quantified (and, ideally, closed), CVA-driven
`optimal_diagonal_control` should be expected to underperform H1's,
independent of the live-update question sections 13/14 address — those
sections' fixes are still correct and necessary, they just aren't
sufficient on their own to match H1's control quality.

### Net status

Run left active at the time of writing (stable, bounded, not unsafe —
just underperforming); no automatic abort, user's call whether to stop
it. Item 5 (gating) and the section 13 bootstrap fix are validated as
working correctly for their stated purpose (no live-refit instability).
The open question this section adds is upstream of both: CVA fit
*accuracy*, not update cadence or gating — worth its own investigation
before further live control testing.

## 16. Root cause found — CVA needs broadband excitation, independent of the closed-loop and noise questions (2026-09-01)

Continuation of section 15's open question ("why does CVA-driven control
underperform H1's, given the FRF was held perfectly static"). User's framing
going in: suspicion of relative inter-channel phase accuracy, plus a specific
memory that Wayne Larimore's own CVA work was used for control, which
argued against writing the whole approach off.

### 16.1 Literature check — closed-loop CVA is real, and this repo's implementation doesn't have the correction

Searched for what Larimore actually published on closed-loop subspace
identification (not just recalled from training): "Maximum likelihood
subspace identification for linear, nonlinear, and closed-loop systems"
and "Closed-loop subspace identification with innovation estimation"
(Larimore), plus the ADAPTx software built around this line of work, and a
2006 survey (Qin, "An overview of subspace identification") that lays out
the mechanism precisely.

The core closed-loop subspace-ID equation relates a block of future outputs
to a past-driven term, a future-input term (`Hf @ Uf`), and a noise term.
Under feedback, `Uf` and the noise term become correlated, which biases the
naive fit. Larimore's fix (and Jansson's SSARX, Shi & MacGregor's method,
same family): pre-estimate `Hf` from a high-order ARX model fit to the
closed-loop data, subtract its predicted contribution from the future-output
block, THEN do the canonical-correlation projection on what's left. This is
an algorithmic correction, not an excitation trick — no dither required.

Checked `globalcva/global_cva_frf.py`'s `global_cva_v2` against this
specifically: the whitening (`Wp`, `Wf` via `_whiten(Spp/Sff, tol)`) is
genuine canonical-correlation weighting, so this is real CVA, not a
disguised N4SID. But the future block is built as `F = Yh[L:L+N]` — raw
future outputs, no ARX pre-estimate, no subtraction of the future-input
effect. That is exactly the "naive" formulation the literature says is
biased under feedback. This gives a mechanistic (not just inferred-from-logs)
explanation for section 13's live-update incident: near-zero-confidence,
wildly-inconsistent refits during control are what the literature predicts
for this specific formulation under feedback.

Important scope limit: this explains section 13 (fitting DURING control,
closed loop). It does NOT explain section 15 — that run's `sysid_frf` was
fit entirely during the open-loop sys-ID phase, before control started, with
`update_tf_during_control=False`. Feedback-induced bias cannot be the cause
of section 15's accuracy gap. That gap needed its own explanation — see 16.3.

### 16.2 Raw-tap pipeline check — no implementation bug found

Before concluding the phase-accuracy hypothesis was structural rather than a
bug, checked `data_collector.py::acquire()` directly for a channel-order,
timing, or transformation-matrix mismatch between the CVA raw tap and the
H1 windowed-FFT path. Both are sourced from the exact same
`acquisition_data` block in the same call (`raw = np.copy(acquisition_data)`
happens first, before any frame chunking), and both apply identical
`response_channel_indices`/`reference_channel_indices`, identical
`response_transformation_matrix`/`reference_transformation_matrix`, and
identical `test_level` normalization. No decimation stage was found between
the ADC read and either path. `global_cva_frf.py::frf_from_ss`'s
discrete-to-frequency mapping (`z = exp(j*2*pi*f*dt)`, `H = C(zI-A)^-1B+D`)
is the standard textbook formula. Conclusion: nothing in the plumbing
explains a channel-specific phase bug; whatever's happening is a property of
the fit itself, not the acquisition path.

### 16.3 Direct CVA-vs-H1-vs-ground-truth comparison — the actual finding

Built `globalcva/cva_vs_h1_phase_comparison.py`: reproduces the exact live
configuration as closely as possible — `global_cva_innovations` (the
function `_run_cva_processing` actually calls live, not the plain
`global_cva_v2` the original validation table used), `lags=40, rank=66,
refine_iters=1`, `cva_window_seconds=2.0` (all live defaults), on the same
noise-free linear ground-truth system (`sdynpy_frame6x12_system.npz` —
confirmed by the user as the system actually loaded for the live runs).
Computes CVA and H1 FRFs against the known ground truth, broken down per
response channel, magnitude and phase separately, across 3 seeds.

First pass used full-bandwidth (0-2560 Hz) white noise excitation, matching
`validate_global_cva.py`'s original methodology. Result: CVA beat H1 on
every one of the 8 response channels, in both magnitude and phase (mean
per-channel magnitude error ~0.6-0.9 dB for CVA vs ~0.8-1.0 dB for H1; mean
phase error ~2-6 deg for CVA vs ~4-6 deg for H1; overall relative error
CVA~0.07-0.08 vs H1~0.13, consistent with the header docstring's original
table). The plot showed CVA tracking truth cleanly through sharp
anti-resonance notches where H1's CSD-ratio estimate spikes badly (20+ dB,
100+ deg errors) because coherence collapses there. Nothing in this
reproduction resembled the live run's "5 of 8 channels off by 5-7 dB"
pattern. This ruled out "CVA is just less accurate than H1 in general, even
open-loop, even noise-free" as the explanation — if anything the idealized
case says the opposite, matching the original validation.

Checked the actual live sys-ID/control excitation spec next:
`examples/sixdrive12resp/code/build_flat_spec_large.py` /
`results/flat_spec_frame6x12.mat`, loaded via
`random_vibration_sys_id_utilities.load_specification`. Confirmed
(`scipy.io.loadmat`, checked the diagonal directly): flat 0.001 g^2/Hz on
each of the 8 control channels, but strictly **band-limited to 100-1000
Hz, zero everywhere else**. This is not what the first offline pass used —
that used full-bandwidth white noise. Reran the identical comparison
(`globalcva/cva_vs_h1_bandlimited.py`) with the excitation changed to match
this exactly: flat magnitude in [100, 1000] Hz, zero outside, random phase,
same amplitude scale, everything else (system, lags, rank, window, seeds)
unchanged.

CVA collapsed:

| Metric (mean across 8 channels) | CVA, full-BW excitation | CVA, 100-1000 Hz excitation | H1, 100-1000 Hz excitation |
|---|---|---|---|
| Mean magnitude error | ~0.6-0.9 dB | ~5.6-6.8 dB | ~0.8-1.0 dB |
| Mean phase error | ~2-6 deg | ~50-59 deg | ~4-6 deg |
| Overall relative error | 0.07-0.08 | 1.02-1.35 | 0.12 |

H1, run on the exact same band-limited data, was essentially unaffected
(relative error ~0.12 either way — expected, since H1's target band already
matched the excited band). CVA's relative error exceeds 1.0 — worse than
predicting zero. The plot
(`globalcva/cva_vs_h1_bandlimited.png`) shows the mechanism clearly: H1
still tracks every resonance and anti-resonance sharply; CVA smooths the
whole shape out and drifts systematically low above ~600 Hz — the signature
of an under-identified subspace model, not an isolated notch artifact.

Checked whether this is a hard cliff or a graded effect: reran with the
band widened to 20-2000 Hz (`globalcva/cva_vs_h1_bandlimited_wide.py`,
still not full 0-2560 Hz). CVA partially recovered — overall relative error
~0.64-0.65 — better than the 100-1000 Hz case but still far short of both
its own full-bandwidth performance and H1's performance at the same width.
Confirms a genuine, graded, dose-response relationship to excitation
bandwidth, not a one-off numerical artifact of one band edge or one seed.

### 16.4 Mechanism and why this reconciles the user's memory of Larimore

H1 estimates each frequency bin independently from local coherent
energy at that bin — restricting excitation to the band being evaluated
costs it nothing. CVA instead builds one joint state-space realization from
a Hankel embedding across `lags=40` samples of delay, and the
canonical-correlation whitening that makes it "CVA" needs the input to be
persistently exciting across the system's full dynamic order — not just the
band the resulting model will be evaluated over — because the past/future
covariance structure being whitened integrates across the whole embedded
lag window regardless of where you plan to use the model afterward. Starve
it of energy outside the target band and that covariance structure becomes
too poorly conditioned to correctly separate a rank=66 realization, even
when scored only on the band that WAS excited. This is a known
persistency-of-excitation result in subspace-ID theory, not specific to
this codebase.

This reconciles the user's recollection better than the closed-loop story
did: standard subspace-ID practice (Larimore's own included) almost
certainly used genuinely broadband identification excitation — PRBS,
multisine, or wideband noise spanning well past the eventual control band
— even when the control band of interest was narrower, because that's what
"persistently exciting of sufficient order" requires for a subspace method.
This Rattlesnake sys-ID phase instead reused the exact same 100-1000 Hz
control-target spec as the identification excitation itself — a completely
reasonable, standard choice for H1 (which is what H1-based FRF estimation
is built for) and precisely the choice that breaks CVA.

### 16.5 Relationship to sections 13/14/15

Three independent, non-overlapping explanations are now on the table for
why live CVA has underperformed H1, and it's worth keeping them separate:

- **Closed-loop bias** (section 13): applies only when CVA is fit DURING
  control (feedback active). Mechanistically explained in 16.1; the known
  literature fix is algorithmic (ARX pre-estimate + subtract), not dither.
  Section 14's dither proposal is a plausible but unconfirmed alternative
  mitigation for this same case — not yet needed if the ARX-style fix is
  pursued instead.
- **Additive sensor noise sensitivity** (`noise_injection_sweep.py`,
  referenced in section 15): applies only when real measurement noise is
  present at realistic levels; irrelevant to the noise-free simulated
  system used in every live run so far.
- **Excitation bandwidth** (this section, 16.3-16.4): applies any time
  CVA's identification excitation is narrower than its embedding needs,
  open-loop or closed-loop, noise-free or not. This is the one that
  actually explains section 15's run — pure open-loop, noise-free, static
  FRF — on its own, without needing either of the other two mechanisms.

### Net status

Root cause of section 15's accuracy gap identified and reproduced offline:
the sys-ID phase's excitation spec (100-1000 Hz flat, matching the eventual
control target) is far too narrow for CVA's identification requirements,
even though it is exactly the right spec for H1. Not yet mapped: precisely
how wide the sys-ID excitation needs to be for this system/lags/rank
combination to reach H1-competitive accuracy (100-1000 fails badly, 20-2000
partially recovers, full 0-2560 fully recovers — the crossover point isn't
pinned down yet). Not yet designed or implemented: decoupling the sys-ID
phase's excitation bandwidth from the control phase's target spec (drive a
wide broadband probe for identification, keep the narrower 100-1000 Hz spec
for the actual control target). No code changes made this session — offline
investigation and scripts only. Scripts and plots saved under
`globalcva/`: `cva_vs_h1_phase_comparison.py`(`.png`),
`cva_vs_h1_bandlimited.py`(`.png`), `cva_vs_h1_bandlimited_wide.py`.

### 16.6 Correction (2026-09-02) — the excitation-bandwidth root cause in 16.3/16.4 is wrong

While investigating how to inject an offline-computed broadband CVA FRF into
a live control test, re-read `components/abstract_sysid_environment.py`'s
`get_sysid_signal_generator()` and found it constructs the sys-ID drive's
`RandomSignalGenerator` with `low_frequency_cutoff=None,
high_frequency_cutoff=None` — hardcoded, not read from the GUI or from the
Specification File. Per `RandomSignalGenerator.__init__`, `None` means
0-to-Nyquist: the live sys-ID excitation is full-bandwidth (0-2560 Hz) by
construction, regardless of `flat_spec_frame6x12.mat`'s 100-1000 Hz band.
That spec file is used for control targeting, not for shaping sys-ID
excitation. `random_vibration_sys_id_environment.py` does not override this
method.

This means 16.3/16.4's headline finding — "the live sys-ID excitation
matches the narrow control-target spec, and that's why CVA collapsed" — does
not describe what actually happened in the live runs. The band-limited-
excitation experiment itself was real and reproducible (and remains a
correct, useful characterization of CVA's general sensitivity to excitation
bandwidth — worth keeping in mind for any future narrowband sys-ID
configuration), but it is not section 15's root cause, since that run's
excitation was already broadband. Section 15's accuracy gap is open again.

Remaining candidate, not yet confirmed: `RandomSignalGenerator.generate_frame()`
builds each frame as an independent Gaussian block, COLA-blended (50%
overlap, Hann window) with the previous block — structurally different from
the single continuous i.i.d. realization the offline validation scripts use.
That changes sample-to-sample correlation structure without necessarily
changing the power spectrum, which H1 (frequency-domain, per-bin) wouldn't
notice but a raw-sample Hankel/subspace method like CVA might. Not tested.

Added a small, additive, off-by-default-pattern capture hook to
`spectral_processing.py::_run_cva_processing` (`CVA_CAPTURE_RAW_DATA` flag,
currently set `True` for this investigation) that writes the raw sys-ID
response/reference buffers and the resulting FRF (plus A/B/C/D and fit
parameters) to `examples/sixdrive12resp/results/cva_captures/
latest_cva_sysid_capture.npz` on every successful live CVA fit, overwriting
so the file left behind after sys-ID completes holds the last (most-
averaged) fit — i.e. exactly the FRF about to be handed to control. Plan:
run live CVA sys-ID once more (same setup as section 11/15: CVA estimator,
`update_tf_during_control=False`), verify the captured FRF against ground
truth offline (same bin-by-bin magnitude+phase method as section 16.3)
*before* starting control, then proceed to the control-phase test with
whichever FRF verifies as accurate.


### 17. Live capture verified against ground truth (2026-09-02) — still no root cause, but three more candidates eliminated

Ran the live CVA sys-ID pass planned in 16.6 (CVA estimator, `update_tf_during_control=False`,
24 averages, `flat_spec_frame6x12.mat`, sdynpy frame 6x12 plant). The capture hook wrote
`examples/sixdrive12resp/results/cva_captures/latest_cva_sysid_capture.npz`: the final
(most-averaged) fit's raw `response_buffer`/`reference_buffer` (8x10240 / 6x10240, i.e. the
live 2.0s sliding window at fs=5120), the resulting `frf`, `A/B/C/D`, and the fit parameters
(lags=40, rank=66, refine_iters=1 — confirmed matching live defaults exactly).

### 17.1 No live-wrapper implementation bug

Recomputed `global_cva_innovations`+`frf_from_ss` offline on the *identical* captured raw
buffers. First attempt disagreed with the live `frf` (relative error 0.15 between the two) —
traced to `spectral_processing.py::_run_cva_processing` passing `tol=1e-10` to
`global_cva_innovations`, not the function's default `tol=1e-8`. Matching `tol=1e-10` in the
offline recompute made live and offline agree to relative error 0.0000 (exact). So the live
wrapper reproduces the standalone, already-validated algorithm exactly on real captured data —
whatever is happening is a property of the data/fit itself, not a bug in `spectral_processing.py`.

### 17.2 The live captured FRF's actual accuracy

Live captured FRF vs ground truth (100-1000 Hz band, where ground truth is defined):
mean magnitude error 2.02 dB, mean phase error 15.36 deg, overall relative error 0.2854.

For scale: the idealized offline validation (continuous i.i.d. excitation, same lags/rank/
refine_iters/window, same plant) gets relative error ~0.07-0.08 (section 16.3). The (now-
retracted-as-root-cause, but still real) band-limited-excitation collapse gets relative error
1.0-1.35 (section 16.3). The live capture sits well above the idealized case and well below the
pathological one — a real, substantial, but not catastrophic gap.

### 17.3 Raw buffer sanity — normal

Checked `response_buffer`/`reference_buffer` directly: RMS per channel uniform and sane
(u ~0.01, y ~0.008-0.016), near-zero DC offset, no clipping (max|u|~0.04, max|y|~0.07, well
under any obvious saturation), Welch PSD of the reference channels flat across 50-2000 Hz as
expected for full-bandwidth white noise. Nothing here suggests corrupted or gapped data.

### 17.4 Candidate: COLA block-generation structure — tested directly, REJECTED

Section 16.6 flagged `RandomSignalGenerator.generate_frame()`'s COLA overlap-add structure
(independent per-frame Gaussian blocks, 50%-overlap Hann-windowed blend with the previous
block) as a plausible remaining cause, since it's structurally different from the continuous
i.i.d. realization the offline validation scripts use. Tested directly: instantiated the actual
`RandomSignalGenerator` class (from `components/signal_generation.py`, same params as live:
`cola_overlap=0.5, cola_window='hann', cola_exponent=0.5, low/high_frequency_cutoff=None`),
drove the same ground-truth linear system through its output, and refit CVA with identical
lags/rank/refine_iters/window. Result (3 seeds): relative error 0.067-0.072 — indistinguishable
from continuous i.i.d. (0.072-0.083 on the same seeds/system). COLA framing alone does not
degrade CVA. This candidate is ruled out.
(`globalcva/cola_excitation_test.py`)

### 17.5 Candidate: fixed u/y timing misalignment — tested, REJECTED as the live explanation, but surfaces a real CVA property worth documenting

Hypothesis: some sub-sample or integer-sample delay between the reference (drive command) and
response (measured) channels in the real acquisition path — common in real DAQ hardware and
plausible in a simulated stand-in for it — could explain the gap, since CVA's fixed-order
state-space fit has no way to represent a delay except by consuming extra states, unlike H1
which represents a pure delay exactly as a linear phase term.

Confirmed CVA is pathologically sensitive to this in general: on clean synthetic data (rel err
0.07 at zero delay), an artificial delay of just 1 sample (0.2 ms) pushes relative error to
0.68; a *fractional* delay of only 0.05 samples (~10 microseconds, via FFT-phase-shift) already
pushes it to ~0.6. This is a genuine, novel, and useful characterization of CVA's fragility to
any inter-channel timing skew — H1 would barely notice a delay this small. (`globalcva/
cva_vs_h1_phase_comparison.py`-style delay sweep, ad hoc script, not saved separately.)

However: searching a wide range of delay corrections (-1.0 to +1.0 samples, applied to the
*live captured* `response_buffer` before refitting) does NOT reproduce this signature and does
NOT materially improve the live fit — best found was relative error 0.239 at +0.10 samples
(modest improvement from the 0.2854 baseline at zero correction), nothing close to recovering
the ~0.07 idealized accuracy the way correcting a real fixed delay should. If a clean fixed
misalignment were the cause, undoing it should collapse the error back toward baseline the same
way the synthetic delay sweep shows a *wrong* delay blowing it up — that recovery isn't there.
Fixed timing misalignment is ruled out as the explanation for the live gap, though the general
delay-sensitivity finding stands on its own and is worth keeping in mind for any real-hardware
deployment.

### 17.6 Net status

Eliminated so far, cumulative across sections 13-17: closed-loop bias (open-loop run), sensor
noise (noise-free sim), excitation bandwidth (confirmed full-bandwidth by construction), COLA
block-generation structure (tested directly), fixed/fractional u-y timing misalignment (tested
directly), and any implementation bug in the live wrapper (live == offline recompute exactly).

Remaining open question: what specifically separates Rattlesnake's actual running simulated
plant (whatever produced these captured buffers) from a clean `scipy.signal.lsim` continuous-
time simulation of the same system, such that CVA's fit lands at relative error ~0.285 instead
of ~0.07-0.08. Not yet tested: the exact discretization/integration method used by Rattlesnake's
real-time (or simulated real-time) engine (ZOH vs impulse-invariant vs continuous-time
integration, output oversampling effects, per-frame boundary handling in how raw samples are
queued in `_run_cva_processing` rolling buffer). No further offline reproduction attempt
found a match this session.

### 17.7 Recommendation for tonight's control-phase test

The live captured FRF does not meet the accuracy bar this investigation established for
"accurate broadband CVA" (idealized case: <1 dB mag, ~2-6 deg phase, relative error ~0.07-0.08).
It has real, substantial error (~2 dB mag, ~15 deg phase, relative error ~0.285) — plausibly
still large enough to reproduce section 15's poor control tracking. Because the offline
recompute on the identical raw data gives the identical (poor) answer, there is no
"offline-computed but more accurate" version of this particular FRF to substitute — the raw
data itself, not the fitting algorithm, is the limiting factor here, and its root cause is not
yet identified. Recommend against proceeding to the control-phase test with this FRF; either
fall back to H1 for tonight's comparison (demonstrably far more robust throughout this whole
investigation), or pause here and continue investigating the plant-simulation-vs-offline-model
gap before attempting CVA-driven control again.

Scripts added this session (in `globalcva/`): `verify_live_capture.py` (sections 17.1-17.2),
`cola_excitation_test.py` (17.4). The delay-sensitivity and live-capture delay-search checks
(17.5) were run ad hoc and not saved as standalone scripts.
