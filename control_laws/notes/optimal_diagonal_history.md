# Optimal diagonal laws: design history and run evidence

What the code comments and docstrings of `optimal_diagonal_control.py`,
`optimal_diagonal_control_fast.py` and `octave_band_switching_control.py` used to
carry besides the code's own description: when something changed and why, which
run showed it, and the numbers measured then.  It was moved here, **verbatim**, so the
code can say what it does now, and why that is right, without the story of how it got there.

Each entry names where the text came from.  Run numbers refer to the rig runs in
`examples/sixdrive12resp/results`.  Where the original claims something the code no
longer does, the entry says so.

In the code, a comment ends with `notes/optimal_diagonal_history.md#<anchor>` when
there is an entry here for it.


## Index

- [Live-FRF safety limit (2026-09-19, run 30)](#frf-safety)
- [Deadlock escape (run 30)](#deadlock)
- [Statistics over the FRF must be masked to in-band bins (runs 21, 28 and 31)](#in-band)
- [Trim freeze after a level change (3 to 25 on 2026-09-25, run 52)](#trim-freeze)
- [Trim leak and the gate-failure counter (run 52; verify_response_trim.py TEST 2)](#trim-leak)
- [The trim's own gate threshold (run 35)](#trim-gate)
- [What kind of loop this is: the mislabelling, and the evidence (runs 29, 31, 32)](#loop-kind)
- [The default of frf_update_threshold: 0.5, raised from 0.05 (2026-09-19, runs 29 and 31)](#frf-update-threshold)
- [Reasons for other defaults (error_domain, drive_rcond, the trim)](#defaults)
- [optimal_diagonal_control_fast: does the unconstrained solve reach coherence 1.0? (run 13, retracted 2026-09-17)](#fast-coherence-claim)
- [optimal_diagonal_control_fast: the safety gate was wrong, then fixed (2026-09-17)](#fast-safety-rule)
- [optimal_diagonal_control_fast: what the FRF-movement gate measures (runs 21, 28, 29; 2026-10-25)](#fast-h-moved)
- [optimal_diagonal_control_fast: state is set before super().__init__() (2026-09-17)](#fast-init-order)
- [optimal_diagonal_control_fast: why the dB objective exists (run 14)](#fast-db-objective)
- [optimal_diagonal_control_fast: warm start (fix 5, 2026-10-25)](#fast-warm-start)
- [optimal_diagonal_control_fast: L-BFGS tolerances on real FRFs](#fast-tolerances)
- [optimal_diagonal_control_fast: the drive-subspace restriction (runs 14, 22-25)](#fast-drive-rcond)
- [optimal_diagonal_control_fast: per-line singular-vector selection (2026-10-24)](#fast-svd-floor)
- [optimal_diagonal_control_fast: the dB-objective coherence guard (fix 6, 2026-10-25)](#fast-db-guard)
- [Smaller dated remarks](#small)

<a id="frf-safety"></a>

## Live-FRF safety limit (2026-09-19, run 30)

Run 30 is the one where the live H1 estimate walked away from the system identification and the law solved against it.

*From* `optimal_diagonal_control.py, the comment above _FRF_DIVERGENCE_LIMIT`:

```text
---------------------------------------------------------------------------
LIVE-FRF SAFETY LIMITS.  Added 2026-09-19 after run 30, where the live H1
estimate walked 22,000x away from the system identification (reported by
this class's own since_initial diagnostic as median 39834, max 500217) and
the law solved against it, commanding 1.8e5 V.  A healthy run sits well
under 1: run 28 logged since_initial median 0.1752, max 0.8920 over its
whole length.  10.0 therefore has an order of magnitude of headroom over
anything ever measured on a working run and still catches run 30 at its
55th call.
```

*From* `optimal_diagonal_control.py, _reject_diverged_frf docstring, second paragraph (from 'Run 30 closed that loop')`:

```text
solve feeds a bad estimate which feeds a worse solve.  Run 30 closed
that loop: since_initial reached median 39834 and max 500217 -- the
live H was four to five orders of magnitude away from the system
identification -- and the law dutifully solved against it and asked
for 1.8e5 V.
```

<a id="deadlock"></a>

## Deadlock escape (run 30)

*From* `optimal_diagonal_control.py, the comment above _DEADLOCK_ESCAPE_DB`:

```text
DEADLOCK ESCAPE.  Run 30's end state: corrupted H -> near-zero commanded
drive (0.0015 V) -> no excitation -> the live FRF stops moving -> nothing
exceeds frf_update_threshold -> no bins scheduled -> the law holds a dead
solution forever.  70+ consecutive calls with every scheduler field zero
and a bit-identical self-prediction.  It cannot recover on its own, because
recovering requires excitation the law is no longer commanding.  These
thresholds only fire in that absorbing state: 20 dB below the target trace
is far outside anything a working run reaches (runs 19-29 all land within
+/-3 dB), and 10 consecutive calls rules out a transient.
```

*From* `optimal_diagonal_control.py, _escape_deadlock_if_starved docstring, first line of the second paragraph`:

```text
The failure this exists for (run 30, calls 109-182): the commanded
```

*From* `optimal_diagonal_control.py, _escape_deadlock_if_starved docstring, third and fourth lines of that paragraph`:

```text
dead solution.  Every scheduler field zero and a bit-identical
self-prediction for seventy consecutive calls.  Nothing in the loop
```

<a id="in-band"></a>

## Statistics over the FRF must be masked to in-band bins (runs 21, 28 and 31)

This lesson was learned three or four times, in `_effective_drive_rcond`, `_h_moved`, the divergence guard and the trim.

*From* `optimal_diagonal_control.py, _reject_diverged_frf, in-band comment`:

```text
IN-BAND ONLY, and this was got wrong the first time.  Run 31 fired
this guard on 986-1161 bins of 2049 on essentially every call while
controlling perfectly well (converged 2.13 V, 17.70 V^2, and the
refined-bin drift never left median 0.07).  2049 - 901 in-band =
1148 out-of-band bins: no drive energy there, so the live estimate
is noise and its relative change is unbounded.  The guard was
reporting noise.  Harmless -- those bins are not controlled -- but
it buried the log and made the threshold untestable.

THIRD TIME this same mistake has been made in this file and its
subclass: the singular-value spread in _effective_drive_rcond, the
movement metric in _h_moved, and now this.  Any statistic taken
over the FRF array on this rig must be masked to in-band bins.
```

*From* `optimal_diagonal_control.py, _update_response_trim docstring, 'IN-BAND ONLY'`:

```text
diagonal is noise, and the ratio is unbounded.  This is the fourth
statistic in this file that has had to learn that lesson.
```

*From* `optimal_diagonal_control_fast.py, _h_moved, 'IN-BAND ONLY' comment`:

```text
IN-BAND ONLY.  Measured over the whole array this metric is dominated
by the out-of-band bins, where there is no excitation, the live
estimate is noise, and the relative change per cycle is enormous.
Run 28 measured it: median 4.37 -- a 437% change every cycle, max
179 -- while the base class's own drift metric, which looks only at
refined (in-band) bins, read 0.17 on the same data.  A factor of 25
between two metrics of the same quantity.  That is why runs 21 and 28
demoted on 99% of cycles: the gate was responding to out-of-band
noise, never to the plant.  Same mistake as measuring the
singular-value spread over all lines -- see _effective_drive_rcond.
```

<a id="trim-freeze"></a>

## Trim freeze after a level change (3 to 25 on 2026-09-25, run 52)

*From* `optimal_diagonal_control.py, the comment above _TRIM_FREEZE_CALLS`:

```text
RESPONSE-ERROR TRIM FREEZE.  components/data_collector.py divides every
acquired frame by the CURRENT test level, so everything this law sees --
FRF, last_response_cpsd, last_drive_cpsd -- is referred to FULL level and
the trim ratio is level-consistent in steady state.  It is NOT consistent
across a level change: a frame that spans the ramp is normalized by the
wrong number.  Rattlesnake calls set_test_level_db on every change, so the
trim sits out this many control cycles afterwards.

RAISED FROM 3 TO 25, 2026-09-25, after run 52.  Three was sized for a single
level change.  A level RAMP is many changes -- run 52 logged 17 of them
getting from 0 dB to -18 dB -- and the freeze expires between steps, so the
trim spent the whole ramp acting on transient error.  25 matches the ~25
control-cycle time constant of the CPSD exponential average (coefficient
0.04), which is how long a measured frame takes to stop carrying the old
level.  Each change re-arms the counter, so a ramp is ONE long freeze and
the trim resumes only once the level has actually been still that long.
```

<a id="trim-leak"></a>

## Trim leak and the gate-failure counter (run 52; verify_response_trim.py TEST 2)

The first excerpt (the comment above `_TRIM_LEAK_DB`) says that bins inside the deadband also decay.  That was corrected in the same session: only bins that failed the solver-agreement gate leak, as the last excerpt explains and the code does.

*From* `optimal_diagonal_control.py, the comment above _TRIM_LEAK_DB`:

```text
RESPONSE-ERROR TRIM LEAK.  Per cycle, in-band bins that do NOT act -- gated
out by the solver-agreement test, or inside the deadband -- decay toward
zero by this much instead of holding.

WHY, and this is the defect run 52 exposed.  The gate is evaluated BEFORE
the error term, and gated bins previously got step_db = 0, i.e. they kept
whatever correction they already carried, permanently.  That blocks harmful
accumulation and corrective UNWINDING equally: a bin that picks up a
spurious trim while the loop is still converging, and whose solver residual
then exceeds the gate, is frozen at that value for the rest of the run.
Run 52 ended with the trim spread across the full +/-3 dB clamp, median
-1.51 dB, after 261 updates, on a plant whose true model gap at that level
is 0.012 dB -- and it cost 30% more drive (11.47 V rms vs 8.84 V) to
deliver the same response as trim-off.  Integrator windup behind a
conditional gate.

A REAL model error re-drives itself every cycle at up to
response_trim_step_db, so a leak an order of magnitude smaller is invisible
to it in steady state.  A spurious one bleeds off in ~limit/leak cycles
(3.0/0.05 = 60).  This makes the trim self-healing rather than dependent on
never having made a mistake.
```

*From* `optimal_diagonal_control.py, the comment above _TRIM_LEAK_AFTER`:

```text
...but ONLY after this many CONSECUTIVE gate failures on the same bin.

Second correction, same session.  Leaking on a single gate failure was also
wrong: max_bins_per_update is 20 against 901 in-band bins, so after the trim
moves a target it takes ~45 cycles before that bin is re-solved, and until
then its cached prediction is stale and the gate fails for a reason that has
nothing to do with reachability.  Leaking immediately therefore bled away
corrections that were merely WAITING THEIR TURN -- verify_response_trim.py
TEST 2 settled at -1.05 dB against a real -2.00 dB bias.

50 cycles is comfortably longer than one full 901/20 = 45-cycle refinement
sweep, so a bin that has failed the gate this many times in a row has had at
least one chance to be re-solved and still cannot reach its target.  That is
the population that wound up in run 52.
```

*From* `optimal_diagonal_control.py, _update_response_trim, the comment above the leak`:

```text
LEAK (see _TRIM_LEAK_DB).  ONLY bins that failed the SOLVER-AGREEMENT
GATE decay toward zero.  NOT bins that passed the gate and merely sat
inside the deadband.

That distinction is the whole fix, and the first version of it was
wrong: leaking deadband bins too made the steady state sit at the
deadband EDGE rather than at zero, because the trim unwound until the
error grew back past 0.5 dB and then re-corrected -- a limit cycle.
verify_response_trim.py TEST 2 caught it: a real -2.00 dB bias
settled at -0.83 dB with 0.83 dB of residual error.

The justification is simply what each mask means.  A bin inside the
deadband has a TRUSTWORTHY measurement saying the correction is
right, so it holds.  A bin that failed the gate has NO trustworthy
measurement at all -- the solver did not reach its own target, so the
residual cannot be attributed to the model -- and a correction held
there is unjustified by construction.  Those are the bins that wound
up in run 52, and those are the ones that bleed off.
```

<a id="trim-gate"></a>

## The trim's own gate threshold (run 35)

*From* `optimal_diagonal_control.py, module docstring, response_trim_gate_db entry ('what run 35 measured')`:

```text
the comment in __init__ and what run 35 measured.
```

*From* `optimal_diagonal_control.py, the comment above the response_trim_gate_db parameter (_BASE_PARAMS)`:

```text
The solver-agreement gate has its OWN threshold, and the separation
is not cosmetic.  It first reused error_threshold_db, on the
reasoning that the scheduler's "worth refining" bar and "the solver
reached what it aimed at" were the same question.  Run 35 says they
are not.  Reconstructing this law's own per-bin prediction from
that run's saved FRF and drive and comparing it with the measured
response: model mismatch is 0.060 dB rms across all 7208 in-band
channel-bins, median +0.008 -- the FRF-update loop leaves
essentially nothing for an error trim to correct on this plant.
Yet at a 1.0 dB gate, 43.6% of gated channel-bins sat outside the
0.5 dB deadband, with error-vs-spec 0.541 dB rms against a model
mismatch of 0.041 dB rms.  Thirteen to one: the trim would have
spent the entire run integrating the SOLVER's own sub-threshold
shortfall -- second-guessing the optimizer that is already trading
these channels against each other -- and calling it model error.
At 0.5 dB only 1.6% of gated channel-bins clear the deadband, so on
a healthy rig the trim stays quiet.  A real bias still wakes it:
the shortfall is predicted-vs-target, computed entirely from H and
X, so a plant that does not match H cannot change it.  The gate
stays open exactly as wide while the error grows.
```

*From* `optimal_diagonal_control.py, _update_response_trim docstring, 'THE GATE HAS ITS OWN THRESHOLD'`:

```text
THE GATE HAS ITS OWN THRESHOLD, not error_threshold_db, and run 35 is
why -- see the comment on response_trim_gate_db in __init__.  Briefly:
a gate loose enough to serve as the scheduler's "worth refining" bar
is loose enough to let the solver's own sub-threshold shortfall
through, and on a rig whose model mismatch measures 0.06 dB rms that
shortfall is thirteen times larger than the thing this loop exists to
correct.
```

<a id="loop-kind"></a>

## What kind of loop this is: the mislabelling, and the evidence (runs 29, 31, 32)

*From* `optimal_diagonal_control.py, module docstring, 'WHAT KIND OF LOOP THIS IS', opening`:

```text
Two different things can be called "closed loop" here and this law has
historically been mislabelled on both counts.  Reports before 2026-09-17
called it a feedback law; the correction made then over-shot and called it
open loop, which understates it just as badly.  Precisely:
```

*From* `optimal_diagonal_control.py, module docstring, 'WHAT KIND OF LOOP THIS IS', end of the CLOSED paragraph`:

```text
because the error is always recomputed with the new H.  This is what runs
29/31/32 demonstrated -- 17.03 / 18.44 / 16.22 V^2 with the FRF update on,
against 153 / 485 / 245 V^2 with it off.
```

<a id="frf-update-threshold"></a>

## The default of frf_update_threshold: 0.5, raised from 0.05 (2026-09-19, runs 29 and 31)

*From* `optimal_diagonal_control.py, the comment above the frf_update_threshold parameter (_BASE_PARAMS)`:

```text
0.5, raised from 0.05 on 2026-09-19 once runs 29 and 31 measured
what this gate actually sees.  0.05 was set against the broken
_h_moved metric, which included out-of-band bins and read median
4.37; corrected and in-band the per-cycle movement is median 0.196,
90th 0.232, with a thin tail to 3.5.  At 0.05 essentially every
call demoted the fast law to the SDP (99% on run 28) AND every
previously-refined bin counted as drifted, so half the refinement
budget went to re-solving old bins forever (cum_frf_updates 10 per
call, 5570 and climbing).  0.5 sits at about the 95th percentile of
the measured distribution: 5% demotion, budget freed.

Changing this default does not alter any archived run: every
profile_01..26 sets field 2 explicitly (verified cell by cell).
```

<a id="defaults"></a>

## Reasons for other defaults (error_domain, drive_rcond, the trim)

*From* `optimal_diagonal_control.py, module docstring, frf_update_threshold line ('default 0.5 since 2026-09-19, earlier 0.05')`:

```text
re-solving an already-refined bin (default 0.5 since
2026-09-19, earlier 0.05)
```

*From* `optimal_diagonal_control.py, the comment above the error_domain parameter (_BASE_PARAMS)`:

```text
'linear' is the DEFAULT and reproduces runs 03-06 exactly.  'db' is
the 2026-09-17 objective change -- better pooled rms, but it spreads
error across every channel instead of sacrificing the two the plant
cannot reach, which is not what this rig wants (see the 8th
extra_parameters field).  Opt in with an 8th value of 1.
```

*From* `optimal_diagonal_control.py, the comment above the drive_rcond parameter (_BASE_PARAMS)`:

```text
Drive-subspace restriction for the FACTORED path only -- see
optimal_diagonal_control_fast._solve_one_bin.  Parsed here so one
parameter string serves both classes.

NOW A MULTIPLE of the plant's own singular-value spread, not an
absolute ratio -- see optimal_diagonal_control_fast.
_effective_drive_rcond.  Runs 22-25 swept the absolute value on the
rig at 0 / 1e-2 / 3e-2 / 5e-2 and 3e-2 was a genuine minimum; this
frame's smallest structural direction is sigma_5/sigma_1 = 0.0152,
so 3e-2 is 2.0x that.  Expressed as a multiple it tracks a
differently conditioned article without re-tuning.  Negative means
an absolute ratio, |value|.
```

*From* `optimal_diagonal_control.py, the comment above the response_trim_gain parameter (_BASE_PARAMS)`:

```text
RESPONSE-ERROR TRIM -- fields 11-14, OFF by default.  Every
behaviour change in this file ships defaulted off (commit
45d417a3); this one is no exception, so an archived profile that
names ten fields or fewer reproduces exactly.
```

<a id="fast-coherence-claim"></a>

## optimal_diagonal_control_fast: does the unconstrained solve reach coherence 1.0? (run 13, retracted 2026-09-17)

The text below was the middle of the module docstring.  It includes a retraction, kept because the mistake (comparing one channel's minimum with other channels' minima) is easy to repeat.

*From* `optimal_diagonal_control_fast.py, module docstring, from 'COUNTER-MEASUREMENT, 2026-09-17' to the prediction for run 14`:

```text
COUNTER-MEASUREMENT, 2026-09-17 (run 13, this 6-drive/8-control system): the
saved drive CPSD showed median pairwise coherence 0.407, 2.4% of pairs above
0.99 and 0.0% at exactly 1.0 -- the claim above did NOT reproduce. Eigenvalue
participation was 1.21 of 6, so the drive was concentrated but not
rank-deficient.

Run 13 is still not admissible evidence, but for ONE reason, not two: its
saved metadata reads control_python_function = pseudoinverse_control and
update_tf_during_control = 1, so profile_13 was never loaded and whatever it
measured is not this law.

RETRACTED 2026-09-17, same day, and worth keeping as a caution about this
metric. This paragraph previously also called run 13's system identification
weak, "minimum multiple coherence 0.492 against 0.75-0.88 for the twelve-run
comparison set". That comparison was invalid: 0.492 is run 13's CHANNEL-7
minimum and 0.75-0.88 is the range of the OTHER channels' minima in the other
runs. Control channel 7 sits between 0.28 and 0.54 in ALL THIRTEEN
identifications taken on this system -- it is a property of the rig, not of
any run. Scored consistently, run 13's identification was the BEST of the set
(worst per-channel minimum 0.531; run 14's is 0.411, run 05's 0.284). Every
run's identification has median coherence 0.998 and 5th percentile ~0.97.
When judging a system ID here, use the median and the 5th percentile, or the
per-channel minima compared like against like -- a bare minimum over all
channels and lines is dominated by channel 7 at the band edge and says
nothing about run quality.

Treat the 1.0-coherence claim as established at 12 drives and UNVERIFIED at 6
until run 14 lands.

PREDICTED for run 14 by offline dry run on run 01's identification (this exact
parameter string, law constructed the way Rattlesnake constructs it, frozen H):
after 25 cycles and 500 refined bins, median pairwise coherence 0.501, 2.8% of
pairs above 0.99, 0.0% at exactly 1.0, participation 1.09 of 6. Close to what
run 13 showed, which is unexplained given run 13 ran a different law -- read
nothing into the resemblance until run 14 is scored.
```

<a id="fast-safety-rule"></a>

## optimal_diagonal_control_fast: the safety gate was wrong, then fixed (2026-09-17)

*From* `optimal_diagonal_control_fast.py, module docstring, 'REVISED 2026-09-17, and the previous version of this paragraph was wrong'`:

```text
REVISED 2026-09-17, and the previous version of this paragraph was wrong
in a way worth recording. The gate used to be `not np.array_equal(H,
H_cache)` -- ANY bitwise difference -- and this docstring claimed that with
TF-update off "H [is] set once at system_id_update() and never re-passed".
That is not what Rattlesnake does: random_vibration_sys_id_data_analysis.py
:431-475 passes a transfer_function to control() EVERY cycle in both
regimes, and system_id_update() is never called during control at all
(:190, only from perform_control_prediction). With the update off the array
happens to be the same frozen sysid_frf each time, so array_equal held and
the fast path ran -- the right outcome by the wrong mechanism. With the
update ON the environment republishes an incrementally-averaged FRF every
cycle (:390-394), so H was never bit-identical, EVERY bin fell through to
the SDP, and this law was byte-for-byte the base class. It was only ever
fast in the configuration it was least needed in, and any comparison of
"fast" against "optimal diagonal" with the update on was a law against
itself. A relative-change threshold fixes that; frf_update_threshold = 0
restores the old any-change behaviour.
```

*From* `optimal_diagonal_control_fast.py, _h_moved docstring, 'CHANGED 2026-09-17'`:

```text
CHANGED 2026-09-17.  This used to be `not np.array_equal(H_clean,
self.H_cache)` -- ANY difference at all, down to the last bit.  That
reads as the conservative choice and is in fact a silent failure:
with "Update Transfer Function During Control" on, the environment
republishes an incrementally-averaged FRF every single cycle
(random_vibration_sys_id_data_analysis.py:390-394), so H is never
bit-identical, every bin falls through to the SDP, and
optimal_diagonal_control_fast is byte-for-byte the base class.  The
law was only ever fast with FRF update OFF -- the configuration it
was least needed in -- and a run comparing "fast" against "optimal
diagonal" with the update on was comparing a law against itself.
```

<a id="fast-h-moved"></a>

## optimal_diagonal_control_fast: what the FRF-movement gate measures (runs 21, 28, 29; 2026-10-25)

*From* `optimal_diagonal_control_fast.py, _h_moved, the comment above H_ref ('REFERENCE (fix, 2026-10-25; field 25)')`:

```text
REFERENCE (fix, 2026-10-25; field 25).  The gate is meant to measure
how much the FRF moved SINCE THE LAST CALL -- its threshold was set
from "per-cycle movement" (median 0.196, 90th 0.232 on runs 29/31).
But it compared against H_cache, which holds each line's FRF at that
line's LAST SOLVE.  With 20 lines solved per call against ~900 in
band, most of the band carries an FRF several calls old, so with the
FRF update on the metric measured accumulated drift, not per-cycle
movement, and once that passed the threshold every later call would
have gone to the SDP.  Now: the previous call's FRF (falls back to
H_cache on the first call).  gate_reference = 0 restores the old
reference.  Both are logged.
```

*From* `optimal_diagonal_control_fast.py, _h_moved, the comment above the movement log`:

```text
Record what the FRF actually does, whether or not it trips the gate.
The demotion threshold should be set from the movement this rig
really produces, and that cannot be recovered afterwards -- only the
FINAL FRF is saved, so a run file shows cumulative drift (about 28%
between the system ID and end of control on run 21) and says nothing
about the per-cycle movement the gate tests.
```

*From* `optimal_diagonal_control_fast.py, _h_moved, the comment inside the log print ('PER-BIN counters')`:

```text
PER-BIN counters, not per-call.  The demotion rate above
counts CALLS; these count the solves that actually took
each path, which is the thing "was the fast law used?"
really asks.  Confirming run 29 needed three indirect
arguments (metadata, the gate log, and a drive-coherence
signature above the SDP's 0.95 cap) only because these
counters existed but were never printed.
```

<a id="fast-init-order"></a>

## optimal_diagonal_control_fast: state is set before super().__init__() (2026-09-17)

*From* `optimal_diagonal_control_fast.py, __init__, the comment above the state set before super().__init__()`:

```text
This subclass's own state is set up BEFORE super().__init__(),
not after.  The base constructor calls _initialize() whenever it
is handed a non-None transfer_function, and _initialize() runs
_refine_batch() -> _solve_one_bin(), which is overridden here and
reads self.bm_rank / self.n_fast_solves / self._h_changed_this_
call.  With those assignments after the super() call the law
crashed with AttributeError: 'optimal_diagonal_control_fast'
object has no attribute 'n_fast_solves' (found 2026-09-17 by a
harness that constructs the law with the FRF in hand).  Rattle-
snake itself builds control laws before system ID, with
transfer_function=None, so the crash never fired in the app -- it
was latent, and would have fired the moment a law was constructed
with an FRF already available.
```

<a id="fast-db-objective"></a>

## optimal_diagonal_control_fast: why the dB objective exists (run 14)

The first lines of the excerpt record that the docstring once called 'db' the default; the base class has always defaulted to 'linear' and the code follows it (corrected 2026-10-25).

*From* `optimal_diagonal_control_fast.py, _bm_obj_and_grad docstring, from 'CORRECTED 2026-10-25' through 'WHY THIS CHANGED'`:

```text
directly.  CORRECTED 2026-10-25: this docstring used to call 'db' the
default since 2026-09-17; the base class has always set 'linear' and
says so, and the code follows the base class.

WHY THIS CHANGED.  The linear objective caps what abandoning a channel
can cost at that channel's own target squared -- drive its response to
zero and the penalty stops growing -- so on a plant whose rows have
unequal gain the solver sells the hardest channels to buy the easy
ones, and is correct to, given what it was asked to minimize.  On the
6-drive/8-control frame with a flat -30 dB specification and row gains
spanning 3.8 to 10.3 dB, that cost 13X+ and 14X+ about 5 dB each: run
14 converged to a self-predicted 9.47 and 8.28 dB rms on those two
channels against a reachability floor of 4.3 and 3.9, while beating
that floor on 8X+.  Pooled, 4.65 dB against a 2.38 dB floor.  Measured
offline on run 14's own FRF, the dB objective takes the pooled figure
to 3.06 dB and 13X+ to 5.52.
```

<a id="fast-warm-start"></a>

## optimal_diagonal_control_fast: warm start (fix 5, 2026-10-25)

*From* `optimal_diagonal_control_fast.py, _bm_solve, the comment above the warm start ('WARM START (fix 5, 2026-10-25; field 24, 0 = off)')`:

```text
WARM START (fix 5, 2026-10-25; field 24, 0 = off).  Start from this
line's CURRENT drive when that scores better than the buzz seed.
Before this, every re-solve restarted from the buzz eigenvectors and
threw away the line's existing solution, so a line re-solved because
the FRF drifted a little was re-found from scratch: slower, and free
to land in a different local minimum of the non-convex factored
problem from one cycle to the next.  Taking the better of the two
starts means a stale or poor current drive can never make it worse.
```

<a id="fast-tolerances"></a>

## optimal_diagonal_control_fast: L-BFGS tolerances on real FRFs

*From* `optimal_diagonal_control_fast.py, _bm_solve, the comment above the L-BFGS tolerances`:

```text
Tolerances matter a lot here: on REAL (not synthetic-random) FRF
data, ~70% of bins hit a 200-iteration cap at tight tolerances
(ftol=1e-14, gtol=1e-10) without visibly improving dB accuracy --
checked directly (bm_accuracy tuning script): loosening to
maxiter=50/ftol=1e-9/gtol=1e-7 cuts time ~5.7x (18.7ms -> 3.3ms
mean on real bins) for only +0.4 dB mean / +2.7 dB worst-case
versus the SDP's own accuracy on the same bins -- a real,
verified speedup (~3.6x vs the SDP mean on real data), much more
modest than synthetic-random-H benchmarks suggested but honest.
```

<a id="fast-drive-rcond"></a>

## optimal_diagonal_control_fast: the drive-subspace restriction (runs 14, 22-25)

The measurement table in the `_restricted_basis` excerpt was taken when the default was an absolute ratio of 1e-2; the default is now a multiple (2.0) of the plant's own singular-value spread.

*From* `optimal_diagonal_control_fast.py, _effective_drive_rcond docstring, 'WHY A MULTIPLE'`:

```text
WHY A MULTIPLE.  An absolute ratio has to be re-tuned for every test
article, because it is a threshold on a quantity that is a property of
the structure.  The rig sweep (runs 22-25) put the optimum at 3.0e-2 on
this frame, and this frame's smallest structural direction sits at
sigma_5/sigma_1 = 0.0152 -- so the tuned optimum is 2.0x that, which is
the default here.  On a differently conditioned article 2.0x tracks the
plant automatically where 3.0e-2 would not.
```

*From* `optimal_diagonal_control_fast.py, _restricted_basis docstring, from 'WHY THE FACTORED PATH NEEDS THIS' to the end`:

```text
WHY THE FACTORED PATH NEEDS THIS AND THE SDP PATH DOES NOT.  The dB
objective removes the ceiling on what missing a channel costs, which
is the point, but it also means the solver will spend unlimited drive
to chase a channel the plant barely reaches.  Unconstrained on run
14's FRF it did exactly that: pooled error 1.93 dB, better than the
rcond-1e-3 reachability floor itself, bought with +10.4 dB more drive
on average, +30.2 dB on the worst bin, at eigenvalue participation
1.02 -- a nearly rank-one drive pushed down a near-singular direction.
The base class's SDP does not need this because its coherence-cap
constraint already restrains it (+2.6 dB); the factored path is
unconstrained by design and has nothing else holding it.

reg is NOT the right lever and was measured not to be: it penalizes
every direction equally, so raising it from 1e-6 to 1.0 gave back the
accuracy (2.31 dB) while still leaving +7.8 dB mean drive and a +25.5
dB worst bin.  Restricting the SUBSPACE is what the reachability floor
itself does (restrict_rcond), and it targets the actual problem.

Measured on run 14, pooled / mean drive / worst bin, against the
as-shipped linear objective at 4.68 dB and 0 dB drive:

    unrestricted   1.93 dB   +10.4 dB   +30.2 dB
    rcond 1e-3     2.44 dB    +5.2 dB   +30.2 dB
    rcond 1e-2     3.83 dB   -12.2 dB    +8.7 dB   <- default
    rcond 5e-2     4.35 dB   -20.7 dB    +8.7 dB

The 1e-2 default is the one point on that curve that is better than
the law as it stands on BOTH axes -- 0.85 dB more accurate for 12 dB
LESS drive -- so it is safe to adopt without knowing the rig's
headroom.  Set drive_rcond to 0 to disable the restriction.
```

*From* `optimal_diagonal_control_fast.py, _restricted_basis, the comment above the subspace restriction ('APPLIES IN BOTH ERROR DOMAINS since 2026-09-18')`:

```text
APPLIES IN BOTH ERROR DOMAINS since 2026-09-18.  It was briefly
gated to 'db' only, to keep error_domain=0 reproducing the
pre-2026-09-17 law exactly.  Run 22 is what that cost: the factored
path in its DEFAULT linear mode then had nothing constraining it at
all -- no coherence cap, which it cannot express, and no subspace
restriction -- and it commanded a drive whose response depends on
inter-drive cancellation the rig cannot physically realize.

THE MEASUREMENT THAT SETTLES IT.  Perturbing the commanded drive by
1%, which is less than the difference between what run 22 commanded
and what its saved drive CPSD actually shows:

    configuration                      rms    level  | +1% drive
    factored, unconstrained           4.64   -0.42   | 11.85  +16.69
    factored + drive_rcond 1e-2       3.73   +1.94   |  4.68   +3.80
    SDP + coherence cap 0.95          4.16   -0.50   |  7.23   +7.92
    SDP, cap off                      4.09   -0.49   |  7.98   +9.09

Run 22 measured +14.29 dB level against -0.39 predicted from the
commanded drive; a 1% realization error predicts +16.69.  The
unconstrained solve is not wrong, it is UNREALIZABLE -- its accuracy
lives in cancellation between drives that survives only if the drive
is reproduced to a precision no rig delivers.  drive_rcond bounds
that sensitivity directly, and does it better than the coherence cap,
which only halves it.

Set drive_rcond to 0 to disable and recover the run-22 behaviour.
```

<a id="fast-svd-floor"></a>

## optimal_diagonal_control_fast: per-line singular-vector selection (2026-10-24)

*From* `optimal_diagonal_control_fast.py, _per_line_k, the comment above the gap search`:

```text
The DEEPEST qualifying gap, not the largest.  Near a resonance one
mode dominates and the largest gap is sigma_1/sigma_2 (20x or more),
which would keep a single direction and abandon every other
channel.  The structural floor is at the bottom of the spectrum.
...and only where the direction BELOW the gap is genuinely weak
(sigma/sigma_1 <= svd_weak_max).  Without this, a line whose only
qualifying gap is a resonance's sigma_1/sigma_2 still keeps one
direction: on the sweep plant with svd_gap_min 6, 16-81 of 361
lines did exactly that.
```

<a id="fast-db-guard"></a>

## optimal_diagonal_control_fast: the dB-objective coherence guard (fix 6, 2026-10-25)

*From* `optimal_diagonal_control_fast.py, _solve_one_bin, the comment above the dB coherence guard ('fix 6, 2026-10-25; field 23')`:

```text
dB-OBJECTIVE COHERENCE GUARD (fix 6, 2026-10-25; field 23).
With the dB objective the unconstrained factored solve drove
MEDIAN pairwise drive coherence (gamma^2) to 0.99-1.00 at 6
drives (compare_optdiag.py, 2026-10-24): fully dependent
drives, the failure this file's top docstring warns about.
The linear objective did not (0.66 unrestricted).
  1 (default): if this line's solution breaks
    max_drive_coherence, re-solve THIS LINE with the SDP,
    which holds the cap as a constraint.
  2: shrink the offending cross terms after the solve
    (cheap, but measured at +1.8 dB error: capping afterwards
    breaks the inter-drive cancellation the solve relied on).
  0: off (pre-2026-10-25).
```

<a id="small"></a>

## Smaller dated remarks

Single sentences that carried a date, a run number or a bug that was found; the code comment now says only what the code does.

*From* `optimal_diagonal_control.py, module docstring, startup_test_level_cap_db entry ('Added 2026-09-17 to match every law in control_laws.py')`:

```text
(default -9.0; 0 dB = full spec-match). Added
2026-09-17 to match every law in
control_laws.py. The clamp is on the predicted
```

*From* `optimal_diagonal_control.py, module docstring, max_solve_ms_per_update entry ('since 2026-10-25')`:

```text
decide.  Empty fields keep their position and mean "default"
(since 2026-10-25), so a 500 ms budget with 400 lines and every
other field at its default is
```

*From* `optimal_diagonal_control.py, the comment above the max_solve_ms_per_update parameter (_BASE_PARAMS), first line`:

```text
TIME BUDGET PER CALL (added 2026-10-25), in milliseconds of line
```

*From* `optimal_diagonal_control.py, __init__, console-throttling comment`:

```text
Console throttling -- see _should_log.  Once every bin is solved and
the FRF is frozen, _refine_batch does no work but still ran two print
statements per control cycle, which on run 14 buried the terminal in
hundreds of identical lines and read as a hung process.  (It was not
hung: newly_refined=0, n_deferred=0, cum_sdp_refinements frozen.)
```

*From* `optimal_diagonal_control.py, __init__, live-FRF guard comment`:

```text
Live-FRF guard state.  Set here as well as in _initialize so a
harness that reaches _refine_batch without going through
_initialize cannot raise AttributeError -- the same latent bug
the fast subclass hit on 2026-09-17 with n_fast_solves.
```

*From* `optimal_diagonal_control.py, _build_sdp_problem, IRLS comment`:

```text
rejects it as non-DPP (tried 2026-09-17, "SDP problem is not
DPP-compliant").
```

*From* `optimal_diagonal_control.py, _initialize, H_initial comment`:

```text
self.H_initial = H_clean.copy()  # diagnostic AND, since 2026-09-19, the
                                 # fallback the divergence guard reverts to
```

*From* `optimal_diagonal_control.py, _update_response_trim docstring, 'WHAT IT INTEGRATES'`:

```text
and stops there.  Worked through on paper 2026-09-21 before any of
this was written; the clamp happening to sit at the right value is
a coincidence, not a design.
```

*From* `optimal_diagonal_control_fast.py, module docstring, 'Empty fields keep their position' (the date and the old dropping behaviour)`:

```text
Empty fields keep their position and mean "use the default" (since
2026-10-25; before that the parser DROPPED them, so every earlier field had to
be written out).  Per-line selection on, everything else default:
```

*From* `optimal_diagonal_control_fast.py, _effective_drive_rcond, the comment above the cache ('Found 2026-09-18 when an FRF sensitivity test reported 10.34 dB')`:

```text
Only cache once the FULL band is in hand.  H_cache is populated by
_initialize before _refine_batch runs, so the control loop always
has it -- but _solve_one_bin called directly (a harness, a unit
test) would otherwise compute the reference from the ONE bin it was
handed and cache that answer forever.  Found 2026-09-18 when an FRF
sensitivity test reported 10.34 dB for a configuration the control
loop scores at 5.95.
```

*From* `optimal_diagonal_control_fast.py, _restricted_basis, 'FLOOR ON THE COUNT' heading (the date)`:

```text
FLOOR ON THE COUNT (2026-10-24).  Every cut is relative to the line's
```

*From* `optimal_diagonal_control_fast.py, _buzz_solve_all docstring, second line ('fix 8, 2026-10-25')`:

```text
coherence-capped (fix 8, 2026-10-25; field 22, default 0 = off).
```

*From* `optimal_diagonal_control_fast.py, _buzz_solve_all docstring, 'the 2026-10-24 comparison'`:

```text
the 2026-10-24 comparison read a MAXIMUM of 0.98-1.00 for the SDP law
```

*From* `octave_band_switching_control.py, set_test_level_db, the comment above the chained call (last line)`:

```text
Added 2026-09-21 with the trim itself.
```
