# Rewriting `_effective_drive_rcond`'s structural-floor rule (2026-11-04)

The drive-subspace restriction in `optimal_diagonal_control_fast` has to know where
the plant stops and the identification noise starts.  Until now the band mode found
that point with the **largest gap** in the median singular-value spectrum.  That rule
inverts on a full-rank plant measured with noise, and it inverted on the first one it
met outside the frame: the band-average 12-drive RC plate, where it kept 2 of 12 drive
directions and the law converged 7.9 dB off its own target
([optdiag_on_rc_plate_2026-11-04.md](optdiag_on_rc_plate_2026-11-04.md)).

This note is the replacement criterion, why it is the one chosen, and the before/after
on every plant in the repo that exercises the path.

## What was actually wrong

Two functions read the spectrum the same way and were wrong the same way:

    _effective_drive_rcond   the band threshold:   ref = med[argmax(gaps)]
    _band_structural_rank    the min-keep floor:   n   = argmax(gaps) + 1  (if >= svd_gap_min)

The rule's own docstring states the assumption: the identification noise floor is "an
isolated step below the structural directions", so the largest gap in the spectrum
must be that step.  On the 6-drive frame it was -- 8.9x between sigma_5 and sigma_6,
against 2.1-3.8x everywhere else.

It is not an assumption a measured full-rank plant satisfies.  Noise does not put a
step under the spectrum; it **floors the bottom of it into a plateau**.  The RC plate's
50-average H1 estimate, median sigma_k/sigma_1:

    1.000  0.5109  0.2105  0.0696  0.0512  0.0252  0.0141  0.0105  0.0087  0.0074  0.0064  0.0055
           1.94    2.45    3.02    1.33    2.10    1.79    1.33    1.20    1.18    1.16    1.21   <- gaps

The true plant runs on down to 5.2e-05.  Directions 7-12 of the estimate are noise
sitting where the structure is three orders lower, and noise against noise has no gaps:
the bottom five adjacent ratios are 1.16-1.21.  So the largest gap left in the spectrum
was the genuine **structural** 3.02x at sigma_3 -> sigma_4, the reference came out at
0.2105, `drive_rcond` 2.0 put the threshold at 0.4206, and 10 of 12 drive directions
went in the bin.

**Why this never showed on the frame plants.** `svd_min_keep` defaults to auto =
structural rank - 1, and on the frame plants `_band_structural_rank` got the rank right,
so the min-keep floor quietly raised `k` on every line and covered for the band
threshold.  On 10x14 the band threshold was keeping 6 of 10 while the floor held k at 9.
On the plate both functions failed together -- rank 3, floor 2, threshold 2 -- so nothing
caught it.  That is the real reason the defect survived: one of the two rules was
masking the other.

## The replacement

`_structural_floor(med)` -- one function, called by both of the above, so they cannot
disagree about where the plant ends.  It returns the index of the lowest direction still
treated as structural, and the name of the detector that decided.  Three detectors,
strongest evidence first:

| detector | test | what it catches |
| --- | --- | --- |
| `svd_floor` | `sigma/sigma_1 < svd_floor` (1e-6) | a direction the model does not have at all -- an exactly rank-deficient plant (6x8's 6th direction sits at 4e-17) |
| `plateau` | a run of >= 2 adjacent ratios below `svd_flat_max` at the **bottom**, whose topmost member is itself <= `svd_weak_max` | the identification noise floor on a full-rank plant, where no gap marks it |
| `gap` | the **deepest** adjacent ratio >= `svd_gap_min` whose **lower** side is <= `svd_weak_max` | an isolated step down onto a floor -- the frame case the old rule was built for |

If none fires there is no identifiable floor, every direction present is structural, and
the reference is the smallest one -- `drive_rcond` still trims from there.

Two of the three conditions are not new: `_per_line_k` has applied "deepest, not
largest" and the weak-lower-side test since 2026-10-24, for exactly this reason
("without this, a line whose only qualifying gap is a resonance's sigma_1/sigma_2 still
keeps one direction").  **The band mode was never given them.**  Most of this change is
backporting a fix the per-line path already had.

`svd_flat_max` is the one new parameter (field 26, default 0 = auto = `sqrt(svd_gap_min)`
= 1.73).  It is derived from the already-tuned gap parameter rather than being a number
of its own: a ratio of `svd_gap_min` marks a floor, so adjacent ratios below its
geometric half are flat.

The plateau's weak-side test matters as much as its flatness test.  A well-conditioned
plant is flat *everywhere* -- no adjacent ratio in 10x14's exact spectrum exceeds 2.4 --
so without it a generous `svd_flat_max` would read the whole spectrum as noise and keep
one direction, which is the collapse this rewrite exists to stop.  Checked directly:
a synthetic flat-but-strong spectrum (1.0 0.9 0.8 0.7 0.6) returns `none` and keeps all
five.

### Alternatives rejected

- **An absolute floor from the averaging statistics or coherence.** The most principled
  option in theory -- the H1 noise floor really is set by `n_avg` and multiple coherence,
  both of which the law is handed.  Rejected because the map from coherence to a
  singular-value *ratio* is not calibrated anywhere in this repo, so adopting it would
  mean shipping an uncalibrated constant and re-deriving the frame's tuned reference at
  the same time.  Worth doing if someone wants to measure it; it is not something to put
  in on the strength of an argument.
- **A fixed threshold relative to `sigma_max`.** Already available as `drive_rcond < 0`,
  and the existing docstring's objection stands: it is a threshold on a property of the
  structure, so it has to be re-tuned per test article.  That is what the multiple was
  introduced to avoid.
- **Plateau detection alone**, without the `gap` detector. Would lose the frame case,
  where the floor genuinely is an isolated step and there is no plateau to find.
- **Splitting the difference with a constant tuned between the two plants.** Not needed:
  the same defaults serve both, because the two plants' floors are different *kinds* of
  floor rather than different values of one.

## The silent failure

Discarding most of the drive space was invisible from outside the law: it still
converged, just to a floor nobody could explain.  The band-mode print now carries the
detector, the retained rank and the spectrum it was read from:

    drive_rcond 2 x plant spread 0.01407 (plateau) -> effective 0.02814; keeps 5 of 12
    drive directions; median sigma_k/sigma_1 [1. 0.5109 0.2105 ...]

and when the restriction keeps less than half the drive space it also raises a
`RuntimeWarning` naming the retained rank, the detector, the threshold, and the two
overrides (`drive_rcond` 0, or a negative absolute ratio).  That fires on the plate
*after* the fix too -- 5 of 12 is still a large cut, and the user should be told.
`_band_structural_rank`'s print names its detector as well.

## Before / after

Run with `control_laws/tests/compare_optdiag_frame.py`'s own `run()` and metrics on the
frame plants, and `control_laws/tests/optdiag_on_rc_plate.py`'s on the plate (`band_db`
and `drive_lbf2` are the band-average study's own definitions).

### The plants the old rule was tuned for: no regression

15 cycles, 20-frame CPSD estimates, both budget configurations, err on the reachable
channels:

| plant | | err (dB) | given up (dB) | drive (dB) | drive &#124;coh&#124; | band keeps |
| --- | --- | --- | --- | --- | --- | --- |
| 6x8 (rank 5 of 6) | before | 1.16 | 9.81 | -12.9 | 0.74 | 4 of 6 |
| | after | 1.16 | 9.81 | -12.9 | 0.74 | 4 of 6 |
| 10x14 (full rank 10) | before | 0.23 | 7.10 | -11.0 | 0.55 | 6 of 10 |
| | after | 0.23 | 7.10 | -11.0 | 0.54 | **9 of 10** |
| 10x14_rankdef (rank 9) | before | 0.34 | 7.08 | -14.3 | 0.49 | 8 of 10 |
| | after | 0.34 | 7.08 | -14.3 | 0.49 | 8 of 10 |

**The real rig data is bit-identical.** The frame plants above are the FE model; the
plant `drive_rcond 2.0` was actually tuned on is the 6-drive rig, and five of its H1
snapshots are stored in `examples/sixdrive12resp/results/h1_snapshots/`.  Old rule and
new rule agree exactly on every one, all via the `gap` detector -- which is the one the
old rule was built for:

| snapshot | reference, old | reference, new | detector | keeps |
| --- | --- | --- | --- | --- |
| `h1_sysid_baseline` | 0.1507 | 0.1507 | gap | 3 of 6 |
| `h1_expavg_0dB` | 0.02404 | 0.02404 | gap | 4 of 6 |
| `h1_expavg_-3dB` | 0.01585 | 0.01585 | gap | 4 of 6 |
| `h1_uncapped_-6dB` | 0.01839 | 0.01839 | gap | 4 of 6 |
| `h1_uncapped_-12dB` | 0.01433 | 0.01433 | gap | 4 of 6 |

`h1_expavg_-3dB`'s spectrum is the one the `_effective_drive_rcond` docstring quotes
(1.000 0.368 0.108 0.054 0.0159 0.00125 here, against the docstring's 1.000 0.319 0.084
0.032 0.015 0.0017).  The FRF the run-14 analysis used,
`results/analysis/frf_frame6x12_H.npz`, is also unchanged (reference 0.03061, 4 of 6,
via `svd_floor` -- it is an exact FRF, its 6th direction at 4e-17).

6x8 and 10x14_rankdef are bit-identical: their floors are the numerically-absent
directions, which the `svd_floor` detector finds in the same place the largest gap did.
The plate's **exact** FRF is also unchanged (8 of 12, via the `gap` detector) -- the one
case the old rule got right stays right.

10x14 is the only frame plant whose output moves, and it moves because the old rule was
wrong there too: its largest gap is 2.31x at sigma_7 -> sigma_8 with a lower side of
0.118, nowhere near a noise floor, on a plant the repo documents as FULL RANK 10.  The
closed-loop result is unchanged to two decimals on every axis because `svd_min_keep` auto
was already holding k at 9; the golden recording differs by 2.3e-04 relative on the last
drive CPSD, first differing at cycle 3.  `golden_optdiag.py` was re-recorded for it.

### The 12-drive RC plate: the fix

On the study's own 1 Hz grid, 15 cycles, scored against the exact plant.  "in-model" is
the error against the law's **own** FRF -- whether it reaches its own target.  These are
the four cases in the RC plate note's results table:

| case | | in-model | truth | band_db | drive_lbf2 | worst drive (lbf) | rank | &#124;coh&#124; | ch > 6 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FRF fixed, default budget | before | 3.63 | 3.89 | 1.55 | 5.52e+04 | 81.3 | 12 | 0.53 | 0 of 48 |
| | after | **2.92** | 3.25 | 1.39 | 9.06e+04 | 98.3 | 12 | 0.51 | 0 of 48 |
| FRF fixed, full band | before | 6.56 | 6.61 | 1.88 | 1.14e+04 | 35.4 | 11 | 0.82 | 21 of 48 |
| | after | **3.01** | 3.21 | 1.29 | 7.87e+04 | 91.2 | 12 | 0.52 | **0 of 48** |
| in-control FRF update | before | 4.05 | 4.11 | 1.81 | 8.27e+04 | 98.8 | 12 | 0.51 | 1 of 48 |
| | after | 12.27 | **3.44** | 2.17 | 1.07e+05 | 114.5 | 12 | 0.50 | **0 of 48** |
| exact analytic FRF | before | 2.92 | 2.92 | 1.17 | 1.77e+11 | 2.12e+05 | 7 | 0.86 | 0 of 48 |
| | after | 2.92 | 2.92 | 1.17 | 1.77e+11 | 2.12e+05 | 7 | 0.86 | 0 of 48 |

The exact-FRF case is **bit-identical** before and after: its floor is a real isolated
step (8.36x at sigma_11 -> sigma_12 onto 5.2e-05) and the `gap` detector lands on exactly
the direction the largest gap did.  The old rule was never wrong on the exact plant --
only on the measured one, which is the only one a rig ever has.

The full-band case is the clean one, because 15 cycles cover the whole band there and the
law reaches full coverage.  in-model error 6.56 -> **3.01 dB** against an achievable floor
of 0.44; drive rank 11 -> 12 and the commanded drive's own effective rank 2 -> 4; 21 of 48
channels over 6 dB -> **none**.

Two further things fall out of it:

- **The monotone worsening with coverage is gone.** Error by cycle, full-band budget:
  before 3.83 4.02 4.29 4.49 4.75 ... 6.61, climbing every cycle as lines were refined.
  After: 3.26 3.26 3.25 3.24 3.24 ... 3.21.  Refinement was moving lines onto a
  2-direction solution, and that was the whole of the effect.
- **The error is no longer in the law's own model.** in-model and truth were 6.56 and
  6.61 before -- it converged 6.6 dB off its own target.  They are now 3.01 and 3.21, so
  what is left is mostly the 11 %-accurate FRF rather than the law's own solution.

It costs drive.  1.14e+04 -> 7.87e+04 lbf^2, 35 lbf -> 91 lbf on the worst drive.  That
is the restriction doing less restricting, and it is the trade the parameter exists to
make -- but it is still 2.6x less drive than `drive_rcond: 0` commands (1.74e+05) for
0.2 dB more error, so the restriction is earning its place rather than being switched off
by the back door.

#### The in-control FRF update case is mixed, and worth its own look

This is the configuration the band-average study actually streamed, and it is the one
case where the fix is not a clean win:

- It **removes the drive excursion.** Before, one cycle of 15 put `drive_lbf2` at
  **3.37e+15**, eleven orders above the rest, and recovered -- the transient form of the
  study's 5.86e+27 divergence.  After, `drive_lbf2` climbs smoothly 1.23e+04 -> 1.07e+05
  with no excursion at all.
- True error improves, 4.11 -> **3.44 dB**, and the last channel over 6 dB goes away.
- But `band_db` **worsens**, 1.81 -> 2.17: the error is less spiky narrowband and more
  biased within bands.
- And the law's own model drifts much further from what it achieves: in-model error
  4.05 -> **12.27 dB** while the true error is 3.44.  Under `tf_update` the FRF is
  re-identified from the *control* drive each cycle, so the drive the law commands
  determines the model it will next solve against.  Before the fix that loop was
  self-consistent in a narrow subspace; after it, the live estimate and the achieved
  response disagree badly even though the response itself is good.

One seed, one plant.  Nothing here argues against the floor fix -- the excursion and the
true error both improve -- but the in-control update path has a feedback problem of its
own that this change does not address and makes easier to see.

### The 12-drive RC plate at 8 Hz (full coverage in 10 cycles)

A coarser grid, where 10 cycles already cover the band, as an independent check that the
1 Hz result is not an artefact of partial coverage:

| budget | | in-model | truth | band_db | drive_lbf2 | rank | &#124;coh&#124; | ch > 6 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default (20 lines/call) | before | 6.46 | 6.50 | 1.98 | 1.33e+04 | 11 | 0.80 | 21 of 48 |
| | after | **3.08** | 3.23 | 1.38 | 7.47e+04 | 11 | 0.52 | **0 of 48** |
| full band (400 lines) | before | 7.93 | 7.93 | 2.77 | 2.17e+03 | 2 | 1.00 | 29 of 48 |
| | after | **3.08** | 3.15 | 1.18 | 6.63e+04 | 4 | 0.62 | **0 of 48** |

At full coverage the band threshold collapsed hardest: 2 of 12 directions, drive |coh|
exactly 1.00, in-model error 7.93 dB.  After the fix, 3.08 dB -- against the 2.81 dB
`drive_rcond: 0` reaches and a 0.44 dB achievable floor -- at |coh| 0.62 and 2.6x less
drive than `drive_rcond: 0` commands.  Same conclusion as the 1 Hz grid, reached from a
worse starting point.

Raw output of every run quoted above is in
[structural_floor_rewrite_2026-11-04_runs.txt](structural_floor_rewrite_2026-11-04_runs.txt).

## Limitations

- `_FALLBACK_SPREAD = 0.0152` is untouched.  It is still a single plant's constant,
  used when the spectrum cannot be measured at all (`H_cache` empty -- a harness calling
  `_solve_one_bin` directly, never the control loop).  Same defect class, different
  code path, left alone deliberately.
- The plateau detector needs the noise floor to be *visible* in the median spectrum.
  A plant identified well enough that its smallest structural direction is still above
  the noise has no plateau and no gap, and then nothing is cut beyond `drive_rcond`'s
  trim of the weakest direction -- correct, but it means the restriction's strength now
  depends on identification quality in a way it did not pretend to before.
- `_MIN_PLATEAU = 2` is hardcoded, not a parameter: two is the shortest run that can
  show flatness at all, so there is nothing to tune between 1 (a single flat step is as
  likely to be two close modes) and 2.
- Measured on one plate identification per seed, three seeds (the plateau detector
  returns 5 of 12 on all three).  The closed-loop numbers are one seed each, the
  harnesses' defaults.
- The `in-model` column under `tf_update` is the error against the LAST
  re-identification from the control drive, which is what the law would next solve
  against.  It is a diagnostic of the live loop, not of the floor rule.
