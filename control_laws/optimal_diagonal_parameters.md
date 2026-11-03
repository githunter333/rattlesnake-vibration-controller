# Parameters of the optimal diagonal laws

*Generated from the laws' parameter tables by `control_laws/tests/make_parameter_reference.py`;
do not edit by hand.*

Three laws read these: `optimal_diagonal_control` (the SDP law),
`optimal_diagonal_control_fast`, and `octave_band_switching_control`.  Each
subclass accepts everything its parent does, plus the parameters listed under it.
Put them in the environment's **Control Parameters** box (or cell 14 of the
environment sheet of a profile).

## Two forms

**Named (use this).**  One `name: value` per line, or separated by `;`.  Names are
case-insensitive, `#` starts a comment, an empty value means "the default".

```
max_bins_per_update: 10
max_drive_coherence: 0.8
response_trim_gain: 1.0      # trim on
```

A name the law does not have, or a value it cannot read, stops the test from
starting with a message naming the line and the nearest valid name.
A string with no `:` anywhere is read as the comma form below, so a named line whose colon
is missing (`reg = 1e-5`, `max_bins_per_update 10`) is refused with a message as well, not run
on defaults.

**Comma-separated (the original form).**  Positional, by the *field* number in the
tables below; an empty field means "the default"; a field that cannot be read
keeps its default with a warning and the later fields are still read.  Every
archived profile uses this form, and it keeps working.

```
1e-6,0.5,20,1.0,0.95,6,-9.0,1,2,1e-2
```

Field numbers are 1-based.  The two forms must not be mixed in one string.

## Choosing the time budget

`max_solve_ms_per_update` defaults to 0 (off): only `max_bins_per_update` limits the work
per call.  Every behaviour change in these laws ships off by default, so archived profiles
reproduce exactly.  But the line count alone does not bound a call.  At 10 drives x 14
controls the SDP law's default 20 lines per call took 1.1 to 1.5 s on its worst call, longer
than a 1 s control cycle.

Starting points, measured on the 10x14 frame plant (`tests/compare_optdiag_frame.py`, 15 cycles,
FRF held fixed, 20-frame CPSD estimates), on the Mac that runs the rig software, 2026-11-02:

| law | setting | final error | lines solved | ms per call, median / worst |
| --- | --- | --- | --- | --- |
| SDP | defaults (20 lines per call) | 0.12 dB | all 180, by cycle 7 | 16 / 1508 |
| SDP | `max_bins_per_update: 400`, `max_solve_ms_per_update: 500` | 0.63 dB | 109 after 15 cycles, still converging | 490 / 512 |
| fast | defaults (20 lines per call) | 0.23 dB | all 180, by cycle 7 | 20 / 105 |
| fast | `max_bins_per_update: 400`, `max_solve_ms_per_update: 500` | 0.23 dB | all 180, during system ID | 18 / 20 |

The errors and the fast law's line counts reproduce the 2026-10-30/31 measurements.  The times,
and the budgeted SDP row (0.34 dB and 148 lines then), depend on the machine, because a
time budget buys fewer lines on a slower one.

For the fast law at 10 drives, set `max_bins_per_update: 400` and `max_solve_ms_per_update: 500`.
For the SDP law, pick a budget that fits the control cycle (about 500 ms) and expect 15 to 20
cycles to converge; starting with the fast law and finishing with the SDP law is an option.
Fully converged, the SDP law is the more accurate (0.12 against 0.23 dB at the same drive
level).

## Drive coherence

In every one of these laws, and in the test and comparison scripts, "drive coherence" is the
magnitude |X_ij| / sqrt(X_ii X_jj) of a pair of drives: 0 for independent drives, 1 for fully
dependent ones.  That is what `max_drive_coherence` caps.  The squared measure is always
called coherence squared (gamma^2); a cap of 0.95 is gamma^2 = 0.9025.

## optimal_diagonal_control (and, through it, the other two)

| name | field | default | meaning |
| --- | --- | --- | --- |
| `reg` | 1 | `1e-06` | Tikhonov-style weight on ||X||_F |
| `frf_update_threshold` | 2 | `0.5` | relative FRF movement (0-1) that triggers re-solving an already-refined line |
| `max_bins_per_update` | 3 | `20` | hard cap on line solves per call |
| `error_threshold_db` | 4 | `1.0` | do not refine a line whose buzz solution is already within this many dB of target |
| `max_drive_coherence` | 5 | `0.95` | cap (0-1) on the magnitude of pairwise drive coherence, |X_ij|/sqrt(X_ii X_jj) (so 0.95 is gamma^2 0.9025), between SDP-solved drives; 1.0 disables it |
| `startup_test_level_cap_db` | 7 | `-9.0` | ceiling on the very first command, dB relative to the spec trace (0 = full match) |
| `error_domain` | 8 | `'linear'` | 'linear' (0) or 'db' (1) objective; legacy strings give 0 or 1, the named form also 'linear'/'db' |
| `n_irls_passes` | 9 | `2` | reweighting passes of the dB objective |
| `drive_rcond` | 10 | `2.0` | fast law: drive directions below this multiple of the plant's singular-value spread are dropped (negative = absolute ratio) |
| `response_trim_gain` | 11 | `0.0` | fraction of the measured response error folded into the target each cycle (0 = trim off) |
| `response_trim_limit_db` | 12 | `3.0` | total authority of the trim, +/- dB |
| `response_trim_step_db` | 13 | `0.5` | most the trim may move in one cycle, dB |
| `response_trim_deadband_db` | 14 | `0.5` | measured errors smaller than this are left alone, dB |
| `response_trim_gate_db` | 15 | `0.5` | how close the solver must have come to its own target before the leftover error counts as model error, dB |
| `max_solve_ms_per_update` | 21 | `0.0` | time budget per call in ms of line solving (0 = off; the line cap alone applies) |

## optimal_diagonal_control_fast adds

| name | field | default | meaning |
| --- | --- | --- | --- |
| `bm_rank` | 6 | `4` | factorization rank of the fast solve; the SDP's constrained solutions rarely need rank above 3-4 |
| `svd_select` | 16 | `0` | 0 = one singular-value cut for the whole band, 1 = each line picks its own |
| `svd_gap_min` | 17 | `3.0` | per-line selection: smallest ratio between adjacent sigmas that marks a line's structural floor |
| `svd_floor` | 18 | `1e-06` | sigma/sigma_1 below this is never kept and never counted as a gap |
| `svd_weak_max` | 19 | `0.05` | a gap counts as a structural floor only if the direction below it has sigma/sigma_1 at or under this |
| `svd_min_keep` | 20 | `0` | fewest directions any line keeps (0 = auto: structural rank - 1; -1 = N - 1; 1 = pre-2026-10-24 behaviour) |
| `cap_baseline` | 22 | `0` | 1 = apply max_drive_coherence to the buzz baseline that unsolved lines sit on |
| `db_coherence_guard` | 23 | `1` | with the dB objective, a fast solution that breaks the coherence cap: 1 = re-solve with the SDP, 2 = shrink its cross terms, 0 = off |
| `warm_start` | 24 | `1` | 1 = start each fast solve from the line's current drive when that scores better than the buzz seed |
| `gate_reference` | 25 | `1` | what the fast/SDP movement gate compares the FRF with: 1 = the previous call's, 0 = each line's at its last solve |

## octave_band_switching_control adds

| name | field | default | meaning |
| --- | --- | --- | --- |
| `frequency_spacing` | 6* | `None` | Hz per FFT line (sample_rate / samples_per_frame); REQUIRED for octave mode to engage |
| `switch_level_db` | 7* | `0.0` | test level (dB) at or above which octave-band mode engages |
| `octave_fraction` | 8* | `6` | fractional-octave denominator: 6 = 1/6 octave, 3 = 1/3 octave |
| `octave_fmin` | 9* | `None` | lowest frequency to band, Hz (default: lowest line with a nonzero target) |
| `octave_fmax` | 10* | `None` | highest frequency to band, Hz (default: highest line with a nonzero target) |

`*` The field number counts only the **non-empty** fields of the comma-separated
string, and these positions are also read by `optimal_diagonal_control` as
`bm_rank`, `startup_test_level_cap_db`, `error_domain`, `n_irls_passes` and
`drive_rcond`, so a comma-separated string with six or more fields sets both.  It
is kept that way so archived strings give the results they always did, and the law
prints a warning.  **Use the named form for this law.**
