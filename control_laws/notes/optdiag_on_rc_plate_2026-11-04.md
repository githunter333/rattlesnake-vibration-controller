# optimal_diagonal_control_fast on the 12-drive RC plate (2026-11-04)

The band-average study's 12-drive / 48-control RC plate example, run through this
repository's current optimal diagonal fast law offline.

Script: `control_laws/tests/optdiag_on_rc_plate.py`
Law:    `control_laws/optimal_diagonal_control_fast.py` at `control-laws-on-dev`
Plant:  `rattlesnake-band-average`, `studies/rc_plate` at `732f483`
Env:    conda env `rattlesnake` (cvxpy 1.8.2), `PYTHONPATH=src` from the repo root

```bash
PYTHONPATH=src python control_laws/tests/optdiag_on_rc_plate.py --df 1 --cycles 15
```

Raw console output of the three runs quoted below:
`optdiag_on_rc_plate_2026-11-04_runs.txt`.

## Which law this is

`control_laws/optimal_diagonal_control_fast.py` at `control-laws-on-dev`
(`9a141e82`) is byte-identical to the band-average repository's archived copy in
`control_laws/hunter_9da1564/`, so this is the same code that produced that
study's `nl8n_hu_odf_tfu` row.  `optimal_diagonal_control.py` is the SDP parent
whose per-bin solve this subclass overrides, not an earlier generation;
`temp/optimal_diagona_controlpy.py` is a stale scratch copy.

## Reconstructing the plant and the specification

The study's two large inputs are GitHub Release downloads and are not in that
checkout.  Both were rebuilt from files that are, and each reconstruction was
checked against a number that repository documents:

| quantity | built from | check |
| --- | --- | --- |
| plant H, 48 x 12, g/lbf | `C (jwI - A)^-1 B + D` from `rc_plate_state_space.npz`, the 48 rows with `channel_unit == 'g'` | a modal sum over `rc_plate_test_shapes.npy` (rigid lifted to 1 Hz, 31 modes below 8192 Hz) agrees to **8e-16 median** relative error, 20-4000 Hz; the README claims 5e-15 against the Release FRF |
| specification, 48 x 48 CPSD, g^2/Hz | `H_barc G_ff H_barc^H`, modal accelerance through `barc_{response,force}_shapes.npy`, `G_ff = 1e-3 lbf^2/Hz` flat 20-4000 Hz, 59 modes at 1 % | `plot_spec.py` records "101Z+ ... 18.5 g of 7.7 to 45.5"; this gives **101Z+ 18.46 g, range 7.66-45.53 g** |

The first 12 rows of the state-space file's `C`/`D` are the force passthrough
Rattlesnake requires (identity in `D`); only the 48 accelerometer rows are
control responses.

## The interface that actually mattered: a MEASURED FRF, not an exact one

`compare_optdiag_frame.py` hands the law the frame plant's exact analytic FRF.
Doing the same here is wrong by six orders of magnitude, and this is the single
most important bridging decision in the script.

| FRF | sigma_min/sigma_max over the band | pinv at rcond 1e-15 |
| --- | --- | --- |
| exact analytic plant | median 5.2e-05, min 6.9e-08 | drive_lbf2 **6.65e+10** |
| 50-average H1 estimate | median 5.3e-03, min 1.3e-04 | drive_lbf2 **8.42e+04** |
| the study's streamed `n_pinv_noclip` row | | drive_lbf2 **8.36e+04** |

Rattlesnake's `pseudoinverse_control` takes `rcond = 1e-15`, i.e. no truncation,
and the study ran it with empty parameters.  The identification noise floor, not
the structure, is what bounds the drive any law can ask for here -- exactly what
`_effective_drive_rcond`'s docstring says about scaling off sigma_min.  So the
script simulates the study's identification (50 averages, flat uncorrelated
random at 4 lbf RMS per drive, H1, 1e-5 g^2/Hz response noise) and hands the law
that estimate, its Gyy, its Gvv and its multiple coherence.  Scoring is always
against the exact plant.  `--frf exact` reproduces the exact-H case; it is not
what the rig does.

With `--tf-update` the FRF is re-estimated **from the control drive**, because
that is what Rattlesnake does: `random_vibration_sys_id_data_analysis.py` hands
a type-2 law `self.control_frf` when "Update System ID During Control" is on,
and `spectral_processing.py` computes it as
`Gyv @ pinv(Gvv, rcond=1e-12, hermitian=True)`.  That matters because a
rank-deficient drive truncates its own identification.

## Results, on the study's own 1 Hz grid, linear plant with noise

`--df 1 --cycles 15`, 20-frame CPSD estimates, FRF held fixed.  3981 specified
lines, 48 channels, 12 drives.  `err` is narrowband RMS dB; `band_db` is the
study's own 1/6-octave score.

| budget | err | band_db | lines refined | ms/call | drive_lbf2 | drive rank | median \|coh\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| defaults (20 lines/call) | 3.89 | **1.55** | 340 / 3981 | 4977 | 5.52e+04 | 12 | 0.53 |
| full band (400 lines, 500 ms) | 6.51 | **1.85** | 1848 / 3981 | 5740 | 1.26e+04 | 11 | 0.78 |

Best achievable on this plant, 40 lines sampled: **0.44 dB RMS median**, 2.75 dB
worst.  So the specification is very nearly reachable here -- 48 controls on 12
drives, but the specification was generated from 12 BARC forces, so it lies
close to a rank-12 image.  Nothing in these numbers is a reachability limit.

The other two configurations, same grid and cycles, default budget:

| configuration | err | band_db | drive_lbf2 | fast/SDP | drive rank | median \|coh\| | ch > 6 dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FRF fixed (case 1) | 3.89 | 1.55 | 5.52e+04 | 340 / 0 | 12 | 0.53 | 21 of 48 |
| in-control FRF update (`--tf-update`) | 4.11 | 1.81 | 8.27e+04 | 340 / **300** | 12 | 0.51 | **1 of 48** |
| exact analytic FRF (`--frf exact`) | 2.92 | 1.17 | **1.77e+11** | 340 / 0 | 7 | 0.86 | 0 of 48 |

Beside the study's `n_` rows (same linear plant with noise), `band_db` / drive:

    this law, defaults    1.55 / 5.5e4        n_pinv_noclip        1.65 / 8.4e4
    this law, full band   1.85 / 1.3e4        n_ba_front_noclip    1.64 / 1.6e4
                                              n_ba_r70_noclip      1.92 / 4.4e3
                                              n_at_noclip          2.52 / 3.1e3

On the banded score it is competitive.  It buys that at pinv-like drive, an
order of magnitude more than the band-average law at the same error.

### Three things that are not competitive

**1.  It gets worse every cycle, monotonically, and refinement is why.**

    err by cycle, defaults    3.31 3.34 3.37 3.40 3.44 3.49 ... 3.84 3.89
    err by cycle, full band   3.79 3.97 4.17 4.39 4.61 4.84 ... 6.36 6.51

`response_trim_gain` is 0 by default, so the measured response is never read:
this is not a feedback loop drifting, it is the scheduler covering more lines
and each newly refined line making the true response worse.  The budget that
refines more lines ends further off.  At `--df 8`, where 15 cycles cover
essentially the whole band, it saturates at 7.93 dB by cycle 3 and then stops
moving -- a fixed point, not a divergence.

**2.  The error is in the law's own model, not in the FRF it was given.**

At `--df 8`, full coverage: error against the law's own `H_meas` is **7.93 dB**
and against the true plant **7.93 dB**.  Identical.  The law is not being
betrayed by an 11 %-accurate FRF; it is converging to a point 7.9 dB off its own
target while the floor is 0.44 dB.

**3.  Five seconds per control call, against a ~1 s control cycle.**

4977 ms at 20 solves/call and 5740 ms at 110 -- about 5 s of fixed per-call cost
at this problem size, independent of the solve budget, in the non-solve path
over 3981 x 48 x 48.  The study's streamed run recorded **median 8.49 s** per
control call (p90 9.07, max 12.88) against 0.47 s for pinv and 0.37 s for
band-average, so this reproduces independently.  It is also why that run made
only 28 control calls in 205 s, and therefore refined at most ~560 of 3981
lines.

<a id="rc-plate-drive-rcond"></a>

## Why: `_effective_drive_rcond`'s gap rule inverts on this plant

> **FIXED 2026-11-04**, after this was written: the gap rule has been replaced by
> `_structural_floor`'s three detectors, and on this plant the in-model error goes
> from 6.56 to 3.01 dB with 0 of 48 channels over 6 dB.  The diagnosis below stands
> as written -- see [structural_floor_rewrite_2026-11-04.md](structural_floor_rewrite_2026-11-04.md)
> for the replacement criterion, the before/after on every plant in the repo, and
> one correction to the account below: `_band_structural_rank` had the same defect,
> and on the frame plants it was the `svd_min_keep` floor built on it that masked
> the band threshold's error.  Both failed together here.

One parameter at a time, `--df 8`, full-band budget, 10 cycles:

| variant | err/model | err/truth | band_db | drive_lbf2 | drive rank | \|coh\| | effective rcond |
| --- | --- | --- | --- | --- | --- | --- | --- |
| defaults | 7.93 | 7.93 | 2.77 | 2.17e+03 | 2 | 1.00 | **0.421** |
| `drive_rcond: 0` | 2.81 | 3.06 | 1.18 | 1.74e+05 | 4 | 0.61 | - |
| `bm_rank: 12` | 7.93 | 7.93 | 2.77 | 2.17e+03 | 2 | 1.00 | 0.421 |
| `drive_rcond: 0` + `bm_rank: 12` | 2.76 | 3.00 | 1.17 | 1.77e+05 | 11 | 0.53 | - |
| `error_domain: db` | 2.94 | 3.27 | 1.55 | 9.46e+04 | 12 | 0.50 | 0.421 |

`bm_rank` changes nothing, so the factorization rank is not the binding
constraint.  `drive_rcond` is everything: the effective threshold comes out at
**0.421**, which keeps **2 of 12** drive directions.

The reason is the gap rule itself.  `_effective_drive_rcond` takes
`drive_rcond` as a multiple of "the smallest structural direction", found as the
largest gap in the median singular-value spectrum.  Median sigma_k/sigma_1 over
the 996 specified lines at `--df 4`:

    exact plant   1.0000 0.5144 0.2104 0.0692 0.0516 0.0236 0.0061 0.0021 0.0008 0.0005 0.0004 0.0001
      gaps          1.94   2.44   3.04   1.34   2.18   3.90   2.91   2.64   1.52   1.20   8.35
      -> largest gap 8.35x at 11->12, reference 0.0004, threshold 0.0009, keeps 8 of 12

    H1 estimate   1.0000 0.5152 0.2109 0.0695 0.0523 0.0252 0.0140 0.0104 0.0087 0.0075 0.0064 0.0054
      gaps          1.94   2.44   3.04   1.33   2.08   1.80   1.36   1.19   1.16   1.17   1.18
      -> largest gap 3.04x at 3->4, reference 0.2109, threshold 0.4218, keeps 2 of 12

The rule assumes the identification noise floor is an isolated step BELOW the
structural directions -- which is what the 6-drive frame it was tuned on looks
like (medians 1.000, 0.319, 0.084, 0.032, 0.015, 0.0017; a clean 8.9x gap at
5->6; reference 0.0152; 2.0x keeps all 5 structural directions).

The RC plate is full rank 12 with a four-decade smooth decay.  The
identification noise does not sit below the structure as one isolated value --
it floors sigma_7 through sigma_12 into a flat plateau (0.0140 down to 0.0054,
adjacent ratios 1.16 to 1.80), so there is no noise-floor gap left to find.  The
largest remaining gap is a genuine STRUCTURAL feature near the TOP of the
spectrum, 3.04x at sigma_3 -> sigma_4, and the rule takes sigma_3 as "the
smallest structural direction".  On the exact FRF the same rule works, because
the near-null directions are still there to be found -- but the exact FRF is not
what the rig hands the law.

So the gap rule is load-bearing in the wrong direction here, and it fails
silently: the law prints its effective rcond and nothing else flags that 10 of
12 drive directions were discarded.

What the price of that truncation actually is, from `achievable_diagonal` with
`restrict_rcond` (24 lines, `--df 8`):

    unrestricted            RMS 0.54 dB median,  3.63 worst
    restrict_rcond 2.5e-2   RMS 2.15 dB median,  7.12 worst
    restrict_rcond 1e-3     RMS 0.89 dB median,  3.40 worst

`error_domain: db` is the best-balanced variant measured: 2.94 dB in-model at
9.46e4 lbf^2 with drive rank 12 and median |coh| 0.50 -- near `drive_rcond: 0`'s
accuracy at half its drive, and the only variant whose drive stays full rank, so
a live identification through it stays well-posed.  None of this was applied to
the law; it is diagnosis.

## Drive coherence reaches 1.0 at 12 drives -- corroborated on a second rig

The module docstring's claim, and `optimal_diagonal_history.md#fast-coherence-claim`'s
"established at 12 drives", reproduce here on an independent 12-drive plant with
48 responses: with the FRF held fixed and the band fully covered, median
pairwise drive |coh| is **1.00** and the commanded drive's median effective rank
is **2 of 12**.  That is the precondition the docstring warns about, and on this
rig it is reached through the `drive_rcond` collapse above rather than through the
unconstrained solve alone -- `drive_rcond: 0` gives |coh| 0.61, not 1.00.

It matters because Rattlesnake's in-control FRF update estimates H from the
commanded drive: `Gyv @ pinv(Gvv, rcond=1e-12, hermitian=True)`.  A rank-2
reference CPSD truncates to rank 2, so 10 of the 12 columns of the new FRF come
back wrong and the next solve is made against them.  That is a self-reinforcing
loop, and the study's streamed `nl8n_hu_odf_tfu` row is where it ends:
band_db 155 dB, drive_lbf2 5.86e+27.

### The exact-FRF run corroborates the gap mechanism from the other side

Case 3 hands the law the exact analytic plant.  The gap rule then finds the
8.35x gap at sigma_11 -> sigma_12, takes reference 0.0004, sets the threshold to
0.0009 and should keep 8 of 12 directions.  The observed commanded drive has
median effective rank **7**, range 4-12 -- against **2** on the measured FRF,
where the rule predicts 2.  The drive rank tracks the gap rule's own arithmetic
in both cases, which is about as direct as this gets without instrumenting the
law.

It also shows why the exact-FRF experiment is the wrong one: it is the most
accurate run of the three (err 2.92 dB, band_db 1.17 dB, no channel over 6 dB)
and completely unusable, at drive_lbf2 1.77e+11 -- 212,000 lbf on the worst
drive.

### The in-control FRF update: the safety rule fires, and a 3.4e+15 excursion

With `--tf-update`, **300 of 640 solves go to the coherence-capped SDP**, so the
`_h_moved` safety rule does what it is for once the FRF genuinely moves.  Note
it did NOT fire when the FRF was re-identified from an independent flat
excitation rather than from the control drive: a fresh 50-average H1 estimate
moves well under `frf_update_threshold` 0.5 in relative Frobenius norm.  It is
the control-drive estimate, poisoned by the drive's own rank, that moves H
enough to trip the gate.

The SDP path is the better-behaved one on the error distribution: **1 of 48
channels over 6 dB**, against 21 with the FRF held fixed, and drive rank stays
12.  But the drive is not safe.  `drive_lbf2` by cycle:

    1.21e+04  8.10e+04  8.08e+04  3.37e+15  8.02e+04  8.14e+04  ... 8.27e+04

One cycle, eleven orders of magnitude, then a recovery.  The true error barely
registered it (3.43 -> 4.09 dB), because the excursion went into near-null
directions -- which is exactly why it would not be caught by a response-based
abort.  It recovered here only because the next cycle's re-identification
happened to be well conditioned.  On the rig, ramping level on a softening
plant, it did not: `nl8n_hu_odf_tfu`, drive_lbf2 5.86e+27.

So this configuration should be treated as having no usable drive bound on this
plant, independent of how good its band error looks.

## Limitations

- The offline loop is 15 control cycles at one test level.  The study's streamed
  run was 28 control calls across a -12 to 0 dB level ramp on the 8 %-softening
  plant, so a monotone trend here is not the same object as that run's number.
- `--df` coarser than 1 Hz changes the answer, because the scheduler's line
  budget then covers a larger fraction of the band per cycle.  Quote 1 Hz.
- The achievable floor is computed on a sample of lines (`--floor-lines`), not
  all 3981.
- The attribution table and the model-vs-truth comparison were run at `--df 8`,
  where 15 cycles cover essentially the whole band.  The effective `drive_rcond`
  (0.421) and the singular-value spectra are grid-independent, but the error
  numbers in that table are the full-coverage ones, not the 1 Hz ones.
- The 3.37e+15 drive excursion is one cycle of one seed.  It has not been swept
  over seeds, so its rate is unknown; what the run establishes is that it
  happens at all, and recovers without any response-side signal.
