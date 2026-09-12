# Systematic Control-Law Comparison — Session Summary, 12 September 2026

Repository: `rattlesnake-vibration-controller`, branch `control-law-development-2026-09`
System: 6 drives, 12 responses, 8 control channels (6-12-8), linear plant,
`sdynpy_frame6x12_system.npz`, 100–1000 Hz, modal damping 0.70–1.19 % (median 0.83 %).

---

## 1. What changed today

The work moved from ad-hoc single runs to a **controlled experiment**: four
control laws, each with the drive-coherence cap on and off, on one plant, with
every other setting held fixed and machine-verified.

Three tools and eight profiles were built to make that possible, and five
separate defects were found and fixed along the way — four of them
methodological rather than software.

---

## 2. Tooling built

### `examples/sixdrive12resp/code/score_runs_sdynpy.py`

Scores saved runs through SDynPy's `RandomVibTest` (`sdynpy.doc.sdynpy_vibration_test`),
which was discovered to be already present and unused. Two obstacles had to be
solved:

- **Runs carry no time data.** Every run in the repository was saved with
  `time_samples = 0`, so `RandomVibTest.load_rattlesnake_streaming_data` raises.
  However the scoring methods touch `self.time_data` only to take `len()` for
  figure bookkeeping; the real work iterates `self.cpsd`. Pre-populating `cpsd`
  from the file's response CPSD and passing a length-1 placeholder drives the
  whole suite without any time data. `plot_kurtosis`, `plot_time_histories` and
  `plot_cpsd_time_subset` genuinely need it and remain unavailable until runs
  are saved with streaming enabled.
- **A law aiming at the achievable target must not be judged against the flat
  spec**, which is provably unreachable with 6 drives and 8 control channels.
  Every run is therefore scored twice — contractual (flat spec) and physical
  (achievable target) — plus a third *reachability* scoring described below.

The achievable target is computed per run from **that run's own measured FRF**
(`--recompute-floor`), not from a saved file. This matters: scoring an August
run against a September projected target shifted channel 8X+ by 3.6 dB rms.
`--floor-decimate 4` solves every fourth line and interpolates in dB, taking the
6-12-8 case from 5 m 40 s to 1 m 32 s while moving the result by 0.01 dB.

`--save` writes per-run `.npz` plus `summary.csv` so a 25-minute sweep never has
to be repeated to ask a new question.

### `examples/sixdrive12resp/code/check_run.py`

A ten-second verdict on a freshly saved run: converged, saved below test level,
saved while overshooting, or level-correct with implausible error. Imports only
netCDF4 and numpy so it starts instantly at the rig. This exists because of
defect 4 below.

### `examples/sixdrive12resp/code/make_achievable_spec.py`

Generates a specification `.mat` whose target is the achievable diagonal rather
than a flat spec, structurally identical to `flat_spec_frame6x12.mat` —
`f` (1, 2561), `cpsd` (8, 8, 2561) complex, diagonal, same in-band line set,
exact zeros out of band. Built and verified; **not yet used**, reserved for
phase two.

### `examples/sixdrive12resp/results/case/profiles/` — eight profiles

`profile_01_matchtrace_capon.xlsx` … `profile_08_congruence_capoff.xlsx`, each a
copy of the working profile differing only in the control law, its parameters,
and the settings corrected below. Launch with:

```
make launch-rattlesnake-profile PROFILE=examples/sixdrive12resp/results/case/profiles/profile_NN_....xlsx
```

which skips both startup dialogs and reads every setting from the file — the
fix for settings silently drifting in the GUI.

---

## 3. Defects and errors found

### 3.1 Sixteen runs were never comparable (methodological)

The first full sweep scored 16 runs as one experiment. They were not:

| Plant | `hardware` | Runs |
|---|---|---|
| linear `sdynpy_frame6x12_system.npz` | 6 | 4 runs |
| shifted `…_shifted_allmodes.npz` | 7 | 3 runs (all the coherence-cap studies) |
| nonlinear `…_nonlinear_allmodes.npz` | 7 | the rest |

`update_tf_during_control` also differed between them. Nothing in the output
revealed this, so the error was invisible. The harness now captures full
provenance and prints **"THESE RUNS ARE NOT DIRECTLY COMPARABLE"**, naming the
setting and which runs hold each value.

### 3.2 A conditioning conclusion that was wrong (analysis error)

An apparent correlation between FRF conditioning and the reachability limit was
reported, based on the "gap" between flat-spec and achievable-target rms. That
inference was invalid — **rms differences do not subtract**, because the measured
response can sit far from both targets in different directions. The true
reachability limit, measured directly, is roughly double to triple what the gap
suggested:

| Run | gap (as reported) | true reachability |
|---|---|---|
| match trace 0db | 1.18 dB | **2.37 dB** |
| match_trace_pinv_law_8cont | 1.32 dB | **3.92 dB** |
| fullnonlinearminus6dbspectra | 1.59 dB | **3.73 dB** |

With the correct metric the conditioning correlation does not hold, and the
clusters that suggested it turned out to be the three *shifted-plant* runs.
**The conditioning hypothesis is retired.**

### 3.3 Segfault on multi-file sweeps (software)

Each run was being opened through several independent netCDF4 handles. HDF5 does
not tolerate closing one handle while others on the same file are live, and the
interpreter died on the *next* file in the list (exit 139). Every read now shares
a single `Dataset` per file. Also fixed: a Qt5/Qt6 class clash in the Anaconda
environment (matplotlib forced to Agg before SDynPy imports pyvista), and
system-identification files crashing the sweep instead of being skipped.

### 3.4 A run that looked good on screen and was garbage in the file

Run 01 converged visibly, but the saved spectral file sat **28.6 dB below
specification** with a drive trace seven orders of magnitude below a working run.
The FRF was fine — reachability came out a clean 2.29 dB — so system ID had
worked; the spectral snapshot had simply been taken before the test level ramped.
Nothing in the file announces this. It would have been repeated across all eight
runs. `check_run.py` was written in response.

### 3.5 Stale profile settings (process)

The profile on disk disagreed with what was actually being run in five places,
all of which had been overridden in the GUI:

| Setting | profile had | actually used |
|---|---|---|
| Update System ID During Control | N | on |
| Frames in CPSD | 10 | 20 |
| Control averaging coefficient | 0.05 | 0.04 |
| Time per read / write | 1 | 0.25 |
| System ID Averages | 5 | 20 |

Worse, runs 01 and 02 differed in **six system-ID settings** because one was
configured in the GUI and the other loaded from a profile — including
`sysid_level` at 0.01 vs 1.0 V RMS, a factor of 100. The identified FRF's median
condition number came out 6475 vs 25928 as a result. The system-ID block now
counts toward comparability, and run 01 was redone.

### 3.6 `optimal_diagonal_control` has no startup level cap (open)

`match_trace_pseudoinverse` and `match_diagonal_congruence` both take
`startup_test_level_cap_db` (default −9 dB). Neither optimal-diagonal law has
any equivalent: they solve the buzz baseline against the full specification and
command it immediately, which is why run 03 started high. Worked around by
ramping the test level. **Not yet fixed** — deliberately deferred so a law is not
changed partway through the comparison.

---

## 4. Results so far

Three of eight runs complete. Provenance verified clean across all three: same
plant, sample rate, spectral settings, control averaging (Exponential 0.04),
FRF update on, and all fourteen system-ID settings.

| Run | Law | Cap | rms dB | % lines out | cond(H) |
|---|---|---|---|---|---|
| 01 | `match_trace_pseudoinverse` | 0.95 | 8.22 | 27.5 | 6976 |
| 02 | `match_trace_pseudoinverse` | off | 7.21 | 21.1 | 25928 |
| 03 | `optimal_diagonal_control` | 0.95 | **5.62** | **11.2** | 3567 |

### The coherence cap costs about 1 dB

Runs 01 and 02 are a genuine controlled pair — only the cap differs. Removing it
gains **1.0 dB rms and 6.4 points of lines-out**. This confirms, properly
controlled, a 1.72 dB effect previously inferred from the shifted-plant runs.

The cap helps least where it seems to matter most: 13X+ and 14X+ barely move,
while the well-controlled channels improve substantially (8X+ by 2.2 dB, 12X+
from 12.2 % to 1.9 % out). **The cap costs precision on the channels that are
otherwise fine**, and does nothing for the two that are struggling.

### `optimal_diagonal_control` is clearly ahead

On the six controllable channels it roughly halves match_trace's error, with
mean errors within a third of a dB of zero rather than the −1 to −3 dB bias
match_trace shows throughout:

```
          optdiag capon        match_trace capoff
8X+    +0.12 mean  1.39 rms      -1.50 mean  3.87 rms
12X+   +0.35 mean  1.25 rms      +0.07 mean  2.74 rms
```

### Channels 13X+ and 14X+ are reachability-limited, as predicted

Every law leaves them at −5 to −7.5 dB mean with 26–48 % of lines out. This is
the structural sacrifice the achievable-response predictor forecast analytically
weeks ago, now confirmed by measurement across three different control laws.

### A caution on run-to-run variance

Median cond(H) came out 3567, 6976 and 25928 across three identifications of the
*same deterministic linear plant* with identical system-ID settings — a factor of
seven. This is why the achievable floor must be recomputed per run, and why
differences between laws smaller than a few tenths of a dB should not be
over-read.

---

## 5. Commits

| Commit | Time | Subject |
|---|---|---|
| `e60faa46` | 16:15 | Add `make_achievable_spec` |
| `98a13569` | 16:31 | Capture run provenance; refuse silent bad comparisons |
| `8337d90c` | 17:03 | Add `check_run` |
| `5fd78eb8` | 17:07 | Add eight run-ready profiles |
| `d37b9ddd` | 17:29 | Profiles: time per read/write 0.25 |
| `bc04dffd` | 17:51 | System-ID settings count toward comparability |

Earlier commits `2208c33e`, `2a32fa2e`, `2342a605`, `b43635c8` cover the path
updates after the `~/Code` move and the scoring harness itself.

---

## 6. Remaining

**Immediate — finish the matrix.** Runs 04–08: optdiag cap off, optdiagfast on
and off, congruence on and off. Then:

```
cd examples/sixdrive12resp/code
python score_runs_sdynpy.py ../results/runs/run??_*_spec.nc4 \
    --recompute-floor --restrict-rcond 1e-3 --floor-decimate 4 \
    --figures ../results/figures/law_comparison \
    --save ../results/analysis/law_comparison
```

**Phase two — the achievable target.** Generate the spec with
`make_achievable_spec.py` from one system ID, repoint the profiles, repeat the
eight runs. This removes reachability from the comparison so the residual is
tracking behaviour alone. Requires a `--flat-spec` option in the harness so those
runs can still be scored against the original target.

**Open items:**

- Add a startup level cap to both optimal-diagonal laws (§3.6).
- Watch whether runs 03 and 04 differ at all: the coherence cap applies only to
  SDP-refined bins, bounded by `max_bins_per_update = 20`. If they are nearly
  identical, raise the bin budget rather than concluding the cap is irrelevant.
- Three stray `.nc4` files sit loose in `control_laws/` and would extend the old
  coherence-cap comparison from three points to five.
- No `match_diagonal_congruence` run exists in the archive; runs 07–08 will be
  its first saved evidence.
- Streaming files total roughly 90–180 MB per run; the eight will come to
  several hundred MB.
- Nothing is pushed. The branch is 15+ commits ahead of `githunter333/main`.

**Longer term:** a fourth system file with the same modes but ~3 % damping would
make damping a genuine single variable; the nonlinear plant with
`RATTLESNAKE_NONLINEARITY_STRENGTH` swept 0 → 1 → 2 remains the cleanest
plant-≠-model experiment available, and needs streaming enabled so kurtosis and
multiple coherence become usable as nonlinearity instruments.
