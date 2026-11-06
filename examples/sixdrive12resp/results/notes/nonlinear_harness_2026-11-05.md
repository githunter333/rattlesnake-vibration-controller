# Running the bounded-softening cases unattended: what it takes, and what blocks it

**2026-11-05 · Hunter · branch `fix/drive-subspace-structural-floor-clean` (not pushed)**

Goal: runs 53 and 54 — `optimal_diagonal_control_fast`, trim OFF and ON, on the
bounded-softening plant. They are the only optimal-diagonal-fast measurement
ever planned against a nonlinear plant and were never made. The fast law's
entire evidence base is the linear frame (runs 5, 6, 15).

## 1. Why they were never run

`components/sdynpy_nonlinear_system_virtual_hardware.py` did not survive the
move to `src/rattlesnake/`. It is on `main`, `optimaldiaglaws` and
`control-law-development-2026-09`, and absent from `control-laws-on-dev` and
from this branch. Without it the profiles cannot load at all.

Ported forward as `HardwareType.SDYNPY_NONLINEAR = 9` with the physics
unchanged. The port announces itself correctly on the bounded-softening file:
`BOUNDED SOFTENING alpha=0.350 (27/33 modes softened)`, and the environment
reports F=2049, matching the historical line count.

## 2. Six incompatibilities, three of them silent

| | fix |
|---|---|
| hardware enum: profiles name 7, which is `SDYNPY_FRF` here | repoint to 9 |
| environment sheet: current layout inserted six rows | relay **by label**, refuse on a missing label |
| `Noise Averages = 0` rejected by current code | set to 10 |
| COLA/CPSD overlap changed from percent to **fraction** | 50 → 0.5 |
| control-channel row spans one column per channel | carry whole rows |
| Control Averaging (Exponential, 0.04) | **unrecoverable** — the current environment hardcodes `AveragingTypes.LINEAR` |

The silent ones mattered most:

- **`metadata.hardware_type`.** `SDynPySystemMetadata.__init__` passes its own
  module's `HARDWARE_TYPE`, so the nonlinear subclass inherited
  `SDYNPY_SYSTEM`. `acquisition.py` and `output.py` dispatch on
  `metadata.hardware_type`, so a profile naming type 9 would have loaded, run
  and logged normally **on the linear integrator** — a nonlinear test that was
  quietly linear. Caught by printing the resolved acquisition class rather than
  trusting the file.
- **Overlap units.** The validator is `0 <= x < 1` while its error message says
  "percentage". A stored value below 1 passes and means something else.
- **The control-channel row.** Carrying only column 2 turned eight control
  channels into one; it surfaced only because the specification was then
  rejected as `(2049,8,8)` against an expected `(2049,1,1)`.

## 3. The blocker: the integrator is ~8x slower than real time

The validation run (profile 42, optimal_diagonal trim OFF, against run 51's
recorded 5.61 dB / 19.2 % out at −18 dB) produced **34 seconds of data in 275
seconds of wall clock**. No starvation warnings and 48 control-law calls, so
nothing failed — the whole pipeline simply runs at about one eighth of real
time.

The cost is structural: 4096 Hz x `Integration Oversampling` 10 x `N_SUB` 4 =
**163,840 derivative evaluations per second of data**, each a 72x72 matvec plus
a 33-mode modal projection, in Python.

**This is why the historical runs were operator-paced.** Profiles 42-45 carry an
empty Test Profile sheet, and `--profile` only configures the GUI without
dialogs; an operator stepped the level and pressed "Save Spectral Data" when the
loop had settled. With the nonlinear plant that wait is roughly eight times the
data duration, which no fixed wall-clock schedule written in data-time will
respect.

My 45 s-per-level schedule therefore ramped through all five levels inside the
first ~34 s of data. The five saved files are **bit-identical** in their
response spectra (response sum 55.0359 at every level), and score 10.32 dB /
79.5 % out — an early, unconverged state sampled five times, not five levels.
Against run 51's 5.61 dB that looks like a failed port; it is a failed
*schedule*.

## 4. What would make it work

- **Pace by data progress, not wall clock.** Have the harness step levels and
  save when the control loop has advanced — a control-call count or stream
  length at the current level — instead of on a timer. Robust to machine speed,
  and the honest analogue of what the operator did.
- Or **scale the schedule by the measured factor** (~8): ~360 s wall per level,
  ~40 min a run, ~2 hours for validation plus 53 and 54. Simple, but wrong again
  on a busier or faster machine.

Reducing `Integration Oversampling` or `N_SUB` would speed it up and change the
plant, so it is not an option for reproducing these runs.

## 5. Reading the eventual comparison

Even with a correct schedule, two differences from runs 49-56 remain and cannot
be removed:

- the control CPSD averaging (Exponential 0.04 then, hardcoded Linear now);
- the **law version** — these profiles point at the current laws, which have
  changed substantially since 2026-09 (structural floor, warm start, dB
  coherence guard, trim behaviour).

So a gap against 5.49-5.61 dB will not be attributable to the hardware port
alone. To isolate the port, run the *historical* law from
`control-law-development-2026-09` on the ported hardware and compare that to
run 51; only then does a current-law number mean what it appears to.

---

## 6. Second blocker: "Save Spectral Data" now saves system-ID data, not control data

The progress-paced harness works — it reads the controller's own frame count off
`CONTROL_UPDATE`, steps the level with `ADJUST_TEST_LEVEL`, and saves at a fixed
number of frames of data a level, machine-speed independent. A smoke test
stepped −18 → −12 dB at frame 6 and saved at frames 6 and 13 as intended.

The saved files are still **bit-identical** between levels, and so is the
specification sum (1.000x where −18 → −12 dB should give 3.98x). The cause is
not the harness:

| version | `save_spectral_data` writes |
|---|---|
| old, `components/random_vibration_sys_id_environment.py` | `last_transfer_function`, `last_coherence`, `last_response_cpsd`, `last_reference_cpsd` — the **live control** spectra |
| current, `process/random_vibration_sys_id_data_analysis.py` | `sysid_data.sysid_frf`, `sysid_coherence`, `sysid_response_cpsd` — the **system-ID** spectra |

The system ID is measured once before control and never changes, so every save
in a run is identical regardless of level, law or elapsed time. That is exactly
what was observed, and it also explains why the first (wall-clock) attempt and
the historical-law attempt produced the same response sum, ~55.0, despite being
different laws.

The historical files confirm the old behaviour and are self-consistent with the
published scores:

| file | response sum | rms_raw | published |
|---|---|---|---|
| `run51_..._spec_m18.nc4` | 7.739 | **5.607 dB** | 5.61 |
| `run51_..._spec.nc4` (0 dB) | 11.361 | **5.493 dB** | 5.49 |
| `run51_..._spec_p04.nc4` | 45.523 | **7.268 dB** | 7.27 |

They vary with level and reproduce `moving_plant_scores.json` to two decimals,
so `response_cpsd_real` in those files is unambiguously live control data.

**This is a regression in the current controller, not an artifact of this
exercise.** Anyone pressing "Save Spectral Data" during a random-vibration run
today gets the pre-control system-ID spectra, with no indication that is what
they are getting. It is also the remaining blocker for runs 53 and 54: the
current code cannot produce the file content the historical scoring reads.

The fix is small and local — the data analysis process already holds
`self.last_response_cpsd`, `self.control_frf`, `self.control_coherence` and
`self.last_drive_cpsd`, assigned in `run_control` from the spectral queue, which
are the four fields the old code wrote. Writing those instead (falling back to
the system-ID arrays before control starts, so a pre-control save still works)
restores the old meaning. That touches shared controller code, so it is worth
confirming as intended before it goes in.

Historical file naming, for whoever wires up the scoring: `_spec_m18`,
`_spec_m03`, `_spec_p04`, and a bare `_spec` for 0 dB — underscore before the
level, `m`/`p` for sign.

## The first validation run was empty, and looked complete

`validate51h` (18:53-20:34, the historical `optimal_diagonal_control` on the
ported nonlinear hardware) produced five spectral files of identical size, five
`[save]` lines, a clean `run exit: 0` and 96 minutes of wall clock. It contains
no control data at all.

The profile manager fired the workbook's own schedule:

```
18:58:35  Profile Manager -- Profile Firing Event Global GlobalCommands.STOP_HARDWARE
18:58:35  Frame 6x12 Random -- Stopping Control
18:58:35  Frame 6x12 Random Data Analysis got STOP_CONTROL
```

`profile_42h_optdiag_bsoft_trimoff_hist.xlsx` carries the full timed schedule —
level steps at 46 / 91 / 136 / 181 s and `Stop Streaming` / `Stop Hardware` at
271 / 275 s — and `run_headless_levels.py` passed it through unchanged while
logging "start-only events". So control ran 280 s of wall clock, about 30 s of
data, and the remaining 4500 s were dead air. The four profile level steps had
already walked all five levels inside those 30 s of data, which is the exact
failure `schedule()`'s docstring was written to prevent; the harness's own
`ADJUST_TEST_LEVEL` commands afterwards went to a stopped environment.

Two things hid it:

- **The pacing signal saturates.** The frame count on a `CONTROL_UPDATE` is the
  number of frames in the averaged CPSD, so it tops out at `frames_in_cpsd`
  (20). A settle target of 60 is unreachable, so each level waited out its
  1200 s cap and the log read `frame 20/80 updates=49` — plausible-looking, and
  in fact 20 minutes of nothing. `updates` was frozen at 49 from t=280 s to the
  end of the run, which is the one number in that line that gave it away.
- **A save after control stops writes the system ID.** Five identical files.

The only reason this was caught before it was scored is the `spectral_data_source`
stamp added with the save fix in the previous section: every file says
`system_id`, `frames 0`. Without it, five plausible files would have been scored
and a number reported.

### What changed (commit 22f45b3a)

| | |
|---|---|
| schedule | the harness keeps the pre-start events and drops every later level step and every stop, naming each dropped event in the log |
| pacing | on the cumulative control-update count, which rises as long as the law runs, not on the saturating CPSD frame count |
| silence | 120 s with no control update stops the run instead of waiting out the cap |
| reporting | the run summary prints each file's `spectral_data_source` and frame count |

`prepare_bsoft_profiles.py --write` still rewrites profiles 42-45 to the
start-only sheet, and is still the right thing to run for them. The harness no
longer depends on it: the `42h` variant built for this validation was never in
`TARGETS`, which is how it kept its schedule.

### Unrelated, found in the same log

`gui_queue_cleanup` died on its first poll on macOS — `multiprocessing.Queue.qsize()`
has no `sem_getvalue()` behind it there and raises `NotImplementedError` — leaving
a traceback at the head of every headless log that reads like a controller fault.
The queue is unbounded (`mp.Queue()` with no maxsize), so nothing deadlocks, but
nothing trims it either and on macOS it never did. The trimmer now stands down
once with a message and keeps the thread alive for a clean join. A headless run
has to drain `gui_update_queue` itself, which `run_headless_levels.py` does.

## The port is validated: `validate51i` reproduces run 51 at full level

Re-run with the fixed harness, 60 control updates a level, all five saves
stamped `control / 20 frames`:

| level | rms_coh | rms_raw | % out | level vs spec | median coh | drive | run 51 rms_raw | run 51 % out |
|---|---|---|---|---|---|---|---|---|
| -18 | 10.04 | **12.91** | 87.1 | +15.78 | 0.461 | 133.8 | 5.61 | 19.2 |
| -12 | 8.15 | 8.15 | 61.8 | +3.99 | 0.999 | 7.9 | -- | -- |
| -6 | 6.35 | 6.33 | 41.8 | +1.53 | 0.998 | 8.6 | -- | -- |
| -3 | 5.86 | 5.79 | 31.4 | +0.99 | 0.994 | 9.4 | -- | -- |
| **+0** | 5.80 | **5.59** | 34.9 | +1.61 | 0.976 | 9.7 | **5.49** | 31.7 |

The full-level point reproduces run 51 to 0.1 dB, and does so per channel, which
is the part that cannot happen by coincidence:

```
validate51i  +0 dB   5.52  2.34  3.49  3.13  2.53  3.01  10.33  8.37
run 51       +0 dB   5.47  2.49  3.46  3.16  2.39  3.33  10.34  7.71
```

Channels 7 and 8 sit at 10.3 and 8.4 dB in both, which is the achievable floor
with 6 drives; the six channels the law can actually place agree within 0.3 dB.
**The nonlinear hardware port reproduces the historical plant.**

### The -18 dB row is a startup snapshot, not a law difference

Its save was taken at update 61, about 30 s of data into the run. Three numbers
say it had not converged: the response is **+15.8 dB above** spec, the median
coherence is **0.46** against 0.98-0.999 at every later level, and the drive is
**133.8** against 7.9-9.7 everywhere else -- a 14x larger drive at the *lowest*
level is not a physical result, it is the initial drive prediction before the
law has corrected it.

The law's own `self_predicted_rms_db_per_channel` falls monotonically across the
whole run -- about 9 dB early, 6.2 by the middle, 5.5 / 2.5 / 3.6 / 3.3 / 2.8 /
3.0 / 10.9 / 8.7 at the end -- so the five levels are five points along ONE
convergence, not five converged states. Run 51 was operator-paced and held each
level for minutes. 60 updates (about 4 minutes of wall clock, 30 s of data) is
enough at the end of a run and not enough at the start of one.

So for the law comparison the settle has to be long enough that each level is
converged on its own: **150 updates a level**, roughly 10 minutes a level and
50 minutes a run at the measured 0.25 updates/s. Runs 53 and 54 are a trim-off /
trim-on pair, so an unconverged level would be charged to the trim.

## The -18 dB point converges, and to run 51's number

`validate51j`: -18 dB held, saved every 60 control updates.

| snapshot | updates | rms_coh | rms_raw | % out | level vs spec | median coh | drive |
|---|---|---|---|---|---|---|---|
| 1 | 61 | 8.82 | 11.17 | 84.7 | +14.03 | 0.492 | 128.5 |
| 2 | 121 | 8.76 | 8.77 | 63.2 | +8.08 | 0.999 | 8.9 |
| 3 | 181 | 6.18 | 6.18 | 37.5 | +1.84 | 0.999 | 8.8 |
| 4 | 241 | 5.74 | 5.74 | 21.6 | +0.43 | 0.999 | 9.0 |
| 5 | 301 | 5.79 | 5.79 | 21.9 | +0.41 | 0.999 | 9.0 |
| 6 | 361 | **5.71** | **5.71** | **21.6** | +0.52 | 0.999 | 9.1 |
| **run 51** | operator | -- | **5.61** | **19.2** | +0.31 | 0.997 | 8.8 |

```
validate51j snapshot 6    4.81  1.66  2.76  2.30  1.58  1.90  11.36  9.30
run 51 at -18 dB          4.33  1.59  2.54  2.23  1.99  1.31  10.85  9.77
```

5.71 dB against 5.61, 21.6 % out against 19.2 %, +0.52 dB of level against
+0.31, and the drive, the coherence and the per-channel pattern all land on run
51's. Snapshot 1 repeats validate51i's -18 dB signature exactly -- +14 dB over
spec, coherence 0.49, drive 128 -- so that transient is reproducible and is what
the earlier save caught, not a difference between this plant and the historical
one. **Both ends of the level range now reproduce run 51: -18 dB to 0.10 dB and
0 dB to 0.10 dB.** The port is validated.

### The settle requirement, measured

The run plateaus at snapshot 4 and holds there: 5.74, 5.79, 5.71, stable to
0.04 dB over 120 further updates. So **a cold start needs about 240 control
updates**, roughly 16 minutes of wall clock, and 60 is only enough once the law
is already converged. That is the whole explanation of validate51i's shape --
its five levels were five points along one convergence, and only the last was
at the plateau.

For runs 53 and 54, listing the first level twice at a 120-update settle gives
the first level 240 updates and every later level 120, which is about 48 minutes
a run. The repeat also keeps the convergence visible rather than assumed.

Unrelated: `Warning! SDynPy integrator ran out of samples!` at shutdown is
benign. The integrator fills a frame with zeros when the output queue is empty
for longer than one frame time, which is what happens once output stops. It is
not a limit on run length.
