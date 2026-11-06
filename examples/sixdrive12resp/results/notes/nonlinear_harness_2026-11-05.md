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
