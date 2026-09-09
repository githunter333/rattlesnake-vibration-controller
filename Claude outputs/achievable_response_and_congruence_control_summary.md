# Achievable-Response Predictor & Diagonal-Congruence Control — Session Summary

**Repo:** `rattlesnake-vibration-controller`
**Dates:** 2026-09-08 / 09
**Branch:** `control-law-development-2026-09` (not pushed; `main` untouched)
**Status:** Both new modules built, verified, committed. The control law has now had **two clean hardware runs** and beats `match_trace` by ~1.4 dB on the current system.

---

## What now exists that didn't

| File | What it is |
|---|---|
| `control_laws/achievable_response.py` | The predictor. Given an FRF and a spec, computes the best response any control law could produce. Does not drive hardware. |
| `control_laws/diagonal_congruence_control.py` | `match_diagonal_congruence` — per-channel feedback control that generalizes `match_trace`. |
| `examples/.../achievable_floor_100_1000Hz.npz` | Per-line floor, projected targets, drive power, conditioning. |
| `examples/.../projected_target_100_1000Hz.npz` | The reachable target the law aims at. **Regenerate after every system ID.** |
| Three figures in `examples/.../results/` | Floor vs frequency; per-channel error; per-channel achieved spectra. |

---

## The predictor

The question it answers: *how much of my response error is the control law's fault, and how much is simply not reachable?* With N drives and M control responses the achievable set is `{diag(H X H^H) : X >= 0}` — a convex cone. With 8 responses from 6 drives it is a **proper subset** of the positive orthant, so parts of a perfectly reasonable flat spec are unreachable.

Formulated in **dB**, not as a linear least-squares residual (which over-weights the loudest channels). `X = L L^H` makes positive semidefiniteness structural, so only numpy/scipy are needed — no cvxpy.

**Verified** against finite-difference Jacobians (1.5e-9), two closed-form cases, exact achievability when M <= N, PSD-ness of every returned drive, and self-consistency. The optimum was identical to **0.000 dB across five independent seeds** on real data.

A diagnostic nearly misled me: `restart_spread_db` fired at 5.2 dB and I first read it as "unreliable". It isn't — best-of-8 restarts equalled best-of-2 exactly. It measures landscape ruggedness, not reliability of the minimum. Docstring corrected.

**Timing:** ~125 ms/line, about 2 minutes for a 901-line band at the defaults. Cost is per line and parallelises trivially.

---

## What it revealed about the rig

- **The floor is not zero.** Median 1.37 dB, only 13.8% of lines exactly achievable, 58% cannot reach 1 dB. A real share of the residual error is structural.
- **Channels 7 and 8 are the sacrificed pair** — within 1 dB on only 32% and 37% of lines, against 96-98% for channels 3-6. Their errors are near **mirror images**: where 7 must run hot, 8 runs cold, and above 600 Hz they swap. Six drives cannot set both independently.
- Channel 7 is the accelerometer that had been used as the visual guide. A good part of its reading was never a control failure.
- **The system ID only excited 98-1002 Hz.** Below that the FRF is noise (phase jitter 0.46-0.73 rad against 0.08 in band). The control band starts at 100 Hz, so this sits just below it — but it is worth knowing the margin is thin.
- **`rcond` truncation is a good trade:** restricting to the 1e-3 subspace costs 0.74 dB of floor while cutting median `trace(X)` from 1.5e3 to 4.9e-2 — four and a half orders of magnitude less drive for well under a dB.

---

## The control law

`match_trace` computes one scalar per line and scales the whole drive CPSD. Robust, because a uniform positive scalar can never make a drive CPSD invalid — and limited, because if channel 3 is 5 dB high and channel 5 is 5 dB low it does nothing.

The generalization is a **diagonal congruence** `X <- D X D`, `D = diag(exp(u))`. Congruence preserves positive semidefiniteness *structurally*: no gains, however wrong, can command an invalid drive. `match_trace` is the special case `D = sqrt(c) I`. Gains come from one Gauss-Newton step using `S_mi = dlog(y_m)/du_i`, solved as a small regularized least-squares problem.

Two properties, both verified:

- **Every row of S sums to exactly 2**, so a uniform channel error reproduces `match_trace`'s update *exactly*. It degrades gracefully to the law it generalizes.
- **H enters only as a direction and is never inverted in the loop.** A mediocre FRF gives a usable direction and feedback cleans up the rest — the difference from the pinv-based laws that caused the drive-imbalance events.

It aims at the **projected target** (the predictor's output) rather than the raw spec, so it never integrates toward something unreachable.

---

## Hardware runs

**Run 1 (09-08), `Ki=0.3, max_step_db=2`.** Prediction accuracy was the striking part:

| ch | simulated | live | diff |
|---|---|---|---|
| 1 | 3.33 | 3.74 | +0.41 |
| 4 | 3.11 | 3.15 | +0.04 |
| 7 | 8.43 | 9.42 | +0.99 |
| 8 | 7.36 | 7.96 | +0.60 |

Mean absolute error **0.53 dB**, correlation 0.98. Drives 1.79-3.90 V, 2.2x spread — against the 09-07 event at 13-18 V with two channels at 148 V and 477 V.

**Run 2 (09-09), `Ki=0.5, max_step_db=3`, more heavily damped rig, fresh projection.** Channels 1-6 mean improved 3.56 -> **3.06 dB**, overall 4.84 -> **4.44 dB**. Drives 1.80-4.60 V.

**Versus `match_trace` on the current system** (181 lines, 16 frames/cycle):

| | mean RMS | median |
|---|---|---|
| achievable floor | 1.46 dB | 1.22 dB |
| `match_trace_pseudoinverse` | 5.42 dB | 5.10 dB |
| `match_trace_pseudoinverse_pi` | 5.41 dB | 5.10 dB |
| **`match_diagonal_congruence`** | **3.98 dB** | **3.75 dB** |

~1.44 dB better, about 36% of the headroom above the floor. Drive imbalance essentially unchanged (5.1x vs 5.3x), so the gain isn't bought with a lopsided drive.

---

## Defects found and fixed

Recorded because several were real and non-obvious:

1. **Per-line floor bug** (`match_trace_pseudoinverse_pi`, both resolve laws). The near-zero-lock floor used `1e-8 x the loudest line in the band` instead of each line's own target. On a spectrum with wide dynamic range it inflated legitimately quiet lines. *This one has since been exercised on hardware in a good run.*
2. **`max_step_db` did not bound the response.** Rows of S sum to 2 but individual entries can be large and opposite-signed, so bounded gains still swung a channel 38 dB. Now enforced on the predicted response, iteratively.
3. **Startup cap compared `trace(drive)` against `trace(spec)`** — mixing drive and response units, so the realised cap depended on FRF scale. Now caps the predicted response.
4. **The ceiling was model-based**, i.e. only as trustworthy as the FRF, in the exact regime it exists to protect against. Now anchored on the measured level.
5. **Taking the stricter of model and measurement looked conservative but wasn't** — an overestimating model held drive down permanently and the test never reached spec.
6. **Trusting measurement alone reintroduced the dead-sensor hazard.** Resolved with a two-threshold ceiling: measurement governs, the model net sits 20 dB looser. Beyond that margin the law deliberately limits level, because "FRF wrong by 30 dB" and "accelerometer failed low" are indistinguishable from these two signals.
7. **Relative import made the law unloadable.** Rattlesnake uses `spec_from_file_location`, so there is no package context. It would have failed in the GUI before running.
8. **The target was read before system ID.** Rattlesnake constructs the law at `INITIALIZE_PARAMETERS`, so the projection could only ever be stale. Now read lazily on the first control cycle.

---

## Two things worth remembering

**The damping really did change.** Confirmed quantitatively: 160 Hz went from zeta 1.25% to **3.75%**, ~247 Hz from 0.81% to **3.23%**. Conditioning improved with it — median 1013 with nothing above 1e4, versus 1435 with 4% above. The achievable floor improved from 1.37 to 1.22 dB median.

**The trace runs ~0.5 dB high, and that is arithmetic, not bias.** The mean of the per-channel dB errors is **-0.18 dB** — the law is doing exactly what it should — yet the linear trace sums to +1.17 dB. A spread in dB always sums high in linear power (Jensen). It is *not* channels 7/8: their shares are 1% and -10%. Reducing it means reducing scatter; more CPSD averaging helps down to a ~1.15 dB floor.

---

## Open items

- **Full non-diagonal congruence.** The residual 2.5 dB above the floor is structural: diagonal congruence rescales drive channels but cannot reshape their cross-terms. A tuning sweep confirmed the current defaults are already optimal.
- **sdynpy testbed.** Every simulation so far uses the captured FRF as both plant and model, so it cannot exercise model mismatch — the regime the earlier laws failed in. `sdynpy` has `assign_modal_damping`, `time_integrate` and a full `simulate_test`, which would close that gap.
- **Optional overall-level trim**, if the test is graded on gRMS rather than per-channel PSD.
- **Phase 2 reachability in the two resolve laws remains unresolved.** They are not in use.
- **Regenerate the projected target after every system ID.** Nothing detects staleness automatically.
- `git gc --prune=now` on the Mac to clear stale lock files this sandbox could not delete.
