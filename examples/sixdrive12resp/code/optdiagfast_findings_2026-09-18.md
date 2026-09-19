# Making optimal_diagonal_control_fast work — runs 19 to 25

## What was wrong

`optimal_diagonal_control_fast` had never actually been tested. Its fast path
was gated on the FRF being **bit-identical** between calls, and with "Update
Transfer Function During Control" on the environment republishes an
incrementally-averaged FRF every cycle, so every bin fell through to the SDP.
Runs 05 and 06 were the base law wearing the fast law's name.

With the gate fixed (relative Frobenius change against `frf_update_threshold`)
and the law tested with the update OFF, the factored path finally ran — and
failed badly. Run 22: **9.91 dB rms, +14.29 dB level, 17.8 V** against the SDP
law's 5.68 / +0.17 / 4.4.

## Why it failed — the solution was unrealizable, not wrong

The solver was not at fault. Offline, the law's own `_solve_one_bin` on run
22's own FRF gives 4.72 dB rms and −0.39 dB level, and the full control loop
reproduces that exactly (901 bins, all on the fast path).

The commanded and measured drives are nearly identical in aggregate —
coherence median 0.494 vs 0.406, 2.5 % of pairs above 0.99 in both, total
drive 795 vs 859 V², a third of a dB apart. But the response they predict
differs by **15 dB**: −0.39 dB level from the commanded drive, +14.81 from the
measured one, through the same FRF.

That is the signature of a solution that depends on cancellation between
drives. The unconstrained factored solve puts drive in a near-null direction
of H; the rig reproduces the commanded CPSD closely but not exactly, and that
small difference destroys the cancellation.

Perturbing the commanded drive by 1 %:

| configuration | rms | level | rms +1 % | level +1 % |
|---|---|---|---|---|
| factored, unconstrained | 4.64 | −0.42 | 11.85 | **+16.69** |
| factored + drive_rcond 1e-2 | 3.73 | +1.94 | 4.68 | +3.80 |
| SDP + coherence cap 0.95 | 4.16 | −0.50 | 7.23 | +7.92 |
| SDP, cap off | 4.09 | −0.49 | 7.98 | +9.09 |

A 1 % realization error predicts +16.69 dB; the rig measured +14.29.

**This rules out a soft coherence penalty.** The coherence cap only halves the
sensitivity (+7.92 against +9.09 uncapped). It was never what protected the
SDP — conditioning was. A soft version of a lever that weak cannot work, and a
sweep confirmed it barely moves the distribution (λ = 1e-4 takes median
coherence from 0.578 to 0.556).

## The fix, and the sweep that set it

`drive_rcond` confines the factored solve to right-singular directions with
σ ≥ `drive_rcond`·σ_max. It had been gated to the dB objective only, so the
factored path in its DEFAULT linear mode had nothing constraining it at all.
It now applies in both domains.

Rig sweep, same profile, FRF update off, startup cap on:

| run | drive_rcond | rms | % out | level | max V | drive V² | easy six |
|---|---|---|---|---|---|---|---|
| 22 | off | 9.91 | 52.9 % | +14.29 | 17.84 | 859.4 | 8.82 |
| 23 | 1e-2 | 6.56 | 26.9 % | +3.55 | 9.23 | 284.9 | 3.72 |
| **24** | **3e-2** | **6.30** | **22.9 %** | **+0.75** | **6.50** | **153.4** | **3.21** |
| 25 | 5e-2 | 6.61 | 25.4 % | +1.17 | 6.99 | 160.7 | 3.70 |
| 19 | SDP + cap | 5.68 | 19.1 % | +0.17 | 4.40 | 67.4 | 2.50 |

3e-2 is a genuine minimum — rms, percent-out, level and drive are all worse on
both sides of it. It is now the default.

Run 23's result calibrated the model: it landed on the +3 % perturbation row,
so the rig's effective drive-realization error is about 3 %, not the 1 % first
assumed. That calibration predicted run 24's level to 0.08 dB (+0.75 measured
against +0.67 predicted). It did NOT predict run 25 — forecast 6.24 dB and
1.58 V, measured 6.61 and 6.99 V. The model finds the region; it should not be
trusted past that.

## Where it leaves the law

The factored path plateaus **0.6 dB short of the SDP** (6.30 vs 5.68) and needs
**2.3× the drive** (153 vs 67 V²), with the easy six at 3.21 dB against 2.50.
It sacrifices the same two channels (13X+, 14X+), so its character matches.

That 0.6 dB is the price of dropping the coherence constraint, bought for
roughly **18× the solve speed** (about 75 ms per 20-bin cycle against 1.3 s).
Reasonable for survey or setup work; finish on the SDP law.

No setting of `drive_rcond` recovers the gap — tightening past 3e-2 starts
removing directions the plant genuinely needs. Closing it would mean
constraining conditioning inside the factored solve, which is real work, not a
parameter.

## How plant-dependent is the threshold? — and it is now automatic

`drive_rcond` is a threshold on the plant's singular-value spread, so an
absolute value has to be re-tuned per test article. It is now expressed as a
MULTIPLE of the plant's own spread instead, measured from the system ID the
law already has. Default 2.0; a negative value is an absolute ratio for
reproducing an old run.

**Finding the reference is the whole trick.** This 8-response/6-drive frame is
rank 5. Measured sigma_k/sigma_1 medians, in band:

    1.0000   0.3189   0.0844   0.0324   0.0152   0.0017

The last value is the identification noise floor, not a direction the plant
has. Scaling off sigma_min would therefore track how good the system ID was
rather than how the structure is conditioned. The floor is found by the
largest gap in the spectrum — 8.9x between sigma_5 and sigma_6, against
2.1-3.8x everywhere else — and the reference is the smallest direction above
it, sigma_5/sigma_1 = 0.0152. The rig-tuned 3.0e-2 is 2.0x that.

Two traps, both hit while building this:

- **Use in-band lines only.** Out-of-band lines are not controlled and are
  conditioned differently; including them moved the reference from 0.0152 to
  0.1435, a factor of 9, turning the tuned 3.0e-2 into a useless 0.29.
- **Use sigma_5, not sigma_min.** Scaling off sigma_min gives 0.0016, which is
  noise, and 18x rather than 2x.

Verified: multiple 2.0 reproduces the hand-tuned 3.0e-2 to within 1-2 % across
five independent system IDs (runs 19, 22, 23, 24, 25), which is the point —
the same number falls out of each identification without being told.

### Damping

RETRACTED. An earlier version of this section claimed, from an analytic FRF
rebuilt from M/K/C, that raising modal damping from 4.00 %/3.34 % to 5 % moved
sigma_min/sigma_1 from 0.089 to 0.153 and that the equivalent threshold was
about 5e-2. That reconstruction was wrong: it produced a spectrum with no
near-zero sixth singular value at all, inconsistent with this plant's
established rank-5 structure and with the measured spread above. The numbers
should not be used.

The qualitative expectation stands on general grounds — more damping means
flatter response, a narrower singular-value spread and a better conditioned
plant; lighter damping means the opposite, and makes the unconstrained
factored path correspondingly more dangerous. But the multiple-of-spread
formulation now handles that automatically, which is why the absolute number
matters less than it did. Measuring it properly on the shifted system remains
worth doing, from a measured system ID rather than a rebuilt analytic FRF.

## Caveat on the metric

Pooled rms and what the trace looks like disagree on this plant. Run 15 (dB
objective) scored the best rms of any run at 4.90 dB but had 30.6 % of lines
outside ±3 dB against the linear objective's 18.6 %, because the dB objective
spreads error instead of sacrificing the two unreachable channels. The runs
above are all linear-objective, which holds the achievable channels tight and
lets 13X+ and 14X+ go — the behaviour wanted here.

## Repeatability, and what is NOT repeatable (runs 26, 27)

Run 26 used the automatic threshold; run 27 repeated it with nothing changed.

| run | rms | % out | level | max V | drive V² |
|---|---|---|---|---|---|
| 27 auto (repeat) | 6.27 | 22.8 % | +0.55 | 9.83 | 245.4 |
| 26 auto | 6.38 | 23.9 % | +3.06 | 12.80 | 484.7 |
| 24 hand 3.0e-2 | 6.30 | 22.9 % | +0.75 | 6.50 | 153.4 |
| offline, both FRFs | 5.96 | — | −0.74 | 2.07 | 16.2 |

**Accuracy repeats; drive does not.** rms holds within 0.1 dB and per-channel
within 0.2 dB across all three, but drive ran 153 → 485 → 245 V² for the same
commanded solution, and all three are 10-30x the offline prediction. That is
the realization-sensitivity finding showing up again: this law sits near the
conditioning limit, so the response is reproducible and the drive is not.

Consequence for reading earlier rows in this document: single-run drive
comparisons were over-interpreted. "2.3x the SDP" should read "150-500 V²
against the SDP's 67". Budget for the spread, not the mean.

The automatic threshold is confirmed by run 27 — it reproduced the hand-tuned
run 24 on every accuracy measure, having derived the threshold from a fresh
identification (0.0302 vs 0.0303) without being told it.

## FRF update on

Solving on H and running on a perturbed plant, which is the question the
demotion gate exists to answer (rms dB / level dB):

| configuration | exact | H off 2 % | H off 5 % | H off 10 % |
|---|---|---|---|---|
| factored, unconstrained | 4.52 / −0.39 | 4.63 / +4.06 | 8.95 / +10.57 | 13.72 / +16.23 |
| factored + auto drive_rcond | 5.90 / −0.74 | 4.33 / −0.56 | 3.78 / +0.23 | 4.13 / +2.25 |
| SDP + coherence cap | 4.05 / −0.50 | 3.56 / +0.80 | 5.11 / +4.68 | 8.61 / +9.55 |

**The restricted factored path is the most FRF-robust of the three.** Slightly
worse with an exact model, clearly better with a wrong one — at 10 % FRF error
it holds +2.25 dB where the SDP goes to +9.55. `drive_rcond` was added for
drive-realization error and turns out to handle model error at least as well,
and better than the coherence cap does.

That undercuts the demotion gate's premise. The gate demotes whenever H moves
more than 5 %, but at 5 % model error the restricted factored path beats the
SDP (3.78 / +0.23 against 5.11 / +4.68), so demoting trades a better solver
for a worse one and pays 18x the time for it.

**Setting the new threshold needs a measurement that does not exist yet.** Only
the FINAL FRF is saved in a run file, so run 21 shows about 28 % cumulative
drift between system ID and end of control and says nothing about the
per-cycle movement the gate tests. `_h_moved` now logs the observed relative
change every call — median, 90th percentile, max and demotion rate — so one
update-on run yields the distribution and the threshold can be set from it
rather than guessed.
