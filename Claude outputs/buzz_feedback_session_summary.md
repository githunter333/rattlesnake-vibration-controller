# buzz_feedback Control Law — Session Summary

**Repo:** `rattlesnake-vibration-controller`, `control_laws/control_laws.py`
**Status at end of session:** Working (no more instant blowups), but converges poorly / drifts slowly out of bounds. One known structural gap identified, fix designed but **not yet implemented**. Nothing committed to git.

## What was built

`buzz_feedback` is a new closed-loop control law, built on top of the existing `buzz_control` design:

- Seeds response cross-terms (coherence/phase) from system ID for the first command, same as `buzz_control`.
- From cycle 2 on, refreshes cross-terms from the live measured response every cycle instead of freezing them at the system-ID snapshot.
- Recomputes the full target response CPSD from scratch each cycle (per-channel corrected diagonal + live coherence/phase), then re-solves the pseudoinverse.
- Per-channel diagonal update is a pure integrator: `target_diag_i *= exp(Ki * log(spec_i / achieved_i))`.

`extra_parameters` format (current): `rcond,max_drive_coherence,startup_test_level_cap_db,Ki,max_step_db,min_correction_frames`

## Chronology of fixes this session

1. **Initial build + verification.** Wrote the class, appended it to `control_laws.py`, built a 26-check (later 39-check) offline verification suite. Found and fixed a real aliasing bug along the way: `target_diag` was a view into `specification`, not a copy, causing silent corruption (`.copy()` added).

2. **Live blowup #1** (Response Error ~800 dB, drive collapsed to 0V). Diagnosed from `Rattlesnake.log`: a single-cycle overcorrection right after the deliberately quiet startup command, while several channels were still near the noise floor. **Fix:** added `max_step_db` — caps how far `target_diag` can move (in dB) per channel per cycle, mirroring `match_trace_pseudoinverse_pi`'s existing convention.

3. **CPSD averaging investigation** (side thread, answered directly from source): confirmed both Linear and Exponential averaging start at exactly 1 raw, unaveraged frame on the very first control cycle — there's no frame-count protection on the response CPSD the way there already is for the FRF (`min_live_frf_frames_before_replacing_sysid_seed = 2`).

4. **Added `min_correction_frames`** — holds the drive completely unchanged (skips the correction entirely) until at least N live frames have accumulated, closing the gap found in step 3.

5. **Live blowup #2** (Response Error ~250 dB, params `1e-15,0.95,-12,0.25,3,4`). Diagnosed from `Rattlesnake.log`: happened on the *exact* cycle the `min_correction_frames` gate released. Proved mathematically that `target_diag` couldn't have moved more than 2x that cycle, yet output jumped ~14 orders of magnitude — meaning the real gap was in the closed-loop branch's full pseudoinverse re-solve (driven by live coherence estimated from only 4 frames, a known-biased estimator), which had no output-magnitude safety net at all. **Fix:** added an output-trace rate limiter, reusing `max_step_db`, clamping each cycle's output trace to at most `exp(max_step_db·ln10/10)` times the *previous* cycle's own output trace. Also tightened the `rcond` default from `1e-15` to `1e-5`, having checked the real captured FRF's SVD directly (worst lines: condition numbers 1000–7000 around 980/972/145/108–110 Hz) — flagged at the time that `1e-5` likely wouldn't fully neutralize those lines on this rig; `~1e-3` looked closer to what was needed.

6. **Live test #3.** No more instant blowup — but "control accuracy seems very poor, giving it plenty of time to stabilize." Pulled the fresh log: drive RMS climbed continuously and monotonically the entire ~3 minute run (0.2 → 180+), never leveling off; measured response climbed right alongside it. Not settling, just slow.

7. **User re-ran at `rcond=2e-3`** (higher than my own suggested floor). Still diverging, just more slowly.

8. **Root-caused with a direct synthetic reproduction** (not guesswork): built a transfer function with one response channel made genuinely hard to control, ran 150 cycles at the live settings across `rcond` = 1e-15, 2e-3, and even 1e-1. Result: the hard-to-control channel's `target_diag` wound up to `2×10¹⁶` **regardless of rcond**, while a perfectly good channel got crushed to `~3×10⁻³⁵` via the shared coherence cross-terms (they scale with `sqrt(target_diag_i · target_diag_j)`, so one channel's runaway integrator drags the whole solve with it). At `rcond=1e-15`, even with the diagonal frozen entirely (falling into the "don't trust this data" hold state every cycle), the output still drifted upward ~1.66×10⁸× over 150 cycles from the coherence reconstruction alone.

## The core finding

`max_step_db` and the output-trace rate limiter are **rate limiters** — they bound how fast something can grow per cycle, not how large it can ultimately get. "At most 2x per cycle, forever" is still unbounded given enough cycles. There is currently no **anti-windup / integrator saturation** in `buzz_feedback`: nothing stops `target_diag` from climbing indefinitely on a channel whose error never actually improves, and because of the shared coherence cross-terms, that runaway isn't isolated — it can poison otherwise-healthy channels too.

## Proposed next step (not yet implemented)

Add an **absolute ceiling**, not another rate limit:
- Clamp `target_diag` so it can never exceed some fixed multiple of the *original* specification (integrator state saturation).
- Clamp output trace against a *fixed* reference (e.g. the raw specification-based solve), not against the previous cycle's own output, since that reference can itself have already drifted arbitrarily high.

A channel that's genuinely unfixable would then pin at the ceiling instead of growing forever, and — just as important — stop dragging the good channels down with it through the cross-terms.

Open question for next session: what ceiling to default to (proposed a starting point of ~20–30 dB above spec, off-but-recommended like the other safety params, but wanted to confirm in simulation first before picking a number).

## Current code state

- `control_laws/control_laws.py`: `buzz_feedback` class and `_parse_buzz_feedback_parameters` fully implemented with all fixes above (Ki, `max_step_db` dual role, `min_correction_frames`, `rcond` default `1e-5`). Lines before the appended section remain byte-identical to the last commit — everything is purely additive.
- `/tmp/verify_buzz_feedback.py` (device-side, ephemeral scratch file — not part of the repo): 39/39 checks passing, including direct reproductions of both live blowup shapes and the output-trace rate limiter's behavior.
- **Nothing has been committed to git.** All changes are on disk on the device only.

## Other threads from this session (still open / deferred, unrelated to buzz_feedback's core issue)

- `make kill-rattlesnake` — improved diagnostics and a broader `pkill` fallback pattern were added to the Makefile, but never confirmed fixed on a real hung process (couldn't verify from the sandboxed environment).
- Adding a full Kp term to `buzz_feedback` (proportional/derivative-style damping, like `match_trace_pseudoinverse_pi` has) — explicitly deferred by request until Ki-only was tried first. Given today's findings, Kp alone would *not* fix the windup (windup is about the *integrator having no ceiling*, not about needing faster/smoother correction) — the anti-windup ceiling is the more directly relevant fix.
- `spectral_processing.py`'s `LinAlgError: Eigenvalues did not converge` — separately deferred, not touched.

## Suggested pick-up point

Implement the `target_diag` / output-trace absolute ceiling described above, verify it against the same synthetic "unfixable channel" reproduction used to diagnose the windup (confirm the channel pins instead of diverging, and that a healthy channel next to it stays healthy), then it should be safe to try live again.
