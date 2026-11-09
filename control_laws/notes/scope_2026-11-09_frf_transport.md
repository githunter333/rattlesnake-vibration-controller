# Scope: fixing the FRF transport lag

**2026-11-09 · scoping only, nothing changed.** Continues the 88-104 s lag
measurement in `report_2026-11-05_frf_update_lag.md`, which lives with the
rc_plate study in the band-average repo
(`studies/rc_plate/notes/`), not here. This note read the transport code rather
than the symptom, and three things in that report need revising before any fix
is chosen.

Line references are to `src/rattlesnake` in THIS repo. The study's runs use the
band-average repo's own copy of that tree, so a change made here does not reach
a study run until it is carried across -- see section 2.

## 1. What the 2026-11-05 report got wrong, and why it matters

**The payload is ~5x larger than the report assumed, and the FRF is a minority
of it.** The report inferred 75.5 MB from the FRF alone. The queue actually
carries a 7-tuple (`spectral_processing.py:710`):

| item | shape | 8193 lines | share |
|---|---|---|---|
| frequencies | (nl,) f64 | 0.1 MB | 0 % |
| **frf** | (nl, 48, 12) c128 | 72 MB | 19 % |
| coherence | (nl, 48) f64 | 3 MB | 1 % |
| **response_spectral_matrix** | (nl, 48, 48) c128 | **288 MB** | **76 %** |
| reference_spectral_matrix | (nl, 12, 12) c128 | 18 MB | 5 % |
| | **total** | **381 MB** | |

At 4097 lines it is 191 MB; at 2049 lines, 95 MB. So shrinking the FRF cannot
help much -- the response CPSD is three quarters of the bytes. It cannot simply
be dropped either: the consumer unpacks it as `last_response_cpsd`, which every
closed-loop law reads.

**It is sent twice.** `random_vibration_sys_id_data_analysis.py:422`, the moment
after receiving, puts `frf`, `coherence`, `last_response_cpsd` and
`last_drive_cpsd` onto `gui_update_queue`. So ~762 MB per update crosses a
pickle boundary, not 381. In headless runs `run_headless.py:89` drains that
queue with a daemon thread and discards it -- the cost is paid to throw it away.

**The queues are unbounded.** `engine.py:116-121` aliases `new_queue` to
`mp.Queue` (or `thqueue.Queue`) and every call site invokes it with no
arguments, so `maxsize` is 0 -- infinite. This is the actual mechanism, and it
reframes the problem: with an unbounded queue and a producer faster than the
transport, latency grows without bound no matter how fast either end is. That is
textbook bufferbloat, and it explains both observations the report could not:

- a 19x faster consumer received only 1.5x more FRFs -- the consumer was never
  the constraint;
- the 4 Hz run, with a 4x smaller payload, got WORSE -- a smaller array lets the
  producer loop more often, so the backlog deepens in count while each transit
  only shortens linearly, and the fixed per-array costs (pickle framing, feeder
  thread handoff, two memory copies) do not shrink at all.

So the lag is a queueing-discipline problem first and a bandwidth problem
second. Any fix that only reduces bytes will reduce the lag proportionally and
leave it unbounded.

## 2. The thing to try first costs one line

`RattlesnakeController.__init__` already takes `threaded`
(`engine.py:77`, default `THREADING = False` at line 51). In threaded mode
`new_queue` becomes `queue.Queue` and the subsystems become `threading.Thread`,
so a put passes the array **by reference** -- no pickle, no copy, no feeder
thread. The engine's own docstring says so:

> Running in the threaded mode results in much quicker data transfer rates as
> commands do not have to be copied in the RAM. This mode allows the control
> laws to respond quicker to changes in response data.

`studies/rc_plate/run_headless.py:87` passes `threaded=False` explicitly, so
every run in this study has been in the mode that copies 762 MB per update.

The docstring also states the risk: "threaded mode can be overloaded if the
processor is not fast enough which can result in a crash." That matters here
because the virtual hardware runs in real time -- a GIL stall in spectral
processing could make acquisition fall behind, which is a different failure than
a stale FRF and a worse one. Spectral processing is mostly numpy, which releases
the GIL inside BLAS calls, so it may be fine; that is an empirical question, not
a judgement call.

**Scope: one line in `run_headless.py`, plus a run with the probe already
written.** If it works, the architectural fix the 2026-11-05 report listed last
("keep the estimator and the control law in one process") is already built and
just switched off.

## 3. Options, in the order I would try them

| # | change | effort | risk | expected effect |
|---|---|---|---|---|
| 0 | stamp the FRF with its age and log it | ~20 lines | none | no fix; makes every other option measurable |
| 1 | `threaded=True` | 1 line | medium (GIL vs real-time acquisition) | removes the copy entirely |
| 2 | bound the queue to 1-2, drop on full at the producer | ~10 lines | low | caps staleness at one transit, ~7 s, from 88-104 |
| 3 | send only in-band lines | ~30 lines | low | bytes fall by the out-of-band fraction |
| 4 | rate-limit the source to every N frames or T seconds | ~15 lines | low | fewer, fresher sends; frees the producer |
| 5 | complex64 instead of complex128 | ~10 lines | medium (numerics) | halves the bytes |
| 6 | shared-memory double buffer, header through the queue | 200-400 lines | medium | staleness structurally impossible |
| 7 | decimate the GUI copy | ~20 lines | low | removes half the total transport |

### 0. Instrument the age (prerequisite)

Put `time.time()` into the tuple at the send, and have the environment report
`now - stamp` when it hands the FRF to the law. Then the lag is a logged number
on every run instead of something that needs `RS_FRF_PROBE` and an offline
index-matching exercise. Nothing else on this list can be judged without it, and
the report asks for it independently.

### 2. Bound the queue (the structural fix, if threading is rejected)

`maxsize=1` with a non-blocking put that drops its own payload when full is the
right discipline for this data: the consumer only ever uses `[-1]`, so an array
that cannot get through immediately has no value -- a fresher one is 7 s behind
it. Do NOT use a blocking put, which would stall spectral processing and could
back pressure into acquisition.

This bounds the lag at one transit time regardless of bandwidth, which is the
property the current design lacks. It does not make the transit faster, so it
pairs with 3, 5 or 6 rather than replacing them.

### 3. Send only in-band lines

The control band is a fraction of the spectrum -- the RC plate spec is 4097
lines against an 8193-line FRF, and the scored band is narrower still. Slicing
to the band before the put is a small change at the producer, but it needs care:
the GUI plots the full spectrum, the saved `.nc4` may expect full arrays, and
the consumer indexes by line. Worth checking what actually needs out-of-band
lines before assuming they can go.

### 6. Shared memory (the proper fix)

Allocate two `multiprocessing.shared_memory` blocks per array at environment
start, sized from the known shapes, and send only `(generation, buffer_index)`
through the queue. The producer writes the idle buffer then publishes its index;
the consumer reads the published one. A sequence counter around the write makes
torn reads detectable without a lock on the hot path.

This removes the bandwidth cost, the per-array overhead and the unbounded
backlog at once, and it is the only option that makes staleness structurally
impossible rather than merely bounded. It is also the only one that touches
shared controller code substantially, which is the reason it is last: the
`save_spectral_data` change from an earlier session is still unconfirmed as
intended, and this would be a much larger intrusion into `src/rattlesnake`.

## 4. What a fix has to be measured against

Any of these should be judged by re-running the two profiles in the report with
the age instrumentation on, and checking:

1. **Age at the law** -- the number this is all about. Target: under one control
   update, which is 0.374 s for band average r70 and 7.06 s for optimal
   diagonal fast.
2. **FRFs received per FRF produced** -- 23/152 and 35/157 today. Options 2 and
   4 deliberately make this ratio *worse* while making the delivered arrays
   fresher, so it is a diagnostic, not a target.
3. **No regression in the fidelity gate** -- the streamed response CPSD against
   `H Sxx H^H` through the exact plant, median 0.00 dB and -0.1 to +0.3 dB on
   `n_ba_r70_noclip`. Option 5 is the one that could move this.
4. **Acquisition keeps up** -- for option 1 especially. A dropped or late
   acquisition frame is a worse failure than a stale FRF, and the real-time
   virtual hardware will show it.

The `_tfu` rows should then be re-run and re-scored, since the report's section
4 confound (a law that drives harder softens more AND is punished more by the
lag) is only removed once the lag is.

## 5. What I did not verify

- Whether `compute_cpsd` can be turned off for any law in this study. The
  closed-loop laws read `last_response_cpsd`, so probably not, but the open-loop
  ones do not and might run with a much smaller payload.
- Whether the GUI and the `.nc4` writer need full-resolution arrays, which
  decides options 3 and 7.
- Whether threaded mode has ever been exercised on this study's profiles at all.
  `THREADING = False` is the default and `run_headless.py` sets it explicitly,
  so quite possibly not.
- The actual transit time of one tuple. Everything above infers it from the
  6-9 s delivery interval; option 0 would measure it directly.
