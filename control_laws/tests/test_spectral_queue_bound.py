"""Checks on the bound placed on the spectral update queue.

Plain asserts, no pytest -- this repo's environment does not carry it.  Run as

    python control_laws/tests/test_spectral_queue_bound.py

Drives the real SpectralProcessingProcess into a real multiprocessing queue
with a consumer that deliberately falls behind, which is the condition that
produced the 88-104 s FRF lag: an unbounded queue plus a producer faster than
the transport.
"""

import multiprocessing as mp
import pathlib
import queue as _queue
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from rattlesnake.process.spectral_processing import (  # noqa: E402
    AveragingTypes, Estimator, SpectralProcessingMetadata,
    SpectralProcessingProcess)

NL, NRESP, NREF = 65, 6, 4
_H_RNG = np.random.default_rng(0)
H_TRUE = (_H_RNG.normal(size=(NL, NRESP, NREF))
          + 1j * _H_RNG.normal(size=(NL, NRESP, NREF)))


class StubCommandQueue:
    def __init__(self):
        self.q = _queue.Queue()
    def put(self, name, message):
        self.q.put((name, message))
    def flush(self, *a, **k):
        out = []
        while not self.q.empty():
            out.append(self.q.get())
        return out


def make_process(out_queue):
    return SpectralProcessingProcess(
        process_name='test',
        command_queue=StubCommandQueue(),
        data_in_queue=_queue.Queue(),
        data_out_queue=out_queue,
        environment_command_queue=StubCommandQueue(),
        gui_update_queue=_queue.Queue(),
        log_file_queue=_queue.Queue(),
        environment_name='env')


def make_frame(rng):
    ref = rng.normal(size=(NREF, NL)) + 1j * rng.normal(size=(NREF, NL))
    return np.einsum('fij,jf->if', H_TRUE, ref), ref


def produce(p, rng, n):
    """Run n spectral updates with nothing reading the output queue."""
    for _ in range(n):
        p.data_in_queue.put(make_frame(rng))
        p.run_spectral_processing(None)


def drain(q):
    """What the consumer does: flush and keep the newest."""
    items = []
    while True:
        try:
            items.append(q.get(timeout=0.3))
        except _queue.Empty:
            break
    return items


def release(*queues):
    """Let the interpreter exit even with a payload still undelivered.

    An mp.Queue hands each item to a feeder thread that writes the pickle into
    a pipe, and the pipe holds only 64 kB on this platform.  A spectral payload
    is larger than that, so the feeder blocks mid-write until something reads.
    At interpreter exit multiprocessing joins that thread, which never returns
    -- the test hangs with no output rather than failing.  cancel_join_thread
    drops the undelivered bytes instead of waiting for a reader, which is what
    a test wants at teardown.  Nothing in the controller relies on this; it is
    purely about letting this script exit.
    """
    for q in queues:
        q.cancel_join_thread()


N = 20
print('=' * 72)
print(f'1. UNBOUNDED queue: {N} updates with the consumer stalled')
q_unbounded = mp.Queue()
p = make_process(q_unbounded)
p.initialize_parameters(SpectralProcessingMetadata(
    AveragingTypes.EXPONENTIAL, 8, 0.25, Estimator.H1,
    NRESP, NREF, 1.0, 2.0 * NL, NL))
t0 = time.time()
produce(p, np.random.default_rng(1), N)
elapsed = time.time() - t0
time.sleep(0.5)
items = drain(q_unbounded)
print(f'   producer time        : {elapsed:.2f} s')
print(f'   payloads backlogged  : {len(items)}')
print(f'   dropped / superseded : {p.dropped_spectral_updates} / '
      f'{p.superseded_spectral_updates}')
print(f'   first payload frames : {items[0][0]} (what a non-flushing '
      f'consumer would read)')
assert len(items) == N, 'unbounded behaviour changed'
assert p.dropped_spectral_updates == 0 and p.superseded_spectral_updates == 0
release(q_unbounded)
print('   -> every update queued, oldest first: unchanged from before')

print()
print(f'2. BOUNDED at 1: the same {N} updates')
q_bounded = mp.Queue(maxsize=1)
p2 = make_process(q_bounded)
p2.initialize_parameters(SpectralProcessingMetadata(
    AveragingTypes.EXPONENTIAL, 8, 0.25, Estimator.H1,
    NRESP, NREF, 1.0, 2.0 * NL, NL))
t0 = time.time()
produce(p2, np.random.default_rng(1), N)
elapsed = time.time() - t0
time.sleep(0.5)
items = drain(q_bounded)
print(f'   producer time        : {elapsed:.2f} s  (never blocked)')
print(f'   payloads backlogged  : {len(items)}')
print(f'   dropped / superseded : {p2.dropped_spectral_updates} / '
      f'{p2.superseded_spectral_updates}')
delivered = items[-1][0]
print(f'   delivered frames     : {delivered} of {N} produced '
      f'-> age {N - delivered} update(s)')
assert len(items) <= 2, f'queue grew past its bound: {len(items)}'
assert delivered >= N - 1, (
    f'consumer got stale data: frames {delivered} of {N}')
release(q_bounded)
print('   -> one payload in flight, and it is the newest one')

print()
print('3. The producer does not block when the consumer never reads')
q3 = mp.Queue(maxsize=1)
p3 = make_process(q3)
p3.initialize_parameters(SpectralProcessingMetadata(
    AveragingTypes.EXPONENTIAL, 8, 0.25, Estimator.H1,
    NRESP, NREF, 1.0, 2.0 * NL, NL))
rng = np.random.default_rng(2)
worst = 0.0
for _ in range(30):
    p3.data_in_queue.put(make_frame(rng))
    t = time.time()
    p3.run_spectral_processing(None)
    worst = max(worst, time.time() - t)
print(f'   slowest single update: {worst * 1e3:.1f} ms over 30 updates')
assert worst < 2.0, 'an update blocked, which would back-pressure acquisition'
release(q3)
print('   -> no stall, so acquisition cannot be starved by the transport')

print()
print('4. Staleness is bounded as the producer/consumer ratio grows')
print('   ratio   unbounded backlog   bounded age')
for ratio in (1, 5, 20, 60):
    qu, qb = mp.Queue(), mp.Queue(maxsize=1)
    pu, pb = make_process(qu), make_process(qb)
    for proc in (pu, pb):
        proc.initialize_parameters(SpectralProcessingMetadata(
            AveragingTypes.EXPONENTIAL, 8, 0.25, Estimator.H1,
            NRESP, NREF, 1.0, 2.0 * NL, NL))
    produce(pu, np.random.default_rng(3), ratio)
    produce(pb, np.random.default_rng(3), ratio)
    time.sleep(0.4)
    iu, ib = drain(qu), drain(qb)
    print(f'   {ratio:5d}   {len(iu):17d}   '
          f'{ratio - ib[-1][0]:5d} update(s)')
    assert ratio - ib[-1][0] <= 1, 'bounded age grew with the ratio'
    release(qu, qb)

print()
print('ALL CHECKS PASSED')
