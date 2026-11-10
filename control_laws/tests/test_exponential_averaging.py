"""Checks on exponential averaging of the during-control spectral estimate.

Plain asserts, no pytest -- this repo's environment does not carry it.  Run as

    python control_laws/tests/test_exponential_averaging.py

Part 1 drives the real ``SpectralProcessingProcess`` with synthetic frames from
a known plant and checks the recursion against the formula in the docstring.
Part 2 checks that the random environment's control path offers the choice and
refuses a coefficient that would freeze the estimate.
"""

import queue
import sys
import types
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve()
                       .parents[2] / 'src'))

from rattlesnake.process.spectral_processing import (
    AveragingTypes, Estimator, SpectralProcessingMetadata,
    SpectralProcessingProcess, SpectralProcessingCommands)


class StubCommandQueue:
    """Stands in for VerboseMessageProcess's command queue."""
    def __init__(self):
        self.q = queue.Queue()
    def put(self, name, message):
        self.q.put((name, message))
    def get(self, *a, **k):
        return self.q.get(*a, **k)
    def flush(self, *a, **k):
        out = []
        while not self.q.empty():
            out.append(self.q.get())
        return out


def make_process():
    p = SpectralProcessingProcess(
        process_name='test',
        command_queue=StubCommandQueue(),
        data_in_queue=queue.Queue(),
        data_out_queue=queue.Queue(),
        environment_command_queue=StubCommandQueue(),
        gui_update_queue=queue.Queue(),
        log_file_queue=queue.Queue(),
        environment_name='env')
    return p


NL, NRESP, NREF = 9, 4, 2
_H_RNG = np.random.default_rng(0)
H_TRUE = (_H_RNG.normal(size=(NL, NRESP, NREF))
          + 1j * _H_RNG.normal(size=(NL, NRESP, NREF)))


def make_frame(rng, scale=1.0):
    """One (response_fft, reference_fft) pair from the known plant."""
    ref = (rng.normal(size=(NREF, NL)) + 1j * rng.normal(size=(NREF, NL))) * scale
    resp = np.einsum('fij,jf->if', H_TRUE, ref)
    return resp, ref


def metadata(averaging_type, coefficient, averages=8):
    return SpectralProcessingMetadata(
        averaging_type, averages, coefficient, Estimator.H1,
        NRESP, NREF, 1.0, 2.0 * NL, NL)


def run(averaging_type, coefficient, n_frames, frames_per_batch=1, scale=1.0,
        seed=1234):
    rng = np.random.default_rng(seed)
    p = make_process()
    p.initialize_parameters(metadata(averaging_type, coefficient))
    sent = []
    frames_fed = []
    for i in range(0, n_frames, frames_per_batch):
        batch = [make_frame(rng, scale) for _ in range(frames_per_batch)]
        frames_fed.extend(batch)
        for f in batch:
            p.data_in_queue.put(f)
        p.run_spectral_processing(None)
        while not p.data_out_queue.empty():
            sent.append(p.data_out_queue.get())
    return p, sent, frames_fed


print('=' * 72)
print('1. EXPONENTIAL recovers the known plant')
alpha = 0.25
p, sent, fed = run(AveragingTypes.EXPONENTIAL, alpha, 400)
frames, freqs, frf, coh, rcpsd, refcpsd, cond = sent[-1]
err = np.abs(frf - H_TRUE) / np.abs(H_TRUE)
print(f'   updates emitted      : {len(sent)}')
print(f'   frames counter       : {frames}  (grows without bound)')
print(f'   median |H_hat-H|/|H| : {np.median(err):.3e}')
print(f'   max    |H_hat-H|/|H| : {np.max(err):.3e}')
assert np.median(err) < 0.05, 'exponential did not recover the plant'

print()
print('2. The recursion weights the newest frame by exactly alpha')
#    Replay the same frames by hand: S <- a*inst + (1-a)*S, first frame seeds it.
S = None
for resp, ref in fed:
    inst = np.einsum('if,jf->fij', resp, np.conj(resp))
    S = inst if S is None else alpha * inst + (1 - alpha) * S
#    The process applies the APSD normalisation to what it sends.
norm = 1.0 / (2.0 * NL) ** 2 * 1.0
expected = S * norm
expected[1:-1] *= 2
rel = np.abs(rcpsd - expected) / np.abs(expected)
print(f'   max relative difference vs hand recursion: {np.max(rel):.3e}')
assert np.max(rel) < 1e-10, 'running matrix does not match the documented recursion'

print()
print('3. LINEAR is unaffected by a non-zero coefficient in the metadata')
pa, sa, _ = run(AveragingTypes.LINEAR, 0.0, 40, seed=99)
pb, sb, _ = run(AveragingTypes.LINEAR, 0.37, 40, seed=99)
same = np.array_equal(sa[-1][4], sb[-1][4]) and np.array_equal(sa[-1][2], sb[-1][2])
print(f'   coefficient 0.0 vs 0.37 give identical output: {same}')
assert same, 'the coefficient leaked into the linear path'

print()
print('4. A batch of several frames is all absorbed, not just the last')
p1, s1, _ = run(AveragingTypes.EXPONENTIAL, alpha, 12, frames_per_batch=4)
print(f'   12 frames in batches of 4 -> frames counter {s1[-1][0]}')
assert s1[-1][0] == 12, 'frames were dropped from the batch'

print()
print('5. Exponential tracks a step change; linear lags it')
for label, atype, coeff in (('linear   ', AveragingTypes.LINEAR, 0.0),
                            ('exponent.', AveragingTypes.EXPONENTIAL, alpha)):
    rng = np.random.default_rng(7)
    pp = make_process()
    pp.initialize_parameters(metadata(atype, coeff, averages=20))
    for _ in range(40):                       # settle at unit scale
        pp.data_in_queue.put(make_frame(rng, 1.0))
        pp.run_spectral_processing(None)
    while not pp.data_out_queue.empty():
        before = pp.data_out_queue.get()
    for _ in range(5):                        # 10x power step, 5 frames
        pp.data_in_queue.put(make_frame(rng, np.sqrt(10.0)))
        pp.run_spectral_processing(None)
    while not pp.data_out_queue.empty():
        after = pp.data_out_queue.get()
    b = np.real(np.einsum('fii->f', before[4])).sum()
    a = np.real(np.einsum('fii->f', after[4])).sum()
    print(f'   {label}: {10*np.log10(a/b):+.2f} dB of the +10.00 dB step '
          f'after 5 frames')

print()


import sys
import pathlib
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve()
                       .parents[2] / 'src'))

from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationEnvironment)
from rattlesnake.process.spectral_processing import AveragingTypes, Estimator


def call(averaging_type, coefficient, frames_in_cpsd=20):
    """Call the real method with only the attributes it reads."""
    stub = SimpleNamespace(
        environment_metadata=SimpleNamespace(
            frames_in_cpsd=frames_in_cpsd,
            num_response_channels=8,
            num_reference_channels=6,
            frequency_spacing=5.0,
            sample_rate=2048.0,
            fft_lines=257,
            sysid_metadata=SimpleNamespace(
                sysid_estimator='H1',
                sysid_averaging_type=averaging_type,
                sysid_exponential_averaging_coefficient=coefficient)))
    return RandomVibrationEnvironment.get_spectral_processing_metadata(stub)


print('6. Linear is still the default behaviour')
m = call('Linear', 0.0)
print(f'   averaging_type {m.averaging_type}, averages {m.averages}, '
      f'coefficient {m.exponential_averaging_coefficient}')
assert m.averaging_type is AveragingTypes.LINEAR
assert m.averages == 20
assert m.frf_estimator is Estimator.H1

print()
print('7. Linear ignores a coefficient that is out of range')
m = call('Linear', 0.0)                  # the historical hard-coded value
assert m.averaging_type is AveragingTypes.LINEAR
print('   no error raised, as before')

print()
print('8. Exponential is selectable and carries its coefficient')
m = call('Exponential', 0.25)
print(f'   averaging_type {m.averaging_type}, '
      f'coefficient {m.exponential_averaging_coefficient}')
assert m.averaging_type is AveragingTypes.EXPONENTIAL
assert m.exponential_averaging_coefficient == 0.25
assert m.compute_cpsd and m.compute_frf and m.compute_coherence

print()
print('9. Exponential refuses a coefficient that would freeze the estimate')
for bad in (0.0, -0.1, 1.5):
    try:
        call('Exponential', bad)
    except ValueError as exc:
        print(f'   coefficient {bad!r:>5} -> ValueError: {str(exc)[:58]}...')
    else:
        raise AssertionError(f'coefficient {bad} was accepted')
m = call('Exponential', 1.0)
print('   coefficient   1.0 -> accepted (no averaging, newest frame only)')
assert m.exponential_averaging_coefficient == 1.0

print()
print('ALL CHECKS PASSED')
