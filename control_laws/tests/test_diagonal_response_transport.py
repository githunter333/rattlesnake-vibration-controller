"""Checks on diagonal-only transport of the response CPSD.

Plain asserts, no pytest -- this repo's environment does not carry it.  Run as

    python control_laws/tests/test_diagonal_response_transport.py

Every control law reads only the diagonal or the trace of the response CPSD,
but the full matrix was pickled onto every spectral update and is the bulk of
the payload.  These checks cover what is transported, that the numbers survive
the round trip, that the off-diagonals can still be fetched, and that other
environments are untouched.
"""

import pathlib
import pickle
import queue as _queue
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from rattlesnake.process.random_vibration_sys_id_data_analysis import (  # noqa: E402
    RandomVibrationDataAnalysisProcess)
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


def make_process(diagonal_only):
    out = _queue.Queue()
    p = SpectralProcessingProcess(
        process_name='test', command_queue=StubCommandQueue(),
        data_in_queue=_queue.Queue(), data_out_queue=out,
        environment_command_queue=StubCommandQueue(),
        gui_update_queue=_queue.Queue(), log_file_queue=_queue.Queue(),
        environment_name='env')
    p.initialize_parameters(SpectralProcessingMetadata(
        AveragingTypes.EXPONENTIAL, 8, 0.25, Estimator.H1,
        NRESP, NREF, 1.0, 2.0 * NL, NL,
        response_cpsd_diagonal_only=diagonal_only))
    return p, out


def update(p, out, rng, n=1):
    """Runs n spectral updates and returns the last payload."""
    payload = None
    for _ in range(n):
        ref = rng.normal(size=(NREF, NL)) + 1j * rng.normal(size=(NREF, NL))
        p.data_in_queue.put((np.einsum('fij,jf->if', H_TRUE, ref), ref))
        p.run_spectral_processing(None)
        while not out.empty():
            payload = out.get()
    return payload


print('=' * 72)
print('1. Default metadata is unchanged, so other environments are untouched')
assert SpectralProcessingMetadata(
    AveragingTypes.LINEAR, 4, 0.0, Estimator.H1,
    NRESP, NREF, 1.0, 2.0 * NL, NL).response_cpsd_diagonal_only is False
p, out = make_process(diagonal_only=False)
full_payload = update(p, out, np.random.default_rng(1), 3)
full_response = full_payload[4]
print(f'   response shape        : {full_response.shape}  (full matrix)')
assert full_response.ndim == 3
assert full_response.shape == (NL, NRESP, NRESP)
print('   -> full matrix still sent when the flag is off')

print()
print('2. Diagonal-only sends the diagonal, and the drive CPSD stays full')
p2, out2 = make_process(diagonal_only=True)
diag_payload = update(p2, out2, np.random.default_rng(1), 3)
diag_response, diag_reference = diag_payload[4], diag_payload[5]
print(f'   response shape        : {diag_response.shape}  (diagonal)')
print(f'   drive cpsd shape      : {diag_reference.shape}  (still full)')
assert diag_response.shape == (NL, NRESP)
assert diag_reference.shape == (NL, NREF, NREF), (
    'the drive CPSD must stay full: laws return it scaled as a matrix')
print('   -> only the response CPSD is reduced')

print()
print('3. The transported diagonal equals the full matrix diagonal')
err = np.abs(diag_response - np.einsum('fii->fi', full_response))
print(f'   max abs difference    : {err.max():.3e}')
assert err.max() < 1e-20, 'the diagonal does not match the full matrix'
print('   -> identical, so nothing the laws read has changed')

print()
print('4. Payload size')
full_bytes = len(pickle.dumps(full_payload))
diag_bytes = len(pickle.dumps(diag_payload))
resp_full = len(pickle.dumps(full_response))
resp_diag = len(pickle.dumps(diag_response))
print(f'   whole payload         : {full_bytes:,} -> {diag_bytes:,} bytes '
      f'({100 * (1 - diag_bytes / full_bytes):.0f}% smaller)')
print(f'   response CPSD alone   : {resp_full:,} -> {resp_diag:,} bytes '
      f'({resp_full / resp_diag:.0f}x smaller)')
assert diag_bytes < full_bytes
assert resp_diag * NRESP <= resp_full * 1.1
print(f'   -> the response CPSD scales as channels, not channels squared')

print()
print('5. A one-shot request puts the full matrix on the NEXT payload only')
p3, out3 = make_process(diagonal_only=True)
rng = np.random.default_rng(4)
shapes = []
for i in range(5):
    if i == 2:
        p3.send_full_response_cpsd(None)
    shapes.append(update(p3, out3, rng)[4].ndim)
print(f'   payload ndim by update: {shapes}')
assert shapes == [2, 2, 3, 2, 2], f'unexpected sequence {shapes}'
print('   -> served once, then back to the diagonal on its own')

print()
print('6. The GUI switch holds the full matrix for as long as it is set')
p4, out4 = make_process(diagonal_only=True)
rng = np.random.default_rng(5)
seq = [update(p4, out4, rng)[4].ndim]
p4.set_response_cpsd_diagonal_only(False)
seq += [update(p4, out4, rng)[4].ndim for _ in range(2)]
p4.set_response_cpsd_diagonal_only(True)
seq += [update(p4, out4, rng)[4].ndim]
print(f'   payload ndim by update: {seq}')
assert seq == [2, 3, 3, 2], f'unexpected sequence {seq}'
print('   -> stays full while a cross-spectrum window is open')

print()
print('7. The consumer puts the diagonal back for the control laws')
expand = RandomVibrationDataAnalysisProcess._expand_response_cpsd
rebuilt = expand(diag_response)
print(f'   expanded shape        : {rebuilt.shape}')
assert rebuilt.shape == (NL, NRESP, NRESP)
d_err = np.abs(np.einsum('fmm->fm', rebuilt) - diag_response).max()
print(f'   diagonal error        : {d_err:.3e}')
assert d_err == 0.0
off = rebuilt.copy()
off.reshape(NL, NRESP * NRESP)[:, :: NRESP + 1] = 0
print(f'   off-diagonal content  : {np.abs(off).max():.3e} (zero, not measured)')
assert np.abs(off).max() == 0.0
assert expand(full_response) is full_response, 'a full matrix must pass through'
print('   -> laws still receive (lines, channels, channels) as documented')

print()
print('8. Each update gets a fresh array, so a law may keep the one it got')
a = expand(diag_response)
b = expand(diag_response)
assert a is not b, 'a reused buffer would change under a law that stored it'
a[:] = 123.0
assert np.abs(np.einsum('fmm->fm', b) - diag_response).max() == 0.0
print('   -> writing one does not disturb the other')

print()
print('ALL CHECKS PASSED')
