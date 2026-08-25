"""How little data does global CVA need, vs classical H1 averaging?"""
import os
import sys
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, _HERE)
from components.sdynpy_system_virtual_hardware import SDynPySystemAcquisition
import components.utilities as util
from global_cva_frf import global_cva_v2, frf_from_ss, modes_from_A

RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))
d = {k: v for k, v in np.load(os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")).items()}


def make_channel(node, direction, ctype, fb):
    return util.Channel(node_number=node, node_direction=direction, comment=None,
                        serial_number=None, triax_dof=None, sensitivity=None, unit=None,
                        make=None, model=None, expiration=None, physical_device='Virtual',
                        physical_channel=None, channel_type=ctype, minimum_value=None,
                        maximum_value=None, coupling=None, excitation_source=None,
                        excitation=None, feedback_device=fb, feedback_channel=None,
                        warning_level=None, abort_level=None)


acq = SDynPySystemAcquisition.__new__(SDynPySystemAcquisition)
acq.channel_indices = {tuple([abs(v) for v in val]): i for i, val in enumerate(d['coordinate'])}
acq.sdynpy_system_data = d
channel_data = [make_channel(7 + i, 'X+', 'Accel', None) for i in range(8)] + \
               [make_channel(1 + i, 'X+', 'Force', 'Shaker') for i in range(6)]
acq.create_response_channels(channel_data)
sys_ss = acq.system

gt = np.load(os.path.join(RESULTS_DIR, "frf_frame6x12_H.npz"))
f_gt, H_gt = gt['f'], gt['H']
idx = np.where((f_gt >= 100) & (f_gt <= 1000))[0]
f_band = f_gt[idx]

fs = 5120.0
dt = 1.0 / fs
LAGS, RANK = 40, 66


def h1_estimate(y, u, nperseg=1024):
    """Classical H1 MIMO estimate: H = Syu @ inv(Suu), Welch-averaged.

    Convention (verified against a pure-delay test case): scipy's
    csd(a, b) = E[conj(A) B]. H1 needs Syu[i,j] = E[Y_i conj(U_j)]
    = csd(u_j, y_i) -- i.e. INPUT first, output second. Passing
    csd(y, u) instead returns the complex conjugate, which flips the
    phase sign and makes the estimate look far worse than it is.
    """
    nout, n = y.shape
    nin = u.shape[0]
    # [i,j] = csd(u_j, u_i)  ->  Suu[i,j] = E[U_i conj(U_j)]
    f_w, Suu = signal.csd(u[None, :, :], u[:, None, :], fs=fs, nperseg=nperseg,
                          noverlap=nperseg // 2, axis=-1)
    # [i,j] = csd(u_j, y_i)  ->  Syu[i,j] = E[Y_i conj(U_j)]
    _, Syu = signal.csd(u[None, :, :], y[:, None, :], fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, axis=-1)
    Suu = np.moveaxis(Suu, -1, 0)   # (F, nin, nin)
    Syu = np.moveaxis(Syu, -1, 0)   # (F, nout, nin)
    H = np.zeros_like(Syu)
    for k in range(Suu.shape[0]):
        H[k] = Syu[k] @ np.linalg.pinv(Suu[k])
    return f_w, H


print(f"{'duration':>9} {'n_blocks':>9} | {'CVA rel_err':>12} {'CVA modes':>10} | {'H1 rel_err':>11}")
print("-" * 62)
results = {}
for duration in [0.25, 0.5, 1.0, 2.0, 4.0]:
    n = int(duration * fs)
    np.random.seed(0)
    t = np.arange(n) / fs
    u = np.random.randn(6, n) * 0.05
    _, yf, _ = signal.lsim(sys_ss, u.T, t, np.zeros(sys_ss.A.shape[0]))
    y = yf[:, acq.response_channels].T

    try:
        r = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
        H_cva = frf_from_ss(r['A'], r['B'], r['C'], r['D'], f_band, dt)
        rel_cva = np.linalg.norm(H_cva - H_gt[idx]) / np.linalg.norm(H_gt[idx])
        fr, zt, mx = modes_from_A(r['A'], dt, f_hi=1100)
        nm = len(fr)
        results[duration] = H_cva
    except Exception as e:
        rel_cva, nm = float('nan'), 0
        print(f"  CVA failed at {duration}s: {e}")

    # H1 with 1024-pt blocks, 50% overlap
    nperseg = min(1024, n // 2)
    n_blocks = max(1, (n - nperseg) // (nperseg // 2) + 1)
    f_w, H_h1 = h1_estimate(y, u, nperseg=nperseg)
    H_h1_interp = np.zeros((len(f_band), 8, 6), dtype=complex)
    for a in range(8):
        for b in range(6):
            H_h1_interp[:, a, b] = np.interp(f_band, f_w, H_h1[:, a, b].real) + \
                                    1j * np.interp(f_band, f_w, H_h1[:, a, b].imag)
    rel_h1 = np.linalg.norm(H_h1_interp - H_gt[idx]) / np.linalg.norm(H_gt[idx])

    print(f"{duration:>8.2f}s {n_blocks:>9d} | {rel_cva:>12.4f} {nm:>10d} | {rel_h1:>11.4f}")

# --- plot: response 8 / excitation 1, shortest workable CVA vs GT ---
resp_i, drive_i = 7, 0
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
axes[0].semilogy(f_band, np.abs(H_gt[idx, resp_i, drive_i]), 'k-', lw=2, label='ground truth')
for dur in [0.5, 1.0, 2.0]:
    if dur in results:
        axes[0].semilogy(f_band, np.abs(results[dur][:, resp_i, drive_i]),
                         lw=1.0, alpha=0.8, label=f'CVA {dur}s')
axes[0].set_ylabel('|H|  (resp 8 / exc 1)')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[1].plot(f_band, np.unwrap(np.angle(H_gt[idx, resp_i, drive_i])) * 180 / np.pi,
             'k-', lw=2, label='ground truth')
for dur in [0.5, 1.0, 2.0]:
    if dur in results:
        axes[1].plot(f_band, np.unwrap(np.angle(results[dur][:, resp_i, drive_i])) * 180 / np.pi,
                     lw=1.0, alpha=0.8, label=f'CVA {dur}s')
axes[1].set_ylabel('phase (deg)'); axes[1].set_xlabel('Frequency (Hz)')
axes[1].grid(True, alpha=0.3)
fig.suptitle(f'Global CVA v2 (lags={LAGS}, rank={RANK}) vs ground truth -- response 8, excitation 1')
fig.tight_layout()
out = os.path.join(RESULTS_DIR, "global_cva_frf_validation.png")
fig.savefig(out, dpi=120)
print(f"\nplot -> {out}")
