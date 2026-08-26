"""
Zoomed-in view: true FRF vs H1 (standard nperseg=1024, matching every H1
estimate used elsewhere in this project) vs CVA-innovations, right at two
of the closely-spaced true-mode pairs, at 2s duration. Answers "how does
H1 handle these vs CVA" visually rather than just as a peak-count table
(see h1_close_pair_resolution.py for that).
"""
import os
import sys
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_innovations, frf_from_ss
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK = 40, 66
DURATION = 2.0

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt_band = info['H_gt_band']

u, y = generate_drive_response(sys_ss, DURATION, fs=FS)

r = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK, refine_iters=1)
H_cva = frf_from_ss(r['A'], r['B'], r['C'], r['D'], f_band, DT)


def h1_estimate(y, u, fs, nperseg=1024):
    nperseg = min(nperseg, y.shape[1] // 2)
    f_w, Suu = sig.csd(u[None, :, :], u[:, None, :], fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, axis=-1)
    _, Syu = sig.csd(u[None, :, :], y[:, None, :], fs=fs, nperseg=nperseg,
                      noverlap=nperseg // 2, axis=-1)
    Suu = np.moveaxis(Suu, -1, 0)
    Syu = np.moveaxis(Syu, -1, 0)
    H = np.zeros_like(Syu)
    for k in range(Suu.shape[0]):
        H[k] = Syu[k] @ np.linalg.pinv(Suu[k])
    return f_w, H


f_h1, H1 = h1_estimate(y, u, FS, nperseg=1024)

# (response node, drive node, true f1, true f2, gap label) -- the two
# pairs with an accompanying strong response, one at each end of the
# gap range tested
zooms = [(13, 6, 246.94, 249.18, "2.24 Hz gap"),
         (12, 1, 412.66, 413.61, "0.95 Hz gap"),
         (7, 4, 451.00, 454.74, "3.75 Hz gap"),
         (11, 3, 821.75, 824.72, "2.97 Hz gap")]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (resp_node, drive_node, f1, f2, label) in zip(axes.ravel(), zooms):
    ri, di = resp_node - 7, drive_node - 1
    lo, hi = f1 - 15, f2 + 15
    band_mask = (f_band >= lo) & (f_band <= hi)
    h1_mask = (f_h1 >= lo) & (f_h1 <= hi)

    ax.semilogy(f_band[band_mask], np.abs(H_gt_band[band_mask, ri, di]),
                'k-', lw=2, label='true')
    ax.semilogy(f_band[band_mask], np.abs(H_cva[band_mask, ri, di]),
                color='tab:green', lw=1.3, label='CVA innovations (2s, lags=40)')
    ax.semilogy(f_h1[h1_mask], np.abs(H1[h1_mask, ri, di]),
                color='tab:blue', lw=1.3, marker='o', ms=3,
                label='H1 (nperseg=1024, 2s)')
    for f in (f1, f2):
        ax.axvline(f, color='gray', ls=':', lw=1, alpha=0.7)
    ax.set_title(f"resp {resp_node}/drive {drive_node}: {f1:.2f}/{f2:.2f} Hz ({label})", fontsize=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Closely-spaced true-mode pairs: true vs CVA-innovations vs H1\n"
             "(dotted lines mark the two true mode frequencies)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
out = os.path.join(RESULTS, "global_cva_close_pairs_h1_vs_cva.png")
fig.savefig(out, dpi=120)
print(f"saved -> {out}")
