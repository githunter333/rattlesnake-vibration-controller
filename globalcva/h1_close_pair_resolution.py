"""
Does classical H1 (Welch/CSD) resolve the same closely-spaced true-mode
pairs that global_cva_v2/global_cva_innovations struggle with (section 4.4
/ section 10)? H1's frequency resolution is fs/nperseg -- a hard FFT bin-
spacing limit, unlike a parametric method (CVA) which can in principle
place two poles closer together than that. This checks it directly rather
than just asserting it from theory: for each close pair, does |H1(f)|
show two distinct local maxima, or one merged blob?

Also sweeps nperseg to show the actual tradeoff: finer frequency
resolution (bigger nperseg) costs averaged blocks (Welch's noise-averaging
benefit), which matters because short-record efficiency is the entire
point of this line of work.
"""
import os
import sys
import numpy as np
import scipy.signal as sig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_true_system import build_system, generate_drive_response

FS = 5120.0

info = build_system()
sys_ss = info['sys_ss']
fn_true_band = info['fn_true_band']

gaps = np.diff(fn_true_band)
close_pairs = [(fn_true_band[i], fn_true_band[i + 1], gaps[i]) for i in np.where(gaps < 5.0)[0]]
print("Close true-mode pairs (<5 Hz apart):")
for f1, f2, g in close_pairs:
    print(f"    {f1:.2f} Hz <-> {f2:.2f} Hz  (gap {g:.2f} Hz)")


def h1_estimate(y, u, fs, nperseg):
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


def n_local_maxima_near(f_w, mag, f_lo, f_hi):
    """Count local maxima of mag within [f_lo, f_hi] (simple, robust to
    single-sample noise: a maximum must exceed both neighbors)."""
    idx = np.where((f_w >= f_lo) & (f_w <= f_hi))[0]
    if len(idx) < 3:
        return 0, []
    seg = mag[idx]
    peaks = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            peaks.append(f_w[idx[i]])
    return len(peaks), peaks


# pick the response/drive pair with the strongest response near each close
# pair's center frequency, using the ground-truth H, so we're looking at
# the channel most likely to show the resonance clearly
gt = info['H_gt_band']
f_band = info['f_band']

DURATIONS = [1.0, 2.0, 4.0]
NPERSEGS = [512, 1024, 2048, 4096, 8192]

for duration in DURATIONS:
    u, y = generate_drive_response(sys_ss, duration, fs=FS)
    n = u.shape[1]
    print(f"\n{'='*78}\nduration={duration:.1f}s  (n={n} samples)")
    for nperseg in NPERSEGS:
        if nperseg > n // 2:
            continue
        freq_res = FS / nperseg
        n_blocks = max(1, (n - nperseg) // (nperseg // 2) + 1)
        f_w, H1 = h1_estimate(y, u, FS, nperseg)
        print(f"\n  nperseg={nperseg:>5d}  freq_res={freq_res:5.2f} Hz  n_blocks_averaged={n_blocks}")
        for f1, f2, g in close_pairs:
            fc = (f1 + f2) / 2
            gt_idx = np.argmin(np.abs(f_band - fc))
            resp_i, drive_i = np.unravel_index(np.argmax(np.abs(gt[max(0, gt_idx - 5):gt_idx + 5])
                                                           .reshape(-1, 8, 6)[0]), (8, 6))
            mag = np.abs(H1[:, resp_i, drive_i])
            n_peaks, peak_fs = n_local_maxima_near(f_w, mag, f1 - 3 * g - 2, f2 + 3 * g + 2)
            resolved = "RESOLVED" if n_peaks >= 2 else "merged/blob"
            peak_str = ", ".join(f"{p:.1f}" for p in peak_fs) if peak_fs else "-"
            print(f"    pair {f1:.2f}/{f2:.2f} Hz (gap {g:.2f}): resp{resp_i+7}/drive{drive_i+1} "
                  f"-> {n_peaks} local max [{peak_str}]  {resolved}")
