"""
Three-way comparison under additive measurement noise: global_cva_v2
(deterministic-only, no error term) vs global_cva_innovations (Kalman-
refined, refine_iters=1) vs classical H1. Same noise/duration grid as
noise_injection_sweep.py, so results are directly comparable to that run.

This is the acid test for the "does an explicit error term help" design
question raised earlier: does the innovations-form refinement recover any
of the noise robustness the deterministic-only fit was shown to lack?
"""
import os
import sys
import time
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_v2, global_cva_innovations, frf_from_ss, modes_from_A
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK = 40, 66
REFINE_ITERS = 1

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt_band = info['H_gt_band']


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


def add_noise(y_clean, frac, seed):
    rng = np.random.RandomState(seed)
    rms = np.sqrt(np.mean(y_clean ** 2, axis=1, keepdims=True))
    return y_clean + rng.randn(*y_clean.shape) * (frac * rms)


def rel_err_of(A, B, C, D):
    H = frf_from_ss(A, B, C, D, f_band, DT)
    return float(np.linalg.norm(H - H_gt_band) / np.linalg.norm(H_gt_band))


def evaluate(u, y):
    out = {}
    try:
        rv2 = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
        out['cva_rel_err'] = rel_err_of(rv2['A'], rv2['B'], rv2['C'], rv2['D'])
    except Exception as ex:
        out['cva_rel_err'] = float('nan')

    try:
        rinn = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK,
                                       refine_iters=REFINE_ITERS)
        out['inn_rel_err'] = rel_err_of(rinn['A'], rinn['B'], rinn['C'], rinn['D'])
        hist = rinn['refine_history']
        out['inn_stable'] = bool(hist) and hist[-1].get('ok', False) and hist[-1]['max_eig'] < 1.0
    except Exception as ex:
        out['inn_rel_err'] = float('nan')
        out['inn_err'] = str(ex)
        out['inn_stable'] = False

    nperseg = min(1024, u.shape[1] // 2)
    f_w, H_h1 = h1_estimate(y, u, fs=FS, nperseg=nperseg)
    H_h1_interp = np.zeros((len(f_band), 8, 6), dtype=complex)
    for a in range(8):
        for b in range(6):
            H_h1_interp[:, a, b] = np.interp(f_band, f_w, H_h1[:, a, b].real) + \
                                    1j * np.interp(f_band, f_w, H_h1[:, a, b].imag)
    out['h1_rel_err'] = float(np.linalg.norm(H_h1_interp - H_gt_band) / np.linalg.norm(H_gt_band))
    return out


DURATIONS = [0.5, 1.0, 2.0, 4.0]
NOISE_FRACS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]

results = {d: {} for d in DURATIONS}
t0 = time.time()
for duration in DURATIONS:
    u, y_clean = generate_drive_response(sys_ss, duration, fs=FS)
    print(f"\nduration={duration:.1f}s")
    print(f"  {'noise':>6} | {'CVA (det)':>10} {'CVA (innov)':>12} {'H1':>10} | stable")
    print("  " + "-" * 58)
    for frac in NOISE_FRACS:
        y_noisy = y_clean if frac == 0.0 else add_noise(y_clean, frac, seed=hash((duration, frac)) % (2**31))
        res = evaluate(u, y_noisy)
        results[duration][frac] = res
        print(f"  {frac:>6.2f} | {res['cva_rel_err']:>10.4f} {res['inn_rel_err']:>12.4f} "
              f"{res['h1_rel_err']:>10.4f} | {res.get('inn_stable')}")

print(f"\ntotal time: {time.time()-t0:.1f}s")

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
for ax, duration in zip(axes.ravel(), DURATIONS):
    fracs = NOISE_FRACS
    cva_errs = [results[duration][f]['cva_rel_err'] for f in fracs]
    inn_errs = [results[duration][f]['inn_rel_err'] for f in fracs]
    h1_errs = [results[duration][f]['h1_rel_err'] for f in fracs]
    ax.plot(fracs, cva_errs, 'o-', color='tab:red', label='CVA (deterministic)')
    ax.plot(fracs, inn_errs, '^-', color='tab:green', label='CVA (innovations, 1 refine)')
    ax.plot(fracs, h1_errs, 's-', color='tab:blue', label='H1')
    ax.set_title(f"duration = {duration:.1f}s")
    ax.set_xlabel("sensor noise (fraction of per-channel RMS)")
    ax.set_ylabel("FRF rel_err")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle(f"CVA deterministic vs innovations-form (Kalman-refined) vs H1, additive measurement noise\n"
             f"(lags={LAGS}, rank={RANK}, refine_iters={REFINE_ITERS})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
out_path = os.path.join(RESULTS, "global_cva_innovations_noise_comparison.png")
fig.savefig(out_path, dpi=120)
print(f"\nplot -> {out_path}")
