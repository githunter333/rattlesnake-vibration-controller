"""
Noise-injection sweep: how does global_cva_v2 (deterministic-only, no
explicit innovations/error term) degrade under additive measurement
(sensor) noise, vs classical H1, at the validated defaults (lags=40,
rank=66)?

Motivated by a design question: many CVA formulations include an explicit
stochastic/innovations term (x(k+1)=Ax+Bu+Ke(k), y(k)=Cx+Du+e(k)), which
global_cva_v2 omits (pure output-error fit). The existing validation is
noise-free, so that omission costs nothing there. This script adds
per-channel-RMS-relative white sensor noise to the response and re-runs
both estimators, to find out empirically whether/where the missing
innovations term becomes a real liability -- before spending the effort to
build one.

Noise model: for each output channel i, additive white Gaussian noise with
std = noise_frac * rms(y_clean[i, :]) (i.e. noise_frac is roughly an
inverse SNR per channel, since channels differ in amplitude by ~2 orders
of magnitude -- see global_cva_frf_pairs.png).
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
from global_cva_frf import global_cva_v2, frf_from_ss, modes_from_A
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK = 40, 66

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt_band = info['H_gt_band']
fn_true_band = info['fn_true_band']
zeta_true_band = info['zeta_true_band']


def h1_estimate(y, u, fs, nperseg=1024):
    """Same conjugation convention verified in validate_global_cva.py:
    csd(a,b) = E[conj(A) B]; H1 needs Syu[i,j]=E[Y_i conj(U_j)]=csd(u_j,y_i)."""
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


def evaluate(u, y, duration):
    out = {}
    try:
        r = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
        H_cva = frf_from_ss(r['A'], r['B'], r['C'], r['D'], f_band, DT)
        out['cva_rel_err'] = float(np.linalg.norm(H_cva - H_gt_band) / np.linalg.norm(H_gt_band))
        fr_id, zeta_id, max_eig = modes_from_A(r['A'], DT, f_hi=1100)
        ratios = []
        for ft, zt in zip(fn_true_band, zeta_true_band):
            if len(fr_id) == 0:
                break
            j = np.argmin(np.abs(fr_id - ft))
            if zt > 0:
                ratios.append(zeta_id[j] / zt)
        out['cva_zeta_bias'] = float(np.median(ratios)) if ratios else float('nan')
        out['cva_n_modes'] = len(fr_id)
        out['cva_max_eig'] = float(max_eig)
    except Exception as e:
        out['cva_rel_err'] = float('nan')
        out['cva_zeta_bias'] = float('nan')
        out['cva_n_modes'] = 0
        out['cva_max_eig'] = float('nan')
        out['cva_err'] = str(e)

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
t_start = time.time()
for duration in DURATIONS:
    u, y_clean = generate_drive_response(sys_ss, duration, fs=FS)
    print(f"\nduration={duration:.1f}s")
    print(f"  {'noise_frac':>10} | {'CVA rel_err':>11} {'zeta_bias':>9} {'n_modes':>7} {'max|eig|':>9} | {'H1 rel_err':>10}")
    print("  " + "-" * 72)
    for frac in NOISE_FRACS:
        y_noisy = y_clean if frac == 0.0 else add_noise(y_clean, frac, seed=hash((duration, frac)) % (2**31))
        res = evaluate(u, y_noisy, duration)
        results[duration][frac] = res
        print(f"  {frac:>10.2f} | {res['cva_rel_err']:>11.4f} {res['cva_zeta_bias']:>9.3f} "
              f"{res['cva_n_modes']:>7d} {res['cva_max_eig']:>9.5f} | {res['h1_rel_err']:>10.4f}")

print(f"\ntotal time: {time.time()-t_start:.1f}s")

# --- plot: rel_err vs noise_frac, CVA vs H1, one panel per duration ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=False)
for ax, duration in zip(axes.ravel(), DURATIONS):
    fracs = NOISE_FRACS
    cva_errs = [results[duration][f]['cva_rel_err'] for f in fracs]
    h1_errs = [results[duration][f]['h1_rel_err'] for f in fracs]
    ax.plot(fracs, cva_errs, 'o-', color='tab:red', label='CVA (no error term)')
    ax.plot(fracs, h1_errs, 's-', color='tab:blue', label='H1')
    ax.set_title(f"duration = {duration:.1f}s")
    ax.set_xlabel("sensor noise (fraction of per-channel RMS)")
    ax.set_ylabel("FRF rel_err")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(f"CVA (deterministic, lags={LAGS}, rank={RANK}) vs H1 under additive measurement noise",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
out_path = os.path.join(RESULTS, "global_cva_noise_injection_sweep.png")
fig.savefig(out_path, dpi=120)
print(f"\nplot -> {out_path}")
