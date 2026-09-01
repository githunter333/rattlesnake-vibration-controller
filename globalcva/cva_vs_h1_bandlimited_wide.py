"""
Same CVA-vs-H1-vs-truth comparison as cva_vs_h1_phase_comparison.py, but
with the excitation changed to match what the real spec file actually
prescribes: flat_spec_frame6x12.mat (loaded by
random_vibration_sys_id_utilities.load_specification, confirmed
2026-09-01 as Nc=8, 100-1000 Hz flat band, 0.001 g^2/Hz, zero elsewhere)
is BAND-LIMITED to 100-1000 Hz, not full-bandwidth (0-2560 Hz) white
noise like the earlier offline reproduction (and validate_global_cva.py)
used. That's a real, confirmed structural difference between the offline
reproduction and the live test, worth checking directly: does CVA's
noise-free advantage over H1 survive band-limited excitation, or does it
shrink/reverse?

Excitation is synthesized directly in the frequency domain per drive
channel (flat magnitude within [100,1000] Hz, zero outside, random
phase, single realization) rather than reproducing the live
COLA-block/SVD-coloring machinery in signal_generation.py -- this
captures the band-limiting itself, which is the property being tested,
without needing to replicate the streaming/blocking implementation
details.
"""
import os
import sys
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RATTLESNAKE_REPO") or os.path.expanduser("~/mnt/rattlesnake-vibration-controller")
GLOBALCVA = os.path.join(REPO, "globalcva")
sys.path.insert(0, GLOBALCVA)
from build_true_system import build_system
from global_cva_frf import global_cva_innovations, frf_from_ss

RESULTS_DIR = os.path.join(REPO, "examples", "sixdrive12resp", "results")
OUT_DIR = _HERE

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK, REFINE_ITERS = 40, 66, 1
WINDOW_SECONDS = 2.0
H1_NPERSEG = 1024
F_LO, F_HI = 20.0, 2000.0

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt = info['H_gt_band']
nout, nin = H_gt.shape[1], H_gt.shape[2]


def bandlimited_excitation(n, nin, fs, f_lo, f_hi, amp, seed):
    """Flat magnitude in [f_lo, f_hi], zero elsewhere, random phase,
    real time-domain signal via irfft -- one clean realization per
    channel, independent across channels (matches the flat_spec's
    zero off-diagonal)."""
    rng = np.random.default_rng(seed)
    nfreq = n // 2 + 1
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    in_band = (freqs >= f_lo) & (freqs <= f_hi)
    u = np.zeros((nin, n))
    for ch in range(nin):
        mag = in_band.astype(float)
        phase = rng.uniform(0, 2 * np.pi, size=nfreq)
        Xv = mag * np.exp(1j * phase)
        Xv[0] = 0
        Xv[-1] = 0
        x = np.fft.irfft(Xv, n=n)
        x = x / np.std(x) * amp
        u[ch] = x
    return u


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


def interp_complex(f_src, H_src, f_dst):
    out = np.zeros((len(f_dst),) + H_src.shape[1:], dtype=complex)
    for a in range(H_src.shape[1]):
        for b in range(H_src.shape[2]):
            out[:, a, b] = (np.interp(f_dst, f_src, H_src[:, a, b].real) +
                             1j * np.interp(f_dst, f_src, H_src[:, a, b].imag))
    return out


def mag_db_err(H_est, H_ref):
    return 20 * np.log10(np.maximum(np.abs(H_est), 1e-30) / np.maximum(np.abs(H_ref), 1e-30))


def phase_err_deg(H_est, H_ref):
    ratio = H_est * np.conj(H_ref) / np.maximum(np.abs(H_est) * np.abs(H_ref), 1e-30)
    return np.angle(ratio, deg=True)


def run_one(seed):
    n = int(WINDOW_SECONDS * FS)
    u = bandlimited_excitation(n, sys_ss.B.shape[1], FS, F_LO, F_HI, amp=0.05, seed=seed)
    t = np.arange(n) / FS
    _, yf, _ = sig.lsim(sys_ss, u.T, t, np.zeros(sys_ss.A.shape[0]))
    y = yf[:, :].T

    result = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK,
                                     refine_iters=REFINE_ITERS)
    H_cva = frf_from_ss(result['A'], result['B'], result['C'], result['D'], f_band, DT)

    f_w, H_h1_raw = h1_estimate(y, u, FS, H1_NPERSEG)
    H_h1 = interp_complex(f_w, H_h1_raw, f_band)
    return H_cva, H_h1, result


def summarize(H_cva, H_h1, label):
    mag_err_cva = mag_db_err(H_cva, H_gt)
    mag_err_h1 = mag_db_err(H_h1, H_gt)
    phase_err_cva = phase_err_deg(H_cva, H_gt)
    phase_err_h1 = phase_err_deg(H_h1, H_gt)

    print(f"\n--- {label} ---")
    print(f"{'resp':>5} | {'CVA magErr(dB)':>18} | {'CVA phaseErr(deg)':>20} "
          f"| {'H1 magErr(dB)':>18} | {'H1 phaseErr(deg)':>20}")
    for r in range(nout):
        mc = np.abs(mag_err_cva[:, r, :]); pc = np.abs(phase_err_cva[:, r, :])
        mh = np.abs(mag_err_h1[:, r, :]); ph = np.abs(phase_err_h1[:, r, :])
        print(f"resp{r+7:2d} | mean={np.mean(mc):5.2f} max={np.max(mc):6.2f} "
              f"| mean={np.mean(pc):6.2f} max={np.max(pc):7.2f}   "
              f"| mean={np.mean(mh):5.2f} max={np.max(mh):6.2f} "
              f"| mean={np.mean(ph):6.2f} max={np.max(ph):7.2f}")
    overall_cva = np.linalg.norm(H_cva - H_gt) / np.linalg.norm(H_gt)
    overall_h1 = np.linalg.norm(H_h1 - H_gt) / np.linalg.norm(H_gt)
    print(f"Overall relative error: CVA={overall_cva:.4f}  H1={overall_h1:.4f}")


print(f"Band-limited ({F_LO}-{F_HI} Hz) excitation, {WINDOW_SECONDS}s window, "
      f"CVA (lags={LAGS},rank={RANK},refine_iters={REFINE_ITERS}) vs H1 (nperseg={H1_NPERSEG}) "
      f"vs ground truth, noise-free")

results = {}
for seed in [0, 1, 2]:
    H_cva, H_h1, r = run_one(seed)
    print(f"\nseed={seed}: CVA fit rank(cc-selected)={r['rank']}")
    summarize(H_cva, H_h1, f"seed={seed}")
    results[seed] = (H_cva, H_h1)

# quick plot: worst CVA channel at seed 0 under band-limited excitation
H_cva_0, H_h1_0 = results[0]
mag_err_cva_0 = np.abs(mag_db_err(H_cva_0, H_gt))
per_resp = mag_err_cva_0.mean(axis=(0, 2))
worst = int(np.argmax(per_resp))
drive_i = int(np.argmax(np.abs(H_gt[:, worst, :]).mean(axis=0)))

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].semilogy(f_band, np.abs(H_gt[:, worst, drive_i]), 'k-', lw=2, label='truth')
axes[0].semilogy(f_band, np.abs(H_cva_0[:, worst, drive_i]), lw=1.2, alpha=0.85, label='CVA')
axes[0].semilogy(f_band, np.abs(H_h1_0[:, worst, drive_i]), lw=1.2, alpha=0.85, label='H1')
axes[0].set_title(f'Band-limited excitation: resp{worst+7}/drive{drive_i+1} (worst CVA channel)')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3); axes[0].set_ylabel('|H|')
axes[1].plot(f_band, mag_db_err(H_cva_0[:, worst, drive_i], H_gt[:, worst, drive_i]), label='CVA magErr (dB)')
axes[1].plot(f_band, mag_db_err(H_h1_0[:, worst, drive_i], H_gt[:, worst, drive_i]), label='H1 magErr (dB)')
axes[1].axhline(0, color='gray', lw=0.5)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
axes[1].set_ylabel('mag err (dB)'); axes[1].set_xlabel('Frequency (Hz)')
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "cva_vs_h1_bandlimited.png")
fig.savefig(out_png, dpi=120)
print(f"\nplot -> {out_png}")
