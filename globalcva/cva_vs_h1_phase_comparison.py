"""
Direct CVA-vs-H1 comparison against known ground truth, per response/drive
channel pair, magnitude AND phase separately -- requested 2026-09-01 to
pin down the FORM of the error seen in the live no-FRF-update control run
(design doc section 15), since that run's sysid_frf was fit OPEN LOOP (no
feedback contamination) on the simulated (effectively noise-free) 6-drive/
12-response/8-control virtual hardware.

Reuses the same machinery as validate_global_cva.py / build_true_system.py
(same linear ground-truth system, same fs=5120, same broadband amp=0.05
excitation) but:
  - calls global_cva_innovations (refine_iters=1), matching what
    spectral_processing.py's _run_cva_processing ACTUALLY calls live, not
    the plain global_cva_v2 the original validation table used
  - uses cva_window_seconds=2.0 (the live default in
    SpectralProcessingMetadata), matching the live sys-ID window
  - breaks the error down PER RESPONSE CHANNEL (mean/max over the 6 drives)
    in both magnitude (dB) and phase (deg), for both CVA and H1, so it's
    directly comparable to section 15's self-predicted-vs-actual per-
    channel table
  - also computes CVA-vs-H1 direct disagreement (not just vs truth) since
    that's the "form of the error" question

ASSUMPTIONS (flagged explicitly, not verified against the live GUI config):
  - the live runs used the base LINEAR ground-truth system
    (sdynpy_frame6x12_system.npz), not the shifted/nonlinear variants that
    also exist in this results folder
  - fs=5120 Hz, broadband randn excitation at the same amplitude scale as
    the existing validation scripts
  - H1 uses nperseg=1024, 50% overlap (matches the existing comparison
    scripts in this repo, not independently re-derived from the live H1
    pipeline's actual frame_size)
If any of these don't match the actual live configuration, the numbers
below characterize the METHOD's noise-free behavior, not necessarily the
literal live run -- worth re-running with corrected assumptions if this
doesn't reproduce the live pattern.
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
from build_true_system import build_system, generate_drive_response
from global_cva_frf import global_cva_innovations, frf_from_ss

RESULTS_DIR = os.path.join(REPO, "examples", "sixdrive12resp", "results")
OUT_DIR = _HERE

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK, REFINE_ITERS = 40, 66, 1     # live validated defaults
WINDOW_SECONDS = 2.0                      # live cva_window_seconds default
H1_NPERSEG = 1024

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt = info['H_gt_band']         # (F, 8, 6)
nout, nin = H_gt.shape[1], H_gt.shape[2]


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
    """|H_est|/|H_ref| in dB, per bin/channel."""
    return 20 * np.log10(np.maximum(np.abs(H_est), 1e-30) / np.maximum(np.abs(H_ref), 1e-30))


def phase_err_deg(H_est, H_ref):
    """Wrapped phase difference in degrees, per bin/channel (angle of the
    ratio -- avoids unwrap ambiguity, correct for a static per-bin diff)."""
    ratio = H_est * np.conj(H_ref) / np.maximum(np.abs(H_est) * np.abs(H_ref), 1e-30)
    return np.angle(ratio, deg=True)


def run_one(seed):
    u, y = generate_drive_response(sys_ss, WINDOW_SECONDS, fs=FS, seed=seed, amp=0.05)

    result = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK,
                                     refine_iters=REFINE_ITERS)
    H_cva = frf_from_ss(result['A'], result['B'], result['C'], result['D'], f_band, DT)

    f_w, H_h1_raw = h1_estimate(y, u, FS, H1_NPERSEG)
    H_h1 = interp_complex(f_w, H_h1_raw, f_band)

    return H_cva, H_h1


def summarize(H_cva, H_h1):
    mag_err_cva = mag_db_err(H_cva, H_gt)          # (F, nout, nin)
    mag_err_h1 = mag_db_err(H_h1, H_gt)
    phase_err_cva = phase_err_deg(H_cva, H_gt)
    phase_err_h1 = phase_err_deg(H_h1, H_gt)
    mag_disagree = mag_db_err(H_cva, H_h1)          # CVA vs H1 directly
    phase_disagree = phase_err_deg(H_cva, H_h1)

    print(f"\n{'resp':>5} | {'CVA magErr(dB)':>22} | {'CVA phaseErr(deg)':>24} "
          f"| {'H1 magErr(dB)':>22} | {'H1 phaseErr(deg)':>24} | {'CVAvH1 mag/phase':>20}")
    print("-" * 145)
    for r in range(nout):
        mc = np.abs(mag_err_cva[:, r, :])
        pc = np.abs(phase_err_cva[:, r, :])
        mh = np.abs(mag_err_h1[:, r, :])
        ph = np.abs(phase_err_h1[:, r, :])
        md = np.abs(mag_disagree[:, r, :])
        pd = np.abs(phase_disagree[:, r, :])
        print(f"resp{r+7:2d} | mean={np.mean(mc):6.2f} max={np.max(mc):6.2f}      "
              f"| mean={np.mean(pc):7.2f} max={np.max(pc):7.2f}        "
              f"| mean={np.mean(mh):6.2f} max={np.max(mh):6.2f}      "
              f"| mean={np.mean(ph):7.2f} max={np.max(ph):7.2f}        "
              f"| mag={np.mean(md):5.2f} phase={np.mean(pd):6.2f}")

    overall_cva_mag = np.linalg.norm(H_cva - H_gt) / np.linalg.norm(H_gt)
    overall_h1_mag = np.linalg.norm(H_h1 - H_gt) / np.linalg.norm(H_gt)
    print(f"\nOverall relative error (complex norm, matches validate_global_cva.py's metric):"
          f"  CVA={overall_cva_mag:.4f}  H1={overall_h1_mag:.4f}")
    return dict(mag_err_cva=mag_err_cva, mag_err_h1=mag_err_h1,
                phase_err_cva=phase_err_cva, phase_err_h1=phase_err_h1,
                mag_disagree=mag_disagree, phase_disagree=phase_disagree)


print(f"CVA (global_cva_innovations, lags={LAGS}, rank={RANK}, refine_iters={REFINE_ITERS}) "
      f"vs H1 (nperseg={H1_NPERSEG}) vs ground truth, {WINDOW_SECONDS}s window, fs={FS}, "
      f"noise-free simulated linear system")

all_seed_results = {}
for seed in [0, 1, 2]:
    print(f"\n{'='*100}\nseed={seed}")
    H_cva, H_h1 = run_one(seed)
    all_seed_results[seed] = summarize(H_cva, H_h1)
    if seed == 0:
        H_cva_0, H_h1_0 = H_cva, H_h1

# --- identify worst and best CVA response channels (seed 0) for plotting ---
mag_err_cva_0 = np.abs(all_seed_results[0]['mag_err_cva'])
per_resp_mean_err = mag_err_cva_0.mean(axis=(0, 2))
worst_resp = int(np.argmax(per_resp_mean_err))
best_resp = int(np.argmin(per_resp_mean_err))
print(f"\nSeed 0: worst CVA response channel = resp{worst_resp+7} "
      f"(mean|magErr|={per_resp_mean_err[worst_resp]:.2f} dB), "
      f"best = resp{best_resp+7} (mean|magErr|={per_resp_mean_err[best_resp]:.2f} dB)")

# --- plot worst and best channel, magnitude + phase, CVA vs H1 vs truth ---
fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
for col, (resp_i, label) in enumerate([(worst_resp, 'worst'), (best_resp, 'best')]):
    drive_i = int(np.argmax(np.abs(H_gt[:, resp_i, :]).mean(axis=0)))
    ax_mag, ax_ph = axes[0, col], axes[1, col]
    ax_mag.semilogy(f_band, np.abs(H_gt[:, resp_i, drive_i]), 'k-', lw=2, label='truth')
    ax_mag.semilogy(f_band, np.abs(H_cva_0[:, resp_i, drive_i]), lw=1.2, alpha=0.85, label='CVA')
    ax_mag.semilogy(f_band, np.abs(H_h1_0[:, resp_i, drive_i]), lw=1.2, alpha=0.85, label='H1')
    ax_mag.set_title(f'resp{resp_i+7}/drive{drive_i+1} ({label} CVA channel)')
    ax_mag.legend(fontsize=8); ax_mag.grid(True, alpha=0.3)
    ax_mag.set_ylabel('|H|')

    ph_gt = np.unwrap(np.angle(H_gt[:, resp_i, drive_i])) * 180 / np.pi
    ph_cva = np.unwrap(np.angle(H_cva_0[:, resp_i, drive_i])) * 180 / np.pi
    ph_h1 = np.unwrap(np.angle(H_h1_0[:, resp_i, drive_i])) * 180 / np.pi
    ax_ph.plot(f_band, ph_gt, 'k-', lw=2, label='truth')
    ax_ph.plot(f_band, ph_cva, lw=1.2, alpha=0.85, label='CVA')
    ax_ph.plot(f_band, ph_h1, lw=1.2, alpha=0.85, label='H1')
    ax_ph.set_ylabel('phase (deg)'); ax_ph.grid(True, alpha=0.3)

    ax_magerr = axes[2, col]
    ax_magerr.plot(f_band, mag_db_err(H_cva_0[:, resp_i, drive_i], H_gt[:, resp_i, drive_i]),
                    label='CVA magErr (dB)')
    ax_magerr.plot(f_band, mag_db_err(H_h1_0[:, resp_i, drive_i], H_gt[:, resp_i, drive_i]),
                    label='H1 magErr (dB)')
    ax_magerr.axhline(0, color='gray', lw=0.5)
    ax_magerr.legend(fontsize=8); ax_magerr.grid(True, alpha=0.3)
    ax_magerr.set_ylabel('mag err (dB)')

    ax_pherr = axes[3, col]
    ax_pherr.plot(f_band, phase_err_deg(H_cva_0[:, resp_i, drive_i], H_gt[:, resp_i, drive_i]),
                   label='CVA phaseErr (deg)')
    ax_pherr.plot(f_band, phase_err_deg(H_h1_0[:, resp_i, drive_i], H_gt[:, resp_i, drive_i]),
                   label='H1 phaseErr (deg)')
    ax_pherr.axhline(0, color='gray', lw=0.5)
    ax_pherr.legend(fontsize=8); ax_pherr.grid(True, alpha=0.3)
    ax_pherr.set_ylabel('phase err (deg)'); ax_pherr.set_xlabel('Frequency (Hz)')

fig.suptitle(f'CVA vs H1 vs ground truth -- {WINDOW_SECONDS}s window, noise-free sim, seed=0')
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "cva_vs_h1_phase_comparison.png")
fig.savefig(out_png, dpi=120)
print(f"\nplot -> {out_png}")
