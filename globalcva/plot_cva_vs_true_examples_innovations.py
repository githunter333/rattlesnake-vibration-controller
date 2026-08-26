"""
Same layout as plot_cva_vs_true_examples.py (same 6 drive/response pairs,
same 1s duration, lags=40, rank=66) but using global_cva_innovations
(Kalman-refined, refine_iters=1) instead of global_cva_v2, for a direct
visual before/after against global_cva_frf_pairs.png.
"""
import os
import sys
import numpy as np
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
DURATION = 1.0

info = build_system()
sys_ss = info['sys_ss']
f_band = info['f_band']
H_gt_band = info['H_gt_band']

u, y = generate_drive_response(sys_ss, DURATION, fs=FS)
r = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK, refine_iters=1)
H_cva = frf_from_ss(r['A'], r['B'], r['C'], r['D'], f_band, DT)

pairs = [(8, 1), (10, 3), (12, 5), (13, 6), (14, 1), (7, 4)]

fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True)
for ax, (resp_node, drive_node) in zip(axes.ravel(), pairs):
    ri, di = resp_node - 7, drive_node - 1
    h_true = H_gt_band[:, ri, di]
    h_cva = H_cva[:, ri, di]
    rel_err_pair = np.linalg.norm(h_cva - h_true) / np.linalg.norm(h_true)
    ax.semilogy(f_band, np.abs(h_true), 'k-', lw=1.8, label='true')
    ax.semilogy(f_band, np.abs(h_cva), color='tab:green', lw=1.0, alpha=0.9,
                label='CVA innovations (1 s)')
    ax.set_title(f"resp node {resp_node} / drive node {drive_node}   (rel err {rel_err_pair:.3f})",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel("Frequency (Hz)")
for ax in axes[:, 0]:
    ax.set_ylabel("|H|")

fig.suptitle(f"Global CVA innovations-form vs true FRF -- {DURATION:.0f}s broadband, "
             f"lags={LAGS}, rank={RANK}, refine_iters=1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
out = os.path.join(RESULTS, "global_cva_frf_pairs_innovations.png")
fig.savefig(out, dpi=120)
print(f"saved -> {out}")

overall_rel_err = np.linalg.norm(H_cva - H_gt_band) / np.linalg.norm(H_gt_band)
print(f"overall rel_err (all 8x6 pairs, {DURATION:.0f}s, lags={LAGS}, rank={RANK}, innovations): {overall_rel_err:.4f}")
