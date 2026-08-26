"""
Same check as check_damping_bias.py (TRUE per-mode damping ratios, from
the raw M,K,C matrices, vs identified damping ratios), but comparing BOTH
global_cva_v2 (deterministic) and global_cva_innovations (Kalman-refined)
side by side, to see whether the innovations-form refinement fixes the
over-damping bias / close-pair-collapse issue from check_damping_bias.py,
not just the aggregate FRF rel_err improvement already shown in
noise_injection_compare_innovations.py.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_v2, global_cva_innovations, modes_from_A
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK = 40, 66
DURATION = 1.0

info = build_system()
u, y = generate_drive_response(info['sys_ss'], DURATION, fs=FS)
fn_true_band = info['fn_true_band']
zeta_true_band = info['zeta_true_band']

gaps = np.diff(fn_true_band)
close_pair_idx = np.where(gaps < 5.0)[0]

r_det = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
r_inn = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK, refine_iters=1)

fr_det, zeta_det, maxeig_det = modes_from_A(r_det['A'], DT, f_hi=1100)
fr_inn, zeta_inn, maxeig_inn = modes_from_A(r_inn['A'], DT, f_hi=1100)

print(f"duration={DURATION}s, lags={LAGS}, rank={RANK}")
print(f"n_modes: det={len(fr_det)}  innovations={len(fr_inn)}   "
      f"max|eig|: det={maxeig_det:.5f}  innovations={maxeig_inn:.5f}")

print(f"\n{'true f':>8} {'true zeta':>9} | {'det f':>8} {'det zeta':>9} {'det ratio':>9} | "
      f"{'inn f':>8} {'inn zeta':>9} {'inn ratio':>9}")
print("-" * 92)

det_ratios, inn_ratios = [], []
det_close_resolved, inn_close_resolved = 0, 0

for ft, zt in zip(fn_true_band, zeta_true_band):
    jd = np.argmin(np.abs(fr_det - ft)) if len(fr_det) else None
    ji = np.argmin(np.abs(fr_inn - ft)) if len(fr_inn) else None
    rd = zeta_det[jd] / zt if (jd is not None and zt > 0) else float('nan')
    ri = zeta_inn[ji] / zt if (ji is not None and zt > 0) else float('nan')
    det_ratios.append(rd)
    inn_ratios.append(ri)
    print(f"{ft:>8.2f} {zt:>9.5f} | {fr_det[jd]:>8.2f} {zeta_det[jd]:>9.5f} {rd:>9.3f} | "
          f"{fr_inn[ji]:>8.2f} {zeta_inn[ji]:>9.5f} {ri:>9.3f}")

det_ratios = np.array(det_ratios)
inn_ratios = np.array(inn_ratios)

for i in close_pair_idx:
    ft1, ft2 = fn_true_band[i], fn_true_band[i + 1]
    jd1 = np.argmin(np.abs(fr_det - ft1)); jd2 = np.argmin(np.abs(fr_det - ft2))
    ji1 = np.argmin(np.abs(fr_inn - ft1)); ji2 = np.argmin(np.abs(fr_inn - ft2))
    if jd1 != jd2:
        det_close_resolved += 1
    if ji1 != ji2:
        inn_close_resolved += 1

print(f"\n{'':22} {'deterministic':>15} {'innovations':>15}")
print(f"{'median zeta ratio':22} {np.median(det_ratios):>15.3f} {np.median(inn_ratios):>15.3f}")
print(f"{'mean zeta ratio':22} {np.mean(det_ratios):>15.3f} {np.mean(inn_ratios):>15.3f}")
print(f"{'fraction over-damped':22} {np.mean(det_ratios > 1):>15.2f} {np.mean(inn_ratios > 1):>15.2f}")
print(f"{'close pairs resolved':22} {det_close_resolved:>12d}/{len(close_pair_idx)} "
      f"{inn_close_resolved:>12d}/{len(close_pair_idx)}")
