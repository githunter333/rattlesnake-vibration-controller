"""
Rank/lags sensitivity sweep for global_cva_v2, plus a check of the automatic
(rank=None) order-selection heuristic against the known-correct rank=66.

Metrics per run:
  rel_err   : ||H_cva - H_gt|| / ||H_gt|| over the full 8x6, 100-1000 Hz band
  n_modes   : modes recovered by modes_from_A (stable, 20 Hz < f < 1100 Hz)
  max_eig   : largest |eigenvalue| of the identified A (stability margin)
  zeta_bias : median(zeta_id/zeta_true) over matched modes (see
              check_damping_bias.py) -- >1 means over-damped, i.e. peaks read low
  n_resolved_close_pairs : of the true-mode pairs <5 Hz apart in-band, how
              many get two DISTINCT identified poles rather than collapsing
              onto one (this is what produced the 438/788 Hz zeta blow-ups
              in check_damping_bias.py)
"""
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_v2, frf_from_ss, modes_from_A
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS

sysinfo = build_system()
sys_ss = sysinfo['sys_ss']
fn_true_band = sysinfo['fn_true_band']
zeta_true_band = sysinfo['zeta_true_band']
f_band = sysinfo['f_band']
H_gt_band = sysinfo['H_gt_band']

gaps = np.diff(fn_true_band)
close_pair_idx = np.where(gaps < 5.0)[0]   # index i means (fn_true_band[i], fn_true_band[i+1]) are close
print(f"true modes in band: {len(fn_true_band)}, close pairs (<5 Hz apart): {len(close_pair_idx)}")
for i in close_pair_idx:
    print(f"    {fn_true_band[i]:.2f} Hz <-> {fn_true_band[i+1]:.2f} Hz  (gap {gaps[i]:.2f} Hz)")

_cache = {}
def get_uy(duration):
    if duration not in _cache:
        _cache[duration] = generate_drive_response(sys_ss, duration, fs=FS)
    return _cache[duration]


def evaluate(u, y, lags, rank):
    t0 = time.time()
    try:
        r = global_cva_v2(y, u, lags=lags, tol=1e-10, rank=rank)
    except Exception as e:
        return dict(ok=False, err=str(e))
    elapsed = time.time() - t0
    H_cva = frf_from_ss(r['A'], r['B'], r['C'], r['D'], f_band, DT)
    rel_err = np.linalg.norm(H_cva - H_gt_band) / np.linalg.norm(H_gt_band)
    fr_id, zeta_id, max_eig = modes_from_A(r['A'], DT, f_hi=1100)

    # match true -> nearest identified, track zeta ratio and close-pair resolution
    ratios = []
    for ft, zt in zip(fn_true_band, zeta_true_band):
        if len(fr_id) == 0:
            break
        j = np.argmin(np.abs(fr_id - ft))
        if zt > 0:
            ratios.append(zeta_id[j] / zt)
    zeta_bias = float(np.median(ratios)) if ratios else float('nan')

    n_resolved = 0
    for i in close_pair_idx:
        ft1, ft2 = fn_true_band[i], fn_true_band[i + 1]
        if len(fr_id) == 0:
            continue
        j1 = np.argmin(np.abs(fr_id - ft1))
        j2 = np.argmin(np.abs(fr_id - ft2))
        if j1 != j2:
            n_resolved += 1

    return dict(ok=True, rel_err=rel_err, n_modes=len(fr_id), max_eig=max_eig,
                zeta_bias=zeta_bias, n_resolved=n_resolved, rank_used=r['rank'], t=elapsed)


def fmt(res):
    if not res['ok']:
        return f"FAILED: {res['err']}"
    return (f"rel_err={res['rel_err']:.4f}  n_modes={res['n_modes']:2d}  "
            f"max|eig|={res['max_eig']:.5f}  zeta_bias={res['zeta_bias']:.3f}  "
            f"close_pairs_resolved={res['n_resolved']}/{len(close_pair_idx)}  "
            f"({res['t']:.1f}s)")


PART = sys.argv[1] if len(sys.argv) > 1 else "ALL"

if PART in ("A", "ALL"):
 print("\n" + "=" * 78)
 print("PART A -- automatic rank heuristic (rank=None) vs forced rank=66, lags=40")
 print("=" * 78)
 for duration in [0.5, 1.0, 2.0, 4.0]:
  u, y = get_uy(duration)
  auto = evaluate(u, y, lags=40, rank=None)
  forced = evaluate(u, y, lags=40, rank=66)
  auto_rank = auto.get('rank_used', 'FAIL')
  print(f"\nduration={duration:>4.1f}s")
  print(f"  auto  (rank_used={auto_rank}):  {fmt(auto)}")
  print(f"  rank=66:                {fmt(forced)}")

if PART in ("B", "ALL"):
 print("\n" + "=" * 78)
 print("PART B -- lags sweep, rank=66 fixed")
 print("=" * 78)
 for duration in [1.0, 2.0]:
  u, y = get_uy(duration)
  print(f"\nduration={duration:.1f}s")
  for lags in [20, 30, 40, 50, 60, 80, 100]:
   res = evaluate(u, y, lags=lags, rank=66)
   print(f"  lags={lags:>4d}:  {fmt(res)}")

if PART in ("C", "ALL"):
 print("\n" + "=" * 78)
 print("PART C -- rank sweep, lags=40 fixed")
 print("=" * 78)
 for duration in [1.0, 2.0]:
  u, y = get_uy(duration)
  print(f"\nduration={duration:.1f}s")
  for rank in [50, 58, 66, 74, 82, 90, 100, 120]:
   res = evaluate(u, y, lags=40, rank=rank)
   print(f"  rank={rank:>4d}:  {fmt(res)}")
