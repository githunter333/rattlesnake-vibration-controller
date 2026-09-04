#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nonlinear_frf_system_allmodes.py

Generalizes build_nonlinear_frf_system.py (mode 1 only) to ALL 33 flexible
modes -- motivated by the fact that a real test article's nonlinearity
won't politely confine itself to one mode; whatever FRF-tracking strategy
is being evaluated needs to cope with all modes shifting some amount, not
just one known mode by a known amount.

Per-mode nonlinearity targets follow the same pattern as
build_shifted_frf_system_allmodes.py's (linear) modal shifts: first 4
modes get a deterministic target, the remaining 29 get randomized targets
(fixed RNG seed for reproducibility) -- so the calibration has to be
robust across a real spread of (frequency, damping, target) combinations,
not hand-tuned for one mode's numbers the way the single-mode script's
brackets originally were.

Each mode gets its own (k3_i, c2_i) via the SAME numerical free-decay
root-solving approach as the single-mode script (not first-order Duffing
formulas), and its own reference amplitude derived from a shared reference
drive level assumption (mode-shaped excitation at REFERENCE_DRIVE_LEVEL_
VRMS, same idea as the single-mode script -- necessarily an idealization,
since a real broadband random test's actual per-mode excitation efficiency
differs from a clean single-mode-shaped sine, same caveat as before).

Run (sdynpy or rattlesnake env -- only needs numpy/scipy):
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python build_nonlinear_frf_system_allmodes.py

Output: ../results/sdynpy_frame6x12_system_nonlinear_allmodes.npz
"""

import os
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
BASELINE_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system_nonlinear_allmodes.npz")

N_RIGID = 3
N_DETERMINISTIC = 4
DET_FREQ_SHIFT_AT_2X = -0.04          # first 4 modes: deterministic target (matches the single-mode script)
DET_ZETA_RATIO_Q0_TO_2Q0 = 1.20
RAND_FREQ_SHIFT_MEAN = -0.03          # remaining 29 modes: randomized targets (still softening, still
RAND_FREQ_SHIFT_STD = 0.015           # damping-increasing -- same qualitative character, randomized magnitude)
RAND_ZETA_RATIO_MEAN = 1.15
RAND_ZETA_RATIO_STD = 0.08
ZETA_RATIO_FLOOR = 1.02               # keep it a genuine (if sometimes small) increase, not a decrease
FREQ_SHIFT_CEIL = -0.005              # keep it a genuine (if sometimes small) softening, not zero/hardening
RNG_SEED = 42

DRIVE_DOF_INDICES = [0, 2, 4, 6, 8, 10]   # nodes 1-6, X-direction (see build_sdynpy_demo_frame6x12.py's dofs())
REFERENCE_DRIVE_LEVEL_VRMS = 2.5   # rescaled to match observed closed-loop drive RMS (~1.7-5.6 V across -12 dB to 0 dB), was 20.0

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------
# 1. Load baseline, get every flexible mode's own mass-normalized
#    eigenvector, frequency, and damping
# ---------------------------------------------------------------------
data = {key: val for key, val in np.load(BASELINE_FILE).items()}
M, C, K = data['mass'], data['damping'], data['stiffness']

eigvals, eigvecs = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)
freq_hz_all = np.sqrt(eigvals) / (2 * np.pi)
n_flex = len(eigvals) - N_RIGID

wn_arr = np.zeros(n_flex)
zeta0_arr = np.zeros(n_flex)
phi_arr = np.zeros((K.shape[0], n_flex))
for m in range(n_flex):
    i = N_RIGID + m
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    wn_arr[m] = np.sqrt(eigvals[i])
    zeta0_arr[m] = (phi @ C @ phi) / (2 * np.sqrt((phi @ K @ phi) * mn))
    phi_arr[:, m] = phi

# ---------------------------------------------------------------------
# 2. Per-mode targets: first N_DETERMINISTIC deterministic, rest random
# ---------------------------------------------------------------------
target_freq_shift = np.empty(n_flex)
target_zeta_ratio = np.empty(n_flex)
for m in range(n_flex):
    if m < N_DETERMINISTIC:
        target_freq_shift[m] = DET_FREQ_SHIFT_AT_2X
        target_zeta_ratio[m] = DET_ZETA_RATIO_Q0_TO_2Q0
    else:
        target_freq_shift[m] = min(rng.normal(RAND_FREQ_SHIFT_MEAN, RAND_FREQ_SHIFT_STD), FREQ_SHIFT_CEIL)
        target_zeta_ratio[m] = max(rng.normal(RAND_ZETA_RATIO_MEAN, RAND_ZETA_RATIO_STD), ZETA_RATIO_FLOOR)

# ---------------------------------------------------------------------
# 3. Per-mode calibration (free-decay root-solving, robust brackets --
#    generalized from the single-mode script's, not hand-tuned per mode)
# ---------------------------------------------------------------------

def simulate_free_decay(q0, k3, c2, wn, zeta0, n_cycles=6):
    T0 = 2 * np.pi / wn
    t_span = (0, n_cycles * T0)
    t_eval = np.linspace(*t_span, 4000)

    def rhs(t, s):
        q, qd = s
        qdd = -2 * zeta0 * wn * qd - wn ** 2 * q - k3 * q ** 3 - c2 * qd * np.abs(qd)
        return [qd, qdd]

    sol = solve_ivp(rhs, t_span, [q0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]


def measure_freq_and_damping(q0, k3, c2, wn, zeta0):
    t, q = simulate_free_decay(q0, k3, c2, wn, zeta0)
    dq = np.diff(q)
    peak_idx = np.where((dq[:-1] > 0) & (dq[1:] <= 0))[0] + 1
    peak_idx = peak_idx[q[peak_idx] > 0]
    if len(peak_idx) < 3:
        return None, None
    t_peaks = t[peak_idx]
    q_peaks = q[peak_idx]
    period = t_peaks[1] - t_peaks[0]
    f_meas = 1.0 / period
    delta = np.log(q_peaks[0] / q_peaks[1])
    zeta_meas = delta / np.sqrt((2 * np.pi) ** 2 + delta ** 2)
    return f_meas, zeta_meas


def calibrate_mode(q0, wn, zeta0, target_shift, target_ratio):
    f_lin = wn / (2 * np.pi)

    def freq_shift_error(k3):
        f_hi, _ = measure_freq_and_damping(2 * q0, k3, 0.0, wn, zeta0)
        if f_hi is None:
            return np.nan
        return (f_hi - f_lin) / f_lin - target_shift

    k3_scale = abs(target_shift) * 8 * wn ** 2 / (3 * (2 * q0) ** 2)
    k3_stability_limit = wn ** 2 / (4 * q0 ** 2)
    # widen geometrically if needed, same robustness pattern as c2's search below
    for mult in [3, 2, 1, 0.5]:
        k3_lo = -min(mult * k3_scale, 0.5 * k3_stability_limit)
        err_lo = freq_shift_error(k3_lo)
        err_hi = freq_shift_error(0.0 + 1e-6 * k3_scale)  # avoid exact 0 (no-op) edge case
        if np.isfinite(err_lo) and np.isfinite(err_hi) and err_lo * err_hi < 0:
            k3 = brentq(freq_shift_error, k3_lo, 0.0, xtol=1e-6 * k3_scale)
            break
    else:
        raise RuntimeError(f"could not bracket k3 for wn={wn:.2f}, target_shift={target_shift}")

    def damping_ratio_error(c2):
        _, z_lo = measure_freq_and_damping(q0, k3, c2, wn, zeta0)
        _, z_hi = measure_freq_and_damping(2 * q0, k3, c2, wn, zeta0)
        if z_lo is None or z_hi is None or z_lo <= 0:
            return np.nan
        return z_hi / z_lo - target_ratio

    x_target = (target_ratio - 1.0) * 0.25 / 0.20 * zeta0  # scaled version of the single-mode script's
                                                             # x_target=0.25*zeta0 relation for a 1.20 ratio
    c2_scale = x_target * 3 * np.pi / (4 * q0)
    mults = np.geomspace(1e-4, 200, 100)
    errs = [damping_ratio_error(m * c2_scale) for m in mults]
    c2_lo = c2_hi = None
    for i in range(len(mults) - 1):
        if np.isfinite(errs[i]) and np.isfinite(errs[i + 1]) and errs[i] * errs[i + 1] < 0:
            c2_lo, c2_hi = mults[i] * c2_scale, mults[i + 1] * c2_scale
            break
    if c2_lo is None:
        raise RuntimeError(f"could not bracket c2 for wn={wn:.2f}, target_ratio={target_ratio}; errs={errs}")
    c2 = brentq(damping_ratio_error, c2_lo, c2_hi, xtol=1e-9 * max(c2_scale, 1e-12))

    return k3, c2


phi_drive = phi_arr[DRIVE_DOF_INDICES, :]              # (6, n_flex)
k3_arr = np.empty(n_flex)
c2_arr = np.empty(n_flex)
q0_ref_arr = np.empty(n_flex)

print(f"Calibrating {n_flex} modes (first {N_DETERMINISTIC} deterministic, "
      f"remaining {n_flex - N_DETERMINISTIC} randomized, seed={RNG_SEED})\n")
print(f"{'mode':>4} {'f_hz':>8} {'zeta0':>7} {'target_shift':>12} {'target_ratio':>12} "
      f"{'q0_ref':>10} {'k3':>14} {'c2':>10}")
for m in range(n_flex):
    wn = wn_arr[m]
    zeta0 = zeta0_arr[m]
    pd = phi_drive[:, m]
    norm_pd = np.linalg.norm(pd)
    if norm_pd < 1e-8:
        # negligible drive coupling -- q0_ref would blow up; skip calibration,
        # nonlinearity effectively never engages for this mode anyway (k3=c2=0)
        k3_arr[m] = 0.0
        c2_arr[m] = 0.0
        q0_ref_arr[m] = 1.0
        print(f"{m+1:>4} {wn/2/np.pi:>8.2f} {zeta0*100:>6.2f}% "
              f"{'--':>12} {'--':>12} {'(no drive coupling)':>10} {'--':>14} {'--':>10}")
        continue
    Q0_modal_force = norm_pd * REFERENCE_DRIVE_LEVEL_VRMS * np.sqrt(2)
    q0_ref = Q0_modal_force / (2 * zeta0 * wn ** 2)
    try:
        k3, c2 = calibrate_mode(q0_ref, wn, zeta0, target_freq_shift[m], target_zeta_ratio[m])
        k3_arr[m] = k3
        c2_arr[m] = c2
        q0_ref_arr[m] = q0_ref
        print(f"{m+1:>4} {wn/2/np.pi:>8.2f} {zeta0*100:>6.2f}% "
              f"{target_freq_shift[m]*100:>11.2f}% {target_zeta_ratio[m]:>12.3f} "
              f"{q0_ref:>10.3e} {k3:>14.3e} {c2:>10.3e}")
    except RuntimeError as e:
        # Some randomized targets don't bracket cleanly for a given mode's
        # (wn, zeta0) -- rather than hand-tune every edge case across 33
        # modes, fall back to no nonlinearity for that one mode (k3=c2=0)
        # and keep going. This is a demo/exploration build; graceful
        # per-mode degradation is the right trade-off over a brittle build
        # that requires every random target to calibrate perfectly.
        k3_arr[m] = 0.0
        c2_arr[m] = 0.0
        q0_ref_arr[m] = q0_ref
        print(f"{m+1:>4} {wn/2/np.pi:>8.2f} {zeta0*100:>6.2f}% "
              f"{target_freq_shift[m]*100:>11.2f}% {target_zeta_ratio[m]:>12.3f} "
              f"{'CALIBRATION FAILED -- falling back to k3=c2=0 (linear) for this mode':>10}")

# ---------------------------------------------------------------------
# 4. Save: baseline system + all per-mode nonlinear arrays
# ---------------------------------------------------------------------
data['nl_target_mode_shapes'] = phi_arr        # (ndof, n_flex)
data['nl_k3s'] = k3_arr
data['nl_c2s'] = c2_arr
data['nl_q0_refs'] = q0_ref_arr
data['nl_target_freqs_hz'] = wn_arr / (2 * np.pi)
data['nl_target_zetas'] = zeta0_arr
data['nl_drive_dof_indices'] = np.array(DRIVE_DOF_INDICES)
data['nl_reference_drive_level_vrms'] = np.array(REFERENCE_DRIVE_LEVEL_VRMS)

np.savez(OUTPUT_FILE, **data)
print(f"\nSaved all-modes nonlinear system to {OUTPUT_FILE}")
