#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nonlinear_frf_system.py

Builds a nonlinear variant of the 6-drive/12-response frame system: same
mass/damping/stiffness (and therefore the same FRF) as sdynpy_frame6x12_
system.npz at low amplitude, but with an amplitude-dependent softening
stiffness (cubic) plus amplitude-dependent damping (quadratic) attached to
ONE target mode in modal coordinates -- so the FRF genuinely changes as the
test level ramps up, rather than being switched between two fixed states
(see build_shifted_frf_system.py for that earlier, discrete approach).

Modal form, mass-normalized modal coordinate q for the target mode (mn=1):
    q'' + 2*zeta0*wn*q' + wn^2*q + k3*q^3 + c2*q'*|q'| = Qn(t)
  k3 < 0 (softening): frequency drops with amplitude, scaling ~ A^2
  c2 > 0: damping ratio grows with amplitude, scaling ~ A (describing-
          function equivalent of quadratic/aerodynamic damping)

This projects onto physical coordinates via the mode's OWN mass-normalized
eigenvector phi (same rank-1 M-orthogonal projection idea used in
build_shifted_frf_system.py, just nonlinear/state-dependent instead of a
constant shift):
    M x'' + C x' + K x + M*phi*(k3*q^3 + c2*q'*|q'|) = F_ext(t)
    q = phi^T M x_disp,  q' = phi^T M x_vel

Calibration: solved numerically (free-decay simulation of the isolated
target-mode SDOF, not just the first-order Duffing/describing-function
formulas) so that going from a reference modal amplitude q0 to 2*q0 lands
close to a user-specified (frequency shift, damping ratio change) target --
NOT hit exactly, since frequency shift scales ~A^2 and damping shift scales
~A, so they don't move together proportionally with level the way a linear
effect would. See the printed calibration summary for what was actually
achieved.

nonlinearity_strength (read live by the hardware class, not baked in here):
multiplies (k3, c2) together, so strength=0 recovers the pure linear
baseline exactly and strength=1 reproduces the calibration below.

Run (sdynpy env):
    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python build_nonlinear_frf_system.py

Output: ../results/sdynpy_frame6x12_system_nonlinear.npz
"""

import os
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
BASELINE_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system_nonlinear.npz")

N_RIGID = 3
TARGET_MODE = 1  # 1-based, among flexible modes -- mode 1 is ~120 Hz
TARGET_FREQ_SHIFT_AT_2X = -0.04   # frequency shift at amplitude 2*q0, relative to the LINEAR frequency
TARGET_ZETA_RATIO_Q0_TO_2Q0 = 1.20  # damping ratio at 2*q0 relative to damping ratio at q0

# Drive nodes 1-6, X-direction -- physical DOF indices per build_sdynpy_demo_frame6x12.py's
# dofs(row,col) convention (row 0 = drive row, 2 DOF/node, X first): 2*col for col=0..5.
DRIVE_DOF_INDICES = [0, 2, 4, 6, 8, 10]
REFERENCE_DRIVE_LEVEL_VRMS = 20.0  # Q0_REF is calibrated so the full ~4%/20% effect lands near
                                    # this drive level (mode-1-shaped excitation across the 6
                                    # shakers, not a uniform in-phase drive -- see the build's
                                    # printed voltage-sweep guidance below for why that matters)

# ---------------------------------------------------------------------
# 1. Load the baseline (linear) system, get the target mode's own
#    mass-normalized eigenvector + natural frequency/damping
# ---------------------------------------------------------------------
data = {key: val for key, val in np.load(BASELINE_FILE).items()}
M, C, K = data['mass'], data['damping'], data['stiffness']

eigvals, eigvecs = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)
freq_hz_all = np.sqrt(eigvals) / (2 * np.pi)

mode_idx = N_RIGID + (TARGET_MODE - 1)
phi = eigvecs[:, mode_idx]
mn = phi @ M @ phi
wn = np.sqrt(eigvals[mode_idx])
f_target = freq_hz_all[mode_idx]
zeta0 = (phi @ C @ phi) / (2 * np.sqrt((phi @ K @ phi) * mn))

print(f"Target mode {TARGET_MODE} (flexible-mode index): {f_target:.3f} Hz, "
      f"zeta0={zeta0*100:.3f}%, mn={mn:.6f} (should be ~1.0, mass-normalized)")

# ---------------------------------------------------------------------
# 1b. Derive a REALISTIC reference amplitude Q0_REF, tied to what this
#     system can actually reach at a plausible drive level -- an earlier
#     version of this script picked Q0_REF=1.0 arbitrarily (abstract
#     mass-normalized modal units) without checking it against the
#     system's real dynamic range, and it turned out to need ~81,000 V RMS
#     to reach even with a mode-1-shaped drive (a uniform in-phase drive
#     across all 6 shakers barely couples into mode 1 at all -- its modal
#     participation factor is tiny; shaping the drive proportional to the
#     mode's own shape at the drive points maximizes coupling, standard
#     single-mode-excitation practice). Q0_REF is instead solved from the
#     resonant steady-state amplitude formula for a modal SDOF driven at
#     its own natural frequency: q_ss = Q0 / (2*zeta0*wn^2), where Q0 is
#     the modal force amplitude from a REFERENCE_DRIVE_LEVEL_VRMS sine
#     drive shaped across the 6 shakers proportional to the mode's own
#     shape at those points.
phi_drive = phi[DRIVE_DOF_INDICES]
phi_drive_unit = phi_drive / np.linalg.norm(phi_drive)
Q0_modal_force = np.linalg.norm(phi_drive) * REFERENCE_DRIVE_LEVEL_VRMS * np.sqrt(2)
Q0_REF = Q0_modal_force / (2 * zeta0 * wn ** 2)
print(f"mode-1-shaped drive weights (unit-normalized, per shaker): {np.round(phi_drive_unit, 3)}")
print(f"Q0_REF derived from a {REFERENCE_DRIVE_LEVEL_VRMS:.0f} V RMS mode-shaped drive: {Q0_REF:.6e} "
      f"(modal-coordinate units)")

# ---------------------------------------------------------------------
# 2. Calibrate k3, c2 on the isolated SDOF (numeric, not first-order
#    formulas -- see nl_sdof_calibration.py prototype this was ported from)
# ---------------------------------------------------------------------

def simulate_free_decay(q0, k3, c2, n_cycles=6):
    T0 = 2 * np.pi / wn
    t_span = (0, n_cycles * T0)
    t_eval = np.linspace(*t_span, 4000)

    def rhs(t, s):
        q, qd = s
        qdd = -2 * zeta0 * wn * qd - wn ** 2 * q - k3 * q ** 3 - c2 * qd * np.abs(qd)
        return [qd, qdd]

    sol = solve_ivp(rhs, t_span, [q0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]


def measure_freq_and_damping(q0, k3, c2):
    t, q = simulate_free_decay(q0, k3, c2)
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


def calibrate(q0):
    f_lin = wn / (2 * np.pi)

    def freq_shift_error(k3):
        f_hi, _ = measure_freq_and_damping(2 * q0, k3, 0.0)
        return (f_hi - f_lin) / f_lin - TARGET_FREQ_SHIFT_AT_2X

    k3_scale = abs(TARGET_FREQ_SHIFT_AT_2X) * 8 * wn ** 2 / (3 * (2 * q0) ** 2)
    k3_stability_limit = wn ** 2 / (4 * q0 ** 2)
    k3_lo, k3_hi = -min(3 * k3_scale, 0.5 * k3_stability_limit), 0.0
    k3 = brentq(freq_shift_error, k3_lo, k3_hi, xtol=1e-6 * k3_scale)

    def damping_ratio_error(c2):
        _, z_lo = measure_freq_and_damping(q0, k3, c2)
        _, z_hi = measure_freq_and_damping(2 * q0, k3, c2)
        return z_hi / z_lo - TARGET_ZETA_RATIO_Q0_TO_2Q0

    x_target = 0.25 * zeta0  # matches TARGET_ZETA_RATIO_Q0_TO_2Q0=1.20; see nl_sdof_calibration.py derivation
    c2_scale = x_target * 3 * np.pi / (4 * q0)
    mults = np.geomspace(0.1, 20, 40)
    errs = [damping_ratio_error(m * c2_scale) for m in mults]
    c2_lo = c2_hi = None
    for i in range(len(mults) - 1):
        if errs[i] * errs[i + 1] < 0:
            c2_lo, c2_hi = mults[i] * c2_scale, mults[i + 1] * c2_scale
            break
    if c2_lo is None:
        raise RuntimeError(f"no sign change found scanning c2; errs={errs}")
    c2 = brentq(damping_ratio_error, c2_lo, c2_hi, xtol=1e-9 * max(c2_scale, 1e-12))

    return k3, c2


k3, c2 = calibrate(Q0_REF)

f_lo, z_lo = measure_freq_and_damping(Q0_REF, k3, c2)
f_hi, z_hi = measure_freq_and_damping(2 * Q0_REF, k3, c2)
f_lin = wn / (2 * np.pi)
print(f"\ncalibrated (strength=1.0): k3={k3:.3f}  c2={c2:.6f}  (reference amplitude q0={Q0_REF})")
print(f"linear:  f={f_lin:.3f} Hz  zeta={zeta0*100:.3f}%")
print(f"at q0:   f={f_lo:.3f} Hz ({(f_lo/f_lin-1)*100:+.2f}%)   zeta={z_lo*100:.3f}% ({(z_lo/zeta0-1)*100:+.1f}%)")
print(f"at 2*q0: f={f_hi:.3f} Hz ({(f_hi/f_lin-1)*100:+.2f}%)   zeta={z_hi*100:.3f}% ({(z_hi/zeta0-1)*100:+.1f}%)")
print(f"change q0->2*q0: freq {(f_hi/f_lo-1)*100:+.2f}%   zeta ratio {z_hi/z_lo:.3f}x  "
      f"(target was freq {TARGET_FREQ_SHIFT_AT_2X*100:+.0f}% at 2*q0 vs linear, "
      f"zeta ratio {TARGET_ZETA_RATIO_Q0_TO_2Q0:.2f}x q0->2*q0 -- approximate by design, not exact)")

# ---------------------------------------------------------------------
# 3. Save: everything from the baseline file, plus the nonlinear terms
# ---------------------------------------------------------------------
data['nl_target_mode_shape'] = phi          # (ndof,) mass-normalized eigenvector, physical DOF order
data['nl_k3'] = np.array(k3)
data['nl_c2'] = np.array(c2)
data['nl_q0_ref'] = np.array(Q0_REF)
data['nl_target_freq_hz'] = np.array(f_target)
data['nl_target_zeta'] = np.array(zeta0)
data['nl_drive_dof_indices'] = np.array(DRIVE_DOF_INDICES)
data['nl_drive_shape_unit'] = phi_drive_unit  # (6,) per-shaker weights maximizing coupling into this mode
data['nl_reference_drive_level_vrms'] = np.array(REFERENCE_DRIVE_LEVEL_VRMS)

np.savez(OUTPUT_FILE, **data)
print(f"\nSaved nonlinear system to {OUTPUT_FILE}")
print("Load this with components/sdynpy_nonlinear_system_virtual_hardware.py's "
      "SDynPyNonlinearSystemAcquisition as the Hardware File -- the baseline linear "
      "hardware/file are completely untouched.")
