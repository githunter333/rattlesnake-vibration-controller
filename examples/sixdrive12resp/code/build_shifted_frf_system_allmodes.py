#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_shifted_frf_system_allmodes.py

Second FRF-change scenario for the FRF-update-during-control study: instead
of shifting just two modes (build_shifted_frf_system.py), this shifts ALL 33
flexible modes:

  - First 4 modes: deterministic shift -- frequency UP 4%, damping DOWN 20%
    of baseline (a stiffening/de-damping trend, e.g. joint seating in).
  - Remaining 29 modes: random shift -- frequency perturbed by
    N(0, 20 Hz) (additive), damping perturbed by N(0, 25%) (multiplicative,
    relative to each mode's own baseline). Fixed RNG seed for reproducibility.

Uses the same exact rank-1 modal update as build_shifted_frf_system.py: for
mode i's mass-normalized eigenvector phi_i, adding
    dk * (M phi_i)(M phi_i)^T / (phi_i^T M phi_i)^2
to K shifts ONLY mode i's frequency -- every other mode's eigenvector is
M-orthogonal to phi_i, so simultaneous per-mode updates don't interact, even
across all 33 modes at once, and phi_i remains an EXACT eigenvector of the
shifted K regardless of how other modes move (so it stays valid even when
random shifts cause two closely-spaced modes to cross/veer in frequency
order). Same construction applied to C for damping ratio, using each mode's
own original eigenvector and exact target frequency (not the re-solved and
re-sorted eigendecomposition, which would misattribute a crossed mode's
target to the wrong eigenvector). Verified below against each mode's own
eigenvector directly.

Run (sdynpy env):
    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python build_shifted_frf_system_allmodes.py

Output: ../results/sdynpy_frame6x12_system_shifted_allmodes.npz
"""

import os
import numpy as np
from scipy.linalg import eigh
import sdynpy as sdpy

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
NOMINAL_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")
SHIFTED_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system_shifted_allmodes.npz")

N_RIGID = 3
N_DETERMINISTIC = 4          # first 4 flexible modes: deterministic shift
FREQ_UP_FRAC = 0.04          # +4% frequency
DAMPING_DOWN_FRAC = 0.20     # -20% damping (relative to baseline)
RANDOM_FREQ_STD_HZ = 20.0    # remaining modes: additive N(0, 20 Hz)
RANDOM_DAMPING_STD_FRAC = 0.25  # remaining modes: multiplicative N(0, 25%)
DAMPING_FLOOR = 0.0005       # 0.05% -- safety floor, avoids non-physical/negative damping
FREQ_FLOOR_HZ = 10.0         # safety floor, avoids a mode landing near/at zero
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)

system = sdpy.System.load(NOMINAL_FILE)
M, K, C = system.mass, system.stiffness, system.damping

eigvals, eigvecs = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)
freq_hz = np.sqrt(eigvals) / (2 * np.pi)
n_flex = len(eigvals) - N_RIGID

# --- Build per-mode targets ---
f_targets = np.zeros(n_flex)
zeta_targets = np.zeros(n_flex)
zeta_baseline = np.zeros(n_flex)
for m in range(n_flex):
    i = N_RIGID + m
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    zeta_baseline[m] = (phi @ C @ phi) / (2 * np.sqrt((phi @ K @ phi) * mn))

for m in range(n_flex):
    if m < N_DETERMINISTIC:
        f_targets[m] = freq_hz[N_RIGID + m] * (1 + FREQ_UP_FRAC)
        zeta_targets[m] = zeta_baseline[m] * (1 - DAMPING_DOWN_FRAC)
    else:
        f_targets[m] = freq_hz[N_RIGID + m] + rng.normal(0, RANDOM_FREQ_STD_HZ)
        zeta_targets[m] = zeta_baseline[m] * (1 + rng.normal(0, RANDOM_DAMPING_STD_FRAC))
    f_targets[m] = max(f_targets[m], FREQ_FLOOR_HZ)
    zeta_targets[m] = max(zeta_targets[m], DAMPING_FLOOR)

# --- Stiffness shift: simultaneous exact rank-1 update per mode ---
K_shifted = K.copy()
for m in range(n_flex):
    i = N_RIGID + m
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    k_old = phi @ K @ phi
    k_new = (2 * np.pi * f_targets[m]) ** 2 * mn
    dk = k_new - k_old
    Mphi = M @ phi
    K_shifted += dk / mn ** 2 * np.outer(Mphi, Mphi)

# --- Damping shift: same construction, using each mode's own ORIGINAL
# eigenvector (still an exact eigenvector of K_shifted regardless of mode
# crossing/relabeling -- see the verification note below) and its exact
# target frequency, rather than re-solving K_shifted and risking picking up
# a different (crossed) mode's eigenvector at the same sorted position.
C_shifted = C.copy()
for m in range(n_flex):
    i = N_RIGID + m
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    omega_target = 2 * np.pi * f_targets[m]
    c_old = phi @ C @ phi
    c_new = 2 * zeta_targets[m] * omega_target * mn
    dc = c_new - c_old
    Mphi = M @ phi
    C_shifted += dc / mn ** 2 * np.outer(Mphi, Mphi)

# --- Verify: check each mode's OWN original eigenvector directly against
# K_shifted/C_shifted, rather than re-solving + re-sorting the eigenproblem.
# Several of these 33 modes are only a few Hz apart, and the random +/-20 Hz
# shifts can swap two adjacent modes' sorted order ("mode veering") -- that's
# a real, harmless physical outcome, but it would corrupt an index-by-sorted-
# index comparison. Using each mode's own (M-orthogonal, and therefore still
# exact) eigenvector sidesteps that entirely.
print(f"{n_flex} flexible modes shifted (first {N_DETERMINISTIC} deterministic, "
      f"remaining {n_flex - N_DETERMINISTIC} random, seed={RNG_SEED})\n")
print(f"{'mode':>4} {'f_before':>9} {'f_after':>9} {'f_target':>9}  "
      f"{'zeta_before':>11} {'zeta_after':>10} {'zeta_target':>11}")
max_f_err = max_z_err = 0.0
n_crossed = 0
sorted_freq_after = np.sqrt(np.clip(eigh(K_shifted, M, eigvals_only=True), 0, None)) / (2 * np.pi)
for m in range(n_flex):
    i = N_RIGID + m
    phi = eigvecs[:, i]  # mode m's OWN original eigenvector -- still exact in K_shifted/C_shifted
    mn = phi @ M @ phi
    k = phi @ K_shifted @ phi
    c = phi @ C_shifted @ phi
    omega_i = np.sqrt(k / mn)
    f_after = omega_i / 2 / np.pi
    zeta_after = c / (2 * np.sqrt(k * mn))
    max_f_err = max(max_f_err, abs(f_after - f_targets[m]))
    max_z_err = max(max_z_err, abs(zeta_after - zeta_targets[m]))
    if abs(sorted_freq_after[i] - f_after) > 1e-6:
        n_crossed += 1
    tag = "det" if m < N_DETERMINISTIC else "rand"
    print(f"{m + 1:>4} {freq_hz[i]:>9.2f} {f_after:>9.2f} {f_targets[m]:>9.2f}  "
          f"{zeta_baseline[m]*100:>10.2f}% {zeta_after*100:>9.2f}% {zeta_targets[m]*100:>10.2f}%  [{tag}]")

print(f"\nMax |frequency error| vs target (own-eigenvector check): {max_f_err:.2e} Hz")
print(f"Max |damping ratio error| vs target (own-eigenvector check): {max_z_err:.2e}")
print(f"Modes whose sorted-order position swapped with a neighbor (mode veering, "
      f"harmless): {n_crossed}/{n_flex}")

system.stiffness[:] = K_shifted
system.damping[:] = C_shifted
system.save(SHIFTED_FILE)
print(f"\nSaved shifted system to {SHIFTED_FILE}")
