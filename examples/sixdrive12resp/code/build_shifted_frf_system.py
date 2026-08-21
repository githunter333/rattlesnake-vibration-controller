#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_shifted_frf_system.py

Builds a second copy of the six-drive/twelve-response frame system with two
targeted modal shifts, for the FRF-change-during-control study: simulates a
resonance shifting and stiffening its damping partway through a test
(nominally due to a level increase), independent of any measurement/
identification noise.

Uses an exact rank-1 modal update: for a target mode i with mass-normalized
eigenvector phi_i, adding
    dk * (M phi_i)(M phi_i)^T / (phi_i^T M phi_i)^2
to K shifts ONLY mode i's stiffness (and therefore frequency) -- every other
mode's eigenvector is M-orthogonal to phi_i, so it passes through unchanged,
not just to first order but exactly (verified below by re-solving the full
eigenproblem). The same construction is applied to C for the damping ratio.

Targets (current -> shifted):
    Mode 1: 120.00 Hz, 4.00% -> 115 Hz, 5%
    Mode 2: 159.79 Hz, 3.34% -> 155 Hz, 5%
Modes 3+ are left untouched.

Run (sdynpy env):
    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python build_shifted_frf_system.py

Output: ../results/sdynpy_frame6x12_system_shifted.npz -- same format as the
nominal system file (loadable with sdpy.System.load), only stiffness/damping
differ.
"""

import os
import numpy as np
from scipy.linalg import eigh
import sdynpy as sdpy

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
NOMINAL_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")
SHIFTED_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system_shifted.npz")

N_RIGID = 3
# (0-based flexible-mode index) -> (target frequency Hz, target damping ratio)
TARGETS = {
    0: (115.0, 0.05),   # mode 1: 120.00 Hz, 4.00% -> 115 Hz, 5%
    1: (155.0, 0.05),   # mode 2: 159.79 Hz, 3.34% -> 155 Hz, 5%
}

system = sdpy.System.load(NOMINAL_FILE)
M, K, C = system.mass, system.stiffness, system.damping

eigvals, eigvecs = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)
freq_hz = np.sqrt(eigvals) / (2 * np.pi)

print("Before shift:")
for j in range(6):
    i = N_RIGID + j
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    zeta = (phi @ C @ phi) / (2 * np.sqrt((phi @ K @ phi) * mn))
    print(f"  Mode {j + 1}: {freq_hz[i]:8.3f} Hz   zeta={zeta * 100:.2f}%")

# --- Stiffness shift (exact rank-1 modal update per target mode) ---
K_shifted = K.copy()
for mode_idx, (f_target, _) in TARGETS.items():
    i = N_RIGID + mode_idx
    phi = eigvecs[:, i]
    mn = phi @ M @ phi
    k_old = phi @ K @ phi
    k_new = (2 * np.pi * f_target) ** 2 * mn
    dk = k_new - k_old
    Mphi = M @ phi
    K_shifted += dk / mn ** 2 * np.outer(Mphi, Mphi)

# Re-solve with the shifted K to get the (slightly reordered/renormalized)
# eigenvectors the damping update needs to target the same physical modes
eigvals2, eigvecs2 = eigh(K_shifted, M)
freq_hz2 = np.sqrt(np.clip(eigvals2, 0, None)) / (2 * np.pi)

# --- Damping shift (same rank-1 construction, using the shifted-K eigenvectors) ---
C_shifted = C.copy()
for mode_idx, (_, zeta_target) in TARGETS.items():
    i = N_RIGID + mode_idx
    phi = eigvecs2[:, i]
    mn = phi @ M @ phi
    omega_i = np.sqrt(max(eigvals2[i], 0))
    c_old = phi @ C @ phi
    c_new = 2 * zeta_target * omega_i * mn
    dc = c_new - c_old
    Mphi = M @ phi
    C_shifted += dc / mn ** 2 * np.outer(Mphi, Mphi)

print("\nAfter shift:")
for j in range(6):
    i = N_RIGID + j
    phi = eigvecs2[:, i]
    mn = phi @ M @ phi
    k = phi @ K_shifted @ phi
    c = phi @ C_shifted @ phi
    omega_i = np.sqrt(k / mn)
    zeta = c / (2 * np.sqrt(k * mn))
    print(f"  Mode {j + 1}: {omega_i / 2 / np.pi:8.3f} Hz   zeta={zeta * 100:.2f}%")

system.stiffness[:] = K_shifted
system.damping[:] = C_shifted
system.save(SHIFTED_FILE)
print(f"\nSaved shifted system to {SHIFTED_FILE}")
