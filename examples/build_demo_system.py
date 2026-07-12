# -*- coding: utf-8 -*-
"""
build_demo_system.py

Builds a synthetic 5-mass, free-free lumped-parameter chain and exports it
as continuous-time state-space matrices (A, B, C, D) in a .mat file that
Rattlesnake's "State Space" virtual hardware (hardware index 5,
components/state_space_virtual_hardware.py) can load directly.

System layout (free-free chain, no ground connection):

    F1 -> [m1]--k1/c1--[m2]--k2/c2--[m3]--k3/c3--[m4]--k4/c4--[m5] <- F2

  * 2 shaker inputs:  forces at mass 1 and mass 5 (the two ends)
  * 5 accelerometer outputs: acceleration at every mass

This mirrors a small free-free MIMO random vibration test article: one
rigid-body mode near 0 Hz plus four flexible bending/axial-type modes,
with two independent drives -- enough to make Rattlesnake's MIMO Random
control law do real work (cross-channel coupling, non-trivial drive
spectral density matrix, etc.) without being unwieldy to set up in the GUI.

Output: demo_system.mat containing A, B, C, D as double arrays.
    A : (10, 10) state matrix       (state = [q (5); qdot (5)])
    B : (10, 2)  input matrix       (forces at mass 1 and mass 5)
    C : (5, 10)  output matrix      (acceleration at each mass)
    D : (5, 2)   feedthrough matrix (acceleration output, so D != 0)

Run:
    python build_demo_system.py

Then in Rattlesnake, select "State Space" as the hardware and point the
hardware file at the resulting demo_system.mat.
"""

import numpy as np
from scipy.io import savemat
from scipy.linalg import eigh

# ---------------------------------------------------------------------
# 1. Physical parameters
# ---------------------------------------------------------------------
n = 5  # number of lumped masses

# Masses (kg) -- slightly non-uniform so modes aren't degenerate/symmetric
m = np.array([0.50, 0.55, 0.60, 0.55, 0.50])

# Spring stiffnesses between adjacent masses (N/m), n-1 = 4 springs
k = np.array([2.0e5, 1.8e5, 1.8e5, 2.0e5])

# Target modal damping ratio for the flexible modes (e.g. 1%)
zeta_target = 0.01

# ---------------------------------------------------------------------
# 2. Mass and stiffness matrices (free-free chain, no ground spring)
# ---------------------------------------------------------------------
M = np.diag(m)

K = np.zeros((n, n))
for i, ki in enumerate(k):
    K[i, i] += ki
    K[i + 1, i + 1] += ki
    K[i, i + 1] -= ki
    K[i + 1, i] -= ki

# ---------------------------------------------------------------------
# 3. Rayleigh damping: C = alpha*M + beta*K, tuned to give ~zeta_target
#    at two representative flexible modes (skip the rigid-body mode)
# ---------------------------------------------------------------------
eigvals, _ = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)          # numerical noise -> 0
omega_n = np.sqrt(eigvals)                    # rad/s, ascending order
freqs_hz = omega_n / (2 * np.pi)

# omega_n[0] ~ 0 (rigid body mode). Use modes 1 and 3 (0-indexed) to
# solve the 2x2 Rayleigh system for alpha, beta.
i1, i2 = 1, 3
w1, w2 = omega_n[i1], omega_n[i2]
A_ray = 0.5 * np.array([[1 / w1, w1],
                         [1 / w2, w2]])
alpha, beta = np.linalg.solve(A_ray, [zeta_target, zeta_target])

C_damp = alpha * M + beta * K

# ---------------------------------------------------------------------
# 4. Input (force) locations: mass 1 and mass 5 (the two ends)
# ---------------------------------------------------------------------
n_inputs = 2
Bf = np.zeros((n, n_inputs))
Bf[0, 0] = 1.0   # shaker 1 drives mass 1
Bf[-1, 1] = 1.0  # shaker 2 drives mass 5

# ---------------------------------------------------------------------
# 5. Output selection: acceleration at every mass
# ---------------------------------------------------------------------
n_outputs = n
Co = np.eye(n_outputs, n)  # every mass has an accelerometer

# ---------------------------------------------------------------------
# 6. Assemble continuous-time state-space matrices
#    state x = [q; qdot], output y = qddot (acceleration)
# ---------------------------------------------------------------------
Minv = np.linalg.inv(M)
zero_n = np.zeros((n, n))
eye_n = np.eye(n)

A = np.block([
    [zero_n,            eye_n],
    [-Minv @ K,   -Minv @ C_damp]
])

B = np.block([
    [np.zeros((n, n_inputs))],
    [Minv @ Bf]
])

C = np.block([
    -Co @ Minv @ K, -Co @ Minv @ C_damp
])

D = Co @ Minv @ Bf

# ---------------------------------------------------------------------
# 7. Save as .mat for Rattlesnake (and for direct use in MATLAB too)
# ---------------------------------------------------------------------
out_file = "demo_system.mat"
savemat(out_file, {"A": A, "B": B, "C": C, "D": D})

# ---------------------------------------------------------------------
# 8. Sanity check printout
# ---------------------------------------------------------------------
print(f"Saved {out_file}")
print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}, D: {D.shape}")
print()
print("Natural frequencies (Hz):")
for i, f in enumerate(freqs_hz):
    tag = "(rigid body)" if i == 0 else ""
    print(f"  Mode {i+1}: {f:8.2f} Hz  {tag}")
print()
print(f"Rayleigh damping: alpha={alpha:.6e}, beta={beta:.6e}")
print(f"Target zeta={zeta_target:.3f} enforced at modes {i1+1} and {i2+1}")
print(f"2 inputs: force at mass 1, force at mass 5")
print(f"5 outputs: acceleration at masses 1-5")
