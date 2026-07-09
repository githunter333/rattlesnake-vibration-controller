# -*- coding: utf-8 -*-
"""
build_sdynpy_demo.py

Builds the same 5-mass, free-free lumped-parameter chain as
build_demo_system.py, but this time as a native SDynPy `System` object,
and uses SDynPy's built-in `create_synthetic_test()` helper
(sdynpy.fileio.sdynpy_rattlesnake) to generate BOTH:

  1. The system .npz file for Rattlesnake's "State Space via SDynPy
     System" virtual hardware (hardware index 6)
  2. A fully populated Rattlesnake profile spreadsheet (.xlsx) --
     channel table, hardware sheet, and MIMO Random environment --
     ready to load directly into rattlesnake.py with no manual GUI
     channel entry required.

Run this from the `sdynpy` conda environment (needs sdynpy, qtpy,
pyqtgraph, openpyxl -- NOT nidaqmx; that import is lazy and only
triggered if NI-DAQmx hardware is actually selected at runtime).

    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples
    python build_sdynpy_demo.py

Then in Rattlesnake: File -> Open Profile -> select the generated
.xlsx. The hardware type and file path are already filled in.
"""

import os
import numpy as np
from scipy.linalg import eigh
import sdynpy as sdpy
from sdynpy.fileio.sdynpy_rattlesnake import create_synthetic_test

# ---------------------------------------------------------------------
# 1. Physical parameters (identical to build_demo_system.py)
# ---------------------------------------------------------------------
n = 5  # number of lumped masses

m = np.array([0.50, 0.55, 0.60, 0.55, 0.50])          # kg
k = np.array([2.0e5, 1.8e5, 1.8e5, 2.0e5])             # N/m, 4 springs
zeta_target = 0.01                                      # target modal damping

M = np.diag(m)

K = np.zeros((n, n))
for i, ki in enumerate(k):
    K[i, i] += ki
    K[i + 1, i + 1] += ki
    K[i, i + 1] -= ki
    K[i + 1, i] -= ki

# Rayleigh damping tuned to zeta_target at modes 2 and 4 (skip rigid body)
eigvals, _ = eigh(K, M)
eigvals = np.clip(eigvals, 0, None)
omega_n = np.sqrt(eigvals)
i1, i2 = 1, 3
w1, w2 = omega_n[i1], omega_n[i2]
A_ray = 0.5 * np.array([[1 / w1, w1], [1 / w2, w2]])
alpha, beta = np.linalg.solve(A_ray, [zeta_target, zeta_target])
C = alpha * M + beta * K

# ---------------------------------------------------------------------
# 2. Coordinates: 5 physical nodes, translational X direction
# ---------------------------------------------------------------------
node_numbers = [1, 2, 3, 4, 5]
coordinate = sdpy.coordinate_array(node=node_numbers, direction='X')

# Excitation (shaker) at the two end masses; response (accel) at all 5
excitation_coordinates = sdpy.coordinate_array(node=[1, 5], direction='X')
response_coordinates = coordinate  # all 5 nodes

# ---------------------------------------------------------------------
# 3. Build the SDynPy System (transformation=None -> identity, i.e.
#    the "state" DOFs are just the physical DOFs directly)
# ---------------------------------------------------------------------
system = sdpy.System(coordinate, mass=M, stiffness=K, damping=C)

# ---------------------------------------------------------------------
# 4. Generate the system file + full Rattlesnake profile spreadsheet
# ---------------------------------------------------------------------
output_dir = os.path.dirname(os.path.abspath(__file__))
system_filename = os.path.join(output_dir, "sdynpy_demo_system.npz")
spreadsheet_file_name = os.path.join(output_dir, "sdynpy_demo_profile.xlsx")

# Path to the cloned rattlesnake-vibration-controller repo root (the
# directory that directly contains the `components` package)
rattlesnake_directory = os.path.expanduser(
    "~/Documents/Code/python/rattlesnake-vibration-controller"
)

sample_rate = 2560          # Hz -- well above the ~174 Hz top flexible mode
time_per_read = 1.0         # seconds per acquisition frame
time_per_write = 1.0        # seconds per output frame

create_synthetic_test(
    spreadsheet_file_name=spreadsheet_file_name,
    system_filename=system_filename,
    system=system,
    excitation_coordinates=excitation_coordinates,
    response_coordinates=response_coordinates,
    rattlesnake_directory=rattlesnake_directory,
    sample_rate=sample_rate,
    time_per_read=time_per_read,
    time_per_write=time_per_write,
    environments=[("random", "Demo Random Test")],
)

print(f"System file written to:      {system_filename}")
print(f"Profile spreadsheet written: {spreadsheet_file_name}")
print()
print("Natural frequencies (Hz):")
for idx, f in enumerate(omega_n / (2 * np.pi)):
    tag = "(rigid body)" if idx == 0 else ""
    print(f"  Mode {idx + 1}: {f:8.2f} Hz  {tag}")
print()
print("Open rattlesnake.py, then File -> Open Profile and select:")
print(f"  {spreadsheet_file_name}")
print("Hardware type and file path are already configured (hardware index 6).")
