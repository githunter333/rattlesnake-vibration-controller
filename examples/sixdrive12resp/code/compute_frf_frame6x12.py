#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_frf_frame6x12.py

Computes the FRF H(f) for the 6-drive/12-response frame system built by
build_sdynpy_demo_frame6x12.py, over a chosen response-node subset (the
control locations) and the full drive set, and saves it as a .npz for
compare_buzz_vs_optimal_diagonal.py to consume.

This needs `sdynpy`, which typically lives in a separate conda env from
`rattlesnake`/`cvxpy` (see build_sdynpy_demo_frame6x12.py's docstring) --
that's why this is a standalone script rather than being folded into
compare_buzz_vs_optimal_diagonal.py: run this one in the `sdynpy` env,
then run the comparison script in the `rattlesnake` env.

Run (from the `sdynpy` conda environment):

    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python compute_frf_frame6x12.py

Output: ../results/frf_frame6x12_H.npz, with f, H (F,M,N complex),
drive_nodes, resp_nodes.
"""

import os
import numpy as np
import sdynpy as sdpy

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SYSTEM_FILE = os.path.join(RESULTS_DIR, "sdynpy_frame6x12_system.npz")

# ---- Control locations: match the "Control Channels" row in the profile ----
# All 6 shaker drives, and the first 8 of the 12 response channels
# (all 6 of Row B [nodes 7-12] + first 2 of Row C [nodes 13-14]) --
# change resp_nodes to try a different (or square, N=6) control set.
drive_nodes = [1, 2, 3, 4, 5, 6]
resp_nodes = [7, 8, 9, 10, 11, 12, 13, 14]

f_low, f_high, df = 100.0, 1000.0, 1.0  # Hz, matches build_flat_spec_large.py

system = sdpy.System.load(SYSTEM_FILE)
exc = sdpy.coordinate_array(node=drive_nodes, direction='X')
resp = sdpy.coordinate_array(node=resp_nodes, direction='X')

f = np.arange(f_low, f_high + df / 2, df)
frf = system.frequency_response(f, responses=resp, references=exc, displacement_derivative=2)
H = np.moveaxis(frf.ordinate, -1, 0)  # (F, M, N)

out_file = os.path.join(RESULTS_DIR, "frf_frame6x12_H.npz")
np.savez(out_file, f=f, H=H, drive_nodes=np.array(drive_nodes), resp_nodes=np.array(resp_nodes))

print(f"H shape (F,M,N): {H.shape}  (M={len(resp_nodes)} responses, N={len(drive_nodes)} drives)")
print(f"Drive nodes:    {drive_nodes}")
print(f"Response nodes: {resp_nodes}")
print(f"Wrote {out_file}")
