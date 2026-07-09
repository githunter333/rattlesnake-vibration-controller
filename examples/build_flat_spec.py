#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:32:50 2026

@author: nhunterjr
"""

# -*- coding: utf-8 -*-
"""
build_flat_spec.py

Python equivalent of build_flat_spec.m. Builds a flat diagonal CPSD control
specification for the 5-mass SDynPy demo, saved as a .mat file in the format
Rattlesnake's MIMO Random environment expects
(components/random_vibration_sys_id_utilities.py :: load_specification).

The .mat branch of load_specification reads:
    f    : frequency vector (Nf,)
    cpsd : CPSD matrix, MATLAB shape (Nc, Nc, Nf); Rattlesnake does
           cpsd.transpose(2,0,1) on load to get (Nf, Nc, Nc)

A flat spec = constant target PSD on the CPSD diagonal (one auto-spectrum
per control channel), zeros off-diagonal (no prescribed coherence).
Nc = 5 control channels (the five accelerometers).
"""

import numpy as np
from scipy.io import savemat

# ---- Parameters: match these to the GUI settings ----
Nc          = 5        # control channels (accelerometers)
sample_rate = 2560.0   # Hz (matches build_sdynpy_demo.py)
df          = 1.0      # Hz frequency resolution (match GUI)
f_low       = 20.0     # Hz flat band lower edge
f_high      = 200.0    # Hz flat band upper edge
psd_level   = 0.001    # g^2/Hz flat level per channel

# ---- Frequency vector: 0 .. Nyquist ----
f_nyq = sample_rate / 2.0
f = np.arange(0.0, f_nyq + df/2, df)   # include Nyquist
Nf = f.size

# ---- Flat diagonal CPSD, shape (Nc, Nc, Nf) to match MATLAB convention ----
cpsd = np.zeros((Nc, Nc, Nf), dtype='complex128')
in_band = (f >= f_low) & (f <= f_high)
diag_idx = np.arange(Nc)
cpsd[diag_idx, diag_idx, :] = psd_level * in_band[np.newaxis, :]

# ---- Save (-v7 equivalent: scipy default is v5, also loadmat-compatible) ----
out_file = "flat_spec_demo.mat"
savemat(out_file, {"f": f, "cpsd": cpsd})

print(f"Wrote {out_file}")
print(f"  f    : {Nf} frequency lines, 0 to {f_nyq:.1f} Hz, df = {df:.3f} Hz")
print(f"  cpsd : {cpsd.shape[0]} x {cpsd.shape[1]} x {cpsd.shape[2]}  (Nc x Nc x Nf)")
print(f"  Flat band: {f_low:.1f} - {f_high:.1f} Hz at {psd_level:g} g^2/Hz on each of {Nc} channels")

# ---- Verify it round-trips exactly the way Rattlesnake loads it ----
from scipy.io import loadmat
d = loadmat(out_file)
f_check = d['f'].squeeze()
cpsd_check = d['cpsd'].transpose(2, 0, 1)   # exactly what Rattlesnake does
print("\nRound-trip check (as Rattlesnake loads it):")
print(f"  f shape:    {f_check.shape}")
print(f"  cpsd shape: {cpsd_check.shape}  (Nf x Nc x Nc)")
# Find a mid-band line and confirm it's a flat diagonal
k = np.argmin(np.abs(f_check - 100.0))
print(f"  CPSD at {f_check[k]:.1f} Hz (should be {psd_level:g} on diagonal, 0 off):")
print(np.real(cpsd_check[k]))