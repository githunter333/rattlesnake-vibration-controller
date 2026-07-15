# -*- coding: utf-8 -*-
"""
build_octave_demo.py

Demonstrates spectral_analysis/fractional_octave.py (narrowband vs 1/6
octave PSD and FRF estimation) using the six-drive/twelve-response frame
system built by examples/sixdrive12resp/code/build_sdynpy_demo_frame6x12.py.

Only ONE drive (node 1, the first shaker) is excited with broadband
random force; only ONE response (node 18, the far corner of the frame,
farthest from node 1) is observed. This reduces the 6-input/12-output
MIMO system to a single input/single output pair with a genuinely
resonant, multi-mode transfer function (node 1 -> node 18 crosses the
whole frame), which is exactly the kind of signal 1/6-octave analysis
is meant to summarize.

Both the input (drive force) and response (acceleration) time series are
analyzed with:
  * a narrowband Welch/CSD estimate (~3 Hz resolution -- "regular
    frequency increment" spectral density and FRF, the kind of thing
    Rattlesnake's MIMO Random environment works with internally), and
  * a 1/6-octave estimate derived from that same narrowband estimate by
    energy-preserving band averaging, from fmin=10 Hz to fmax=2000 Hz.

Run (from the `sdynpy` conda environment -- needs sdynpy, scipy,
matplotlib):

    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/octave_vs_narrowband/code
    python build_octave_demo.py

Outputs (time series + comparison plot) are written to ../results/.
"""

import os
import sys
import numpy as np
from scipy.signal import lsim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sdynpy as sdpy

REPO_ROOT = os.path.expanduser("~/Documents/Code/python/rattlesnake-vibration-controller")
sys.path.insert(0, REPO_ROOT)
from spectral_analysis.fractional_octave import (
    time_series_to_octave_psd, time_series_to_octave_frf, time_series_to_octave_coherence,
)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SIXDRIVE_RESULTS = os.path.join(REPO_ROOT, "examples", "sixdrive12resp", "results")

# ---------------------------------------------------------------------
# 1. Load the six-drive/twelve-response system and reduce to one
#    input (node 1, force X) -> one output (node 18, acceleration X)
# ---------------------------------------------------------------------
system_filename = os.path.join(SIXDRIVE_RESULTS, "sdynpy_frame6x12_system.npz")
system = sdpy.System.load(system_filename)

excitation_node = 1
response_node = 18

excitation_coordinates = sdpy.coordinate_array(node=[excitation_node], direction='X')
response_coordinates = sdpy.coordinate_array(node=[response_node], direction='X')

A, B, C, D = system.to_state_space(
    response_coordinates={2: response_coordinates},  # 2 = acceleration
    excitation_coordinates=excitation_coordinates,
    output_excitation_signals=False,
)

# ---------------------------------------------------------------------
# 2. Simulate a broadband random force at node 1, observe acceleration
#    at node 18
# ---------------------------------------------------------------------
fs = 4096.0          # Hz -- Nyquist 2048 Hz, comfortably above fmax=2000
duration = 90.0       # s -- long record for smooth narrowband + octave estimates
n_samples = int(fs * duration)
t = np.arange(n_samples) / fs

rng = np.random.default_rng(0)
force = rng.standard_normal(n_samples)  # unit-variance broadband white noise, N

_, accel, _ = lsim((A, B, C, D), U=force, T=t)
accel = np.asarray(accel).squeeze()

np.savez(
    os.path.join(RESULTS_DIR, "node1_to_node18_time_series.npz"),
    t=t, force=force, accel=accel, fs=fs,
    excitation_node=excitation_node, response_node=response_node,
)

# ---------------------------------------------------------------------
# 3. Narrowband + 1/6-octave PSD (input and response), FRF, and coherence
# ---------------------------------------------------------------------
fmin, fmax, fraction, df = 10.0, 2000.0, 6, 3.0

psd_force = time_series_to_octave_psd(force, fs, fmin=fmin, fmax=fmax, fraction=fraction, df=df)
psd_accel = time_series_to_octave_psd(accel, fs, fmin=fmin, fmax=fmax, fraction=fraction, df=df)
frf = time_series_to_octave_frf(force, accel, fs, fmin=fmin, fmax=fmax, fraction=fraction, df=df)
coh = time_series_to_octave_coherence(force, accel, fs, fmin=fmin, fmax=fmax, fraction=fraction, df=df)

n_oct_nan = np.isnan(frf['frf_octave']).sum()
print(f"1/6-octave bands: {len(frf['f_octave'])} total, {n_oct_nan} outside the "
      f"narrowband frequency coverage (should be 0 since fmax={fmax:.0f} Hz < Nyquist)")

# ---------------------------------------------------------------------
# 4. Plot: input PSD, response PSD, FRF magnitude, FRF phase, coherence --
#    narrowband vs 1/6-octave, all log-frequency
# ---------------------------------------------------------------------
NARROW_STYLE = dict(color='0.35', lw=1.1, alpha=0.9)

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 2)
axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])
ax_coh = fig.add_subplot(gs[2, :])

ax = axes[0, 0]
ax.loglog(psd_force['f_narrow'], psd_force['psd_narrow'], **NARROW_STYLE, label=f'narrowband (df={df:.0f} Hz)')
ax.loglog(psd_force['f_octave'], psd_force['psd_octave'], 'o-', color='C0', label='1/6 octave')
ax.set_title(f"Input force PSD -- node {excitation_node} X")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (N^2/Hz)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[0, 1]
ax.loglog(psd_accel['f_narrow'], psd_accel['psd_narrow'], **NARROW_STYLE, label=f'narrowband (df={df:.0f} Hz)')
ax.loglog(psd_accel['f_octave'], psd_accel['psd_octave'], 'o-', color='C1', label='1/6 octave')
ax.set_title(f"Response accel PSD -- node {response_node} X")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (g^2/Hz)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[1, 0]
ax.loglog(frf['f_narrow'], np.abs(frf['frf_narrow']), **NARROW_STYLE, label=f'narrowband (df={df:.0f} Hz)')
ax.loglog(frf['f_octave'], np.abs(frf['frf_octave']), 'o-', color='C2', label='1/6 octave')
ax.set_title(f"FRF magnitude: node {response_node} / node {excitation_node}")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|H| (g/N)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[1, 1]
ax.semilogx(frf['f_narrow'], np.angle(frf['frf_narrow'], deg=True), **NARROW_STYLE, label=f'narrowband (df={df:.0f} Hz)')
ax.semilogx(frf['f_octave'], np.angle(frf['frf_octave'], deg=True), 'o-', color='C3', label='1/6 octave')
ax.set_title(f"FRF phase: node {response_node} / node {excitation_node}")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Phase (deg)")
ax.set_xlim(fmin, fmax); ax.set_ylim(-180, 180); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax_coh.semilogx(coh['f_narrow'], coh['coherence_narrow'], **NARROW_STYLE, label=f'narrowband (df={df:.0f} Hz)')
ax_coh.semilogx(coh['f_octave'], coh['coherence_octave'], 'o-', color='C4', label='1/6 octave')
ax_coh.set_title(f"Coherence: node {response_node} / node {excitation_node}")
ax_coh.set_xlabel("Frequency (Hz)"); ax_coh.set_ylabel("gamma^2")
ax_coh.set_xlim(fmin, fmax); ax_coh.set_ylim(0, 1.05); ax_coh.legend(fontsize=8); ax_coh.grid(True, which='both', alpha=0.3)

fig.suptitle(f"Narrowband vs 1/6-octave: single drive (node {excitation_node}) -> "
             f"single response (node {response_node}), six-drive/twelve-response frame")
fig.tight_layout()
plot_filename = os.path.join(RESULTS_DIR, "narrowband_vs_octave_node1_to_node18.png")
fig.savefig(plot_filename, dpi=150)

print(f"\nTime series written to: {os.path.join(RESULTS_DIR, 'node1_to_node18_time_series.npz')}")
print(f"Plot written to:        {plot_filename}")
print(f"\n{duration:.0f} s at {fs:.0f} Hz, narrowband df={df:.1f} Hz, "
      f"1/6-octave {fmin:.0f}-{fmax:.0f} Hz")
