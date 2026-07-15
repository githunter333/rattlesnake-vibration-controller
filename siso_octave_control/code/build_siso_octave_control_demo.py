# -*- coding: utf-8 -*-
"""
build_siso_octave_control_demo.py

Closed-loop SISO random vibration control, leveled on a 1/6-octave basis,
using the same single-input/single-output pair (node 1 drive -> node 18
response) from the six-drive/twelve-response frame system as
spectral_analysis/code/build_octave_demo.py.

Drive SYNTHESIS and response MEASUREMENT both stay on the full narrowband
(Welch) frequency grid throughout -- a real time-domain drive signal is
synthesized every iteration, and the actual response is measured with
ordinary narrowband Welch/CSD analysis. The question is how much of the
CONTROL DECISION is allowed to use narrowband-resolution information.

Two scenarios are run back to back from the same system-ID measurement,
differing only in how the very first drive estimate is built:

  * 'narrowband H1 init' -- the initial drive PSD is target / |H(f)|^2
    using the FULL narrowband H1 estimate (683 individual gain values).
    This does fine-grained, line-by-line equalization from the start.

  * 'octave-only H init' -- the initial drive PSD is target / |H_oct|^2
    using only the 1/6-OCTAVE-AVERAGED H1 estimate (46 gain values, one
    per band, applied flat across every narrowband line in that band).
    No step anywhere in this scenario ever sees narrowband-resolution
    gain information.

Both scenarios then run the same iterative loop, which was already
octave-only: measure the achieved narrowband response, band-average it,
compare to the target per band, and multiply the drive PSD in that band
by the scalar ratio target/achieved. That part alone can drive the
1/6-octave average to the target in either scenario -- what it CANNOT do
is cancel narrowband structure (individual resonances/antiresonances)
inside a band, since it only ever applies one number per band. The
'octave-only H init' scenario is expected to converge to the same
octave-band levels as 'narrowband H1 init', but with visible leftover
narrowband ripple within each band, because it never had access to the
within-band shape of H(f) to cancel in the first place.

Run (from the `sdynpy` conda environment):

    conda activate sdynpy
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/siso_octave_control/code
    python build_siso_octave_control_demo.py

Outputs are written to ../results/.
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
    octave_band_frequencies, narrowband_cross_spectra, octave_band_psd,
    octave_band_frf, narrowband_coherence,
)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SIXDRIVE_RESULTS = os.path.join(REPO_ROOT, "examples", "sixdrive12resp", "results")

# ---------------------------------------------------------------------
# 1. Load the SISO pair: node 1 (drive, force X) -> node 18 (response, accel X)
# ---------------------------------------------------------------------
system = sdpy.System.load(os.path.join(SIXDRIVE_RESULTS, "sdynpy_frame6x12_system.npz"))
excitation_node, response_node = 1, 18
A, B, C, D = system.to_state_space(
    response_coordinates={2: sdpy.coordinate_array(node=[response_node], direction='X')},
    excitation_coordinates=sdpy.coordinate_array(node=[excitation_node], direction='X'),
    output_excitation_signals=False,
)

fs = 4096.0
duration = 60.0
n_samples = int(fs * duration)
t = np.arange(n_samples) / fs
rng = np.random.default_rng(42)

fmin, fmax, fraction, df = 100.0, 1000.0, 6, 3.0
n_iterations = 5

centers, lower, upper = octave_band_frequencies(fmin, fmax, fraction)
target_level = 1e-3  # g^2/Hz, flat
target_octave = np.full(centers.shape, target_level)


def run_system(drive_time):
    _, resp_time, _ = lsim((A, B, C, D), U=drive_time, T=t)
    return np.asarray(resp_time).squeeze()


def expand_octave_to_narrowband(f_nb, lower, upper, values):
    """Broadcast one value per 1/6-octave band across every narrowband
    line that falls in that band; 0 outside [lower[0], upper[-1])."""
    out = np.zeros_like(f_nb)
    for lo, hi, v in zip(lower, upper, values):
        out[(f_nb >= lo) & (f_nb < hi)] = v
    return out


def synthesize_time_series(freqs_shape, psd_shape, fs, n_samples, rng):
    """Random-phase IFFT synthesis of a real time series matching a target
    one-sided narrowband PSD (interpolated onto the exact rfft grid for
    n_samples)."""
    freqs_full = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    psd_full = np.interp(freqs_full, freqs_shape, psd_shape, left=0.0, right=0.0)
    c = np.full(freqs_full.size, 2.0)
    c[0] = 1.0
    if n_samples % 2 == 0:
        c[-1] = 1.0
    mag = np.sqrt(np.clip(psd_full, 0, None) * fs * n_samples / c)
    phase = rng.uniform(0, 2 * np.pi, size=freqs_full.size)
    Xk = mag * np.exp(1j * phase)
    Xk[0] = 0.0
    if n_samples % 2 == 0:
        Xk[-1] = mag[-1]
    return np.fft.irfft(Xk, n=n_samples)


# ---------------------------------------------------------------------
# 2. System ID: one flat broadband drive, shared by both scenarios
# ---------------------------------------------------------------------
sysid_drive = rng.standard_normal(n_samples)
sysid_resp = run_system(sysid_drive)
f_nb, Sxx0, Syy0, Sxy0 = narrowband_cross_spectra(sysid_drive, sysid_resp, fs, df=df)

target_narrow = expand_octave_to_narrowband(f_nb, lower, upper, target_octave)
in_band = target_narrow > 0.0

# -- initial drive PSD, narrowband H1 (fine-grained, 683 gain values) --
with np.errstate(invalid='ignore', divide='ignore'):
    H_mag2_narrow = np.abs(Sxy0 / Sxx0) ** 2
H_mag2_narrow_reg = np.maximum(H_mag2_narrow, np.max(H_mag2_narrow) * 1e-6)
drive_psd_narrowband_init = np.zeros_like(f_nb)
drive_psd_narrowband_init[in_band] = target_narrow[in_band] / H_mag2_narrow_reg[in_band]

# -- initial drive PSD, 1/6-octave-averaged H1 (coarse, 46 gain values) --
H_octave = octave_band_frf(f_nb, Sxx0, Sxy0, lower, upper)
H_mag2_octave = np.abs(H_octave) ** 2
H_mag2_octave_reg = np.maximum(H_mag2_octave, np.nanmax(H_mag2_octave) * 1e-6)
H_mag2_octave_narrow = expand_octave_to_narrowband(f_nb, lower, upper, H_mag2_octave_reg)
drive_psd_octave_init = np.zeros_like(f_nb)
drive_psd_octave_init[in_band] = target_narrow[in_band] / H_mag2_octave_narrow[in_band]


# ---------------------------------------------------------------------
# 3. Closed-loop iteration, shared by both scenarios
# ---------------------------------------------------------------------
def run_control_loop(drive_psd_narrow, label):
    drive_psd_narrow = drive_psd_narrow.copy()
    history_achieved_octave, history_rms_db_error, history_coherence_narrow = [], [], []
    final_drive_time = final_resp_time = final_f_nb = final_Sxx = final_Syy = None

    for it in range(n_iterations):
        drive_time = synthesize_time_series(f_nb, drive_psd_narrow, fs, n_samples, rng)
        resp_time = run_system(drive_time)

        f_nb_it, Sxx_it, Syy_it, Sxy_it = narrowband_cross_spectra(drive_time, resp_time, fs, df=df)
        achieved_octave = octave_band_psd(f_nb_it, Syy_it, lower, upper)
        db_error = 10 * np.log10(achieved_octave / target_octave)
        rms_db_error = np.sqrt(np.nanmean(db_error ** 2))

        history_achieved_octave.append(achieved_octave)
        history_rms_db_error.append(rms_db_error)
        history_coherence_narrow.append(narrowband_coherence(Sxx_it, Syy_it, Sxy_it))
        print(f"[{label}] iteration {it}: RMS dB error = {rms_db_error:.3f} dB "
              f"(band max = {np.nanmax(np.abs(db_error)):.3f} dB)")

        correction = target_octave / achieved_octave
        correction_narrow = expand_octave_to_narrowband(f_nb_it, lower, upper, correction)
        drive_psd_narrow[in_band] *= correction_narrow[in_band]

        final_drive_time, final_resp_time = drive_time, resp_time
        final_f_nb, final_Sxx, final_Syy = f_nb_it, Sxx_it, Syy_it

    return dict(
        label=label,
        history_achieved_octave=history_achieved_octave,
        history_rms_db_error=history_rms_db_error,
        history_coherence_narrow=history_coherence_narrow,
        final_drive_time=final_drive_time, final_resp_time=final_resp_time,
        final_f_nb=final_f_nb, final_Sxx=final_Sxx, final_Syy=final_Syy,
    )


result_nb = run_control_loop(drive_psd_narrowband_init, "narrowband H1 init")
result_oct = run_control_loop(drive_psd_octave_init, "octave-only H init")

np.savez(
    os.path.join(RESULTS_DIR, "siso_octave_control_time_series.npz"),
    t=t, fs=fs, excitation_node=excitation_node, response_node=response_node,
    drive_narrowband_init=result_nb['final_drive_time'], response_narrowband_init=result_nb['final_resp_time'],
    drive_octave_init=result_oct['final_drive_time'], response_octave_init=result_oct['final_resp_time'],
)

# ---------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------
NARROW_STYLE = dict(lw=1.0, alpha=0.9)
COLOR_NB, COLOR_OCT = 'C0', 'C3'

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

ax = axes[0, 0]
ax.plot(range(n_iterations), result_nb['history_rms_db_error'], 'o-', color=COLOR_NB, label=result_nb['label'])
ax.plot(range(n_iterations), result_oct['history_rms_db_error'], 'o-', color=COLOR_OCT, label=result_oct['label'])
ax.set_title("RMS 1/6-octave error vs iteration")
ax.set_xlabel("Iteration"); ax.set_ylabel("RMS error (dB)")
ax.set_xticks(range(n_iterations)); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.loglog(result_nb['final_f_nb'], result_nb['final_Syy'], color=COLOR_NB, **NARROW_STYLE, label=result_nb['label'])
ax.loglog(result_oct['final_f_nb'], result_oct['final_Syy'], color=COLOR_OCT, **NARROW_STYLE, label=result_oct['label'])
ax.loglog(centers, target_octave, 's--', color='k', ms=4, label='1/6 octave target')
ax.set_title(f"Final narrowband response PSD -- node {response_node} X")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (g^2/Hz)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[0, 2]
zoom_lo, zoom_hi = 380.0, 520.0
mask_nb = (result_nb['final_f_nb'] >= zoom_lo) & (result_nb['final_f_nb'] <= zoom_hi)
mask_oct = (result_oct['final_f_nb'] >= zoom_lo) & (result_oct['final_f_nb'] <= zoom_hi)
ax.semilogy(result_nb['final_f_nb'][mask_nb], result_nb['final_Syy'][mask_nb], color=COLOR_NB, **NARROW_STYLE, label=result_nb['label'])
ax.semilogy(result_oct['final_f_nb'][mask_oct], result_oct['final_Syy'][mask_oct], color=COLOR_OCT, **NARROW_STYLE, label=result_oct['label'])
ax.axhline(target_level, color='k', ls='--', lw=0.8, label='target')
ax.set_title(f"Zoomed {zoom_lo:.0f}-{zoom_hi:.0f} Hz (densely modal)")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (g^2/Hz)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.loglog(f_nb, drive_psd_narrowband_init, color=COLOR_NB, **NARROW_STYLE, label=result_nb['label'])
ax.loglog(f_nb, drive_psd_octave_init, color=COLOR_OCT, **NARROW_STYLE, label=result_oct['label'])
ax.set_title("Initial drive PSD (iteration 0 starting point)")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Drive PSD (N^2/Hz)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[1, 1]
ax.semilogx(result_nb['final_f_nb'], result_nb['history_coherence_narrow'][-1], color=COLOR_NB, **NARROW_STYLE, label=result_nb['label'])
ax.semilogx(result_oct['final_f_nb'], result_oct['history_coherence_narrow'][-1], color=COLOR_OCT, **NARROW_STYLE, label=result_oct['label'])
ax.set_title(f"Final coherence -- node {response_node} / node {excitation_node}")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("gamma^2")
ax.set_xlim(fmin, fmax); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

ax = axes[1, 2]
ax.semilogx(centers, result_nb['history_achieved_octave'][-1], 'o-', color=COLOR_NB, label=result_nb['label'])
ax.semilogx(centers, result_oct['history_achieved_octave'][-1], 'o-', color=COLOR_OCT, label=result_oct['label'])
ax.semilogx(centers, target_octave, 's--', color='k', ms=4, label='target')
ax.set_yscale('log')
ax.set_title("Final 1/6-octave achieved vs target\n(both scenarios hit the same octave spec)")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (g^2/Hz)")
ax.set_xlim(fmin, fmax); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

fig.suptitle(f"Narrowband H1 init vs octave-only H init: node {excitation_node} -> node {response_node}, "
             f"flat {target_level:g} g^2/Hz target, {fmin:.0f}-{fmax:.0f} Hz")
fig.tight_layout()
plot_filename = os.path.join(RESULTS_DIR, "siso_octave_control_init_comparison.png")
fig.savefig(plot_filename, dpi=150)

print(f"\nTime series written to: {os.path.join(RESULTS_DIR, 'siso_octave_control_time_series.npz')}")
print(f"Plot written to:        {plot_filename}")
