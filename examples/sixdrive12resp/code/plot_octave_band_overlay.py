# -*- coding: utf-8 -*-
"""
plot_octave_band_overlay.py

Visualizes octave_band_switching_control's sixth-octave control level on
the SAME narrowband frequency axis Rattlesnake's own Run tab plots use --
producing a "staircase": flat within each 1/6-octave band, stepping at
each band edge -- overlaid on the raw narrowband response/specification.

Reuses spectral_analysis/fractional_octave.py's octave_band_frequencies()
and octave_band_psd() for the band edges and energy-preserving band
averaging (the SAME math used in the validated SISO octave-control demo),
NOT the plain-complex-mean-of-H approximation that lives inside
octave_band_switching_control.py for speed. That makes this plot a good
independent check of how much error that approximation is costing, since
it band-averages the actual measured response power directly.

Input
-----
A netCDF file saved from Rattlesnake's Run tab "Save Current Spectral
Data" button (components/random_vibration_sys_id_environment.py,
RandomVibrationEnvironment.save_spectral_data). Confirmed by inspecting an
actual saved file from this repo (examples/sixdrive12resp/results/
nonlinearspectraldata082126.nc4): under a group named after the running
environment (e.g. "Random" for a direct RANDOM launch, or the profile's
sheet name like "Frame 6x12 Random" if loaded from a Combined profile),
the group holds BOTH the measured data and the specification actually used
for that run -- so no separate spec file is needed, and there's no
axis-order or frequency-grid mismatch to worry about:
    response_cpsd_real/imag              (fft_lines, spec_channels, spec_channels)
    drive_cpsd_real/imag                 (fft_lines, drive_channels, drive_channels)
    specification_frequency_lines        (fft_lines,) -- actual Hz values, confirmed
    specification_cpsd_matrix_real/imag  (fft_lines, spec_channels, spec_channels)
plus sample_rate (global attr) and samples_per_frame (group attr), used
only as a cross-check against specification_frequency_lines.

Usage
-----
    python plot_octave_band_overlay.py SPECTRAL_DATA.nc4 \
        --environment "Frame 6x12 Random" --fmin 100 --fmax 1000 --fraction 6 \
        --out overlay.png

Defaults for --fmin/--fmax/--fraction match the sixdrive12resp demo's
current octave_band_switching_control extra_parameters string
(...,1.0,0.0,6 -> frequency_spacing=1.0, switch_level_db=0.0, octave_fraction=6)
and its auto-detected band range (100-1000 Hz, confirmed in gui_debug.log:
"_build_bands: 20 bands over 100.00-1000.00 Hz (1/6 octave)"). --environment
has no single correct default -- it depends on how you launched Rattlesnake
(bare "Random" for `make launch-rattlesnake`, or the profile sheet name if
loaded from a saved Combined profile) -- so it's left required-in-spirit
with a guess; the script lists available groups if the name doesn't match.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4 as nc4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from spectral_analysis.fractional_octave import octave_band_frequencies, octave_band_psd


def load_spectral_data(nc4_path, environment_name):
    """Returns (freq, response_diag_psd[F,channels], drive_diag_psd[F,drives],
    spec_diag[F,channels])."""
    ds = nc4.Dataset(nc4_path, 'r')
    try:
        if environment_name not in ds.groups:
            available = [g for g in ds.groups if g != 'channels']
            raise KeyError(
                f'No group "{environment_name}" in {nc4_path}. Available groups: {available}')
        group = ds.groups[environment_name]

        resp_re = np.array(group.variables['response_cpsd_real'][:])
        resp_im = np.array(group.variables['response_cpsd_imag'][:])
        response_cpsd = resp_re + 1j * resp_im  # (F, channels, channels)

        drive_re = np.array(group.variables['drive_cpsd_real'][:])
        drive_im = np.array(group.variables['drive_cpsd_imag'][:])
        drive_cpsd = drive_re + 1j * drive_im  # (F, drives, drives)

        spec_re = np.array(group.variables['specification_cpsd_matrix_real'][:])
        spec_im = np.array(group.variables['specification_cpsd_matrix_imag'][:])
        spec_cpsd = spec_re + 1j * spec_im  # (F, channels, channels)

        freq = np.array(group.variables['specification_frequency_lines'][:])

        # Cross-check against sample_rate/samples_per_frame when available (belt-and-suspenders;
        # specification_frequency_lines is the authoritative axis actually saved with the data).
        try:
            expected_spacing = float(ds.sample_rate) / float(group.samples_per_frame)
            actual_spacing = freq[1] - freq[0]
            if not np.isclose(expected_spacing, actual_spacing, rtol=1e-6):
                print(f'WARNING: frequency_spacing from sample_rate/samples_per_frame '
                      f'({expected_spacing:g} Hz) does not match specification_frequency_lines '
                      f'spacing ({actual_spacing:g} Hz) -- using the latter.')
        except (AttributeError, KeyError):
            pass

        response_diag = np.einsum('fcc->fc', response_cpsd).real
        drive_diag = np.einsum('fdd->fd', drive_cpsd).real
        spec_diag = np.einsum('fcc->fc', spec_cpsd).real
        return freq, response_diag, drive_diag, spec_diag
    finally:
        ds.close()


def band_of_line(freq, lower, upper):
    """Assigns each narrowband line to a band index (or -1 if outside all bands)."""
    band_idx = np.full(freq.shape, -1, dtype=int)
    for b, (lo, hi) in enumerate(zip(lower, upper)):
        band_idx[(freq >= lo) & (freq < hi)] = b
    return band_idx


def octave_staircase(freq, narrowband_psd, lower, upper, band_idx):
    """Band-averages narrowband_psd (energy-preserving) and broadcasts each
    band's level back across its narrowband lines, NaN outside [lower[0], upper[-1])."""
    band_levels = octave_band_psd(freq, narrowband_psd, lower, upper)
    staircase = np.full(freq.shape, np.nan)
    in_band = band_idx >= 0
    staircase[in_band] = band_levels[band_idx[in_band]]
    return staircase, band_levels


def plot_overlay(freq, narrowband, spec, staircase, channel_labels, title, out_path):
    n_channels = narrowband.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2.2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        ax.semilogy(freq, spec[:, c], color='tab:blue', lw=1, alpha=0.6, label='Specification')
        ax.semilogy(freq, narrowband[:, c], color='tab:red', lw=0.6, alpha=0.5,
                    label='Narrowband response')
        ax.semilogy(freq, staircase[:, c], color='tab:orange', lw=2.0, drawstyle='steps-post',
                    label='1/6-octave band level')
        ax.set_ylabel(channel_labels[c])
        ax.grid(True, which='both', alpha=0.3)
        if c == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spectral_data_nc4', help='File saved via "Save Current Spectral Data"')
    parser.add_argument('--environment', default='Random',
                        help='Group name inside the nc4 -- "Random" for a bare '
                             '`make launch-rattlesnake` (RANDOM) launch, or the profile sheet '
                             'name (e.g. "Frame 6x12 Random") if loaded from a saved Combined '
                             'profile. Run without a matching group to see what is available.')
    parser.add_argument('--fmin', type=float, default=100.0)
    parser.add_argument('--fmax', type=float, default=1000.0)
    parser.add_argument('--fraction', type=int, default=6)
    parser.add_argument('--source', choices=['response', 'drive'], default='response',
                        help='Which diagonal PSD to band-average and plot. Default: response '
                             '(what the Run tab error plot shows).')
    parser.add_argument('--out', default='octave_band_overlay.png')
    args = parser.parse_args()

    freq, response_diag, drive_diag, spec_diag = load_spectral_data(
        args.spectral_data_nc4, args.environment)

    narrowband = response_diag if args.source == 'response' else drive_diag
    n_channels = narrowband.shape[1]
    if args.source == 'response' and spec_diag.shape[1] != n_channels:
        raise ValueError(
            f'Spec has {spec_diag.shape[1]} channels but response has {n_channels}; '
            'check --environment / file pairing.')

    centers, lower, upper = octave_band_frequencies(args.fmin, args.fmax, args.fraction)
    idx = band_of_line(freq, lower, upper)

    staircase = np.full_like(narrowband, np.nan)
    for c in range(n_channels):
        staircase[:, c], _ = octave_staircase(freq, narrowband[:, c], lower, upper, idx)

    labels = [f'Ch {c + 1}' for c in range(n_channels)]
    if args.source == 'response':
        plot_overlay(freq, narrowband, spec_diag, staircase, labels,
                    f'Response vs 1/6-octave band level ({args.fmin:g}-{args.fmax:g} Hz)',
                    args.out)
    else:
        # No meaningful "specification" for drive voltage; plot narrowband alone under the staircase
        plot_overlay(freq, narrowband, narrowband, staircase, labels,
                    f'Drive CPSD vs 1/6-octave band level ({args.fmin:g}-{args.fmax:g} Hz)',
                    args.out)


if __name__ == '__main__':
    main()
