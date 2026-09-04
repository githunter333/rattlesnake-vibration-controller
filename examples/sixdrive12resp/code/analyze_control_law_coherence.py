#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_control_law_coherence.py

Reads one or more "Save Spectral Data" .nc4 files written by Rattlesnake's
random-vibration sys-ID/control environment (RandomVibrationSysIdEnvironment.
save_spectral_data) and reports the ACHIEVED pairwise drive-to-drive
coherence -- i.e. what a control law actually produced on the plant, not
what we assume it produced.

This is the empirical counterpart to control_laws.py's _cap_drive_coherence:
that function enforces a coherence cap in the control-law math; this script
measures, from real saved spectral data, whether/how much coherence a given
law+test-level combination is actually running at, so the uncapped-vs-
capped, level-by-level comparison (match trace -> pseudoinverse -> buzz ->
optimal diagonal) has real numbers behind it rather than an assumption
that "no cap" means "coherence hits 1" everywhere.

Each .nc4 written by save_spectral_data has:
  - top-level attrs: sample_rate, time_per_read (= samples_per_read /
    sample_rate, i.e. the FFT frame period), hardware, hardware_file, ...
  - exactly one group, named after the environment (e.g. "Frame 6x12
    Random"), containing:
      drive_cpsd_real, drive_cpsd_imag   (fft_lines, drive_channels, drive_channels)
      frf_coherence                       (fft_lines, specification_channels)  <- response-side FRF coherence, NOT what we want here
      frf_data_real/imag, response_cpsd_real/imag, *_noise_* -- unused here

There's no explicit frequency array stored, so frequency is reconstructed
from sample_rate/time_per_read (df = 1/time_per_read, standard periodogram
relationship for the frame length that was actually used) -- flagged
"approximate" in the printout since it depends on time_per_read exactly
equalling the analysis frame period, which should hold for Rattlesnake's
own writes but is worth a sanity check against the GUI's displayed
frequency spacing if the numbers look off.

Usage:
    python analyze_control_law_coherence.py file1.nc4 [file2.nc4 ...] \
        [--band FLO FHI] [--cap 0.95] [--label file1.nc4=uncapped_-6dB ...]

    --band FLO FHI   restrict stats to this frequency band in Hz
                     (default: whole spectrum)
    --cap C          coherence threshold to report exceedance stats against
                     (default 0.95, matching the control-law cap parameter)
    --label F=NAME   friendly name for a file in the comparison table
                     (repeatable; default label is the filename)

With multiple files, prints a side-by-side comparison table (one row per
file) so you can see max/mean coherence and %-of-(bin,pair)-combos-over-cap
trend as test level increases and/or as the 0.95 cap is turned on.
"""

import sys
import os
import argparse
import numpy as np

try:
    import netCDF4 as nc4
except ImportError:
    print("ERROR: netCDF4 not installed in this environment. On the "
          "Rattlesnake machine's own python/conda env this should already "
          "be present (Rattlesnake itself depends on it for streaming "
          "files); run this script with that same interpreter.", file=sys.stderr)
    raise

DEFAULT_NL_SYSTEM_FILE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results",
    "sdynpy_frame6x12_system_nonlinear_allmodes.npz"))


def load_nl_reference_vrms(path=DEFAULT_NL_SYSTEM_FILE):
    """Reads the nonlinear system's calibration reference voltage
    (nl_reference_drive_level_vrms, written by
    build_nonlinear_frf_system_allmodes.py) straight from the npz, so this
    script's printed "1x"/"2x" comparison points stay correct after that
    script's REFERENCE_DRIVE_LEVEL_VRMS is recalibrated -- rather than a
    hardcoded number here silently going stale. Returns None (with a
    warning on stderr) if the file is missing or doesn't have the field,
    so callers can fall back to omitting the comparison rather than
    printing a wrong number.
    """
    if not os.path.exists(path):
        print(f"WARNING: nonlinear system file not found at {path!r} -- "
              f"skipping REFERENCE_DRIVE_LEVEL_VRMS comparison in the RMS "
              f"table (pass --nl-system to point at the right file).", file=sys.stderr)
        return None
    try:
        with np.load(path) as d:
            if 'nl_reference_drive_level_vrms' not in d:
                print(f"WARNING: {path!r} has no nl_reference_drive_level_vrms field -- "
                      f"skipping REFERENCE_DRIVE_LEVEL_VRMS comparison.", file=sys.stderr)
                return None
            return float(d['nl_reference_drive_level_vrms'])
    except Exception as e:
        print(f"WARNING: could not read nl_reference_drive_level_vrms from {path!r} ({e}) -- "
              f"skipping REFERENCE_DRIVE_LEVEL_VRMS comparison.", file=sys.stderr)
        return None


def load_drive_cpsd(path):
    """Returns (freqs_hz, drive_cpsd[fft_lines, N, N] complex, group_name, sample_rate)."""
    ds = nc4.Dataset(path, 'r')
    try:
        sample_rate = float(ds.sample_rate)
        time_per_read = float(ds.time_per_read)
        # save_spectral_data always also creates a 'channels' group (holding
        # per-channel calibration metadata, from the '/channels/'+label
        # variable-creation calls) alongside the actual environment group --
        # so pick the group that actually has drive_cpsd_real rather than
        # assuming there's only one group.
        candidate_names = [name for name, grp in ds.groups.items() if 'drive_cpsd_real' in grp.variables]
        if len(candidate_names) != 1:
            raise RuntimeError(f"{path}: expected exactly one group with drive_cpsd_real, "
                                f"found {candidate_names} (all groups: {list(ds.groups.keys())})")
        g = ds.groups[candidate_names[0]]
        group_names = candidate_names
        real = np.array(g.variables['drive_cpsd_real'][...])
        imag = np.array(g.variables['drive_cpsd_imag'][...])
        cpsd = real + 1j * imag
        n_lines = cpsd.shape[0]
        # BUGFIX (2026-09-02): frequency spacing is sample_rate/nfft (nfft =
        # 2*(fft_lines-1)), NOT 1/time_per_read -- those two only agree if
        # the acquisition frame length exactly equals the FFT block length,
        # which isn't the case here (e.g. sample_rate=4096, time_per_read=
        # 0.25s implies df=4Hz, but fft_lines=2049 -> nfft=4096 -> the real
        # df is 1Hz). Using 1/time_per_read silently used a 4x-too-coarse,
        # mislabeled frequency axis -- a "20-1000Hz" band selection with the
        # wrong df was actually only pulling in true frequencies up to
        # ~250Hz. sample_rate/nfft is the only formula that's correct
        # regardless of any relationship between the acquisition frame and
        # the FFT block.
        nfft = 2 * (n_lines - 1)
        df = sample_rate / nfft
        freqs = np.arange(n_lines) * df
        return freqs, cpsd, group_names[0], sample_rate
    finally:
        ds.close()


def per_channel_rms(cpsd, freqs, band=None):
    """RMS drive voltage per channel from the diagonal of the CPSD (one-sided
    PSD convention, V^2/Hz): rms_i = sqrt(sum(diag_i) * df), summed over the
    given band (default: whole spectrum). This is the direct, no-guessing
    measure of "how hard was this channel actually driven" for a given
    saved snapshot -- use it to correlate against the nonlinear system's
    calibrated "1x"/"2x" reference drive voltage (see
    load_nl_reference_vrms / build_nonlinear_frf_system_allmodes.py)
    rather than guessing from the GUI's Target Test Level dB alone, since
    that idealized calibration assumed a single mode-shaped sine, not a
    real broadband random spec.
    """
    diag = np.real(np.diagonal(cpsd, axis1=1, axis2=2))  # (F, N)
    if band is not None:
        flo, fhi = band
        mask = (freqs >= flo) & (freqs <= fhi)
    else:
        mask = np.ones_like(freqs, dtype=bool)
    if len(freqs) > 1:
        df = freqs[1] - freqs[0]
    else:
        df = 1.0
    return np.sqrt(np.sum(diag[mask], axis=0) * df)  # (N,)


def pairwise_coherence(cpsd):
    """cpsd: (F, N, N) complex -> coherence: (F, N, N) real, diag=1, NaN where
    a diagonal auto-spectrum is ~0 (no drive on that channel)."""
    F, N, _ = cpsd.shape
    diag = np.real(np.diagonal(cpsd, axis1=1, axis2=2))  # (F, N)
    denom = np.sqrt(diag[:, :, None] * diag[:, None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        coh = np.abs(cpsd) / denom
    coh[denom <= 1e-30] = np.nan
    return coh


def summarize(path, label, band, cap):
    freqs, cpsd, group_name, sample_rate = load_drive_cpsd(path)
    coh = pairwise_coherence(cpsd)
    N = coh.shape[-1]
    iu = np.triu_indices(N, k=1)

    if band is not None:
        flo, fhi = band
        mask = (freqs >= flo) & (freqs <= fhi)
    else:
        mask = np.ones_like(freqs, dtype=bool)

    off_diag = coh[mask][:, iu[0], iu[1]]  # (F_band, n_pairs)
    valid = ~np.isnan(off_diag)
    vals = off_diag[valid]

    n_total = vals.size
    n_over = int(np.sum(vals > cap))
    pct_over = 100.0 * n_over / n_total if n_total else float('nan')
    max_coh = float(np.max(vals)) if n_total else float('nan')
    mean_coh = float(np.mean(vals)) if n_total else float('nan')
    median_coh = float(np.median(vals)) if n_total else float('nan')

    # which pair/bin hits the max (for a concrete pointer, not just a number)
    if n_total:
        flat_idx = np.nanargmax(np.where(np.isnan(off_diag), -np.inf, off_diag))
        fi, pi = np.unravel_index(flat_idx, off_diag.shape)
        worst_pair = (int(iu[0][pi]) + 1, int(iu[1][pi]) + 1)  # 1-based drive channel numbers
        worst_freq = float(freqs[mask][fi])
    else:
        worst_pair = None
        worst_freq = None

    rms = per_channel_rms(cpsd, freqs, band)

    return dict(
        path=path, label=label, group_name=group_name, sample_rate=sample_rate,
        n_drive=N, band=band, n_bins=int(mask.sum()), n_pairs=len(iu[0]),
        max_coh=max_coh, mean_coh=mean_coh, median_coh=median_coh,
        pct_over_cap=pct_over, n_over_cap=n_over, n_total=n_total,
        worst_pair=worst_pair, worst_freq_hz=worst_freq,
        rms_per_channel=rms, rms_mean=float(np.mean(rms)), rms_max=float(np.max(rms)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--band', nargs=2, type=float, metavar=('FLO', 'FHI'), default=None)
    ap.add_argument('--cap', type=float, default=0.95)
    ap.add_argument('--label', action='append', default=[], metavar='FILE=NAME')
    ap.add_argument('--nl-system', default=DEFAULT_NL_SYSTEM_FILE, metavar='NPZ',
                     help='Path to the nonlinear system npz to pull the calibrated '
                          'reference drive voltage from for the RMS comparison '
                          '(default: %(default)s)')
    args = ap.parse_args()

    labels = {}
    for item in args.label:
        f, name = item.split('=', 1)
        labels[f] = name

    rows = []
    for f in args.files:
        label = labels.get(f, f.rsplit('/', 1)[-1])
        rows.append(summarize(f, label, args.band, args.cap))

    band_str = f"{args.band[0]:.0f}-{args.band[1]:.0f} Hz" if args.band else "full spectrum"
    print(f"\nDrive-to-drive coherence summary ({band_str}, cap={args.cap})\n")
    hdr = f"{'label':<28} {'n_drive':>7} {'max':>7} {'mean':>7} {'median':>7} {'%>cap':>7} {'worst pair @ Hz':>20}"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        worst = f"({r['worst_pair'][0]},{r['worst_pair'][1]}) @ {r['worst_freq_hz']:.1f}" if r['worst_pair'] else "--"
        print(f"{r['label']:<28} {r['n_drive']:>7} {r['max_coh']:>7.3f} {r['mean_coh']:>7.3f} "
              f"{r['median_coh']:>7.3f} {r['pct_over_cap']:>6.2f}% {worst:>20}")
    print()

    ref_vrms = load_nl_reference_vrms(args.nl_system)
    if ref_vrms is not None:
        print(f"Per-channel drive RMS voltage ({band_str}) -- compare against the nonlinear\n"
              f"system's calibrated REFERENCE_DRIVE_LEVEL_VRMS={ref_vrms:g} ('1x reference', "
              f"modest engagement)\n"
              f"and 'x2 reference'={2*ref_vrms:g} (full calibrated target shift) design points\n"
              f"(from {args.nl_system}):\n")
    else:
        print(f"Per-channel drive RMS voltage ({band_str}):\n")
    hdr2 = f"{'label':<28} " + " ".join(f"ch{i+1:>5}" for i in range(rows[0]['n_drive'])) + f" {'mean':>7} {'max':>7}"
    print(hdr2)
    print('-' * len(hdr2))
    for r in rows:
        chvals = " ".join(f"{v:7.2f}" for v in r['rms_per_channel'])
        print(f"{r['label']:<28} {chvals} {r['rms_mean']:>7.2f} {r['rms_max']:>7.2f}")
    print()


if __name__ == '__main__':
    main()
