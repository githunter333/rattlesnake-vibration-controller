"""Quick sanity check on a saved Rattlesnake run, before trusting it.

Answers the one question that matters immediately after a run: did the file
capture the CONVERGED control state, or a snapshot from before the test level
came up?  The first saved run of the 2026-09-12 comparison looked fine on
screen but landed in the file 28.6 dB below specification with a drive trace
seven orders of magnitude below a working run -- caught only by scoring it.

Deliberately imports nothing but netCDF4 and numpy, so it starts instantly and
runs in any environment, unlike the full scoring harness.

Usage
-----
    python check_run.py RUN.nc4 [RUN2.nc4 ...]
"""

from __future__ import annotations

import os
import sys

import netCDF4 as nc4
import numpy as np

BAND = (100.0, 1000.0)


def check(path, band=BAND):
    """Print a one-screen verdict for one run file."""
    name = os.path.basename(path)
    try:
        dataset = nc4.Dataset(path, 'r')
    except OSError as exc:
        print(f'{name}: cannot open ({exc})')
        return False
    try:
        groups = [g for g in dataset.groups if g != 'channels']
        if not groups:
            print(f'{name}: no environment group')
            return False
        group = dataset[groups[0]]
        # A system-ID run carries reference_cpsd_* where a control run carries
        # drive_cpsd_*, and has no controlled response to judge.
        missing = [v for v in ('response_cpsd_real', 'drive_cpsd_real',
                               'specification_cpsd_matrix_real',
                               'specification_frequency_lines')
                   if v not in group.variables]
        if missing:
            kind = ('system-identification run'
                    if 'reference_cpsd_real' in group.variables
                    else 'not a controlled random run')
            print(f'{name}: {kind} -- nothing to check')
            return None

        frequencies = np.array(group['specification_frequency_lines'][:], dtype=float)
        in_band = (frequencies >= band[0]) & (frequencies <= band[1])
        response = np.einsum('fmm->fm', np.array(
            group['response_cpsd_real'][:], dtype=float))[in_band]
        spec = np.einsum('fmm->fm', np.array(
            group['specification_cpsd_matrix_real'][:], dtype=float))[in_band]
        drive = np.einsum('fnn->fn', np.array(
            group['drive_cpsd_real'][:], dtype=float))[in_band]

        response_trace = response.sum(axis=1).mean()
        spec_trace = spec.sum(axis=1).mean()
        level_db = 10.0 * np.log10(max(response_trace, 1e-300)
                                   / max(spec_trace, 1e-300))

        # Per-channel error, the same quantity the scoring harness reports.
        with np.errstate(divide='ignore', invalid='ignore'):
            error_db = 10.0 * np.log10(np.maximum(response, 1e-300)
                                       / np.maximum(spec, 1e-300))
        valid = np.isfinite(error_db) & (spec > 0)
        rms_db = float(np.sqrt(np.mean(error_db[valid] ** 2))) if valid.any() else np.nan
        percent_out = (100.0 * np.mean(np.abs(error_db[valid]) > 6.0)
                       if valid.any() else np.nan)

        law = (group.getncattr('control_python_function')
               if 'control_python_function' in group.ncattrs() else '?')
        params = (str(group.getncattr('control_python_function_parameters')).strip()
                  if 'control_python_function_parameters' in group.ncattrs() else '')
        averaging = (group.getncattr('control_averaging_type')
                     if 'control_averaging_type' in group.ncattrs() else 'Linear')
        coefficient = (group.getncattr('control_averaging_coefficient')
                       if 'control_averaging_coefficient' in group.ncattrs() else 0.0)
        plant = (os.path.basename(str(dataset.getncattr('hardware_file')))
                 if 'hardware_file' in dataset.ncattrs() else '?')
        samples = (len(dataset.dimensions['time_samples'])
                   if 'time_samples' in dataset.dimensions else 0)

        # A converged run sits within a couple of dB of the specification in
        # total power.  Far below means the snapshot predates the level ramp;
        # far above means it was still overshooting.
        if level_db < -6.0:
            verdict = ('NOT CONVERGED -- saved below test level; '
                       'the level had not come up when this was written')
        elif level_db > 6.0:
            verdict = 'NOT CONVERGED -- saved while overshooting'
        elif rms_db > 15.0:
            verdict = 'SUSPECT -- level is right but per-channel error is very large'
        else:
            verdict = 'looks converged'

        print(f'{name}')
        print(f'   plant {plant}   law {law} [{params or "defaults"}]')
        print(f'   averaging {averaging} {coefficient:g}   '
              f'streaming samples {samples}')
        print(f'   total power vs spec  {level_db:+7.2f} dB')
        print(f'   mean drive trace     {drive.sum(axis=1).mean():.4e}')
        print(f'   per-channel rms      {rms_db:7.2f} dB, {percent_out:.1f}% of lines '
              f'outside +/-6 dB')
        print(f'   -> {verdict}')
        return abs(level_db) <= 6.0 and rms_db <= 15.0
    finally:
        dataset.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    outcomes = [check(p) for p in sys.argv[1:]]
    judged = [o for o in outcomes if o is not None]
    skipped = len(outcomes) - len(judged)
    note = f' ({skipped} skipped)' if skipped else ''
    print(f'\n{sum(judged)}/{len(judged)} run(s) look converged{note}')
    raise SystemExit(0 if judged and all(judged) else 1)


if __name__ == '__main__':
    main()
