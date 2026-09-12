"""Score Rattlesnake random-vibration runs with SDynPy's RandomVibTest.

Why this exists
---------------
Every control-law comparison so far used our own CPSD estimation choices and
our own eyeball on when a run had converged.  SDynPy ships a standardised
diagnostic suite (``sdynpy.doc.sdynpy_vibration_test.RandomVibTest``) that
uses the spectral processing parameters recorded in the file itself, which
removes both confounds and produces the metric an actual test is judged by
(percent of lines outside tolerance).

Two things had to be solved to use it here:

1. ``RandomVibTest.load_rattlesnake_streaming_data`` needs streaming time
   histories.  Our runs were saved with ``time_samples = 0`` -- spectral data
   only -- so that entry point raises.  But the scoring methods
   (``plot_control_vs_spec``, ``plot_percent_lines_out``, ``plot_rms_error``,
   ``plot_response``, ``plot_rms_level``) touch ``self.time_data`` ONLY to
   take ``len()`` for figure bookkeeping; the real work iterates
   ``self.cpsd``.  So pre-populating ``cpsd`` from the file's response CPSD
   and passing a length-1 placeholder ``time_data`` drives the whole suite
   without any time data.  ``plot_kurtosis``, ``plot_time_histories`` and
   ``plot_cpsd_time_subset`` genuinely need time data and stay unavailable.

2. A law driving toward the *projected* (best-achievable) target must not be
   judged against the flat spec stored in the file -- that target is provably
   unreachable with 6 drives and 8 control channels, so the law would be
   condemned for doing exactly what it was designed to do.  ``score_run``
   therefore scores twice: against the flat spec (contractual: "would this
   test pass as written") and against the projected target (physical: "is the
   law delivering what is actually achievable").  The gap between the two is
   the achievable floor, expressed in the same metric as everything else.

Usage
-----
    python score_runs_sdynpy.py RUN.nc4 [RUN2.nc4 ...] \
        [--projected ../results/analysis/projected_target_100_1000Hz.npz] \
        [--tolerance-db 6] [--figures ../results/figures/sdynpy_scoring]
"""

from __future__ import annotations

import argparse
import os
import warnings

# This is a batch tool that never needs an interactive window.  It must be
# said BEFORE sdynpy is imported, because sdynpy pulls in pyvista and hence
# Qt: on an Anaconda env carrying both PyQt5 and Qt6 the two register the
# same Objective-C classes and the process segfaults the moment matplotlib
# tries to open a Qt canvas.  Agg avoids the canvas entirely.
os.environ.setdefault('MPLBACKEND', 'Agg')

# sdynpy imports pyvista, which initializes Qt at import time.  On a headless
# machine that aborts the process before any of our code runs, so ask Qt for
# the offscreen platform there.  A Mac or an X/Wayland session has a real
# display and is left alone.
import sys
if sys.platform not in ('darwin', 'win32') and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Agg', force=True)

import netCDF4 as nc4
import numpy as np

from sdynpy.core.sdynpy_data import power_spectral_density_array
from sdynpy.fileio.sdynpy_rattlesnake import read_random_spectral_data
from sdynpy.doc.sdynpy_vibration_test import RandomVibTest

_TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------
# Loading a spectral-only run
# --------------------------------------------------------------------------
# A controlled random run and a system-identification run share most of their
# variables; the discriminator is that control writes drive_cpsd_* while system
# ID writes reference_cpsd_* instead.  Require everything
# read_random_spectral_data actually dereferences, or the file fails deep
# inside SDynPy rather than being skipped cleanly.
REQUIRED_VARIABLES = ('response_cpsd_real', 'response_cpsd_imag',
                      'specification_cpsd_matrix_real',
                      'specification_cpsd_matrix_imag',
                      'specification_frequency_lines',
                      'drive_cpsd_real', 'drive_cpsd_imag')


def _band_psd(matrix, frequencies, control_coordinate):
    """Build the (2, n_control) lower/upper band array SDynPy expects.

    Mirrors ``RattlesnakeRandomEnvironmentData.specification_warning_psd``.
    Returns None when the band is absent or entirely zero/NaN, which is how
    an unpopulated profile column shows up -- and is the case for every
    profile in this study.
    """
    if matrix is None:
        return None
    matrix = np.asarray(matrix)
    if matrix.size == 0 or not np.any(np.isfinite(matrix) & (matrix != 0)):
        return None
    coord = np.tile(control_coordinate, (2, 1)).T
    low = power_spectral_density_array(
        abscissa=frequencies,
        ordinate=np.moveaxis(matrix[0, ...], 0, -1),
        coordinate=coord)
    high = power_spectral_density_array(
        abscissa=frequencies,
        ordinate=np.moveaxis(matrix[-1, ...], 0, -1),
        coordinate=coord)
    return np.concatenate((low[np.newaxis, :], high[np.newaxis, :]))


# Attributes that fully describe how a run was produced.  Comparing runs that
# differ in any of these is comparing different experiments -- the first sweep
# of this campaign unknowingly mixed three different plants (linear, shifted
# and nonlinear), which produced an apparent "conditioning" effect that was
# really just the plant changing underneath.
PROVENANCE_ROOT = ('hardware', 'hardware_file', 'sample_rate',
                   'time_per_read', 'time_per_write', 'output_oversample')
PROVENANCE_GROUP = ('control_python_script', 'control_python_function',
                    'control_python_function_type',
                    'control_python_function_parameters',
                    'control_averaging_type', 'control_averaging_coefficient',
                    'frames_in_cpsd', 'cpsd_window', 'cpsd_overlap',
                    'samples_per_frame', 'update_tf_during_control',
                    'sysid_averaging_type', 'sysid_exponential_averaging_coefficient',
                    'sysid_averages', 'sysid_estimator', 'sysid_level')

# Of those, the ones that MUST agree for a set of runs to be comparable.  The
# control law and its parameters are deliberately absent: those are the
# independent variable.
MUST_MATCH = ('hardware', 'hardware_file', 'sample_rate', 'frames_in_cpsd',
              'cpsd_window', 'cpsd_overlap', 'samples_per_frame',
              'control_averaging_type', 'control_averaging_coefficient',
              'update_tf_during_control')


def read_provenance(dataset, environment):
    """Collect how a run was produced: plant, law, parameters, averaging.

    control_averaging_type / control_averaging_coefficient are absent on runs
    predating that feature; they are reported as 'Linear' / 0.0 to match what
    Rattlesnake itself assumes when reloading such a profile.
    """
    out = {}
    for key in PROVENANCE_ROOT:
        if key in dataset.ncattrs():
            out[key] = dataset.getncattr(key)
    group = dataset[environment]
    for key in PROVENANCE_GROUP:
        if key in group.ncattrs():
            out[key] = group.getncattr(key)
    out.setdefault('control_averaging_type', 'Linear')
    out.setdefault('control_averaging_coefficient', 0.0)
    if 'hardware_file' in out:
        plant = os.path.basename(str(out['hardware_file']))
        # The distinguishing part of these names is the SUFFIX
        # (_nonlinear_allmodes, _shifted_allmodes, or nothing), so a
        # left-truncated display keeps what actually tells them apart.
        out['plant'] = plant
        out['plant_short'] = plant if len(plant) <= 30 else '..' + plant[-28:]
    params = str(out.get('control_python_function_parameters', '')).strip()
    out['control_python_function_parameters'] = params
    out['parameters_shown'] = params if params else '(defaults)'
    return out


def compare_provenance(results):
    """Return {key: {value: [run names]}} for MUST_MATCH keys that disagree."""
    disagreements = {}
    for key in MUST_MATCH:
        seen = {}
        for r in results:
            value = r.get('provenance', {}).get(key, '(absent)')
            seen.setdefault(str(value), []).append(r['name'])
        if len(seen) > 1:
            disagreements[key] = seen
    return disagreements


def describe_dataset(dataset):
    """Return (is_scorable, reason) for an already-open Dataset.

    Takes an open handle rather than a path deliberately.  HDF5 does not
    tolerate the same file being open through several netCDF4 handles while
    one of them is closed -- doing so segfaults the interpreter on the NEXT
    file, which is exactly what a directory sweep does.  Every read in this
    module therefore shares one handle per file, closed once by score_run.
    """
    groups = [g for g in dataset.groups if g != 'channels']
    if not groups:
        return False, 'no environment group'
    group = dataset[groups[0]]
    missing = [v for v in REQUIRED_VARIABLES if v not in group.variables]
    if missing:
        kind = ('system-identification run'
                if 'reference_cpsd_real' in group.variables
                else 'not a controlled random run')
        return False, (f'{kind}; environment {groups[0]!r} has no '
                       + ', '.join(missing))
    return True, groups[0]


def bands_from_spec(specification, tolerance_db):
    """Build a (2, n_control) lower/upper band array at +/-tolerance_db.

    Rattlesnake writes the warning and abort matrices as all-NaN when the
    profile's warning/abort columns were never filled in, which is the case
    for every run in this study.  SDynPy's plotting routines dereference both
    bands unconditionally, so they are synthesized here from whichever
    specification is currently being scored against.  Same arithmetic as
    RandomVibTest.set_tolerance_limit_psd.
    """
    asd = specification.get_asd()
    low = asd / 10.0 ** (tolerance_db / 10.0)
    high = asd * 10.0 ** (tolerance_db / 10.0)
    return np.concatenate((low[np.newaxis, :], high[np.newaxis, :]))


def load_spectral_run(dataset, coordinate_override_column=None, default_unit='EU'):
    """Build a RandomVibTest from a Rattlesnake spectral-only nc4 file.

    Returns
    -------
    test : RandomVibTest
        Populated with the measured response CPSD and the specification,
        warning and abort bands read from the file.
    extras : dict
        ``drive_cpsd``, ``frequencies``, ``control_coordinate``,
        ``environment``, and ``n_drives``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Hand SDynPy the open handle; given a path it would open its own and
        # never close it, leaving two live handles on one file.
        response_cpsd, spec_cpsd, drive_cpsd = read_random_spectral_data(
            dataset, coordinate_override_column)

    if True:
        environment = [g for g in dataset.groups if g != 'channels'][0]
        group = dataset[environment]
        frequencies = np.array(group['specification_frequency_lines'][:], dtype=float)
        warning = _band_psd(np.array(group['specification_warning_matrix'][:]),
                            frequencies, response_cpsd.get_asd().response_coordinate.flatten()) \
            if 'specification_warning_matrix' in group.variables else None
        abort = _band_psd(np.array(group['specification_abort_matrix'][:]),
                          frequencies, response_cpsd.get_asd().response_coordinate.flatten()) \
            if 'specification_abort_matrix' in group.variables else None
        n_drives = drive_cpsd.shape[0]
        # Units for the control channels, in the same order as the CPSD.
        # SDynPy's axis labelling dereferences self.units unconditionally, so
        # a None here breaks plot_control_vs_spec and plot_response.  The
        # channel table is frequently blank in these profiles; fall back to a
        # neutral label rather than inventing a physical unit.
        environment_index = np.where(
            np.array(dataset['environment_names'][:]) == environment)[0][0]
        active = np.array(
            dataset['environment_active_channels'][:, environment_index]).astype(bool)
        all_units = np.array(dataset['channels']['unit'][:], dtype='<U32')
        control_indices = np.array(group['control_channel_indices'][:])
        units = all_units[active][control_indices]
        units = np.array([u if str(u).strip() else default_unit for u in units],
                         dtype='<U32')

    control_coordinate = response_cpsd.get_asd().response_coordinate.flatten()

    test = RandomVibTest(
        coordinate=control_coordinate,
        control_coordinate=control_coordinate,
        response_coordinate=control_coordinate,
        excitation_coordinate=None,
        time_data=[None],              # placeholder: only len() is ever taken
        cpsd=[response_cpsd],          # pre-populated, so compute_cpsd never runs
        specification_cpsd=spec_cpsd,
        specification_warning_psd=warning,
        specification_abort_psd=abort,
        units=units,
    )
    # The FRF the run was actually controlling with, and its conditioning over
    # the specified band.  Conditioning turns out to predict the reachability
    # limit better than any other quantity in the file, so it is carried
    # alongside the spectra rather than recomputed later.
    frf = (np.array(group['frf_data_real'][:], dtype=float)
           + 1j * np.array(group['frf_data_imag'][:], dtype=float))
    spec_diag = np.real(spec_cpsd.get_asd().ordinate).T          # (F, M)
    with np.errstate(invalid='ignore'):
        in_band = np.isfinite(spec_diag).all(axis=1) & (spec_diag > 0).any(axis=1)
    if np.any(in_band):
        singular = np.linalg.svd(frf[in_band], compute_uv=False)
        frf_condition = singular[:, 0] / np.maximum(singular[:, -1], _TINY)
    else:
        frf_condition = np.array([])

    provenance = read_provenance(dataset, environment)

    extras = dict(drive_cpsd=drive_cpsd, frequencies=frequencies,
                  provenance=provenance,
                  control_coordinate=control_coordinate,
                  environment=environment, n_drives=n_drives,
                  frf=frf, frf_condition=frf_condition,
                  bands_from_file=(warning is not None and abort is not None))
    return test, extras


# --------------------------------------------------------------------------
# Per-run achievable floor, computed from the run's OWN measured FRF
# --------------------------------------------------------------------------
def _load_achievable_diagonal():
    """Import ``achievable_diagonal`` from the repository's control_laws folder.

    Imported lazily and by explicit path: this script sits in
    examples/<case>/code/ and control_laws/ is three levels up, and the
    control_laws package is not installed.  Falls back to a plain import so
    the function still resolves if the module is already on sys.path.
    """
    try:
        from achievable_response import achievable_diagonal
        return achievable_diagonal
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, '..', '..', '..', 'control_laws', 'achievable_response.py'))
    if not os.path.exists(candidate):
        raise ImportError(
            'could not locate achievable_response.py; expected it at '
            f'{candidate}.  Pass --projected instead of --recompute-floor, or '
            'put control_laws on PYTHONPATH.')
    import importlib.util
    spec = importlib.util.spec_from_file_location('_achievable_response', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.achievable_diagonal



def read_run_frf(dataset):
    """Return (frequencies, H) where H is (F, M, N) from the run file itself.

    Every Rattlesnake random run stores the FRF it was actually controlling
    with.  Using it -- rather than a projected target saved from some other
    system identification -- makes the achievable floor exact for THIS run,
    which matters because the rig's damping has changed measurably between
    test campaigns.
    """
    environment = [g for g in dataset.groups if g != 'channels'][0]
    group = dataset[environment]
    frequencies = np.array(group['specification_frequency_lines'][:], dtype=float)
    frf = (np.array(group['frf_data_real'][:], dtype=float)
           + 1j * np.array(group['frf_data_imag'][:], dtype=float))
    spec = np.array(group['specification_cpsd_matrix_real'][:], dtype=float)
    target = np.einsum('fmm->fm', spec)
    return frequencies, frf, target


def achievable_floor_from_run(dataset, achievable_diagonal, band=None,
                              decimate=1, **kwargs):
    """Compute the achievable diagonal response for a run from its own FRF.

    Parameters
    ----------
    achievable_diagonal : callable
        ``control_laws.achievable_response.achievable_diagonal``, passed in so
        this module does not depend on the repository layout.
    band : (low, high) or None
        Frequency limits to solve over.  When None, every line whose
        specification is finite and positive is solved.
    kwargs
        Forwarded to ``achievable_diagonal`` (rank, n_restarts, rcond,
        restrict_rcond, max_nfev, progress...).

    Returns
    -------
    PowerSpectralDensityArray on the run's frequency lines (NaN off-band),
    plus the raw predictor result dict.
    """
    frequencies, frf, target = read_run_frf(dataset)
    valid = np.isfinite(target).all(axis=1) & (np.max(target, axis=1) > 0)
    if band is not None:
        valid &= (frequencies >= band[0]) & (frequencies <= band[1])
    line_indices = np.flatnonzero(valid)
    if decimate > 1:
        # Solving every line of a 901-line band takes minutes.  The floor is a
        # smooth function of frequency away from resonances, so solving every
        # Nth line and interpolating in dB costs very little accuracy for a
        # near-linear speedup.  Endpoints are always kept so the interpolation
        # never extrapolates.
        kept = np.unique(np.concatenate([line_indices[::decimate],
                                         line_indices[-1:]]))
        line_indices = kept
    result = achievable_diagonal(frf, target, line_indices=line_indices, **kwargs)

    if decimate > 1:
        achieved = np.array(result['achieved'], dtype=float)
        solved = np.array(result['solved'], dtype=bool)
        idx = np.flatnonzero(solved)
        fill = np.flatnonzero(valid)
        for m in range(achieved.shape[1]):
            with np.errstate(divide='ignore', invalid='ignore'):
                db = 10.0 * np.log10(np.maximum(achieved[idx, m], _TINY))
            achieved[fill, m] = 10.0 ** (
                np.interp(frequencies[fill], frequencies[idx], db) / 10.0)
        result = dict(result)
        result['achieved'] = achieved
        result['interpolated_from_n_lines'] = int(idx.size)
    return result, frequencies


def achieved_to_psd(result, frequencies, control_coordinate):
    """Wrap a predictor result's ``achieved`` array as a PowerSpectralDensityArray."""
    achieved = np.array(result['achieved'], dtype=float)     # (F, M) NaN off-band
    coord = np.tile(control_coordinate, (2, 1)).T
    return power_spectral_density_array(abscissa=frequencies,
                                        ordinate=achieved.T,
                                        coordinate=coord)


# --------------------------------------------------------------------------
# Projected-target adapter
# --------------------------------------------------------------------------
def projected_target_psd(npz_path, frequencies, control_coordinate):
    """Convert a saved projected-target .npz into a PowerSpectralDensityArray.

    The predictor writes ``target_diag`` (F, M) on its own frequency grid,
    which generally differs from the run's FFT lines (2049 vs 1025 for the
    6x12x8 case).  Interpolation is done on the dB scale, which is the space
    the target is specified and judged in; lines outside the predictor's band
    come back NaN so they are excluded from scoring rather than silently
    treated as zero-target.
    """
    data = np.load(npz_path)
    source_f = np.asarray(data['frequencies'], dtype=float)
    target = np.asarray(data['target_diag'], dtype=float)      # (F_src, M)
    if target.shape[0] != source_f.size:
        raise ValueError(f'{npz_path}: target_diag has {target.shape[0]} rows '
                         f'but {source_f.size} frequencies')
    n_control = len(control_coordinate)
    if target.shape[1] != n_control:
        raise ValueError(f'{npz_path}: target_diag has {target.shape[1]} channels '
                         f'but the run has {n_control} control channels')

    valid = np.all(np.isfinite(target), axis=1) & (np.max(target, axis=1) > 0)
    if not np.any(valid):
        raise ValueError(f'{npz_path}: no finite positive target lines')
    fs, ts = source_f[valid], target[valid]
    lo, hi = fs.min(), fs.max()

    out = np.full((frequencies.size, n_control), np.nan)
    inside = (frequencies >= lo) & (frequencies <= hi)
    for m in range(n_control):
        with np.errstate(divide='ignore'):
            db = 10.0 * np.log10(np.maximum(ts[:, m], _TINY))
        out[inside, m] = 10.0 ** (np.interp(frequencies[inside], fs, db) / 10.0)

    coord = np.tile(control_coordinate, (2, 1)).T
    return power_spectral_density_array(abscissa=frequencies,
                                        ordinate=out.T,
                                        coordinate=coord)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _compare(measured, target, tolerance_db):
    """Per-channel dB comparison of two (M, F) ordinate arrays.

    Only lines where the target is finite and strictly positive are scored, so
    out-of-band lines and the NaN padding on a projected target are excluded
    rather than counted as failures.
    """
    measured = np.real(measured)
    target = np.real(target)
    with np.errstate(invalid='ignore'):
        band = np.isfinite(target) & (target > 0) & np.isfinite(measured)

    with np.errstate(divide='ignore', invalid='ignore'):
        error_db = 10.0 * np.log10(np.maximum(measured, _TINY)
                                   / np.maximum(target, _TINY))
    error_db[~band] = np.nan

    n_lines = band.sum(axis=1)
    with np.errstate(invalid='ignore'):
        mean_db = np.nanmean(error_db, axis=1)
        rms_db = np.sqrt(np.nanmean(error_db ** 2, axis=1))
        outside = np.nansum(np.abs(error_db) > tolerance_db, axis=1)
    percent_out = 100.0 * outside / np.maximum(n_lines, 1)

    total = int(band.sum())
    flat = error_db[band]
    return dict(
        error_db=error_db,
        mean_db=mean_db,
        rms_db=rms_db,
        percent_out=percent_out,
        n_lines=n_lines,
        overall_rms_db=float(np.sqrt(np.mean(flat ** 2))) if total else np.nan,
        overall_percent_out=float(100.0 * np.sum(np.abs(flat) > tolerance_db)
                                  / total) if total else np.nan,
        tolerance_db=tolerance_db,
    )


def score_against(test, specification, tolerance_db=6.0):
    """Per-channel dB error of the measured response against a specification."""
    return _compare(test.cpsd[0].get_asd().ordinate,
                    specification.get_asd().ordinate, tolerance_db)


def reachability(projected, specification, tolerance_db=6.0):
    """How far the BEST ACHIEVABLE response sits from the specification.

    This is the reachability limit stated directly, rather than inferred from
    the difference between two scorings.  A near-zero result means the
    specification is essentially achievable with the drives available and
    every bit of the measured error belongs to the control law; a large one
    means part of the error is unreachable no matter what the law does.

    Empirically this tracks the conditioning of the identified FRF far more
    than it tracks the drive/control channel count: on this rig, runs with
    median cond(H) around 110 come back near 0.1 dB while runs in the
    thousands come back at 1.2-1.6 dB.
    """
    return _compare(projected.get_asd().ordinate,
                    specification.get_asd().ordinate, tolerance_db)


def _fmt_table(label, coords, stats):
    lines = [f'  {label}',
             f'    {"channel":10s} {"lines":>6s} {"mean dB":>9s} {"rms dB":>8s} '
             f'{"% out":>7s}']
    for i, c in enumerate(coords):
        lines.append(f'    {str(c):10s} {stats["n_lines"][i]:6d} '
                     f'{stats["mean_db"][i]:9.2f} {stats["rms_db"][i]:8.2f} '
                     f'{stats["percent_out"][i]:7.1f}')
    lines.append(f'    {"OVERALL":10s} {"":6s} {"":9s} '
                 f'{stats["overall_rms_db"]:8.2f} {stats["overall_percent_out"]:7.1f}')
    return '\n'.join(lines)


def score_run(nc4_path, projected_npz=None, tolerance_db=6.0,
              figure_dir=None, verbose=True, recompute_floor=False,
              floor_band=(100.0, 1000.0), floor_kwargs=None, save_dir=None):
    """Score one run against the flat spec and against what is achievable.

    The achievable target comes either from a saved projected-target .npz
    (``projected_npz``) or, preferably, from this run's OWN measured FRF
    (``recompute_floor=True``).  The saved file is only valid for runs from
    the same system identification; the rig's damping has changed measurably
    between campaigns, so a stale target quietly biases the comparison.
    """
    name = os.path.splitext(os.path.basename(nc4_path))[0]
    try:
        dataset = nc4.Dataset(nc4_path, 'r')
    except OSError as exc:
        return dict(name=name, path=nc4_path, skipped=f'cannot open ({exc})')
    try:
        return _score_open_run(dataset, nc4_path, name, projected_npz,
                               tolerance_db, figure_dir, verbose,
                               recompute_floor, floor_band, floor_kwargs,
                               save_dir)
    finally:
        dataset.close()


def _score_open_run(dataset, nc4_path, name, projected_npz, tolerance_db,
                    figure_dir, verbose, recompute_floor, floor_band,
                    floor_kwargs, save_dir=None):
    """Body of score_run, with the file's single Dataset handle supplied."""
    scorable, reason = describe_dataset(dataset)
    if not scorable:
        return dict(name=name, path=nc4_path, skipped=reason)

    test, extras = load_spectral_run(dataset)
    coords = extras['control_coordinate']

    flat_spec = test.specification_cpsd
    result = dict(name=name, path=nc4_path, coordinate=coords,
                  n_drives=extras['n_drives'], n_control=len(coords),
                  environment=extras['environment'],
                  bands_from_file=extras['bands_from_file'],
                  median_cond_frf=(float(np.median(extras['frf_condition']))
                                   if extras['frf_condition'].size else float('nan')),
                  provenance=extras['provenance'])
    result['contractual'] = score_against(test, flat_spec, tolerance_db)

    projected = None
    if recompute_floor:
        achievable_diagonal = _load_achievable_diagonal()
        floor_result, floor_frequencies = achievable_floor_from_run(
            dataset, achievable_diagonal, band=floor_band, **(floor_kwargs or {}))
        n_solved = floor_result.get('interpolated_from_n_lines')
        projected = achieved_to_psd(floor_result, floor_frequencies, coords)
        result['floor_result'] = floor_result
        result['floor_source'] = (
            "recomputed from this run's FRF"
            + (f', {n_solved} lines solved + interpolated' if n_solved else ''))
    elif projected_npz is not None:
        projected = projected_target_psd(projected_npz, extras['frequencies'], coords)
        result['floor_source'] = f'saved file {os.path.basename(projected_npz)}'
    if projected is not None:
        result['physical'] = score_against(test, projected, tolerance_db)
        result['reachability'] = reachability(projected, flat_spec, tolerance_db)
        result['projected_psd'] = projected

    if verbose:
        print(f'\n=== {name}')
        print(f'  environment {extras["environment"]!r}, '
              f'{len(coords)} control channels, {extras["n_drives"]} drives, '
              f'tolerance +/-{tolerance_db:g} dB, '
              f'median cond(H) {result["median_cond_frf"]:.0f}')
        prov = extras['provenance']
        print(f'  plant       {prov.get("plant", "?")}  '
              f'(hardware {prov.get("hardware", "?")})')
        print(f'  law         {prov.get("control_python_function", "?")}  '
              f'[{prov["parameters_shown"]}]')
        print(f'  averaging   control {prov["control_averaging_type"]} '
              f'coef {prov["control_averaging_coefficient"]:g}, '
              f'{prov.get("frames_in_cpsd", "?")} frames, '
              f'FRF update {"on" if prov.get("update_tf_during_control") else "off"}')
        if not extras['bands_from_file']:
            print('  note: profile carries no warning/abort levels (all NaN); '
                  'tolerance bands synthesized from the specification')
        print(_fmt_table('vs FLAT SPEC (contractual)', coords, result['contractual']))
        if 'physical' in result:
            print(_fmt_table(f'vs ACHIEVABLE TARGET ({result["floor_source"]})',
                             coords, result['physical']))
            print(_fmt_table('REACHABILITY: best achievable vs flat spec',
                             coords, result['reachability']))
            reach = result['reachability']['overall_rms_db']
            if reach < 0.5:
                verdict = ('the flat spec is essentially achievable here, so '
                           'nearly all of the error above belongs to the law')
            elif reach < 1.5:
                verdict = 'a modest part of the error above is unreachable'
            else:
                verdict = ('a substantial part of the error above is '
                           'unreachable regardless of the law')
            print(f'    best achievable is {reach:.2f} dB rms from the flat '
                  f'spec -- {verdict}')
            gap = (result['contractual']['overall_rms_db']
                   - result['physical']['overall_rms_db'])
            print(f'    gap (flat - projected) = {gap:.2f} dB  '
                  f'<- the part of the error that is unreachable, not the law')

    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
        # Figures for the contractual view: flat spec with synthesized bands.
        test.specification_cpsd = flat_spec
        test.specification_warning_psd = bands_from_spec(flat_spec, tolerance_db / 2.0)
        test.specification_abort_psd = bands_from_spec(flat_spec, tolerance_db)
        _save_figures(test, figure_dir, f'{name}_vs_flat')
        # Same measured data, judged against what is actually achievable.
        if projected is not None:
            test.specification_cpsd = projected
            test.specification_warning_psd = bands_from_spec(projected, tolerance_db / 2.0)
            test.specification_abort_psd = bands_from_spec(projected, tolerance_db)
            _save_figures(test, figure_dir, f'{name}_vs_projected')
            test.specification_cpsd = flat_spec

    if save_dir is not None:
        path = save_result(result, test, extras, save_dir)
        if verbose:
            print(f'    saved {os.path.relpath(path)}')
    return result


def save_result(result, test, extras, save_dir):
    """Write one run's scoring to <save_dir>/<run>_scoring.npz.

    A --recompute-floor sweep costs ~25 minutes for this campaign, and the
    interesting questions tend to arrive afterwards ("does the gap track FRF
    conditioning?").  Everything needed to answer them without recomputing is
    written here: the measured, specification and achievable spectra, all
    three scorings per channel, the FRF, and the predictor's own diagnostics.

    Loads back with np.load(path, allow_pickle=False); string arrays come back
    as numpy unicode arrays.
    """
    os.makedirs(save_dir, exist_ok=True)
    out = {
        'frequencies': extras['frequencies'],
        'control_coordinate': np.array([str(c) for c in result['coordinate']]),
        'n_drives': np.array(result['n_drives']),
        'n_control': np.array(result['n_control']),
        'environment': np.array(result['environment']),
        'tolerance_db': np.array(result['contractual']['tolerance_db']),
        'bands_from_file': np.array(result['bands_from_file']),
        'floor_source': np.array(result.get('floor_source', '')),
        'measured_asd': np.real(test.cpsd[0].get_asd().ordinate),
        'spec_asd': np.real(test.specification_cpsd.get_asd().ordinate),
        'frf': extras['frf'],
        'frf_condition': extras['frf_condition'],
    }
    for key, value in result.get('provenance', {}).items():
        out[f'prov__{key}'] = np.array(value)
    for scoring in ('contractual', 'physical', 'reachability'):
        stats = result.get(scoring)
        if stats is None:
            continue
        for key in ('error_db', 'mean_db', 'rms_db', 'percent_out', 'n_lines',
                    'overall_rms_db', 'overall_percent_out'):
            out[f'{scoring}__{key}'] = np.asarray(stats[key])
    if 'projected_psd' in result:
        out['achievable_asd'] = np.real(result['projected_psd'].get_asd().ordinate)
    floor = result.get('floor_result')
    if floor is not None:
        for key in ('best_rms_db', 'best_minimax_db', 'drive_trace',
                    'exactly_achievable', 'solved'):
            if key in floor:
                out[f'floor__{key}'] = np.asarray(floor[key])
    path = os.path.join(save_dir, f'{result["name"]}_scoring.npz')
    np.savez_compressed(path, **out)
    return path


def write_summary_csv(results, save_dir):
    """One row per run: the three scorings side by side, for sorting in a sheet."""
    import csv
    path = os.path.join(save_dir, 'summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run', 'plant', 'law', 'parameters',
                         'control_averaging_type', 'control_averaging_coefficient',
                         'frames_in_cpsd', 'sample_rate', 'update_tf_during_control',
                         'n_control', 'n_drives', 'median_cond_frf',
                         'flat_rms_db', 'flat_percent_out',
                         'achievable_rms_db', 'achievable_percent_out',
                         'reachability_rms_db', 'reachability_percent_out',
                         'floor_source'])
        for r in results:
            prov = r.get('provenance', {})
            row = [r['name'], prov.get('plant', ''),
                   prov.get('control_python_function', ''),
                   prov.get('control_python_function_parameters', ''),
                   prov.get('control_averaging_type', ''),
                   prov.get('control_averaging_coefficient', ''),
                   prov.get('frames_in_cpsd', ''), prov.get('sample_rate', ''),
                   prov.get('update_tf_during_control', ''),
                   r['n_control'], r['n_drives'],
                   f'{r.get("median_cond_frf", float("nan")):.1f}',
                   f'{r["contractual"]["overall_rms_db"]:.3f}',
                   f'{r["contractual"]["overall_percent_out"]:.2f}']
            for key in ('physical', 'reachability'):
                if key in r:
                    row += [f'{r[key]["overall_rms_db"]:.3f}',
                            f'{r[key]["overall_percent_out"]:.2f}']
                else:
                    row += ['', '']
            row.append(r.get('floor_source', ''))
            writer.writerow(row)
    return path


def _save_figures(test, figure_dir, name):
    """Emit the subset of the standard suite that works without time data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def _flatten(obj):
        """Some plot methods return one figure per data set; the paginating
        ones (control_vs_spec, response) return a LIST of figures per data
        set, so the return value is nested to arbitrary depth."""
        if obj is None:
            return
        if isinstance(obj, plt.Figure):
            yield obj
            return
        if isinstance(obj, (list, tuple, np.ndarray)):
            for item in obj:
                yield from _flatten(item)

    for method, stem in [('plot_control_vs_spec', 'control_vs_spec'),
                         ('plot_percent_lines_out', 'percent_lines_out'),
                         ('plot_rms_error', 'rms_error'),
                         ('plot_rms_level', 'rms_level'),
                         ('plot_response', 'response')]:
        try:
            figures, _ = getattr(test, method)()
            collected = list(_flatten(figures))
            for i, figure in enumerate(collected):
                suffix = '' if len(collected) == 1 else f'_p{i + 1}'
                figure.savefig(os.path.join(figure_dir, f'{name}_{stem}{suffix}.png'),
                               dpi=150, bbox_inches='tight')
                plt.close(figure)
        except Exception as exc:                      # noqa: BLE001
            print(f'    [skip] {method}: {type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='Rattlesnake .nc4 run files')
    parser.add_argument('--projected', default=None,
                        help='projected-target .npz from the achievable-response predictor')
    parser.add_argument('--tolerance-db', type=float, default=6.0)
    parser.add_argument('--recompute-floor', action='store_true',
                        help="compute the achievable floor from each run's own "
                             "measured FRF instead of a saved projected target "
                             "(slower, but exact for that run's system state)")
    parser.add_argument('--floor-band', type=float, nargs=2, default=(100.0, 1000.0),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--restrict-rcond', type=float, default=None,
                        help='drive-subspace truncation passed to the predictor')
    parser.add_argument('--n-restarts', type=int, default=2)
    parser.add_argument('--floor-decimate', type=int, default=1,
                        help='solve every Nth line of the floor and interpolate '
                             'the rest in dB (N=4 keeps the whole run near a '
                             'minute; N=1 solves every line)')
    parser.add_argument('--figures', default=None, help='directory for figures')
    parser.add_argument('--save', default=None, metavar='DIR',
                        help='write <run>_scoring.npz per run plus summary.csv, '
                             'so a long sweep can be re-analysed without '
                             'recomputing the achievable floor')
    args = parser.parse_args()

    floor_kwargs = dict(n_restarts=args.n_restarts, decimate=args.floor_decimate)
    if args.restrict_rcond is not None:
        floor_kwargs['restrict_rcond'] = args.restrict_rcond
    results = []
    for run in args.runs:
        try:
            outcome = score_run(run, args.projected, args.tolerance_db,
                                args.figures,
                                recompute_floor=args.recompute_floor,
                                floor_band=tuple(args.floor_band),
                                floor_kwargs=floor_kwargs,
                                save_dir=args.save)
        except Exception as exc:                       # noqa: BLE001
            print(f'\n=== {os.path.basename(run)}\n'
                  f'  [failed] {type(exc).__name__}: {exc}')
            continue
        if outcome.get('skipped'):
            print(f'\n=== {os.path.basename(run)}\n'
                  f'  [skipped] {outcome["skipped"]}')
            continue
        results.append(outcome)

    if not results:
        print('\nNo scorable runs.')
        return results
    if len(results) > 1:
        has_floor = 'physical' in results[0]
        print('\n=== SUMMARY (overall rms dB / percent lines out) ===')
        print(f'{"run":30s} {"plant":30s} {"law":26s} {"parameters":20s} {"avg":>16s}')
        for r in results:
            prov = r.get('provenance', {})
            avg = (f'{prov.get("control_averaging_type", "?")} '
                   f'{prov.get("control_averaging_coefficient", 0.0):g}')
            print(f'{r["name"][:30]:30s} {str(prov.get("plant_short", "?")):30s} '
                  f'{str(prov.get("control_python_function", "?"))[:26]:26s} '
                  f'{prov.get("parameters_shown", "?")[:20]:20s} {avg:>16s}')
        print()
        header = f'{"run":42s} {"cond(H)":>9s} {"flat":>14s}'
        if has_floor:
            header += f' {"achievable":>14s} {"unreachable":>14s}'
        print(header)
        for r in results:
            row = (f'{r["name"][:42]:42s} '
                   f'{r.get("median_cond_frf", float("nan")):9.0f} '
                   f'{r["contractual"]["overall_rms_db"]:7.2f} '
                   f'{r["contractual"]["overall_percent_out"]:6.1f}')
            if 'physical' in r:
                row += (f'  {r["physical"]["overall_rms_db"]:7.2f} '
                        f'{r["physical"]["overall_percent_out"]:6.1f}')
                row += (f'  {r["reachability"]["overall_rms_db"]:7.2f} '
                        f'{r["reachability"]["overall_percent_out"]:6.1f}')
            print(row)
        if has_floor:
            print('\n  "unreachable" is how far the BEST ACHIEVABLE response sits\n'
                  '  from the flat spec.  Near zero means the spec is reachable and\n'
                  '  the error is the law\'s; large means part of it is not.')

        disagreements = compare_provenance(results)
        if disagreements:
            print('\n  *** THESE RUNS ARE NOT DIRECTLY COMPARABLE ***')
            print('  The following were not held constant across the set:')
            for key, values in disagreements.items():
                print(f'    {key}:')
                for value, names in values.items():
                    if key == 'hardware_file':
                        value = os.path.basename(value)
                    shown = ', '.join(n[:30] for n in names[:4])
                    more = f' (+{len(names) - 4} more)' if len(names) > 4 else ''
                    print(f'      {value}  <- {shown}{more}')
            print('  Differences between these runs reflect the change above at '
                  'least as much as\n  the control law.  Compare only within a '
                  'group that shares all of them.')
        else:
            print('\n  provenance check: all runs share plant, sample rate, '
                  'averaging and\n  spectral settings -- differences are '
                  'attributable to the law and its parameters.')

    if args.save:
        path = write_summary_csv(results, args.save)
        print(f'\nwrote {path}')
    return results


if __name__ == '__main__':
    main()
