"""Generate a Rattlesnake specification .mat whose target is the BEST
ACHIEVABLE response rather than a flat spec.

Why
---
With 6 drives and 8 control channels the flat spec is not reachable, and the
unreachable share measured on this rig is 2.4-3.9 dB rms -- larger than the
differences between the control laws being compared.  Ranking laws against
the flat spec therefore mostly measures how each one fails at an impossible
task.  Pointing every law at the achievable diagonal instead removes
reachability from the comparison, so the residual is tracking and convergence
behaviour alone.

What it writes
--------------
The same structure as the existing flat spec: ``f`` (1, F) and ``cpsd``
(M, M, F) complex128.  The output is DIAGONAL, matching the flat spec's own
structure (verified: its off-diagonal is identically zero).  Off-band lines
stay exactly zero, so Rattlesnake's in-band detection is unchanged.

The achievable diagonal is reachable; the full matrix formed by pairing it
with zero cross-terms is not, in the sense that no drive CPSD produces
exactly that matrix including the zeros.  That is deliberate -- it matches
how real specifications are written, and the scored quantity is the diagonal.
A feedback law correcting on measured per-channel error converges to it; a
law whose cross-term structure is fixed open-loop by one startup solve may
not, which is a result about that law rather than a defect here.

Ordering
--------
The FRF must come from a system identification performed on the SAME plant
the runs will use, and every run being compared must share ONE generated
target.  Regenerating between runs silently destroys comparability.

Usage
-----
    python make_achievable_spec.py SYSID_OR_RUN.nc4 \
        --source-spec ../results/case/flat_spec_frame6x12.mat \
        --output ../results/case/achievable_spec_frame6x12.mat \
        [--band 100 1000] [--restrict-rcond 1e-3] [--decimate 4]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import netCDF4 as nc4
import scipy.io as sio

_TINY = np.finfo(float).tiny


def _load_achievable_diagonal():
    """Import achievable_diagonal from the repository's control_laws folder."""
    try:
        from achievable_response import achievable_diagonal
        return achievable_diagonal
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, '..', '..', '..', 'control_laws', 'achievable_response.py'))
    if not os.path.exists(candidate):
        raise ImportError(f'could not locate achievable_response.py at {candidate}')
    import importlib.util
    spec = importlib.util.spec_from_file_location('_achievable_response', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.achievable_diagonal


def read_frf_and_target(nc4_path):
    """Return (frequencies, H, target_diag) from a system-ID or control run."""
    dataset = nc4.Dataset(nc4_path, 'r')
    try:
        environment = [g for g in dataset.groups if g != 'channels'][0]
        group = dataset[environment]
        for required in ('frf_data_real', 'frf_data_imag',
                         'specification_cpsd_matrix_real',
                         'specification_frequency_lines'):
            if required not in group.variables:
                raise ValueError(f'{nc4_path}: environment {environment!r} has no {required}')
        frequencies = np.array(group['specification_frequency_lines'][:], dtype=float)
        frf = (np.array(group['frf_data_real'][:], dtype=float)
               + 1j * np.array(group['frf_data_imag'][:], dtype=float))
        spec = np.array(group['specification_cpsd_matrix_real'][:], dtype=float)
        target = np.einsum('fmm->fm', spec)
    finally:
        dataset.close()
    return frequencies, frf, target


def build_achievable_spec(nc4_path, source_spec_path, band=(100.0, 1000.0),
                          decimate=1, progress=True, **floor_kwargs):
    """Compute the achievable diagonal and place it on the source spec's grid.

    Returns (f, cpsd, report) ready for scipy.io.savemat.
    """
    source = sio.loadmat(source_spec_path)
    if 'f' not in source or 'cpsd' not in source:
        raise ValueError(f"{source_spec_path}: expected 'f' and 'cpsd' variables")
    f_out = np.asarray(source['f'], dtype=float).reshape(-1)          # (F_out,)
    cpsd_in = np.asarray(source['cpsd'])                              # (M, M, F_out)
    n_control = cpsd_in.shape[0]

    # The band the source spec actually specifies.  Regenerating only where the
    # original was nonzero keeps Rattlesnake's in-band detection identical.
    source_diag = np.einsum('mmf->fm', cpsd_in).real                  # (F_out, M)
    specified = (source_diag > 0).any(axis=1)
    if not np.any(specified):
        raise ValueError(f'{source_spec_path}: specification is zero everywhere')

    frequencies, frf, target = read_frf_and_target(nc4_path)
    if frf.shape[1] != n_control:
        raise ValueError(f'FRF has {frf.shape[1]} control channels, source spec has {n_control}')

    valid = np.isfinite(target).all(axis=1) & (target > 0).any(axis=1)
    valid &= (frequencies >= band[0]) & (frequencies <= band[1])
    line_indices = np.flatnonzero(valid)
    if line_indices.size == 0:
        raise ValueError('no lines to solve in the requested band')
    if decimate > 1:
        line_indices = np.unique(np.concatenate([line_indices[::decimate],
                                                 line_indices[-1:]]))

    achievable_diagonal = _load_achievable_diagonal()
    if progress:
        print(f'  solving {line_indices.size} lines '
              f'({band[0]:g}-{band[1]:g} Hz, decimate={decimate})...', flush=True)
    result = achievable_diagonal(frf, target, line_indices=line_indices,
                                 **floor_kwargs)

    achieved = np.asarray(result['achieved'], dtype=float)            # (F_src, M)
    solved = np.asarray(result['solved'], dtype=bool)
    solved_idx = np.flatnonzero(solved)
    if solved_idx.size == 0:
        raise ValueError('the predictor solved no lines')

    # Interpolate in dB onto the source spec's own frequency grid, over exactly
    # the lines the source specified.  dB is the space the target is written and
    # judged in, so interpolating there avoids biasing deep notches.
    out_diag = np.zeros((f_out.size, n_control), dtype=float)
    fill = np.flatnonzero(specified)
    for m in range(n_control):
        with np.errstate(divide='ignore'):
            db = 10.0 * np.log10(np.maximum(achieved[solved_idx, m], _TINY))
        out_diag[fill, m] = 10.0 ** (
            np.interp(f_out[fill], frequencies[solved_idx], db) / 10.0)

    cpsd_out = np.zeros((n_control, n_control, f_out.size), dtype=complex)
    for m in range(n_control):
        cpsd_out[m, m, :] = out_diag[:, m]

    with np.errstate(divide='ignore', invalid='ignore'):
        change_db = 10.0 * np.log10(np.maximum(out_diag[fill], _TINY)
                                    / np.maximum(source_diag[fill], _TINY))
    report = dict(
        n_lines_solved=int(solved_idx.size),
        n_lines_written=int(fill.size),
        band_hz=(float(f_out[fill].min()), float(f_out[fill].max())),
        per_channel_mean_db=change_db.mean(axis=0),
        per_channel_min_db=change_db.min(axis=0),
        per_channel_max_db=change_db.max(axis=0),
        overall_rms_db=float(np.sqrt(np.mean(change_db ** 2))),
        source_spec=source_spec_path,
        frf_source=nc4_path,
    )
    return f_out.reshape(1, -1), cpsd_out, report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frf_source', help='system-ID or control run .nc4 carrying the FRF')
    parser.add_argument('--source-spec', required=True,
                        help='existing spec .mat supplying the frequency grid and band')
    parser.add_argument('--output', required=True, help='spec .mat to write')
    parser.add_argument('--band', type=float, nargs=2, default=(100.0, 1000.0),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--restrict-rcond', type=float, default=None)
    parser.add_argument('--n-restarts', type=int, default=2)
    parser.add_argument('--decimate', type=int, default=4)
    parser.add_argument('--force', action='store_true',
                        help='overwrite the output if it already exists')
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f'{args.output} exists; pass --force to overwrite.  '
                         'Regenerating a target midway through a comparison '
                         'destroys comparability between runs.')

    floor_kwargs = dict(n_restarts=args.n_restarts)
    if args.restrict_rcond is not None:
        floor_kwargs['restrict_rcond'] = args.restrict_rcond

    f, cpsd, report = build_achievable_spec(
        args.frf_source, args.source_spec, band=tuple(args.band),
        decimate=args.decimate, **floor_kwargs)

    sio.savemat(args.output, {'f': f, 'cpsd': cpsd})

    print(f'\nwrote {args.output}')
    print(f'  FRF from      {report["frf_source"]}')
    print(f'  grid from     {report["source_spec"]}')
    print(f'  band          {report["band_hz"][0]:.0f}-{report["band_hz"][1]:.0f} Hz, '
          f'{report["n_lines_written"]} lines written, '
          f'{report["n_lines_solved"]} solved')
    print('\n  change from the source spec, per channel (dB):')
    print(f'    {"ch":>4s} {"mean":>8s} {"min":>8s} {"max":>8s}')
    for i in range(report['per_channel_mean_db'].size):
        print(f'    {i + 1:4d} {report["per_channel_mean_db"][i]:8.2f} '
              f'{report["per_channel_min_db"][i]:8.2f} '
              f'{report["per_channel_max_db"][i]:8.2f}')
    print(f'    overall rms change {report["overall_rms_db"]:.2f} dB')
    print('\n  Every run being compared must use THIS file.  Regenerating it '
          'between runs\n  silently destroys comparability.')


if __name__ == '__main__':
    main()
