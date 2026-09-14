"""Rebuild report_data.json for the 12-run 6-12-8 comparison.

Every quantity is recomputed here from the run files with one documented
method, so the table is internally consistent.  The achievable-floor columns
SUPERSEDE the values in the 2026-09-12 report: those could not be reproduced
(run01 recomputes to ~2.2 dB against a reported 3.87), the settings that
produced them were not recorded, and a column whose provenance is unknown is
worth less than one that is merely different.

TWO floors are computed for every run, both from that run's OWN control FRF,
every 5th in-band line (100-1000 Hz), log-interpolated between solved lines:

  reach_*        restrict_rcond = 1e-3.  The OPERATIONAL floor: the best any
                 of these laws could do while inverting at the rcond they all
                 actually use.  This reproduces the 2026-09-12 report's
                 numbers (run01 3.91 here vs 3.87 there, the difference being
                 decimation 5 vs 4), so the column is continuous with it.
  rank_*         no rcond restriction, but computed on the measured FRF
                 TRUNCATED to its physically real rank.  The plant is rank 5,
                 not 6: computed analytically from M, K and C the sixth
                 singular value is ~1e-17, and in the measured FRF it comes
                 back at ~3e-4 of the first -- identification noise in a
                 direction the six shakers cannot actually reach.

An unrestricted floor on the raw measured FRF returns ~2.3 dB, but that number
is the optimizer exploiting that noise direction and should not be quoted: it
was reported as a real 1.6 dB headroom in the first draft of the 2026-09-14
report and is wrong.  Truncated to rank 5 the same computation gives 3.97 dB,
and the analytic rank-5 plant gives 4.07 dB, both consistent with the
operational floor.  rcond = 1e-3 is therefore doing the right thing -- it
rejects a direction that does not exist -- and lowering it would invert noise.

Decimation was checked against every-2nd and every-20th line on run01 and
moves the result by 0.05 dB or less.
"""
import importlib.util, json, os, sys, time
import numpy as np
import netCDF4

np.seterr(all='ignore')
BAND = (100.0, 1000.0)
DECIMATE = 5
RESTRICT_RCOND = 1e-3
RUNS = '../results/runs'
CAPPOS = {'match_trace_pseudoinverse': 1, 'match_diagonal_congruence': 1,
          'pseudoinverse_control': 1, 'buzz_control': 1,
          'optimal_diagonal_control': 4, 'optimal_diagonal_control_fast': 4}
SHORT = {'match_trace_pseudoinverse': 'match_trace',
         'optimal_diagonal_control': 'optimal_diagonal',
         'optimal_diagonal_control_fast': 'optimal_diagonal_fast',
         'match_diagonal_congruence': 'congruence',
         'pseudoinverse_control': 'pseudoinverse', 'buzz_control': 'buzz'}
OPEN_LOOP = {'pseudoinverse_control', 'buzz_control'}

_s = importlib.util.spec_from_file_location(
    'ar', '../../../control_laws/achievable_response.py')
ar = importlib.util.module_from_spec(_s); _s.loader.exec_module(ar)


def stats(measured, target, tol=6.0):
    band = np.isfinite(target) & (target > 0) & np.isfinite(measured)
    e = 10.0 * np.log10(np.maximum(measured, 1e-300) / np.maximum(target, 1e-300))
    e[~band] = np.nan
    return dict(rms=float(np.sqrt(np.nanmean(e ** 2))),
                pout=float(100.0 * np.nansum(np.abs(e) > tol) / max(band.sum(), 1)),
                mean_db=[float(v) for v in np.nanmean(e, axis=0)])


def one(path):
    ds = netCDF4.Dataset(path)
    g = ds.groups['Frame 6x12 Random']
    f = np.array(g['specification_frequency_lines'][:], float)
    H = np.array(g['frf_data_real'][:]) + 1j * np.array(g['frf_data_imag'][:])
    S = np.array(g['specification_cpsd_matrix_real'][:], float)
    R = np.array(g['response_cpsd_real'][:], float)
    D = np.array(g['drive_cpsd_real'][:], float)
    coh = np.array(g['frf_coherence'][:], float)
    law = g.getncattr('control_python_function')
    par = [p.strip() for p in
           str(g.getncattr('control_python_function_parameters')).split(',')]
    stream = os.path.getsize(path.replace('_spec.nc4', '_stream.nc4')) \
        if os.path.exists(path.replace('_spec.nc4', '_stream.nc4')) else 0
    ds.close()

    tgt = np.einsum('fmm->fm', S)
    resp = np.einsum('fmm->fm', R)
    inb = (np.isfinite(tgt).all(axis=1) & (tgt.max(axis=1) > 0)
           & (f >= BAND[0]) & (f <= BAND[1]))
    idx = np.flatnonzero(inb)

    # --- achievable floor from this run's own FRF ---
    li = np.unique(np.concatenate([idx[::DECIMATE], idx[-1:]]))

    def floor(**kw):
        res = ar.achievable_diagonal(H, tgt, line_indices=li, **kw)
        a = np.array(res['achieved'], float)
        solved = np.flatnonzero(np.array(res['solved'], bool))
        for m in range(a.shape[1]):
            db = 10.0 * np.log10(np.maximum(a[solved, m], 1e-300))
            a[idx, m] = 10.0 ** (np.interp(f[idx], f[solved], db) / 10.0)
        return a

    ach = floor(restrict_rcond=RESTRICT_RCOND)   # operational floor

    # Physically real rank, and the floor over that subspace only.
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    ratio = sv / np.maximum(sv[:, :1], 1e-300)
    # In-band only: out-of-band lines carry a much noisier sixth direction and
    # would wrongly report the plant as full rank.
    numeric_rank = int(np.median((ratio[inb] > 1e-3).sum(axis=1)))
    sv_trunc = sv.copy()
    sv_trunc[:, numeric_rank:] = 0.0
    H_saved = H
    H = (U * sv_trunc[:, None, :]) @ Vh
    ideal = floor()
    H = H_saved

    flat = stats(resp[inb], tgt[inb])
    reach = stats(ach[inb], tgt[inb])          # floor vs specification
    against = stats(resp[inb], ach[inb])       # run vs floor
    ideal_reach = stats(ideal[inb], tgt[inb])   # rank-truncated floor

    cap = float(par[CAPPOS[law]])
    return dict(
        run=os.path.basename(path)[:5], law=SHORT[law], law_full=law,
        loop='open' if law in OPEN_LOOP else 'feedback',
        cap='off' if cap >= 1.0 else f'{cap:g}', params=','.join(par),
        rms=flat['rms'], pout=flat['pout'], mean_db=flat['mean_db'],
        reach_rms=reach['rms'], reach_pout=reach['pout'],
        reach_mean_db=reach['mean_db'],
        ach_rms=against['rms'], ach_pout=against['pout'],
        rank_rms=ideal_reach['rms'], rank_pout=ideal_reach['pout'],
        numeric_rank=numeric_rank,
        sv6_ratio=float(np.median(ratio[inb, -1])),
        drive_trace=float(np.einsum('fnn->f', D)[inb].mean()),
        level=float(10 * np.log10(resp[inb].sum() / tgt[inb].sum())),
        cond=float(np.median(np.linalg.cond(H[inb]))),
        min_mult_coh=float(coh[inb].min()),
        stream_bytes=int(stream))


CACHE = '.floor_cache'


def main():
    """Each per-run result is cached, so this can be run in batches that fit a
    short shell timeout and re-run safely without recomputing finished runs.
    With no arguments it assembles report_data.json from the cache."""
    os.makedirs(CACHE, exist_ok=True)
    paths = sorted(f'{RUNS}/{n}' for n in os.listdir(RUNS)
                   if n.startswith('run') and n.endswith('_spec.nc4'))
    wanted = sys.argv[1:]
    if wanted != ['assemble']:
        for p in paths:
            run = os.path.basename(p)[:5]
            if wanted and run not in wanted:
                continue
            cached = os.path.join(CACHE, run + '.json')
            if os.path.exists(cached):
                print(f'  {run} cached', flush=True)
                continue
            t = time.time()
            r = one(p)
            json.dump(r, open(cached, 'w'), indent=1)
            print(f"  {run} {r['law']:22s} rms {r['rms']:5.2f} "
                  f"floor {r['reach_rms']:5.2f} vs-floor {r['ach_rms']:5.2f} "
                  f"({time.time()-t:.0f}s)", flush=True)
    have = [os.path.basename(p)[:5] for p in paths
            if os.path.exists(os.path.join(CACHE, os.path.basename(p)[:5] + '.json'))]
    out = [json.load(open(os.path.join(CACHE, r + '.json'))) for r in have]
    if len(out) == len(paths):
        json.dump(out, open('report_data.json', 'w'), indent=1)
        print(f'wrote report_data.json with {len(out)} runs')
    else:
        print(f'{len(out)}/{len(paths)} runs cached -- '
              f'missing {sorted(set(os.path.basename(p)[:5] for p in paths) - set(have))}')


if __name__ == '__main__':
    main()
