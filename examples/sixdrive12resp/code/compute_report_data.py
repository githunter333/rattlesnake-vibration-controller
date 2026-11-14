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

A THIRD floor was added 2026-11-14, budget_*, which is the one to quote
when asking what a law is leaving on the table:

  budget_*       solved under a ridge on trace(X) whose weight is bisected
                 until the floor spends the SAME band-mean drive the run
                 did.  reach_* and rank_* may spend anything, and do -- the
                 rank-5 floor wants about +26 dB more drive than the laws
                 use, so the headroom they report is not headroom a real
                 controller could claim.  budget_ach_rms is the run measured
                 against this floor.  budget_match_db records how closely
                 the floor's drive ended up matching the law's, both as
                 means over every in-band line; budget_coarse_db is the
                 bisection's own residual on its coarse line set, and
                 budget_w the multiplier it used.

REVISED 2026-11-14 (same day, after the first 52-run pass).  Two things were
wrong and both are fixed above.  The budget solves ran on raw H with the
default rcond, while reach_* ran with the per-run rank cutoff, so on 9 of 51
runs the budget floor came out BELOW the free floor -- impossible for a
constrained optimum, and caused by the optimizer reaching back through the
fictitious sixth singular direction once a loose budget made w small.  Both
budget solves now take the same restrict_rcond as the operational floor,
which makes budget_rms an upper bound on reach_rms by construction.  And
budget_match_db used to be the bisection's residual on its own 40x-decimated
line set; that was self-consistent but it is not what the number looks like
it says, and on a peaky drive spectrum the coarse mean overstates the law's
drive by up to 10 dB.  The honest match is now reported and the coarse
residual kept beside it as budget_coarse_db.

Two things worth knowing about it.  The budget is the BAND MEAN, so the
floor may move drive between frequencies freely; that is deliberately
generous, making budget_rms a lower bound on the unavoidable error at that
power.  And a drive budget makes the fictitious sixth singular direction
self-penalising -- it costs enormous drive for almost no response -- so
budget_* computed on the raw measured FRF and on the rank-truncated one
agree to 0.02 dB once the budget binds, where the unconstrained floors
differ by 1.7 dB.  The budget column is therefore a check on the rank test
below as well as a result in its own right.

REVISED 2026-11-13: the 1e-3 thresholds above are gone, replaced by the gap
test in plant_rank().  Everything this docstring says remains true OF THE
SEPTEMBER RUNS -- there sigma_6/sigma_1 is 1e-4 to 4e-4, a fixed 1e-3 sits
cleanly between the real and the noise directions, and the new test agrees
with the old one on 34 of those 38 runs.  What it got wrong was treating
1e-3 as a property of the method rather than of that particular, clean,
LINEAR measurement.  Identification noise rises with test level and with
nonlinearity: on the November 6x8 softened runs sigma_6/sigma_1 reaches
2e-3 at 0 dB and 2e-2 at strength 22, crosses the threshold, and both floor
columns then describe the optimizer reaching through a direction the six
shakers do not have.  reach_rms fell to 0.20 dB on one of those runs.  The
paragraph above anticipated this failure for the unrestricted floor and did
not notice that the operational floor shares the threshold and so shares
the failure.  The gap test has no constant in it and returns rank 5 on 51
of the 52 runs on disk.
"""
import importlib.util, json, os, sys, time
import numpy as np
import netCDF4

np.seterr(all='ignore')
BAND = (100.0, 1000.0)
DECIMATE = 5
BUDGET_DECIMATE = 40      # coarse line set for the drive-budget bisection
BUDGET_ITERS = 8          # log-w bisection steps
BUDGET_TOL_DB = 0.3       # close enough on the drive trace to stop early
RESTRICT_RCOND = 1e-3     # fallback only; see plant_rank()
RUNS = '../results/runs'
CAPPOS = {'match_trace_pseudoinverse': 1, 'match_diagonal_congruence': 1,
          'pseudoinverse_control': 1, 'buzz_control': 1,
          'optimal_diagonal_control': 4, 'optimal_diagonal_control_fast': 4}
SHORT = {'match_trace_pseudoinverse': 'match_trace',
         'optimal_diagonal_control': 'optimal_diagonal',
         'optimal_diagonal_control_fast': 'optimal_diagonal_fast',
         'match_diagonal_congruence': 'congruence',
         'pseudoinverse_control': 'pseudoinverse', 'buzz_control': 'buzz'}
# WHAT EACH LAW CLOSES ON.  This has now been got wrong twice in opposite
# directions, so it is spelled out rather than expressed as one boolean.
# Before 2026-09-17 the optimal-diagonal family was called feedback; the
# correction made then called it open loop, which over-shot, because it reads
# as though the plant model were frozen when re-solving against a moving FRF
# is exactly what those laws do.  Two separate loops:
#
#   'error'  -- integrates the measured response error.  match_trace_
#               pseudoinverse (trace ratio) and match_diagonal_congruence
#               (per-drive log error).
#   'model'  -- closed on the IDENTIFIED PLANT MODEL, open on the response
#               error.  The optimal-diagonal family hands the live FRF to
#               _refine_batch every cycle and re-solves drifted and degraded
#               bins, but its objective is the PREDICTED response
#               diag(H X H^H) and, with response_trim_gain at 0 (the
#               default, and what every run scored here used), it never reads
#               last_response_cpsd.
#   'none'   -- one-shot synthesis, replayed unchanged.  pseudoinverse_control
#               and buzz_control.
#
# A run taken with response_trim_gain > 0 is NOT 'model' -- it closes both
# loops.  Detected from the parameter string below rather than assumed, since
# from 2026-09-21 the same law can be run either way.
LOOP = {'match_trace_pseudoinverse': 'error',
        'match_diagonal_congruence': 'error',
        'optimal_diagonal_control': 'model',
        'optimal_diagonal_control_fast': 'model',
        'pseudoinverse_control': 'none',
        'buzz_control': 'none'}
# Position of response_trim_gain in the parameter string, for the two laws
# that have one.
TRIM_POS = {'optimal_diagonal_control': 10, 'optimal_diagonal_control_fast': 10}


def _loop_of(law, par):
    base = LOOP.get(law, 'none')
    i = TRIM_POS.get(law)
    if i is not None and len(par) > i:
        try:
            if float(par[i]) > 0:
                return 'model+error'
        except ValueError:
            pass
    return base

# How each law applies the drive-coherence cap. This, not the loop type, is
# what predicts whether the cap costs anything: the optimal-diagonal laws
# solve SUBJECT TO it as an SDP constraint (optimal_diagonal_control.py:248),
# every other law post-processes it onto an already-computed solve via
# _cap_drive_coherence.
CAP_IN_SOLVE = {'optimal_diagonal_control', 'optimal_diagonal_control_fast'}

_s = importlib.util.spec_from_file_location(
    'ar', '../../../control_laws/achievable_response.py')
ar = importlib.util.module_from_spec(_s); _s.loader.exec_module(ar)


def plant_rank(sv, inb):
    """Rank of the plant, from the GAP in the singular value spectrum.

    Until 2026-11-13 this was a fixed ratio test, sigma_k/sigma_1 > 1e-3.
    That compares the measurement against a constant while the thing it has
    to separate -- the identification noise floor -- moves with the test.
    On this frame sigma_6/sigma_1 is 3e-4 at -12 dB but 2e-3 at 0 dB and
    2e-2 once the plant is softened hard, so above -12 dB the test returned
    6, the truncation did nothing, and BOTH floor columns reported the
    optimizer exploiting a direction the six shakers cannot reach.  The
    module docstring already warned about exactly that failure for the
    unrestricted floor; the same fixed threshold was silently failing the
    operational one too.

    The gap is the robust feature.  A real rank deficiency separates
    sigma_K from sigma_{K+1} by far more than neighbouring singular values
    WITHIN the reachable subspace are separated: that within-subspace
    spacing is 1.8-3.2x at every level and on both plants, while the step
    across the deficiency is 2.7-314x.  So take the largest consecutive
    ratio instead of a fixed cutoff.

    Checked on all 52 runs on disk -- 38 September 6-12-8 runs and 14
    November 6x8 nonlinear ones -- it returns 5 on 51, at every test level,
    both plants, linear and softened, where the fixed test returns 6 on 14
    of them.  The exception is run30, where the law diverged (drive trace
    2e10) and the FRF is pathological; the fixed test returns 2 there as
    well, so this is not a regression.

    Returns (rank, margin), margin being how far the winning gap beats the
    largest other one.  Margin near 1 means the spectrum shows no clear
    deficiency and the rank should not be leaned on: it runs 1.3-1.6 on the
    hardest softened runs and 10-100 on clean ones.
    """
    g = np.median(sv[inb, :-1] / np.maximum(sv[inb, 1:], 1e-300), axis=0)
    k = int(np.argmax(g))
    margin = float(g[k] / max(float(np.max(np.delete(g, k))), 1e-300))
    return k + 1, margin


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

    # Physically real rank, in-band only: out-of-band lines carry a much
    # noisier sixth direction and would wrongly report the plant full rank.
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    ratio = sv / np.maximum(sv[:, :1], 1e-300)
    numeric_rank, rank_margin = plant_rank(sv, inb)

    # Operational floor.  The cutoff goes BETWEEN the last real singular
    # value and the first noise one -- the geometric mean of their in-band
    # medians -- rather than at a fixed 1e-3, for the reason given in
    # plant_rank: a constant cannot track a noise floor that moves with the
    # test.  At -12 dB this lands near the old 1e-3 and reproduces it; above
    # 0 dB the old value no longer separated the two and the "operational"
    # floor was quietly inverting noise.
    med = np.median(ratio[inb], axis=0)
    if numeric_rank < sv.shape[1]:
        restrict = float(np.sqrt(med[numeric_rank - 1] * med[numeric_rank]))
    else:
        restrict = RESTRICT_RCOND
    ach = floor(restrict_rcond=restrict)

    # ... and the floor over the real subspace only.
    sv_trunc = sv.copy()
    sv_trunc[:, numeric_rank:] = 0.0
    H_saved = H
    H = (U * sv_trunc[:, None, :]) @ Vh
    ideal = floor()
    H = H_saved

    # --- floor AT THE DRIVE THE LAW ACTUALLY SPENT ---------------------
    # The two floors above are free to spend any drive they like, and they
    # do: the rank-5 floor wants ~+26 dB more than the laws use.  A floor
    # that assumes 400x the drive is not headroom a real controller could
    # claim, so this one is solved under a ridge on trace(X) whose weight is
    # bisected until the floor's own band-mean drive matches the run's.
    #
    # The budget is the BAND MEAN, so the floor may move drive between
    # frequencies as it likes.  That is deliberately generous -- it gives a
    # lower bound on the unavoidable error at that total power, which is the
    # conservative direction for a claim about what the law is leaving.
    #
    # Bisection is on log w because the trace falls roughly as a power of w
    # (measured: w 0.3 -> 100 moves the trace over 3.5 decades monotonically).
    lb = np.unique(np.concatenate([idx[::BUDGET_DECIMATE], idx[-1:]]))
    law_dt = np.einsum('fnn->f', D)
    law_trace = float(law_dt[lb].mean())        # diagnostic only, lb mean
    law_trace_full = float(law_dt[idx].mean())  # bisection target

    def coarse_trace(w):
        # One restart: the pseudoinverse start is deterministic and the
        # bisection only needs the trace, not a polished minimum.
        r = ar.achievable_diagonal(H, tgt, line_indices=lb, drive_weight=w,
                                   restrict_rcond=restrict,
                                   compute_minimax=False, n_restarts=1)
        sol = np.flatnonzero(np.array(r['solved'], bool))
        if sol.size == 0:
            return np.nan
        # Estimate the floor's FULL-BAND mean drive the same way the final
        # solve's is estimated -- log-interpolate the solved lines over every
        # in-band line, then average -- so the bisection compares a full-band
        # mean against the law's full-band mean.  Matching an lb-mean to an
        # lb-mean is self-consistent but hands the floor the wrong budget:
        # the law's drive is peaky and its lb-mean runs 2-10 dB above its
        # true mean.  The floor's drive is smooth in frequency, so recovering
        # its mean from the coarse set by interpolation is accurate where
        # sampling the law's would not be.
        dtc = np.array(r['drive_trace'], float)
        ldc = np.log10(np.maximum(dtc[sol], 1e-300))
        return float(np.mean(10.0 ** np.interp(f[idx], f[sol], ldc)))

    # Keep the BEST w seen, not the last one.  trace(w) is monotone in the
    # large but not bin-by-bin -- neighbouring w can land in different local
    # minima -- so a plain bisection can finish a step away from its own best
    # bracket.  Measured: keeping the last w left one run 0.85 dB off target
    # where its best probe was inside 0.1 dB.
    lo, hi = 1e-2, 1e3
    w_used, best_err = np.nan, np.inf
    for _ in range(BUDGET_ITERS):
        w = float(np.sqrt(lo * hi))
        tr = coarse_trace(w)
        if not np.isfinite(tr):
            break
        err_db = 10.0 * np.log10(max(tr, 1e-300) / max(law_trace_full, 1e-300))
        if abs(err_db) < best_err:
            best_err, w_used = abs(err_db), w
        if abs(err_db) <= BUDGET_TOL_DB:
            break
        if tr > law_trace_full:   # spending too much -> penalise harder
            lo = w
        else:
            hi = w

    if np.isfinite(w_used):
        # final evaluation on the standard line set, interpolated like the rest
        # n_restarts=1 here too: measured 3.296 vs 3.295 dB against the
        # default 2 on lg_spec0_2, for half the time.  Consistent with the
        # note in achievable_diagonal that best-of-8 equals best-of-2.
        r = ar.achievable_diagonal(H, tgt, line_indices=li, drive_weight=w_used,
                                   restrict_rcond=restrict,
                                   compute_minimax=False, n_restarts=1)
        a = np.array(r['achieved'], float)
        sol = np.flatnonzero(np.array(r['solved'], bool))
        for m in range(a.shape[1]):
            db = 10.0 * np.log10(np.maximum(a[sol, m], 1e-300))
            a[idx, m] = 10.0 ** (np.interp(f[idx], f[sol], db) / 10.0)
        budget = stats(a[inb], tgt[inb])
        budget_vs = stats(resp[inb], a[inb])
        # The bisection matches an lb-mean to an lb-mean, which is
        # self-consistent but samples a peaky drive spectrum coarsely: on
        # run26 the law's lb-mean is 10x its full-band mean, so the budget
        # handed to the floor was 10 dB more generous than intended.  Report
        # the match the way it will be read -- the final solve's drive
        # against the law's, both over every in-band line, the floor
        # interpolated exactly as its response is -- and keep the
        # bisection's own residual separately as budget_coarse_db.
        dt = np.array(r['drive_trace'], float)
        budget_trace_li = float(np.nanmean(dt[sol]))
        ldb = np.log10(np.maximum(dt[sol], 1e-300))
        budget_trace = float(np.mean(10.0 ** np.interp(f[idx], f[sol], ldb)))
        budget_match_db = 10.0 * np.log10(max(budget_trace, 1e-300)
                                          / max(law_trace_full, 1e-300))
    else:
        budget = budget_vs = dict(rms=float('nan'), pout=float('nan'),
                                  mean_db=[])
        budget_trace = budget_trace_li = budget_match_db = float('nan')

    flat = stats(resp[inb], tgt[inb])
    reach = stats(ach[inb], tgt[inb])          # floor vs specification
    against = stats(resp[inb], ach[inb])       # run vs floor
    ideal_reach = stats(ideal[inb], tgt[inb])   # rank-truncated floor

    cap = float(par[CAPPOS[law]])
    return dict(
        run=os.path.basename(path)[:5], law=SHORT[law], law_full=law,
        loop=_loop_of(law, par),
        cap_mode='in-solve' if law in CAP_IN_SOLVE else 'post-process',
        cap='off' if cap >= 1.0 else f'{cap:g}', params=','.join(par),
        rms=flat['rms'], pout=flat['pout'], mean_db=flat['mean_db'],
        reach_rms=reach['rms'], reach_pout=reach['pout'],
        reach_mean_db=reach['mean_db'],
        ach_rms=against['rms'], ach_pout=against['pout'],
        rank_rms=ideal_reach['rms'], rank_pout=ideal_reach['pout'],
        budget_rms=budget['rms'], budget_pout=budget['pout'],
        budget_ach_rms=budget_vs['rms'], budget_w=float(w_used),
        budget_trace=budget_trace, budget_law_trace=law_trace_full,
        budget_trace_li=budget_trace_li, budget_law_trace_lb=law_trace,
        budget_match_db=float(budget_match_db),
        budget_coarse_db=float(best_err) if np.isfinite(best_err) else float('nan'),
        numeric_rank=numeric_rank, rank_margin=rank_margin,
        restrict_rcond_used=restrict,
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
                  f"budget {r['budget_rms']:5.2f} "
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
