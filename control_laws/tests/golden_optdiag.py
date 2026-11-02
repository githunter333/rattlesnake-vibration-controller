"""
Golden-output regression test for the optimal diagonal laws.

Purpose: let the laws be cleaned up (named parameters, splitting the long
methods, retiring options) without silently changing what they output.  It
runs fixed closed-loop scenarios on the frame plants (frame_plant.py) with a
seeded measurement-noise generator, and compares what each law returns with a
recorded reference.  The laws use no random numbers, and two runs on one
machine are bit-identical (checked 2026-11-02), so the comparison is exact
first and numeric second:

  exact    every cycle's drive CPSD hashes identically to the recorded one.
  numeric  not bit-identical, but within tolerance: the recorded first/last
           drive CPSD agree to --rtol (relative to the largest element), the
           per-cycle rms dB error to --atol-db, and every solve counter
           matches exactly.  Expected on another machine or BLAS; a refactor
           that only reorders floating-point arithmetic would land here too.
  FAIL     anything else.

Scenarios (SCENARIOS below) cover every parameter field of the laws at a
non-default value, so a parser change that shifts or drops a field shows up as
a different output.  Time-budget runs use a budget so large that the line
count, not the clock, decides the work per call; a wall-clock limit would make
the reference machine-dependent.

Usage (from the repo root):
    python control_laws/tests/golden_optdiag.py                  check against the recorded reference
    python control_laws/tests/golden_optdiag.py --strict         ...and require bit-identical output
    python control_laws/tests/golden_optdiag.py --record         write the reference (see below)
    python control_laws/tests/golden_optdiag.py --list           list the scenarios
    python control_laws/tests/golden_optdiag.py --only sdp_default fast_10x14_default

Run in the conda env that has cvxpy ('rattlesnake').  src/ is put on the path
automatically; the laws are loaded the way Rattlesnake loads a control script.

WHEN TO RE-RECORD: only after a change that is MEANT to alter output, and say
so in the commit message.  A refactor that is meant to preserve behavior must
pass this test unchanged.  Scenarios tagged 'legacy_collision' record a
behavior known to be wrong (octave_band_switching_control reads the same
parameter positions as the base class); they are expected to change when that
is fixed.

Files: control_laws/tests/golden/optdiag_golden.json (manifest, hashes, series,
counters) and optdiag_golden.npz (first and last in-band drive CPSD per
scenario, complex64).
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(LAW_DIR)
GOLDEN_DIR = os.path.join(HERE, 'golden')
GOLDEN_JSON = os.path.join(GOLDEN_DIR, 'optdiag_golden.json')
GOLDEN_NPZ = os.path.join(GOLDEN_DIR, 'optdiag_golden.npz')

sys.path.insert(0, HERE)
try:
    import rattlesnake.utilities  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(REPO, 'src'))
import coherence_sweep as cs                          # noqa: E402
from frame_plant import frame_plant                   # noqa: E402
from rattlesnake.utilities import load_control_module  # noqa: E402

LAW_FILES = {
    'SDP': ('optimal_diagonal_control.py', 'optimal_diagonal_control'),
    'fast': ('optimal_diagonal_control_fast.py', 'optimal_diagonal_control_fast'),
    'octave': ('octave_band_switching_control.py', 'octave_band_switching_control'),
}
COUNTERS = ['n_sdp_refinements', 'n_stale_refinements', 'n_fast_solves', 'n_safe_solves',
            'n_frf_updates', 'n_trim_updates', 'n_octave_switches', 'n_octave_band_solves']
SEED = 1
N_FRAMES = 20


def fields(values):
    """Comma-separated extra_parameters string from {field index: value}.
    Written this way so a scenario names its fields instead of counting commas."""
    if not values:
        return ''
    out = [''] * (max(values) + 1)
    for i, v in values.items():
        out[i] = repr(v) if isinstance(v, float) else str(v)
    return ','.join(out)


def scenario(name, plant, law, params='', n_cycles=6, level_db=None, frf_drift=0.0,
             frf_walk=False, tags=()):
    """frf_drift: each control cycle hands the law an FRF whose every element is
    multiplied by 1 + frf_drift * (complex unit-variance noise), as if "update
    transfer function during control" were on; 0 = the FRF never changes.
    frf_walk: the noise accumulates from cycle to cycle (a random walk), so the
    movement since the previous call stays small while the movement since a
    line's last solve grows -- the only case where field 24 (gate_reference)
    can change anything."""
    return dict(name=name, plant=plant, law=law, params=params, n_cycles=n_cycles,
                level_db=level_db, frf_drift=frf_drift, frf_walk=frf_walk, tags=list(tags))


# Field indices (0-based), from the laws' docstrings:
#  0 reg  1 frf_update_threshold  2 max_bins_per_update  3 error_threshold_db
#  4 max_drive_coherence  5 bm_rank (fast)  6 startup_test_level_cap_db
#  7 error_domain (0 linear / 1 dB)  8 n_irls_passes  9 drive_rcond
#  10-14 response trim: gain, limit_db, step_db, deadband_db, gate_db
#  15-19 svd_select, svd_gap_min, svd_floor, svd_weak_max, svd_min_keep (fast)
#  20 max_solve_ms_per_update  21 cap_baseline  22 db_coherence_guard
#  23 warm_start  24 gate_reference
#
# Every field a scenario sets is given a value that DIFFERS from its default AND
# is large enough to change the output on these plants (checked by deleting and
# by shifting each parameter read in a copy of the laws: error_threshold_db
# does nothing below ~12 dB here, svd_floor nothing below the plant's small
# singular ratios, the trim limit nothing until it is smaller than the trim's
# own travel), so a parser that drops, shifts or ignores it changes the output.  Field 1
# (the FRF-movement threshold) and field 24 only matter when the FRF moves, so
# they are set in the frf_drift scenarios.  The time budget (field 20) is
# 0.001 ms: the scheduler always runs the first solve of a call, so that is
# exactly one line per call on any machine, where a budget a real clock could
# hit would make the reference machine-dependent.
SCENARIOS = [
    scenario('sdp_default', '6x8', 'SDP'),
    scenario('sdp_db_irls_cap', '6x8', 'SDP',
             fields({0: 1e-5, 2: 10, 3: 14.0, 4: 0.8, 6: -12.0, 7: 1, 8: 3})),
    scenario('sdp_trim_on', '6x8', 'SDP',
             fields({10: 1.0, 11: 0.5, 12: 0.4, 13: 0.3, 14: 0.4})),
    scenario('sdp_time_budget_one_line', '6x8', 'SDP', fields({2: 15, 20: 0.001})),
    scenario('sdp_frf_drift', '6x8', 'SDP', fields({1: 0.02}), frf_drift=0.05),
    scenario('fast_default', '6x8', 'fast'),
    scenario('fast_rank_svd_rcond', '6x8', 'fast',
             fields({0: 1e-5, 2: 10, 3: 0.5, 4: 0.8, 5: 3, 9: 0.5,
                     15: 1, 16: 2.0, 17: 0.03, 18: 0.2, 19: 2})),
    scenario('fast_db_guard_flags', '6x8', 'fast', fields({7: 1, 22: 2})),
    # Fields 21 and 23 are in scenarios of their own: set next to a neighbour
    # with the same value, a parser that read the wrong position would not show.
    # cap_baseline only acts on lines still unsolved, hence the small line cap
    # and a coherence cap tight enough to bind.
    scenario('fast_cap_baseline', '6x8', 'fast', fields({2: 10, 4: 0.5, 21: 1})),
    scenario('fast_warm_start_off', '6x8', 'fast', fields({23: 0})),
    scenario('fast_trim_on', '6x8', 'fast',
             fields({10: 1.0, 11: 0.5, 12: 0.4, 13: 0.3, 14: 0.4})),
    scenario('fast_time_budget_one_line', '6x8', 'fast', fields({20: 0.001})),
    scenario('fast_frf_drift', '6x8', 'fast', fields({1: 0.02}), frf_drift=0.05),
    scenario('fast_frf_walk', '6x8', 'fast', fields({1: 0.05}), frf_drift=0.03, frf_walk=True),
    scenario('fast_frf_walk_gate_ref0', '6x8', 'fast', fields({1: 0.05, 24: 0}),
             frf_drift=0.03, frf_walk=True),
    scenario('octave_no_spacing', '6x8', 'octave'),
    scenario('octave_engaged_legacy_string', '6x8', 'octave',
             '1e-6,0.5,20,1.0,0.95,5,0,6,100,1000', n_cycles=3, level_db=0.0,
             tags=['legacy_collision']),
    scenario('sdp_10x14_default', '10x14', 'SDP'),
    scenario('fast_10x14_default', '10x14', 'fast'),
    scenario('fast_10x14_rankdef', '10x14_rankdef', 'fast', n_cycles=5),
]


def load_laws(law_dir=LAW_DIR):
    laws = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for key, (fname, cls) in LAW_FILES.items():
            laws[key] = getattr(load_control_module(os.path.join(law_dir, fname)), cls)
    return laws


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24]


def run_scenario(sc, laws, plants):
    """Closed loop exactly as compare_optdiag_frame.run: system ID, a first
    control call with no measurement, then n_cycles cycles each fed a measured
    response (seeded, N_FRAMES Gaussian frames)."""
    if sc['plant'] not in plants:
        plants[sc['plant']] = frame_plant(sc['plant'])
    p = plants[sc['plant']]
    rng = np.random.default_rng(SEED)
    rng_h = np.random.default_rng(SEED + 1000)      # only used when frf_drift > 0
    H, sel = p['H'], p['band']
    noise_diag = np.real(np.einsum('fii->fi', p['gxx_noise']))
    s_d = np.real(np.einsum('fii->fi', p['spec']))[sel]
    args = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
    nf = N_FRAMES

    def err_rms(X):
        y = np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj()))[sel]
        return float(np.sqrt(np.mean((10*np.log10(np.maximum(y, 1e-300)/s_d))**2)))

    hashes, errs, drive = [], [], []

    def record(X):
        hashes.append(sha(X))
        errs.append(err_rms(X))
        drive.append(float(10*np.log10(np.mean(np.real(np.einsum('fii->f', X))[sel]))))

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        law = laws[sc['law']](p['spec'], p['warn'], p['abort'], sc['params'], None,
                              *args, None, 0, nf, None, None)
        if sc['level_db'] is not None:
            law.set_test_level_db(sc['level_db'])
        law.system_id_update(H, *args, p['coh'], nf, nf)
        X = law.control(H, p['coh'], nf, nf, None, None)
        first = np.array(X[sel], dtype=np.complex64)
        record(X)
        walk = np.zeros(H.shape, complex)
        for _ in range(sc['n_cycles']):
            Gyy, Gxx, coh = cs.measure(H, X, noise_diag, N_FRAMES, rng)
            H_law = H
            if sc['frf_drift']:
                z = (rng_h.standard_normal(H.shape) + 1j*rng_h.standard_normal(H.shape))/np.sqrt(2)
                if sc['frf_walk']:
                    walk = walk + sc['frf_drift']*z
                    H_law = H*(1 + walk)
                else:
                    H_law = H*(1 + sc['frf_drift']*z)
            X = law.control(H_law, coh, nf, nf, Gyy, Gxx)
            record(X)
    last = np.array(X[sel], dtype=np.complex64)
    counters = {k: int(getattr(law, k)) for k in COUNTERS if hasattr(law, k)}
    if getattr(law, 'sdp_refined', None) is not None:
        counters['refined_lines'] = int(np.sum(law.sdp_refined))
    return dict(hashes=hashes, err_rms=errs, drive_db=drive, counters=counters,
                first=first, last=last, seconds=time.perf_counter() - t0)


def law_sources(law_dir=LAW_DIR):
    return {fname: hashlib.sha256(open(os.path.join(law_dir, fname), 'rb').read()).hexdigest()[:16]
            for fname, _ in LAW_FILES.values()}


def environment():
    import scipy
    try:
        import cvxpy
        cvx = cvxpy.__version__
    except ImportError:
        cvx = None
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                cvxpy=cvx, machine=platform.machine(), system=platform.system())


def record(selected, law_dir=LAW_DIR):
    laws, plants = load_laws(law_dir), {}
    manifest = dict(recorded=time.strftime('%Y-%m-%d %H:%M:%S'), environment=environment(),
                    law_sources=law_sources(law_dir), seed=SEED, n_frames=N_FRAMES, scenarios={})
    snaps = {}
    for sc in selected:
        r = run_scenario(sc, laws, plants)
        manifest['scenarios'][sc['name']] = dict(
            definition=sc, hashes=r['hashes'], err_rms=r['err_rms'], drive_db=r['drive_db'],
            counters=r['counters'])
        snaps[sc['name'] + '__first'] = r['first']
        snaps[sc['name'] + '__last'] = r['last']
        print(f"recorded {sc['name']:32s} {r['seconds']:5.1f} s  final err {r['err_rms'][-1]:.3f} dB", flush=True)
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(GOLDEN_JSON, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    np.savez_compressed(GOLDEN_NPZ, **snaps)
    print(f'\nwrote {os.path.relpath(GOLDEN_JSON, REPO)} and {os.path.relpath(GOLDEN_NPZ, REPO)}')


def check(selected, strict, rtol, atol_db, law_dir=LAW_DIR):
    with open(GOLDEN_JSON) as fh:
        gold = json.load(fh)
    snaps = np.load(GOLDEN_NPZ)
    laws, plants = load_laws(law_dir), {}
    env_now, env_then = environment(), gold['environment']
    if env_now != env_then:
        diff = {k: (env_then.get(k), env_now.get(k)) for k in env_now if env_now[k] != env_then.get(k)}
        print(f'note: environment differs from the recording (recorded -> now): {diff}')
        print('      bit-identical output is not guaranteed across environments; numeric matches are expected there.\n')
    src_now = law_sources(law_dir)
    changed = [f for f, h in src_now.items() if gold['law_sources'].get(f) != h]
    print('law source changed since recording: ' + (', '.join(changed) if changed else 'none') + '\n')

    n_exact = n_numeric = n_fail = 0
    for sc in selected:
        g = gold['scenarios'].get(sc['name'])
        if g is None:
            print(f"FAIL     {sc['name']:32s} not in the recording (run --record)")
            n_fail += 1
            continue
        if g['definition'] != sc:
            print(f"FAIL     {sc['name']:32s} scenario definition differs from the recording (run --record if intended)")
            n_fail += 1
            continue
        r = run_scenario(sc, laws, plants)
        tag = f"  [{', '.join(sc['tags'])}]" if sc['tags'] else ''
        problems = []
        if r['counters'] != g['counters']:
            problems.append(f"counters {g['counters']} -> {r['counters']}")
        err_d = float(np.max(np.abs(np.array(r['err_rms']) - np.array(g['err_rms']))))
        if err_d > atol_db:
            problems.append(f'per-cycle error differs by up to {err_d:.2e} dB')
        rel = {}
        for key in ('first', 'last'):
            ref = snaps[f"{sc['name']}__{key}"]
            d = np.max(np.abs(r[key].astype(complex) - ref.astype(complex)))
            rel[key] = float(d/max(np.max(np.abs(ref)), 1e-300))
            if rel[key] > rtol:
                problems.append(f'{key} drive CPSD differs by {rel[key]:.2e} (relative)')
        first_bad = next((i for i, (a, b) in enumerate(zip(r['hashes'], g['hashes'])) if a != b), None)
        exact = r['hashes'] == g['hashes']
        if problems:
            n_fail += 1
            print(f"FAIL     {sc['name']:32s} {r['seconds']:5.1f} s{tag}")
            for pr in problems:
                print(f'           - {pr}')
            if first_bad is not None:
                print(f'           - first differing cycle: {first_bad} (0 = the first control call)')
        elif exact:
            n_exact += 1
            print(f"PASS     {sc['name']:32s} {r['seconds']:5.1f} s  exact{tag}")
        elif strict:
            n_fail += 1
            print(f"FAIL     {sc['name']:32s} {r['seconds']:5.1f} s  not bit-identical (--strict); "
                  f"drive CPSD rel diff first {rel['first']:.1e}, last {rel['last']:.1e}, "
                  f"first differing cycle {first_bad}{tag}")
        else:
            n_numeric += 1
            print(f"PASS     {sc['name']:32s} {r['seconds']:5.1f} s  numeric (not bit-identical; rel diff "
                  f"first {rel['first']:.1e}, last {rel['last']:.1e}; first differing cycle {first_bad}){tag}")
    print(f'\n{len(selected)} scenarios: {n_exact} exact, {n_numeric} numeric, {n_fail} failed')
    return 0 if n_fail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--record', action='store_true', help='write the reference instead of checking')
    ap.add_argument('--strict', action='store_true', help='require bit-identical output')
    ap.add_argument('--list', action='store_true', help='list the scenarios and exit')
    ap.add_argument('--only', nargs='+', metavar='NAME', help='run only these scenarios')
    ap.add_argument('--law-dir', default=LAW_DIR, metavar='DIR',
                    help='load the laws from DIR instead of control_laws/ (to test a copy or a candidate refactor)')
    ap.add_argument('--rtol', type=float, default=1e-5, help='relative tolerance on the drive CPSD (default 1e-5)')
    ap.add_argument('--atol-db', type=float, default=1e-4, help='tolerance on per-cycle rms dB error (default 1e-4)')
    a = ap.parse_args()
    names = [s['name'] for s in SCENARIOS]
    if a.list:
        for s in SCENARIOS:
            print(f"{s['name']:32s} {s['plant']:14s} {s['law']:7s} {s['n_cycles']} cycles  "
                  f"params={s['params']!r}" + (f" level_db={s['level_db']}" if s['level_db'] is not None else '')
                  + (f" frf_drift={s['frf_drift']}" + (' (walk)' if s['frf_walk'] else '') if s['frf_drift'] else ''))
        return 0
    selected = SCENARIOS
    if a.only:
        bad = [n for n in a.only if n not in names]
        if bad:
            ap.error(f'unknown scenario(s): {bad}; see --list')
        selected = [s for s in SCENARIOS if s['name'] in a.only]
    if a.record:
        if a.only:
            ap.error('--record rewrites the whole reference; do not combine it with --only')
        record(selected, a.law_dir)
        return 0
    if not os.path.exists(GOLDEN_JSON):
        print('no recorded reference; run with --record first')
        return 1
    return check(selected, a.strict, a.rtol, a.atol_db, a.law_dir)


if __name__ == '__main__':
    sys.exit(main())
