"""
Golden test for HOW THE OPTIMAL DIAGONAL LAWS READ THEIR PARAMETER STRING.

golden_optdiag.py proves the laws' output does not change; this proves the
parameters they resolve from a string do not change, without running a solve.
Each string below is handed to each law's constructor (no FRF, so nothing is
solved) and the resulting parameter attributes are compared with a recording
made from the code as it was before the parameter parser was rewritten
(step 1b of the optimal diagonal cleanup).

The strings are
  * every distinct string in the archived profiles under examples/ (15, all
    comma-separated; none for the octave law),
  * every string the golden_optdiag scenarios use,
  * edge cases: empty fields, trailing commas, spaces, more fields than any
    law reads, negative values (the max/abs transforms), integers written in
    scientific notation, values that do not parse, and 5-10 field strings
    for the octave law (whose positions 5-9 the base class also reads).

Each string is applied to ALL THREE laws, so the collision between the octave
law and the base class is part of the record.

Strings tagged 'bad_value' contain a field that does not parse.  The old parser
silently abandoned that field AND every field after it; the new one keeps the
defaults only for the field that failed.  Those are expected to change and are
reported separately; everything else must match exactly.

Usage (from the repo root):
    python control_laws/tests/golden_params.py              check against the recording
    python control_laws/tests/golden_params.py --record     (re)write it
    python control_laws/tests/golden_params.py --law-dir DIR  check a copy of the laws

Recording: control_laws/tests/golden/params_golden.json
"""
import argparse
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_optdiag as gopt                                  # noqa: E402
from frame_plant import frame_plant                            # noqa: E402

GOLDEN = os.path.join(gopt.GOLDEN_DIR, 'params_golden.json')

# The parameter attributes of each law, as they are named today.
BASE_ATTRS = [
    'reg', 'frf_update_threshold', 'max_bins_per_update', 'error_threshold_db',
    'max_drive_coherence', 'startup_test_level_cap_db', 'error_domain',
    'n_irls_passes', 'drive_rcond', 'max_solve_ms_per_update',
    'response_trim_gain', 'response_trim_limit_db', 'response_trim_step_db',
    'response_trim_deadband_db', 'response_trim_gate_db',
]
FAST_ATTRS = [
    'bm_rank', 'svd_select', 'svd_gap_min', 'svd_floor', 'svd_weak_max',
    'svd_min_keep', 'cap_baseline', 'db_coherence_guard', 'warm_start',
    'gate_reference',
]
OCTAVE_ATTRS = [
    'frequency_spacing', 'switch_level_db', 'octave_fraction', 'octave_fmin',
    'octave_fmax',
]
LAW_ATTRS = {'SDP': BASE_ATTRS, 'fast': BASE_ATTRS + FAST_ATTRS,
             'octave': BASE_ATTRS + OCTAVE_ATTRS}

# Distinct strings from the archived profiles (examples/**/*.xlsx, 2026-11-02).
ARCHIVED = [
    '1e-6,0.05,20,1.0,1.0',
    '1e-6,0.05,20,1.0,0.95',
    '1e-6,0.05,20,1.0,0.95,6',
    '1e-6,0.5,20,1.0,0.95,6',
    '1e-6,0.05,20,1.0,1.0,6',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,1,2,1e-2',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,1,2,1e-3',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,1,2,3e-3',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,0,2,3e-2',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,0,2,5e-2',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,0,2,2.0,0,3.0,0.5,0.5,0.5',
    '1e-6,0.05,20,1.0,0.95,6,-9.0,0,2,2.0,0.5,3.0,0.5,0.5,0.5',
    '1e-6,0.5,20,1.0,0.95,6,-9.0,0,2,2.0,0,3.0,0.5,0.5,0.5',
    '1e-6,0.5,20,1.0,0.95,6,-9.0,0,2,2.0,0.5,3.0,0.5,0.5,0.5',
]

# (string, tags) -- edge cases
EDGE = [
    ('', ()), (' ', ()), (',', ()), (',,,', ()),
    ('1e-6,', ()), ('1e-6,0.5,20,', ()), (' 1e-6 , 0.5 , 20 ', ()),
    (',,200', ()), (',,,,,,,,,,,,,,,,,,,,,,,,,', ()),
    ('1e-6,0.5,20,1.0,0.95,4,-9,0,2,2.0,0,3,0.5,0.5,0.5,1,3,1e-6,0.05,0,0,0,1,1,1', ()),
    ('1e-6,0.5,20,1.0,0.95,4,-9,0,2,2.0,0,3,0.5,0.5,0.5,1,3,1e-6,0.05,0,0,0,1,1,1,9,9,9', ()),
    ('2e1', ()), (',,2e1,,,3e0', ()), (',,,,,,,2.7,2.7', ()),
    ('0', ()), ('-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1', ()),
    (',,,,,,,,,,-1,-2,-0.4,-0.3,-0.4,,0.5,-1e-4,-0.2', ()),
    (',,,,,,,,,,,,,,,,0.1', ()),
    ('0.8', ()), ('1e-6,0.5,20,1.0,0.95,0,0,0,0,0,0,0,0,0,0', ()),
    # values that do not parse: old parser stops at the first one
    ('abc', ('bad_value',)),
    ('1e-6,abc,20,1.0,0.95,6', ('bad_value',)),
    ('1e-6,0.5,20,x,0.95,6,-9.0,1,2,1e-2', ('bad_value',)),
    (',,,,,,,,,,,,,,,,,,,,,,,,zzz', ('bad_value',)),
    ('1e-6,0.5,20,1.0,0.95,6,-9.0,1,2,1e-2,1.0,2.0,nope,0.3,0.4', ('bad_value',)),
    # the octave law: 5 to 10 fields, and the form that engages octave mode
    ('1e-6,0.5,20,1.0,0.95,5', ()),
    ('1e-6,0.5,20,1.0,0.95,5,0', ()),
    ('1e-6,0.5,20,1.0,0.95,5,0,6', ()),
    ('1e-6,0.5,20,1.0,0.95,5,0,6,100', ()),
    ('1e-6,0.5,20,1.0,0.95,5,0,6,100,1000', ()),
    ('1e-6,0.5,20,1.0,0.95,5,-3,3,200,800', ()),
    ('1e-6,0.5,20,1.0,0.95,5,0,6,,1000', ()),
]


def strings():
    seen, out = set(), []
    def add(s, tags=()):
        if s not in seen:
            seen.add(s)
            out.append((s, tuple(tags)))
    for s in ARCHIVED:
        add(s, ('archived',))
    for sc in gopt.SCENARIOS:
        add(sc['params'], ('scenario',))
    for s, t in EDGE:
        add(s, t)
    return out


def resolve(cls, spec, warn, abort, params):
    """Parameter attributes of a law built with this string; or the error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            law = cls(spec, warn, abort, params)
    except Exception as e:                              # noqa: BLE001
        return {'error': f'{type(e).__name__}: {e}'}
    return {a: (repr(getattr(law, a)) if hasattr(law, a) else '<missing>')
            for a in LAW_ATTRS_FOR[cls]}


LAW_ATTRS_FOR = {}


def run_all(law_dir):
    laws = gopt.load_laws(law_dir)
    for key, cls in laws.items():
        LAW_ATTRS_FOR[cls] = LAW_ATTRS[key]
    p = frame_plant('6x8')
    result = {}
    for s, _ in strings():
        result[s] = {key: resolve(cls, p['spec'], p['warn'], p['abort'], s)
                     for key, cls in laws.items()}
    return result


def record(law_dir):
    res = run_all(law_dir)
    manifest = dict(
        note='parameters each law resolves from each string, recorded from the code '
             'before the parser rewrite (step 1b); see golden_params.py',
        law_sources=gopt.law_sources(law_dir), environment=gopt.environment(),
        strings=[{'string': s, 'tags': list(t), 'resolved': res[s]} for s, t in strings()])
    os.makedirs(gopt.GOLDEN_DIR, exist_ok=True)
    with open(GOLDEN, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    n_err = sum(1 for s in res for k in res[s] if 'error' in res[s][k])
    print(f'recorded {len(res)} strings x {len(LAW_ATTRS)} laws '
          f'({n_err} constructions raised) -> {os.path.relpath(GOLDEN, os.path.dirname(HERE))}')


def check(law_dir):
    with open(GOLDEN) as fh:
        gold = json.load(fh)
    res = run_all(law_dir)
    changed = [f for f, h in gopt.law_sources(law_dir).items() if gold['law_sources'].get(f) != h]
    print('law source changed since recording: ' + (', '.join(changed) if changed else 'none') + '\n')
    n_same = 0
    hard, expected = [], []
    for rec in gold['strings']:
        s, tags = rec['string'], rec['tags']
        for key, was in rec['resolved'].items():
            now = res.get(s, {}).get(key)
            if now == was:
                n_same += 1
                continue
            diffs = ({k: (was.get(k), now.get(k) if now else None) for k in set(was) | set(now or {})
                      if was.get(k) != (now or {}).get(k)})
            (expected if 'bad_value' in tags else hard).append((s, key, diffs))
    total = sum(len(r['resolved']) for r in gold['strings'])
    print(f'{n_same} of {total} (string, law) pairs identical')
    for label, rows in (('UNEXPECTED CHANGES', hard), ('expected changes (bad_value strings)', expected)):
        if rows:
            print(f'\n{label}: {len(rows)}')
            for s, key, diffs in rows[:25]:
                print(f'  {key:6s} {s!r}')
                for k, (a, b) in list(diffs.items())[:6]:
                    print(f'           {k}: {a} -> {b}')
    print('\n' + ('FAIL' if hard else 'PASS') + f' ({len(hard)} unexpected, {len(expected)} expected)')
    return 1 if hard else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--law-dir', default=gopt.LAW_DIR)
    a = ap.parse_args()
    if a.record:
        record(a.law_dir)
        return 0
    if not os.path.exists(GOLDEN):
        print('no recording; run with --record first')
        return 1
    return check(a.law_dir)


if __name__ == '__main__':
    sys.exit(main())
