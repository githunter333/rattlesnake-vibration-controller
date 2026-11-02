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
    # blank fields AHEAD of the octave law's own positions: it counts only the
    # non-empty fields, the base class counts them all
    ('1e-6,,20,1.0,0.95,5', ()),
    ('1e-6,,20,1.0,0.95,5,0,6,100,1000', ()),
    (',,,,,5,0,6,100,1000', ()),
    ('1e-6,0.5,20,1.0,0.95,,5,0,6,100,1000', ()),
]


# The named form's strings.  They cannot be recorded from the old code (it has no
# named form); they are recorded from the rewritten parser, error messages
# included, and pin it from then on.
NAMED = [
    'reg: 1e-5',
    'reg: 1e-5\nmax_bins_per_update: 10\nmax_drive_coherence: 0.8',
    'reg: 1e-5; max_bins_per_update: 10; max_drive_coherence: 0.8',
    '# a comment\n\n  REG :  2e-6  # trailing\nMax_Bins_Per_Update: 3',
    'error_domain: db', 'error_domain: linear', 'error_domain: 1',
    'reg:\nmax_bins_per_update: 7',
    'svd_select: 1\nsvd_min_keep: -1\nbm_rank: 2',
    'response_trim_gain: -2\nresponse_trim_limit_db: -1.5\nsvd_gap_min: 0.2\nn_irls_passes: -4',
    'max_solve_ms_per_update: 250\nmax_bins_per_update: 400',
    'frequency_spacing: 5\nswitch_level_db: -3\noctave_fraction: 3\noctave_fmin: 200\noctave_fmax: 800',
    'frequency_spacing: 5\noctave_fmin: 100\noctave_fmax: 1000',
    # mistakes
    'max_bin_per_update: 5', 'rgu: 1', 'bm_rank: 3', 'frequency_spacing: 5',
    'reg: abc', 'reg: 1\nreg: 2', 'reg: 1\n20', 'octave_fmin: 100\noctave_fmax: 50',
]

# What each position of the legacy comma string means, written out here by hand
# and NOT read from the laws' tables, so the twin check below can catch a wrong
# name or position in a table.  (The octave law's five are left out: its legacy
# reading overlaps the base class's, so it has no clean named equivalent.)
LEGACY_NAME_OF = {
    0: 'reg', 1: 'frf_update_threshold', 2: 'max_bins_per_update', 3: 'error_threshold_db',
    4: 'max_drive_coherence', 5: 'bm_rank', 6: 'startup_test_level_cap_db', 7: 'error_domain',
    8: 'n_irls_passes', 9: 'drive_rcond', 10: 'response_trim_gain', 11: 'response_trim_limit_db',
    12: 'response_trim_step_db', 13: 'response_trim_deadband_db', 14: 'response_trim_gate_db',
    15: 'svd_select', 16: 'svd_gap_min', 17: 'svd_floor', 18: 'svd_weak_max',
    19: 'svd_min_keep', 20: 'max_solve_ms_per_update', 21: 'cap_baseline',
    22: 'db_coherence_guard', 23: 'warm_start', 24: 'gate_reference',
}


def to_named(key, legacy):
    """The named string with the same parameters as this legacy string, for the
    SDP or fast law."""
    out = []
    for i, raw in enumerate(x.strip() for x in legacy.split(',')):
        name = LEGACY_NAME_OF.get(i)
        if raw and name in LAW_ATTRS[key]:
            out.append(f'{name}: {raw}')
    return '\n'.join(out)


def strings():
    seen, out = set(), []
    def add(s, tags=()):
        if s not in seen:
            seen.add(s)
            out.append((s, tuple(tags)))
    for s in ARCHIVED:
        add(s, ('archived',))
    for sc in gopt.SCENARIOS:
        if ':' not in sc['params']:                # the named twins are added below
            add(sc['params'], ('scenario',))
    for s, t in EDGE:
        add(s, t)
    for s in NAMED:
        add(s, ('named',))
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


def run_pairs(law_dir, pairs):
    """Resolved parameters for [(law key, string), ...]."""
    laws = gopt.load_laws(law_dir)
    for key, cls in laws.items():
        LAW_ATTRS_FOR[cls] = LAW_ATTRS[key]
    p = frame_plant('6x8')
    return [resolve(laws[k], p['spec'], p['warn'], p['abort'], s) for k, s in pairs]


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
    # the named twin of every legacy string must resolve to the same parameters
    pairs, legacy_of = [], []
    for rec in gold['strings']:
        s, tags = rec['string'], rec['tags']
        if ':' in s or 'bad_value' in tags:
            continue
        for key in ('SDP', 'fast'):
            pairs.append((key, to_named(key, s)))
            legacy_of.append((key, s))
    twins = run_pairs(law_dir, pairs)
    bad_twins = [(k, s, t, r) for (k, s), (_, t), r in zip(legacy_of, pairs, twins)
                 if r != res[s][k]]
    print(f'\nnamed twins: {len(pairs) - len(bad_twins)} of {len(pairs)} legacy strings resolve '
          'identically when written with names')
    for k, s, t, r in bad_twins[:10]:
        print(f'  {k:5s} {s!r} -> {t!r} differs')
    hard = hard + [(s, k, {}) for k, s, t, r in bad_twins]
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
