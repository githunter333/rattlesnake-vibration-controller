"""Collect the random control laws into one loadable folder, from their repos.

    python control_laws/tests/make_law_set.py            build it
    python control_laws/tests/make_law_set.py --check    exit 1 if anything drifted

Every file is extracted read-only with `git show <ref>:<path>` and its blob hash
is verified against `git rev-parse <ref>:<path>` in the source repo, so the
folder holds repo bytes and the manifest cannot drift from the code.  Nothing is
fetched and no source repo is modified.

WHY ONE FLAT FOLDER AND NOT TWO
-------------------------------
Rattlesnake loads a law with importlib from a path, with no package context, so
a law that needs a sibling finds it by path next to itself.  Three laws do:
`optimal_diagonal_control_fast` and `octave_band_switching_control` load
`optimal_diagonal_control.py`, `broyden_congruence_control` loads
`diagonal_congruence_control.py`, and `optimal_diagonal_control`,
`diagonal_congruence_control`, `independent_drive_control` and
`pareto_target_control` all load `control_laws.py`.  Splitting the set by loop
status would separate those pairs and break them, so `random/` is flat and the
loop status of each law is recorded in the manifest instead.
`random_not_closed_loop/` holds only the law that cannot close in any
configuration AND has no sibling to lose.

WHAT IS NOT COMMITTED
---------------------
`bzwink/rattlesnake-band-average` is a PRIVATE repo (GitHub API returns 404
unauthenticated, checked 2026-11-09) and its band-average study is unreleased.
The two files sourced from it are written into the folder so the set is
complete and runnable here, and are listed in .gitignore so they cannot be
pushed to this public fork.  `private=True` below is what drives that.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
CONTROLLER = os.path.dirname(LAW_DIR)
HOME = os.path.expanduser('~')
BAND_AVERAGE = os.path.join(HOME, 'Code/python/rattlesnake-band-average')
FORCEFINDER = os.path.join(HOME, 'Code/python/forcefinder')
HERTZFORGE = os.path.join(HOME, 'Code/python/KSC/Hertzforge')
OUT_DIR = os.path.join(LAW_DIR, 'law_set')

# group, out name, source repo, ref, path in that repo, loop status, laws
LAWS = [
    ('random', 'control_laws.py', CONTROLLER, 'HEAD', 'control_laws/control_laws.py',
     'mixed', 'match_trace_pseudoinverse, match_trace_pseudoinverse_pi, '
     'buzz_feedback, match_trace_pi_resolve, match_trace_resolve (closed); '
     'pseudoinverse_control, buzz_control + _generator + _class (open)', False),
    ('random', 'optimal_diagonal_control.py', CONTROLLER, 'HEAD',
     'control_laws/optimal_diagonal_control.py',
     'param', 'optimal_diagonal_control -- closed with response_trim_gain > 0', False),
    ('random', 'optimal_diagonal_control_fast.py', CONTROLLER, 'HEAD',
     'control_laws/optimal_diagonal_control_fast.py',
     'param', 'optimal_diagonal_control_fast -- inherits the trim gain', False),
    ('random', 'octave_band_switching_control.py', CONTROLLER, 'HEAD',
     'control_laws/octave_band_switching_control.py',
     'param', 'octave_band_switching_control -- inherits the trim gain', False),
    ('random', 'diagonal_congruence_control.py', CONTROLLER, 'HEAD',
     'control_laws/diagonal_congruence_control.py',
     'closed', 'match_diagonal_congruence', False),
    ('random', 'broyden_congruence_control.py', CONTROLLER, 'HEAD',
     'control_laws/broyden_congruence_control.py',
     'closed', 'match_diagonal_congruence_broyden', False),
    ('random', 'independent_drive_control.py', CONTROLLER, 'HEAD',
     'control_laws/independent_drive_control.py',
     'open', 'independent_drive_control, independent_drive_generator -- '
     'Smallwood, the startup solve of a closed-loop law', False),
    ('random', 'pareto_target_control.py', CONTROLLER, 'HEAD',
     'control_laws/pareto_target_control.py',
     'open', 'pareto_optimal_control, pareto_optimal_generator -- Coletti et al., '
     'target definition rather than a control law', False),
    ('random', 'achievable_response.py', CONTROLLER, 'HEAD',
     'control_laws/achievable_response.py',
     'n/a', 'not a law -- the reachable-target floor, and the scoring denominator', False),
    ('random', 'random_vibration_sys_id_control_law.py', CONTROLLER, 'HEAD',
     'src/rattlesnake/environment/random_vibration_sys_id_control_law.py',
     'mixed', "Rattlesnake's built-ins: match_trace_pseudoinverse (closed); "
     'pseudoinverse_control, buzz_control + _generator + _class (open, no caps)', False),
    ('random', 'spr_random_control_law.py', FORCEFINDER, 'main',
     'src/forcefinder/rattlesnake_control_laws/spr_random_control_law.py',
     'param', "RandomControlSourcePathReceiver -- Carter's SPR law, closed with "
     'match_trace: True (default False). Needs the forcefinder package', False),
    ('random', 'forcefinder_spr_control_law.py', BAND_AVERAGE, 'origin/band-average-study',
     'control_laws/forcefinder_spr_control_law.py',
     'param', 'spr_random_generator -- the Rattlesnake adapter the study runs. '
     'PRIVATE SOURCE, not committed', True),
    ('random', 'band_average_2026_11_04.py', BAND_AVERAGE, 'origin/band-average-study',
     'control_laws/band_average_2026_11_04.py',
     'closed', 'band_average, band_average_generator -- open with '
     'gain_correction: False. PRIVATE SOURCE, not committed', True),
    ('random_not_closed_loop', 'hertzforge_drive_prediction.py', HERTZFORGE, 'main',
     'src/hertzforge/core/drive_prediction.py',
     'open', 'predict_drives, regularized_inverse -- open-loop pre-test '
     'prediction, not a Rattlesnake law and it does not iterate. MIT, '
     'Brandon Zwink and Kevin Cross', False),
]

LOOP_LABEL = {
    'closed': 'closed loop as shipped',
    'param': 'one parameter away from closed',
    'open': 'open loop, no switch',
    'mixed': 'both, in one file',
    'n/a': 'not a law',
}


def git(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args), check=True,
                          capture_output=True).stdout


def source_blob(repo, ref, path):
    return git(repo, 'rev-parse', '%s:%s' % (ref, path)).decode().strip()


def last_commit(repo, ref, path):
    out = git(repo, 'log', '-1', '--format=%ad %h %an', '--date=short',
              ref, '--', path).decode().strip()
    return out or '(no commit touches this path on that ref)'


def repo_label(repo):
    return os.path.basename(repo)


def build(check_only):
    rows, problems = [], []
    for group, name, repo, ref, path, loop, laws, private in LAWS:
        try:
            want = source_blob(repo, ref, path)
        except subprocess.CalledProcessError:
            problems.append('%s: cannot resolve %s:%s in %s'
                            % (name, ref, path, repo_label(repo)))
            continue
        dest_dir = os.path.join(OUT_DIR, group)
        dest = os.path.join(dest_dir, name)
        if check_only:
            if not os.path.exists(dest):
                problems.append('%s: missing from the set' % name)
                continue
            have = git(CONTROLLER, 'hash-object', dest).decode().strip()
            if have != want:
                problems.append('%s: DRIFTED -- set has %s, %s %s has %s'
                                % (name, have[:10], repo_label(repo), ref, want[:10]))
        else:
            os.makedirs(dest_dir, exist_ok=True)
            blob = git(repo, 'show', '%s:%s' % (ref, path))
            with open(dest, 'wb') as fh:
                fh.write(blob)
            have = git(CONTROLLER, 'hash-object', dest).decode().strip()
            if have != want:
                problems.append('%s: wrote %s but source is %s' % (name, have, want))
        rows.append(dict(group=group, name=name, repo=repo_label(repo), ref=ref,
                         path=path, loop=loop, laws=laws, private=private,
                         blob=want[:10], commit=last_commit(repo, ref, path)))
    return rows, problems


def write_manifest(rows):
    lines = [
        '# The random control laws, collected',
        '',
        '*Generated by `control_laws/tests/make_law_set.py`; do not edit by hand.*',
        '*`--check` fails when any file here has drifted from its source repo.*',
        '',
        'Every file was extracted with `git show <ref>:<path>` and its blob hash',
        'verified against the source repo, so these are repo bytes. Nothing was',
        'fetched and no source repo was modified. Edit a law in its own repo and',
        'rebuild; do not edit it here.',
        '',
        '`random/` is flat on purpose -- several laws load a sibling by path, so',
        'splitting the set by loop status would break them. The loop column carries',
        'that information instead.',
        '',
    ]
    for group in ('random', 'random_not_closed_loop'):
        g = [r for r in rows if r['group'] == group]
        if not g:
            continue
        lines += ['## `%s/` -- %d files' % (group, len(g)), '',
                  '| file | loop | laws | source | last commit | blob | in git |',
                  '|---|---|---|---|---|---|---|']
        for r in g:
            lines.append('| `%s` | %s | %s | %s `%s` `%s` | %s | `%s` | %s |' % (
                r['name'], LOOP_LABEL[r['loop']], r['laws'], r['repo'], r['ref'],
                r['path'], r['commit'], r['blob'],
                'NO -- private source' if r['private'] else 'yes'))
        lines.append('')
    priv = [r['name'] for r in rows if r['private']]
    if priv:
        lines += [
            '## Not committed',
            '',
            'These come from `bzwink/rattlesnake-band-average`, which is a private',
            'repo and holds an unreleased study. They are written here so the set is',
            'complete and runnable on this machine, and are listed in `.gitignore` so',
            'they cannot reach this public fork:',
            '',
        ] + ['- `%s`' % n for n in priv] + ['']
    lines += [
        '## Cannot run from this folder alone',
        '',
        '- `spr_random_control_law.py` and `forcefinder_spr_control_law.py` need the',
        '  `forcefinder` package importable.',
        '- `hertzforge_drive_prediction.py` has relative imports and cannot run',
        '  outside the Hertzforge package at all. It is here for reference.',
        '',
    ]
    out = os.path.join(OUT_DIR, 'MANIFEST.md')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='verify the set against its sources; exit 1 on drift')
    args = ap.parse_args()
    rows, problems = build(args.check)
    if args.check:
        for p in problems:
            print('  ' + p)
        print('%d files checked, %d problem(s)' % (len(rows), len(problems)))
        return 1 if problems else 0
    man = write_manifest(rows)
    for r in rows:
        print('  %-38s %-22s %s%s' % (r['name'], r['repo'] + ' ' + r['ref'],
                                      r['blob'],
                                      '   (gitignored)' if r['private'] else ''))
    for p in problems:
        print('  PROBLEM: ' + p)
    print('\nwrote %s' % man)
    print('%d files, %d of them not committed' % (
        len(rows), sum(r['private'] for r in rows)))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
