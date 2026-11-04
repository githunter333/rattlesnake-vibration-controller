"""
Requested / best achievable / achieved control levels at three representative
channels of the band-average 12-drive RC plate.

One figure, three stacked subplots -- a typical X, Y and Z control channel --
each showing four curves as ASD against frequency on log-log axes:

  requested        the reconstructed BARC specification, with its +/-3 dB
                   warning band shaded and its +/-6 dB abort limits dotted
  best achievable  max over X >= 0 of diag(H X H^H) against that spec, from
                   control_laws/achievable_response.achievable_diagonal on the
                   EXACT plant -- the reachability floor, independent of any
                   control law.  Computed on log-spaced lines (it costs about
                   2 s a line), so it is drawn as markers, not a dense curve.
  achieved         the converged response of optimal_diagonal_control_fast
                   with the 2026-11-04 structural-floor rule
  achieved, pre-fix  the same law at --before-rev, whose largest-gap rule kept
                   2 of 12 drive directions (dashed, de-emphasized)

Everything is scored against the EXACT plant while the law is handed the
simulated 50-average H1 estimate, exactly as optdiag_on_rc_plate.py does -- see
that script's "a MEASURED FRF, not an exact one" note for why that matters.

CHOOSING THE CHANNELS.  `--channels auto` (the default) picks, within each axis,
the channel whose specification ASD is closest in SHAPE to that axis's median
ASD curve: smallest median |dB deviation from the per-axis median curve| over
in-band lines.  Shape rather than overall RMS, because these are log-log ASD
plots and a channel can sit at a typical level while having an untypical
spectrum.  The criterion reads only the SPECIFICATION, never any result, so it
cannot be steered by which channel happens to make the law look good.

Usage (from the repo root, in the 'rattlesnake' conda env):
    python control_laws/tests/plot_optdiag_on_rc_plate.py
    python control_laws/tests/plot_optdiag_on_rc_plate.py --channels 101Z+,104Y+,105X+
    python control_laws/tests/plot_optdiag_on_rc_plate.py --floor-lines 60   (quick look)

About 9 minutes at the defaults: two 15-cycle closed-loop runs plus the floor.

The pre-fix curve needs the law as it was before the fix.  It is written to
control_laws/_prefix_law_<pid>.py (the law resolves its base class as a sibling,
so it has to sit in that directory) and removed again on the way out.
"""
import argparse
import contextlib
import io
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(LAW_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'src'))
import optdiag_on_rc_plate as D                                    # noqa: E402
from rattlesnake.utilities import load_control_module              # noqa: E402

FULLBAND = 'max_bins_per_update: 400\nmax_solve_ms_per_update: 500'
BEFORE_REV = '3a68a9ba'          # the last commit before the structural-floor fix
WARN_DB, ABORT_DB = 3.0, 6.0


def axis_of(name):
    """'X' from '105X+'."""
    return str(name)[-2]


def pick_channels(p):
    """One channel per axis: the one whose spec ASD is closest in shape to that
    axis's median ASD curve.  Returns [(index, name, deviation_db)] and, for the
    record, the whole ranking per axis."""
    names = [str(c) for c in p['controls']]
    ls = 10*np.log10(np.maximum(p['spec_asd'][p['band']], 1e-300))
    out, ranking = [], {}
    for ax in 'XYZ':
        cols = [i for i, nm in enumerate(names) if axis_of(nm) == ax]
        med = np.median(ls[:, cols], axis=1)
        dev = np.median(np.abs(ls[:, cols] - med[:, None]), axis=0)
        order = np.argsort(dev)
        ranking[ax] = [(cols[k], names[cols[k]], float(dev[k])) for k in order]
        out.append(ranking[ax][0])
    return out, ranking


def response_asd(H, X):
    """diag(H X H^H) -- the control autospectra a drive CPSD produces."""
    return np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj()))


def prefix_law(rev):
    """The law as of `rev`, loaded from a sibling copy in control_laws/."""
    path = os.path.join(LAW_DIR, f'_prefix_law_{os.getpid()}.py')
    src = subprocess.run(['git', '-C', REPO, 'show',
                          f'{rev}:control_laws/optimal_diagonal_control_fast.py'],
                         capture_output=True, text=True, check=True).stdout
    with io.open(path, 'w', encoding='utf-8') as fh:
        fh.write(src)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod = load_control_module(path)
        return mod.optimal_diagonal_control_fast, path
    except Exception:
        os.remove(path)
        raise


def achievable(p, n_lines, seed=0):
    """Reachability floor on log-spaced in-band lines: the best diag(H X H^H)
    any drive can produce against this spec, on the exact plant."""
    ar = load_control_module(os.path.join(LAW_DIR, 'achievable_response.py'))
    idx = np.flatnonzero(p['band'])
    f = p['f'][idx]
    # log-spaced, because the figure's x axis is logarithmic
    want = np.geomspace(f[0], f[-1], min(n_lines, idx.size))
    take = np.unique(np.searchsorted(f, want).clip(0, idx.size - 1))
    lines = idx[take]
    with contextlib.redirect_stdout(io.StringIO()):
        out = ar.achievable_diagonal(p['H'], p['spec_asd'], line_indices=lines, seed=seed)
    return lines, out


def rms_db(achieved, spec, band, ch):
    """rms dB error of one channel over the specified lines."""
    e = 10*np.log10(np.maximum(achieved[band, ch], 1e-300)/spec[band, ch])
    return float(np.sqrt(np.mean(e**2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-average', default=D.DEFAULT_BAND_AVERAGE)
    ap.add_argument('--df', type=float, default=1.0, help='line spacing, Hz (default: %(default)s)')
    ap.add_argument('--cycles', type=int, default=15)
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--floor-lines', type=int, default=180,
                    help='lines to compute the reachability floor on (~2 s each)')
    ap.add_argument('--channels', default='auto',
                    help="'auto', or a comma-separated list like 105X+,104Y+,109Z+")
    ap.add_argument('--before-rev', default=BEFORE_REV,
                    help='git revision for the pre-fix curve; "none" to omit it')
    ap.add_argument('--out', default=os.path.join(LAW_DIR, 'notes'),
                    help='directory for the PNG (default: %(default)s)')
    ap.add_argument('--name', default='optdiag_rc_plate_levels_2026-11-04.png')
    ap.add_argument('--dpi', type=int, default=130)
    args = ap.parse_args()

    p = D.rc_plate_plant(band_average=args.band_average, df=args.df)
    print(D.summary(p), flush=True)
    names = [str(c) for c in p['controls']]
    band, f, spec = p['band'], p['f'], p['spec_asd']

    auto, ranking = pick_channels(p)
    if args.channels == 'auto':
        chosen = auto
    else:
        chosen = []
        for nm in args.channels.split(','):
            nm = nm.strip()
            if nm not in names:
                ap.error(f'unknown channel {nm!r}; have {", ".join(names)}')
            i = names.index(nm)
            dev = dict((n, d) for _, n, d in ranking[axis_of(nm)])[nm]
            chosen.append((i, nm, dev))
    for ax in 'XYZ':
        line = ', '.join(f'{n} {d:.2f}' for _, n, d in ranking[ax][:4])
        print(f'  {ax} most typical (name, dev dB): {line}', flush=True)
    print('  plotting: ' + ', '.join(f'{n} (dev {d:.2f} dB)' for _, n, d in chosen), flush=True)

    print('\nclosed loop, post-fix law...', flush=True)
    r_after = D.run(p, load_control_module(
        os.path.join(LAW_DIR, 'optimal_diagonal_control_fast.py')).optimal_diagonal_control_fast,
        FULLBAND, args.cycles, args.frames)
    y_after = response_asd(p['H'], r_after['X'])

    y_before, r_before = None, None
    if args.before_rev.lower() != 'none':
        print(f'closed loop, law at {args.before_rev}...', flush=True)
        cls_before, tmp = prefix_law(args.before_rev)
        try:
            r_before = D.run(p, cls_before, FULLBAND, args.cycles, args.frames)
        finally:
            os.remove(tmp)
        y_before = response_asd(p['H'], r_before['X'])

    print(f'reachability floor on ~{args.floor_lines} lines...', flush=True)
    lines, flo = achievable(p, args.floor_lines)
    y_floor = flo['achieved']
    print(f'  solved {flo["n_lines_solved"]} lines; rms dB error over all 48 channels: '
          f'median {np.nanmedian(flo["best_rms_db"][lines]):.2f}, '
          f'max {np.nanmax(flo["best_rms_db"][lines]):.2f}', flush=True)

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10.5, 4.0*len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)
    inb = np.flatnonzero(band)
    for ax_, (ci, nm, dev) in zip(axes, chosen):
        s = spec[inb, ci]
        ax_.fill_between(f[inb], s*10**(-WARN_DB/10), s*10**(WARN_DB/10),
                         color='0.85', label=f'+/-{WARN_DB:g} dB warning band')
        ax_.plot(f[inb], s*10**(-ABORT_DB/10), ':', color='0.55', lw=1.0)
        ax_.plot(f[inb], s*10**(ABORT_DB/10), ':', color='0.55', lw=1.0,
                 label=f'+/-{ABORT_DB:g} dB abort limit')
        ax_.plot(f[inb], s, '-', color='k', lw=1.8, label='requested (BARC spec)')
        if y_before is not None:
            e = rms_db(y_before, spec, band, ci)
            ax_.plot(f[inb], y_before[inb, ci], '--', color='tab:red', lw=1.0, alpha=0.55,
                     label=f'achieved, pre-fix ({e:.2f} dB rms)')
        e_after = rms_db(y_after, spec, band, ci)
        ax_.plot(f[inb], y_after[inb, ci], '-', color='tab:blue', lw=1.1,
                 label=f'achieved, post-fix ({e_after:.2f} dB rms)')
        good = lines[np.isfinite(y_floor[lines, ci]) & (y_floor[lines, ci] > 0)]
        ef = 10*np.log10(np.maximum(y_floor[good, ci], 1e-300)/spec[good, ci])
        ax_.plot(f[good], y_floor[good, ci], 'o', color='tab:green', ms=3.4,
                 label=f'best achievable ({np.sqrt(np.mean(ef**2)):.2f} dB rms, '
                       f'{good.size} lines)')
        ax_.set(xscale='log', yscale='log', ylabel='ASD (g$^2$/Hz)')
        ax_.set_xlim(f[inb][0], f[inb][-1])
        ax_.set_title(f'{nm}  --  spec RMS {np.sqrt(s.sum()*p["df"]):.1f} g, '
                      f'shape deviation {dev:.2f} dB from the {axis_of(nm)} median',
                      fontsize=10)
        ax_.grid(True, which='both', alpha=0.25)
        ax_.legend(fontsize=7.5, loc='lower left', ncol=2, framealpha=0.9)
    axes[-1].set_xlabel('frequency (Hz)')
    fig.suptitle('optimal_diagonal_control_fast on the 12-drive RC plate: requested, '
                 'best achievable, achieved\n'
                 f'{int(band.sum())} specified lines at {p["df"]:g} Hz, {args.cycles} cycles, '
                 'full-band budget, measured 50-average FRF; scored against the exact plant',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name)
    fig.savefig(path, dpi=args.dpi)
    print(f'\nwrote {path}', flush=True)

    print('\nper-channel rms dB error over the specified lines:')
    print(f"  {'channel':9s} {'requested RMS':>13s} {'pre-fix':>8s} {'post-fix':>9s} {'floor':>7s}")
    for ci, nm, _ in chosen:
        fl = lines[np.isfinite(y_floor[lines, ci])]
        ef = 10*np.log10(np.maximum(y_floor[fl, ci], 1e-300)/spec[fl, ci])
        pre = f'{rms_db(y_before, spec, band, ci):8.2f}' if y_before is not None else f'{"-":>8s}'
        print(f"  {nm:9s} {np.sqrt(spec[band, ci].sum()*p['df']):12.1f} g {pre} "
              f"{rms_db(y_after, spec, band, ci):9.2f} {np.sqrt(np.mean(ef**2)):7.2f}")


if __name__ == '__main__':
    main()
