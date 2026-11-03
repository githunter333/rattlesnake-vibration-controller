"""
Response spectral densities of the three optimal diagonal laws on a frame plant:
the target spec, the best achievable spec, and what the law produced.

    python control_laws/tests/plot_law_spectra.py [plant] [n_cycles] [n_frames] [outdir]
    defaults: 10x14 15 20 examples/tendrive14resp/results/plots

One PNG per law, one panel per control channel (file names: see SLUGS and plot_path):
  target spec      dashed, the specification autospectrum
  best achievable the best a law could do, per line, with the channels the SDP and fast laws
                   reach (11 of 14 on 10x14) HELD ON TARGET and the rest (7x, 13x, 14x on 10x14)
                   as close as they can then get, in rms dB.  reference_spectra() solves it with
                   achievable_response's dB residuals, weighted 100:1 toward the reachable channels.
                   The key also quotes the JOINT optimum, achievable_response.achievable_diagonal:
                   the best rms dB over all channels at once, which trades the reachable channels
                   off against the others and so is not a floor for them
  control          the TRUE response autospectrum diag(H X H^H) after the last cycle

Each law runs its defaults (band_average: df 5, octave order 6, drive RMS limit off), closed loop, as in
compare_optdiag_frame.py (the FRF is held fixed, responses are 'measured' from n_frames Gaussian frames).  The octave law is switched
into octave mode with set_test_level_db(0), since nothing in Rattlesnake does that yet.

Run in the 'rattlesnake' conda env with PYTHONPATH=src.
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import compare_optdiag_frame as cf                     # noqa: E402
from frame_plant import frame_plant                    # noqa: E402
import achievable_response as ar                       # noqa: E402
from achievable_response import achievable_diagonal    # noqa: E402
from scipy.optimize import least_squares               # noqa: E402
from rattlesnake.utilities import load_control_module  # noqa: E402

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
GRID_MAJOR, FRAME = '#c9c8c0', '#6b6a66'            # stronger major gridlines; the panel outline
C_ACHIEVABLE, C_CONTROL = '#2a78d6', '#eb6834'          # categorical slots 1 and 2
Y_LO, Y_HI = 1e-5, 1e-1                                  # +-20 dB around the 1e-3 spec

# df = the plant's line spacing; drive_rms_limit off because the plant's drive units are arbitrary
BAND_PARAMS = 'df: 5\noct_order: 6\ndrive_rms_limit: none'
OCTAVE_PARAMS = 'frequency_spacing: 5\nswitch_level_db: 0\noctave_fraction: 6\noctave_fmin: 100\noctave_fmax: 1000'


def laws():
    out = dict(cf.load())
    with contextlib.redirect_stdout(io.StringIO()):
        mod = load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))

    class OctaveEngaged(mod.octave_band_switching_control):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.set_test_level_db(0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        band = load_control_module(os.path.join(cf.LAW_DIR, 'band_average.py')).band_average
    return [('SDP law (optimal_diagonal_control)', out['SDP'], ''),
            ('FAST law (optimal_diagonal_control_fast)', out['fast'], ''),
            ('OCTAVE law (octave_band_switching_control): 1/6-octave bands, mean-of-H averaging',
             OctaveEngaged, OCTAVE_PARAMS),
            ('BAND-AVERAGE law (band_average): 1/6-octave bands, exact cross-power averaging', band, BAND_PARAMS),
            ('BAND-AVERAGE law (band_average), drive coherence capped at 0.95', band,
             BAND_PARAMS + '\nmax_drive_coherence: 0.95')]


# File-name stem of each entry of laws(), in order; a plot is spectra_<plant>_<stem>_<view>.png
# with view 'lines' (every line) or 'bands' (1/6-octave band averages).
SLUGS = ['sdp_law', 'fast_law', 'octave_law', 'band_average_law', 'band_average_law_capped']


def plot_path(outdir, plant, slug, view):
    return os.path.join(outdir, f'spectra_{plant}_{slug}_{view}.png')


def _held_line(Hb, y, reach, weight, max_nfev=300):
    """One line: diag(H X H^H) in rms dB, the `reach` channels weighted `weight` to 1."""
    from_L = lambda v: ar._unpack(v, Hb.shape[1], Hb.shape[1])
    N = Hb.shape[1]
    y_log = np.log(np.maximum(y, 1e-300))
    w = np.where(reach, weight, 1.0)
    fun = lambda v: w*ar._resid(v, Hb, y_log, N, N)
    jac = lambda v: w[:, None]*ar._jac(v, Hb, y_log, N, N)
    best = None
    for L0 in (ar._init_pinv(Hb[reach], y[reach], 1e-8), ar._init_pinv(Hb, y, 1e-8)):
        sol = least_squares(fun, ar._pack(L0), jac=jac, method='trf', max_nfev=max_nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    return ar._diag_response(Hb, from_L(best.x))[0]


def reference_spectra(p, lost, weight=100.0):
    """(held, joint), each (F, M), NaN off the specified band.

    held   best achievable with the reachable channels (~lost) held on target (weighted `weight`:1,
           which keeps them within 0.01 dB) and the others as close as they can then get
    joint  achievable_response.achievable_diagonal: best rms dB over all channels at once"""
    H, sel = p['H'], p['band']
    spec = np.real(np.einsum('fii->fi', p['spec']))
    idx = np.flatnonzero(sel)
    held = np.full(spec.shape, np.nan)
    for i in idx:
        held[i] = _held_line(H[i], spec[i], ~lost, weight)
    joint = achievable_diagonal(H, spec, line_indices=idx, compute_minimax=False)['achieved']
    return held, joint


def response(H, X):
    return np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj()))


def panel(ax, f, target, best, got, label, given_up, err_db, steps=False):
    ax.set_facecolor(SURFACE)
    ds = 'steps-mid' if steps else 'default'
    ax.semilogy(f, best, color=C_ACHIEVABLE, lw=2.8, zorder=2, solid_capstyle='round', drawstyle=ds)
    ax.semilogy(f, got, color=C_CONTROL, lw=1.4, zorder=3, drawstyle=ds)
    ax.semilogy(f, target, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4, drawstyle=ds)
    lo = min(Y_LO, 0.7*min(got.min(), best.min()))
    hi = max(Y_HI, 1.5*max(got.max(), best.max()))
    ax.set_ylim(lo, hi)                                  # one window for every panel (+-20 dB), wider only if needed
    ax.set_xlim(f[0], f[-1])
    ax.grid(True, which='major', color=GRID_MAJOR, lw=0.7)
    ax.grid(True, which='minor', axis='y', color=GRID, lw=0.4, ls=':')
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    ax.grid(True, which='minor', axis='x', color=GRID, lw=0.4, ls=':')
    ax.set_axisbelow(True)
    for s in ax.spines.values():                        # a full frame around every panel
        s.set_visible(True)
        s.set_color(FRAME)
        s.set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=7, length=2.5, direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=1.5, labelleft=False, labelbottom=False)
    ax.set_title(label + ('  (given up by SDP/fast)' if given_up else ''), loc='left', fontsize=9, color=INK, pad=3)
    ax.set_title(f'{err_db:.2f} dB rms', loc='right', fontsize=8, color=INK2, pad=3)


def figure(name, p, X, best, lost, names, plant, outpath, bands=None, joint=None):
    """bands = (masks, centers): plot the 1/6-octave band AVERAGES (power mean over each band's lines)
    instead of every line."""
    f, sel = p['f'], p['band']
    spec_full = np.real(np.einsum('fii->fi', p['spec']))
    if bands is None:
        fb, grid = f[sel], (lambda a: a[sel])
    else:
        masks, fb = bands
        grid = lambda a: np.array([a[m].mean(axis=0) for m in masks])
    spec, got, bst = grid(spec_full), grid(response(p['H'], X)), grid(best)
    jnt = None if joint is None else grid(joint)
    err = np.sqrt(np.mean((10*np.log10(got/spec))**2, axis=0))
    M = spec.shape[1]
    ncol = 3
    nrow = int(np.ceil((M + 1)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.05*nrow + 0.9), sharex=True, facecolor=SURFACE)
    axes = axes.ravel()
    for m in range(M):
        panel(axes[m], fb, spec[:, m], bst[:, m], got[:, m], f'control {names[m]}', lost[m], err[m], steps=bands is not None)
    for ax in axes[M:]:
        ax.axis('off')
    for ax in axes[-ncol:]:
        if ax.axison:
            ax.set_xlabel('frequency (Hz)', fontsize=8, color=INK2)
    key = axes[M]                                        # first spare cell holds the key
    key.axis('off')
    handles = [plt.Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3))),
               plt.Line2D([], [], color=C_ACHIEVABLE, lw=2.6),
               plt.Line2D([], [], color=C_CONTROL, lw=1.1)]
    key.legend(handles, ['target spec', 'best achievable (reachable channels on target)', 'control (this law)'], loc='upper left',
               frameon=False, fontsize=9, labelcolor=INK)
    reach = ~lost
    rms_db = lambda y, cols: np.sqrt(np.mean((10*np.log10(y[:, cols]/spec[:, cols]))**2))
    everyone = np.ones(M, dtype=bool)
    joint_txt = '' if jnt is None else f'\n  (joint 14-channel compromise: {rms_db(jnt, everyone):.2f} all, {rms_db(jnt, reach):.2f} reachable)'
    key.text(0.0, 0.46,
             'rms error vs the spec (dB)\n'
             f'  all {M} channels:   this law {rms_db(got, everyone):.2f},  best achievable {rms_db(bst, everyone):.2f}\n'
             f'  {int(reach.sum())} reachable:      this law {rms_db(got, reach):.2f},  best achievable {rms_db(bst, reach):.2f}' + joint_txt,
             transform=key.transAxes, fontsize=8.5, color=INK2, va='top', linespacing=1.5)
    fig.supylabel('response autospectrum (units$^2$/Hz)', fontsize=9, color=INK2, x=0.005)
    view = f"every {p['df']:g} Hz line" if bands is None else '1/6-octave band averages'
    fig.text(0.015, 0.995, name, fontsize=12, color=INK, ha='left', va='top')
    fig.text(0.015, 0.972, f'{plant} plant  |  view: {view}', fontsize=9.5, color=INK2, ha='left', va='top')
    fig.tight_layout(rect=(0.01, 0, 1, 0.955))
    fig.savefig(outpath, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def score_table(p, runs, lost, spec_d, refs=()):
    """Printed table: rms dB error per law, per line and per 1/6-octave band (100-1000 Hz)."""
    f, H, sel = p['f'], p['H'], p['band']
    with contextlib.redirect_stdout(io.StringIO()):
        mod = load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))
    _, lo, hi = mod._octave_band_frequencies(100.0, 1000.0, 6)
    masks = [m for m in ((f >= l) & (f < h) & sel for l, h in zip(lo, hi)) if m.any()]   # in-band lines only
    reach, everyone = ~lost, np.ones(lost.size, dtype=bool)
    rms = lambda e, cols: float(np.sqrt(np.mean(e[:, cols]**2)))
    print(f'\nrms dB error vs spec: {len(masks)} bands of 1/6 octave, {int(sel.sum())} lines')
    print(f"{'':72s} {'line, 11 reach':>14s} {'line, all 14':>13s} {'band, 11 reach':>15s} {'band, all 14':>13s}")
    for label, X in runs:
        y = response(H, X)
        line = 10*np.log10(y[sel]/spec_d[sel])
        band = np.array([10*np.log10(y[m].mean(0)/spec_d[m].mean(0)) for m in masks])
        print(f'{label:72s} {rms(line, reach):14.2f} {rms(line, everyone):13.2f} {rms(band, reach):15.2f} {rms(band, everyone):13.2f}')
    for label, y in refs:                                 # reference spectra scored the same way
        line = 10*np.log10(y[sel]/spec_d[sel])
        band = np.array([10*np.log10(y[m].mean(0)/spec_d[m].mean(0)) for m in masks])
        print(f'{label:72s} {rms(line, reach):14.2f} {rms(line, everyone):13.2f} {rms(band, reach):15.2f} {rms(band, everyone):13.2f}')


def main():
    plant = sys.argv[1] if len(sys.argv) > 1 else '10x14'
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    root = os.path.dirname(os.path.dirname(HERE))
    outdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(root, 'examples', 'tendrive14resp', 'results', 'plots')
    os.makedirs(outdir, exist_ok=True)
    p = frame_plant(plant)
    names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
    spec_d = np.real(np.einsum('fii->fi', p['spec']))
    runs = []
    for label, cls, params in laws():
        print('running', label, flush=True)
        runs.append((label, cf.run(p, cls, params, n_cycles, n_frames)['X']))
    sel = p['band']
    final = np.array([np.sqrt(np.mean((10*np.log10(response(p['H'], X)[sel]/spec_d[sel]))**2, axis=0)) for _, X in runs])
    lost = np.all(final[:2] > 3.0, axis=0)               # channels the SDP and fast laws both give up
    print('best achievable references ...', flush=True)
    best, joint = reference_spectra(p, lost)
    score_table(p, runs, lost, spec_d, refs=[('best achievable (reachable held on target)', best),
                                               ('joint 14-channel compromise', joint)])
    for (label, X), short in zip(runs, SLUGS):
        path = plot_path(outdir, plant, short, 'lines')
        figure(label, p, X, best, lost, names, plant, path, joint=joint)
        print('wrote', path)


if __name__ == '__main__':
    main()
