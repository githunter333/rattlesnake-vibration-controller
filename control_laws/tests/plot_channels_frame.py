"""
Per-channel plots for a frame plant: for every control channel, the target
spec, the final response each optimal diagonal law achieves, and the best
achievable response (achievable_response.achievable_diagonal: the drive, of
any shape, that minimizes the rms dB error over ALL channels jointly at each
line).  It is a joint optimum, not a per-channel bound: it will give up some
dB on one channel to gain more on another, so a law can sit below it on a
channel the law favours (and the laws, minimizing a LINEAR error, favour
different channels than the dB floor does).

Laws and settings, as compare_optdiag_frame.py found best on 10x14:
  SDP   default line budget (20 lines/call)
  fast  full band (max_bins_per_update 400, max_solve_ms_per_update 500)
Both run closed loop for n_cycles with 20-frame CPSD estimates; the plotted
response is the TRUE response to the final drive, H X H^H.

Usage: python control_laws/tests/plot_channels_frame.py [plant] [n_cycles] [out.png]
       defaults: 10x14 15 channels_<plant>.png
Also writes <out>.npz with every curve.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, LAW_DIR)
from achievable_response import achievable_diagonal       # noqa: E402
from compare_optdiag_frame import load, run              # noqa: E402
from frame_plant import frame_plant                      # noqa: E402

# Validated default categorical palette (dataviz skill): slot 1 blue, slot 2 orange.
C_SDP, C_FAST = '#2a78d6', '#eb6834'
C_SPEC, C_BEST = '#1f1f1e', '#a8a7a0'
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'


def main():
    plant = sys.argv[1] if len(sys.argv) > 1 else '10x14'
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    out = sys.argv[3] if len(sys.argv) > 3 else f'channels_{plant}.png'
    p = frame_plant(plant)
    H, f, sel = p['H'], p['f'], p['band']
    spec = np.real(np.einsum('fii->fi', p['spec']))
    laws = dict(load())
    t = time.perf_counter()
    X_sdp = run(p, laws['SDP'], '', n_cycles, 20)['X']
    X_fast = run(p, laws['fast'], ',,400,,,,,,,,,,,,,,,,,,500', n_cycles, 20)['X']
    print(f'laws run in {time.perf_counter() - t:.1f} s')
    t = time.perf_counter()
    best = achievable_diagonal(H, np.where(sel[:, None], spec, 0.0), compute_minimax=False)
    print(f'achievable floor in {time.perf_counter() - t:.1f} s')
    resp = lambda X: np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj()))
    y_sdp, y_fast, y_best = resp(X_sdp), resp(X_fast), best['achieved']
    names = [f"{n}{'xy'[d - 1]}" for n, d in p['controls']]
    np.savez(os.path.splitext(out)[0] + '.npz', f=f, band=sel, spec=spec, sdp=y_sdp, fast=y_fast,
             best=y_best, channels=np.array(names))

    def rms(y, m):
        return np.sqrt(np.mean((10*np.log10(np.maximum(y[sel, m], 1e-300)/spec[sel, m]))**2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(names)
    ncol = 4 if n > 8 else 2
    nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 2.9*nrow + 0.9), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ylo = np.nanmin([np.nanmin(np.where(y[sel] > 0, y[sel], np.nan)) for y in (y_sdp, y_fast, y_best)])
    yhi = np.nanmax([np.nanmax(y[sel]) for y in (y_sdp, y_fast, y_best)])
    ylo, yhi = max(ylo, spec[sel].max()*1e-4)/1.5, yhi*1.5
    for m in range(nrow*ncol):
        ax = axes.flat[m]
        if m >= n:
            ax.axis('off')
            continue
        ax.plot(f[sel], y_best[sel, m], color=C_BEST, lw=4.0, solid_capstyle='round', zorder=1)
        ax.plot(f[sel], spec[sel, m], color=C_SPEC, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.plot(f[sel], y_sdp[sel, m], color=C_SDP, lw=1.5, zorder=3)
        ax.plot(f[sel], y_fast[sel, m], color=C_FAST, lw=1.5, zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(90, 1100)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, which='major', color=GRID, lw=0.8)
        ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, which='both')
        ax.set_title(f'{names[m]}', loc='left', fontsize=10.5, fontweight='bold', color=INK, pad=14)
        ax.text(0.0, 1.015, f'rms error: SDP {rms(y_sdp, m):.1f}  fast {rms(y_fast, m):.1f}  '
                f'best {rms(y_best, m):.1f} dB', transform=ax.transAxes, fontsize=8, color=MUTED, va='bottom')
        if m % ncol == 0:
            ax.set_ylabel('ASD (units²/Hz)', fontsize=8.5, color=MUTED)
        if m // ncol == nrow - 1 or m + ncol >= n:
            ax.set_xlabel('Frequency (Hz)', fontsize=8.5, color=MUTED)
            ax.tick_params(labelbottom=True)
    handles = [Line2D([], [], color=C_SPEC, lw=1.2, ls=(0, (4, 3)), label='Target spec'),
               Line2D([], [], color=C_BEST, lw=4.0, label='Best achievable (least rms dB over all channels jointly)'),
               Line2D([], [], color=C_SDP, lw=1.5, label='SDP law, final'),
               Line2D([], [], color=C_FAST, lw=1.5, label='Fast law, final')]
    fig.legend(handles=handles, loc='upper right', ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.99, 0.995), labelcolor=INK)
    fig.suptitle(f'Frame {plant}: control autospectra after {n_cycles} cycles', x=0.01, y=0.995,
                 ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor='white')
    print('wrote', out)


if __name__ == '__main__':
    main()
