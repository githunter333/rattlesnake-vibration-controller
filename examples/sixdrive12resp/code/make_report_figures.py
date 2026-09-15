"""Figures for the 6-12-8 control-law comparison (12 runs, 2026-09-14).

Colour does one job: it encodes LOOP TYPE (feedback vs open-loop), which is the
distinction the twelve-run matrix turned out to be about.  Cap setting is
carried by position/shape, law identity by direct labels.  Palette slots are
the validated categorical defaults; the scatter uses only the first two, which
clear the all-pairs gate.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

OUT = 'figs'
os.makedirs(OUT, exist_ok=True)

LOOP = {'feedback': '#2a78d6', 'open': '#eb6834'}
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#a8326e']
INK, INK2, INK3, GRID, SURF = '#0b0b0b', '#52514e', '#78766f', '#d8d7d2', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
})

rows = json.load(open('report_data.json'))
by = {r['run']: r for r in rows}
CH = ['7X+', '8X+', '9X+', '10X+', '11X+', '12X+', '13X+', '14X+']
NICE = {'match_trace': 'match trace', 'optimal_diagonal': 'opt diagonal',
        'optimal_diagonal_fast': 'opt diag fast', 'congruence': 'congruence',
        'pseudoinverse': 'pseudoinverse', 'buzz': 'buzz'}
SHORTLAW = {'match_trace': 'mtrace', 'optimal_diagonal': 'optdiag',
            'optimal_diagonal_fast': 'optdiag f', 'congruence': 'congr',
            'pseudoinverse': 'pinv', 'buzz': 'buzz'}
FAMILIES = ['match_trace', 'optimal_diagonal', 'optimal_diagonal_fast',
            'congruence', 'pseudoinverse', 'buzz']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print('  ', name)


def pair(fam):
    on = next(r for r in rows if r['law'] == fam and r['cap'] != 'off')
    off = next(r for r in rows if r['law'] == fam and r['cap'] == 'off')
    return on, off


# --- 1. the cap inverts by family ------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.2))
order = sorted(FAMILIES, key=lambda f: pair(f)[1]['rms'])
for y, fam in enumerate(order):
    on, off = pair(fam)
    c = LOOP[on['loop']]
    ax.plot([on['rms'], off['rms']], [y, y], color=c, lw=2.4, alpha=0.45,
            solid_capstyle='round', zorder=2)
    ax.scatter([on['rms']], [y], s=95, facecolor=SURF, edgecolor=c,
               linewidth=2.2, zorder=3)
    ax.scatter([off['rms']], [y], s=95, color=c, edgecolor=SURF,
               linewidth=1.6, zorder=3)
    d = off['rms'] - on['rms']
    ax.text(max(on['rms'], off['rms']) + 0.22, y, f'{d:+.2f} dB',
            va='center', fontsize=8, color=INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([NICE[f] for f in order])
ax.set_xlabel('overall rms error vs flat specification (dB)  — lower is better')
ax.set_xlim(4.6, 12.2)
ax.grid(axis='x')
ax.invert_yaxis()
h = [plt.Line2D([], [], marker='o', ls='', ms=9, markerfacecolor=SURF,
                markeredgecolor=INK2, markeredgewidth=2, label='coherence cap on (0.95)'),
     plt.Line2D([], [], marker='o', ls='', ms=9, color=INK2,
                markeredgecolor=SURF, label='cap off (1.0)'),
     plt.Line2D([], [], color=LOOP['feedback'], lw=3, label='feedback law'),
     plt.Line2D([], [], color=LOOP['open'], lw=3, label='open-loop law')]
ax.legend(handles=h, frameon=False, fontsize=8, ncol=2, loc='upper right')
ax.set_title('Removing the coherence cap: 2.0–3.5 dB for the open-loop laws, '
             '0.1 dB or less for three of the four feedback laws',
             loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'cap_effect.png')

# --- 2. how much error belongs to the law ----------------------------------
fig, ax = plt.subplots(figsize=(12.4, 4.2))
srt = sorted(rows, key=lambda r: r['ach_rms'])
x = np.arange(len(srt))
ax.bar(x, [r['ach_rms'] for r in srt], 0.6,
       color=[LOOP[r['loop']] for r in srt], edgecolor=SURF, linewidth=1.2,
       label='error against the achievable floor (the law’s own)')
ax.plot(x, [r['rms'] for r in srt], 'D', ms=6, color=INK2,
        markeredgecolor=SURF, markeredgewidth=1.2, ls='none',
        label='total error against the flat specification')
for i, r in enumerate(srt):
    ax.text(i, r['ach_rms'] - 0.32, f"{r['ach_rms']:.2f}", ha='center',
            fontsize=7.4, color=SURF, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{r['run'][3:]}\n{NICE[r['law']]}\ncap {r['cap']}"
                    for r in srt], fontsize=7.2)
ax.set_ylabel('rms error (dB)')
ax.grid(axis='y')
ax.set_ylim(0, max(r['rms'] for r in rows) * 1.22)
ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
ax.set_title('Two separate scorings of the same run\nbar: error against what '
             'this plant can actually deliver.  diamond: error against the flat '
             'specification.\nThey are independent rms figures and do not '
             'subtract — the floor itself is 3.57–3.92 dB rms.',
             loc='left', fontsize=10, color=INK, pad=10)
save(fig, 'error_vs_floor.png')

# --- 3. the two acceptance criteria ----------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.8))
lo, hi = np.log10(0.04), np.log10(6.1)
for r in rows:
    s = 70 + 300 * (np.log10(r['drive_trace']) - lo) / (hi - lo)
    filled = r['cap'] == 'off'
    ax.scatter(r['rms'], r['pout'], s=max(s, 62),
               facecolor=LOOP[r['loop']] if filled else SURF,
               edgecolor=LOOP[r['loop']] if not filled else SURF,
               linewidth=2.0 if not filled else 1.6, alpha=0.9, zorder=3)
    if r['law'] in ('optimal_diagonal', 'optimal_diagonal_fast'):
        continue          # labelled once as a cluster, below
    ax.annotate(f"{r['run'][3:]} {NICE[r['law']]}", (r['rms'], r['pout']),
                textcoords='offset points', xytext=(11, -3), fontsize=7.2,
                color=INK2)
cl = [r for r in rows if r['law'].startswith('optimal_diagonal')]
ax.annotate('03–06  opt diagonal and opt diag fast\n(all four within 0.04 dB rms)',
            (max(r['rms'] for r in cl), np.mean([r['pout'] for r in cl])),
            textcoords='offset points', xytext=(20, 0), fontsize=7.2,
            color=INK2, va='center')
ax.set_xlabel('overall rms error (dB)  — lower is better')
ax.set_ylabel('% of lines outside ±6 dB  — lower is better')
ax.set_xlim(4.6, 12.4)
ax.grid(True)
h = [plt.Line2D([], [], marker='o', ls='', ms=9, color=LOOP['feedback'],
                markeredgecolor=SURF, label='feedback'),
     plt.Line2D([], [], marker='o', ls='', ms=9, color=LOOP['open'],
                markeredgecolor=SURF, label='open-loop'),
     plt.Line2D([], [], marker='o', ls='', ms=9, markerfacecolor=SURF,
                markeredgecolor=INK2, markeredgewidth=2, label='cap on')]
ax.legend(handles=h, frameon=False, fontsize=8, loc='upper left')
ax.set_title('The two acceptance criteria mostly agree now\nmarker area ∝ log '
             'drive power', loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'rms_vs_pout.png')

# --- 4. drive power ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8))
srt = sorted(rows, key=lambda r: r['drive_trace'])
ax.bar(range(len(srt)), [r['drive_trace'] for r in srt], 0.6,
       color=[LOOP[r['loop']] for r in srt], edgecolor=SURF, linewidth=1.2)
for i, r in enumerate(srt):
    ax.text(i, r['drive_trace'] * 1.3, f"{r['drive_trace']:.3g}", ha='center',
            fontsize=7.4, color=INK2)
ax.set_yscale('log')
ax.set_xticks(range(len(srt)))
ax.set_xticklabels([f"{r['run'][3:]}\n{SHORTLAW[r['law']]}\ncap {r['cap']}"
                    for r in srt], fontsize=7.6)
ax.set_ylabel('mean drive trace (V², log scale)')
ax.grid(axis='y')
ax.set_title('Drive power spans a factor of 128 across the twelve runs',
             loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'drive_power.png')

# --- 5. per-channel, each law at its correct cap setting --------------------
BEST = [('match_trace', 'run02'), ('optimal_diagonal', 'run03'),
        ('congruence', 'run08'), ('buzz', 'run12'), ('pseudoinverse', 'run10')]
fig, ax = plt.subplots(figsize=(9.6, 4.0))
xi = np.arange(8)
floor = np.mean([by[r]['reach_mean_db'] for r in by], axis=0)
ax.bar(xi, floor, 0.68, color=INK3, alpha=0.22, edgecolor=SURF, linewidth=1.2,
       label='best achievable (reachability floor)', zorder=1)
for i, (fam, run) in enumerate(BEST):
    v = by[run]['mean_db']
    ax.plot(xi, v, 'o-', color=SER[i], lw=2, ms=7, markeredgecolor=SURF,
            markeredgewidth=1.4, zorder=3)
    dy = {'optimal_diagonal': 7, 'buzz': -10}.get(fam, -2)
    ax.annotate(NICE[fam], (7, v[7]), textcoords='offset points',
                xytext=(9, dy), fontsize=7.6, color=SER[i], weight='bold')
ax.axhline(0, color=INK2, lw=1)
for y in (6, -6):
    ax.axhline(y, color=INK2, lw=1, ls=':')
ax.text(-0.45, 6.3, '±6 dB tolerance', fontsize=7.5, color=INK2)
ax.set_xticks(xi); ax.set_xticklabels(CH)
ax.set_xlim(-0.6, 9.1)
ax.set_ylabel('mean error vs flat spec (dB)'); ax.set_xlabel('control channel')
ax.grid(axis='y')
ax.legend(frameon=False, fontsize=8, loc='lower left')
ax.set_title('The same two channels dominate every law’s error\n13X+ and 14X+ '
             'are the reachability-limited pair; each law is shown at its own '
             'best cap setting', loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'per_channel_mean.png')

# --- 6. system characterisation --------------------------------------------
sysfile = ('../results/case/sdynpy_frame6x12_system.npz')
d = np.load(sysfile, allow_pickle=True)
M, K, Cm = d['mass'], d['stiffness'], d['damping']
w2, phi = eigh(K, M)
w = np.sqrt(np.maximum(w2, 0)); fn = w / (2 * np.pi)
zeta = np.diag(phi.T @ Cm @ phi) / (2 * np.maximum(w, 1e-12))
keep = (fn >= 100) & (fn <= 1000)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
a = axes[0]
a.hist(zeta[keep] * 100, bins=12, color=SER[0], edgecolor=SURF)
a.set_xlabel('modal damping ζ (%)'); a.set_ylabel('modes'); a.grid(axis='y')
a.set_title(f'{keep.sum()} modes in 100–1000 Hz\nζ = {zeta[keep].min()*100:.2f}–'
            f'{zeta[keep].max()*100:.2f} %, median {np.median(zeta[keep])*100:.2f} %',
            loc='left', fontsize=9.5, pad=8)

a = axes[1]
xr = np.arange(len(rows))
a.bar(xr - 0.19, [r['reach_rms'] for r in rows], 0.36, color=SER[0],
      edgecolor=SURF, linewidth=1.1, label='at rcond 1e-3 (what the laws use)')
a.bar(xr + 0.19, [r['rank_rms'] for r in rows], 0.36, color=INK3, alpha=0.45,
      edgecolor=SURF, linewidth=1.1, label='unrestricted, over the real rank 5')
a.set_xticks(xr)
a.set_xticklabels([r['run'][3:] for r in rows], fontsize=7.4)
a.set_ylabel('reachability floor (dB rms)'); a.set_xlabel('run')
a.set_ylim(0, 5)
a.grid(axis='y')
a.legend(frameon=False, fontsize=7.2, loc='upper left')
a.set_title('The floor is a property of the plant:\nrcond 1e-3 costs nothing '
            '(≤0.21 dB)', loc='left', fontsize=9.5, pad=8)

a = axes[2]
a.bar(range(len(rows)), [r['cond'] for r in rows], 0.6,
      color=[LOOP[r['loop']] for r in rows], edgecolor=SURF, linewidth=1.2)
a.set_yscale('log')
a.set_xticks(range(len(rows)))
a.set_xticklabels([r['run'][3:] for r in rows], fontsize=7.4)
a.set_ylabel('median cond(H)'); a.set_xlabel('run')
a.grid(axis='y')
a.set_title('Identification scatter: same plant,\nsame settings, factor of ten',
            loc='left', fontsize=9.5, pad=8)
save(fig, 'system_characterisation.png')
print('done')
