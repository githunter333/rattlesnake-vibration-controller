"""Figures for modal_shift_and_damping.py / verify_modal_shift_damping.py."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'figures', 'report3wk'))
os.makedirs(OUT, exist_ok=True)

C = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False})

d = json.load(open(os.path.join(N, 'modal_shift_and_damping.json')))
soft = np.array(d['modes']['softened'], bool)
lv = np.array([r['level_db'] for r in d['levels']], float)
sh = np.array([r['shift_pct'] for r in d['levels']])[:, soft]
zr = np.array([r['zeta_ratio'] for r in d['levels']])[:, soft]
zs = np.array([r['zeta_from_softening'] for r in d['levels']])[:, soft]

# ---- FIG 8: how far the plant moves, per level --------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                             gridspec_kw={'hspace': 0.28})

a1.fill_between(lv, sh.min(1), sh.max(1), color=C[0], alpha=0.16, lw=0,
                label='spread across the 27 softened modes')
a1.plot(lv, np.median(sh, 1), 'o-', color=C[0], lw=2, ms=6,
        label='median softened mode', zorder=3)
a1.set_yscale('log'); a1.set_ylabel('frequency shift\n(% below linear)')
med = np.median(sh, 1)
for x in (0.0, 4.0):
    y = med[list(lv).index(x)]
    a1.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(-16, 4),
                ha='right', fontsize=8, color=INK2)
a1.annotate('the 6 modes below 300 Hz: identically 0 at every level',
            xy=(-18, 1), xytext=(-17.6, 0.0035), fontsize=7.8, color=INK2)
a1.set_ylim(2e-3, 90)
a1.set_title('Frequency — 1% was the design target at 0 dB, and 0 dB lands at 0.995%',
             fontsize=9, color=INK, pad=6, loc='left')
a1.legend(fontsize=8, loc='upper left')

zr1, zs1 = zr - 1.0, zs - 1.0
a2.fill_between(lv, zr1.min(1), zr1.max(1), color=C[0], alpha=0.16, lw=0,
                label='spread across the 27 softened modes')
a2.plot(lv, np.median(zr1, 1), 'o-', color=C[0], lw=2, ms=6,
        label='median, total', zorder=3)
a2.plot(lv, np.median(zs1, 1), 's--', color=C[1], lw=2, ms=5,
        label='of which $1/\\sqrt{g}$ — softening alone, no $c_2$ term', zorder=3)
a2.axhline(0.10, color=INK2, lw=1.0, ls=':')
a2.text(-17.6, 0.115, 'design intent: +0.10 at the design amplitude',
        fontsize=7.6, color=INK2)
a2.set_yscale('log')
a2.set_ylabel('damping increase\n$\\zeta/\\zeta_0 - 1$')
a2.set_xlabel('test level (dB re full level)'); a2.set_xticks(lv)
a2.set_ylim(8e-5, 4.0)
for x in (0.0,):
    y = np.median(zr1, 1)[list(lv).index(x)]
    a2.annotate(f'+{y:.3f}', (x, y), textcoords='offset points', xytext=(6, -13),
                ha='left', fontsize=8, color=INK2)
a2.set_title('Damping — +0.124 at 0 dB, not the +0.10 intended',
             fontsize=9, color=INK, pad=6, loc='left')
a2.legend(fontsize=8, loc='lower right')
fig.suptitle('How far the bounded-softening plant actually moves, per mode and per level',
             fontsize=9.8, color=INK, x=0.012, ha='left', y=0.985)
fig.savefig(os.path.join(OUT, 'fig8_modal_shift_damping.png'), dpi=200,
            bbox_inches='tight')
plt.close(fig)

# ---- FIG 9: equivalent-linear vs what a modal fit would read ------------
w = json.load(open(os.path.join(N, 'modal_shift_damping_white_+0dB.json')))['modes']
sm = np.array([m['shift_meas'] for m in w])
ap = np.array([m['apparent_over_true'] for m in w])
sp = np.array([m['shift_pred_at_meas_sigma'] for m in w])

fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
fig.subplots_adjust(top=0.78, wspace=0.34)
b1.plot([0, 3], [0, 3], color=INK2, lw=1.0, ls=':', zorder=1)
b1.plot(sp, sm, 'o', color=C[0], ms=8, zorder=3)
b1.set_xlabel('predicted, equivalent-linear  (%)')
b1.set_ylabel('measured from the FRF peak  (%)')
b1.set_xlim(0, 3.1); b1.set_ylim(0, 3.1)
b1.text(2.95, 2.75, '1:1', fontsize=8, color=INK2, ha='right')
b1.set_title('Frequency shift: the peak moves\nless than the equivalent-linear model',
             fontsize=9, color=INK, pad=6, loc='left')

b2.axhline(1.0, color=INK2, lw=1.0, ls=':', zorder=1)
b2.plot(sm, ap, 'o', color=C[1], ms=8, zorder=3)
b2.set_xlabel('frequency shift measured  (%)')
b2.set_ylabel('apparent $\\zeta$ / true $\\zeta$')
b2.set_ylim(0.95, 1.8)
b2.set_title('Damping: a half-power fit reads high,\nand the error grows with the shift',
             fontsize=9, color=INK, pad=6, loc='left')
fig.suptitle('Nine softened modes, direct time integration — the two answers differ, '
             'and both are correct',
             fontsize=9.8, color=INK, x=0.012, ha='left', y=1.02)
fig.savefig(os.path.join(OUT, 'fig9_equivalent_vs_apparent.png'), dpi=200,
            bbox_inches='tight')
plt.close(fig)
print('wrote fig8_modal_shift_damping.png, fig9_equivalent_vs_apparent.png')
