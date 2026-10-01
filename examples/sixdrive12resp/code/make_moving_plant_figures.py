# -*- coding: utf-8 -*-
"""Score every saved spectrum of the four moving-plant configurations.

The definition is validated against the stored runs_51_52_55_optdiag.json
values to three decimals: coherent-column rms is the per-channel rms of
10*log10(gamma^2 * S_yy / S_spec) over the in-band lines.
"""
import json, os
import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
NOTES = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
GROUP = 'Frame 6x12 Random'

CONFIGS = {
 'mt_off': ('match_trace, refresh OFF', 'run49_matchtrace_bsoft_norefresh_spec%s.nc4',
            [(-18, '_m18'), (-12, '_m12'), (-6, '_m06'), (0, ''),
             (2, '_p02'), (4, '_p04'), (8, '_p08')]),
 'mt_on':  ('match_trace, refresh ON',  'run50_matchtrace_bsoft_refresh005_spec%s.nc4',
            [(-18, '_m18'), (-12, '_m12'), (-6, '_m06'), (0, '')]),
 'od_off': ('optimal_diagonal, trim OFF', 'run51_optdiag_bsoft_trimoff_spec%s.nc4',
            [(-18, '_m18'), (0, ''), (4, '_p04')]),
 'od_on':  ('optimal_diagonal, trim ON',  'run55_optdiag_bsoft_trimon_fixed_spec%s.nc4',
            [(-18, '_m18'), (-12, '_m12'), (-6, '_m06'), (-3, '_m03'), (0, '')]),
}


def score(path):
    ds = netCDF4.Dataset(path)
    g = ds[GROUP]
    S = np.array(g['response_cpsd_real'][:])
    P = np.array(g['specification_cpsd_matrix_real'][:])
    c = np.clip(np.array(g['frf_coherence'][:]), 0.0, 1.0)
    D = np.array(g['drive_cpsd_real'][:])
    ds.close()
    ib = np.einsum('fii->fi', P).max(axis=1) > 0
    Sd, Pd, cd = np.einsum('fii->fi', S)[ib], np.einsum('fii->fi', P)[ib], c[ib]
    e_raw = 10 * np.log10(Sd / Pd)
    e_coh = 10 * np.log10(np.maximum(cd, 1e-12) * Sd / Pd)
    return dict(rms_raw=float(np.sqrt((e_raw ** 2).mean())),
                rms_coh=float(np.sqrt((e_coh ** 2).mean())),
                pct_out=float(100 * np.mean(np.abs(e_raw) > 3.0)),
                level=float(10 * np.log10(Sd.sum() / Pd.sum())),
                coh=float(np.median(cd)),
                drive=float(np.sqrt(np.trace(D, axis1=1, axis2=2).real.sum())),
                ch_raw=np.sqrt((e_raw ** 2).mean(axis=0)).tolist(),
                ch_coh=np.sqrt((e_coh ** 2).mean(axis=0)).tolist())


out = {}
for key, (label, pat, levels) in CONFIGS.items():
    rows = []
    for lv, suf in levels:
        p = os.path.join(RUNS, pat % suf)
        if not os.path.exists(p):
            print('MISSING', os.path.basename(p)); continue
        r = score(p); r['level_db'] = lv
        rows.append(r)
    out[key] = dict(label=label, rows=rows)
    print(f'{label}')
    print('   lvl   rms_coh  rms_raw  %out   level   coh    drive')
    for r in rows:
        print('  %+4d   %7.3f  %7.3f %5.1f  %+6.2f  %.4f %6.2f'
              % (r['level_db'], r['rms_coh'], r['rms_raw'], r['pct_out'],
                 r['level'], r['coh'], r['drive']))
with open(os.path.join(NOTES, 'moving_plant_scores.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
print('\nwrote moving_plant_scores.json')
# -*- coding: utf-8 -*-
"""Report figures: optimal_diagonal and match_trace_pseudoinverse on the
bounded-softening (moving) plant.

Colour follows the LAW, line style follows the configuration, so the pairing
reads without consulting the legend twice.  Palette validated with the dataviz
validator: #1a5fa0 / #cc5a28, dE 30.0 normal vision, 20.4 under protanopia.

All three panels are on the same test-level axis because the point of the
figure is that they disagree: the law that wins on accuracy loses on level.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'figures', 'report3wk'))
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(NOTES, 'moving_plant_scores.json')))

OD, MT = '#1a5fa0', '#cc5a28'
INK, INK2 = '#0b0b0b', '#52514e'
GRID_MAJ, GRID_MIN = '#cfcecb', '#e8e7e3'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.8, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.axisbelow': True, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
    'font.family': 'DejaVu Sans'})

SERIES = [
 ('od_off', OD, '-',  'o', 'optimal_diagonal, trim OFF'),
 ('od_on',  OD, '--', 's', 'optimal_diagonal, trim ON'),
 ('mt_on',  MT, '-',  'o', 'match_trace_pinv, FRF refresh ON'),
 ('mt_off', MT, '--', 's', 'match_trace_pinv, FRF refresh OFF'),
]


def grid(ax, ymaj, ymin):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
    ax.grid(True, which='major', color=GRID_MAJ, lw=0.6)
    ax.grid(True, which='minor', color=GRID_MIN, lw=0.45)
    ax.tick_params(which='minor', length=2.2, color='#b9b8b4')
    ax.tick_params(which='major', length=4.0, color='#9a9994')


def get(key, field):
    rows = D[key]['rows']
    return ([r['level_db'] for r in rows], [r[field] for r in rows])


fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True,
                                 gridspec_kw={'hspace': 0.16})
for key, col, ls, mk, lab in SERIES:
    x, y = get(key, 'rms_coh')
    a1.plot(x, y, ls=ls, marker=mk, color=col, lw=1.6, ms=5.5, label=lab, zorder=4)
    x, y = get(key, 'level')
    a2.plot(x, y, ls=ls, marker=mk, color=col, lw=1.6, ms=5.5, zorder=4)
    x, y = get(key, 'coh')
    a3.plot(x, y, ls=ls, marker=mk, color=col, lw=1.6, ms=5.5, zorder=4)

grid(a1, 0.5, 0.25); a1.set_ylim(5.2, 8.7)
a1.set_ylabel('coherent-column rms error\n(dB, lower is better)')
a1.set_title('(a)  Accuracy — optimal_diagonal is 1.9 dB better and stays there '
             'as the plant moves', loc='left', pad=7, color=INK)
# legend is placed at figure level below panel (c) -- see fig.legend() later

grid(a2, 2.0, 0.5); a2.set_ylim(-1.2, 9.0)
a2.axhspan(-0.5, 0.5, color='#eef3f8', lw=0, zorder=0)
a2.axhline(0.0, color='#9a9994', lw=0.9, zorder=1)
a2.text(8.6, -0.80, '±0.5 dB level acceptance band', fontsize=8,
        color=INK2, ha='right')
a2.set_ylabel('total level error\n(dB re specification)')
a2.set_title('(b)  Level — and loses it completely: +8.00 dB at +4 dB demand, '
             'against +0.51 dB', loc='left', pad=7, color=INK)
a2.annotate('the law is blind to this:\ndrive rose 0.9 dB for a\n4 dB increase in demand',
            xy=(4, 8.0), xytext=(-5.5, 6.4), fontsize=8, color=INK2, va='center',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=-0.15'))

grid(a3, 0.2, 0.05); a3.set_ylim(0.0, 1.08)
a3.set_ylabel('median multiple coherence\n(in band)')
a3.set_xlabel('test level (dB re full level)')
a3.set_xlim(-19, 9)
a3.set_title('(c)  Model quality — what each law pays for its result',
             loc='left', pad=7, color=INK)
a3.annotate('refresh ON feeds control-time estimates\ncarrying distortion back into the solve',
            xy=(0, 0.70), xytext=(-18.4, 0.30), fontsize=8, color=INK2, va='center',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=0.15'))

fig.suptitle('Both laws on the moving plant — bounded-softening article, runs 49, 50, 51, 55',
             fontsize=10, color=INK, x=0.012, ha='left', y=0.995)
h, lb = a1.get_legend_handles_labels()
fig.legend(h, lb, loc='lower center', bbox_to_anchor=(0.54, -0.035),
           ncol=2, columnspacing=2.6, handlelength=2.6)
fig.align_ylabels([a1, a2, a3])
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig13_moving_plant_both_laws.{ext}'), dpi=300,
                bbox_inches='tight')
plt.close(fig)
print('wrote fig13_moving_plant_both_laws.png / .pdf')
# -*- coding: utf-8 -*-
"""Per-channel error, both laws, at the near-linear and full-level states.

The eight control channels are served by a rank-5 plant, so two of them must
be conceded whatever the law does.  Both laws concede the same two.  What
separates them is how well they serve the six they can reach -- and that gap
is already there at -18 dB, before the plant has moved at all, so it is a
rank consequence rather than a nonlinearity consequence.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'figures', 'report3wk'))
D = json.load(open(os.path.join(NOTES, 'moving_plant_scores.json')))

OD, MT = '#1a5fa0', '#cc5a28'
INK, INK2 = '#0b0b0b', '#52514e'
GRID_MAJ, GRID_MIN = '#cfcecb', '#e8e7e3'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.8, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.axisbelow': True, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
    'font.family': 'DejaVu Sans'})


def ch(key, lvl):
    return np.array([r['ch_coh'] for r in D[key]['rows'] if r['level_db'] == lvl][0])


x = np.arange(1, 9)
w = 0.38
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True)
fig.subplots_adjust(wspace=0.08)

for ax, lvl, ttl in ((b1, -18, '(a)  −18 dB — plant effectively linear'),
                     (b2, 0, '(b)  0 dB — plant softened 1%')):
    od, mt = ch('od_off', lvl), ch('mt_on', lvl)
    ax.bar(x - w/2, od, w * 0.92, color=OD, label='optimal_diagonal, trim OFF', zorder=3)
    ax.bar(x + w/2, mt, w * 0.92, color=MT, label='match_trace_pinv, refresh ON', zorder=3)
    ax.axvspan(6.5, 8.5, color='#f4f3f0', lw=0, zorder=0)
    ax.set_xticks(x)
    ax.set_xlabel('control channel')
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, axis='y', which='major', color=GRID_MAJ, lw=0.6)
    ax.grid(True, axis='y', which='minor', color=GRID_MIN, lw=0.45)
    ax.tick_params(which='minor', length=0)
    ax.set_xlim(0.4, 8.6)
    ax.set_ylim(0, 15.5)
    ax.set_title(ttl, loc='left', pad=7, color=INK)
    ax.text(7.5, 14.6, 'conceded', fontsize=8, color=INK2, ha='center')

b1.set_ylabel('coherent-column rms error  (dB)')
b1.legend(loc='upper left', ncol=1)
b1.annotate('channels 7 and 8 are the\ntwo a rank-5 plant cannot\n'
            'reach. Both laws concede\nthem, and already do so\nhere, before the plant\nhas moved.',
            xy=(6.55, 10.6), xytext=(0.75, 10.4), fontsize=8, color=INK2, va='top',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=-0.18'))
b2.annotate('the whole difference is in\nthe six they can reach:\n'
            '1.9\u20135.1 dB vs 4.0\u20136.4 dB',
            xy=(2.2, 4.9), xytext=(0.75, 10.4), fontsize=8, color=INK2, va='top',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=0.18'))

fig.suptitle('Where each law puts its error — per control channel, bounded-softening plant',
             fontsize=10, color=INK, x=0.012, ha='left', y=1.0)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig14_moving_plant_per_channel.{ext}'), dpi=300,
                bbox_inches='tight')
plt.close(fig)
print('wrote fig14_moving_plant_per_channel.png / .pdf')
