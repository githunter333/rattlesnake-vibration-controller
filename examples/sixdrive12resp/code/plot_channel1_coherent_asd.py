"""
plot_channel1_coherent_asd.py

Control channel 1 (node 7 X+, accelerometer) of the bounded-softening nonlinear
plant, run 51 (optimal_diagonal, trim off), at two test levels.

  top     the measured response auto-spectrum  S_yy[1,1]
  bottom  the same weighted by that channel's MULTIPLE COHERENCE, gamma^2 * S_yy

The second is the part of the response the six drives explain linearly.  The
gap between the panels is everything else: distortion, extraneous input and
estimator noise.

TWO THINGS TO KNOW WHEN READING THESE.

1. The saved response and drive CPSDs are referenced to the FULL-LEVEL
   specification, not absolute at the test level -- the -18 dB response matches
   the full spec to -0.17 dB and the saved specification matrix is identical at
   every level.  That is why the two traces nearly overlie instead of sitting
   18 dB apart.  The axis labels say so.

2. gamma^2 is computed against a LIVE FRF, so the equivalent-linear part of the
   softening -- the whole frequency shift and damping rise -- counts as
   coherent.  Only the residual appears in the gap.  The saved estimate also
   carries a positive bias of about (p/n_d)(1-gamma^2) with p = 6 drives and
   n_d = 20 frames per CPSD, so the coherent trace is an UPPER bound; the
   bias-corrected 0 dB curve is drawn dashed.

Outputs PNG at 300 dpi for documents and PDF for vector use.
"""
import os
import numpy as np
import netCDF4
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'figures', 'report3wk'))
os.makedirs(OUT, exist_ok=True)

CH = 0
GROUP = 'Frame 6x12 Random'
N_DRIVE, N_AVG = 6, 20
BAND = (100.0, 1000.0)
HARM = (1250.0, 1450.0)
FMAX = 1600.0
CASES = [('run51_optdiag_bsoft_trimoff_spec_m18.nc4', '−18 dB  — near linear'),
         ('run51_optdiag_bsoft_trimoff_spec.nc4',      '0 dB  — full level')]

RAMP = ['#8cbbe8', '#10365c']            # ordered variable -> one hue, light to dark
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
GRID_MAJ, GRID_MIN = '#cfcecа'.replace('а', 'a'), '#e8e7e3'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.8,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.axisbelow': True,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
    'font.family': 'DejaVu Sans'})


def read(fn):
    ds = netCDF4.Dataset(os.path.join(RUNS, fn))
    g = ds[GROUP]
    f = np.array(g['specification_frequency_lines'][:], float)
    Syy = np.array(g['response_cpsd_real'][:])[:, CH, CH]
    coh = np.clip(np.array(g['frf_coherence'][:])[:, CH], 0.0, 1.0)
    spec = np.array(g['specification_cpsd_matrix_real'][:])[:, CH, CH]
    ds.close()
    return f, Syy, coh, spec > 0


def style(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.grid(True, which='major', color=GRID_MAJ, lw=0.6, ls='-')
    ax.grid(True, which='minor', color=GRID_MIN, lw=0.45, ls='-')
    ax.tick_params(which='minor', length=2.2, color='#b9b8b4')
    ax.tick_params(which='major', length=4.0, color='#9a9994')
    for x in BAND + HARM:
        ax.axvline(x, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=2)


data = [read(fn) + (lab,) for fn, lab in CASES]
f, inb = data[0][0], data[0][3]
m = f <= FMAX
adj = N_AVG / (N_AVG - N_DRIVE)
YLIM = (1e-9, 3e-1)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.16})
for ax in (a1, a2):
    style(ax)
a1.set_ylim(*YLIM)
a2.set_xlim(0, FMAX)

for (fk, Syy, coh, _, lab), c in zip(data, RAMP):
    a1.plot(fk[m], Syy[m], lw=1.0, color=c, label=lab, zorder=4)
    a2.plot(fk[m], (coh * Syy)[m], lw=1.0, color=c, label=lab, zorder=4)
g2a = 1.0 - (1.0 - data[-1][2]) * adj
g2a = np.where(g2a > 0.0, g2a, np.nan)
a2.plot(f[m], (g2a * data[-1][1])[m], lw=0.7, color='#d98a63', ls=(0, (3, 2)),
        label='0 dB, bias-corrected', zorder=3)

for ax, band_y in ((a1, 1.1e-1), (a2, 1.1e-1)):
    ax.text(BAND[0] + 18, band_y, 'control band  100–1000 Hz',
            fontsize=8, color=INK2, ha='left', va='center')
    ax.text(np.mean(HARM), band_y, '3f', fontsize=8, color=INK2,
            ha='center', va='center')

a1.set_ylabel('response ASD\n(G$^2$/Hz, re full level)')
a1.set_title('(a)  Measured response — control channel 1 (node 7 X+), '
             'run 51, optimal_diagonal, trim off', loc='left', pad=7, color=INK)
a1.legend(loc='lower center', ncol=2, title='test level', title_fontsize=8.5,
          bbox_to_anchor=(0.52, -0.02))
a1.annotate('+15.0 dB over spec at 294 Hz at 0 dB,\n'
            'against +5.1 dB at 306 Hz at −18 dB —\n'
            'the linear / softened band boundary',
            xy=(310, 3.0e-2), xytext=(600, 1.4e-2), fontsize=8, color=INK2,
            va='center',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=0.12'))

a2.set_ylabel('$\\gamma^2 \\times$ response ASD\n(G$^2$/Hz, re full level)')
a2.set_xlabel('frequency (Hz)')
a2.set_title('(b)  Coherent part — the portion the six drives explain linearly',
             loc='left', pad=7, color=INK)
a2.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.52, -0.02))
a2.annotate('3f is incoherent at both levels\n($\\gamma^2$ = 0.13 and 0.11):\n'
            'the weighting removes it',
            xy=(1320, 8e-8), xytext=(430, 9e-7), fontsize=8, color=INK2,
            va='center',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=-0.10'))

fig.align_ylabels([a1, a2])
for ext, dpi in (('png', 300), ('pdf', 300)):
    fig.savefig(os.path.join(OUT, f'fig10_ch1_asd_and_coherent.{ext}'), dpi=dpi,
                bbox_inches='tight')
plt.close(fig)

print('channel 1, in-band 100-1000 Hz')
print('  level     rms G    coherent rms G   bias-corr    incoherent %   median g^2')
for fk, Syy, coh, ib, lab in data:
    tot = np.sqrt(np.sum(Syy[ib]))
    co = np.sqrt(np.sum((coh * Syy)[ib]))
    ga = np.clip(1.0 - (1.0 - coh) * adj, 0.0, 1.0)
    coa = np.sqrt(np.sum((ga * Syy)[ib]))
    print('  %-22s %7.4f   %7.4f      %7.4f     %8.2f      %.4f'
          % (lab, tot, co, coa, 100 * (1 - co ** 2 / tot ** 2), np.median(coh[ib])))
h = (f >= HARM[0]) & (f <= HARM[1])
for fk, Syy, coh, ib, lab in data:
    print('  %-22s 3f band: ASD sum %.3e, median g^2 %.4f'
          % (lab, np.sum(Syy[h]), np.median(coh[h])))
print('wrote fig10_ch1_asd_and_coherent.png / .pdf')
