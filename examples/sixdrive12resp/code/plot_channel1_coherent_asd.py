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
   an effective n_d of 19.31 (20 frames, 50% overlap, measured by
   verify_coherence_bias.py), so the coherent trace is an UPPER bound; the
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
# frames_in_cpsd is 20 but the frames overlap 50% with a Hann window, so
# they are not 20 independent observations.  verify_coherence_bias.py
# measures the effective count through the real Welch chain: 19.31.
N_AVG_EFF = 19.31
BAND = (100.0, 1000.0)
HARM = (1250.0, 1450.0)
FMAX = 1600.0
CASES = [('run51_optdiag_bsoft_trimoff_spec_m18.nc4', '−18 dB  — near linear'),
         ('run51_optdiag_bsoft_trimoff_spec.nc4',      '0 dB  — full level')]

RAMP = ['#5aa5e4', '#1a5fa0']            # ordered variable -> one hue, light to dark
ORANGE = '#cc5a28'                       # the bias-corrected trace
# validated: node dataviz/scripts/validate_palette.js '#5aa5e4,#1a5fa0,#cc5a28'
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
adj = N_AVG_EFF / (N_AVG_EFF - N_DRIVE)
YLIM = (1e-9, 3e-1)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.16})
for ax in (a1, a2):
    style(ax)
a1.set_ylim(*YLIM)
a2.set_xlim(0, FMAX)

for (fk, Syy, coh, _, lab), c in zip(data, RAMP):
    a1.plot(fk[m], Syy[m], lw=1.1, color=c, label=lab, zorder=4)
    a2.plot(fk[m], (coh * Syy)[m], lw=1.1, color=c, label=lab, zorder=4)
g2a = 1.0 - (1.0 - data[-1][2]) * adj
g2a = np.where(g2a > 0.0, g2a, np.nan)
a2.plot(f[m], (g2a * data[-1][1])[m], lw=0.7, color=ORANGE, ls=(0, (3, 2)),
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


# =========================================================================
# ZOOM: 200-400 Hz, the boundary between the linear and softened modes
# =========================================================================
# Four modes in this window are exactly linear (214.37, 246.94, 249.18,
# 286.25 Hz) and three are softened (300.86, 306.47, 335.24 Hz).  At 0 dB the
# softened three come down 0.97-0.99% while the linear four do not move at
# all, so the gap between the top linear mode and the first softened mode
# closes from 14.61 to 11.70 Hz, a 20% narrowing.  The control failure sits in
# that gap rather than on either peak, and it is only half coherent.
ZOOM = (200.0, 400.0)
ZYLIM = (5e-4, 3.5e-1)
LIN_MODES = np.array([214.37, 246.94, 249.18, 286.25])
SOFT_MODES = np.array([300.86, 306.47, 335.24])
SOFT_AT_0 = np.array([297.95, 303.47, 331.91])


def style_zoom(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which='major', color=GRID_MAJ, lw=0.6, ls='-')
    ax.grid(True, which='minor', color=GRID_MIN, lw=0.45, ls='-')
    ax.tick_params(which='minor', length=2.2, color='#b9b8b4')
    ax.tick_params(which='major', length=4.0, color='#9a9994')
    for x in LIN_MODES:
        ax.axvline(x, color='#a3a29d', lw=1.0, zorder=2)
    for x in SOFT_MODES:
        ax.axvline(x, color=RAMP[0], lw=1.1, ls=(0, (5, 3)), zorder=2)
    for x in SOFT_AT_0:
        ax.axvline(x, color=RAMP[1], lw=1.1, ls=(0, (5, 3)), zorder=2)


ds = netCDF4.Dataset(os.path.join(RUNS, CASES[0][0]))
SPEC = np.array(ds[GROUP]['specification_cpsd_matrix_real'][:])[:, CH, CH]
ds.close()
zm = (f >= ZOOM[0]) & (f <= ZOOM[1])

figz, (z1, z2) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, sharey=True,
                              gridspec_kw={'hspace': 0.17})
for ax in (z1, z2):
    style_zoom(ax)
z1.set_ylim(*ZYLIM)
z2.set_xlim(*ZOOM)

z1.plot(f[zm], SPEC[zm], lw=1.5, color='#7a7975', ls=(0, (1, 2)),
        label='specification', zorder=3)
for (fk, Syy, coh, _, lab), c in zip(data, RAMP):
    z1.plot(fk[zm], Syy[zm], lw=1.5, color=c, label=lab, zorder=4)
    z2.plot(fk[zm], (coh * Syy)[zm], lw=1.5, color=c, label=lab, zorder=4)
g2az = 1.0 - (1.0 - data[-1][2]) * adj
g2az = np.where(g2az > 0.0, g2az, np.nan)
z2.plot(f[zm], (g2az * data[-1][1])[zm], lw=0.9, color=ORANGE, ls=(0, (3, 2)),
        alpha=0.95, label='0 dB, bias-corrected', zorder=3)

z1.annotate('', xy=(SOFT_AT_0[0], 1.7e-1), xytext=(SOFT_MODES[0], 1.7e-1),
            arrowprops=dict(arrowstyle='-|>', color=RAMP[1], lw=1.4,
                            shrinkA=0, shrinkB=0))
z1.text(299.4, 2.6e-1, '300.86 \u2192 297.95 Hz  (\u22120.97%)', fontsize=8,
        color=RAMP[1], ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none'))
z1.text(286.25 - 2.5, 1.7e-1, 'fixed', fontsize=8, color='#6e6d69',
        ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none'))
z1.annotate('+15.0 dB over spec at 294 Hz \u2014\nthe miss is in the GAP between the\n'
            'fixed mode at 286.2 Hz and the\nsoftened one, not on either peak',
            xy=(292, 3.6e-2), xytext=(202, 1.5e-1), fontsize=8.5, color=INK2,
            va='top',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0,
                            connectionstyle='arc3,rad=0.18'))
z1.set_ylabel('response ASD\n(G$^2$/Hz, re full level)')
z1.set_title('(a)  Measured response, 200\u2013400 Hz \u2014 control channel 1, run 51, '
             'optimal_diagonal', loc='left', pad=7, color=INK)
z1.legend(loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.72))

z2.annotate('288\u2013300 Hz at 0 dB: +14.07 dB over spec,\n'
            'of which +9.84 dB coherent, or +4.86 dB\n'
            'after the bias correction.  Median $\\gamma^2$ is\n'
            '0.40 here, against 0.997 at \u221218 dB.',
            xy=(293, 1.5e-2), xytext=(202, 2.9e-1), fontsize=8.5, color=INK2,
            va='top',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0,
                            connectionstyle='arc3,rad=0.18'))
z2.set_ylabel('$\\gamma^2 \\times$ response ASD\n(G$^2$/Hz, re full level)')
z2.set_xlabel('frequency (Hz)')
z2.set_title('(b)  Coherent part \u2014 the excursion is part control miss and '
             'part distortion', loc='left', pad=7, color=INK)
z2.legend(loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.80))

figz.text(0.012, 0.012,
          'Vertical rules: solid grey, the four modes that stay linear.  '
          'Dashed, the three softened modes — light at −18 dB, dark at their '
          '0 dB positions.',
          fontsize=8, color=INK2, ha='left')
figz.align_ylabels([z1, z2])
for ext in ('png', 'pdf'):
    figz.savefig(os.path.join(OUT, f'fig11_ch1_zoom_200_400.{ext}'), dpi=300,
                 bbox_inches='tight')
plt.close(figz)

print()
print('200-400 Hz window, channel 1')
for lo, hi, lab in ((288.0, 300.0, '288-300 Hz (the gap)'), ZOOM + ('200-400 Hz',)):
    w = (f >= lo) & (f <= hi)
    for (fk, Syy, coh, _, L), tag in zip(data, ('-18 dB', '  0 dB')):
        ca = np.clip(1.0 - (1.0 - coh) * adj, 0.0, 1.0)
        print('  %-22s %s: total %+6.2f dB, coherent %+6.2f dB '
              '(corrected %+6.2f), median gamma^2 %.4f'
              % (lab, tag, 10 * np.log10(Syy[w].sum() / SPEC[w].sum()),
                 10 * np.log10((coh * Syy)[w].sum() / SPEC[w].sum()),
                 10 * np.log10((ca * Syy)[w].sum() / SPEC[w].sum()),
                 np.median(coh[w])))
print('  gap top-linear to first-softened: %.2f Hz at -18 dB, %.2f Hz at 0 dB'
      % (SOFT_MODES[0] - LIN_MODES[-1], SOFT_AT_0[0] - LIN_MODES[-1]))
print('wrote fig11_ch1_zoom_200_400.png / .pdf')


# =========================================================================
# OVERLAY at 0 dB: total, coherent, and bias-corrected coherent
# =========================================================================
# The three are nested: total >= coherent >= bias-corrected.  The first gap is
# the response the six drives do not explain linearly; the second is the part
# of the "coherent" power that is really six complex coefficients fitted from
# twenty averages, which a response unrelated to the drives would also show.
f0, S0, c0, ib0, lab0 = data[-1]
c0a_plot = 1.0 - (1.0 - c0) * adj
c0a_plot = np.where(c0a_plot > 0.0, c0a_plot, np.nan)
c0a_int = np.clip(1.0 - (1.0 - c0) * adj, 0.0, 1.0)

rms_spec = np.sqrt(SPEC[ib0].sum())
rms_tot = np.sqrt(S0[ib0].sum())
rms_coh = np.sqrt((c0 * S0)[ib0].sum())
rms_cor = np.sqrt((c0a_int * S0)[ib0].sum())

figo, ax = plt.subplots(figsize=(7.2, 4.8))
style(ax)
ax.set_ylim(1e-9, 3e-1)
ax.set_xlim(0, FMAX)

ax.plot(f0[m], SPEC[m], lw=1.1, color='#7a7975', ls=(0, (1, 2)),
        label='specification', zorder=3)
ax.plot(f0[m], S0[m], lw=1.3, color=RAMP[1], label='total response', zorder=6)
ax.plot(f0[m], (c0 * S0)[m], lw=1.3, color=RAMP[0],
        label='coherent  ($\\gamma^2 \\times$ total)', zorder=5)
ax.plot(f0[m], (c0a_plot * S0)[m], lw=1.1, color=ORANGE, ls=(0, (3, 2)),
        label='coherent, bias-corrected', zorder=4)

ax.text(BAND[0] + 18, 1.1e-1, 'control band  100\u20131000 Hz',
        fontsize=8, color=INK2, ha='left', va='center')
ax.text(np.mean(HARM), 1.1e-1, '3f', fontsize=8, color=INK2,
        ha='center', va='center')
ax.text(1035, 6.0e-5,
        'out of band the coherent part is\ntwo decades below the total \u2014\n'
        'none of it comes from the drives',
        fontsize=8, color=INK2, ha='left', va='top')

stats = ('in-band rms, 100\u20131000 Hz\n'
         'specification    %.4f G\n'
         'total            %.4f G   %+5.2f dB\n'
         'coherent         %.4f G   %+5.2f dB\n'
         'bias-corrected   %.4f G   %+5.2f dB'
         % (rms_spec, rms_tot, 20*np.log10(rms_tot/rms_spec),
            rms_coh, 20*np.log10(rms_coh/rms_spec),
            rms_cor, 20*np.log10(rms_cor/rms_spec)))
ax.text(150, 2.6e-6, stats, fontsize=8, color=INK, ha='left', va='top',
        family='DejaVu Sans Mono',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#cfcecа'.replace('а', 'a'),
                  lw=0.8))

ax.set_ylabel('ASD  (G$^2$/Hz, referenced to full level)')
ax.set_xlabel('frequency (Hz)')
ax.set_title('Control channel 1 at 0 dB \u2014 how much of the response the six '
             'drives explain linearly\nrun 51, optimal_diagonal, trim off',
             loc='left', pad=8, color=INK)
ax.legend(loc='lower left', ncol=1, bbox_to_anchor=(0.005, 0.015))

for ext in ('png', 'pdf'):
    figo.savefig(os.path.join(OUT, f'fig12_ch1_0dB_overlay.{ext}'), dpi=300,
                 bbox_inches='tight')
plt.close(figo)

print()
print('channel 1 at 0 dB, in-band 100-1000 Hz   (specification %.4f G)' % rms_spec)
for nm, v in (('total', rms_tot), ('coherent', rms_coh),
              ('coherent, bias-corrected', rms_cor)):
    print('  %-26s %.4f G   %+6.2f dB re spec' % (nm, v, 20 * np.log10(v / rms_spec)))
print('wrote fig12_ch1_0dB_overlay.png / .pdf')
