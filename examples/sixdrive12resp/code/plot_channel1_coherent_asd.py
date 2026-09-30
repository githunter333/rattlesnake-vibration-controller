"""
plot_channel1_coherent_asd.py

Control channel 1 (node 7 X+, accelerometer) of the bounded-softening nonlinear
plant, run 51 (optimal_diagonal, trim off), at two test levels.

  top     the measured response auto-spectrum  S_yy[1,1]
  bottom  the same weighted by that channel's MULTIPLE COHERENCE, gamma^2 * S_yy

The second is the part of the response that the six drives explain linearly.
The gap between the panels is everything else: distortion, extraneous input and
estimator noise.  Note that gamma^2 is computed against a LIVE FRF, so the
equivalent-linear part of the softening -- the whole frequency shift and damping
rise -- counts as coherent.  Only the residual shows up in the gap.

The saved multiple coherence carries a positive bias of roughly (p/n_d)(1-g^2)
with p = 6 drives and n_d = 20 frames per CPSD, so the coherent trace here is
an upper bound on the linear part.  Bias-corrected curves are drawn dashed.
"""
import os
import numpy as np
import netCDF4
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'figures', 'report3wk'))
os.makedirs(OUT, exist_ok=True)

CH = 0                                  # first control channel
GROUP = 'Frame 6x12 Random'
N_DRIVE, N_AVG = 6, 20                  # for the multiple-coherence bias
CASES = [('run51_optdiag_bsoft_trimoff_spec_m18.nc4', '−18 dB  — near linear'),
         ('run51_optdiag_bsoft_trimoff_spec.nc4',      '0 dB  — full level')]

# sequential ramp: level is an ordered variable, so one hue light -> dark
RAMP = ['#8cbbe8', '#10365c']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BANDFILL, HARMFILL = '#eef3f8', '#fbf0e9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False})


def read(fn):
    ds = netCDF4.Dataset(os.path.join(RUNS, fn))
    g = ds[GROUP]
    f = np.array(g['specification_frequency_lines'][:], float)
    Syy = np.array(g['response_cpsd_real'][:])[:, CH, CH]
    coh = np.array(g['frf_coherence'][:])[:, CH]
    spec = np.array(g['specification_cpsd_matrix_real'][:])[:, CH, CH]
    ds.close()
    return f, Syy, np.clip(coh, 0.0, 1.0), spec > 0


data = [read(fn) + (lab,) for fn, lab in CASES]
f = data[0][0]
inb = data[0][3]
FMAX = 1600.0
m = f <= FMAX
adj = N_AVG / (N_AVG - N_DRIVE)          # 20/14 = 1.4286

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.22})
for ax in (a1, a2):
    ax.axvspan(100, 1000, color=BANDFILL, lw=0, zorder=0)
    ax.axvspan(1250, 1450, color=HARMFILL, lw=0, zorder=0)
    ax.set_yscale('log')

for (fk, Syy, coh, _, lab), c in zip(data, RAMP):
    a1.plot(fk[m], Syy[m], lw=1.4, color=c, label=lab, zorder=3)
    a2.plot(fk[m], (coh * Syy)[m], lw=1.4, color=c, label=lab, zorder=3)
    g2a = 1.0 - (1.0 - coh) * adj
    g2a = np.where(g2a > 0.0, g2a, np.nan)      # mask where the correction clips
    a2.plot(fk[m], (g2a * Syy)[m], lw=0.9, color=c, ls='--', alpha=0.85, zorder=2)

a1.set_ylabel('response ASD  (G$^2$/Hz,\nreferenced to full level)')
a1.set_title('Measured — control channel 1 (node 7 X+), run 51, optimal_diagonal',
             fontsize=9.5, color=INK, pad=6, loc='left')
a1.legend(fontsize=8, loc='lower center', ncol=2, title='test level',
          title_fontsize=8)
a1.text(600, 4e-2, 'control band 100–1000 Hz', fontsize=7.6, color=INK2, ha='center')
a1.annotate('+15.0 dB over spec at 294 Hz at 0 dB;\n+5.1 dB at 306 Hz at \u221218 dB \u2014 this is\n'
            'the linear / softened band boundary',
            xy=(294, 2.6e-2), xytext=(330, 4e-7), fontsize=7.4, color=INK2,
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                            connectionstyle='arc3,rad=-0.15'))
a1.text(1350, 4e-2, '3f band', fontsize=7.6, color=INK2, ha='center')

a2.set_ylabel('$\\gamma^2 \\times$ response ASD\n(G$^2$/Hz, re full level)')
a2.set_xlabel('frequency (Hz)')
a2.set_xlim(0, FMAX)
a2.set_title('Coherent part — what the six drives explain linearly '
             '(dashed: bias-corrected)', fontsize=9.5, color=INK, pad=6, loc='left')
a2.annotate('3f is almost entirely incoherent:\nthe weighting removes it',
            xy=(1330, 6e-8), xytext=(1090, 2e-6), fontsize=7.6, color=INK2,
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
a1.set_ylim(1e-9, 2e-1)
fig.suptitle('Response spectral density and its coherent part, control channel 1',
             fontsize=10, color=INK, x=0.012, ha='left', y=0.985)
fig.savefig(os.path.join(OUT, 'fig10_ch1_asd_and_coherent.png'), dpi=200,
            bbox_inches='tight')
plt.close(fig)

# --- numbers for the caption ---------------------------------------------
print('channel 1, in-band 100-1000 Hz')
print('  level     rms G      coherent rms G   incoherent %   median gamma^2')
for fk, Syy, coh, ib, lab in data:
    tot = np.sqrt(np.sum(Syy[ib]))
    co = np.sqrt(np.sum((coh * Syy)[ib]))
    g2a = np.clip(1.0 - (1.0 - coh) * adj, 0.0, 1.0)
    coa = np.sqrt(np.sum((g2a * Syy)[ib]))
    print('  %-7s  %8.4f   %8.4f (%8.4f)  %8.2f      %.4f'
          % (lab, tot, co, coa, 100 * (1 - co ** 2 / tot ** 2), np.median(coh[ib])))
h = (f >= 1250) & (f <= 1450)
for fk, Syy, coh, ib, lab in data:
    print('  %-7s  3f band: ASD sum %.3e, median gamma^2 %.4f'
          % (lab, np.sum(Syy[h]), np.median(coh[h])))
print('wrote fig10_ch1_asd_and_coherent.png')
