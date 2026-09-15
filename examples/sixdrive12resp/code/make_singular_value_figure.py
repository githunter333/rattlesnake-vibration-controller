"""Singular values of the 6-12-8 FRF: 8 control responses from 6 drives.

Plots the measured in-control FRF (what the laws actually invert) against the
FRF computed exactly from the model's mass, stiffness and damping.  The two
agree on sigma_1..sigma_5 and disagree completely on sigma_6: the plant is
rank 5, so the analytic sixth singular value is ~1e-16 while the measured one
sits at the identification noise floor.

sigma_i are ORDERED, so they get a single-hue ordinal ramp (validated), not
categorical colours.  sigma_6 is drawn apart because it is not a physical
direction.
"""
import json
import os
import numpy as np
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figs'; os.makedirs(OUT, exist_ok=True)
RUN = '../results/runs/run03_optdiag_capon_spec.nc4'
SYS = '../results/case/sdynpy_frame6x12_system.npz'
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # ordinal, validated
NULLC = '#78766f'
INK, INK2, INK3, GRID, SURF = '#0b0b0b', '#52514e', '#78766f', '#d8d7d2', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True})


def analytic(f):
    z = np.load(SYS, allow_pickle=True)
    M, K, C, co = z['mass'], z['stiffness'], z['damping'], z['coordinate']
    node = np.array([int(c[0]) for c in co]); dirn = np.array([int(c[1]) for c in co])
    dr = [np.flatnonzero((node == n) & (dirn == 1))[0] for n in range(1, 7)]
    rs = [np.flatnonzero((node == n) & (dirn == 1))[0] for n in range(7, 15)]
    H = np.empty((len(f), 8, 6), complex)
    for i, fi in enumerate(f):
        w = 2*np.pi*fi
        R = np.linalg.solve(K - w**2*M + 1j*w*C, np.eye(M.shape[0])[:, dr])
        H[i] = -w**2 * R[rs, :]
    return H


f = np.arange(100., 1000.001, 1.0)
sva = np.linalg.svd(analytic(f), compute_uv=False)
ds = netCDF4.Dataset(RUN); g = ds.groups['Frame 6x12 Random']
fm = np.array(g['specification_frequency_lines'][:], float)
Hm = (np.array(g['frf_data_real'][:]) + 1j*np.array(g['frf_data_imag'][:]))
Hm = Hm[np.isin(np.round(fm, 6), np.round(f, 6))]; ds.close()
svm = np.linalg.svd(Hm, compute_uv=False)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4),
                         gridspec_kw={'width_ratios': [1.5, 1]})

a = axes[0]
for i in range(5):
    a.semilogy(f, svm[:, i], lw=1.3, color=RAMP[i], label=f'σ{i+1}')
a.semilogy(f, svm[:, 5], lw=1.1, color=NULLC, ls='--', label='σ6 (measured)')
a.set_xlim(100, 1000)
a.set_ylim(3e-5, 4e2)
a.text(995, 5e-5,
       'the analytic σ6 is ≈ 4×10⁻¹⁶, eleven decades below this trace:\n'
       'the measured σ6 is identification noise, not a sixth direction',
       fontsize=7.6, color=INK2, ha='right', va='bottom')
a.set_xlabel('frequency (Hz)')
a.set_ylabel('singular value   (m/s²)/N')
a.grid(True, which='major')
a.legend(frameon=False, fontsize=8, ncol=6, loc='upper left',
         columnspacing=1.1, handlelength=1.4, bbox_to_anchor=(0, 1.0))
a.set_title('Singular values of the 8×6 FRF over the control band\nfive physical '
            'directions spanning 37 dB, then nothing',
            loc='left', fontsize=10.5, color=INK, pad=10)

b = axes[1]
x = np.arange(6)
for k, (sv, lab, mk) in enumerate([(sva, 'analytic', 'o'), (svm, 'measured', 's')]):
    r = 20*np.log10(np.maximum(sv/sv[:, :1], 1e-300))
    med = np.median(r, axis=0)
    lo = np.percentile(r, 10, axis=0); hi = np.percentile(r, 90, axis=0)
    off = (k - 0.5)*0.16
    for i in range(6):
        c = RAMP[i] if i < 5 else NULLC
        b.plot([x[i]+off, x[i]+off], [lo[i], hi[i]], color=c, lw=1.3, alpha=0.5)
        b.plot(x[i]+off, med[i], mk, ms=7, color=c if k else SURF,
               markeredgecolor=c, markeredgewidth=1.8, zorder=3)
    if k == 0:
        for i in range(5):
            b.annotate(f'{med[i]:.0f}', (x[i]+off, med[i]), textcoords='offset points',
                       xytext=(-13, -3), fontsize=7.4, color=INK2, ha='right')
b.set_xticks(x); b.set_xticklabels([f'σ{i+1}' for i in range(6)])
b.set_ylim(-95, 16)
b.set_ylabel('dB relative to σ1')
b.grid(axis='y')
h = [plt.Line2D([], [], marker='o', ls='', ms=8, markerfacecolor=SURF,
                markeredgecolor=INK2, markeredgewidth=1.8, label='analytic (exact)'),
     plt.Line2D([], [], marker='s', ls='', ms=8, color=INK2, label='measured (run 03)')]
b.legend(handles=h, frameon=False, fontsize=8, loc='lower left')
b.set_title('Spread relative to σ1\nmarker = median, bar = 10th–90th percentile',
            loc='left', fontsize=10.5, color=INK, pad=10)
b.annotate('analytic σ6: −327 dB\n(off scale — exactly zero)', (5, -88),
           textcoords='offset points', xytext=(-4, 0), fontsize=7.4,
           color=INK2, ha='right', va='center')

fig.savefig(os.path.join(OUT, 'singular_values.png'), dpi=170,
            bbox_inches='tight', facecolor=SURF)


def summarise(sv):
    return [dict(i=i+1,
                 lo=float(sv[:, i].min()), med=float(np.median(sv[:, i])),
                 hi=float(sv[:, i].max()),
                 ratio=float(np.median(sv[:, i]/sv[:, 0])),
                 db=float(20*np.log10(max(np.median(sv[:, i]/sv[:, 0]), 1e-300))))
            for i in range(sv.shape[1])]


json.dump(dict(band=[100.0, 1000.0], n_lines=int(len(f)),
               units='(m/s^2)/N', source_run=os.path.basename(RUN),
               analytic=summarise(sva), measured=summarise(svm),
               cond_15_analytic=float(np.median(sva[:, 0]/sva[:, 4])),
               cond_15_measured=float(np.median(svm[:, 0]/svm[:, 4])),
               cond_16_measured=float(np.median(svm[:, 0]/svm[:, 5]))),
          open('singular_value_data.json', 'w'), indent=1)
print('wrote figs/singular_values.png and singular_value_data.json')
