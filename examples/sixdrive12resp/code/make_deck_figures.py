"""Deck-scale variants of two report figures.

The report figures are sized for a page: 12.6 in wide with 9 pt type. Scaled
to fit a slide they render at roughly 4 pt, which does not project. These
variants carry the same data at slide proportions with type that survives the
reduction -- fewer panels, larger fonts, thicker marks.
"""
import os
import numpy as np
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

OUT = 'figs'
RUN = '../results/runs/run03_optdiag_capon_spec.nc4'
SYS = '../results/case/sdynpy_frame6x12_system.npz'
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
NULLC = '#78766f'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d8d7d2', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'font.size': 15, 'axes.spines.top': False,
    'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'xtick.labelsize': 14, 'ytick.labelsize': 14})


def frf(f):
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


# ---- singular values, single panel, deck scale -----------------------------
f = np.arange(100., 1000.001, 1.0)
ds = netCDF4.Dataset(RUN); g = ds.groups['Frame 6x12 Random']
fm = np.array(g['specification_frequency_lines'][:], float)
Hm = (np.array(g['frf_data_real'][:]) + 1j*np.array(g['frf_data_imag'][:]))
Hm = Hm[np.isin(np.round(fm, 6), np.round(f, 6))]; ds.close()
svm = np.linalg.svd(Hm, compute_uv=False)

fig, a = plt.subplots(figsize=(11.6, 4.6))
for i in range(5):
    a.semilogy(f, svm[:, i], lw=2.0, color=RAMP[i], label=f'σ{i+1}')
a.semilogy(f, svm[:, 5], lw=1.6, color=NULLC, ls='--', label='σ6')
a.set_xlim(100, 1000); a.set_ylim(3e-5, 4e2)
a.set_xlabel('frequency (Hz)', fontsize=15)
a.set_ylabel('singular value   (m/s²)/N', fontsize=15)
a.grid(True)
a.legend(frameon=False, fontsize=14, ncol=6, loc='upper left',
         columnspacing=1.5, handlelength=1.8)
# The note that belongs here lives in the slide's own callout box, so the
# plot stays clean rather than printing text over the sigma-6 trace.
fig.savefig(os.path.join(OUT, 'deck_singular_values.png'), dpi=170,
            bbox_inches='tight', facecolor=SURF)
plt.close(fig)

# ---- damping + conditioning, two panels, deck scale -------------------------
z = np.load(SYS, allow_pickle=True)
M_, K_, C_ = z['mass'], z['stiffness'], z['damping']
w2, phi = eigh(K_, M_); w = np.sqrt(np.maximum(w2, 0)); fn = w/(2*np.pi)
zeta = np.diag(phi.T @ C_ @ phi) / (2*np.maximum(w, 1e-12))
keep = (fn >= 100) & (fn <= 1000)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
a = axes[0]
a.hist(zeta[keep]*100, bins=12, color=RAMP[2], edgecolor=SURF)
a.set_xlabel('modal damping ζ (%)', fontsize=15)
a.set_ylabel('modes', fontsize=15); a.grid(axis='y')
a.set_title(f'{keep.sum()} modes in the control band\nζ = {zeta[keep].min()*100:.2f}–'
            f'{zeta[keep].max()*100:.2f} %, median {np.median(zeta[keep])*100:.2f} %',
            loc='left', fontsize=15, pad=10)

a = axes[1]
sv = np.linalg.svd(frf(f), compute_uv=False)
a.semilogy(f, sv[:, 0]/sv[:, 4], lw=2.0, color=RAMP[2], label='σ1/σ5  (real)')
a.semilogy(f, svm[:, 0]/svm[:, 5], lw=1.8, color=NULLC, ls='--',
           label='σ1/σ6  (noise floor)')
a.set_xlim(100, 1000)
a.set_xlabel('frequency (Hz)', fontsize=15)
a.set_ylabel('condition number', fontsize=15)
a.grid(True)
a.legend(frameon=False, fontsize=14, loc='upper left')
a.set_title('Conditioning: which number to quote\nmedian 68 against median 3567',
            loc='left', fontsize=15, pad=10)
fig.savefig(os.path.join(OUT, 'deck_system.png'), dpi=170, bbox_inches='tight',
            facecolor=SURF)
plt.close(fig)
print('wrote figs/deck_singular_values.png and figs/deck_system.png')
