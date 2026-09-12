"""Generate the figures for the 6-12-8 control-law comparison report."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

OUT = 'figs'
os.makedirs(OUT, exist_ok=True)

# Validated 3-slot categorical palette (light mode, all-pairs PASS)
C = {'match_trace': '#2a78d6', 'optimal_diagonal': '#eb6834', 'congruence': '#1baf7a'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d8d7d2', '#fcfcfb'

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

# Representative run per law family, cap ON (the operationally sane setting).
FAM = [('match_trace', 'run01', 'match_trace_pseudoinverse'),
       ('optimal_diagonal', 'run03', 'optimal_diagonal_control'),
       ('congruence', 'run07', 'match_diagonal_congruence')]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches='tight',
                facecolor=SURF)
    plt.close(fig)
    print('  ', name)


# --- 1. per-channel mean error, three laws ---------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(8); w = 0.26
for i, (fam, run, label) in enumerate(FAM):
    v = by[run]['mean_db']
    ax.bar(x + (i - 1) * w, v, w * 0.92, color=C[fam], label=label,
           edgecolor=SURF, linewidth=1.2)
ax.axhline(0, color=INK2, lw=1)
for y, s in [(6, '+6 dB tolerance'), (-6, '-6 dB tolerance')]:
    ax.axhline(y, color=INK2, lw=1, ls=':')
    ax.text(7.55, y + 0.25, s, fontsize=7.5, color=INK2, ha='right')
ax.set_xticks(x); ax.set_xticklabels(CH)
ax.set_ylabel('mean error (dB)'); ax.set_xlabel('control channel')
ax.set_ylim(-8, 8); ax.grid(axis='y')
ax.legend(frameon=False, ncol=3, fontsize=8, loc='lower left')
ax.set_title('Mean per-channel error against the flat specification',
             loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'per_channel_mean.png')

# --- 2. achievable floor vs achieved ---------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
z = np.load('scoring/run07_congruence_capon_spec_scoring.npz')
reach = z['reachability__mean_db']              # achievable vs flat spec
ax.bar(x, reach, 0.62, color=INK2, alpha=0.28, edgecolor=SURF, linewidth=1.2,
       label='best achievable (reachability limit)')
for i, (fam, run, label) in enumerate(FAM):
    ax.plot(x, by[run]['mean_db'], 'o-', color=C[fam], lw=2, ms=7,
            markeredgecolor=SURF, markeredgewidth=1.4, label=label)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x); ax.set_xticklabels(CH)
ax.set_ylabel('mean error vs flat spec (dB)'); ax.set_xlabel('control channel')
ax.grid(axis='y'); ax.legend(frameon=False, fontsize=8, loc='lower left')
ax.set_title('What is reachable, and what each law delivered',
             loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'achievable_vs_achieved.png')

# --- 3. rms vs percent out, drive as size ----------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for r in rows:
    fam = 'optimal_diagonal' if r['law'].startswith('optimal') else \
          ('congruence' if r['law'] == 'congruence' else 'match_trace')
    s = 60 + 260 * (np.log10(r['drive_trace']) - np.log10(4e-2)) / \
        (np.log10(6.1) - np.log10(4e-2))
    ax.scatter(r['rms'], r['pout'], s=max(s, 55), color=C[fam], alpha=0.85,
               edgecolor=SURF, linewidth=1.6, zorder=3)
    ax.annotate(f"{r['run'][3:]} {r['cap']}", (r['rms'], r['pout']),
                textcoords='offset points', xytext=(9, -3), fontsize=7.4,
                color=INK2)
ax.set_xlabel('overall rms error (dB)  — lower is better')
ax.set_ylabel('% of lines outside ±6 dB  — lower is better')
ax.grid(True)
ax.set_title('The two acceptance criteria disagree\nmarker area ∝ log drive power',
             loc='left', fontsize=10.5, color=INK, pad=10)
handles = [plt.Line2D([], [], marker='o', ls='', color=C[f], ms=8,
                      markeredgecolor=SURF, label=f) for f in C]
ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left')
save(fig, 'rms_vs_pout.png')

# --- 4. drive power ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.6))
order = sorted(rows, key=lambda r: r['drive_trace'])
cols = ['#eb6834' if r['law'].startswith('optimal') else
        ('#1baf7a' if r['law'] == 'congruence' else '#2a78d6') for r in order]
lab = [f"{r['run'][3:]}  {r['law'].replace('optimal_diagonal_fast','optdiag fast').replace('optimal_diagonal','optdiag').replace('match_trace','match trace')}\n{r['cap']}"
       for r in order]
ax.bar(range(8), [r['drive_trace'] for r in order], 0.62, color=cols,
       edgecolor=SURF, linewidth=1.2)
for i, r in enumerate(order):
    ax.text(i, r['drive_trace'] * 1.25, f"{r['drive_trace']:.3g}",
            ha='center', fontsize=7.6, color=INK2)
ax.set_yscale('log'); ax.set_xticks(range(8))
ax.set_xticklabels(lab, fontsize=7.6)
ax.set_xlim(-0.6, 7.6)
ax.set_ylabel('mean drive trace (V², log scale)')
ax.grid(axis='y')
ax.set_title('Drive power spans a factor of 128 across the eight runs',
             loc='left', fontsize=10.5, color=INK, pad=10)
save(fig, 'drive_power.png')

# --- 5. system characterisation --------------------------------------------
sysfile = ('/mnt/user-data/uploads/nhunterjr--Code--python--rattlesnake-'
           'vibration-controller/examples/sixdrive12resp/results/case/'
           'sdynpy_frame6x12_system.npz')
d = np.load(sysfile, allow_pickle=True)
M, K, Cm = d['mass'], d['stiffness'], d['damping']
w2, phi = eigh(K, M); w = np.sqrt(np.maximum(w2, 0)); fn = w / (2 * np.pi)
zeta = np.diag(phi.T @ Cm @ phi) / (2 * np.maximum(w, 1e-12))
keep = (fn >= 100) & (fn <= 1000)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
a = axes[0]
a.hist(zeta[keep] * 100, bins=12, color=C['match_trace'], edgecolor=SURF)
a.set_xlabel('modal damping ζ (%)'); a.set_ylabel('modes')
a.grid(axis='y')
a.set_title(f'{keep.sum()} modes in 100–1000 Hz\nζ = {zeta[keep].min()*100:.2f}–'
            f'{zeta[keep].max()*100:.2f} %, median {np.median(zeta[keep])*100:.2f} %',
            loc='left', fontsize=9.5, pad=8)

a = axes[1]
zz = np.load('scoring/run03_optdiag_capon_spec_scoring.npz')
H = zz['frf']; f = zz['frequencies']
band = (f >= 100) & (f <= 1000)
sv = np.linalg.svd(H[band], compute_uv=False)
for i in range(sv.shape[1]):
    a.semilogy(f[band], sv[:, i], lw=1.1,
               color=plt.cm.viridis(i / max(sv.shape[1] - 1, 1)))
a.set_xlabel('frequency (Hz)'); a.set_ylabel('singular value')
a.grid(True)
a.set_title('FRF singular values: 6 drives,\n8 control channels — 2 directions absent',
            loc='left', fontsize=9.5, pad=8)

a = axes[2]
cond = [ (r['run'], c) for r, c in zip(rows,
        [6976, 25928, 3567, 2493, 3457, 2447, 3354, 8676]) ]
a.bar(range(8), [c for _, c in cond], 0.62,
      color=[C['match_trace']]*2 + [C['optimal_diagonal']]*4 + [C['congruence']]*2,
      edgecolor=SURF, linewidth=1.2)
a.set_yscale('log'); a.set_xticks(range(8))
a.set_xticklabels([n[3:] for n, _ in cond], fontsize=8)
a.set_ylabel('median cond(H)'); a.set_xlabel('run')
a.grid(axis='y')
a.set_title('Identification scatter: same plant,\nsame settings, factor of 10',
            loc='left', fontsize=9.5, pad=8)
save(fig, 'system_characterisation.png')
print('done')
