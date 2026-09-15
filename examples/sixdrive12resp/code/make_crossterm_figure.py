"""Cross-term magnitudes and coherences across the twelve-run matrix.

The flat specification has zero off-diagonal terms.  Nothing in any of these
laws controls the off-diagonals -- they all match the DIAGONAL -- so whatever
cross-structure appears is a by-product.  This measures it, writes
crossterm_data.json for build_report.py, and renders figs/cross_terms.png.

Kept separate from compute_report_data.py because these statistics are cheap
(no optimisation) while that script's per-run achievable floors take ~45 s
each; folding them in would mean re-solving twelve floors to add a column.

The reference value the whole section turns on -- the cross-channel coherence
of the BEST ACHIEVABLE response -- is recomputed here from the floor solver's
own returned drive CPSD rather than quoted, so it cannot drift away from the
floor numbers in report_data.json.
"""
import importlib.util
import json, os
import numpy as np
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figs'; os.makedirs(OUT, exist_ok=True)
RUNS = '../results/runs'
LOOP = {'feedback': '#2a78d6', 'open': '#eb6834'}
INK, INK2, INK3, GRID, SURF = '#0b0b0b', '#52514e', '#78766f', '#d8d7d2', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True})

CAPPOS = {'match_trace_pseudoinverse': 1, 'match_diagonal_congruence': 1,
          'pseudoinverse_control': 1, 'buzz_control': 1,
          'optimal_diagonal_control': 4, 'optimal_diagonal_control_fast': 4}
SH = {'match_trace_pseudoinverse': 'match trace', 'optimal_diagonal_control': 'opt diagonal',
      'optimal_diagonal_control_fast': 'opt diag fast', 'match_diagonal_congruence': 'congruence',
      'pseudoinverse_control': 'pseudoinverse', 'buzz_control': 'buzz'}
OPEN = {'pseudoinverse_control', 'buzz_control'}
FLOOR_RUN = 'run03_optdiag_capon_spec.nc4'   # any run's FRF; they agree to ~0.2 dB


def floor_coherence(path, decimate=20):
    """Cross-channel coherence of the best-achievable response on this FRF."""
    spec = importlib.util.spec_from_file_location(
        'ar', '../../../control_laws/achievable_response.py')
    ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
    ds = netCDF4.Dataset(path); g = ds.groups['Frame 6x12 Random']
    f = np.array(g['specification_frequency_lines'][:], float)
    H = np.array(g['frf_data_real'][:]) + 1j*np.array(g['frf_data_imag'][:])
    S = np.array(g['specification_cpsd_matrix_real'][:], float); ds.close()
    tgt = np.einsum('fmm->fm', S)
    ok = np.isfinite(tgt).all(1) & (tgt.max(1) > 0) & (f >= 100) & (f <= 1000)
    li = np.flatnonzero(ok)[::decimate]
    r = ar.achievable_diagonal(H, tgt, line_indices=li, restrict_rcond=1e-3,
                               return_drives=True)
    X = np.asarray(r['drive_cpsd']); solved = np.flatnonzero(np.array(r['solved'], bool))
    Xs = X[solved] if X.shape[0] == H.shape[0] else X
    R = H[solved] @ Xs @ np.conj(np.swapaxes(H[solved], 1, 2))
    dr = np.real(np.einsum('fii->fi', R)); i, j = np.triu_indices(8, 1)
    gr = np.abs(R[:, i, j]) / np.sqrt(np.maximum(dr[:, i]*dr[:, j], 1e-300))
    dx = np.real(np.einsum('fii->fi', Xs)); a, b = np.triu_indices(6, 1)
    gd = np.abs(Xs[:, a, b]) / np.sqrt(np.maximum(dx[:, a]*dx[:, b], 1e-300))
    w = np.maximum(np.linalg.eigvalsh(R), 0)
    return dict(resp_med=float(np.median(gr)), resp_p90=float(np.percentile(gr, 90)),
                drive_med=float(np.median(gd)),
                drive_over_cap=float(100*np.mean(gd > 0.95)),
                part=float(np.median(w.sum(1)**2/np.maximum((w**2).sum(1), 1e-300))),
                n_lines=int(solved.size), source=os.path.basename(path))

FLOOR = floor_coherence(os.path.join(RUNS, FLOOR_RUN))
print('best-achievable response: |g| median %.3f, drive |g| median %.3f, '
      '%.0f%% of drive pairs above the 0.95 cap'
      % (FLOOR['resp_med'], FLOOR['drive_med'], FLOOR['drive_over_cap']))

rows = []
for name in sorted(n for n in os.listdir(RUNS) if n.startswith('run') and n.endswith('_spec.nc4')):
    ds = netCDF4.Dataset(os.path.join(RUNS, name)); g = ds.groups['Frame 6x12 Random']
    S = np.array(g['specification_cpsd_matrix_real'][:])
    R = np.array(g['response_cpsd_real'][:]) + 1j*np.array(g['response_cpsd_imag'][:])
    D = np.array(g['drive_cpsd_real'][:]) + 1j*np.array(g['drive_cpsd_imag'][:])
    law = g.getncattr('control_python_function')
    par = [p.strip() for p in str(g.getncattr('control_python_function_parameters')).split(',')]
    ds.close()
    inb = np.einsum('fii->f', S) > 0
    def gam(X, n):
        Xb = X[inb]; d = np.real(np.einsum('fii->fi', Xb)); i, j = np.triu_indices(n, 1)
        return np.abs(Xb[:, i, j]) / np.sqrt(np.maximum(d[:, i]*d[:, j], 1e-300))
    gr, gd = gam(R, 8), gam(D, 6)
    w = np.maximum(np.linalg.eigvalsh(R[inb]), 0)
    rows.append(dict(run=name[:5], law=SH[law],
                     loop='open' if law in OPEN else 'feedback',
                     cap='off' if float(par[CAPPOS[law]]) >= 1.0 else 'on',
                     gr=gr.ravel(), gd=gd.ravel(),
                     part=float(np.median(w.sum(1)**2/np.maximum((w**2).sum(1), 1e-300)))))

rows.sort(key=lambda r: np.median(r['gr']))
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                         gridspec_kw={'width_ratios': [1.35, 1]})

a = axes[0]
for y, r in enumerate(rows):
    c = LOOP[r['loop']]
    q = np.percentile(r['gr'], [10, 25, 50, 75, 90])
    a.plot([q[0], q[4]], [y, y], color=c, lw=1.4, alpha=0.5, solid_capstyle='round')
    a.plot([q[1], q[3]], [y, y], color=c, lw=5.5, alpha=0.85, solid_capstyle='butt')
    a.plot([q[2]], [y], 'o', ms=7, color=SURF, markeredgecolor=c, markeredgewidth=2, zorder=4)
a.axvline(FLOOR['resp_med'], color=INK2, lw=1.4, ls='--')
a.text(FLOOR['resp_med'] - 0.018, -0.45,
       'best achievable\nresponse (%.2f)' % FLOOR['resp_med'],
       fontsize=7.6, color=INK2, ha='right', va='bottom')
a.axvline(0, color=INK2, lw=1.2)
a.text(0.014, -0.45, 'specification\n(zero cross-terms)', fontsize=7.6,
       color=INK2, va='bottom')
a.set_yticks(range(len(rows)))
a.set_yticklabels([f"{r['run'][3:]}  {r['law']}  cap {r['cap']}" for r in rows], fontsize=8)
a.set_xlim(-0.03, 1.05)
a.set_xlabel('response cross-channel coherence |γ|   (28 pairs × 901 lines)')
a.set_ylim(len(rows)-0.4, -1.25); a.grid(axis='x')
a.set_title('Response: nothing gets near the zero the spec asked for\nand the best '
            'possible response is the most correlated of all',
            loc='left', fontsize=10, color=INK, pad=10)

b = axes[1]
for y, r in enumerate(rows):
    c = LOOP[r['loop']]
    q = np.percentile(r['gd'], [10, 25, 50, 75, 90])
    b.plot([q[0], q[4]], [y, y], color=c, lw=1.4, alpha=0.5, solid_capstyle='round')
    b.plot([q[1], q[3]], [y, y], color=c, lw=5.5, alpha=0.85, solid_capstyle='butt')
    b.plot([q[2]], [y], 'o', ms=7, color=SURF, markeredgecolor=c, markeredgewidth=2, zorder=4)
b.axvline(0.95, color=INK2, lw=1.4, ls=':')
b.text(0.938, -0.45, 'cap 0.95', fontsize=7.6, color=INK2, ha='right', va='bottom')
b.set_yticks(range(len(rows))); b.set_yticklabels([])
b.set_xlim(-0.03, 1.05)
b.set_xlabel('drive cross-channel coherence |γ|   (15 pairs × 901 lines)')
b.set_ylim(len(rows)-0.4, -1.25); b.grid(axis='x')
h = [plt.Line2D([], [], color=LOOP['feedback'], lw=5, label='feedback law'),
     plt.Line2D([], [], color=LOOP['open'], lw=5, label='open-loop law')]
a.legend(handles=h, frameon=False, fontsize=8, loc='lower left')
b.set_title('Drive: the cap trims the top decile only\n(the optimal drive would '
            'breach it on half its pairs)',
            loc='left', fontsize=10, color=INK, pad=10)

fig.savefig(os.path.join(OUT, 'cross_terms.png'), dpi=170, bbox_inches='tight',
            facecolor=SURF)
json.dump(dict(floor=FLOOR,
               runs=[{k: (float(np.median(r['gr'])) if k == 'resp_med' else
                          float(np.percentile(r['gr'], 90)) if k == 'resp_p90' else
                          float(np.median(r['gd'])) if k == 'drive_med' else
                          float(np.percentile(r['gd'], 90)) if k == 'drive_p90' else
                          float(100*np.mean(r['gd'] > 0.95)) if k == 'drive_over_cap' else
                          r[k])
                      for k in ('run', 'law', 'loop', 'cap', 'part',
                                'resp_med', 'resp_p90', 'drive_med', 'drive_p90',
                                'drive_over_cap')}
                     for r in rows]),
          open('crossterm_data.json', 'w'), indent=1)
print('wrote figs/cross_terms.png and crossterm_data.json')
print(f'{"run":6}{"law":15}{"cap":5}{"resp med":>9}{"resp p90":>9}{"drv med":>9}{"eig part":>9}')
for r in rows:
    print(f"{r['run']:6}{r['law']:15}{r['cap']:5}{np.median(r['gr']):9.3f}"
          f"{np.percentile(r['gr'],90):9.3f}{np.median(r['gd']):9.3f}{r['part']:9.2f}")
