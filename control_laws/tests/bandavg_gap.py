"""
Why does band_average miss its own prediction?  (2026-11-03)

The law's knee sweep (band_average._resolve_auto_kappa) predicts the MEAN |dB| band error over
all channels and covered bands, from the band-space model r_b = A_b vec(P_b).  The plots and score
tables use RMS dB.  This script scores one run both ways, in the law's own band definition
(T_narrow_to_octave @ response vs spec_asd_octave, covered bands, all channels), and sweeps the
condition number, to separate a difference of metric from a real shortfall of the solve.

    python control_laws/tests/bandavg_gap.py        (rattlesnake env, PYTHONPATH=src, repo root)
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import compare_optdiag_frame as cf                   # noqa: E402
import plot_law_spectra as pl                        # noqa: E402
from frame_plant import frame_plant                  # noqa: E402

p = frame_plant('10x14')
H, sel = p['H'], p['band']
spec = np.real(np.einsum('fii->fi', p['spec']))
names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
Band = pl.laws()[3][1]
HARD = [names.index(n) for n in ('7x', '13x', '14x')]
reach = np.ones(len(names), bool)
reach[HARD] = False
everyone = np.ones(len(names), bool)


def own_errors(law, X):
    """(covered bands, channels) dB error in the law's own band definition."""
    y = pl.response(H, X)
    yo = law.T_narrow_to_octave @ y
    cov = law.spec_asd_octave.max(axis=1) > 0
    return 10*np.log10(np.clip(yo[cov], 1e-300, None)/law.spec_asd_octave[cov])


def run(params, n_cycles=15):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = cf.run(p, Band, params, n_cycles, 20)
        law = Band(specification=p['spec'], warning_levels=p['warn'], abort_levels=p['abort'], extra_parameters=params)
        args = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
        law.system_id_update(H, *args, p['coh'], 20, 20)
        X1 = law.control(H, p['coh'], 20, 20, None, None)            # the first solve, no loop
    return law, r['X'], X1, out.getvalue()


def row(label, law, X):
    e = own_errors(law, X)
    mean_abs = lambda a: float(np.mean(np.abs(a)))
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    drive = 10*np.log10(np.mean(np.real(np.einsum('fii->f', X))[sel]))
    print(f'{label:34s} {mean_abs(e):9.2f} {rms(e):9.2f} {mean_abs(e[:, reach]):10.2f} {rms(e[:, reach]):10.2f} {drive:8.1f}')


if __name__ == '__main__':
    print(f"{'':34s} {'mean|dB|':>9s} {'rms dB':>9s} {'mean|dB|':>10s} {'rms dB':>10s} {'drive':>8s}")
    print(f"{'':34s} {'all 14':>9s} {'all 14':>9s} {'11 reach':>10s} {'11 reach':>10s} {'(dB)':>8s}")
    law, Xc, X1, log = run(pl.BAND_PARAMS)
    ak = law.auto_kappa
    i = int(np.argmin(np.abs(ak['grid'] - ak['selected'])))
    print(f"law's prediction at its chosen kappa {ak['selected']:.0f}: mean|dB| {ak['error_db'][i]:.2f} (best on the grid {ak['error_db'].min():.2f})")
    row('auto kappa, first solve (no loop)', law, X1)
    row('auto kappa, after 15 cycles', law, Xc)
    for label, extra in (('kappa 30', 'condition_number: 30'), ('kappa 100', 'condition_number: 100'),
                         ('kappa 1e3', 'condition_number: 1000'), ('kappa 1e4', 'condition_number: 10000'),
                         ('kappa 1e5', 'condition_number: 100000'), ('regularization: none', 'regularization: none')):
        law, Xc, X1, log = run(pl.BAND_PARAMS + '\n' + extra)
        row(label + ', first solve', law, X1)

    # ---- the sweep's prediction against the real solve, kappa by kappa ----------------------
    law, Xc, X1, log = run(pl.BAND_PARAMS)
    ak = law.auto_kappa
    cov = law.spec_asd_octave.max(axis=1) > 0

    def real_solve_error(kappa):
        """The sweep's own metric (mean |dB| over covered bands and all channels, ratio clipped at
        1e-6) evaluated on what the real solve, _solve_bands(kappa), produces."""
        out = law._solve_bands(kappa=kappa)
        yo = (law.T_narrow_to_octave @ pl.response(H, out))[cov]
        ratio = np.clip(yo/law.spec_asd_octave[cov], 1e-6, None)
        return float(np.mean(np.abs(10*np.log10(ratio))))

    grid = ak['grid']
    real = np.array([real_solve_error(k) for k in grid])
    print("\nmean |dB| over all channels, by condition number kappa (the sweep's metric):")
    print(f"{'kappa':>8s} {'sweep predicts':>15s} {'real solve gives':>17s}")
    for k in (10, 18, 30, 56, 100, 178, 316, 1000, 1e4, 1e5):
        j = int(np.argmin(np.abs(np.log(grid/k))))
        print(f"{grid[j]:8.0f} {ak['error_db'][j]:15.2f} {real[j]:17.2f}")
    knee_real = float(grid[int(np.argmax(real <= real.min() + 0.1))])
    print(f"\nknee chosen by the sweep: kappa {ak['knee']:.0f};  knee if the real solve were used: kappa {knee_real:.0f}")
    law2, Xc2, X12, _ = run(pl.BAND_PARAMS + f'\ncondition_number: {knee_real:.0f}')
    row(f'kappa {knee_real:.0f} (real-solve knee), first solve', law2, X12)
    row(f'kappa {knee_real:.0f} (real-solve knee), 15 cycles', law2, Xc2)
