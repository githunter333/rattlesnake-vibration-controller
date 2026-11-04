"""
band_average: the version in this repo's control_laws/ against the 2026-11-04 version in
rattlesnake-band-average, on the 10x14 frame plant, closed loop (the same runs as
bandavg_gap.py and plot_bandavg_bands.py).

The new version chooses the regularization threshold kappa band by band and scores each
candidate with the solve the law outputs (PSD re-fit included), which is what bandavg_gap.py
found the old sweep did not do.  This script runs both, with the old law at its automatic
kappa (18 on this plant) and at kappa 100, and the new law with each kappa_rule, and scores
them in the law's own bands:

  rms dB   all 14 channels / the 11 the SDP law reaches, 1/6-octave bands as the law defines
           them (T_narrow_to_octave @ response against spec_asd_octave, covered bands);
           the same per line; the drive level; and the time of the first solve (the sweep)

    python control_laws/tests/bandavg_versions.py [path/to/band_average_2026_11_04.py]

Run in the 'rattlesnake' env with PYTHONPATH=src, from the repository root.  The default path is
~/Code/python/rattlesnake-band-average/control_laws/band_average_2026_11_04.py.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import compare_optdiag_frame as cf                   # noqa: E402
import plot_law_spectra as pl                        # noqa: E402
from frame_plant import frame_plant                  # noqa: E402

NEW_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Code/python/rattlesnake-band-average/control_laws/band_average_2026_11_04.py')

p = frame_plant('10x14')
H, sel = p['H'], p['band']
names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
reach = np.array([n not in ('7x', '13x', '14x') for n in names])         # the SDP/fast laws' given-up set
everyone = np.ones(len(names), bool)
OLD = pl.laws()[3][1]
with contextlib.redirect_stdout(io.StringIO()):
    NEW = pl.load_control_module(NEW_PATH).band_average
spec = np.real(np.einsum('fii->fi', p['spec']))


def build(cls, params):
    """(law after its first solve, seconds that took, X of that first solve)."""
    args = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        law = cls(specification=p['spec'], warning_levels=p['warn'], abort_levels=p['abort'], extra_parameters=params)
        t0 = time.perf_counter()
        law.system_id_update(H, *args, p['coh'], 20, 20)
        X1 = law.control(H, p['coh'], 20, 20, None, None)
        dt = time.perf_counter() - t0
    return law, dt, X1, out.getvalue()


def own_err(law, X):
    yo = law.T_narrow_to_octave @ pl.response(H, X)
    cov = law.spec_asd_octave.max(axis=1) > 0
    return 10*np.log10(np.clip(yo[cov], 1e-300, None)/law.spec_asd_octave[cov])


def line_err(X):
    return 10*np.log10(pl.response(H, X)[sel]/spec[sel])


rms = lambda a, c: float(np.sqrt(np.mean(a[:, c]**2)))
RUNS = [
    ('OLD  auto kappa (as shipped)', OLD, pl.BAND_PARAMS),
    ('OLD  condition_number 100', OLD, pl.BAND_PARAMS + '\ncondition_number: 100'),
    ('NEW  kappa_rule front (default)', NEW, pl.BAND_PARAMS),
    ('NEW  kappa_rule knee', NEW, pl.BAND_PARAMS + '\nkappa_rule: knee'),
    ('NEW  kappa_rule lcurve', NEW, pl.BAND_PARAMS + '\nkappa_rule: lcurve'),
    ('NEW  condition_number 100', NEW, pl.BAND_PARAMS + '\ncondition_number: 100'),
]

if __name__ == '__main__':
    print(f"new law: {NEW_PATH}\n")
    print(f"{'':36s} {'bands, own def.':>17s}   {'per line':>15s} {'drive':>7s} {'1st solve':>10s}")
    print(f"{'':36s} {'all14 / 11 reach':>17s}   {'all14 / 11 reach':>15s} {'(dB)':>7s} {'(s)':>10s}")
    for label, cls, params in RUNS:
        law, dt, X1, log = build(cls, params)
        with contextlib.redirect_stdout(io.StringIO()):
            Xc = cf.run(p, cls, params, 15, 20)['X']
        eb, el = own_err(law, Xc), line_err(Xc)
        drive = 10*np.log10(np.mean(np.real(np.einsum('fii->f', Xc))[sel]))
        print(f"{label:36s} {rms(eb, everyone):7.2f} / {rms(eb, reach):<7.2f}   {rms(el, everyone):6.2f} / {rms(el, reach):<6.2f} {drive:7.1f} {dt:10.2f}")
        note = [ln for ln in log.splitlines() if 'condition_number' in ln]
        if note:
            print(f"{'':36s} {note[0].strip()}")
