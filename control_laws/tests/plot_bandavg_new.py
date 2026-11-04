"""
Plots of the 2026-11-04 band_average law (rattlesnake-band-average repo) on the 10x14 frame plant,
in the layout of plot_law_spectra.py: target / best achievable / control, one panel per control
channel, per line and as band averages in the law's own 1/6-octave bands.

    python control_laws/tests/plot_bandavg_new.py [path/to/band_average_2026_11_04.py]

Writes spectra_10x14_band_average_law_2026_11_04_{lines,bands}.png into
examples/tendrive14resp/results/plots/ and prints the law's score under the three band
definitions of plot_bandavg_bands.py, next to the SDP and fast laws and the two references.
The law runs its default kappa_rule ('front') with df 5, oct_order 6 and the drive RMS limit
off (the plant's drive units are arbitrary).  Run in the 'rattlesnake' env with PYTHONPATH=src,
from the repository root.
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

NEW_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Code/python/rattlesnake-band-average/control_laws/band_average_2026_11_04.py')
SLUG = 'band_average_law_2026_11_04'
TITLE = 'BAND-AVERAGE law 2026-11-04 (band_average_2026_11_04): kappa chosen per band from the real solve'

p = frame_plant('10x14')
H, f, sel = p['H'], p['f'], p['band']
spec = np.real(np.einsum('fii->fi', p['spec']))
names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
out = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'examples', 'tendrive14resp', 'results', 'plots')
os.makedirs(out, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    NEW = pl.load_control_module(NEW_PATH).band_average

laws = pl.laws()
refs = {lab: cf.run(p, cls, par, 15, 20)['X'] for lab, cls, par in laws[:2]}          # SDP, fast
fin = np.array([np.sqrt(np.mean((10*np.log10(pl.response(H, X)[sel]/spec[sel]))**2, axis=0)) for X in refs.values()])
lost = np.all(fin > 3.0, axis=0)
reach, everyone = ~lost, np.ones(lost.size, bool)
best, joint = pl.reference_spectra(p, lost)
with contextlib.redirect_stdout(io.StringIO()):
    X = cf.run(p, NEW, pl.BAND_PARAMS, 15, 20)['X']
    law = NEW(specification=p['spec'], warning_levels=p['warn'], abort_levels=p['abort'], extra_parameters=pl.BAND_PARAMS)

# the law's own bands: its edges, 100-1000 Hz lines only
cov = np.flatnonzero(law.spec_asd_octave.max(axis=1) > 0)
pairs = [((law._b_of == b) & law._inb & sel, np.sqrt(law.band_lb[b]*law.band_ub[b])) for b in cov]
pairs = [(m, c) for m, c in pairs if m.any()]
bands = ([m for m, _ in pairs], np.array([c for _, c in pairs]))
with contextlib.redirect_stdout(io.StringIO()):
    mod = pl.load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))
_, lo, hi = mod._octave_band_frequencies(100.0, 1000.0, 6)
masks2 = [m for m in ((f >= l) & (f < h) & sel for l, h in zip(lo, hi)) if m.any()]

for view, b in (('lines', None), ('bands', bands)):
    path = pl.plot_path(out, '10x14', SLUG, view)
    pl.figure(TITLE, p, X, best, lost, names, '10x14', path, bands=b, joint=joint)
    print('wrote', path)

rms = lambda a, c: float(np.sqrt(np.mean(a[:, c]**2)))
band_err = lambda y, ms: np.array([10*np.log10(y[m].mean(0)/spec[m].mean(0)) for m in ms])
own = lambda y: 10*np.log10(np.clip((law.T_narrow_to_octave @ y)[cov], 1e-300, None)/law.spec_asd_octave[cov])
print(f"\nrms dB error vs the spec, 11 reachable / all 14 channels ({len(masks2)} base-2 bands, {len(bands[0])} law bands)")
print(f"{'':56s} {'per line':>13s} {'base-2 bands':>14s} {'law bands':>14s} {'law definition':>15s}")
rows = [('SDP law', refs[laws[0][0]]), ('FAST law', refs[laws[1][0]]), ('BAND-AVERAGE 2026-11-04 (front)', X)]
for lab, Xr in rows:
    y = pl.response(H, Xr)
    cells = [10*np.log10(y[sel]/spec[sel]), band_err(y, masks2), band_err(y, bands[0]), own(y)]
    print(f"{lab:56s} " + ' '.join(f'{rms(c, reach):5.2f} / {rms(c, everyone):<5.2f}' for c in cells))
for lab, y in (('best achievable (reachable held on target)', best), ('joint 14-channel compromise', joint)):
    cells = [10*np.log10(y[sel]/spec[sel]), band_err(y, masks2), band_err(y, bands[0])]
    print(f"{lab:56s} " + ' '.join(f'{rms(c, reach):5.2f} / {rms(c, everyone):<5.2f}' for c in cells))
