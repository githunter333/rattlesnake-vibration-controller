"""
The five laws scored three ways, and the band_average law plotted as 1/6-octave band averages.

Scores (rms dB vs the spec) for 11 reachable channels and for all 14:
  base-2 bands          the octave law's band edges, 100-1000 Hz lines only
  law bands, in-band    band_average's own ANSI base-ten edges, 100-1000 Hz lines only
  law bands, law def.   band_average's own definition: T_narrow_to_octave @ response, so a
                        partly covered edge band reads low on both sides
Run in the 'rattlesnake' env with PYTHONPATH=src, from the repository root:

    python control_laws/tests/plot_bandavg_bands.py

Writes spectra_10x14_band_average_law_bands.png and ..._band_average_law_capped_bands.png into
examples/tendrive14resp/results/plots/.  The band_average runs use df 5, oct_order 6 and the drive
RMS limit off (the plant's drive units are arbitrary).
"""
import sys, os, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import compare_optdiag_frame as cf, plot_law_spectra as pl
from frame_plant import frame_plant
from achievable_response import achievable_diagonal
p = frame_plant('10x14'); H, f, sel = p['H'], p['f'], p['band']
spec_d = np.real(np.einsum('fii->fi', p['spec']))
names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
out = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'examples', 'tendrive14resp', 'results', 'plots')
laws = pl.laws()
runs = [(lab, cf.run(p, cls, par, 15, 20)['X']) for lab, cls, par in laws]
fin = np.array([np.sqrt(np.mean((10*np.log10(pl.response(H, X)[sel]/spec_d[sel]))**2, axis=0)) for _, X in runs[:2]])
lost = np.all(fin > 3.0, axis=0); best, joint = pl.reference_spectra(p, lost); reach = ~lost; everyone = np.ones(lost.size, bool)

# the band-average law's own band machinery
with contextlib.redirect_stdout(io.StringIO()):
    law = laws[3][1](specification=p['spec'], warning_levels=p['warn'], abort_levels=p['abort'], extra_parameters=pl.BAND_PARAMS)
T, spec_oct = law.T_narrow_to_octave, law.spec_asd_octave                  # (bands, lines), (bands, ctrl)
covered = spec_oct.max(axis=1) > 0
print(f"law's bands overlapping the spec: {int(covered.sum())} (edges {law.band_lb[covered][0]:.1f}-{law.band_ub[covered][-1]:.1f} Hz)")
# (B) law's band edges, in-band lines only
masks10 = [(law._b_of == b) & law._inb & sel for b in np.flatnonzero(covered)]
masks10 = [m for m in masks10 if m.any()]
cent10 = np.array([np.sqrt(law.band_lb[b]*law.band_ub[b]) for b in np.flatnonzero(covered)])[:len(masks10)]
# base-2 bands used earlier
with contextlib.redirect_stdout(io.StringIO()):
    mod = pl.load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))
_, lo, hi = mod._octave_band_frequencies(100.0, 1000.0, 6)
masks2 = [m for m in ((f >= l) & (f < h) & sel for l, h in zip(lo, hi)) if m.any()]

rms = lambda a, c: float(np.sqrt(np.mean(a[:, c]**2)))
def band_err(y, masks): return np.array([10*np.log10(y[m].mean(0)/spec_d[m].mean(0)) for m in masks])
def own_err(y):                                                            # (A): the law's own definition
    yo = T @ y
    return 10*np.log10(np.maximum(yo[covered], 1e-300)/spec_oct[covered])
print(f"\n{'rms dB error vs spec':72s} {'base-2 bands':>16s} {'law bands, in-band lines':>26s} {'law bands, law definition':>27s}")
print(f"{'':72s} {'11 reach':>8s} {'all 14':>7s} {'11 reach':>13s} {'all 14':>12s} {'11 reach':>14s} {'all 14':>12s}")
for lab, X in runs:
    y = pl.response(H, X)
    e2, e10, eo = band_err(y, masks2), band_err(y, masks10), own_err(y)
    print(f"{lab:72s} {rms(e2, reach):8.2f} {rms(e2, everyone):7.2f} {rms(e10, reach):13.2f} {rms(e10, everyone):12.2f} {rms(eo, reach):14.2f} {rms(eo, everyone):12.2f}")
for lab, ref in (('best achievable (reachable held on target)', best), ('joint 14-channel compromise', joint)):
    b2, b10 = band_err(ref, masks2), band_err(ref, masks10)
    print(f"{lab:72s} {rms(b2, reach):8.2f} {rms(b2, everyone):7.2f} {rms(b10, reach):13.2f} {rms(b10, everyone):12.2f}")

for (lab, X), slug in ((runs[3], pl.SLUGS[3]), (runs[4], pl.SLUGS[4])):
    path = pl.plot_path(out, '10x14', slug, 'bands')
    pl.figure(lab, p, X, best, lost, names, '10x14', path, bands=(masks10, cent10), joint=joint); print('wrote', path)
