"""
Plots of the octave law with the exact band cross-power (octave_exact_c.OctaveExactC), per line and
as 1/6-octave band averages, and of the shipped octave law as band averages.  Same layout as
plot_law_spectra.py.  Run in the 'rattlesnake' env with PYTHONPATH=src, from the repository root:

    python control_laws/tests/plot_octave_exact_c.py

Writes spectra_10x14_octave_exactC_experiment_{lines,bands}.png and spectra_10x14_octave_law_bands.png into
examples/tendrive14resp/results/plots/ and prints the in-band-lines score table.
"""
import sys, os, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from octave_exact_c import OctaveExactC, pl, cf, frame_plant
from achievable_response import achievable_diagonal
p = frame_plant('10x14'); spec_d = np.real(np.einsum('fii->fi', p['spec']))
names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
out = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'examples', 'tendrive14resp', 'results', 'plots')
best = achievable_diagonal(p['H'], spec_d, line_indices=np.flatnonzero(p['band']))['achieved']
laws = pl.laws()
X = {lab: cf.run(p, cls, par, 15, 20)['X'] for lab, cls, par in laws[:2]}          # SDP, fast -> the given-up set
fin = np.array([np.sqrt(np.mean((10*np.log10(pl.response(p['H'], x)[p['band']]/spec_d[p['band']]))**2, axis=0)) for x in X.values()])
lost = np.all(fin > 3.0, axis=0)
shipped = cf.run(p, laws[2][1], pl.OCTAVE_PARAMS, 15, 20)['X']
exact = cf.run(p, OctaveExactC, pl.OCTAVE_PARAMS, 15, 20)['X']
f = p['f']
with contextlib.redirect_stdout(io.StringIO()):
    mod = pl.load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))
cen, lo, hi = mod._octave_band_frequencies(100.0, 1000.0, 6)
pairs = [(m, c) for m, c in zip(((f >= l) & (f < h) & p['band'] for l, h in zip(lo, hi)), cen) if m.any()]
bands = ([m for m, _ in pairs], np.array([c for _, c in pairs]))
T1 = 'EXPERIMENT, not a repo law: OCTAVE law with exact band cross-power averaging (octave_exact_c)'
T0 = laws[2][0]                                  # the octave law as shipped
for title, Xx, slug, view, b in ((T1, exact, 'octave_exactC_experiment', 'lines', None), (T1, exact, 'octave_exactC_experiment', 'bands', bands),
                                 (T0, shipped, 'octave_law', 'bands', bands)):
    path = pl.plot_path(out, '10x14', slug, view)
    pl.figure(title, p, Xx, best, lost, names, '10x14', path, bands=b); print('wrote', path)

reach = ~lost; everyone = np.ones(lost.size, bool)
rms = lambda a, c: float(np.sqrt(np.mean(a[:, c]**2)))
print(f"\n{'in-band lines only; 1/6-octave bands':44s} {'line,11':>8s} {'line,14':>8s} {'band,11':>8s} {'band,14':>8s}")
for lab, Xx in (('SDP', X[laws[0][0]]), ('fast', X[laws[1][0]]), ('octave as shipped (mean of H)', shipped), ('octave, exact band cross-power', exact)):
    y = pl.response(p['H'], Xx); line = 10*np.log10(y[p['band']]/spec_d[p['band']])
    band = np.array([10*np.log10(y[m].mean(0)/spec_d[m].mean(0)) for m in bands[0]])
    print(f"{lab:44s} {rms(line, reach):8.2f} {rms(line, everyone):8.2f} {rms(band, reach):8.2f} {rms(band, everyone):8.2f}")
bb = np.array([10*np.log10(best[m].mean(0)/spec_d[m].mean(0)) for m in bands[0]])
print(f"{'best achievable (per-line optimum over 14 ch)':44s} {'':8s} {'':8s} {rms(bb, reach):8.2f} {rms(bb, everyone):8.2f}")
