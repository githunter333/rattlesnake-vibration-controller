"""
EXPERIMENT (not a control law): octave_band_switching_control with the exact band cross-power.

The octave law solves each 1/6-octave band against the plain complex mean of H over the band's
lines.  The response a flat-in-band drive X produces, averaged over the band, is instead
    mean_lines( h_m X h_m^H )  =  sum_ij C[m,i,j] X[i,j],   C[m] = mean_lines( h_m^H-outer h_m ),
which needs only the per-line H the law already receives.  OctaveExactC swaps C in, in both the
solve and the scheduler's error estimate, and nothing else.

    python control_laws/tests/octave_exact_c.py        (rattlesnake conda env, PYTHONPATH=src)

prints the shipped law and this variant, with 20-frame and exact CPSDs, scored per line and in
1/6-octave bands (100-1000 Hz lines only).  10x14 plant, 15 cycles.  2026-11-03: the band error
on the 11 reachable channels fell from 2.21 to 0.32 dB.
"""
import sys, os, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import cvxpy as cp
import compare_optdiag_frame as cf, plot_law_spectra as pl
from frame_plant import frame_plant

OCT = pl.laws()[2][1].__mro__[1]            # octave_band_switching_control (OctaveEngaged's parent)

class OctaveExactC(OCT):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_test_level_db(0.0)

    def _band_average_H(self, H_clean):
        Hb = super()._band_average_H(H_clean)
        self._Hb = Hb
        C = []
        for lines in self._lines_of_band:
            if lines.size == 0:
                C.append(np.zeros((self.M, self.N, self.N), dtype=complex)); continue
            Hl = H_clean[lines]                                   # (L, M, N)
            C.append(np.einsum('lmi,lmj->mij', Hl, Hl.conj())/lines.size)
        self._Cb = np.stack(C)                                    # (bands, M, N, N)
        return Hb

    def _band_of(self, H):
        for b in range(len(self._Hb)):
            if np.array_equal(self._Hb[b], H):
                return b
        raise RuntimeError('band not found')

    def _err_db_oct(self, H_band_clean, indices):
        Y = np.real(np.einsum('bmij,bij->bm', self._Cb[indices], self._oct_output_cpsd[indices]))
        target = np.maximum(self._oct_y_target[indices], 1e-30)
        return np.max(np.abs(10*np.log10(np.maximum(Y, 1e-30)/target)), axis=1)

    def _solve_one_bin(self, H, y_target, X_warm=None):          # same as the base, with C in place of h h^H
        C = self._Cb[self._band_of(H)]
        M, N = H.shape
        if self._sdp_prob is None or self._sdp_shape != (M, N):
            self._sdp_prob, self._sdp_X, self._sdp_W_params, self._sdp_y_param = self._build_sdp_problem(M, N)
            self._sdp_shape = (M, N)
        w = np.ones(M)
        for m in range(M):
            self._sdp_W_params[m].value = w[m]*C[m]
        self._sdp_y_param.value = w*y_target
        self._sdp_prob.solve(solver=cp.CLARABEL, warm_start=True)
        return self._sdp_X.value

if __name__ == '__main__':
    p = frame_plant('10x14'); spec_d = np.real(np.einsum('fii->fi', p['spec']))
    f, H, sel = p['f'], p['H'], p['band']
    with contextlib.redirect_stdout(io.StringIO()):
        mod = pl.load_control_module(os.path.join(cf.LAW_DIR, 'octave_band_switching_control.py'))
    _, lo, hi = mod._octave_band_frequencies(100.0, 1000.0, 6)
    masks = [m for m in ((f >= l) & (f < h) & sel for l, h in zip(lo, hi)) if m.any()]      # 100-1000 Hz lines only

    sdp = dict(cf.load())['SDP']
    e = 10*np.log10(pl.response(H, cf.run(p, sdp, '', 15, 20)['X'])[sel]/spec_d[sel])
    reach = np.sqrt(np.mean(e**2, axis=0)) < 3.0
    everyone = np.ones(reach.size, bool)
    rms = lambda a, c: float(np.sqrt(np.mean(a[:, c]**2)))
    print(f"{'':44s} {'line,11':>8s} {'line,14':>8s} {'band,11':>8s} {'band,14':>8s}")
    for label, cls in (('octave, mean of H (as shipped)', pl.laws()[2][1]), ('octave, exact band cross-power', OctaveExactC)):
        for nf in (20, 0):
            r = cf.run(p, cls, pl.OCTAVE_PARAMS, 15, nf)
            y = pl.response(H, r['X'])
            line = 10*np.log10(y[sel]/spec_d[sel])
            band = np.array([10*np.log10(y[m].mean(0)/spec_d[m].mean(0)) for m in masks])
            print(f"{label + (' [exact CPSD]' if nf == 0 else ' [20 frames]'):44s} {rms(line, reach):8.2f} {rms(line, everyone):8.2f} {rms(band, reach):8.2f} {rms(band, everyone):8.2f}")
