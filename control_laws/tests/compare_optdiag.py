"""
optimal_diagonal_control (SDP per line) vs optimal_diagonal_control_fast
(factored solve per line), closed loop on the coherence-sweep simulator.

Same synthetic 8-control x 6-drive plant, same scenarios and finite-frame
CPSD estimates as coherence_sweep.py.  Both laws get the SAME parameter
string, so any difference comes from the per-line solver and what it allows:

  optimal_diagonal_control       convex SDP, drive coherence capped at
                                 max_drive_coherence (0.95), linear or dB
                                 objective, no drive-subspace restriction
  optimal_diagonal_control_fast  unconstrained factored solve (no coherence
                                 cap), drive restricted to the strongest
                                 singular directions (drive_rcond, per-line
                                 floor svd_min_keep)

The simulator hands every law the SAME (system-ID) FRF every cycle, so the
fast law's movement gate never fires and it stays on the fast path
throughout.  That is the configuration with "Update Transfer Function During
Control" OFF.  With it ON, a moving FRF demotes the fast law to the SDP.

Reported per law and scenario:
  err      rms dB error of the TRUE response autospectra vs spec, 100-380 Hz,
           at the end of the run
  cyc<1dB  first cycle at which the error is within 1 dB of its final value
  drive    mean drive trace, dB (relative units)
  coh      median / max pairwise drive coherence over in-band lines
  s/call   mean wall time per control() call (the first call, which also
           does the initial solves, is reported separately as 'init s')

Usage: python control_laws/tests/compare_optdiag.py [n_cycles] [n_frames] [bins]
   bins = max_bins_per_update for both laws (default 20, the law's default)
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import coherence_sweep as cs                          # noqa: E402
from rattlesnake.utilities import load_control_module  # noqa: E402

# Full parameter string (the parser drops empty fields, so every field is
# written out).  {bins}, {dom} and {rc} are filled per run.
PARAMS = '1e-6,0.5,{bins},1.0,0.95,4,-9,{dom},2,{rc},0,3,0.5,0.5,0.5,0,3,1e-6,0.05,0'


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        sdp = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control.py'))
        fast = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control_fast.py'))
    return sdp.optimal_diagonal_control, fast.optimal_diagonal_control_fast


def drive_coherence(X, band):
    Xb = X[band]
    d = np.real(np.einsum('fii->fi', Xb))
    den = d[:, :, None]*d[:, None, :]
    coh = np.abs(Xb)**2/np.where(den > 0, den, 1.0)
    iu = np.triu_indices(X.shape[1], 1)
    c = coh[:, iu[0], iu[1]].ravel()
    return float(np.median(c)), float(np.max(c))


def run_one(p, law_cls, params, n_cycles, n_frames, seed=1):
    rng = np.random.default_rng(seed)
    Hm, spec = p['H_model'], p['spec']
    nf = n_frames if n_frames > 0 else 1000
    coh1 = np.ones((cs.N_FREQ, cs.N_RESP))
    args = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
    f = np.arange(cs.N_FREQ)*1.0
    sub = (f >= 100) & (f <= 380) & p['band']

    def score(X):
        H = p['H_true']
        y = np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj())) + p['noise_diag']
        e = 10*np.log10(np.maximum(y[sub], 1e-300)/p['spec_diag'][sub])
        return float(np.sqrt(np.mean(e**2)))

    hist, times = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        law = law_cls(spec, p['warn'], p['abort'], params, None, *args, None, 0, nf, None, None)
        t0 = time.perf_counter()
        law.system_id_update(Hm, *args, coh1, nf, nf)
        X = law.control(Hm, coh1, nf, nf, None, None)
        t_init = time.perf_counter() - t0
        hist.append(score(X))
        for _ in range(n_cycles):
            Gyy, Gxx, coh = cs.measure(p['H_true'], X, p['noise_diag'], n_frames, rng)
            t0 = time.perf_counter()
            X = law.control(Hm, coh, nf, nf, Gyy, Gxx)
            times.append(time.perf_counter() - t0)
            hist.append(score(X))
    h = np.array(hist)
    within = np.where(np.abs(h - h[-1]) <= 1.0)[0]
    # first cycle from which it STAYS within 1 dB of the final value
    settle = next((int(i) for i in within if np.all(np.abs(h[i:] - h[-1]) <= 1.0)), len(h) - 1)
    drv = 10*np.log10(np.mean(np.real(np.einsum('fii->f', X))[sub]))
    cmed, cmax = drive_coherence(X, p['band'])
    extra = {}
    if hasattr(law, 'n_fast_solves'):
        extra = {'fast': law.n_fast_solves, 'sdp': law.n_safe_solves}
    return dict(err=h[-1], settle=settle, drive=drv, coh_med=cmed, coh_max=cmax,
                t_call=float(np.mean(times)) if times else 0.0, t_init=t_init,
                refined=int(np.sum(law.sdp_refined)), extra=extra, hist=h)


def main():
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    bins = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    SDP, FAST = load()
    cases = [('clean', 0.0), ('noise', 0.5), ('nonlinear', 0.05), ('illcond', 1.0)]
    configs = [
        ('linear objective, drive_rcond 2.0 (defaults)', 0, 2.0),
        ('linear objective, drive_rcond 0 (fast unrestricted)', 0, 0.0),
        ('dB objective, drive_rcond 2.0', 1, 2.0),
    ]
    print(f'{n_cycles} cycles, ' + (f'{n_frames}-frame CPSD estimates' if n_frames else
                                    'exact CPSDs') + f', max_bins_per_update {bins}')
    print('err = rms dB error 100-380 Hz at the end; cyc<1dB = cycles until it stays within '
          '1 dB of that; coh = median/max drive coherence\n')
    hdr = (f"{'case':15s} {'law':5s} {'err':>6s} {'cyc<1dB':>8s} {'drive':>7s} {'coh med/max':>12s} "
           f"{'init s':>7s} {'s/call':>7s}  solves")
    for cname, dom, rc in configs:
        params = PARAMS.format(bins=bins, dom=dom, rc=rc)
        print(f'=== {cname}\n    {params}')
        print(hdr)
        print('-'*len(hdr))
        for name, lvl in cases:
            p = cs.scenario(name, lvl)
            label = name if name == 'clean' else f'{name} {lvl:g}'
            for lname, cls in [('SDP', SDP), ('fast', FAST)]:
                r = run_one(p, cls, params, n_cycles, n_frames)
                sol = (f"{r['extra']['fast']} fast / {r['extra']['sdp']} SDP" if r['extra']
                       else 'all SDP')
                print(f"{label:15s} {lname:5s} {r['err']:6.2f} {r['settle']:8d} {r['drive']:7.1f} "
                      f"{r['coh_med']:5.2f}/{r['coh_max']:5.2f} {r['t_init']:7.2f} "
                      f"{r['t_call']:7.3f}  {sol}", flush=True)
        print()


if __name__ == '__main__':
    main()
