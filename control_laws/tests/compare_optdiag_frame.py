"""
optimal_diagonal_control (SDP) vs optimal_diagonal_control_fast on a frame
plant (frame_plant.py), closed loop, comparing the default line budget with
letting the time budget decide.

Configurations (same for both laws, everything else at its default):
  default     max_bins_per_update 20, no time budget
  full band   max_bins_per_update 400, max_solve_ms_per_update 500

Each cycle the response is "measured" from n_frames Gaussian frames
(coherence_sweep.measure; 0 = exact CPSDs) and handed to the law, as
Rattlesnake would.  The FRF handed to the law never changes (FRF update off),
so the fast law stays on its fast path.

Reported per law and configuration:
  err       rms dB error of the TRUE response autospectra against the spec,
            over the specified lines (100-1000 Hz), after each cycle; split
            into the channels the law can reach (err) and the ones it gives up
            (the channels whose final self-predicted error is over 3 dB --
            13x and 14x on the 10x14 layout, 14 controls on 10 drives), which
            otherwise dominate the rms and hide convergence
  solved    lines solved so far (system_id_update's first batch included)
  ms/call   wall time of each control() call
  drive |coh|  median pairwise drive coherence MAGNITUDE |X_ij|/sqrt(X_ii X_jj) over the
            specified lines: the measure max_drive_coherence caps (square it for gamma^2)

Usage: python control_laws/tests/compare_optdiag_frame.py [plant] [n_cycles] [n_frames]
       defaults: 10x14 15 20
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
from frame_plant import frame_plant, summary          # noqa: E402
from rattlesnake.utilities import load_control_module  # noqa: E402

CONFIGS = [
    ('default (20 lines/call)', ''),
    ('full band (400 lines, 500 ms)', ',,400,,,,,,,,,,,,,,,,,,500'),
]


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        sdp = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control.py'))
        fast = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control_fast.py'))
    return [('SDP', sdp.optimal_diagonal_control), ('fast', fast.optimal_diagonal_control_fast)]


def run(p, cls, params, n_cycles, n_frames, seed=1):
    rng = np.random.default_rng(seed)
    H = p['H']
    noise_diag = np.real(np.einsum('fii->fi', p['gxx_noise']))
    sel = p['band']
    s_d = np.real(np.einsum('fii->fi', p['spec']))[sel]
    args = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
    nf = n_frames if n_frames > 0 else 1000

    def err(X):
        y = np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj()))[sel]
        return 10*np.log10(np.maximum(y, 1e-300)/s_d)       # (lines, channels) dB

    errs, ms, lines, solved = [], [], [], []
    n_solves = lambda law: law.n_fast_solves + law.n_safe_solves if hasattr(law, 'n_fast_solves') \
        else law.n_sdp_refinements + getattr(law, 'n_stale_refinements', 0) \
        if hasattr(law, 'n_sdp_refinements') else 0            # laws with no per-line solves (band_average)
    refined_of = lambda law: getattr(law, 'sdp_refined', np.zeros(p['f'].size, dtype=bool))
    with contextlib.redirect_stdout(io.StringIO()):
        law = cls(p['spec'], p['warn'], p['abort'], params, None, *args, None, 0, nf, None, None)
        t0 = time.perf_counter()
        law.system_id_update(H, *args, p['coh'], nf, nf)
        X = law.control(H, p['coh'], nf, nf, None, None)
        t_init = time.perf_counter() - t0
        errs.append(err(X))
        solved.append(int(np.sum(refined_of(law)[sel])))
        for _ in range(n_cycles):
            Gyy, Gxx, coh = cs.measure(H, X, noise_diag, n_frames, rng)
            s0 = n_solves(law)
            t0 = time.perf_counter()
            X = law.control(H, coh, nf, nf, Gyy, Gxx)
            ms.append(1e3*(time.perf_counter() - t0))
            lines.append(n_solves(law) - s0)
            errs.append(err(X))
            solved.append(int(np.sum(refined_of(law)[sel])))
    drive = 10*np.log10(np.mean(np.real(np.einsum('fii->f', X))[sel]))
    d = np.real(np.einsum('fii->fi', X))[sel]
    c = np.abs(X[sel])/np.sqrt(np.maximum(d[:, :, None]*d[:, None, :], 1e-300))
    iu = np.triu_indices(X.shape[1], 1)
    coh_med = float(np.median(c[:, iu[0], iu[1]]))
    refined = int(np.sum(refined_of(law)[sel]))
    return dict(X=X, errs=np.array(errs), solved=np.array(solved), ms=np.array(ms), lines=np.array(lines), t_init=t_init,
                drive=drive, coh=coh_med, refined=refined, n_sel=int(sel.sum()))


def main():
    plant = sys.argv[1] if len(sys.argv) > 1 else '10x14'
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    p = frame_plant(plant)
    print(summary(p))
    print(f'{n_cycles} cycles, ' + (f'{n_frames}-frame CPSD estimates' if n_frames else 'exact CPSDs') + '\n')
    laws = load()
    results = {}
    for cname, params in CONFIGS:
        for lname, cls in laws:
            results[(cname, lname)] = run(p, cls, params, n_cycles, n_frames)
    # channels given up: final error over 3 dB rms for every law (same channels for all)
    fin = np.array([np.sqrt(np.mean(r['errs'][-1]**2, axis=0)) for r in results.values()])
    lost = np.all(fin > 3.0, axis=0)
    names = [f"{n}{'xy'[d-1]}" for n, d in p['controls']]
    print('channels given up (final rms > 3 dB for every law):', [nm for nm, l in zip(names, lost) if l] or 'none')
    rms = lambda e, m: float(np.sqrt(np.mean(e[:, m]**2))) if m.any() else float('nan')
    print(f"\n{'':36s} {'err':>6s} {'given up':>9s}  {'all solved':>11s}  {'compute to':>11s}  {'ms/call':>14s}  {'lines/call':>10s}  drive  drive |coh|")
    print(f"{'':36s} {'(dB)':>6s} {'(dB)':>9s}  {'by cycle':>11s}  {'that point':>11s}  {'median / max':>14s}  {'median':>10s}  (dB)   median")
    for (cname, lname), r in results.items():
        full = np.flatnonzero(r['solved'] >= r['solved'].max())
        c_full = int(full[0])
        secs = r['t_init'] + r['ms'][:c_full].sum()/1e3
        print(f"{cname:30s} {lname:5s} {rms(r['errs'][-1], ~lost):6.2f} {rms(r['errs'][-1], lost):9.2f}  "
              f"{c_full:4d} ({r['solved'].max()}) {secs:9.1f} s  {np.median(r['ms']):6.0f} /{r['ms'].max():6.0f}  "
              f"{np.median(r['lines']):10.0f}  {r['drive']:+5.1f}  {r['coh']:.2f}", flush=True)
    print('\nerr (dB) on the reachable channels, by cycle (cycle 0 = the first control call):')
    print(f"{'cycle':>36s} " + ' '.join(f'{k:5d}' for k in range(n_cycles + 1)))
    for (cname, lname), r in results.items():
        print(f'{cname:30s} {lname:5s} ' + ' '.join(f'{rms(e, ~lost):5.2f}' for e in r['errs']))


if __name__ == '__main__':
    main()
