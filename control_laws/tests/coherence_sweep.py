"""
Closed-loop comparison of diagonal control laws as coherence degrades.

Synthetic 8-response x 6-drive modal plant, 1 Hz lines, 20-380 Hz band.
Each control cycle:
  1. the law returns a drive CPSD X
  2. the TRUE plant produces Gyy = H_true X H_true^H + Gnn
  3. the law is handed CPSDs ESTIMATED from n_frames random frames of that
     drive and response (complex Gaussian), the way a controller measures them
The score is the error of the TRUE expected response autospectra against the
specification -- what the test article actually experiences -- not the noisy
measurement the law sees.

Three ways to get low coherence, plus a clean baseline:
  clean      H_model = H_true, no noise
  noise      uncorrelated response noise on channels 2, 5, 7 sized so their
             coherence AT SPECIFICATION LEVEL is gamma2 (noise = (1-gamma2) *
             spec, so the spec is still reachable)
  nonlinear  the plant at test level has every mode softened by `shift`
             (frequency down, damping up 50%) relative to the system-ID FRF
             the laws were given, plus 5% distortion noise
  illcond    the system-ID FRF is wrong along the plant's weakest drive
             direction by `err` relative error (what nearly dependent system-ID
             drives do to an H1 estimate)

Laws: match_trace_pseudoinverse, match_diagonal_congruence,
match_diagonal_congruence_broyden, optimal_diagonal_control_fast, each with
default parameters.

This is a model, not a rig.  It does not reproduce Rattlesnake's running
averages (each cycle here is measured from fresh frames), the COLA output
process, or hardware dynamics.

Usage: python control_laws/tests/coherence_sweep.py [n_cycles] [n_frames]
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
from check_control_laws_on_dev import synthetic_plant  # noqa: E402
from rattlesnake.utilities import load_control_module   # noqa: E402

N_FREQ, N_RESP, N_DRIVE = 401, 8, 6
# optimal_diagonal_control parses its string with empty fields DROPPED, so
# every field up to the last one set must be given explicitly.  This is the
# default string with drive_rcond = 0 and a budget that covers the band.
OD_FULL = '1e-6,0.05,400,1.0,0.95,4,-9,0,2,0'
NOISY_CHANNELS = [2, 5, 7]


def modal_frf(shift=0.0, damp_scale=1.0, seed=0):
    """Same plant as synthetic_plant, with optional softening."""
    rng = np.random.default_rng(seed)
    f = np.arange(N_FREQ)*1.0
    w = 2*np.pi*f
    n_modes = 10
    wn = 2*np.pi*np.linspace(40, 340, n_modes)*(1.0 - shift)
    zeta = 0.02*damp_scale
    phi_r = rng.standard_normal((N_RESP, n_modes))
    phi_d = rng.standard_normal((N_DRIVE, n_modes))
    H = np.zeros((N_FREQ, N_RESP, N_DRIVE), complex)
    wn0 = 2*np.pi*np.linspace(40, 340, n_modes)
    for k in range(n_modes):
        den = wn[k]**2 - w**2 + 2j*zeta*wn[k]*w
        H += np.outer(phi_r[:, k], phi_d[:, k])[None]*(wn0[k]**2/den)[:, None, None]
    H += 0.01*(rng.standard_normal(H.shape) + 1j*rng.standard_normal(H.shape))
    return H


def scenario(name, level):
    p = synthetic_plant(n_freq=N_FREQ, n_resp=N_RESP, n_drive=N_DRIVE)
    spec_diag = np.real(np.einsum('fii->fi', p['spec']))
    H_true = p['H'].copy()
    H_model = p['H'].copy()
    noise = np.zeros((N_FREQ, N_RESP))
    if name == 'noise':
        g2 = level
        for c in NOISY_CHANNELS:
            noise[:, c] = (1.0 - g2)*spec_diag[:, c]
    elif name == 'nonlinear':
        H_true = modal_frf(shift=level, damp_scale=1.5)
        noise += 0.05*spec_diag
    elif name == 'illcond':
        rng = np.random.default_rng(7)
        _, s, Vh = np.linalg.svd(H_true)
        v_weak = Vh[:, -1, :].conj()                       # (F, N)
        u_err = (rng.standard_normal((N_FREQ, N_RESP))
                 + 1j*rng.standard_normal((N_FREQ, N_RESP)))
        u_err /= np.linalg.norm(u_err, axis=1, keepdims=True)
        # error of size err * ||H|| along the weakest direction
        mag = level*np.linalg.norm(H_true, axis=(1, 2))
        H_model = H_true + (mag[:, None, None]*u_err[:, :, None]*v_weak[:, None, :].conj())
    p['H_true'], p['H_model'], p['noise_diag'] = H_true, H_model, noise
    p['spec_diag'] = spec_diag
    return p


def measure(H, X, noise_diag, n_frames, rng):
    """Estimate (Gyy, Gxx, coherence) from n_frames Gaussian frames.
    n_frames = 0 returns the exact (expected) CPSDs -- no estimation noise."""
    F, M, N = H.shape
    Xh = (X + X.conj().transpose(0, 2, 1))/2
    if n_frames == 0:
        Gyy = H @ Xh @ H.conj().transpose(0, 2, 1)
        Gyy = Gyy + noise_diag[:, :, None]*np.eye(M)[None]
        sig = np.real(np.einsum('fmm->fm', Gyy)) - noise_diag
        return Gyy, Xh, sig/np.maximum(sig + noise_diag, 1e-300)
    lam, V = np.linalg.eigh(Xh)
    L = V*np.sqrt(np.clip(lam, 0, None))[:, None, :]
    z = (rng.standard_normal((F, N, n_frames)) + 1j*rng.standard_normal((F, N, n_frames)))/np.sqrt(2)
    v = L @ z
    e = (rng.standard_normal((F, M, n_frames)) + 1j*rng.standard_normal((F, M, n_frames)))/np.sqrt(2)
    y = H @ v + np.sqrt(noise_diag)[:, :, None]*e
    Gxx = v @ v.conj().transpose(0, 2, 1)/n_frames
    Gyy = y @ y.conj().transpose(0, 2, 1)/n_frames
    sig = np.real(np.einsum('fmn,fnk,fmk->fm', H, Xh, H.conj()))
    coh = sig/np.maximum(sig + noise_diag, 1e-300)
    return Gyy, Gxx, coh


def true_error(p, X):
    H = p['H_true']
    y = np.real(np.einsum('fmn,fnk,fmk->fm', H, X, H.conj())) + p['noise_diag']
    b = p['band']
    e = 10*np.log10(np.maximum(y[b], 1e-300)/p['spec_diag'][b])
    return np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e**2, axis=0))


def load_laws():
    with contextlib.redirect_stdout(io.StringIO()):
        cl = load_control_module(os.path.join(LAW_DIR, 'control_laws.py'))
        dc = load_control_module(os.path.join(LAW_DIR, 'diagonal_congruence_control.py'))
        br = load_control_module(os.path.join(LAW_DIR, 'broyden_congruence_control.py'))
        od = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control_fast.py'))
    # name: (kind, law, extra_parameters)
    return {
        'match_trace': ('fn', cl.match_trace_pseudoinverse, ''),
        'congruence': ('cls', dc.match_diagonal_congruence, ''),
        'cong_broyden': ('cls', br.match_diagonal_congruence_broyden, ''),
        'cong_b g=0': ('cls', br.match_diagonal_congruence_broyden, ',,,,,,,,0'),
        'cong_b p=0': ('cls', br.match_diagonal_congruence_broyden, ',,,,,,,,,0'),
        'cong_b measX': ('cls', br.match_diagonal_congruence_broyden, ',,,,,,,,,,,,,,0'),
        'optdiag rc=0': ('cls', od.optimal_diagonal_control_fast, OD_FULL),
        # hybrid: first command from the model-based optimal diagonal solve,
        # then diagonal-congruence feedback from that shape
        'hyb congr': ('hyb', dc.match_diagonal_congruence, ''),
        'hyb broyden': ('hyb', br.match_diagonal_congruence_broyden, ''),
        'hyb broy cap1': ('hyb', br.match_diagonal_congruence_broyden, ',1.0'),
        'optdiag_fast': ('cls', od.optimal_diagonal_control_fast, ''),
    }


def run(p, kind, law, params, n_cycles, n_frames, seed=1):
    rng = np.random.default_rng(seed)
    Hm, spec = p['H_model'], p['spec']
    coh_sysid = np.ones((N_FREQ, N_RESP))
    args_sysid = (p['gxx_noise'], p['gff_noise'], p['gxx_sysid'], p['gff_sysid'])
    hist = []
    nf = n_frames if n_frames > 0 else 1000     # what the law is told
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if kind == 'hyb':
            od = load_laws()['optdiag_fast'][1]
            seed_law = od(spec, p['warn'], p['abort'], OD_FULL, None, *args_sysid, None, 0,
                          n_frames, None, None)
            seed_law.system_id_update(Hm, *args_sysid, coh_sysid, nf, nf)
            obj = law(spec, p['warn'], p['abort'], params, None, *args_sysid, None, 0, n_frames,
                      None, None)
            obj.system_id_update(Hm, *args_sysid, coh_sysid, nf, nf)
            obj.control(Hm, coh_sysid, nf, nf, None, None)      # initialise its state
            X = seed_law.control(Hm, coh_sysid, nf, nf, None, None)
            if hasattr(obj, '_remember_command'):
                obj._reset_learning()
                obj._remember_command(X)
            kind = 'cls'
        elif kind == 'cls':
            obj = law(spec, p['warn'], p['abort'], params, None, *args_sysid, None, 0, n_frames,
                      None, None)
            obj.system_id_update(Hm, *args_sysid, coh_sysid, nf, nf)
            X = obj.control(Hm, coh_sysid, nf, nf, None, None)
        else:
            X = law(spec, p['warn'], p['abort'], Hm, *args_sysid, coh_sysid, nf, nf,
                    params, None, None)
        hist.append(true_error(p, X)[0])
        for _ in range(n_cycles):
            Gyy, Gxx, coh = measure(p['H_true'], X, p['noise_diag'], n_frames, rng)
            if kind == 'cls':
                X = obj.control(Hm, coh, nf, nf, Gyy, Gxx)
            else:
                X = law(spec, p['warn'], p['abort'], Hm, *args_sysid, coh, nf,
                        nf, params, Gyy, Gxx)
            hist.append(true_error(p, X)[0])
    final, per_ch = true_error(p, X)
    drive_db = 10*np.log10(np.mean(np.real(np.einsum('fii->f', X))[p['band']]))
    return np.array(hist), per_ch, drive_db


def main():
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    laws = load_laws()
    cases = [('clean', 0.0), ('noise', 0.8), ('noise', 0.5), ('noise', 0.3),
             ('nonlinear', 0.02), ('nonlinear', 0.05),
             ('illcond', 0.3), ('illcond', 1.0)]
    print(f'{n_cycles} control cycles, ' + (f'{n_frames} frames per CPSD estimate' if n_frames else 'EXACT CPSDs (no estimation noise)'))
    print('score = rms dB error of the TRUE response autospectra vs spec, band 20-380 Hz')
    print('mean(10-end) averages cycles 10..end; worst ch = worst channel rms at the end\n')
    hdr = f"{'case':16s} {'law':13s} {'cyc1':>6s} {'cyc5':>6s} {'mean(10-end)':>12s} {'final':>6s} {'worst ch':>9s} {'drive dB':>8s} {'time':>6s}"
    print(hdr)
    print('-'*len(hdr))
    rows = []
    for name, level in cases:
        if only and name not in only:
            continue
        p = scenario(name, level)
        label = name if name == 'clean' else f'{name} {level:g}'
        for lname, (kind, law, params) in laws.items():
            t0 = time.perf_counter()
            try:
                h, per_ch, drive_db = run(p, kind, law, params, n_cycles, n_frames)
                dt = time.perf_counter() - t0
                print(f'{label:16s} {lname:13s} {h[1]:6.2f} {h[min(5, len(h)-1)]:6.2f} '
                      f'{np.mean(h[10:]):12.2f} {h[-1]:6.2f} {np.max(per_ch):9.2f} '
                      f'{drive_db:8.1f} {dt:5.1f}s', flush=True)
                rows.append((label, lname, h))
            except Exception as ex:
                print(f'{label:16s} {lname:13s} FAILED: {type(ex).__name__}: {ex}', flush=True)
        print()
    np.save(os.path.join(HERE, 'coherence_sweep_histories.npy'),
            np.array([(r[0], r[1], r[2]) for r in rows], dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
