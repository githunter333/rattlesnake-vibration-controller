"""
Test candidate root cause from design doc section 16.6: does the live
sys-ID excitation generator's COLA block structure (RandomSignalGenerator
in components/signal_generation.py -- independent per-frame Gaussian
blocks, 50%-overlap Hann-windowed overlap-add) degrade CVA's fit relative
to the continuous i.i.d. white-noise realization used in the earlier
offline validation (cva_vs_h1_phase_comparison.py), even though both are
full-bandwidth?

Builds excitation two ways, drives the SAME ground-truth linear system
through each, and fits CVA (identical lags/rank/refine_iters/window as
live) on each. If COLA-blended excitation alone reproduces the live
capture's ~0.29 relative error (vs ~0.07-0.08 for continuous i.i.d.),
that confirms the block-generation structure as the root cause.
"""
import sys
import numpy as np
sys.path.insert(0, 'globalcva')
sys.path.insert(0, '.')
from build_true_system import build_system
from global_cva_frf import global_cva_innovations, frf_from_ss
import scipy.signal as signal
from components.signal_generation import RandomSignalGenerator  # NOTE: offline (cloud) runs used a components_offline stand-in without qtpy; on the actual dev machine components.signal_generation imports fine directly

FS = 5120.0
LAGS, RANK, REFINE = 40, 66, 1
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
AMP = 0.05  # matches build_true_system.generate_drive_response's default scale


def mag_db_err(H_est, H_ref):
    return 20 * np.log10(np.abs(H_est) / np.maximum(np.abs(H_ref), 1e-30))


def phase_err_deg(H_est, H_ref):
    ratio = H_est * np.conj(H_ref)
    denom = np.maximum(np.abs(H_est) * np.abs(H_ref), 1e-30)
    return np.angle(ratio / denom, deg=True)


def interp_complex(f_src, H_src, f_dst):
    nout, nin = H_src.shape[1], H_src.shape[2]
    out = np.zeros((len(f_dst), nout, nin), dtype=complex)
    for o in range(nout):
        for i in range(nin):
            out[:, o, i] = (np.interp(f_dst, f_src, H_src[:, o, i].real)
                             + 1j * np.interp(f_dst, f_src, H_src[:, o, i].imag))
    return out


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def make_iid_excitation(nin, n_total, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(nin, n_total) * AMP


def make_cola_excitation(nin, n_total, seed, num_samples_per_frame=5120,
                          cola_overlap=0.5, cola_window='hann', cola_exponent=0.5):
    np.random.seed(seed)
    gen = RandomSignalGenerator(
        rms=AMP, sample_rate=FS, num_samples_per_frame=num_samples_per_frame,
        num_signals=nin, low_frequency_cutoff=None, high_frequency_cutoff=None,
        cola_overlap=cola_overlap, cola_window=cola_window, cola_exponent=cola_exponent,
        output_oversample=1)
    chunks = []
    got = 0
    # generous burn-in so the COLA queue is past its zero-initialized startup
    # transient before we start keeping samples
    burn_in_calls = 10
    for _ in range(burn_in_calls):
        gen.generate_frame()
    while got < n_total:
        frame, _ = gen.generate_frame()
        chunks.append(frame)
        got += frame.shape[-1]
    u = np.concatenate(chunks, axis=-1)[:, :n_total]
    return u


def fit_and_score(u, y, f_band, H_gt_band):
    # take the most recent WINDOW_SAMPLES, matching the live sliding-window buffer
    y_win = y[:, -WINDOW_SAMPLES:]
    u_win = u[:, -WINDOW_SAMPLES:]
    result = global_cva_innovations(y_win, u_win, lags=LAGS, rank=RANK,
                                     refine_iters=REFINE, tol=1e-10)
    freqs = np.arange(WINDOW_SAMPLES // 2 + 1) * (FS / WINDOW_SAMPLES)
    # match live: frequencies = arange(num_frequency_lines)*frequency_spacing, but for
    # a like-for-like comparison against ground truth we just need freqs spanning
    # 100-1000 Hz; use the same nperseg=5120 grid as the live capture used
    freqs = np.arange(2561) * (FS / 5120.0) * (5120.0 / 5120.0)  # 0..2560 step 1 Hz (matches capture)
    dt = 1.0 / FS
    H = frf_from_ss(result['A'], result['B'], result['C'], result['D'], freqs, dt)
    H_on_gt = interp_complex(freqs, H, f_band)
    mag_errs = [np.mean(np.abs(mag_db_err(H_on_gt[:, o, :], H_gt_band[:, o, :])))
                for o in range(H_gt_band.shape[1])]
    phase_errs = [np.mean(np.abs(phase_err_deg(H_on_gt[:, o, :], H_gt_band[:, o, :])))
                  for o in range(H_gt_band.shape[1])]
    overall = rel_err(H_on_gt.reshape(len(f_band), -1), H_gt_band.reshape(len(f_band), -1))
    return np.mean(mag_errs), np.mean(phase_errs), overall


if __name__ == '__main__':
    sysd = build_system()
    sys_ss = sysd['sys_ss']
    f_band, H_gt_band = sysd['f_band'], sysd['H_gt_band']
    nin = sys_ss.B.shape[1]

    duration = 4.0  # seconds of simulated drive, so the kept 2.0s window is past startup
    n_total = int(duration * FS)
    t = np.arange(n_total) / FS

    print(f'{"seed":>4} | {"mode":>18} | {"mag err (dB)":>13} | {"phase err (deg)":>16} | {"rel err":>8}')
    print('-' * 75)
    for seed in (0, 1, 2):
        u_iid = make_iid_excitation(nin, n_total, seed)
        _, y_iid, _ = signal.lsim(sys_ss, u_iid.T, t, np.zeros(sys_ss.A.shape[0]))
        y_iid = y_iid.T
        mag, ph, rel = fit_and_score(u_iid, y_iid, f_band, H_gt_band)
        print(f'{seed:>4} | {"continuous i.i.d.":>18} | {mag:>13.2f} | {ph:>16.2f} | {rel:>8.4f}')

        u_cola = make_cola_excitation(nin, n_total, seed)
        _, y_cola, _ = signal.lsim(sys_ss, u_cola.T, t, np.zeros(sys_ss.A.shape[0]))
        y_cola = y_cola.T
        mag, ph, rel = fit_and_score(u_cola, y_cola, f_band, H_gt_band)
        print(f'{seed:>4} | {"COLA-blended":>18} | {mag:>13.2f} | {ph:>16.2f} | {rel:>8.4f}')
        print()

    print('For reference, the live captured fit: mag=2.02 dB, phase=15.36 deg, rel=0.2854')
