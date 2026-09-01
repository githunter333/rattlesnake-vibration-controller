"""
Verify the live-captured CVA sys-ID FRF (examples/sixdrive12resp/results/
cva_captures/latest_cva_sysid_capture.npz) against:
  (a) ground truth (frf_frame6x12_H.npz, 100-1000 Hz)
  (b) an offline recompute of global_cva_innovations on the IDENTICAL raw
      response/reference buffers that were captured live

(a) tells us whether the live FRF that would be handed to control is
accurate. (b) tells us whether spectral_processing.py's live wrapper
reproduces the standalone, already-validated global_cva_frf.py algorithm
on the exact same data -- i.e. whether there's an implementation bug in
the live wrapper, isolated from any question about the excitation itself
(COLA framing, closed-loop, bandwidth, etc.), since both (a)'s "frf" key
and (b)'s recompute start from the same raw samples.
"""
import numpy as np
import sys
sys.path.insert(0, 'globalcva')
from global_cva_frf import global_cva_innovations, frf_from_ss

CAP = 'examples/sixdrive12resp/results/cva_captures/latest_cva_sysid_capture.npz'
GT = 'examples/sixdrive12resp/results/frf_frame6x12_H.npz'


def interp_complex(f_src, H_src, f_dst):
    # H_src: (nf, nout, nin) complex, interpolate each (out,in) pair onto f_dst
    nout, nin = H_src.shape[1], H_src.shape[2]
    out = np.zeros((len(f_dst), nout, nin), dtype=complex)
    for o in range(nout):
        for i in range(nin):
            re = np.interp(f_dst, f_src, H_src[:, o, i].real)
            im = np.interp(f_dst, f_src, H_src[:, o, i].imag)
            out[:, o, i] = re + 1j * im
    return out


def mag_db_err(H_est, H_ref):
    return 20 * np.log10(np.abs(H_est) / np.maximum(np.abs(H_ref), 1e-30))


def phase_err_deg(H_est, H_ref):
    ratio = H_est * np.conj(H_ref)
    denom = np.abs(H_est) * np.abs(H_ref)
    denom = np.maximum(denom, 1e-30)
    return np.angle(ratio / denom, deg=True)


def rel_err(H_est, H_ref):
    return np.linalg.norm(H_est - H_ref) / np.linalg.norm(H_ref)


def summarize(name, H_est_on_gt, H_gt):
    nout = H_gt.shape[1]
    print(f'--- {name} ---')
    mag_means, phase_means = [], []
    for o in range(nout):
        me = mag_db_err(H_est_on_gt[:, o, :], H_gt[:, o, :])
        pe = phase_err_deg(H_est_on_gt[:, o, :], H_gt[:, o, :])
        mag_means.append(np.mean(np.abs(me)))
        phase_means.append(np.mean(np.abs(pe)))
        print(f'  resp ch {o}: mean |mag err| = {np.mean(np.abs(me)):6.2f} dB, '
              f'mean |phase err| = {np.mean(np.abs(pe)):6.2f} deg')
    overall = rel_err(H_est_on_gt.reshape(len(H_gt), -1), H_gt.reshape(len(H_gt), -1))
    print(f'  MEAN across channels: mag={np.mean(mag_means):.2f} dB, '
          f'phase={np.mean(phase_means):.2f} deg, overall relative error={overall:.4f}')
    print()
    return overall


if __name__ == '__main__':
    cap = np.load(CAP, allow_pickle=True)
    gt = np.load(GT, allow_pickle=True)

    f_gt = gt['f']
    H_gt = gt['H']  # (901, 8, 6)

    f_live = cap['frequencies']
    H_live = cap['frf']  # (2561, 8, 6), already the FRF as computed live

    print(f"Capture: sample_rate={int(cap['sample_rate'])}, lags={int(cap['cva_lags'])}, "
          f"rank={int(cap['cva_rank'])}, refine_iters={int(cap['cva_refine_iters'])}, "
          f"window={float(cap['cva_window_seconds'])}s, "
          f"response_buffer={cap['response_buffer'].shape}, "
          f"reference_buffer={cap['reference_buffer'].shape}")
    print()

    # (a) live-computed FRF vs ground truth, restricted to the 100-1000 Hz band
    #     where ground truth is defined
    H_live_on_gt = interp_complex(f_live, H_live, f_gt)
    err_a = summarize('(a) LIVE captured FRF vs ground truth (100-1000 Hz)', H_live_on_gt, H_gt)

    # (b) offline recompute of global_cva_innovations on the identical raw
    #     buffers, same params, compared to ground truth
    y = cap['response_buffer']
    u = cap['reference_buffer']
    dt = 1.0 / float(cap['sample_rate'])
    result = global_cva_innovations(y, u, lags=int(cap['cva_lags']), rank=int(cap['cva_rank']),
                                     refine_iters=int(cap['cva_refine_iters']), tol=1e-10)
    A, B, C, D = result['A'], result['B'], result['C'], result['D']
    H_offline = frf_from_ss(A, B, C, D, f_live, dt)
    H_offline_on_gt = interp_complex(f_live, H_offline, f_gt)
    err_b = summarize('(b) OFFLINE recompute (same raw buffers) vs ground truth', H_offline_on_gt, H_gt)

    # (c) live-computed FRF vs offline recompute directly -- isolates whether
    #     spectral_processing.py's live wrapper matches the standalone algorithm
    #     on identical data
    disagree = rel_err(H_live.reshape(len(f_live), -1), H_offline.reshape(len(f_live), -1))
    print(f'(c) LIVE vs OFFLINE recompute, direct disagreement (full 0-2560 Hz grid): '
          f'relative error = {disagree:.4f}')
    print()

    print('=== SUMMARY ===')
    print(f'(a) live vs truth      overall relative error = {err_a:.4f}')
    print(f'(b) offline vs truth   overall relative error = {err_b:.4f}')
    print(f'(c) live vs offline    direct disagreement     = {disagree:.4f}')
