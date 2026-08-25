"""Shared helper: rebuild the 6-drive/12-response physical state-space
directly from the raw M,C,K/transformation/coordinate arrays (matches
SDynPySystemAcquisition._build_state_space -- verified rel_err=0.0000
against the saved ground truth), plus the true per-mode frequencies and
damping ratios from the same M,K,C. Used by check_damping_bias.py and
sweep_rank_lags.py so both work from one verified system build.
"""
import os
import numpy as np
import scipy.signal as signal
import scipy.linalg as la

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "examples", "sixdrive12resp", "results"))


def build_system():
    d = {k: v for k, v in np.load(os.path.join(RESULTS, "sdynpy_frame6x12_system.npz")).items()}
    M, C, K = d['mass'], d['damping'], d['stiffness']
    coord = d['coordinate']
    transformation = d['transformation']

    if d['enforce_symmetry']:
        M = (M + M.T) / 2
        C = (C + C.T) / 2
        K = (K + K.T) / 2

    channel_indices_map = {(int(row['node']), abs(int(row['direction']))): i
                            for i, row in enumerate(coord)}

    def phi_rows(nodes, direction_code=1):
        idxs = [channel_indices_map[(n, direction_code)] for n in nodes]
        signs = np.array([np.sign(direction_code) * np.sign(coord[i]['direction']) for i in idxs])
        return transformation[idxs, :] * signs[:, None]

    phi_excitation = phi_rows(range(1, 7))    # 6 drives, force, X+
    phi_response = phi_rows(range(7, 15))     # 8 accel responses, X+

    ndofs = M.shape[0]
    Minv_K = np.linalg.solve(M, K)
    Minv_C = np.linalg.solve(M, C)

    A_state = np.block([[np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                         [-Minv_K, -Minv_C]])
    B_state = np.block([[np.zeros((ndofs, phi_excitation.shape[0]))],
                         [np.linalg.solve(M, phi_excitation.T)]])
    C_accel = np.block([-phi_response @ Minv_K, -phi_response @ Minv_C])
    D_accel = phi_response @ np.linalg.solve(M, phi_excitation.T)

    sys_ss = signal.StateSpace(A_state, B_state, C_accel, D_accel)

    # true modal frequencies + damping (proportional damping here -> exact)
    evals, Phi = la.eigh(K, M)
    wn_true = np.sqrt(np.clip(evals, 0, None))
    fn_true = wn_true / (2 * np.pi)
    C_modal = Phi.T @ C @ Phi
    zeta_true = np.diag(C_modal) / (2 * wn_true)

    gt = np.load(os.path.join(RESULTS, "frf_frame6x12_H.npz"))
    f_gt, H_gt = gt['f'], gt['H']
    idx_band = np.where((f_gt >= 100) & (f_gt <= 1000))[0]
    f_band = f_gt[idx_band]
    H_gt_band = H_gt[idx_band]

    band_mask = (fn_true >= 100) & (fn_true <= 1000)
    order = np.argsort(fn_true[band_mask])
    fn_true_band = fn_true[band_mask][order]
    zeta_true_band = zeta_true[band_mask][order]

    return dict(sys_ss=sys_ss, fn_true_band=fn_true_band, zeta_true_band=zeta_true_band,
                f_band=f_band, H_gt_band=H_gt_band)


def generate_drive_response(sys_ss, duration, fs=5120.0, seed=0, amp=0.05):
    n = int(duration * fs)
    np.random.seed(seed)
    t = np.arange(n) / fs
    u = np.random.randn(6, n) * amp
    _, yf, _ = signal.lsim(sys_ss, u.T, t, np.zeros(sys_ss.A.shape[0]))
    return u, yf.T
