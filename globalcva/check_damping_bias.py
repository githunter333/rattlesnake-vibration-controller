"""
Cross-check: TRUE per-mode damping ratios (from the raw M,K,C matrices) vs
the damping ratios global_cva_v2 identifies, for the modes in 100-1000 Hz.
Written to test the "CVA-identified poles read over-damped -> peaks come in
low vs ground truth" hypothesis noted while reviewing
global_cva_frf_validation.png / global_cva_frf_pairs.png.

Rebuilds the same physical state-space that
SDynPySystemAcquisition._build_state_space builds in
components/sdynpy_system_virtual_hardware.py, directly from the raw
mass/damping/stiffness/transformation/coordinate arrays -- this sidesteps
importing the components/ package (which pulls in the full Qt GUI stack).
Verified against the saved ground-truth FRF: rel_err = 0.0000.

Result (1s, lags=40, rank=66): median zeta_id/zeta_true ~= 1.08, 76% of
matched modes come back over-damped. The largest outliers are closely-spaced
mode pairs (<~4 Hz apart) that collapse onto a single identified pole, whose
damping then has to broaden to explain both -- a second, larger-effect
mechanism on top of the general shrinkage bias.
"""
import numpy as np
import scipy.signal as signal
import scipy.linalg as la
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_v2, frf_from_ss, modes_from_A

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
d = {k: v for k, v in np.load(os.path.join(RESULTS, "sdynpy_frame6x12_system.npz")).items()}
M, C, K = d['mass'], d['damping'], d['stiffness']
coord = d['coordinate']
transformation = d['transformation']

if d['enforce_symmetry']:
    M = (M + M.T) / 2
    C = (C + C.T) / 2
    K = (K + K.T) / 2

# --- replicate create_response_channels' channel_indices / phi construction ---
channel_indices_map = {(int(row['node']), abs(int(row['direction']))): i
                        for i, row in enumerate(coord)}

def phi_rows(nodes, direction_code=1):
    idxs = [channel_indices_map[(n, direction_code)] for n in nodes]
    signs = np.array([np.sign(direction_code) * np.sign(coord[i]['direction']) for i in idxs])
    rows = transformation[idxs, :] * signs[:, None]
    return rows

drive_nodes = list(range(1, 7))      # 6 drives, force, X+
resp_nodes = list(range(7, 15))      # 8 accel responses, X+

phi_excitation = phi_rows(drive_nodes)   # (6, ndofs)
phi_response = phi_rows(resp_nodes)      # (8, ndofs)

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

# --- sanity check: compare against the saved ground-truth FRF ---
gt = np.load(os.path.join(RESULTS, "frf_frame6x12_H.npz"))
f_gt, H_gt = gt['f'], gt['H']
idx_band = np.where((f_gt >= 100) & (f_gt <= 1000))[0]
f_band = f_gt[idx_band]
w = 2 * np.pi * f_band
H_mine = np.zeros((len(f_band), 8, 6), dtype=complex)
I72 = np.eye(A_state.shape[0])
for i, wi in enumerate(w):
    H_mine[i] = C_accel @ np.linalg.solve(1j * wi * I72 - A_state, B_state) + D_accel
rel_err_recon = np.linalg.norm(H_mine - H_gt[idx_band]) / np.linalg.norm(H_gt[idx_band])
print(f"sanity check -- my hand-built continuous SS vs saved ground truth H: rel_err = {rel_err_recon:.4f}")

# --- TRUE modal frequencies + damping ratios, straight from M,K,C ---
evals, Phi = la.eigh(K, M)
evals = np.clip(evals, 0, None)
wn_true = np.sqrt(evals)                      # rad/s
fn_true = wn_true / (2 * np.pi)
C_modal = Phi.T @ C @ Phi
zeta_true_diag = np.diag(C_modal) / (2 * wn_true)
off_diag_frac = np.linalg.norm(C_modal - np.diag(np.diag(C_modal))) / np.linalg.norm(np.diag(np.diag(C_modal)))
print(f"C_modal off-diagonal / diagonal norm ratio (proportional-damping check): {off_diag_frac:.4f}")

band_mask_true = (fn_true >= 100) & (fn_true <= 1000)
print(f"\nTrue modes in band: {band_mask_true.sum()}")

# --- generate broadband drive + response, matching validate_global_cva.py ---
fs = 5120.0
dt = 1.0 / fs
LAGS, RANK = 40, 66
duration = 1.0
n = int(duration * fs)
np.random.seed(0)
t = np.arange(n) / fs
u = np.random.randn(6, n) * 0.05
_, yf, _ = signal.lsim(sys_ss, u.T, t, np.zeros(sys_ss.A.shape[0]))
y = yf.T

r = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
fr_id, zeta_id, max_eig = modes_from_A(r['A'], dt, f_hi=1100)
print(f"\nIdentified modes (CVA, {duration}s): {len(fr_id)}, max|eig|={max_eig:.5f}")

# --- match identified modes to nearest true mode in band and compare zeta ---
fn_true_band = fn_true[band_mask_true]
zeta_true_band = zeta_true_diag[band_mask_true]
order = np.argsort(fn_true_band)
fn_true_band = fn_true_band[order]
zeta_true_band = zeta_true_band[order]

print(f"\n{'true f (Hz)':>12} {'true zeta':>10} | {'id f (Hz)':>10} {'id zeta':>9} | {'df (Hz)':>8} {'zeta_id/zeta_true':>18}")
print("-" * 80)
rows = []
for ft, zt in zip(fn_true_band, zeta_true_band):
    j = np.argmin(np.abs(fr_id - ft))
    df = fr_id[j] - ft
    ratio = zeta_id[j] / zt if zt > 0 else float('nan')
    rows.append(ratio)
    print(f"{ft:>12.2f} {zt:>10.5f} | {fr_id[j]:>10.2f} {zeta_id[j]:>9.5f} | {df:>8.2f} {ratio:>18.3f}")

rows = np.array(rows)
print(f"\nmedian zeta_id/zeta_true = {np.median(rows):.3f}   mean = {np.mean(rows):.3f}   "
      f"fraction >1 (over-damped) = {np.mean(rows > 1):.2f}")
