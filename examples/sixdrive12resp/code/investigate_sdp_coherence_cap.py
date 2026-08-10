#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigate_sdp_coherence_cap.py

Companion to investigate_buzz_coherence_cap.py: quantifies what
optimal_diagonal_control's own max_drive_coherence=0.95 cap costs (or
doesn't cost) in diagonal tracking accuracy, by running the SDP to full
convergence twice on the same system -- once with the cap at its default
0.95, once disabled (1.0) -- and comparing.

Run (rattlesnake env, needs cvxpy):

    conda activate rattlesnake
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python investigate_sdp_coherence_cap.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.expanduser("~/Documents/Code/python/rattlesnake-vibration-controller")
sys.path.insert(0, REPO_ROOT)
from control_laws.optimal_diagonal_control import optimal_diagonal_control

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# ---------------------------------------------------------------------
# 0. Load FRF + build the same flat spec + survey CPSD used throughout
# ---------------------------------------------------------------------
H_FILE = os.path.join(RESULTS_DIR, "frf_frame6x12_H.npz")
data = np.load(H_FILE)
f = data["f"]
H = data["H"]
drive_nodes = data["drive_nodes"]
resp_nodes = data["resp_nodes"]
F, M, N = H.shape
print(f"Loaded H: F={F} bins ({f[0]:.0f}-{f[-1]:.0f} Hz), M={M} responses, N={N} drives")

psd_level = 0.001
target_diag = np.full((F, M), psd_level)
specification = np.zeros((F, M, M), dtype=complex)
idx = np.arange(M)
specification[:, idx, idx] = psd_level

survey_response_cpsd = np.einsum('fmn,fkn->fmk', H, H.conj())
in_band = np.any(target_diag > 0, axis=1)
pair_i, pair_j = np.triu_indices(N, k=1)


def run_sdp(extra_parameters, label):
    ctrl = optimal_diagonal_control(
        specification=specification, warning_levels=None, abort_levels=None,
        extra_parameters=extra_parameters, transfer_function=None,
    )
    t0 = time.time()
    ctrl.system_id_update(transfer_function=H, sysid_response_cpsd=survey_response_cpsd)
    n_calls = 1
    while ctrl.n_deferred > 0:
        ctrl.control(transfer_function=H)
        n_calls += 1
    elapsed = time.time() - t0
    print(f"[{label}] converged after {n_calls} calls, {ctrl.n_sdp_refinements} SDP refinements, "
          f"{ctrl.n_solver_failures} solver failures, {elapsed:.1f}s, "
          f"max_drive_coherence={ctrl.max_drive_coherence:g}")
    return ctrl.output_cpsd.copy()


# ---------------------------------------------------------------------
# 1. Run both: default (capped, 0.95) vs. uncapped (1.0)
# ---------------------------------------------------------------------
sdp_capped = run_sdp("", "SDP capped (0.95, default)")
sdp_uncapped = run_sdp("1e-6,0.05,20,1.0,1.0", "SDP uncapped (1.0)")

# ---------------------------------------------------------------------
# 2. Confirm the cap actually held, and see how high uncapped naturally goes
# ---------------------------------------------------------------------
def offdiag_coherence(Sxx):
    coh = np.zeros((F, N, N))
    for fi in np.where(in_band)[0]:
        d = np.real(np.diagonal(Sxx[fi]))
        den = np.outer(d, d)
        den = np.where(den == 0.0, 1.0, den)
        coh[fi] = np.abs(Sxx[fi]) ** 2 / den
    return np.sqrt(np.clip(coh[in_band][:, pair_i, pair_j], 0, None))

coh_capped = offdiag_coherence(sdp_capped)
coh_uncapped = offdiag_coherence(sdp_uncapped)
print(f"\nDrive-to-drive coherence -- capped run:   mean={coh_capped.mean():.4f}, max={coh_capped.max():.4f}")
print(f"Drive-to-drive coherence -- uncapped run: mean={coh_uncapped.mean():.4f}, max={coh_uncapped.max():.4f}, "
      f"fraction > 0.95: {100*np.mean(coh_uncapped > 0.95):.2f}%")

# ---------------------------------------------------------------------
# 3. Score: per-DOF dB error + drive RMS, capped vs. uncapped
# ---------------------------------------------------------------------
def diag_error_db(Sxx):
    Y = np.einsum('fmn,fnk,flk->fml', H, Sxx, H.conj())
    achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
    target = np.maximum(target_diag, 1e-30)
    return 10 * np.log10(achieved / target)

def drive_rms(Sxx):
    diag_psd = np.real(np.einsum('fnn->fn', Sxx))
    return np.sqrt(np.trapezoid(diag_psd[in_band], f[in_band], axis=0))

err_capped = diag_error_db(sdp_capped)[in_band]
err_uncapped = diag_error_db(sdp_uncapped)[in_band]
rms_capped = drive_rms(sdp_capped)
rms_uncapped = drive_rms(sdp_uncapped)

print("\nPer-DOF RMS-across-frequency dB error, SDP capped(0.95) vs. uncapped(1.0):")
print(f"{'node':>6} {'capped':>10} {'uncapped':>10} {'delta':>8}")
for m in range(M):
    rc = np.sqrt(np.mean(err_capped[:, m] ** 2))
    ru = np.sqrt(np.mean(err_uncapped[:, m] ** 2))
    print(f"{resp_nodes[m]:>6} {rc:>10.3f} {ru:>10.3f} {rc - ru:>8.3f}")

overall_capped = np.sqrt(np.mean(err_capped ** 2))
overall_uncapped = np.sqrt(np.mean(err_uncapped ** 2))
print(f"\nOverall RMS dB error: capped={overall_capped:.3f}  uncapped={overall_uncapped:.3f}  "
      f"(cap costs {overall_capped - overall_uncapped:+.3f} dB)")

print(f"\nPer-drive RMS (100-1000 Hz band):")
print(f"{'node':>6} {'capped':>10} {'uncapped':>10} {'ratio':>8}")
for n in range(N):
    print(f"{drive_nodes[n]:>6} {rms_capped[n]:>10.4f} {rms_uncapped[n]:>10.4f} "
          f"{rms_capped[n]/rms_uncapped[n]:>8.3f}")
total_capped = np.sqrt(np.sum(rms_capped ** 2))
total_uncapped = np.sqrt(np.sum(rms_uncapped ** 2))
print(f"\nOverall drive RMS (RSS): capped={total_capped:.4f}  uncapped={total_uncapped:.4f}  "
      f"ratio={total_capped/total_uncapped:.3f}")

np.savez(
    os.path.join(RESULTS_DIR, "sdp_coherence_cap_investigation.npz"),
    f=f, resp_nodes=resp_nodes, drive_nodes=drive_nodes,
    err_capped=err_capped, err_uncapped=err_uncapped,
    rms_capped=rms_capped, rms_uncapped=rms_uncapped,
    coh_capped=coh_capped, coh_uncapped=coh_uncapped,
)

# ---------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------
COLOR_CAP, COLOR_UNCAP = 'C1', 'C2'
ncols = 4
nrows = int(np.ceil(M / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
f_in = f[in_band]
for m in range(M):
    ax = axes[m]
    ax.plot(f_in, err_uncapped[:, m], color=COLOR_UNCAP, lw=0.8, alpha=0.9, label='SDP uncapped')
    ax.plot(f_in, err_capped[:, m], color=COLOR_CAP, lw=0.8, alpha=0.9, label='SDP capped (0.95)')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_title(f"node {resp_nodes[m]}", fontsize=10)
    ax.grid(True, which='both', alpha=0.4)
    if m == 0:
        ax.legend(fontsize=8)
for m in range(M, len(axes)):
    axes[m].axis('off')
for ax in axes[-ncols:]:
    ax.set_xlabel("Frequency (Hz)")
for r in range(nrows):
    axes[r * ncols].set_ylabel("dB error")
fig.suptitle("SDP diagonal error: uncapped vs. drive-coherence-capped (0.95)")
fig.tight_layout()
plot1 = os.path.join(RESULTS_DIR, "sdp_coherence_cap_per_dof_error.png")
fig.savefig(plot1, dpi=150)

print(f"\nData written to: {os.path.join(RESULTS_DIR, 'sdp_coherence_cap_investigation.npz')}")
print(f"Plot written to: {plot1}")
