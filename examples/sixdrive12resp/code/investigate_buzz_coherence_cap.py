#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigate_buzz_coherence_cap.py

Investigates whether capping pairwise drive-to-drive coherence at
max_drive_coherence=0.95 (the cap added to optimal_diagonal_control's SDP)
would matter for buzz_control_class-style closed-form control, which was
left uncapped on the reasoning that its survey/cross-term source already
comes from independent drives.

Two things this script does:
  1. MEASURES the natural (uncapped) drive-to-drive coherence in buzz's own
     output_cpsd -- not the survey's response coherence, but the coherence
     that actually falls out of Hpinv @ modified_spec @ Hpinv^H once it's
     been through the pseudo-inverse sandwich. This is the empirical check
     of whether buzz's drives stay "independent enough" on their own.
  2. Builds a CAPPED variant by post-processing buzz's per-bin drive CPSD:
     for every drive pair whose coherence exceeds the cap, shrink that
     cross term's magnitude down to the cap (preserving phase and every
     diagonal/auto-spectrum value), then re-projects onto the PSD cone
     (clips any negative eigenvalues introduced by the per-pair shrink) so
     the result is still a physically valid CPSD.
  3. Scores natural vs. capped buzz on diagonal tracking error and drive
     RMS, the same metrics used throughout this session's comparisons.

Run (rattlesnake env, needs cvxpy only because it imports
optimal_diagonal_control for its buzz/coherence helper methods -- no SDP
solving actually happens here):

    conda activate rattlesnake
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python investigate_buzz_coherence_cap.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.expanduser("~/Documents/Code/python/rattlesnake-vibration-controller")
sys.path.insert(0, REPO_ROOT)
from control_laws.optimal_diagonal_control import optimal_diagonal_control

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
MAX_DRIVE_COHERENCE = 0.95

# ---------------------------------------------------------------------
# 0. Load FRF (regenerate with `make frf` if this doesn't match the
#    current system damping) and build the same flat spec + survey CPSD
#    used throughout this session's comparisons.
# ---------------------------------------------------------------------
H_FILE = os.path.join(RESULTS_DIR, "frf_frame6x12_H.npz")
data = np.load(H_FILE)
f = data["f"]
H = data["H"]                      # (F, M, N)
drive_nodes = data["drive_nodes"]
resp_nodes = data["resp_nodes"]
F, M, N = H.shape
print(f"Loaded H: F={F} bins ({f[0]:.0f}-{f[-1]:.0f} Hz), M={M} responses, N={N} drives")

psd_level = 0.001
target_diag = np.full((F, M), psd_level)
specification = np.zeros((F, M, M), dtype=complex)
idx = np.arange(M)
specification[:, idx, idx] = psd_level

survey_response_cpsd = np.einsum('fmn,fkn->fmk', H, H.conj())  # independent-unit-drive survey

# ---------------------------------------------------------------------
# 1. Natural (uncapped) buzz solve
# ---------------------------------------------------------------------
ctrl = optimal_diagonal_control(
    specification=specification, warning_levels=None, abort_levels=None,
    extra_parameters="", transfer_function=None,
)
buzz_natural = ctrl._buzz_solve_all(H, survey_response_cpsd)  # (F, N, N)

# ---------------------------------------------------------------------
# 2. Measure buzz's own drive-to-drive coherence (not the survey's
#    response coherence -- the coherence actually present in the
#    resulting drive CPSD)
# ---------------------------------------------------------------------
in_band = np.any(target_diag > 0, axis=1)
drive_coh_natural = np.zeros((F, N, N))
for fi in np.where(in_band)[0]:
    drive_coh_natural[fi] = ctrl._cpsd_coherence(buzz_natural[fi])

pair_i, pair_j = np.triu_indices(N, k=1)
coh_offdiag = drive_coh_natural[in_band][:, pair_i, pair_j]  # (n_in_band, n_pairs)
print(f"\nNatural buzz drive-to-drive coherence ({len(pair_i)} pairs x {in_band.sum()} in-band bins):")
print(f"  mean: {coh_offdiag.mean():.4f}")
print(f"  95th percentile: {np.percentile(coh_offdiag, 95):.4f}")
print(f"  max: {coh_offdiag.max():.4f}  "
      f"(pair drives {drive_nodes[pair_i[np.unravel_index(np.argmax(coh_offdiag), coh_offdiag.shape)[1]]]}, "
      f"{drive_nodes[pair_j[np.unravel_index(np.argmax(coh_offdiag), coh_offdiag.shape)[1]]]})")
frac_over_cap = np.mean(coh_offdiag > MAX_DRIVE_COHERENCE)
print(f"  fraction of (pair, bin) combinations exceeding {MAX_DRIVE_COHERENCE}: {100*frac_over_cap:.2f}%")

# ---------------------------------------------------------------------
# 3. Capped variant: shrink over-cap cross terms, then re-project onto
#    the PSD cone (clip negative eigenvalues) so it's still a valid CPSD
# ---------------------------------------------------------------------
def cap_coherence(Xf, cap):
    X = Xf.copy()
    n = X.shape[0]
    diagX = np.real(np.diag(X))
    for a in range(n):
        for b in range(a + 1, n):
            denom = np.sqrt(max(diagX[a] * diagX[b], 1e-30))
            coh_ab = np.abs(X[a, b]) / denom if denom > 0 else 0.0
            if coh_ab > cap:
                limit_mag = cap * denom
                scale = limit_mag / max(np.abs(X[a, b]), 1e-30)
                X[a, b] *= scale
                X[b, a] = np.conj(X[a, b])
    # re-project onto PSD cone (per-pair shrink can occasionally push a
    # tiny negative eigenvalue given how tightly wound some bins are)
    w, v = np.linalg.eigh(X)
    if np.any(w < 0):
        w_clipped = np.clip(w, 0, None)
        X = (v * w_clipped) @ v.conj().T
    return X

buzz_capped = buzz_natural.copy()
n_bins_modified = 0
for fi in np.where(in_band)[0]:
    if np.any(drive_coh_natural[fi][pair_i, pair_j] > MAX_DRIVE_COHERENCE):
        buzz_capped[fi] = cap_coherence(buzz_natural[fi], MAX_DRIVE_COHERENCE)
        n_bins_modified += 1
print(f"\n{n_bins_modified}/{in_band.sum()} in-band bins had at least one drive pair "
      f"above {MAX_DRIVE_COHERENCE} coherence and were capped")

# ---------------------------------------------------------------------
# 4. Score: per-DOF dB diagonal error + drive RMS, natural vs. capped
# ---------------------------------------------------------------------
def diag_error_db(Sxx):
    Y = np.einsum('fmn,fnk,flk->fml', H, Sxx, H.conj())
    achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
    target = np.maximum(target_diag, 1e-30)
    return 10 * np.log10(achieved / target)

def drive_rms(Sxx):
    diag_psd = np.real(np.einsum('fnn->fn', Sxx))
    return np.sqrt(np.trapezoid(diag_psd[in_band], f[in_band], axis=0))

err_natural = diag_error_db(buzz_natural)[in_band]
err_capped = diag_error_db(buzz_capped)[in_band]
rms_natural = drive_rms(buzz_natural)
rms_capped = drive_rms(buzz_capped)

print("\nPer-DOF RMS-across-frequency dB error, natural vs. capped buzz:")
print(f"{'node':>6} {'natural':>10} {'capped':>10} {'delta':>8}")
for m in range(M):
    rn = np.sqrt(np.mean(err_natural[:, m] ** 2))
    rc = np.sqrt(np.mean(err_capped[:, m] ** 2))
    print(f"{resp_nodes[m]:>6} {rn:>10.3f} {rc:>10.3f} {rc - rn:>8.3f}")

overall_natural = np.sqrt(np.mean(err_natural ** 2))
overall_capped = np.sqrt(np.mean(err_capped ** 2))
print(f"\nOverall RMS dB error: natural={overall_natural:.3f}  capped={overall_capped:.3f}  "
      f"(delta {overall_capped - overall_natural:+.3f} dB)")

print(f"\nPer-drive RMS (100-1000 Hz band): ")
print(f"{'node':>6} {'natural':>10} {'capped':>10} {'ratio':>8}")
for n in range(N):
    print(f"{drive_nodes[n]:>6} {rms_natural[n]:>10.4f} {rms_capped[n]:>10.4f} "
          f"{rms_capped[n]/rms_natural[n]:>8.3f}")
total_natural = np.sqrt(np.sum(rms_natural ** 2))
total_capped = np.sqrt(np.sum(rms_capped ** 2))
print(f"\nOverall drive RMS (RSS): natural={total_natural:.4f}  capped={total_capped:.4f}  "
      f"ratio={total_capped/total_natural:.3f}")

np.savez(
    os.path.join(RESULTS_DIR, "buzz_coherence_cap_investigation.npz"),
    f=f, resp_nodes=resp_nodes, drive_nodes=drive_nodes,
    err_natural=err_natural, err_capped=err_capped,
    rms_natural=rms_natural, rms_capped=rms_capped,
    coh_offdiag=coh_offdiag, max_drive_coherence=MAX_DRIVE_COHERENCE,
)

# ---------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------
COLOR_NAT, COLOR_CAP = 'C0', 'C1'
ncols = 4
nrows = int(np.ceil(M / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
f_in = f[in_band]
for m in range(M):
    ax = axes[m]
    ax.plot(f_in, err_natural[:, m], color=COLOR_NAT, lw=0.8, alpha=0.9, label='buzz (uncapped)')
    ax.plot(f_in, err_capped[:, m], color=COLOR_CAP, lw=0.8, alpha=0.9, label=f'buzz (coh<={MAX_DRIVE_COHERENCE})')
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
fig.suptitle(f"Buzz diagonal error: natural vs. drive-coherence-capped ({MAX_DRIVE_COHERENCE})")
fig.tight_layout()
plot1 = os.path.join(RESULTS_DIR, "buzz_coherence_cap_per_dof_error.png")
fig.savefig(plot1, dpi=150)

fig2, ax2 = plt.subplots(figsize=(7, 5))
counts, bins_, _ = ax2.hist(coh_offdiag.ravel(), bins=60, color=COLOR_NAT, alpha=0.8)
ax2.axvline(MAX_DRIVE_COHERENCE, color='k', ls='--', lw=1.2, label=f'cap = {MAX_DRIVE_COHERENCE}')
ax2.set_xlabel("Drive-to-drive coherence (natural buzz)")
ax2.set_ylabel("Count (pair x in-band bin)")
ax2.set_title("Distribution of natural buzz drive-to-drive coherence")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
plot2 = os.path.join(RESULTS_DIR, "buzz_coherence_distribution.png")
fig2.savefig(plot2, dpi=150)

print(f"\nData written to:  {os.path.join(RESULTS_DIR, 'buzz_coherence_cap_investigation.npz')}")
print(f"Plot written to:  {plot1}")
print(f"Plot written to:  {plot2}")
