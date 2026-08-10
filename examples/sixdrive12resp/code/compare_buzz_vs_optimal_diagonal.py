#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_buzz_vs_optimal_diagonal.py

Compares the plain "buzz" closed-form baseline against the fully SDP-
converged result from control_laws/optimal_diagonal_control.py, on the
6-drive/12-response frame system, controlling M=8 of the 12 response
channels (nodes 7-14, i.e. all of Row B + first 2 of Row C) -- a tall
8x6 problem (M > N), which is the case where optimal-diagonal is
expected to pull ahead of buzz.

H(f) is precomputed separately by compute_frf_frame6x12.py (run in the
`sdynpy` conda env, which has sdynpy but not cvxpy) and loaded here
from ../results/frf_frame6x12_H.npz, since this script needs to run in
the `rattlesnake` env (which has cvxpy but not sdynpy) to actually
exercise optimal_diagonal_control.

Baseline definition
--------------------
"Buzz baseline" = optimal_diagonal_control's own `_buzz_solve_all`
(the closed-form H+ pinv synthesis with cross terms pulled from a
survey/buzz CPSD), captured BEFORE any SDP refinement touches it.

Note this deviates slightly from "output_cpsd right after init": the
class's own `_initialize()` always spends its first SDP batch
immediately (see optimal_diagonal_control.py's docstring, step 1-2),
so "right after init" is already partly SDP-refined for
max_bins_per_update bins. To get a clean, uncontaminated buzz-only
baseline for comparison, `_buzz_solve_all` is called directly here
instead of reading `output_cpsd` right after construction. The survey
CPSD used for both the buzz baseline's cross terms AND to seed
optimal_diagonal_control's own initialization is the same:
H @ H^H per bin (equivalent to an independent-unit-drive buzz test on
this noise-free synthetic system).

The "optimal diagonal" result is obtained the way Rattlesnake actually
drives this control law in practice: one system_id_update() call with
the survey CPSD (establishes cross terms + spends the first SDP
batch), then repeated control() calls until n_deferred == 0 (every
in-band bin is either SDP-refined or already within
error_threshold_db of target under buzz).

Score: per-DOF dB diagonal error at each frequency,
    10*log10( diag(H @ Sxx @ H^H) / target )
for both baseline and optimal-diagonal drive CPSDs Sxx.

Run (from the `rattlesnake` conda env, which has cvxpy):

    conda activate rattlesnake
    cd ~/Documents/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python compare_buzz_vs_optimal_diagonal.py

Outputs are written to ../results/.
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
# 0. Load the precomputed FRF (built by compute_frf_frame6x12.py in the
#    `sdynpy` env -- run that script first if this .npz doesn't exist)
# ---------------------------------------------------------------------
H_FILE = os.environ.get(
    "H_FRAME6X12_NPZ",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results", "frf_frame6x12_H.npz")),
)
if not os.path.exists(H_FILE):
    raise FileNotFoundError(
        f"{H_FILE} not found -- run compute_frf_frame6x12.py first (in the `sdynpy` conda env)."
    )
data = np.load(H_FILE)
f = data["f"]                      # (F,) Hz, 100-1000 at 1 Hz resolution
H = data["H"]                      # (F, M, N) complex, M=8, N=6
drive_nodes = data["drive_nodes"]
resp_nodes = data["resp_nodes"]

F, M, N = H.shape
print(f"Loaded H: F={F} bins ({f[0]:.0f}-{f[-1]:.0f} Hz), M={M} control responses, N={N} drives (tall: M>N)")
print(f"Drive nodes:   {drive_nodes.tolist()}")
print(f"Response nodes:{resp_nodes.tolist()}")

# ---------------------------------------------------------------------
# 1. Flat target spec: (F, M, M) diagonal, matches build_flat_spec_large.py's level
# ---------------------------------------------------------------------
psd_level = 0.001  # g^2/Hz
target_diag = np.full((F, M), psd_level)
specification = np.zeros((F, M, M), dtype=complex)
idx = np.arange(M)
specification[:, idx, idx] = psd_level

# ---------------------------------------------------------------------
# 2. Survey/buzz CPSD: independent unit-level drives -> response CPSD
#    = H @ I @ H^H per bin. This is what a real buzz/system-ID pass
#    with uncorrelated references measures on this noise-free system.
# ---------------------------------------------------------------------
survey_response_cpsd = np.einsum('fmn,fkn->fmk', H, H.conj())  # (F,M,M), N-dim identity contracted

# ---------------------------------------------------------------------
# 3. Buzz baseline: optimal_diagonal_control's own closed-form solve,
#    captured before any SDP refinement.
# ---------------------------------------------------------------------
ctrl = optimal_diagonal_control(
    specification=specification,
    warning_levels=None,
    abort_levels=None,
    extra_parameters="",
    transfer_function=None,   # do NOT auto-initialize (that would spend the first SDP batch)
)
buzz_cpsd = ctrl._buzz_solve_all(H, survey_response_cpsd)  # (F, N, N) drive CPSD

# ---------------------------------------------------------------------
# 4. Optimal-diagonal result: system_id_update() (real cross terms +
#    first SDP batch), then control() until n_deferred == 0.
# ---------------------------------------------------------------------
t0 = time.time()
ctrl.system_id_update(transfer_function=H, sysid_response_cpsd=survey_response_cpsd)
n_calls = 1
print(f"call {n_calls}: n_sdp_refinements={ctrl.n_sdp_refinements}, n_deferred={ctrl.n_deferred}")
while ctrl.n_deferred > 0:
    ctrl.control(transfer_function=H)
    n_calls += 1
    print(f"call {n_calls}: n_sdp_refinements={ctrl.n_sdp_refinements}, n_deferred={ctrl.n_deferred}")
opt_cpsd = ctrl.output_cpsd.copy()
elapsed = time.time() - t0
print(f"Converged after {n_calls} control()/system_id_update() calls, "
      f"{ctrl.n_sdp_refinements} total SDP refinements, {elapsed:.1f}s")

# ---------------------------------------------------------------------
# 5. Score: per-DOF dB diagonal error, 10*log10(diag(H Sxx H^H)/target)
# ---------------------------------------------------------------------
def diag_error_db(Sxx):
    Y = np.einsum('fmn,fnk,flk->fml', H, Sxx, H.conj())
    achieved = np.maximum(np.real(np.einsum('fmm->fm', Y)), 1e-30)
    return 10 * np.log10(achieved / target_diag)

err_buzz = diag_error_db(buzz_cpsd)   # (F, M)
err_opt = diag_error_db(opt_cpsd)     # (F, M)

print("\nPer-DOF summary (dB error, response node):")
print(f"{'node':>6} {'buzz mean':>10} {'buzz max|.|':>12} {'opt mean':>10} {'opt max|.|':>12}")
for m in range(M):
    print(f"{resp_nodes[m]:>6} {np.mean(err_buzz[:, m]):>10.3f} {np.max(np.abs(err_buzz[:, m])):>12.3f} "
          f"{np.mean(err_opt[:, m]):>10.3f} {np.max(np.abs(err_opt[:, m])):>12.3f}")

overall_buzz_rms = np.sqrt(np.mean(err_buzz ** 2))
overall_opt_rms = np.sqrt(np.mean(err_opt ** 2))
overall_buzz_max = np.max(np.abs(err_buzz))
overall_opt_max = np.max(np.abs(err_opt))
print(f"\nOverall RMS dB error:  buzz={overall_buzz_rms:.3f}  optimal-diagonal={overall_opt_rms:.3f}")
print(f"Overall max |dB error|: buzz={overall_buzz_max:.3f}  optimal-diagonal={overall_opt_max:.3f}")

np.savez(
    os.path.join(RESULTS_DIR, "buzz_vs_optimal_diagonal_comparison.npz"),
    f=f, resp_nodes=resp_nodes, drive_nodes=drive_nodes,
    err_buzz=err_buzz, err_opt=err_opt,
    buzz_cpsd=buzz_cpsd, opt_cpsd=opt_cpsd,
    n_calls=n_calls, n_sdp_refinements=ctrl.n_sdp_refinements,
)

# ---------------------------------------------------------------------
# 6. Plots: per-DOF error vs frequency (grid) + mean/max summary table
# ---------------------------------------------------------------------
COLOR_BUZZ, COLOR_OPT = 'C0', 'C3'
ncols = 4
nrows = int(np.ceil(M / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
for m in range(M):
    ax = axes[m]
    ax.plot(f, err_buzz[:, m], color=COLOR_BUZZ, lw=0.8, alpha=0.9, label='buzz')
    ax.plot(f, err_opt[:, m], color=COLOR_OPT, lw=0.8, alpha=0.9, label='optimal-diagonal')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_title(f"node {resp_nodes[m]}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if m == 0:
        ax.legend(fontsize=8)
for m in range(M, len(axes)):
    axes[m].axis('off')
for ax in axes[-ncols:]:
    ax.set_xlabel("Frequency (Hz)")
for r in range(nrows):
    axes[r * ncols].set_ylabel("dB error")
fig.suptitle(f"Per-DOF diagonal error: buzz vs optimal-diagonal (M={M}, N={N}, tall)")
fig.tight_layout()
plot1 = os.path.join(RESULTS_DIR, "buzz_vs_optimal_diagonal_per_dof_error.png")
fig.savefig(plot1, dpi=150)

# Summary table figure
fig2, ax2 = plt.subplots(figsize=(8, 0.5 + 0.35 * (M + 1)))
ax2.axis('off')
col_labels = ["Node", "Buzz mean (dB)", "Buzz max|.| (dB)", "Opt mean (dB)", "Opt max|.| (dB)"]
rows = []
for m in range(M):
    rows.append([
        f"{resp_nodes[m]}",
        f"{np.mean(err_buzz[:, m]):.3f}",
        f"{np.max(np.abs(err_buzz[:, m])):.3f}",
        f"{np.mean(err_opt[:, m]):.3f}",
        f"{np.max(np.abs(err_opt[:, m])):.3f}",
    ])
rows.append([
    "ALL", f"{np.mean(err_buzz):.3f}", f"{overall_buzz_max:.3f}",
    f"{np.mean(err_opt):.3f}", f"{overall_opt_max:.3f}",
])
table = ax2.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
for j in range(len(col_labels)):
    table[0, j].set_text_props(weight='bold')
for j in range(len(col_labels)):
    table[len(rows), j].set_text_props(weight='bold')
fig2.suptitle("Mean / max |dB error| summary: buzz vs optimal-diagonal")
plot2 = os.path.join(RESULTS_DIR, "buzz_vs_optimal_diagonal_summary_table.png")
fig2.savefig(plot2, dpi=150, bbox_inches='tight')

print(f"\nData written to:  {os.path.join(RESULTS_DIR, 'buzz_vs_optimal_diagonal_comparison.npz')}")
print(f"Plot written to:  {plot1}")
print(f"Table written to: {plot2}")
