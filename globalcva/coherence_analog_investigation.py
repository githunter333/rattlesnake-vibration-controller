"""
coherence_analog_investigation.py

Investigates design-doc section 7 item 3 / section 10: CVA has no native
coherence, but optimal_diagonal_control and the GUI both used to be
thought to depend on it (corrected 2026-08-29 -- confirmed by reading the
actual control law code that nothing in the Python control math actually
consumes `multiple_coherence`; it's diagnostic-only). This script builds
and empirically grades THREE candidate coherence-analogs against a
known-truth reference, using the same synthetic 6-drive/8-response system
and noise-injection machinery as the rest of globalcva/.

Rattlesnake's real, live coherence (components/spectral_processing.py,
~line 440) is MULTIPLE coherence:
    coherence_i(f) = Re[ Gyu_i(f) @ pinv(Guu(f)) @ Gyu_i(f)^H ] / Gyy_ii(f)
algebraically identical to Re[(H1 @ Guu @ H1^H)_ii] / Gyy_ii since H1 =
Gyu @ pinv(Guu) by construction.

Candidate (a) SUBSTITUTION: plug an external H(f) into that exact
algebraic formula, using the same MEASURED Guu/Gyy. FIRST EMPIRICAL
FINDING (see below): this is NOT bounded in [0,1] for ANY H other than
the data's own H1 -- including the exact TRUE system H. The formula's
boundedness is a special property of H1 being the per-bin least-squares-
optimal solution (Cauchy-Schwarz forces the cross-term between prediction
and residual to vanish); a mismatched H leaves a nonzero cross-term that
can push the ratio above 1 or below 0. This isn't a CVA defect -- it's
structural to the algebraic-substitution approach, confirmed by testing
the true system H, not just CVA's.

Candidate (b) INNOVATIONS-BASED: decompose global_cva_innovations' own
model-predicted PSD into signal-path + noise-path terms using its fitted
Kalman gain/noise covariances (both PSD-additive by construction, so
provably bounded in [0,1]). Empirically: with lags=40/rank=66/
refine_iters=1 it collapses to near-zero almost immediately -- dominated
by unmodeled-dynamics/model-order truncation being absorbed into the
"noise" term, not by actual measurement noise. Measures model self-
consistency/order-adequacy, not signal-vs-noise fraction.

Candidate (c) EXPLAINED-VARIANCE / RESIDUAL-BASED (2026-08-29, user's
question: "how much of the total response is explained by H times the
input?"): rather than going through the Gxf/pinv(Gff)/Gxf^H algebraic
identity (which implicitly assumes H is the per-bin LS-optimal solution),
compute the model's prediction DIRECTLY and difference it against the
actual measured response:
    Yhat_block(f) = H(f) @ U_block(f)     for every Welch block, every H
    e_block(f)    = Y_block(f) - Yhat_block(f)
    coherence_c_i(f) = 1 - mean_blocks|e_block_i(f)|^2 / mean_blocks|Y_block_i(f)|^2
This is a literal, block-wise "fraction of response variance explained,"
using the SAME Welch segmentation for numerator and denominator so the
(unknown, irrelevant) overall PSD normalization constant cancels in the
ratio. Because mean|e|^2 >= 0 always, this is upper-bounded at 1 by
construction for ANY H, unlike candidate (a) -- no optimality assumption
required. It CAN go negative (model actively worse than predicting zero),
which is informative, not broken; clip to 0 in practice like a
nonnegative-R^2 convention.
"""
import os
import sys
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from global_cva_frf import global_cva_v2, global_cva_innovations, frf_from_ss
from build_true_system import build_system, generate_drive_response

FS = 5120.0
DT = 1.0 / FS
LAGS, RANK = 40, 66
REFINE_ITERS = 1
DURATION = 2.0
NOISE_FRACS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
FMIN, FMAX = 100.0, 1000.0

info = build_system()
sys_ss = info['sys_ss']


def add_noise(y_clean, frac, seed):
    rng = np.random.RandomState(seed)
    rms = np.sqrt(np.mean(y_clean ** 2, axis=1, keepdims=True))
    return y_clean + rng.randn(*y_clean.shape) * (frac * rms)


def continuous_frf(sys_ss, freqs_hz):
    A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
    n = A.shape[0]
    I = np.eye(n)
    H = np.zeros((len(freqs_hz), C.shape[0], B.shape[1]), dtype=complex)
    for i, f in enumerate(freqs_hz):
        s = 1j * 2 * np.pi * f
        H[i] = C @ np.linalg.solve(s * I - A, B) + D
    return H


def csd_matrices(u, y, fs, nperseg):
    f_w, Guu = sig.csd(u[None, :, :], u[:, None, :], fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, axis=-1)
    _, Gyu = sig.csd(u[None, :, :], y[:, None, :], fs=fs, nperseg=nperseg,
                      noverlap=nperseg // 2, axis=-1)
    _, Gyy = sig.csd(y[None, :, :], y[:, None, :], fs=fs, nperseg=nperseg,
                      noverlap=nperseg // 2, axis=-1)
    return f_w, np.moveaxis(Guu, -1, 0), np.moveaxis(Gyu, -1, 0), np.moveaxis(Gyy, -1, 0)


def multiple_coherence(H, Guu, Gyy):
    pred = np.einsum('fij,fjk,flk->fil', H, Guu, H.conj())
    num = np.real(np.einsum('fii->fi', pred))
    den = np.real(np.einsum('fii->fi', Gyy))
    return num / den


def cva_innovations_coherence(A, B, C, D, K, R, P, f_w, dt, Guu):
    Hu = frf_from_ss(A, B, C, D, f_w, dt)
    n = A.shape[0]
    nout = C.shape[0]
    He = np.zeros((len(f_w), nout, nout), dtype=complex)
    for i, f in enumerate(f_w):
        z = np.exp(1j * 2 * np.pi * f * dt)
        He[i] = C @ np.linalg.solve(z * np.eye(n) - A, K) + np.eye(nout)
    Re_cov = C @ P @ C.T + R
    sig_pred = np.einsum('fij,fjk,flk->fil', Hu, Guu, Hu.conj())
    noise_pred = np.einsum('fij,jk,flk->fil', He, Re_cov, He.conj())
    num = np.real(np.einsum('fii->fi', sig_pred))
    den = np.real(np.einsum('fii->fi', sig_pred + noise_pred))
    return num / den


def welch_blocks(x, nperseg, noverlap, window='hann'):
    """x: (nch, N) -> freqs (nf,), blocks (n_blocks, nch, nf) rfft of each
    detrended (constant), windowed block. No PSD normalization applied --
    fine for a ratio of two quantities computed through this SAME pipeline."""
    win = sig.get_window(window, nperseg)
    step = nperseg - noverlap
    n = x.shape[1]
    starts = list(range(0, n - nperseg + 1, step))
    blocks = np.zeros((len(starts), x.shape[0], nperseg // 2 + 1), dtype=complex)
    for bi, st in enumerate(starts):
        seg = x[:, st:st + nperseg]
        seg = seg - seg.mean(axis=1, keepdims=True)
        blocks[bi] = np.fft.rfft(seg * win[None, :], axis=1)
    freqs = np.fft.rfftfreq(nperseg, d=DT)
    return freqs, blocks


def explained_variance_coherence(H, Uf, Yf):
    """H: (nf,nout,nin). Uf: (nb,nin,nf). Yf: (nb,nout,nf).
    Returns coherence_c (nf,nout), UNCLIPPED (so callers can see raw min/max
    before deciding whether/how to clip)."""
    Yhat = np.einsum('fij,bjf->bif', H, Uf)
    E = Yf - Yhat
    Gee = np.mean(np.abs(E) ** 2, axis=0)   # (nout, nf)
    Gyy = np.mean(np.abs(Yf) ** 2, axis=0)  # (nout, nf)
    Gyy_safe = np.where(Gyy == 0, 1.0, Gyy)
    coh = 1.0 - (Gee / Gyy_safe)
    return coh.T  # (nf, nout)


u, y_clean = generate_drive_response(sys_ss, DURATION, fs=FS)
nperseg = min(1024, u.shape[1] // 2)

results = {}
print(f"{'frac':>5} | {'a: H1':>16} {'a: TRUE':>16} {'a: CVA':>16} | "
      f"{'b: CVA-innov':>16} | {'c: H1':>16} {'c: TRUE':>16} {'c: CVA':>16}")
print(f"{'':>5} | " + " ".join([f"{'min':>7} {'max':>8}"] * 6))
print("-" * 150)

for frac in NOISE_FRACS:
    y = y_clean if frac == 0.0 else add_noise(y_clean, frac, seed=hash(('coh', frac)) % (2**31))

    f_w, Guu, Gyu, Gyy = csd_matrices(u, y, FS, nperseg)
    band = (f_w >= FMIN) & (f_w <= FMAX)

    H1 = np.zeros_like(Gyu)
    for k in range(Guu.shape[0]):
        H1[k] = Gyu[k] @ np.linalg.pinv(Guu[k])
    coh_a_h1 = multiple_coherence(H1, Guu, Gyy)
    H_gt = continuous_frf(sys_ss, f_w)
    coh_a_true = multiple_coherence(H_gt, Guu, Gyy)

    rv2 = global_cva_v2(y, u, lags=LAGS, tol=1e-10, rank=RANK)
    H_cva = frf_from_ss(rv2['A'], rv2['B'], rv2['C'], rv2['D'], f_w, DT)
    coh_a_cva = multiple_coherence(H_cva, Guu, Gyy)

    rinn = global_cva_innovations(y, u, lags=LAGS, tol=1e-10, rank=RANK,
                                   refine_iters=REFINE_ITERS)
    coh_b = cva_innovations_coherence(rinn['A'], rinn['B'], rinn['C'], rinn['D'],
                                       rinn['K'], rinn['R'], rinn['P'], f_w, DT, Guu)

    f_wb, Uf = welch_blocks(u, nperseg, nperseg // 2)
    _, Yf = welch_blocks(y, nperseg, nperseg // 2)
    band_b = (f_wb >= FMIN) & (f_wb <= FMAX)

    coh_c_h1 = explained_variance_coherence(H1, Uf, Yf)
    H_gt_b = continuous_frf(sys_ss, f_wb)
    coh_c_true = explained_variance_coherence(H_gt_b, Uf, Yf)
    H_cva_b = frf_from_ss(rv2['A'], rv2['B'], rv2['C'], rv2['D'], f_wb, DT)
    coh_c_cva = explained_variance_coherence(H_cva_b, Uf, Yf)

    results[frac] = dict(f_w=f_w, band=band, coh_a_h1=coh_a_h1, coh_a_true=coh_a_true,
                          coh_a_cva=coh_a_cva, coh_b=coh_b,
                          f_wb=f_wb, band_b=band_b, coh_c_h1=coh_c_h1,
                          coh_c_true=coh_c_true, coh_c_cva=coh_c_cva)

    def mm(arr, m):
        return arr[m].min(), arr[m].max()

    vals = [mm(coh_a_h1, band), mm(coh_a_true, band), mm(coh_a_cva, band),
            mm(coh_b, band), mm(coh_c_h1, band_b), mm(coh_c_true, band_b),
            mm(coh_c_cva, band_b)]
    print(f"{frac:>5.2f} | " + " ".join(f"{v[0]:>7.3f} {v[1]:>8.3f}" for v in vals))

fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey=True)
for ax, frac in zip(axes.ravel(), NOISE_FRACS):
    r = results[frac]
    ch = 0
    ax.plot(r['f_w'][r['band']], r['coh_a_true'][r['band'], ch], color='black', lw=1.2,
            alpha=0.5, label='(a) TRUE, algebraic')
    ax.plot(r['f_w'][r['band']], r['coh_a_cva'][r['band'], ch], color='tab:red', lw=1,
            alpha=0.6, label='(a) CVA, algebraic')
    ax.plot(r['f_w'][r['band']], r['coh_b'][r['band'], ch], color='tab:green', lw=1,
            alpha=0.8, label='(b) CVA, innovations')
    ax.plot(r['f_wb'][r['band_b']], r['coh_c_true'][r['band_b'], ch], color='dimgray', lw=1.8,
            label='(c) TRUE, explained-var')
    ax.plot(r['f_wb'][r['band_b']], r['coh_c_cva'][r['band_b'], ch], color='tab:orange', lw=1.8,
            label='(c) CVA, explained-var')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.axhline(0.0, color='gray', lw=0.5, ls='--')
    ax.set_title(f'noise frac={frac:.2f}')
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.3)
    if frac == NOISE_FRACS[0]:
        ax.legend(fontsize=6.5, loc='lower left')
for ax in axes[-1]:
    ax.set_xlabel('Frequency (Hz)')
for ax in axes[:, 0]:
    ax.set_ylabel('coherence (ch 1)')
fig.suptitle(f'Coherence-analog candidates (a/b/c) vs TRUE, duration={DURATION}s, '
             f'lags={LAGS} rank={RANK} refine_iters={REFINE_ITERS}')
fig.tight_layout(rect=[0, 0, 1, 0.95])
RESULTS = os.path.normpath(os.path.join(_HERE, "..", "examples", "sixdrive12resp", "results"))
out_path = os.path.join(RESULTS, "coherence_analog_investigation.png")
fig.savefig(out_path, dpi=130)
print(f"\nplot -> {out_path}")
