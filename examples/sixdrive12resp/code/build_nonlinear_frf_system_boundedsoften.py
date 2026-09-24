#!/usr/bin/env python3
"""
build_nonlinear_frf_system_boundedsoften.py

The fourth -- and structurally different -- nonlinear variant of the
6-drive/12-response frame.  Every earlier one used a softening CUBIC and every
one of the four runs taken on them (42, 44, 46, 48) ended in escape.

WHY THE CUBIC COULD NEVER WORK, in one line: the shift available at escape is
(3/8) = 37.5% no matter how large |k3| is, because raising |k3| raises the
shift and lowers q_esc = wn/sqrt|k3| at exactly the same rate.  With 4-sigma
random peaks that caps the usable shift near 0.5-1%, and escapecal's MARGIN=10
put it at 0.375%.  Run 48 escaped anyway.  You cannot buy effect with strength.

WHAT CHANGES HERE.  The kernel is bounded (bounded_softening.py):

    g(q) = alpha + (1-alpha) exp(-(q/qt)^2),   F(q) = wn^2 q g(q)

F stays positive for all q, so the potential is unbounded and there is no well
to escape from.  This is not a bigger margin, it is the removal of the failure
mode.  Consequences:

  * TARGET SHIFT IS A FREE PARAMETER.  5% here vs 0.375% for escapecal -- 13x.
  * STRENGTH HAS NO CEILING.  RATTLESNAKE_NONLINEARITY_STRENGTH scales qt by
    1/sqrt(strength); alpha is untouched, so the plant is bounded at any value.
    escapecal's "DESIGN POINT IS STRENGTH 1, there is no headroom" is void.
  * SIZING BIAS FLIPS.  escapecal used max(q_control, q_sysid) because
    under-estimating amplitude meant escape.  Here under-estimating only means
    more shift than designed, so we size on q_control, the amplitude at the
    condition we actually want to hit the target.

THE ONE NEW HAZARD, AND IT IS NOT AN AMPLITUDE HAZARD.  The shift is quadratic
in modal amplitude, and the flat system-ID drive excites the modes differently
from the shaped control drive -- measured per-mode ratios 0.25 to 1.2
(escapecal).  So at a 5% design target the worst mode identifies with up to
~7% of softening ALREADY IN ITS FRF, mode-dependently.  That would contaminate
the very baseline the article compares against, and is a candidate explanation
for run 48's system ID "looking significantly noisier than the linear system
sysid" -- part of that was not noise, it was each resonance sitting at a
different shifted frequency.  This builder prints the required system-ID
backoff in dB.  RUN SYSTEM ID AT THAT LEVEL, not at 0 dB.

MODE SELECTION.  In-band modes only (100-1000 Hz).  Out-of-band modes get
qt = inf, which makes the nonlinear term identically zero -- they are not
approximately linear, they are exactly linear.  They contribute a near-static
compliance term to the in-band FRF, so softening them would add integration
cost and unverifiable behaviour while changing nothing the control loop sees.
It also leaves the out-of-band modes as an untouched reference.

Run:
    cd ~/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python3 build_nonlinear_frf_system_boundedsoften.py

Output: ../results/case/sdynpy_frame6x12_system_nonlinear_boundedsoften.npz
"""
import os
import numpy as np
import scipy.linalg as sla
import netCDF4

from bounded_softening import (ALPHA_DEFAULT, BAND_DEFAULT, select_in_band,
                               qt_for_target_shift, freq_ratio_random,
                               k3_equivalent, sysid_backoff_db, self_test)

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.normpath(os.path.join(HERE, '..', 'results', 'case'))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
LINEAR = os.path.join(CASE, 'sdynpy_frame6x12_system.npz')
ALLMODES = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_allmodes.npz')
REF_RUN = os.path.join(RUNS, 'run40_matchtrace_refresh005_spec.nc4')
OUT = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_boundedsoften.npz')

# 1.0%, NOT the 5% first tried.  The softening loop -- soften, response grows,
# soften more -- has a gain that passes 1.0 at a 2.0% target on this plant with
# run 40's shaped drive, at which point the plant JUMPS past the design point
# to a far-softened stable state instead of settling (measured: designing 5%
# produced 29%).  1.0% holds the gain at 0.51, a factor of two of margin, and
# is still 2.7x the shift escapecal's cubic actually realised -- with no escape
# possible at any level or strength.  Raise it only after re-running
# verify_bounded_softening.py; the printed loop gain is a hard guard.
TARGET_SHIFT = 0.01      # frequency shift at 0 dB on every softened mode
ALPHA = ALPHA_DEFAULT    # 0.35; above the 0.3086 monotone-tangent-stiffness bound
ZETA_RATIO = 1.10        # damping ratio at the design amplitude, vs linear
ALLOW_SYSID_SHIFT = 0.005
BAND = BAND_DEFAULT
LEVELS_DB = (-18.0, -12.0, -6.0, 0.0)

self_test(verbose=True)

lin = dict(np.load(LINEAR, allow_pickle=True))
src = np.load(ALLMODES, allow_pickle=True)
M, C, K = lin['mass'], lin['damping'], lin['stiffness']
didx = np.asarray(src['nl_drive_dof_indices'])
ndof = M.shape[0]

# --- modes -----------------------------------------------------------------
ev, V = sla.eigh(K, M)
wn = np.sqrt(np.maximum(ev, 0.0))
o = np.argsort(wn); wn, V = wn[o], V[:, o]
flex = np.where(wn / (2 * np.pi) > 1.0)[0]
PHI = V[:, flex]
wn_f = wn[flex]
fn_f = wn_f / (2 * np.pi)
zeta0 = np.einsum('di,de,ei->i', PHI, C, PHI) / (2 * wn_f)
n_modes = len(flex)
print(f'{n_modes} flexible modes, {fn_f.min():.1f} - {fn_f.max():.1f} Hz')

W = PHI.T @ M


def modal_std(f, S, didx):
    """Modal response std (per mode) under a drive CPSD S on the linear plant."""
    df = f[1] - f[0]
    w = 2 * np.pi * f
    use = np.zeros(len(f), bool)
    use[(f >= BAND[0]) & (f <= BAND[1])] = True
    Hq = np.zeros((len(f), n_modes, len(didx)), complex)
    for i in np.where(use)[0]:
        Hq[i] = W @ np.linalg.solve(-(w[i] ** 2) * M + 1j * w[i] * C + K,
                                    np.eye(ndof)[:, didx])
    Sq = np.real(np.einsum('fmk,fkl,fml->fm', Hq.conj(), S, Hq))
    return np.sqrt(np.sum(np.maximum(Sq[use], 0.0), axis=0) * df)


# --- modal amplitude under the REAL 0 dB control drive (run 40) ------------
ds = netCDF4.Dataset(REF_RUN); g = ds['Frame 6x12 Random']
D = np.array(g['drive_cpsd_real'][:]) + 1j * np.array(g['drive_cpsd_imag'][:])
f = np.array(g['specification_frequency_lines'][:], float)
S_spec = np.array(g['specification_cpsd_matrix_real'][:]); ds.close()
df = f[1] - f[0]
inb = (np.real(np.einsum('fmm->fm', S_spec)).max(axis=1) > 0) & \
      (f >= BAND[0]) & (f <= BAND[1])
q_control = modal_std(f, D, didx)
print(f'q_control (run 40, 0 dB): {q_control.min():.3e} - {q_control.max():.3e}')

# --- modal amplitude under a flat 1 V rms system-ID drive ------------------
f_flat = np.arange(0, 1025) * 2.0
inb_flat = (f_flat >= BAND[0]) & (f_flat <= BAND[1])
df_flat = f_flat[1] - f_flat[0]
Sflat = np.zeros((len(f_flat), len(didx), len(didx)), complex)
Sflat[inb_flat] = np.eye(len(didx)) * (1.0 ** 2 / (len(didx) * inb_flat.sum() * df_flat))
q_sysid = modal_std(f_flat, Sflat, didx)
print(f'q_sysid   (flat 1 V):     {q_sysid.min():.3e} - {q_sysid.max():.3e}')

# --- SELF-CONSISTENT calibration -------------------------------------------
# FIRST ATTEMPT SIZED qt ON THE LINEAR RESPONSE AND OVERSHOT BY 6x.
# Designed 5% at 0 dB; direct time integration (verify_bounded_softening.py)
# measured 29%.  Two feedbacks were missing, both of which AMPLIFY:
#
#   1. Softening lowers the modal stiffness, which raises the modal response,
#      which softens further.  For a white force this alone is mild
#      (sigma ~ g^-0.75, fixed point near 5.8%).
#   2. The control drive is SHAPED -- match_trace notches it at each resonance
#      to flatten the response.  When the resonance moves it slides off its own
#      notch into a much hotter part of the drive spectrum, and THAT is the
#      large term.  A white-noise estimate cannot see it.
#
# So qt must be sized on the modal response AT THE SOFTENED RESONANCE, not at
# the linear one.  Since the target stiffness ratio g_t = (1-target)^2 is known
# up front, no iteration is needed for the design point: evaluate the response
# with the resonance already moved to g_t, then choose qt to make that state
# self-consistent.  Other levels are reported by damped fixed-point iteration.
PhiD = PHI[didx, :]                                        # (n_drive, n_modes)
SQ = np.maximum(np.real(np.einsum('km,fkl,lm->fm', PhiD.conj(), D, PhiD)), 0.0)
Sflat_Q = np.maximum(np.real(np.einsum('km,fkl,lm->fm',
                                       PhiD.conj(), Sflat, PhiD)), 0.0)
wn2 = wn_f ** 2


def modal_sigma(g, S, fgrid, scale=1.0, zr=1.0):
    """Modal response std with each mode's stiffness scaled by g."""
    w = 2 * np.pi * fgrid
    dfg = fgrid[1] - fgrid[0]
    g = np.broadcast_to(np.atleast_1d(np.asarray(g, float)), (n_modes,))
    H = 1.0 / (-(w[:, None] ** 2)
               + 2j * zr * zeta0[None, :] * wn_f[None, :] * w[:, None]
               + g[None, :] * wn2[None, :])
    return np.sqrt(np.sum(S * np.abs(H) ** 2, axis=0) * dfg) * scale


def solve_g(qt, S, fgrid, scale=1.0, n_it=400):
    """Damped fixed point for the self-consistent stiffness ratio."""
    g = np.ones(n_modes)
    for _ in range(n_it):
        sg = modal_sigma(g, S, fgrid, scale)
        gn = ALPHA + (1 - ALPHA) * (1 + 2 * (sg / qt) ** 2) ** -1.5
        gn = np.where(np.isfinite(gn), gn, 1.0)
        if np.max(np.abs(gn - g)) < 1e-13:
            g = gn
            break
        g = 0.7 * g + 0.3 * gn
    return g, modal_sigma(g, S, fgrid, scale)


# cross-check the closed-form response against the builder's own q_control
chk = modal_sigma(np.ones(n_modes), SQ, f)
print(f'modal_sigma(g=1) vs q_control: max rel dev '
      f'{np.max(np.abs(chk - q_control) / q_control):.3%}')

mask = select_in_band(fn_f, BAND)
g_t = (1.0 - TARGET_SHIFT) ** 2
sig_t = modal_sigma(np.full(n_modes, g_t), SQ, f)          # response AT the target
u = ((1.0 - ALPHA) / (g_t - ALPHA)) ** (2.0 / 3.0)
qt = np.full(n_modes, np.inf)
qt[mask] = sig_t[mask] * np.sqrt(2.0 / (u - 1.0))
print(f'response growth from softening: sigma(g_t)/sigma(1) = '
      f'{np.min(sig_t/chk):.2f} - {np.max(sig_t/chk):.2f}x   '
      f'(this is the factor the first attempt missed)')

c2 = np.zeros(n_modes)
c2[mask] = (ZETA_RATIO - 1.0) * zeta0[mask] * 3 * np.pi / (4 * sig_t[mask])

# --- is the fixed point stable? -------------------------------------------
# The loop is  sigma -> g(sigma) -> sigma(g).  If its gain reaches 1 the plant
# jumps instead of settling, which would be a genuine (bounded) bifurcation
# rather than a controlled softening.  Measure the gain numerically at the
# design point.
eps = 1e-3
g_hi = ALPHA + (1 - ALPHA) * (1 + 2 * ((sig_t * (1 + eps)) / qt) ** 2) ** -1.5
s_hi = modal_sigma(g_hi, SQ, f)
gain = np.abs((s_hi - sig_t) / (sig_t * eps))
gmax = float(np.nanmax(gain[mask]))
print(f'fixed-point loop gain at design point: {gmax:.3f} (must stay below 1.0)')
if gmax >= 1.0:
    raise SystemExit(
        f'\nABORT: loop gain {gmax:.3f} >= 1 -- the design point is UNSTABLE.\n'
        'The plant will jump past the target to a far-softened stable state\n'
        'rather than settle at it.  Lower TARGET_SHIFT (gain passes 1.0 near\n'
        '2.0% on this plant) or accept the jump deliberately.')
if gmax > 0.7:
    print(f'  WARNING: gain {gmax:.3f} leaves little margin; 1.0% target gives ~0.51.')

print(f'\nsoftening {int(mask.sum())}/{n_modes} modes (band {BAND[0]:g}-{BAND[1]:g} Hz); '
      f'alpha={ALPHA:.3f}, target {100*TARGET_SHIFT:.1f}% at 0 dB')
print('  mode      f_Hz        qt      k3_equiv  ' +
      '  '.join(f'{d:+.0f}dB' for d in LEVELS_DB))
shifts = {}
for dbl in LEVELS_DB:
    gg, _ = solve_g(qt, SQ, f, 10 ** (dbl / 20.0))
    shifts[dbl] = 100 * (1 - np.sqrt(gg))
for i in range(n_modes):
    if not mask[i]:
        continue
    print(f'  {i:4d}  {fn_f[i]:8.2f}  {qt[i]:.4e}  {k3_equivalent(wn_f[i], ALPHA, qt[i]):+.3e}  '
          + '  '.join(f'{shifts[dbl][i]:6.3f}%' for dbl in LEVELS_DB))
for dbl in LEVELS_DB:
    v = shifts[dbl][mask]
    print(f'  {dbl:+6.1f} dB : median {np.median(v):6.3f}%  range [{v.min():.3f}, {v.max():.3f}]')

# --- system-ID contamination, on the FLAT drive ---------------------------
g_sys, _ = solve_g(qt, Sflat_Q, f_flat, 1.0)
sh_sys = 100 * (1 - np.sqrt(g_sys))
worst_mode = int(np.argmax(np.where(mask, sh_sys, -np.inf)))
print(f'\nSYSTEM ID CHECK (flat 1 V rms drive, self-consistent)')
print(f'  softening baked into the identified FRF at full sysid level: '
      f'median {np.median(sh_sys[mask]):.2f}%, worst {sh_sys[worst_mode]:.2f}% '
      f'(mode {worst_mode} at {fn_f[worst_mode]:.1f} Hz)')
need_db = None
for cand in np.arange(0.0, -40.1, -0.5):
    gs, _ = solve_g(qt, Sflat_Q, f_flat, 10 ** (cand / 20.0))
    if np.max(100 * (1 - np.sqrt(gs))[mask]) <= 100 * ALLOW_SYSID_SHIFT:
        need_db = float(cand)
        break
print(f'  ==> RUN SYSTEM ID AT {need_db:+.1f} dB to hold every mode under '
      f'{100*ALLOW_SYSID_SHIFT:.1f}% softening')

# --- escape check ----------------------------------------------------------
qq = np.linspace(0.0, 1e4, 200001)
worst_force = np.inf
for i in np.where(mask)[0]:
    F = wn2[i] * qq * (ALPHA + (1 - ALPHA) * np.exp(-(qq / qt[i]) ** 2))
    worst_force = min(worst_force, F[1:].min())
print(f'\nESCAPE CHECK: min restoring force over all softened modes out to '
      f'q = 1e4 is {worst_force:.4e} -- positive, so no escape at any level '
      f'or any strength.')

# --- write -----------------------------------------------------------------
out = dict(lin)
out['nl_target_mode_shapes'] = PHI
out['nl_soft_qts'] = qt
out['nl_soft_wn2s'] = wn2
out['nl_soft_alpha'] = np.array(float(ALPHA))
out['nl_soft_mask'] = mask
out['nl_c2s'] = c2
out['nl_q0_refs'] = q_control
out['nl_q_design_refs'] = sig_t
out['nl_q_sysid_refs'] = q_sysid
out['nl_target_freqs_hz'] = fn_f
out['nl_target_zetas'] = zeta0
out['nl_drive_dof_indices'] = didx
out['nl_design_target_shift'] = np.array(float(TARGET_SHIFT))
out['nl_sysid_backoff_db'] = np.array(float(need_db if need_db is not None else -40.0))
out['nl_reference_drive_level_vrms'] = np.array(
    float(np.sqrt(np.sum(np.real(np.einsum('fnn->f', D))[inb]) * df)))
np.savez(OUT, **out)
print(f'\nwrote {OUT}')
print(f'  reference drive level: {float(out["nl_reference_drive_level_vrms"]):.3f} V rms (run 40)')
print('  strength has NO ceiling on this file -- qt scales as 1/sqrt(strength), alpha is fixed.')
