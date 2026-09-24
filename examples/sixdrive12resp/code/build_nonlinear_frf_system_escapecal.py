#!/usr/bin/env python3
"""
build_nonlinear_frf_system_escapecal.py

A third nonlinear variant of the 6-drive/12-response frame, calibrated against
ESCAPE rather than against a fixed reference drive level.

WHY THIS EXISTS (2026-09-23).  Run 44 on
sdynpy_frame6x12_system_nonlinear_allmodes.npz at strength 100 overflowed the
time integration:

    sdynpy_nonlinear_system_virtual_hardware.py:151:
        RuntimeWarning: overflow encountered in power
        return k3s * q ** 3 + c2s * qd * np.abs(qd)

That is not a jump or a bifurcation, it is ESCAPE.  A softening cubic gives a
finite potential well: the restoring force wn^2*q + k3*q^3 turns negative at

    q_esc = wn / sqrt(|k3|)

and beyond that the solution runs away.  The Duffing "jump" criterion
(shift/zeta approaching 1), which is what the strength was chosen against,
never binds -- escape binds first and by a wide margin.

THE TRAP, stated plainly, because it cost a run: you cannot buy frequency
shift by turning up k3.

    shift    ~  (3/8) |k3| A^2 / wn^2        rises linearly with |k3|
    q_esc    =  wn / sqrt(|k3|)              falls as 1/sqrt(|k3|)

so the shift available AT escape is (3/8) = 37.5%, fixed, whatever |k3| is.
Raising the strength multiplier raises the shift and moves the cliff toward
you at exactly the same rate.  With a random drive whose peaks touch ~4 sigma
you need q_esc > 4*q_rms, which caps the usable shift at 0.375/16 = 2.3%, and
with any real margin nearer 0.5-1%.

Measured on the existing all-modes file, min over modes of q_esc/(4 q_rms):

    strength    1   4.06  safe        4   2.03  marginal
               10   1.28  marginal   25   0.81  ESCAPES
              100   0.41  ESCAPES

The binding mode was #31 at 837 Hz, where the randomised k3 draw landed at
2.7e18 -- and that mode is not the one carrying the shape change.  A single
uniform multiplier cannot serve both, which is the real defect of calibrating
per-mode targets independently of where each mode's cliff is.

WHAT THIS BUILDER DOES INSTEAD.  Each mode's k3 is set from its OWN escape
amplitude relative to the modal amplitude that the actual closed-loop drive
produces in it:

    |k3_i| = ( wn_i / (MARGIN * q_rms_i) )^2

q_rms_i is computed from a REAL run's commanded drive CPSD (run 40, match_trace
at 0 dB on the linear plant), projected onto mode i through the physical model
-- not assumed from a nominal voltage.  The consequence is that

    shift_i  =  (3/8) / MARGIN^2      for EVERY mode, identically,

so the article softens uniformly across the band and every mode carries the
same escape margin.  At MARGIN = 6 that is a 1.04% first-order shift with
q_esc = 6*q_rms, i.e. 1.5x headroom on 4-sigma peaks.

The first-order formula overestimates: checked against the single-mode file's
own numerical calibration it reads 8.9% where the builder solved -4.0%, a
factor of 2.2.  So expect a realised shift nearer 0.5%.  That is still 35x
the 0.013% the original all-modes file gave at a safe strength.

DESIGN POINT IS STRENGTH 1.  Unlike the earlier files there is no headroom to
turn it up -- strength s scales |k3| by s and q_esc by 1/sqrt(s), so s = 2.25
already eats the entire 1.5x margin.  RATTLESNAKE_NONLINEARITY_STRENGTH must
be 1 (or below) for this article.

Run:
    conda activate sdynpy    # or any env with numpy/scipy/netCDF4
    cd ~/Code/python/rattlesnake-vibration-controller/examples/sixdrive12resp/code
    python build_nonlinear_frf_system_escapecal.py

Output: ../results/case/sdynpy_frame6x12_system_nonlinear_escapecal.npz
"""
import os
import numpy as np
import scipy.linalg as sla
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.normpath(os.path.join(HERE, '..', 'results', 'case'))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
LINEAR = os.path.join(CASE, 'sdynpy_frame6x12_system.npz')
ALLMODES = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_allmodes.npz')
REF_RUN = os.path.join(RUNS, 'run40_matchtrace_refresh005_spec.nc4')
OUT = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_escapecal.npz')

MARGIN = 10.0       # q_esc / q_scale, against the WORST-CASE modal amplitude
ZETA_RATIO = 1.10   # damping ratio at q_scale, relative to the linear value
BAND = (100.0, 1000.0)

lin = dict(np.load(LINEAR, allow_pickle=True))
src = np.load(ALLMODES, allow_pickle=True)
M, C, K = lin['mass'], lin['damping'], lin['stiffness']
didx = np.asarray(src['nl_drive_dof_indices'])
ndof = M.shape[0]

# --- modes -----------------------------------------------------------------
ev, V = sla.eigh(K, M)                      # V is M-orthonormal
wn = np.sqrt(np.maximum(ev, 0.0))
o = np.argsort(wn); wn, V = wn[o], V[:, o]
flex = np.where(wn/(2*np.pi) > 1.0)[0]
PHI = V[:, flex]                            # (ndof, n_modes)
wn_f = wn[flex]
fn_f = wn_f/(2*np.pi)
zeta0 = np.einsum('di,de,ei->i', PHI, C, PHI)/(2*wn_f)
print(f'{len(flex)} flexible modes, {fn_f.min():.1f} - {fn_f.max():.1f} Hz')

# --- modal amplitude produced by a REAL commanded drive --------------------
ds = netCDF4.Dataset(REF_RUN); g = ds['Frame 6x12 Random']
D = np.array(g['drive_cpsd_real'][:]) + 1j*np.array(g['drive_cpsd_imag'][:])
f = np.array(g['specification_frequency_lines'][:], float)
S = np.array(g['specification_cpsd_matrix_real'][:]); ds.close()
df = f[1] - f[0]
sd = np.real(np.einsum('fmm->fm', S))
inb = (sd.max(axis=1) > 0) & (f >= BAND[0]) & (f <= BAND[1])
w = 2*np.pi*f
W = PHI.T @ M                                # q = W x
Hq = np.zeros((len(f), len(flex), len(didx)), complex)
for i in np.where(inb)[0]:
    Hq[i] = W @ np.linalg.solve(-(w[i]**2)*M + 1j*w[i]*C + K,
                                np.eye(ndof)[:, didx])
Sq = np.real(np.einsum('fmk,fkl,fml->fm', Hq.conj(), D, Hq))
q_rms = np.sqrt(np.sum(np.maximum(Sq[inb], 0.0), axis=0)*df)
print(f'modal amplitude from run 40 drive: q_rms {q_rms.min():.3e} - {q_rms.max():.3e}')

# --- calibrate against escape ----------------------------------------------
# --- the amplitude every mode is calibrated against -------------------------
# FIRST ATTEMPT, 2026-09-23, AND WHY IT FAILED.  Each mode was given its own
# reference, |k3_i| = (wn_i/(MARGIN*q_rms_i))^2 with q_rms_i from the control
# drive.  Run 46's system ID overflowed at strength 1.
#
#   q_rms under the control drive spans 4.27e-08 to 2.27e-06, a factor of 53,
#   so k3 spanned 3.07e15 to 3.19e20 -- a factor of 104,000.
#
# Two things go wrong with that.  A weakly driven mode gets an enormous k3 and
# a correspondingly tiny escape amplitude, and it is exactly those modes whose
# amplitude is least predictable: a flat system-ID drive and the shaped control
# drive excite the modes completely differently, with per-mode ratios from 0.25
# to 1.2.  Under a flat 1 V rms drive the tightest mode retained only 1.267x
# margin on 4-sigma peaks -- and the integration starts from rest, so the
# switch-on transient overshoots steady-state rms by more than that.
#
# A mode that is barely excited also contributes almost nothing to the FRF
# shape change, so there is no benefit in making it violently nonlinear.
#
# NOW: one amplitude scale for all modes, taken as the worst case over both
# drive conditions, and MARGIN raised to 10 so 4-sigma peaks sit at 2.5x
# headroom.  k3 then varies only as wn^2 -- a factor of 56 across the band
# instead of 104,000 -- and the shift concentrates on the well-excited modes,
# which is both safe and physically sensible.
f_flat = np.arange(0, 1025)*2.0
w_flat = 2*np.pi*f_flat
inb_flat = (f_flat >= BAND[0]) & (f_flat <= BAND[1])
df_flat = f_flat[1] - f_flat[0]
Sflat = np.zeros((len(f_flat), len(didx), len(didx)), complex)
Sflat[inb_flat] = np.eye(len(didx))*(1.0**2/(len(didx)*inb_flat.sum()*df_flat))
Hf = np.zeros((len(f_flat), len(flex), len(didx)), complex)
for i in np.where(inb_flat)[0]:
    Hf[i] = W @ np.linalg.solve(-(w_flat[i]**2)*M + 1j*w_flat[i]*C + K,
                                np.eye(ndof)[:, didx])
q_flat = np.sqrt(np.sum(np.maximum(np.real(np.einsum(
    'fmk,fkl,fml->fm', Hf.conj(), Sflat, Hf))[inb_flat], 0.0), axis=0)*df_flat)
# PER MODE, but referenced to the WORST CASE over both drive conditions.
# A single global scale (tried second) is safe but useless: it puts the whole
# shift on the one strongest mode, the median LINE barely moves, and d_shape
# came out at 0.0025 -- below the 0.0070 noise floor.  The defect in the FIRST
# attempt was not per-mode referencing, it was referencing the control drive
# alone, which does not bound what a flat system-ID drive does to the same
# mode (per-mode ratios 0.25 to 1.2).  Taking the max of the two, with MARGIN
# raised to 10, makes q_esc >= 10*q for EVERY mode under EITHER drive, so the
# 4-sigma margin is 2.5 everywhere by construction -- and every mode still
# gets its shift.
q_worst = np.maximum(q_rms, q_flat)
print(f'q control max {q_rms.max():.3e};  flat 1 V sysid max {q_flat.max():.3e}')
print(f'worst-case per-mode reference: {q_worst.min():.3e} - {q_worst.max():.3e}')
q_esc = MARGIN*q_worst
k3 = -(wn_f/q_esc)**2
# describing-function equivalent damping for c2*qd*|qd| at amplitude q_rms:
#   zeta_extra ~= (4/(3 pi)) * c2 * q_rms
c2 = (ZETA_RATIO - 1.0)*zeta0*3*np.pi/(4*q_worst)
shift_pred = (3.0/8.0)*np.abs(k3)*q_rms**2/wn_f**2
print(f'first-order shift, every mode: {100*shift_pred.min():.4f} - '
      f'{100*shift_pred.max():.4f} %   (3/8/MARGIN^2 = {100*0.375/MARGIN**2:.4f} %)')
worst = q_worst
print(f'q_esc/(4*worst-case q) margin: {(q_esc/(4*worst)).min():.3f} min, {(q_esc/(4*worst)).max():.1f} max')
print(f'k3 spread: factor {np.abs(k3).max()/np.abs(k3).min():.0f}')

out = dict(lin)
out['nl_target_mode_shapes'] = PHI
out['nl_k3s'] = k3
out['nl_c2s'] = c2
out['nl_q0_refs'] = q_worst        # per mode, worst case over both drives
out['nl_target_freqs_hz'] = fn_f
out['nl_target_zetas'] = zeta0
out['nl_drive_dof_indices'] = didx
out['nl_reference_drive_level_vrms'] = np.array(
    float(np.sqrt(np.sum(np.real(np.einsum('fnn->f', D))[inb])*df)))
np.savez(OUT, **out)
print(f'\nwrote {OUT}')
print(f'  reference drive level recorded: '
      f'{float(out["nl_reference_drive_level_vrms"]):.3f} V rms (run 40, measured)')
print('  DESIGN POINT IS STRENGTH 1 -- there is no headroom above it.')
