#!/usr/bin/env python3
"""
verify_bounded_softening.py

MEASURE, do not assert.  Four amplitude predictions were made on the cubic
nonlinear article and all four were wrong, so before any live run this script
integrates the plant's own modal equations in the time domain at the drive
levels the article will actually use, and reports:

  1. does it overflow (the run 42/44/46/48 failure) -- at strength 1 and far above
  2. what frequency shift is REALISED per mode, vs the 5% the builder designed

In modal coordinates the nonlinear term is diagonal -- nl_i depends only on
q_i -- and so is the linear part, so the 33 modes decouple completely and can
be integrated independently and cheaply.  This is the same RK4 with the same
N_SUB=4 the virtual hardware uses, driven by modal forces synthesised from
run 40's measured drive CPSD.
"""
import os, sys
import numpy as np
import scipy.linalg as sla
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.normpath(os.path.join(HERE, '..', 'results', 'case'))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
NPZ = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_boundedsoften.npz')
REF_RUN = os.path.join(RUNS, 'run40_matchtrace_refresh005_spec.nc4')

FS = 4096.0
T_BLOCK = 2.0
N_BLOCK = 4
N_SUB = 4
LEVELS_DB = [float(x) for x in (sys.argv[1:] or ['-18', '-12', '-6', '0'])]
STRENGTHS = [1.0, 25.0]

d = np.load(NPZ, allow_pickle=True)
PHI = d['nl_target_mode_shapes']
qt0 = d['nl_soft_qts']; wn2 = d['nl_soft_wn2s']; alpha = float(d['nl_soft_alpha'])
c2 = d['nl_c2s']; fn = d['nl_target_freqs_hz']; zeta = d['nl_target_zetas']
q_design = d['nl_q0_refs']
didx = np.asarray(d['nl_drive_dof_indices'])
wn = np.sqrt(wn2); nm = len(wn)

# --- per-mode modal force auto-PSD from run 40's measured drive ------------
ds = netCDF4.Dataset(REF_RUN); g = ds['Frame 6x12 Random']
D = np.array(g['drive_cpsd_real'][:]) + 1j * np.array(g['drive_cpsd_imag'][:])
fD = np.array(g['specification_frequency_lines'][:], float); ds.close()
PhiD = PHI[didx, :]                                   # (n_drive, n_modes)
SQ = np.real(np.einsum('km,fkl,lm->fm', PhiD.conj(), D, PhiD))   # (nf, n_modes)
SQ = np.maximum(SQ, 0.0)

nfft = int(FS * T_BLOCK)
fsyn = np.fft.rfftfreq(nfft, 1 / FS)
dfs = fsyn[1] - fsyn[0]
SQi = np.zeros((len(fsyn), nm))
for m in range(nm):
    SQi[:, m] = np.interp(fsyn, fD, SQ[:, m], left=0.0, right=0.0)

rng = np.random.default_rng(12345)


def synth(scale):
    """Random-phase realisation of the modal forces, one block."""
    amp = np.sqrt(SQi * dfs / 2.0) * nfft * scale
    ph = rng.uniform(0, 2 * np.pi, SQi.shape)
    X = amp * np.exp(1j * ph)
    X[0] = 0.0; X[-1] = np.real(X[-1])
    return np.fft.irfft(X, n=nfft, axis=0)


def integrate(Q, qt, active):
    """RK4, N_SUB substeps per sample, modal and decoupled. Returns q(t)."""
    dt = 1.0 / FS / N_SUB
    n = Q.shape[0]
    q = np.zeros(nm); v = np.zeros(nm)
    out = np.empty((n, nm))
    two_zw = 2 * zeta * wn

    def acc(qq, vv, F):
        a = F - two_zw * vv - wn2 * qq
        if active:
            a -= (wn2 * qq * (1.0 - alpha) * np.expm1(-(qq / qt) ** 2)
                  + c2 * vv * np.abs(vv))
        return a

    for i in range(n):
        F0 = Q[i]; F1 = Q[(i + 1) % n]
        for s in range(N_SUB):
            Fa = F0 + (F1 - F0) * (s / N_SUB)
            Fb = F0 + (F1 - F0) * ((s + 0.5) / N_SUB)
            Fc = F0 + (F1 - F0) * ((s + 1.0) / N_SUB)
            k1v = acc(q, v, Fa); k1q = v
            k2v = acc(q + .5 * dt * k1q, v + .5 * dt * k1v, Fb); k2q = v + .5 * dt * k1v
            k3v = acc(q + .5 * dt * k2q, v + .5 * dt * k2v, Fb); k3q = v + .5 * dt * k2v
            k4v = acc(q + dt * k3q, v + dt * k3v, Fc);           k4q = v + dt * k3v
            q = q + (dt / 6) * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[i] = q
    return out


def measured_shift(q, qt):
    """
    Frequency shift measured from the trajectory, NOT from a PSD peak.

    Statistical linearisation defines k_eq = E[F(q) q]/E[q^2]; F here is the
    true nonlinear restoring force the integrator used, so this is the realised
    equivalent stiffness with no spectral-resolution limit and no windowing
    bias.  The system is modally decoupled, so k_eq sets the resonance directly
    and 1 - sqrt(k_eq/wn^2) IS the frequency shift.  A PSD peak at 0.5 Hz
    resolution cannot resolve the 0.35% expected at -12 dB; this can.
    """
    F = wn2 * q * (alpha + (1.0 - alpha) * np.exp(-(q / qt) ** 2))
    keq = np.mean(F * q, axis=0) / np.mean(q ** 2, axis=0)
    return 100.0 * (1.0 - np.sqrt(keq / wn2))


print(f'fs={FS:g} Hz, {N_BLOCK} x {T_BLOCK:g} s blocks, N_SUB={N_SUB}, '
      f'df={dfs:g} Hz, {nm} modes')
print(f'design amplitude q_control: {q_design.min():.3e} - {q_design.max():.3e}')

for strength in STRENGTHS:
    qt = qt0 / np.sqrt(strength)
    print(f'\n================  STRENGTH {strength:g}  ================')
    for db in LEVELS_DB:
        sc = 10 ** (db / 20.0)
        blocks_lin, blocks_nl = [], []
        for b in range(N_BLOCK):
            Q = synth(sc)
            blocks_lin.append(integrate(Q, qt, False))
            blocks_nl.append(integrate(Q, qt, True))
        ql = np.concatenate(blocks_lin); qn = np.concatenate(blocks_nl)
        if not (np.all(np.isfinite(ql)) and np.all(np.isfinite(qn))):
            print(f'  {db:+6.1f} dB  ***  NON-FINITE -- ESCAPED  ***')
            continue
        shift = measured_shift(qn, qt)
        sig = qn.std(axis=0)
        pred = 100 * (1 - np.sqrt(alpha + (1 - alpha)
                                  * (1 + 2 * (sig / qt) ** 2) ** -1.5))
        lin_ratio = ql.std(axis=0) / (q_design * sc)
        rel = np.max(np.abs(shift - pred) / np.maximum(pred, 1e-12))
        print(f'  {db:+6.1f} dB  linear sigma/design {lin_ratio.min():.3f}-{lin_ratio.max():.3f}'
              f'   max|q|/qt {np.max(np.abs(qn) / qt):.2f}')
        print(f'            measured  shift med {np.median(shift):6.3f}%  '
              f'[{shift.min():6.3f}, {shift.max():6.3f}]')
        print(f'            predicted shift med {np.median(pred):6.3f}%  '
              f'[{pred.min():6.3f}, {pred.max():6.3f}]   max rel err {rel:.2%}')
