"""
modal_shift_and_damping.py

How much does the bounded-softening nonlinear plant actually move each mode's
FREQUENCY and DAMPING, as a function of test level?

Two effects move the damping RATIO, and they are easy to conflate:

  1. The quadratic damping term  c2*qd*|qd|.  Its equivalent linear coefficient
     under a GAUSSIAN process comes from power balance, not from the sinusoidal
     describing function:

         E[|qd|^3] = 2*sqrt(2/pi) * sigma_v^3
         c_eq      = 2*sqrt(2/pi) * c2 * sigma_v

     The builder sized c2 with the SINUSOIDAL result  c_eq = (8/3pi)*c2*V,
     which under random excitation understates the increment by (3/4)*sqrt(2pi)
     = 1.880.  Designed zeta/zeta0 = 1.10 at the design amplitude is therefore
     1.188 in the random-vibration test the article is actually used in.

  2. Softening alone raises the damping RATIO even with c2 = 0.  The equation
     is  qdd + 2*zeta0*wn*qd + g*wn^2*q = f, so the damping COEFFICIENT is
     fixed while the natural frequency falls to wn*sqrt(g).  The ratio is
     therefore zeta0/sqrt(g) -- it rises as 1/sqrt(g) for free.

Everything here is the equivalent-linear (statistical linearization) answer:
the properties of the best linear system matching the response.  A modal fit
to a MEASURED FRF returns something broader still, because the resonance
frequency wanders with the instantaneous amplitude and smears the peak.  That
apparent damping is estimated separately in verify_modal_shift_damping.py.

Outputs modal_shift_and_damping.json next to the other run notes.
"""
import os
import json
import numpy as np
import scipy.linalg as sla
import netCDF4

from bounded_softening import BAND_DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.normpath(os.path.join(HERE, '..', 'results', 'case'))
RUNS = os.path.normpath(os.path.join(HERE, '..', 'results', 'runs'))
NOTES = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
LINEAR = os.path.join(CASE, 'sdynpy_frame6x12_system.npz')
NONLIN = os.path.join(CASE, 'sdynpy_frame6x12_system_nonlinear_boundedsoften.npz')
REF_RUN = os.path.join(RUNS, 'run40_matchtrace_refresh005_spec.nc4')
BAND = BAND_DEFAULT
LEVELS_DB = (-18.0, -12.0, -6.0, -3.0, 0.0, 2.0, 4.0, 8.0)

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
SIN_TO_GAUSS = 0.75 * np.sqrt(2.0 * np.pi)      # 1.8800


def build():
    lin = dict(np.load(LINEAR, allow_pickle=True))
    nl = dict(np.load(NONLIN, allow_pickle=True))
    M, C, K = lin['mass'], lin['damping'], lin['stiffness']

    ev, V = sla.eigh(K, M)
    wn = np.sqrt(np.maximum(ev, 0.0))
    o = np.argsort(wn); wn, V = wn[o], V[:, o]
    flex = np.where(wn / (2 * np.pi) > 1.0)[0]
    PHI, wn_f = V[:, flex], wn[flex]
    zeta0 = np.einsum('di,de,ei->i', PHI, C, PHI) / (2 * wn_f)

    didx = np.asarray(nl['nl_drive_dof_indices'])
    ds = netCDF4.Dataset(REF_RUN); grp = ds['Frame 6x12 Random']
    D = np.array(grp['drive_cpsd_real'][:]) + 1j * np.array(grp['drive_cpsd_imag'][:])
    f = np.array(grp['specification_frequency_lines'][:], float); ds.close()

    PhiD = PHI[didx, :]
    SQ = np.maximum(np.real(np.einsum('km,fkl,lm->fm', PhiD.conj(), D, PhiD)), 0.0)
    return dict(f=f, SQ=SQ, wn=wn_f, zeta0=zeta0,
                fn=wn_f / (2 * np.pi), wn2=wn_f ** 2,
                qt=nl['nl_soft_qts'], c2=nl['nl_c2s'],
                mask=nl['nl_soft_mask'], alpha=float(nl['nl_soft_alpha']))


def moments(P, g, zr, scale):
    """sigma_q and sigma_v for every mode, given stiffness ratio g and a
    multiplier zr on the LINEAR damping coefficient."""
    f, SQ = P['f'], P['SQ']
    w = 2 * np.pi * f
    dfg = f[1] - f[0]
    H = 1.0 / (-(w[:, None] ** 2)
               + 2j * zr[None, :] * P['zeta0'][None, :] * P['wn'][None, :] * w[:, None]
               + g[None, :] * P['wn2'][None, :])
    A = SQ * np.abs(H) ** 2
    sq = np.sqrt(np.sum(A, axis=0) * dfg) * scale
    sv = np.sqrt(np.sum(A * (w[:, None] ** 2), axis=0) * dfg) * scale
    return sq, sv


def solve_level(P, scale, n_it=600, relax=0.3):
    """Joint fixed point on (stiffness ratio, damping multiplier)."""
    n = len(P['wn'])
    g = np.ones(n); zr = np.ones(n)
    for _ in range(n_it):
        sq, sv = moments(P, g, zr, scale)
        gn = np.where(P['mask'],
                      P['alpha'] + (1 - P['alpha']) * (1 + 2 * (sq / P['qt']) ** 2) ** -1.5,
                      1.0)
        # power balance: c_eq = 2*sqrt(2/pi)*c2*sigma_v, and the linear
        # coefficient is 2*zeta0*wn, so the multiplier is 1 + that over it.
        zn = 1.0 + SQRT_2_OVER_PI * P['c2'] * sv / (P['zeta0'] * P['wn'])
        gn = np.where(np.isfinite(gn), gn, 1.0)
        zn = np.where(np.isfinite(zn), zn, 1.0)
        if max(np.max(np.abs(gn - g)), np.max(np.abs(zn - zr))) < 1e-13:
            g, zr = gn, zn
            break
        g = (1 - relax) * g + relax * gn
        zr = (1 - relax) * zr + relax * zn
    sq, sv = moments(P, g, zr, scale)
    root_g = np.sqrt(g)
    return dict(g=g, zr=zr, sq=sq, sv=sv,
                f_ratio=root_g,                     # f_eff / f_linear
                shift_pct=(1.0 - root_g) * 100.0,   # softening, positive
                zeta_ratio=zr / root_g,             # zeta_eff / zeta0
                zeta_from_quadratic=zr,             # the c2 term alone
                zeta_from_softening=1.0 / root_g)   # the 1/sqrt(g) term alone


def main():
    P = build()
    out = {'modes': {'f_hz': P['fn'].tolist(),
                     'zeta0': P['zeta0'].tolist(),
                     'softened': P['mask'].tolist()},
           'sinusoidal_to_gaussian_factor': SIN_TO_GAUSS,
           'levels': []}
    print(f"{len(P['fn'])} modes, {P['fn'].min():.1f}-{P['fn'].max():.1f} Hz, "
          f"{int(P['mask'].sum())} softened")
    print()
    print(' level   ---- frequency shift % ----   ---- zeta/zeta0 ----   '
          'of which 1/sqrt(g)')
    print('  dB      median     max    (linear)    median     max        median')
    print(' ' + '-' * 76)
    for L in LEVELS_DB:
        r = solve_level(P, 10.0 ** (L / 20.0))
        m = P['mask']
        row = dict(level_db=L,
                   shift_pct=r['shift_pct'].tolist(),
                   zeta_ratio=r['zeta_ratio'].tolist(),
                   zeta_from_quadratic=r['zeta_from_quadratic'].tolist(),
                   zeta_from_softening=r['zeta_from_softening'].tolist(),
                   sigma_q=r['sq'].tolist(), sigma_v=r['sv'].tolist())
        out['levels'].append(row)
        print(f"{L:+6.0f}   {np.median(r['shift_pct'][m]):7.3f} {r['shift_pct'][m].max():7.3f}"
              f"  {np.max(np.abs(r['shift_pct'][~m])):9.2e}"
              f"   {np.median(r['zeta_ratio'][m]):8.4f} {r['zeta_ratio'][m].max():7.4f}"
              f"     {np.median(r['zeta_from_softening'][m]):8.4f}")
    with open(os.path.join(NOTES, 'modal_shift_and_damping.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print()
    print('wrote modal_shift_and_damping.json')


if __name__ == '__main__':
    main()
