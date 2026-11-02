"""
verify_modal_shift_damping.py

Direct time integration check on modal_shift_and_damping.py.

For each selected mode the same band-limited Gaussian modal force drives two
simulations -- one fully LINEAR (g=1, c2=0) and one NONLINEAR -- with identical
noise and identical integrator, so the integrator's own frequency bias cancels
in the ratio.  Many independent realisations run in parallel to buy statistics.

Three things are measured, none of them by curve fitting:

  f_eff    peak of the response PSD (parabolic interpolation), nonlinear
           relative to linear.
  zeta_eq  the TRUE equivalent-linear damping, from measured power balance:
               c = (2*zeta0*wn*E[v^2] + c2*E[|v|^3]) / E[v^2]
           This also tests the Gaussian assumption directly, by reporting the
           measured E[|v|^3]/sigma_v^3 against 2*sqrt(2/pi) = 1.5958.
  zeta_app the APPARENT damping a modal fit returns, from the half-power width
           of the response PSD.  It is larger, because the resonance wanders
           with instantaneous amplitude and smears the peak.  This is the
           number an FRF-based modal fit on measured data would report.

Usage:  python3 verify_modal_shift_damping.py [level_db] [n_modes]
"""
import os
import sys
import json
import numpy as np

import modal_shift_and_damping as M

FS = 20000.0
SKIP = 0
T_REC = 20.0
N_REAL = 24
SEED = 20260929
E_ABS_V3_GAUSS = 2.0 * np.sqrt(2.0 / np.pi)      # 1.59577


def make_force(SQ_i, fgrid, n, fs, n_real, rng, scale):
    """n_real independent realisations with one-sided modal force PSD SQ_i."""
    nf = n // 2 + 1
    fx = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.interp(fx, fgrid, SQ_i, left=0.0, right=0.0)
    amp = np.sqrt(np.maximum(P, 0.0) * fs * n / 2.0)
    ph = rng.uniform(0, 2 * np.pi, size=(n_real, nf))
    X = amp[None, :] * np.exp(1j * ph)
    X[:, 0] = 0.0
    if n % 2 == 0:
        X[:, -1] = np.abs(X[:, -1])
    return (np.fft.irfft(X, n=n, axis=1) * scale).astype(np.float64)


def integrate(F, dt, wn, zeta0, wn2, alpha, qt, c2, nonlinear):
    """Central difference, velocity by central estimate with one correction."""
    n_real, n = F.shape
    q = np.zeros(n_real); qm = np.zeros(n_real)
    global SKIP
    v = np.zeros(n_real)
    c_lin = 2.0 * zeta0 * wn
    Q = np.empty((n_real, n), np.float32)
    s2 = 0.0; s3 = 0.0                      # velocity moments, accumulated
    skip = n // 20                          # drop the start-up transient
    for k in range(n):
        if nonlinear:
            fk = wn2 * q + wn2 * q * (1.0 - alpha) * np.expm1(-(q / qt) ** 2)
            fd = c_lin * v + c2 * v * np.abs(v)
        else:
            fk = wn2 * q
            fd = c_lin * v
        a = F[:, k] - fd - fk
        qp = 2.0 * q - qm + dt * dt * a
        v = (qp - qm) / (2.0 * dt)
        Q[:, k] = q
        if k >= skip:
            s2 += float(np.sum(v * v)); s3 += float(np.sum(np.abs(v) ** 3))
        qm, q = q, qp
    cnt = (n - skip) * n_real
    SKIP = skip
    return Q[:, skip:], s2 / cnt, s3 / cnt


def frf_peak_and_halfpower(Q, F, fs, f_expect, span=0.06):
    """Peak frequency and half-power bandwidth of |H|^2 = |S_fq/S_ff|^2.

    The modal force here is the SHAPED control drive, which match_trace
    notches at each resonance, so the raw response PSD does not peak at the
    mode.  Dividing by the force spectrum removes the shaping -- this is the
    same H1 estimate a measurement would form -- and the search is confined
    to a window around the linear natural frequency."""
    from scipy import signal
    nper = min(1 << 16, Q.shape[1] // 4)
    fx, Sff = signal.welch(F, fs=fs, nperseg=nper, axis=-1)
    _, Sfq = signal.csd(F, Q.astype(np.float64), fs=fs, nperseg=nper, axis=-1)
    H2 = np.abs(np.mean(Sfq, axis=0) / np.mean(Sff, axis=0)) ** 2
    band = (fx > f_expect * (1 - span)) & (fx < f_expect * (1 + span))
    S = np.where(band, H2, 0.0)
    i = int(np.argmax(S))
    if 0 < i < len(S) - 1:
        y0, y1, y2 = S[i - 1], S[i], S[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    else:
        d = 0.0
    fpk = fx[i] + d * (fx[1] - fx[0])
    half = S[i] / 2.0
    lo = i
    while lo > 0 and S[lo] > half:
        lo -= 1
    hi = i
    while hi < len(S) - 1 and S[hi] > half:
        hi += 1
    def cross(a, b):
        if S[a] == S[b]:
            return fx[a]
        return fx[a] + (fx[b] - fx[a]) * (S[a] - half) / (S[a] - S[b])
    bw = cross(hi - 1, hi) - cross(lo + 1, lo)
    return fpk, bw


def white_test(P, level, nmod):
    """Frequency shift against the kernel, on clean white-band excitation.

    The shaped control drive is NOTCHED at every resonance, so an H1 estimate
    formed from it is starved of excitation exactly where the peak is and the
    measured peak frequency is biased by however deep that mode's notch is.
    That contaminates the frequency check but not the response check, which is
    why the two are separated.

    Here each mode is driven by flat noise over a band around itself, sigma_q
    is MEASURED from the simulation, and the closed-form Gaussian prediction
    is evaluated at that measured sigma_q.  Nothing is assumed about what
    response the drive produces -- only that the kernel softens by the
    predicted amount once the response is known.
    """
    scale = 10.0 ** (level / 20.0)
    ref = M.solve_level(P, scale)
    soft = np.where(P['mask'])[0]
    pick = soft[np.linspace(0, len(soft) - 1, nmod).astype(int)]
    n = int(FS * T_REC)
    dt = 1.0 / FS
    rng = np.random.default_rng(SEED + 7)
    rows = []
    print(f'white-band check, response sized on the {level:+.0f} dB state')
    print()
    print('  mode    sigma_q      shift %  (at the MEASURED sigma_q)    '
          'apparent zeta')
    print('   Hz      sim        pred     meas    meas/pred            '
          '/true zeta')
    print(' ' + '-' * 74)
    for i in pick:
        fn = P['fn'][i]
        band = (P['f'] > fn * 0.75) & (P['f'] < fn * 1.25)
        SQw = np.where(band, 1.0, 0.0)
        # flat level chosen to reproduce the linear response of the real drive
        lin_ref = np.sqrt(np.sum(P['SQ'][band, i]) / max(band.sum(), 1))
        F = make_force(SQw * lin_ref ** 2, P['f'], n, FS, N_REAL, rng, scale)
        Qlin, _, _ = integrate(F, dt, P['wn'][i], P['zeta0'][i], P['wn2'][i],
                               P['alpha'], P['qt'][i], P['c2'][i], False)
        Qnl, mv2, mv3 = integrate(F, dt, P['wn'][i], P['zeta0'][i], P['wn2'][i],
                                  P['alpha'], P['qt'][i], P['c2'][i], True)
        Ft = F[:, SKIP:]
        f_lin, bw_lin = frf_peak_and_halfpower(Qlin, Ft, FS, fn)
        f_nl, bw_nl = frf_peak_and_halfpower(Qnl, Ft, FS, fn)
        sq = float(np.sqrt(np.mean(Qnl.astype(np.float64) ** 2)))
        g = P['alpha'] + (1 - P['alpha']) * (1 + 2 * (sq / P['qt'][i]) ** 2) ** -1.5
        shift_pred = (1.0 - np.sqrt(g)) * 100.0
        shift_meas = (1.0 - f_nl / f_lin) * 100.0
        # true equivalent-linear damping ratio from power balance, vs the
        # apparent one a half-power fit returns
        c_lin = 2.0 * P['zeta0'][i] * P['wn'][i]
        zr = (c_lin + P['c2'][i] * mv3 / mv2) / c_lin
        zeta_true = zr / np.sqrt(g)
        zeta_app = (bw_nl / f_nl) / (bw_lin / f_lin)
        rows.append(dict(mode_hz=float(fn), sigma_q_sim=sq,
                         shift_pred_at_meas_sigma=float(shift_pred),
                         shift_meas=float(shift_meas),
                         zeta_true_over_zeta0=float(zeta_true),
                         zeta_apparent_over_zeta0=float(zeta_app),
                         apparent_over_true=float(zeta_app / zeta_true)))
        r = rows[-1]
        print(f'{fn:7.1f} {sq:11.4e} {shift_pred:9.3f} {shift_meas:8.3f}'
              f'    {shift_meas/shift_pred:8.3f}          {r["apparent_over_true"]:9.3f}')
    out = os.path.join(M.NOTES, f'modal_shift_damping_white_{int(level):+d}dB.json')
    with open(out, 'w') as fh:
        json.dump(dict(level_db=level, modes=rows), fh, indent=1)
    print()
    print('wrote', os.path.basename(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'white':
        P = M.build()
        white_test(P, float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
                   int(sys.argv[3]) if len(sys.argv) > 3 else 4)
        return
    level = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    nmod = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    scale = 10.0 ** (level / 20.0)

    P = M.build()
    pred = M.solve_level(P, scale)
    soft = np.where(P['mask'])[0]
    # spread the sample across the softened band
    pick = soft[np.linspace(0, len(soft) - 1, nmod).astype(int)]

    n = int(FS * T_REC)
    dt = 1.0 / FS
    rng = np.random.default_rng(SEED)
    rows = []
    print(f'level {level:+.0f} dB   fs={FS:.0f} Hz  T={T_REC:.0f} s  '
          f'x{N_REAL} realisations  ({N_REAL*T_REC:.0f} s per mode)')
    print()
    print('  mode      shift %         c_eq / c_lin        E|v|^3/sv^3   zeta_app')
    print('   Hz     pred    meas     pred    meas   ratio    meas  (1.5958) '
          ' /zeta_eq')
    print(' ' + '-' * 78)
    for i in pick:
        F = make_force(P['SQ'][:, i], P['f'], n, FS, N_REAL, rng, scale)
        Qlin, _, _ = integrate(F, dt, P['wn'][i], P['zeta0'][i], P['wn2'][i],
                               P['alpha'], P['qt'][i], P['c2'][i], False)
        Qnl, mv2, mv3 = integrate(F, dt, P['wn'][i], P['zeta0'][i], P['wn2'][i],
                                  P['alpha'], P['qt'][i], P['c2'][i], True)
        Ft = F[:, SKIP:]
        f_lin, bw_lin = frf_peak_and_halfpower(Qlin, Ft, FS, P['fn'][i])
        f_nl, bw_nl = frf_peak_and_halfpower(Qnl, Ft, FS, P['fn'][i])
        shift_meas = (1.0 - f_nl / f_lin) * 100.0

        sv = np.sqrt(mv2)
        m3 = mv3
        c_lin = 2.0 * P['zeta0'][i] * P['wn'][i]
        c_meas = c_lin + P['c2'][i] * m3 / mv2
        zr_meas = c_meas / c_lin
        zeta_eq_meas = zr_meas / (f_nl / f_lin)          # re zeta0
        zeta_app = (bw_nl / f_nl) / (bw_lin / f_lin)     # re the linear fit
        rows.append(dict(mode_hz=float(P['fn'][i]),
                         shift_pred=float(pred['shift_pct'][i]),
                         shift_meas=float(shift_meas),
                         zr_pred=float(pred['zr'][i]),
                         zr_meas=float(zr_meas),
                         f_lin_hz=float(f_lin), f_nl_hz=float(f_nl),
                         bw_lin_hz=float(bw_lin), bw_nl_hz=float(bw_nl),
                         zeta_ratio_pred=float(pred['zeta_ratio'][i]),
                         zeta_eq_meas=float(zeta_eq_meas),
                         absv3_over_sv3=float(m3 / sv ** 3),
                         zeta_apparent_over_eq=float(zeta_app / zeta_eq_meas),
                         sigma_q_sim=float(np.sqrt(np.mean(Qnl.astype(np.float64) ** 2))),
                         sigma_q_pred=float(pred['sq'][i])))
        r = rows[-1]
        print(f'{P["fn"][i]:7.1f} {r["shift_pred"]:7.3f} {r["shift_meas"]:7.3f}'
              f'  {r["zr_pred"]:8.4f} {r["zr_meas"]:7.4f}'
              f' {r["zr_meas"]/r["zr_pred"]:6.3f}'
              f'   {r["absv3_over_sv3"]:8.4f}'
              f'    {r["zeta_apparent_over_eq"]:7.3f}')
    out = os.path.join(M.NOTES, f'modal_shift_damping_verify_{int(level):+d}dB.json')
    with open(out, 'w') as fh:
        json.dump(dict(level_db=level, fs=FS, t_rec=T_REC, n_real=N_REAL,
                       gauss_absv3=E_ABS_V3_GAUSS, modes=rows), fh, indent=1)
    print()
    print('wrote', os.path.basename(out))


if __name__ == '__main__':
    main()
