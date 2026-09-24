#!/usr/bin/env python3
"""
bounded_softening.py

Per-mode BOUNDED softening for the 33-mode frame plant -- the replacement for
the softening cubic that escaped on runs 42/44/46/48.

THE KERNEL
    g(q)  = alpha + (1 - alpha) * exp(-(q/qt)^2)
    F(q)  = wn^2 * q * g(q)

and the virtual hardware carries only the DEPARTURE from linear,

    nl_term(q) = F(q) - wn^2 q = wn^2 q (1-alpha) * (exp(-(q/qt)^2) - 1)   <= 0.

WHY IT CANNOT ESCAPE.  alpha > 0 makes both factors of F positive for q > 0,
so the restoring force never reverses sign and the potential grows without
bound, V(q) -> 0.5*alpha*wn^2*q^2.  There is no finite well.  Contrast the
cubic F = wn^2 q + k3 q^3, k3 < 0, which reaches F = 0 at q_esc = wn/sqrt|k3|
and runs away beyond it.  That is the entire failure mode of the four
nonlinear runs, and this removes it structurally rather than by margin.

The idea is the user's, from a Mathematica/MATLAB SDOF demo whose stiffness
    Ks(x,r) = x*(exp(-1e6*|x/3|) + 0.5*r)
saturates onto a residual linear stiffness instead of going negative.  Two
changes were made porting it here:

  1. NORMALIZED so g(0) = 1.  In the original, Ks'(0) = 1 + 0.5r, so the
     small-amplitude natural frequency is w*sqrt(1+0.5r), not w.  Here the
     small-amplitude modal frequency is EXACTLY the measured wn, which is what
     lets the article start from the same linear FRF it always did.
  2. GAUSSIAN kernel exp(-(q/qt)^2) instead of exp(-|q|/qt).  The Gaussian
     matches the cubic term for term at small amplitude:
         g ~ 1 - (1-alpha) q^2/qt^2   <->   k3 = -(1-alpha) wn^2/qt^2
     so the whole escapecal calibration maps across with qt = q_esc*sqrt(1-alpha),
     and the shift stays quadratic in level (6 dB of level -> 4x the shift)
     instead of linear.  The |q| form has a kink at q = 0 and gives a shift
     linear in level, which compresses the contrast across a level change.

STATISTICAL LINEARIZATION (this is a RANDOM vibration test, not a sine dwell).
For q ~ N(0, sigma^2) the equivalent stiffness has a closed form:

    k_eq/wn^2 = alpha + (1-alpha) * (1 + 2 sigma^2/qt^2)^(-3/2)
    f(sigma)/f_linear = sqrt(k_eq/wn^2)

Its small-sigma limit is 1 - 1.5(1-alpha) sigma^2/qt^2, identical to the
cubic's 1 + 1.5 k3 sigma^2/wn^2 -- checked numerically in self_test().
Inverting it for qt is also closed form, so no root-finding is needed.

STRENGTH.  RATTLESNAKE_NONLINEARITY_STRENGTH scales qt as qt/sqrt(strength),
which reproduces k3_eff = strength*k3 at small amplitude AND leaves alpha
untouched.  So unlike escapecal -- where the design point was strength 1 with
no headroom -- strength here may be set to anything at all, including 1000,
and the plant still cannot escape.  It only softens harder.

MODE SELECTION.  Only modes resonating inside the control band are softened;
everything else gets qt = inf, which makes g identically 1 -- bit-exactly
linear, not approximately.  Out-of-band modes contribute a near-static
compliance term to the in-band FRF, so softening them adds integration cost
and risk while changing nothing the control loop can observe.
"""
import numpy as np

# min over u of exp(-u^2)(1-2u^2) = -0.446260...; below this alpha the tangent
# stiffness dF/dq dips negative over a band.  Still bounded, still no escape,
# but jumpy and stiff to integrate, so we stay above it.
ALPHA_MONOTONE_MIN = 0.30862
ALPHA_DEFAULT = 0.35
BAND_DEFAULT = (100.0, 1000.0)


# ------------------------------------------------------------------ kernel --
def soften_g(q, alpha, qt):
    """Stiffness multiplier. g(0) = 1, g(inf) = alpha. qt = inf -> exactly 1."""
    q = np.asarray(q, float)
    return alpha + (1.0 - alpha) * np.exp(-(q / qt) ** 2)


def soften_nl_term(q, wn2, alpha, qt):
    """
    The DEPARTURE from the linear restoring force, which is what the virtual
    hardware adds:  wn^2 q (g(q) - 1) = wn^2 q (1-alpha) (exp(-(q/qt)^2) - 1).

    expm1 keeps this exact as q -> 0 (where the departure must vanish) and
    saturates at -1 for large q, so it cannot overflow at any amplitude --
    unlike k3*q**3, which is what actually raised the RuntimeWarning on run 44.
    """
    q = np.asarray(q, float)
    return wn2 * q * (1.0 - alpha) * np.expm1(-(q / qt) ** 2)


def qt_from_q_esc(q_esc, alpha):
    """Map an escapecal cubic calibration onto this kernel."""
    return np.asarray(q_esc, float) * np.sqrt(1.0 - alpha)


def k3_equivalent(wn, alpha, qt):
    """The cubic coefficient this kernel reproduces at small amplitude."""
    return -(1.0 - alpha) * np.asarray(wn, float) ** 2 / np.asarray(qt, float) ** 2


# ------------------------------------------- statistical linearization ------
def freq_ratio_random(sigma, alpha, qt):
    """f/f_linear for a zero-mean Gaussian modal response of std `sigma`."""
    sigma = np.asarray(sigma, float)
    keq = alpha + (1.0 - alpha) * (1.0 + 2.0 * (sigma / qt) ** 2) ** -1.5
    return np.sqrt(keq)


def qt_for_target_shift(sigma, target_shift, alpha):
    """
    Closed-form inverse: qt such that freq_ratio_random(sigma) = 1-target_shift.
    Requires (1-target)^2 > alpha, i.e. the target must be inside the kernel's
    asymptotic reach -- max achievable shift is 1 - sqrt(alpha).
    """
    r2 = (1.0 - target_shift) ** 2
    if np.any(r2 <= alpha):
        raise ValueError(
            "target shift %.3f unreachable at alpha=%.3f (max %.3f)"
            % (target_shift, alpha, 1.0 - np.sqrt(alpha)))
    u = ((1.0 - alpha) / (r2 - alpha)) ** (2.0 / 3.0)   # = 1 + 2 sigma^2/qt^2
    return np.asarray(sigma, float) * np.sqrt(2.0 / (u - 1.0))


def freq_ratio_sine(A, alpha, qt, n=20001):
    """First-harmonic describing function, for cross-checking the random one."""
    th = np.linspace(0.0, 2.0 * np.pi, n)
    s = np.sin(th)
    keq = np.trapezoid(soften_g(A * s, alpha, qt) * s ** 2, th) / np.pi
    return np.sqrt(max(keq, 0.0))


# ----------------------------------------------------------- mode select ----
def select_in_band(mode_freqs_hz, band=BAND_DEFAULT):
    f = np.asarray(mode_freqs_hz, float)
    return (f >= band[0]) & (f <= band[1])


def calibrate(mode_freqs_hz, q_control, target_shift=0.05, alpha=ALPHA_DEFAULT,
              band=BAND_DEFAULT):
    """
    Size qt per mode from the modal response std under the 0 dB control drive.

    NOTE the deliberate change of bias from escapecal.  That builder used
    max(q_control, q_sysid) because under-estimating the amplitude meant
    ESCAPE.  Here under-estimating merely means a larger shift than designed,
    which is harmless, so we size on the amplitude at the condition we actually
    want to hit the target -- the 0 dB control run.
    """
    f = np.asarray(mode_freqs_hz, float)
    qc = np.asarray(q_control, float)
    mask = select_in_band(f, band)
    qt = np.full(f.shape, np.inf)
    qt[mask] = qt_for_target_shift(qc[mask], target_shift, alpha)
    return mask, qt


def sysid_backoff_db(q_control, q_sysid, target_shift, mask,
                     allow_shift=0.005, alpha=ALPHA_DEFAULT):
    """
    How far the SYSTEM ID drive must be backed off so the identified FRF is a
    genuinely small-amplitude reference rather than an already-softened one.

    The shift is quadratic in amplitude, so a mode identified at q_sysid picks
    up target_shift*(q_sysid/q_control)^2 of softening.  On this plant the
    per-mode ratio q_sysid/q_control spans 0.25-1.2 (measured, escapecal), so
    the worst mode can identify with MORE softening than the control run shows.
    Un-backed-off, that contaminates the baseline mode-dependently -- which is
    a candidate explanation for run 48's system ID "looking significantly
    noisier than the linear system sysid".
    """
    qc = np.asarray(q_control, float)
    qs = np.asarray(q_sysid, float)
    r = np.where(mask, qs / np.where(qc > 0, qc, np.inf), 0.0)
    worst = float(r.max())
    shift_now = target_shift * worst ** 2
    need_db = 20.0 * np.log10(np.sqrt(allow_shift / target_shift) / worst)
    return worst, shift_now, need_db


# ------------------------------------------------------------- self test ----
def self_test(verbose=True):
    # 1. normalization and saturation
    assert abs(soften_g(0.0, 0.35, 1.0) - 1.0) < 1e-14
    assert abs(soften_g(1e6, 0.35, 1.0) - 0.35) < 1e-12
    assert soften_nl_term(0.0, 1.0, 0.35, 1.0) == 0.0
    assert np.isfinite(soften_nl_term(1e30, 1.0, 0.35, 1.0))     # cubic overflows here
    assert soften_g(5.0, 0.35, np.inf) == 1.0                    # qt=inf -> exactly linear
    assert soften_nl_term(5.0, 1.0, 0.35, np.inf) == 0.0

    # 2. NO ESCAPE: restoring force strictly positive, potential unbounded
    for alpha in (0.15, 0.35, 0.60):
        q = np.linspace(1e-9, 500.0, 1_000_001)
        F = q + soften_nl_term(q, 1.0, alpha, 1.0)
        assert F.min() > 0.0, "force reversed at alpha=%g" % alpha
        assert np.trapezoid(F, q) > 0.4 * alpha * 500.0 ** 2

    # 3. tangent-stiffness monotonicity bound
    q = np.linspace(0.0, 20.0, 1_000_001)
    g_lo = np.gradient(q + soften_nl_term(q, 1.0, ALPHA_MONOTONE_MIN - 0.05, 1.0), q).min()
    g_hi = np.gradient(q + soften_nl_term(q, 1.0, ALPHA_MONOTONE_MIN + 0.05, 1.0), q).min()
    assert g_lo < 0.0 < g_hi, "monotonicity bound wrong (%.4f, %.4f)" % (g_lo, g_hi)
    assert ALPHA_DEFAULT > ALPHA_MONOTONE_MIN

    # 4. random SL reduces to the cubic at small amplitude (the drop-in claim)
    alpha, qt, wn = 0.35, 1.0, 1.0
    k3 = k3_equivalent(wn, alpha, qt)
    # agreement is O(sigma^2) relative, so test the CONVERGENCE RATE, not just
    # a tolerance -- halving sigma must quarter the relative discrepancy.
    errs = []
    for sig in (0.20, 0.10, 0.05, 0.025):
        sat = 1.0 - freq_ratio_random(sig, alpha, qt)
        cub = 1.0 - np.sqrt(1.0 + 3.0 * k3 * sig ** 2 / wn ** 2)
        errs.append(abs(sat - cub) / cub)
    assert errs[-1] < 2e-3, "cubic mismatch at sigma=0.025: %.5f" % errs[-1]
    for a, b in zip(errs[:-1], errs[1:]):
        assert 3.5 < a / b < 4.6, "not 2nd-order convergent: ratio %.3f" % (a / b)

    # 5. closed-form inverse round-trips
    for tgt in (0.02, 0.05, 0.10, 0.25):
        qt_ = qt_for_target_shift(1.0, tgt, 0.35)
        assert abs((1.0 - freq_ratio_random(1.0, 0.35, qt_)) - tgt) < 1e-12

    # 6. escapecal mapping is consistent
    q_esc, alpha = 3.0, 0.35
    qt_ = qt_from_q_esc(q_esc, alpha)
    assert abs(k3_equivalent(1.0, alpha, qt_) - (-(1.0 / q_esc ** 2))) < 1e-12

    # 7. level scaling is quadratic TO LEADING ORDER only.  At a 5% design
    # shift, -6 dB gives 0.271 of it rather than 0.250 -- 8% high, because the
    # kernel is already slightly saturated at the design point.  Design tables
    # must therefore evaluate freq_ratio_random at each level rather than
    # scaling the 0 dB number by 4^n.
    qt_ = qt_for_target_shift(1.0, 0.05, 0.35)
    r = (1.0 - freq_ratio_random(0.5, 0.35, qt_)) / 0.05
    assert 0.24 < r < 0.28, "level scaling badly non-quadratic: %.4f" % r
    tiny = qt_for_target_shift(1.0, 1e-4, 0.35)
    r0 = (1.0 - freq_ratio_random(0.5, 0.35, tiny)) / 1e-4
    assert abs(r0 - 0.25) < 1e-3, "small-signal limit not quadratic: %.5f" % r0

    if verbose:
        print("bounded_softening self_test: all 7 checks pass")
    return True


if __name__ == "__main__":
    self_test()
