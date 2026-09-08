"""
diagonal_congruence_control.py

A feedback control law that corrects EVERY control channel, not just the
total power per frequency line, while keeping the one structural property
that makes match_trace_pseudoinverse safe.

WHAT match_trace CAN AND CANNOT DO
----------------------------------
match_trace_pseudoinverse computes one real scalar per frequency line --
spec_trace/response_trace -- and multiplies the whole drive CPSD by it.  That
is why it is robust: a uniform positive scalar can never turn a valid drive
CPSD into an invalid one, and after startup it never inverts the FRF again.
It is also why it is limited: if channel 3 is 5 dB high and channel 5 is 5 dB
low, the trace is correct and the law does nothing at all.

THE GENERALIZATION
------------------
Replace the scalar with a DIAGONAL CONGRUENCE:

    X <- D X D,     D = diag(exp(u_1) ... exp(u_N)),  u real

Congruence preserves positive semidefiniteness structurally: if X >= 0 then
D X D >= 0 for ANY real diagonal D.  So, exactly as with match_trace, no
choice of gains -- however wrong the FRF, however noisy the measurement --
can command a physically invalid drive.  match_trace is the special case
D = sqrt(c) I.  What is gained is N independent knobs instead of one, which
is what per-channel correction requires.

CHOOSING THE GAINS
------------------
Let y_m = (H X H^H)_mm be the predicted response auto-spectrum.  Then

    S_mi = d log(y_m) / d u_i = 2 Re( H_mi (X H^H)_im ) / y_m

is the M x N sensitivity of each response to each drive gain.  Given the
MEASURED log error eps_m = log(target_m / achieved_m), solve the small
regularized least-squares problem

    u = argmin || S u - eps ||^2 + lambda ||u||^2

apply a gain, clamp it, and update.  This is one Gauss-Newton step on the
log-domain diagonal-matching problem, restricted to diagonal congruence.

Two properties worth knowing, both checked in the verification suite:

  * Every row of S sums to exactly 2, because scaling all gains together by
    t scales X by e^(2t) and hence y by e^(2t).  A consequence is that when
    the error is the SAME on every channel the solution is u_i = eps/2 for
    all i -- the law reproduces match_trace's update exactly.  It degrades
    gracefully to the law it generalizes.

  * H enters only through S, i.e. only as a DIRECTION.  It is never inverted
    in the loop.  A mediocre FRF still yields a usable direction and the
    feedback cleans up the rest, which is precisely what a pinv-based law
    cannot do -- that is where the drive-imbalance events came from.

AIMING AT WHAT IS REACHABLE
---------------------------
With more control responses than drives, parts of a specification are simply
not achievable (see achievable_response.py).  A law that integrates toward an
unreachable target climbs until something stops it, which is how the earlier
resolve laws reached their anti-windup ceilings.  This law therefore accepts
a PROJECTED target -- the closest reachable response, computed offline by
achievable_response.achievable_diagonal -- and drives to that instead of the
raw specification.  Supply it as the last extra_parameters field:

    from control_laws.achievable_response import achievable_diagonal
    r = achievable_diagonal(H, spec_diag)
    t = spec_diag.copy(); s = r['solved']; t[s] = r['achieved'][s]
    np.savez('projected_target.npz', target_diag=t)     # shape (F, M)

The projection is computed OFFLINE on purpose: it costs ~2 minutes for a
900-line band, which does not fit a control frame.  It is therefore tied to
the FRF it was computed from: REGENERATE IT AFTER A NEW SYSTEM ID.  What is
reachable is a property of the structure, so if the structure changes -- a
fixture reworked, damping altered -- a stale projection aims at the wrong
place.  Nothing detects this automatically.  Without it the law still
runs, aiming at the raw specification; it simply has no way to know which
lines are hopeless.

SAFETY
------
  * congruence  -- the commanded drive is always a valid CPSD, unconditionally
  * max_step_db -- bounds how far any one channel's PREDICTED RESPONSE can
                   move per cycle.  Enforced on the response itself, not on
                   the gains: bounded gains do not imply a bounded response
                   move, because the sensitivity rows sum to 2 but individual
                   entries can be large and opposite in sign.
  * ceiling_db  -- the response may not exceed target by more than this,
                   judged from the MEASURED level extrapolated over one
                   bounded step rather than from the model alone, so the limit
                   survives a wrong or drifting FRF; a physically meaningful anti-windup rather than an
                   arbitrary integrator limit.  It is deliberately NOT subject
                   to max_step_db: coming down from an overshoot is a safety
                   action and happens in one cycle, not over the many cycles a
                   step limit would impose.  max_step_db governs the feedback
                   update; the ceiling only ever reduces level.
  * resonance_sensitivity -- per-line gain scheduling, reusing
                   match_trace_pseudoinverse_pi's proven half-power detector,
                   because a resonance carries dynamical phase lag and has a
                   much lower safe gain than a flat line
  * max_drive_coherence -- defaults to 0.95, NOT 1.0; uncapped drive coherence
                   lets shakers fight each other

extra_parameters (comma separated, all optional):
    rcond, max_drive_coherence, startup_test_level_cap_db, Ki, max_step_db,
    ridge, resonance_sensitivity, ceiling_db, projected_target_file

    rcond                  1e-3  pseudoinverse cutoff, STARTUP SOLVE ONLY
    max_drive_coherence    0.95  pairwise drive coherence cap
    startup_test_level_cap_db -9.0  cap on the first, unguarded command
    Ki                     0.5   loop gain on the Gauss-Newton step
    max_step_db            3.0   max response change per channel per cycle
    ridge                  0.1   Tikhonov weight, relative to trace(S^T S)/N
    resonance_sensitivity  0.5   per-line gain scheduling; 0 disables
    ceiling_db             12.0  predicted response may not exceed target by
                                 more than this; 0 disables
    projected_target_file  none  .npz holding target_diag of shape (F, M)
"""

import numpy as np

from .control_laws import (trace, cpsd_autospectra, _cap_drive_coherence,
                           _half_power_resonance_baseline)

__all__ = ['match_diagonal_congruence', 'parse_diagonal_congruence_parameters']

_TINY = 1e-300

# How much looser the model-based ceiling is than the measurement-based one.
# Not a tuning knob.  It is the amount of FRF error the law will tolerate
# before it stops trying to reach specification and starts protecting instead.
#
# The boundary is real and unavoidable: "the FRF is wrong by 30 dB" and "a
# control accelerometer has failed low" produce IDENTICAL evidence -- the
# model says far too hot, the measurement says far too cold -- and no amount
# of cleverness distinguishes them from these two signals alone.  So a choice
# is forced.  Within the margin the measurement is believed and the test
# reaches level.  Beyond it the law limits drive and the test will sit below
# specification, which is the correct response to evidence that something is
# fundamentally wrong: an FRF that far off is a system-ID problem, not
# something a control law should push through.
MODEL_MARGIN_DB = 20.0


def parse_diagonal_congruence_parameters(extra_parameters,
                                         default_startup_test_level_cap_db=-9.0):
    """Parse the comma-separated extra_parameters string.  Any field that does
    not parse as a number is taken to be the projected-target file path, so the
    path may appear anywhere (in practice last)."""
    parts = [p.strip() for p in extra_parameters.split(',')] if extra_parameters else []
    path = None
    nums = []
    for p in parts:
        if p == '':
            nums.append(None)
            continue
        try:
            nums.append(float(p))
        except ValueError:
            path = p
    def _get(i, default):
        if len(nums) > i and nums[i] is not None:
            return nums[i]
        return default
    return (_get(0, 1e-3),                                  # rcond
            _get(1, 0.95),                                  # max_drive_coherence
            _get(2, default_startup_test_level_cap_db),     # startup cap
            _get(3, 0.5),                                   # Ki
            _get(4, 3.0),                                   # max_step_db
            _get(5, 0.1),                                   # ridge
            _get(6, 0.5),                                   # resonance_sensitivity
            _get(7, 12.0),                                  # ceiling_db
            path)                                           # projected target file


def _load_projected_target(path, shape):
    """Load target_diag from an .npz and check it matches the specification."""
    if not path:
        return None, 'no projected target supplied; aiming at the raw specification'
    try:
        with np.load(path) as z:
            if 'target_diag' not in z:
                return None, f"{path}: no 'target_diag' array; aiming at the raw specification"
            t = np.asarray(z['target_diag'], dtype=float)
    except Exception as e:
        return None, f'{path}: could not be read ({type(e).__name__}); aiming at the raw specification'
    if t.shape != shape:
        return None, (f'{path}: target_diag is {t.shape}, specification diagonal is '
                      f'{shape}; aiming at the raw specification')
    return t, f'projected target loaded from {path}'


def _gain_schedule(transfer_function, resonance_sensitivity, n_lines):
    """Per-line gain scale, 1/(1 + s*(peakiness-1)), reusing
    match_trace_pseudoinverse_pi's self-scaling half-power peak detector."""
    if not (resonance_sensitivity > 0) or transfer_function is None:
        return np.ones(n_lines)
    line_gain = np.linalg.norm(transfer_function, axis=(1, 2))
    baseline = _half_power_resonance_baseline(line_gain)
    peakiness = line_gain/np.maximum(baseline, np.finfo(float).tiny)
    return 1.0/(1.0 + resonance_sensitivity*np.maximum(0.0, peakiness - 1.0))


class match_diagonal_congruence:
    """Per-channel diagonal-congruence feedback control.  See module docstring."""

    def __init__(self, specification, warning_levels, abort_levels, extra_parameters,
                 transfer_function=None, noise_response_cpsd=None,
                 noise_reference_cpsd=None, sysid_response_cpsd=None,
                 sysid_reference_cpsd=None, multiple_coherence=None, frames=None,
                 total_frames=None, last_response_cpsd=None, last_output_cpsd=None):
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        (self.rcond, self.max_drive_coherence, self.startup_test_level_cap_db,
         self.Ki, self.max_step_db, self.ridge, self.resonance_sensitivity,
         self.ceiling_db, self.target_path) = \
            parse_diagonal_congruence_parameters(extra_parameters)

        self.spec_diag = np.real(cpsd_autospectra(specification))
        self.unspecified = ~np.isfinite(self.spec_diag) | (self.spec_diag <= 0)
        target, msg = _load_projected_target(self.target_path, self.spec_diag.shape)
        self.target_diag = self.spec_diag.copy() if target is None else target.copy()
        # never aim above the specification itself, whatever the file says
        both = ~self.unspecified & np.isfinite(self.target_diag) & (self.target_diag > 0)
        self.target_diag[~both] = self.spec_diag[~both]
        self.n_cycles = 0
        print(f'[match_diagonal_congruence] rcond={self.rcond:g} '
              f'max_drive_coherence={self.max_drive_coherence:g} '
              f'Ki={self.Ki:g} max_step_db={self.max_step_db:g} ridge={self.ridge:g} '
              f'resonance_sensitivity={self.resonance_sensitivity:g} '
              f'ceiling_db={self.ceiling_db:g}; {msg}', flush=True)

    def system_id_update(self, transfer_function=None, noise_response_cpsd=None,
                         noise_reference_cpsd=None, sysid_response_cpsd=None,
                         sysid_reference_cpsd=None, multiple_coherence=None,
                         frames=None, total_frames=None):
        pass    # nothing needed from the system-ID phase

    def control(self, transfer_function=None, multiple_coherence=None, frames=None,
                total_frames=None, last_response_cpsd=None, last_output_cpsd=None):
        H = transfer_function
        if last_output_cpsd is None:
            return self._startup(H)

        self.n_cycles += 1
        X = last_output_cpsd
        F, N, _ = X.shape
        M = self.spec_diag.shape[1]

        achieved = np.real(cpsd_autospectra(last_response_cpsd))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.target_diag/achieved
        # a channel measuring <=0 or non-finite this cycle is not trusted: its
        # row is dropped from the solve rather than driven to zero.
        good = np.isfinite(ratio) & (ratio > 0) & ~self.unspecified
        eps = np.zeros_like(self.spec_diag)
        eps[good] = np.log(ratio[good])

        # sensitivity S_mi = 2 Re(H_mi (X H^H)_im) / y_m, from the MODEL
        C = X @ H.conjugate().transpose(0, 2, 1)                     # (F,N,M)
        y_pred = np.real(np.einsum('fmn,fnm->fm', H, C))             # (F,M)
        y_safe = np.where(y_pred > 0, y_pred, 1.0)
        S = 2.0*np.real(np.einsum('fmi,fim->fmi', H, C))/y_safe[:, :, None]
        S = np.where((y_pred > 0)[:, :, None] & good[:, :, None], S, 0.0)
        eps = np.where(good, eps, 0.0)

        # regularized least squares, per line: (S^T S + lam I) u = S^T eps
        StS = np.einsum('fmi,fmj->fij', S, S)                        # (F,N,N)
        Ste = np.einsum('fmi,fm->fi', S, eps)                        # (F,N)
        lam = self.ridge*np.maximum(np.einsum('fii->f', StS), _TINY)/N
        A = StS + lam[:, None, None]*np.eye(N)[None, :, :]
        try:
            u = np.linalg.solve(A, Ste[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            u = np.einsum('fij,fj->fi', np.linalg.pinv(A), Ste)
        u = np.where(np.isfinite(u), u, 0.0)

        # gain and per-line resonance scheduling
        u = u*self.Ki*_gain_schedule(H, self.resonance_sensitivity, F)[:, None]
        lines_ok = np.all(np.isfinite(u), axis=1) & ~np.all(self.unspecified, axis=1)
        u[~lines_ok] = 0.0

        # Step limit.  NOTE: clamping u itself does NOT bound the response
        # move.  Each row of S sums to 2, but individual entries can be large
        # and of opposite sign, so a bounded gain vector can still swing one
        # channel a long way.  The limit is therefore enforced on the actual
        # predicted per-channel response change, by scaling u back until it
        # complies.  Scaling is re-checked because the map u -> response is
        # exponential, not linear; a handful of passes converges tightly.
        if self.max_step_db > 0:
            max_step_nat = self.max_step_db*np.log(10.0)/10.0
            for _ in range(6):
                move = self._predicted_move(H, X, u, y_pred)
                worst = np.max(np.abs(move), axis=1)
                if np.all(worst <= max_step_nat*(1.0 + 1e-9)):
                    break
                s = np.where(worst > max_step_nat, max_step_nat/np.maximum(worst, _TINY), 1.0)
                u = u*s[:, None]

        D = np.exp(u)
        output = X*D[:, :, None]*D[:, None, :]                       # congruence
        move = self._predicted_move(H, X, u, y_pred)
        output = self._apply_ceiling(H, output, achieved, good, move)
        output = self._floor_and_silence(output)
        return _cap_drive_coherence(output, self.max_drive_coherence)

    # ------------------------------------------------------------------
    def _predicted_move(self, H, X, u, y_prev):
        """Exact per-channel log change in the predicted response produced by
        the congruence X -> D X D with D = diag(exp(u))."""
        D = np.exp(u)
        Xn = X*D[:, :, None]*D[:, None, :]
        Yn = np.einsum('fmn,fnk,flk->fml', H, Xn, H.conjugate())
        yn = np.real(np.einsum('fmm->fm', Yn))
        ok = (yn > 0) & (y_prev > 0)
        out = np.zeros_like(yn)
        out[ok] = np.log(yn[ok]/y_prev[ok])
        return out

    def _startup(self, H):
        tf_pinv = np.linalg.pinv(H, self.rcond)
        output = tf_pinv@self.specification@tf_pinv.conjugate().transpose(0, 2, 1)
        # Cap the PREDICTED RESPONSE level, not the drive trace.  The rest of
        # this family compares trace(drive) against trace(specification),
        # which mixes units -- drive CPSD against response CPSD -- so the
        # realised cap depends on the FRF's scale.  Since H is in hand here,
        # cap the quantity actually meant: predicted response trace relative
        # to the specification trace.
        spec_trace = np.real(trace(self.specification))
        Y = np.einsum('fmn,fnk,flk->fml', H, output, H.conjugate())
        resp_trace = np.real(trace(Y))
        max_power_ratio = 10.0**(self.startup_test_level_cap_db/10.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(1.0, max_power_ratio*spec_trace/resp_trace)
        scale[~np.isfinite(scale)] = 1.0
        scale[resp_trace <= 0] = 1.0
        output = output*scale[:, np.newaxis, np.newaxis]
        self.n_cycles = 0
        return _cap_drive_coherence(self._floor_and_silence(output),
                                    self.max_drive_coherence)

    def _apply_ceiling(self, H, output, achieved=None, good=None, move=None):
        """The response may not exceed the target by more than ceiling_db.
        Enforced as a per-line real scalar, which is itself a congruence and
        so cannot break positive semidefiniteness.

        Two estimates of the response are used and the more restrictive wins:

          measured-anchored -- achieved * exp(move).  The ABSOLUTE level comes
              from the measurement and is therefore model-free; only the small,
              already-bounded step is extrapolated with H.  Even a badly wrong
              FRF can only distort this by at most max_step_db.
          model-predicted   -- diag(H X H^H).  Used where this cycle's
              measurement cannot be trusted, and as a floor of last resort.

        Anchoring on the measurement matters when 'Update Transfer Function
        During Control' is on, or whenever H has drifted: a purely model-based
        ceiling is only as good as the model, which is precisely the thing that
        cannot be relied on in the regime this limit exists to protect against.
        """
        if not (self.ceiling_db > 0):
            return output
        lim = self.target_diag*10.0**(self.ceiling_db/10.0)
        valid = ~self.unspecified & (lim > 0)

        # The model net sits MODEL_MARGIN_DB looser than the measured one.  A
        # mildly wrong model must not hold the drive down (that would stop the
        # test ever reaching specification), but a model insisting the response
        # is orders of magnitude over must still be able to intervene -- that
        # is the protection against a measurement that has failed low, e.g. a
        # dead accelerometer, which would otherwise invite unbounded push.
        Y = np.einsum('fmn,fnk,flk->fml', H, output, H.conjugate())
        y_model = np.real(np.einsum('fmm->fm', Y))
        lim_model = lim*10.0**(MODEL_MARGIN_DB/10.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            over = np.where(valid, y_model/lim_model, 0.0)
        over = np.where(np.isfinite(over), over, 0.0)

        if achieved is not None and move is not None and good is not None:
            y_meas = achieved*np.exp(move)
            with np.errstate(divide='ignore', invalid='ignore'):
                over_m = np.where(valid & good, y_meas/lim, 0.0)
            over_m = np.where(np.isfinite(over_m), over_m, 0.0)
            # Where this cycle's measurement is trustworthy it REPLACES the
            # model estimate rather than being combined with it.  The measured
            # level is the actual level; the model's is a guess.  Taking the
            # stricter of the two looks conservative but is not: a model that
            # OVERestimates the response then holds the drive down forever and
            # the test never reaches specification -- a functional failure
            # dressed up as caution.  The model is used only where the
            # measurement cannot be trusted.
            over = np.maximum(over, np.where(valid & good, over_m, 0.0))

        worst = np.max(over, axis=1)
        scale = np.where(worst > 1.0, 1.0/np.maximum(worst, _TINY), 1.0)
        return output*scale[:, np.newaxis, np.newaxis]

    def _floor_and_silence(self, output):
        """Silence unspecified lines; keep specified lines off an unrecoverable
        zero using a floor relative to EACH line's own target (not the loudest
        line in the band -- see the floor fix in control_laws.py)."""
        all_unspec = np.all(self.unspecified, axis=1)
        output[all_unspec] = 0.0
        spec_trace = np.sum(np.where(self.unspecified, 0.0, self.spec_diag), axis=1)
        floor = 1e-8*spec_trace
        out_trace = np.real(trace(output))
        needs = (~all_unspec) & (out_trace < floor) & (floor > 0)
        if np.any(needs):
            scale = np.ones_like(out_trace)
            with np.errstate(divide='ignore', invalid='ignore'):
                scale[needs] = floor[needs]/np.maximum(out_trace[needs], _TINY)
            scale[~np.isfinite(scale)] = 1.0
            output = output*scale[:, np.newaxis, np.newaxis]
        return output
