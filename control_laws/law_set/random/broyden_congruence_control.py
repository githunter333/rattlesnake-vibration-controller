"""
broyden_congruence_control.py

PROTOTYPE, 2026-10-22.  match_diagonal_congruence with the sensitivity matrix
LEARNED in the loop instead of computed from the FRF every cycle.

WHY
---
match_diagonal_congruence updates the drive by a diagonal congruence
X <- D X D, D = diag(exp(u)), and chooses u by one regularised Gauss-Newton
step on the log autospectrum error, using

    S_mi = d log y_m / d u_i = 2 Re( H_mi (X H^H)_im ) / y_m        (model)

H enters only as a direction, which already tolerates a lot of model error.
But it is still the SYSTEM-ID H, taken at system-ID level, without the
response noise in y_m.  Three things make that direction wrong exactly when
coherence is low:

  * response noise / uncorrelated inputs.  The true measured response is
    H X H^H + Gnn, and Gnn does not move with u, so the TRUE sensitivity is
    the model's scaled by the signal fraction (roughly the coherence).  The
    model over-predicts what a gain change will do.
  * nonlinearity.  The plant at test level is not the plant at system-ID
    level; the true local sensitivity at the operating point is what matters.
  * ill-conditioning.  H is poorly determined along weak drive directions,
    so S is too.

The measurement already contains the true local sensitivity.  Each cycle
the loop knows the drive change it commanded and sees the response change it
got.  So S is refined with a Broyden (secant) update:

    du   = 0.5 log( diag C_{k-1} / diag C_{k-2} )  COMMANDED drive change
    dy   = log( y_k / y_{k-1} )                     MEASURED response change
    S   += g * w * (dy - S du) du^T / (du^T du + delta)

du is taken from the drive this law RETURNED (after its own ceiling, floor
and coherence cap), not from the measured drive CPSD: the commanded change is
known exactly, whereas a measured one carries the same ~1/sqrt(frames)
estimation noise as the response, and the per-cycle gain changes are small
enough that a measured du is mostly noise (found in the first sweep).  This
assumes the hardware realises the commanded drive SHAPE, which the measured
drive CPSD can be used to check.

w is a signal-to-noise weight on each row.  A measured log autospectrum from
n frames has variance ~1/n, so a cycle-to-cycle difference has ~2/n.  The
expected secant signal on row m is (S du)_m.  w_m = r^2/(r^2 + 2/n), the
recursive-least-squares shrinkage: a secant whose expected signal is below
the measurement noise teaches S almost nothing.  Without it the first sweep
showed S being steered by estimation noise and the law doing WORSE than the
fixed-model version.

The FRF is used once, to seed S on the first feedback cycle, and after that
only through model_anchor and the inherited safety logic (startup solve,
model ceiling margin).

COHERENCE WEIGHTING
-------------------
NOT a noise argument: the variance of a measured autospectrum depends on the
frame count, not on coherence.  The argument is controllability.  A channel
with multiple coherence c has only a fraction ~c of its response coming
from the drives; the rest does not move when the drive changes.  Its error is
therefore only partly actionable, and pushing hard on it costs the other
channels.  Rows of the Gauss-Newton solve are weighted by c**p.  This is a
heuristic; coherence_power = 0 turns it off for comparison.

STEP SAFETY
-----------
The parent bounds the predicted response move with the MODEL.  Here the move
is predicted with the LEARNED S as well, and the larger of the two is held
to max_step_db.  The ridge also has an absolute floor taken from the model
sensitivity, so a learned S that collapses toward zero on a line (response
dominated by noise -- nothing the drive does moves it) yields a small step,
not a huge one.  This is the anti-windup for noise-floor lines.

WHAT THE COHERENCE SWEEP SHOWED (tests/coherence_sweep.py, 2026-10-22)
-------------------------------------------------------------------
Synthetic 8x6 plant, 40 cycles, 30-frame CPSD estimates; final rms dB error
of the true response against spec.  Defaults unless noted.

  case            parent  this law  hybrid*  optdiag rc=0 (model only)
  clean            2.75     2.67     0.54      0.12
  noise  g2=0.5    2.29     2.18     1.00      1.08
  noise  g2=0.3    2.31     2.30     1.28      1.41
  softened 5%      2.92     2.82     2.44      4.62
  FRF wrong x1     4.53     4.14     3.66      2.66

  * hybrid = first command from optimal_diagonal_control_fast (drive_rcond 0),
    then this law with max_drive_coherence 1.0.

  - Learning S is worth 0.1-0.5 dB over the parent, most where the FRF is
    wrong.  Modest.  Coherence weighting moved results by <= 0.1 dB.
  - Building on the COMMANDED drive (base_on_command) is worth more: 0.3-0.6
    dB here, and it stops match_trace_pseudoinverse's 1.7 -> 4.4 dB drift.
  - A diagonal congruence only rescales the drives, N knobs per line.  It
    corrects level and per-drive balance from measurement, but it cannot
    create drive SHAPE (inter-drive coherence and phase); that still comes
    from the model.  Hence the hybrid, and why it is the best or close to it
    whenever the plant or the noise differs from the model, while a model-only
    solve stays best when the model is good.
  - max_drive_coherence 0.95 destroys an optimal shape that needs
    near-coherent drives; the hybrid numbers need it lifted, at ~9 dB more
    drive relying on inter-drive cancellation.  The simulator realises the
    drive exactly, so it flatters that choice.  Not validated on hardware.

extra_parameters: the nine match_diagonal_congruence fields, then
    broyden_gain        0.6   g above; 0 freezes S at the model seed
                              (0.3 until 2026-10-23; the phase columns need
                              faster learning, and 0.6 is no worse for the
                              magnitude columns)
    coherence_power     1.0   p above; 0 disables coherence weighting
    model_anchor        0.05  fraction of S pulled back to the current model
                              each cycle (keeps S from drifting on lines the
                              loop never excites); 0 = pure learning
    min_du_db           0.05  drive changes smaller than this (dB, per line)
                              carry no usable secant information and are
                              skipped
    dither_db           0.0   optional zero-mean random gain perturbation
                              (dB) added each cycle for persistent excitation
    split_level         1     1 = take the uniform (level) part of the step
                              unregularised and ridge only the shape part;
                              0 = the parent's single ridged solve
    base_on_command     1     1 = apply the congruence to the drive this law
                              COMMANDED last cycle; 0 = to the measured drive
                              CPSD, as the parent does (see _drive_base)
    realize_tol_db      3.0   if the measured drive's in-band trace differs
                              from the commanded one by more than this (median
                              dB), the hardware is not realising the command
                              and the measured drive is used that cycle
    phase_steering      1     1 = also steer each drive's PHASE (see below);
                              0 = magnitudes only, as before 2026-10-23
    max_phase_step_deg  15    most any drive's phase may move in one cycle

PHASE STEERING (added 2026-10-23)
---------------------------------
With a real diagonal D, X <- D X D scales each cross term X_ij by d_i d_j > 0:
drive magnitudes change, but the relative PHASE between drives (and their
coherence) stays whatever the startup solve chose.  Yet each response
autospectrum depends on those phases, through interference:
y_m = sum_ij H_mi X_ij conj(H_mj).  With magnitudes only there are N real
knobs per line against M autospectrum targets (6 vs 8 on the frame the sweep
models), which is most of the 2-3 dB floor this family showed.

So D is made complex, d_i = exp(u_i + j theta_i), applied as X <- D X D^H.
That is still a congruence -- the drive stays a valid CPSD for ANY u, theta --
and it multiplies X_ij by d_i conj(d_j): magnitudes scale as before, the
relative phase of drives i and j rotates by theta_i - theta_j, and the
coherence between drives is unchanged.  2N-1 knobs per line (a common phase
does nothing).

The phase sensitivity is the imaginary-part twin of the magnitude one:

    d log y_m / d theta_i = -2 Im( H_mi (X H^H)_im ) / y_m

Each row of it sums to zero (a common phase rotation leaves every y_m alone);
that is enforced on the learned matrix.  The Broyden update learns the phase
columns exactly like the magnitude columns: the commanded phase change is
known (the ceiling, floor and coherence cap are real scalings and never move
a phase), and the response change is measured.

WHAT THE SWEEP SHOWED FOR PHASE (2026-10-23, broyden_gain 0.6, mean of
cycles 20-40, rms dB, 20-380 Hz):

                      30-frame CPSDs                 exact CPSDs, 100 cycles
  start / case      no phase   phase                 no phase   phase
  pinv  clean         2.58     2.56                    2.13*    1.84*
  pinv  noise 0.5     2.13     1.98
  pinv  soft 5%       2.63     2.79                    2.39*    2.21*
  pinv  FRF err 1     3.64     4.10
  hybrid clean        0.39     0.47                    1.00*    0.66*
  hybrid noise 0.5    0.80     0.92
  hybrid soft 5%      2.37     2.49
  hybrid FRF err 1    3.01     3.60
  (* broyden_gain 0.3, final value)

  - Structurally phase helps: with exact measurements it lowers the floor
    every time, still falling at 100 cycles.
  - With 30-frame estimates it helps only response-noise cases from a pinv
    start (~0.15 dB) and HURTS from the hybrid start and wherever the FRF is
    wrong (0.1-0.6 dB).  The phase columns are information-hungry: their
    true values change fast as the drive phases move, the model anchor pulls
    them toward the model's phase, which is exactly what is wrong when the
    FRF is, and each noisy secant spreads over 2N columns instead of N.
  - With model_anchor 0 the learned S drifts and the law fails (4-6 dB), so
    the anchor is load-bearing.

Risk: phase steering works through interference between coherent drives, so
it relies on the hardware reproducing relative drive phase.  Milder than
lifting the coherence cap -- inter-drive coherence is untouched -- but the
same kind of exposure.  max_phase_step_deg, the response step limit and the
measured-anchored ceiling bound it.
"""

import importlib.util as _ilu
import os as _os

import numpy as np

_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    '_rattlesnake_diagonal_congruence_base',
    _os.path.join(_here, 'diagonal_congruence_control.py'))
_base = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_base)

match_diagonal_congruence = _base.match_diagonal_congruence
_gain_schedule = _base._gain_schedule
_cap_drive_coherence = _base._cap_drive_coherence
cpsd_autospectra = _base.cpsd_autospectra

_TINY = 1e-300
_LN10_10 = np.log(10.0)/10.0       # dB -> natural log of power ratio


def _parse_broyden_fields(extra_parameters):
    """The numeric fields after the parent's eight numeric ones."""
    nums = []
    for p in (extra_parameters.split(',') if extra_parameters else []):
        p = p.strip()
        if p == '':
            nums.append(None)
            continue
        try:
            nums.append(float(p))
        except ValueError:
            pass     # the projected-target path; the parent reads it
    def _get(i, default):
        return nums[i] if len(nums) > i and nums[i] is not None else default
    return (_get(8, 0.6), _get(9, 1.0), _get(10, 0.05), _get(11, 0.05), _get(12, 0.0),
            _get(13, 1.0), _get(14, 1.0), _get(15, 3.0), _get(16, 1.0), _get(17, 15.0))


def _model_sensitivity(H, X, with_phase=False):
    """Model sensitivity of log y_m to the drive log-gains u_i and, with
    with_phase, to the drive phases theta_i, plus the model y_m.

        a_mi = H_mi (X H^H)_im           (sum_i a_mi = y_m)
        d log y_m / d u_i     =  2 Re(a_mi) / y_m
        d log y_m / d theta_i = -2 Im(a_mi) / y_m

    Returns S of shape (F, M, N), or (F, M, 2N) = [magnitude | phase]."""
    C = X @ H.conjugate().transpose(0, 2, 1)
    y = np.real(np.einsum('fmn,fnm->fm', H, C))
    y_safe = np.where(y > 0, y, 1.0)
    a = np.einsum('fmi,fim->fmi', H, C)/y_safe[:, :, None]
    S = 2.0*np.real(a)
    if with_phase:
        S = np.concatenate([S, -2.0*np.imag(a)], axis=2)
    return np.where((y > 0)[:, :, None], S, 0.0), y


class match_diagonal_congruence_broyden(match_diagonal_congruence):
    """Diagonal-congruence feedback with a Broyden-learned sensitivity."""

    def __init__(self, specification, warning_levels, abort_levels, extra_parameters,
                 *args, **kwargs):
        super().__init__(specification, warning_levels, abort_levels, extra_parameters,
                         *args, **kwargs)
        (self.broyden_gain, self.coherence_power, self.model_anchor,
         self.min_du_db, self.dither_db, split, base_cmd,
         self.realize_tol_db, phase, self.max_phase_step_deg) = \
            _parse_broyden_fields(extra_parameters)
        self.split_level = split != 0
        self.base_on_command = base_cmd != 0
        self.phase_steering = phase != 0
        self._reset_learning()
        self._rng = np.random.default_rng(12345)
        print(f'[match_diagonal_congruence_broyden] broyden_gain={self.broyden_gain:g} '
              f'coherence_power={self.coherence_power:g} model_anchor={self.model_anchor:g} '
              f'min_du_db={self.min_du_db:g} dither_db={self.dither_db:g} '
              f'split_level={int(self.split_level)} '
              f'base_on_command={int(self.base_on_command)} '
              f'realize_tol_db={self.realize_tol_db:g} '
              f'phase_steering={int(self.phase_steering)} '
              f'max_phase_step_deg={self.max_phase_step_deg:g}', flush=True)

    def _reset_learning(self):
        self.S_hat = None
        self._cmd_last = None         # diag of the drive returned last call
        self._cmd_prev = None         # ... and the call before
        self._cmd_full = None         # full drive CPSD returned last call
        self._theta_last = None       # drive phases applied in the last call
        self.n_measured_fallbacks = 0
        self._y_prev = None
        self._good_prev = None
        self.n_broyden_updates = 0
        self.last_row_sum = None      # diagnostic: ~2 x signal fraction per channel

    # Explicit overrides: Rattlesnake's dropdown only lists a class whose OWN
    # __dict__ defines both methods.
    def system_id_update(self, *args, **kwargs):
        return super().system_id_update(*args, **kwargs)

    def control(self, transfer_function=None, multiple_coherence=None, frames=None,
                total_frames=None, last_response_cpsd=None, last_output_cpsd=None):
        H = transfer_function
        self._ensure_target()
        if last_output_cpsd is None:
            self._reset_learning()
            out = self._startup(H)
            self._remember_command(out)
            return out

        self.n_cycles += 1
        X = self._drive_base(last_output_cpsd)
        F, N, _ = X.shape

        # --- measured error, exactly as the parent ---
        achieved = np.real(cpsd_autospectra(last_response_cpsd))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.target_diag/achieved
        good = np.isfinite(ratio) & (ratio > 0) & ~self.unspecified
        eps = np.where(good, np.log(np.where(good, ratio, 1.0)), 0.0)

        S_model, y_model = _model_sensitivity(H, X, with_phase=self.phase_steering)
        P = S_model.shape[2]                 # N, or 2N with phase steering

        # --- learn S from (commanded drive change, measured response change) ---
        if self.S_hat is None:
            self.S_hat = S_model.copy()
        else:
            self._broyden_update(achieved, good, frames, S_model)
            if self.model_anchor > 0:
                self.S_hat = (1.0 - self.model_anchor)*self.S_hat + self.model_anchor*S_model
        self._y_prev = achieved.copy()
        self._good_prev = good.copy()
        S = np.where(good[:, :, None], self.S_hat, 0.0)
        self.last_row_sum = np.sum(self.S_hat[:, :, :N], axis=2)

        # --- coherence-weighted, regularised Gauss-Newton step ---
        w = self._row_weights(multiple_coherence, good)
        Sw = S*w[:, :, None]
        StS = np.einsum('fmi,fmj->fij', Sw, S)
        # LEVEL / SHAPE SPLIT.  A uniform gain u = c*1 moves every channel by
        # its row sum r_m (2 for a clean linear channel), exactly match_trace's
        # update.  Solving the whole step through the ridge does NOT preserve
        # that direction -- 1 is not an eigenvector of S^T S + lam I -- so in
        # the parent law every level correction also distorts the drive shape,
        # and a run started from an optimal shape drifts away from it (seen in
        # the sweep: 0.12 dB optimal start -> ~2 dB).  So take the level part
        # unregularised, by weighted least squares on c alone, and put only
        # the residual through the ridge.
        c = np.zeros(F)
        if self.split_level:
            r = np.sum(S[:, :, :N], axis=2)      # magnitude columns only (F,M)
            num = np.sum(w*r*eps, axis=1)
            den = np.sum(w*r*r, axis=1)
            c = np.where(den > 0, num/np.maximum(den, _TINY), 0.0)
            eps = eps - r*c[:, None]
        Ste = np.einsum('fmi,fm->fi', Sw, eps)
        # ridge relative to the larger of learned and MODEL curvature: a learned
        # S that collapses toward zero must not turn into an enormous step
        Sm = np.where(good[:, :, None], S_model, 0.0)
        StS_model = np.einsum('fmi,fmj->fij', Sm*w[:, :, None], Sm)
        scale = np.maximum(np.einsum('fii->f', StS), np.einsum('fii->f', StS_model))
        lam = self.ridge*np.maximum(scale, _TINY)/P
        A = StS + lam[:, None, None]*np.eye(P)[None]
        try:
            z = np.linalg.solve(A, Ste[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            z = np.einsum('fij,fj->fi', np.linalg.pinv(A), Ste)
        z = np.where(np.isfinite(z), z, 0.0)
        z[:, :N] += c[:, None]
        if P > N:
            # a common phase does nothing; keep the phases zero-mean so the
            # step limits act on relative phase only
            z[:, N:] -= np.mean(z[:, N:], axis=1, keepdims=True)
        z = z*self.Ki*_gain_schedule(H, self.resonance_sensitivity, F)[:, None]
        if self.dither_db > 0:
            z[:, :N] += self._rng.standard_normal((F, N))*self.dither_db*_LN10_10/2.0
        lines_ok = ~np.all(self.unspecified, axis=1) & ~np.all(~good, axis=1)
        z[~lines_ok] = 0.0

        # --- step limit on BOTH predicted moves (learned and model) ---
        if self.max_step_db > 0:
            max_step = self.max_step_db*_LN10_10
            max_ph = np.deg2rad(self.max_phase_step_deg)
            for _ in range(6):
                move_l = np.einsum('fmi,fi->fm', self.S_hat, z)
                move_m = self._predicted_move_c(H, X, z, y_model, N)
                worst = np.maximum(np.max(np.abs(move_l), axis=1),
                                   np.max(np.abs(move_m), axis=1))
                # a drive gain may also never move more than max_step on its own
                worst = np.maximum(worst, 2.0*np.max(np.abs(z[:, :N]), axis=1))
                ratio = worst/max_step
                if P > N:
                    ratio = np.maximum(ratio, np.max(np.abs(z[:, N:]), axis=1)/max_ph)
                if np.all(ratio <= 1.0 + 1e-9):
                    break
                z = z*np.where(ratio > 1.0, 1.0/np.maximum(ratio, _TINY), 1.0)[:, None]

        D = self._gains(z, N)
        output = X*D[:, :, None]*D.conjugate()[:, None, :]          # D X D^H
        self._theta_applied = z[:, N:].copy() if P > N else None
        # measured-anchored ceiling extrapolates with the LEARNED sensitivity
        move = np.einsum('fmi,fi->fm', self.S_hat, z)
        output = self._apply_ceiling(H, output, achieved, good, move)
        output = self._floor_and_silence(output)
        output = _cap_drive_coherence(output, self.max_drive_coherence)
        self._remember_command(output)
        return output

    # ------------------------------------------------------------------
    def _drive_base(self, measured):
        """The drive CPSD the next congruence is applied to.

        The parent scales the MEASURED drive CPSD.  That estimate carries
        ~1/sqrt(frames) noise, and multiplying it in every cycle turns the
        noise into a random walk with no restoring force: in the coherence
        sweep match_trace_pseudoinverse drifted from 1.7 to 4.4 dB over 40
        cycles that way, and holds at 1.6 dB when given its own command.  So
        build on the drive this law COMMANDED, which is known exactly --
        unless the measurement says the hardware is not realising it (median
        in-band trace mismatch above realize_tol_db), in which case the
        measured drive is the truth and is used instead."""
        cmd = self._cmd_full
        if (not self.base_on_command or cmd is None or measured is None
                or cmd.shape != measured.shape):
            return measured
        band = ~np.all(self.unspecified, axis=1)
        tm = np.real(np.einsum('fii->f', measured))[band]
        tc = np.real(np.einsum('fii->f', cmd))[band]
        ok = (tm > 0) & (tc > 0)
        if not np.any(ok):
            return measured
        mismatch_db = np.median(np.abs(10*np.log10(tm[ok]/tc[ok])))
        if mismatch_db > self.realize_tol_db:
            self.n_measured_fallbacks += 1
            if self.n_measured_fallbacks <= 3 or self.n_measured_fallbacks % 25 == 0:
                print(f'[match_diagonal_congruence_broyden] measured drive differs from the '
                      f'commanded drive by {mismatch_db:.1f} dB (median, in band) -- '
                      f'building on the MEASURED drive this cycle', flush=True)
            return measured
        return cmd

    @staticmethod
    def _gains(z, N):
        """Complex diagonal gains exp(u + j theta) (real if no phase part)."""
        if z.shape[1] > N:
            return np.exp(z[:, :N] + 1j*z[:, N:])
        return np.exp(z)

    def _predicted_move_c(self, H, X, z, y_prev, N):
        """Exact model change in log y_m for X -> D X D^H, D = gains(z)."""
        D = self._gains(z, N)
        Xn = X*D[:, :, None]*D.conjugate()[:, None, :]
        yn = np.real(np.einsum('fmn,fnk,fmk->fm', H, Xn, H.conjugate()))
        ok = (yn > 0) & (y_prev > 0)
        out = np.zeros_like(yn)
        out[ok] = np.log(yn[ok]/y_prev[ok])
        return out

    def _remember_command(self, output):
        self._theta_last = getattr(self, '_theta_applied', None)
        self._theta_applied = None
        self._cmd_full = output.copy()
        self._cmd_prev = self._cmd_last
        self._cmd_last = np.real(np.einsum('fii->fi', output)).copy()

    def _row_weights(self, multiple_coherence, good):
        w = np.ones(good.shape)
        if self.coherence_power > 0 and multiple_coherence is not None:
            c = np.asarray(np.real(multiple_coherence), dtype=float)
            if c.shape == good.shape:
                w = np.clip(np.nan_to_num(c, nan=0.0), 0.0, 1.0)**self.coherence_power
        return np.where(good, w, 0.0)

    def _broyden_update(self, achieved, good, frames, S_model):
        """Secant update.  The response measured now (y_k) was produced by the
        drive returned LAST call (_cmd_last); the one before (y_{k-1}) by the
        drive returned the call before that (_cmd_prev)."""
        if (self.broyden_gain <= 0 or self._cmd_prev is None
                or self._y_prev is None or self._good_prev is None):
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            du = 0.5*np.log(self._cmd_last/self._cmd_prev)       # (F,N)
            dy = np.log(achieved/self._y_prev)                   # (F,M)
        du = np.where(np.isfinite(du), du, 0.0)
        N = du.shape[1]
        P = self.S_hat.shape[2]
        if P > N:
            # commanded phase change: exactly the phases applied last call (the
            # ceiling, floor and coherence cap are real scalings)
            dth = self._theta_last if self._theta_last is not None else np.zeros_like(du)
            du = np.concatenate([du, dth], axis=1)                  # (F, 2N)
        rows_ok = good & self._good_prev & np.isfinite(dy)
        dy = np.where(rows_ok, dy, 0.0)
        du2 = np.sum(du**2, axis=1)                              # (F,)
        min_du = self.min_du_db*_LN10_10/2.0
        lines = du2 > (min_du**2)
        if not np.any(lines):
            return
        pred = np.einsum('fmi,fi->fm', self.S_hat, du)           # expected (S du)
        innov = np.where(rows_ok, dy - pred, 0.0)
        # signal-to-noise shrinkage: var(dy) ~ 2/frames for log autospectra
        n = float(frames) if frames is not None and np.isfinite(frames) and frames > 0 else 1.0
        noise_var = 2.0/n
        w = pred**2/(pred**2 + noise_var)
        innov = innov*w
        delta = 0.1*min_du**2
        step = (self.broyden_gain*innov[:, :, None]*du[:, None, :]
                /(du2 + delta)[:, None, None])
        step[~lines] = 0.0
        self.S_hat = self.S_hat + step
        # Physical bounds.
        #  * Entry size: individual sensitivities CAN exceed 2 when drives
        #    interfere (a_mi can be larger than y_m), so the bound is relative
        #    to the model's own entry, not a flat +-2 (the first version
        #    clipped at +-2, which is wrong for coherent drives).
        bound = np.maximum(2.0, 2.0*np.abs(S_model))
        self.S_hat = np.clip(self.S_hat, -bound, bound)
        #  * Magnitude row sum = sensitivity to a uniform gain: in [0, 2], 2 for
        #    a noise-free linear channel, less where part of the response does
        #    not come from the drives.
        Sm = self.S_hat[:, :, :N]
        rs = np.sum(Sm, axis=2, keepdims=True)
        Sm += (np.clip(rs, 0.0, 2.0) - rs)/N
        #  * Phase row sum = sensitivity to a common phase: exactly 0.
        if P > N:
            Sp = self.S_hat[:, :, N:]
            Sp -= np.mean(Sp, axis=2, keepdims=True)
        self.n_broyden_updates += int(np.count_nonzero(lines))
