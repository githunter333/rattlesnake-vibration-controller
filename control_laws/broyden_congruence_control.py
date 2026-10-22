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

extra_parameters: the nine match_diagonal_congruence fields, then
    broyden_gain        0.3   g above; 0 freezes S at the model seed
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
    return (_get(8, 0.3), _get(9, 1.0), _get(10, 0.05), _get(11, 0.05), _get(12, 0.0),
            _get(13, 1.0))


def _model_sensitivity(H, X):
    """S_mi = 2 Re(H_mi (X H^H)_im) / y_m and the model y_m."""
    C = X @ H.conjugate().transpose(0, 2, 1)
    y = np.real(np.einsum('fmn,fnm->fm', H, C))
    y_safe = np.where(y > 0, y, 1.0)
    S = 2.0*np.real(np.einsum('fmi,fim->fmi', H, C))/y_safe[:, :, None]
    return np.where((y > 0)[:, :, None], S, 0.0), y


class match_diagonal_congruence_broyden(match_diagonal_congruence):
    """Diagonal-congruence feedback with a Broyden-learned sensitivity."""

    def __init__(self, specification, warning_levels, abort_levels, extra_parameters,
                 *args, **kwargs):
        super().__init__(specification, warning_levels, abort_levels, extra_parameters,
                         *args, **kwargs)
        (self.broyden_gain, self.coherence_power, self.model_anchor,
         self.min_du_db, self.dither_db, split) = _parse_broyden_fields(extra_parameters)
        self.split_level = split != 0
        self._reset_learning()
        self._rng = np.random.default_rng(12345)
        print(f'[match_diagonal_congruence_broyden] broyden_gain={self.broyden_gain:g} '
              f'coherence_power={self.coherence_power:g} model_anchor={self.model_anchor:g} '
              f'min_du_db={self.min_du_db:g} dither_db={self.dither_db:g} '
              f'split_level={int(self.split_level)}', flush=True)

    def _reset_learning(self):
        self.S_hat = None
        self._cmd_last = None         # diag of the drive returned last call
        self._cmd_prev = None         # ... and the call before
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
        X = last_output_cpsd
        F, N, _ = X.shape

        # --- measured error, exactly as the parent ---
        achieved = np.real(cpsd_autospectra(last_response_cpsd))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.target_diag/achieved
        good = np.isfinite(ratio) & (ratio > 0) & ~self.unspecified
        eps = np.where(good, np.log(np.where(good, ratio, 1.0)), 0.0)

        S_model, y_model = _model_sensitivity(H, X)

        # --- learn S from (commanded drive change, measured response change) ---
        if self.S_hat is None:
            self.S_hat = S_model.copy()
        else:
            self._broyden_update(achieved, good, frames)
            if self.model_anchor > 0:
                self.S_hat = (1.0 - self.model_anchor)*self.S_hat + self.model_anchor*S_model
        self._y_prev = achieved.copy()
        self._good_prev = good.copy()
        S = np.where(good[:, :, None], self.S_hat, 0.0)
        self.last_row_sum = np.sum(self.S_hat, axis=2)

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
            r = np.sum(S, axis=2)                                    # (F,M)
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
        lam = self.ridge*np.maximum(scale, _TINY)/N
        A = StS + lam[:, None, None]*np.eye(N)[None]
        try:
            u = np.linalg.solve(A, Ste[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            u = np.einsum('fij,fj->fi', np.linalg.pinv(A), Ste)
        u = np.where(np.isfinite(u), u, 0.0) + c[:, None]
        u = u*self.Ki*_gain_schedule(H, self.resonance_sensitivity, F)[:, None]
        if self.dither_db > 0:
            u = u + self._rng.standard_normal(u.shape)*self.dither_db*_LN10_10/2.0
        lines_ok = ~np.all(self.unspecified, axis=1) & ~np.all(~good, axis=1)
        u[~lines_ok] = 0.0

        # --- step limit on BOTH predicted moves (learned and model) ---
        if self.max_step_db > 0:
            max_step = self.max_step_db*_LN10_10
            for _ in range(6):
                move_l = np.einsum('fmi,fi->fm', self.S_hat, u)
                move_m = self._predicted_move(H, X, u, y_model)
                worst = np.maximum(np.max(np.abs(move_l), axis=1),
                                   np.max(np.abs(move_m), axis=1))
                # a drive gain may also never move more than max_step on its own
                worst = np.maximum(worst, 2.0*np.max(np.abs(u), axis=1))
                if np.all(worst <= max_step*(1.0 + 1e-9)):
                    break
                s = np.where(worst > max_step, max_step/np.maximum(worst, _TINY), 1.0)
                u = u*s[:, None]

        D = np.exp(u)
        output = X*D[:, :, None]*D[:, None, :]
        # measured-anchored ceiling extrapolates with the LEARNED sensitivity
        move = np.einsum('fmi,fi->fm', self.S_hat, u)
        output = self._apply_ceiling(H, output, achieved, good, move)
        output = self._floor_and_silence(output)
        output = _cap_drive_coherence(output, self.max_drive_coherence)
        self._remember_command(output)
        return output

    # ------------------------------------------------------------------
    def _remember_command(self, output):
        self._cmd_prev = self._cmd_last
        self._cmd_last = np.real(np.einsum('fii->fi', output)).copy()

    def _row_weights(self, multiple_coherence, good):
        w = np.ones(good.shape)
        if self.coherence_power > 0 and multiple_coherence is not None:
            c = np.asarray(np.real(multiple_coherence), dtype=float)
            if c.shape == good.shape:
                w = np.clip(np.nan_to_num(c, nan=0.0), 0.0, 1.0)**self.coherence_power
        return np.where(good, w, 0.0)

    def _broyden_update(self, achieved, good, frames):
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
        # physical bounds: a drive cannot change a response by more than all of
        # it (|S| <= 2), and the uniform-scaling sensitivity (row sum) lies in
        # [0, 2] -- 2 for a noise-free linear channel, less where part of the
        # response does not come from the drives
        self.S_hat = np.clip(self.S_hat, -2.0, 2.0)
        rs = np.sum(self.S_hat, axis=2, keepdims=True)
        fix = np.clip(rs, 0.0, 2.0) - rs
        self.S_hat = self.S_hat + fix/self.S_hat.shape[2]
        self.n_broyden_updates += int(np.count_nonzero(lines))
