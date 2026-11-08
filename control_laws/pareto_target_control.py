# -*- coding: utf-8 -*-
"""Pareto-optimal target definition for MIMO random vibration.

Coletti, Davis & Schultz, "Pareto-optimal target definition for multi-axis
random vibration testing", Mechanical Systems and Signal Processing 249 (2026)
art. 114075, DOI 10.1016/j.ymssp.2026.114075 (SAND2026-18747J).  Equation and
section numbers below are the paper's.

THE IDEA
--------
Only the DIAGONAL of the response CPSD is specified.  The drive-minimizing
family -- Smallwood's extreme inputs/outputs method, Musella's minimum drives
method, and the independent drives (buzz) method -- answers the freedom in the
cross terms by holding the diagonal fixed and choosing off-diagonals that
minimize drive energy.  This paper's central observation is that the
off-diagonal terms strongly affect the control RESIDUALS as well, so holding
the diagonal fixed does not hold the achieved response fixed, and optimizing
energy alone trades away accuracy without accounting for it.

So both objectives are optimized at once.  For a tradeoff parameter lambda,

    min_X  sum_i |spec_ii - (H X H^H)_ii|  +  lambda * Tr(X),   X >= 0   (2)

Sweeping lambda traces the Pareto curve: no point on it is beaten in BOTH
control error and input energy by any other feasible solution.  The positive
semidefinite constraint is applied DURING the optimization rather than
retroactively, which is the first of the paper's three criticisms of the
existing methods.

The solution X is the open-loop drive CPSD -- for open-loop control the target
CPSD never has to be specified at all.  For closed-loop control a target is
derived FROM that drive (Section 2.2) such that regularized pseudoinverse
control reproduces it:

    min_Y  sum_i |Y_ii - spec_ii|   s.t.  H+(l) Y H+(l)^H = X,  Y >= 0   (5)

    H+(l) = sum_i v_i sigma_i/(sigma_i^2 + l) u_i^H                      (6)

Note that the SAME lambda is the Pareto tradeoff parameter in (2) and the
Tikhonov parameter in (6).  That is the paper's second criticism made
constructive: the older derivations assume an UNREGULARIZED pseudoinverse, so
their optimality does not survive the regularization every real controller
applies.  Here the regularization is inside the constraint.

HOW THIS RELATES TO THE LAWS ALREADY HERE
-----------------------------------------
independent_drive_control.py is this paper's IDM baseline.  Its Section 3.2.1
is our buzz construction exactly -- cross terms taken from the correlation and
phase of H H^H, the response to uncorrelated unit drives, their Eq. (10).  So
the two modules are the same problem answered two ways, and
check_pareto_target.py scores them against each other on the same plants.

The paper's third criticism lands on a defect this folder has already measured
independently.  independent_drive_control's docstring records that rank
truncation breaks Smallwood's step 3.4, because the congruence guarantees the
target diagonal only if H H^+ = I; that law fixes it by setting the level
explicitly after the solve.  Coletti et al. fix the same thing at the root, by
putting H+(lambda) inside the constraint that defines the target.  Theirs is
the better answer; ours is the one available to a law that must not re-solve.

A feasibility note, because the obvious reading of the paper here is wrong and
this module measured it.  The paper says exact control is possible if
n_resp <= n_drive^2, a count of the real parameters in a Hermitian X.  That is
NECESSARY and not sufficient.  The binding constraint is that H X H^H must lie
in the range of H, and on a rank-deficient plant that range is small.

Measured on the frame model, flat specification, 181 in-band lines:

    4 responses on 6 drives (rank 4)    feasible nearly everywhere
    5 responses on 6 drives (rank 5)    feasible nearly everywhere
    6 responses on 6 drives (rank 5)    feasible nearly everywhere
    8 responses on 6 drives (rank 5)    0 of 181, by a wide margin -- the
                                        smallest relative residual on any line
                                        is 2.8e-3 and the median is 1.15e-1

The 6x12 rig is the last of those, and 8 <= 36 holds, so the parameter count
would have predicted feasibility.  It is not feasible.  The practical rule is
n_resp <= rank(H), not n_resp <= n_drive^2.

This matters for reading independent_drive_control, which found 0 of 181 on the
same rig and 0 of 291 on the RC plate with the drive restricted to its
diagonal.  On the rig, that result is NOT an artifact of the diagonal
restriction: a full Hermitian drive cannot do it either.  `exact_feasible`
below answers the question properly for any plant and specification.

WHAT IS FAITHFUL AND WHAT IS NOT
--------------------------------
Faithful: Eqs. (2), (3), (4), (5), (6), the three target cases of Section 2.2,
Eq. (9)'s underdetermined energy refinement, and the automatic lambda selection
of Section 3.1 including its 10 % exclusions.

Solved with cvxpy rather than the paper's Appendix B barrier method, which is
an implementation detail the paper itself allows ("commercial solvers [23, 24]
or ... the Barrier method").  CLARABEL measures about 11 ms per line on a 6
drive problem, inside the paper's quoted few hundredths of a second.  Interior
point methods return eigenvalues a few parts in 1e8 negative, so every returned
CPSD is projected onto the PSD cone before it leaves this module.

NOT faithful, and off by default: `weighted`.  The paper's objective is a sum of
ABSOLUTE residuals, and Section 4.3 is candid about what that costs -- in bins
with very low specifications the Pareto solution can drive the response near
zero, which barely moves the absolute residual and is unbounded in dB.  Their
Fig. 8c shows it.  This study scores in dB (mean |dB| over bands), so the
paper's own objective is the wrong metric for us in exactly the bins we care
about.  `weighted` divides each residual by spec_ii, which stays convex and is
the relative-error surrogate the paper names as future work ("future work
should develop a convex approximation of the decibel error").  It is off by
default so the default reproduces the paper.

VERIFIED
--------
Loaded the way Rattlesnake loads a control law (spec_from_file_location, no
package context) and exercised on 2026-11-08 by check_pareto_target.py.  The
frame model at 5 Hz, flat specification, 181 in-band lines over 100-1000 Hz;
the RC plate at 48 accelerometers against the BARC specification.

  decoupled 6 x 6 plant       exact control feasible, taken by the exact
                              branch, 4.0e-16 dB error and every off-diagonal
                              drive term exactly 0
  drive CPSD                  PSD to -1.1e-16 relative eigenvalue
  -9 dB startup ceiling       predicted trace re spec exactly -9.00 dB
  generator form              identical output to the function form
  Eq. (9) refinement          input energy 5.70999 -> 5.70984 with the output
                              diagonal unchanged to 1.8e-15
  even-determined target      Z Y Z^H reproduces X to 1.1e-5

THREE RESULTS THAT SHOULD CHANGE HOW THIS IS USED.

1. At lambda = 0 this method beats the independent drives method on the rig in
   the metric this study scores: mean |dB| 1.990 against
   independent_drive_control's 2.937.  That is the result worth having.  The
   Pareto comparison against the paper's own IDM baseline is more modest -- 7
   of 14 grid points strictly dominated, and at the two selected points the
   Pareto solution has the lower residual (0.873 against 0.997) with slightly
   MORE input energy (0.0224 against 0.0204), so it does not dominate there.
   That is consistent with the paper's own experimental section, which claims
   only "moderate improvements in control accuracy over the IDM".

2. Section 3.1's automatic lambda selection is unusable for this study, and
   measurably so.  On the rig it picks lambda = 10, which costs 15.940 mean
   |dB| where lambda = 0 costs 1.990 -- a 14 dB penalty -- and drives 290 of
   1448 in-band bins more than 30 dB below their specification, where
   lambda = 0 drives 12.

   The cause is not a bug, it is a scale mismatch in the criterion.
   log(residuals) + log(inputs) weights two axes equally, and on this rig the
   input energy spans a factor of 4e11 across the grid while the residual spans
   6.8x.  So the criterion is effectively minimizing input energy alone.  This
   is Section 4.3's own caveat arriving through the selection rule rather than
   through the objective.

   Two ways out are provided, both departures: `weighted=True`, which recovers
   almost all of it (2.245 against 15.940 at the SAME lambda = 10), and
   `select='db'` on pareto_curve, which scores the grid in mean |dB| and
   abandons the energy objective.  The law's own default is lambda = 0.0, which
   is the measured best point in dB on this plant -- chosen on that
   measurement, not from the paper.

3. For an OVERDETERMINED closed-loop target, use mode='optimal'.  The cheap
   Z+ X Z+^H of Section 2.2 fails at both ends of the curve for different
   reasons.  At lambda = 0 it is exactly the predicted response H X H^H, whose
   diagonal is fine (1.990 mean |dB|) but which does not satisfy Eq. (5)'s
   constraint on a rank-deficient plant -- Z Y Z^H came back 9.5e-2 off X,
   because Z+ Z is the projector onto the row space of H and H here is rank 5
   of 6.  At lambda = 10 it satisfies the constraint to 1.0e-3 and the target
   itself is numerically absurd, 175 mean |dB| off the specification, because
   Z+ inverts a factor sigma/(sigma^2 + lambda) that is tiny in the weak
   directions.  Eq. (5) solved properly gives both at once: constraint 6.1e-5
   and 4.386 mean |dB|.  It costs about 0.37 s per line at 8 responses and was
   not attempted at 48.

EXACT FEASIBILITY, MEASURED
---------------------------
  frame, 4 resp x 6 drives, rank 4    159 / 181 feasible
  frame, 5 resp x 6 drives, rank 5    157 / 181
  frame, 6 resp x 6 drives, rank 5    127 / 181
  frame, 8 resp x 6 drives, rank 5      0 / 181   (the 6x12 rig)
  RC plate, 48 x 12, rank 12            0 /  49   (every 6th line)

The first three rows are feasible wherever the solver can show it; their
relative residuals sit at 5e-8 to 1e-7, which is interior point accuracy, and
the lines counted infeasible are the handful with a genuinely large residual.
The last two rows are infeasible by a wide margin -- the rig's SMALLEST
relative residual on any line is 2.8e-3 and the plate's is 1.5e-4.

Caveat on the plate row: 42 of its 49 solves returned 'optimal_inaccurate'.
The conclusion survives, since the smallest residual is still 15x the
tolerance, but the plate is at the edge of what CLARABEL and SCS do well at
this size and its numbers should not be quoted more precisely than that.

INTENDED WORKFLOW
-----------------
This is a PRE-TEST target definition, not a real-time law.  Solving the whole
lambda grid costs tens of seconds for a few hundred lines, which the paper says
plainly is "often acceptable for control predictions, but not for real-time
closed-loop control".  So:

    1. offline, once:  pareto_curve(spec, H)  ->  lambda_hat and the curve
    2. put that lambda in the law's parameter string, or hand the derived
       target to any existing law in this folder

`pareto_optimal_control` below is the open-loop law for step 2, solving Eq. (2)
at a FIXED lambda.  It does not search the grid, because a control law may not
stall its first command for a minute.

PARAMETERS
----------
Comma separated, all optional, defaulted incrementally like every other law
here:

    lam, max_drive_coherence, startup_test_level_cap_db, weighted, set_trace

    lam                         the Pareto tradeoff parameter.  0.0 (default)
                                is maximum accuracy, the left end of the curve,
                                and is the measured best point in dB on the
                                rig; larger prefers lower drive energy.  Read
                                it off pareto_curve, do not guess it -- its
                                units are response^2/input^2 and so are plant
                                specific -- and read VERIFIED item 2 before
                                taking Section 3.1's automatic choice.
    max_drive_coherence         cap on |X_ij|/sqrt(X_ii X_jj).  Default 1.0, no
                                cap.
    startup_test_level_cap_db   ceiling on the predicted response trace
                                relative to the specification's.  Default -9.0,
                                the same guard every law in this module uses on
                                an unfed-back first command.
    weighted                    0 (default) is the paper's absolute residual;
                                1 weights each residual by 1/spec_ii.
    set_trace                   0 (default) leaves the solved level alone.  1
                                forces the predicted trace to the
                                specification's, which DEFEATS the tradeoff --
                                undershooting to save energy is the whole point
                                of lambda > 0 -- and is offered only for
                                comparison against the match-trace family.

FEEDBACK
--------
THIS LAW IS OPEN LOOP.  last_response_cpsd is accepted and never read.  The
paper's closed-loop path is to derive a target with `pareto_target` and run an
existing closed-loop law against it; that is target definition, not a law, and
it lives in this module as a function rather than as a control law.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
"""
import importlib.util
import os

import numpy as np

try:
    import cvxpy as cp
except Exception:                                   # cvxpy is optional here
    cp = None


def _load_sibling(name):
    """Load a module sitting next to this file BY PATH.

    Rattlesnake loads a control law with importlib.spec_from_file_location and
    no package context, so a normal import of a sibling fails.  Every law here
    that needs another one does this; see optimal_diagonal_control.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(name)[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cl = _load_sibling('control_laws.py')
_cap_drive_coherence = _cl._cap_drive_coherence
_apply_startup_level_cap_from_trace = _cl._apply_startup_level_cap_from_trace
_set_predicted_response_trace = _cl._set_predicted_response_trace

# The paper's sparse grid, Section 3.1: {0, 1e-6, 1e-5, ..., 1e6}.
DEFAULT_LAMBDAS = (0.0,) + tuple(10.0**np.arange(-6.0, 6.5, 1.0))

# Interior point solvers return the PSD constraint satisfied to a tolerance,
# not exactly.  Eigenvalues this far negative relative to the largest are
# treated as solver noise and clipped; anything worse is reported.
_PSD_SLACK = 1e-6

# Relative residual below which a line counts as EXACTLY controlled:
# sum_i |spec_ii - achieved_ii| / sum_i spec_ii.
#
# Feasibility is decided on this and NOT on the status or slack of Eq. (B.3)'s
# solve.  Measured reason: on the 6x12 rig, Eq. (B.3) returned 'infeasible' on
# 162 of 181 lines and 'optimal_inaccurate' on 18, and those 18 carried slacks
# from -7.6e3 against a median of -1.2e1 on the rest -- the sign of a slack
# variable whose scale is arbitrary disagreed with the strict infeasibility of
# its neighbours.  independent_drive_control carries the same lesson from the
# other direction: a sign test is not a specification test.
#
# The VALUE is calibrated on the measured separation, on the frame model at
# 181 in-band lines, between what the solver can reach and what the plant
# cannot:
#
#   4, 5, 6 responses on 6 drives   relative residual median 5e-8 to 1e-7,
#                                   max 1.9e-1 -- feasible nearly everywhere,
#                                   at interior point accuracy
#   8 responses on 6 drives         median 1.15e-1, MINIMUM 2.8e-3 -- not one
#                                   line is close to exact
#
# So anything from about 1e-6 to 1e-4 separates the two cleanly, and 1e-5 sits
# in the middle of that.  Note this is four orders looser than
# independent_drive_control's 1e-10, and for a reason: that module's solve is
# np.linalg.lstsq, which is direct and lands at 1e-16, while every solve here
# is interior point and lands at 1e-7.
_EXACT_RTOL = 1e-5


def _solver():
    """CLARABEL where available -- measured 5x faster than SCS on these -- then
    SCS, then whatever cvxpy picks."""
    if cp is None:
        return None
    installed = set(cp.installed_solvers())
    for name in ('CLARABEL', 'SCS'):
        if name in installed:
            return name
    return None


def _solver_ladder(preferred):
    """The solvers to try in order.

    CLARABEL is the fastest of these by about 5x on a 6-drive line, and it
    fails outright on the RC plate's 48-response feasibility problem -- so a
    ladder is not belt and braces, it is required to cover the study's own
    plants.  SCS is slower and more tolerant.
    """
    if cp is None:
        return []
    installed = set(cp.installed_solvers())
    order = [preferred] if preferred else []
    order += [name for name in ('CLARABEL', 'SCS') if name in installed]
    seen, ladder = set(), []
    for name in order:
        if name and name not in seen:
            seen.add(name)
            ladder.append(name)
    return ladder or [None]


def _solve(problem, solver):
    """Solve with the ladder, returning the status of whichever worked.

    A solver that raises is a failure like any other, so it is caught and the
    next one tried; the status reported is the LAST one attempted, so a caller
    that checks it sees a real failure rather than a stale success.
    """
    status = 'solver_error'
    for name in _solver_ladder(solver):
        try:
            problem.solve(solver=name)
        except Exception:
            continue
        status = problem.status
        if status in ('optimal', 'optimal_inaccurate'):
            return status
    return status


def _require_cvxpy():
    if cp is None:
        raise RuntimeError(
            'pareto_target_control needs cvxpy.  Every model in Coletti et al. '
            'is a semidefinite program; there is no closed form to fall back '
            'on.  Install cvxpy, or use independent_drive_control, which is '
            'this paper\'s IDM baseline and needs only numpy.')


def _spec_diagonal(specification):
    """The specified response auto-spectra, real and non-negative."""
    d = np.real(np.einsum('fmm->fm', np.asarray(specification)))
    return np.maximum(d, 0.0)


def _project_psd(X):
    """Clip negative eigenvalues, preserving the Hermitian part.

    Returns the projected matrix and the most negative eigenvalue it found,
    relative to the largest in magnitude, so a caller can tell solver noise
    from a solve that actually failed.
    """
    X = 0.5*(X + X.conjugate().T)
    w, V = np.linalg.eigh(X)
    scale = max(float(np.abs(w).max()), 1e-300)
    worst = float(w.min())/scale
    if w.min() < 0.0:
        X = (V*np.maximum(w, 0.0)) @ V.conjugate().T
        X = 0.5*(X + X.conjugate().T)
    return X, worst


def _regularized_pinv(H, lam):
    """Eq. (6): sum over i of v_i sigma_i/(sigma_i^2 + lam) u_i^H.

    Batched over frequency.  lam = 0 is the ordinary pseudoinverse.  The sum
    runs to rank(H), so directions numerically absent from H are dropped
    rather than being handed to the regularizer.
    """
    H = np.asarray(H)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    cutoff = s[:, :1]*np.finfo(float).eps*max(H.shape[-2:])
    filt = np.where(s > cutoff, s/(s**2 + lam), 0.0)
    return (Vh.conjugate().transpose(0, 2, 1)*filt[:, None, :]) \
        @ U.conjugate().transpose(0, 2, 1)


def _relative_residual(H, y, X):
    """sum_i |y_i - (H X H^H)_ii| / sum_i y_i, at one line."""
    scale = float(np.sum(y))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.inf
    achieved = np.real(np.diag(H @ X @ H.conjugate().T))
    return float(np.sum(np.abs(y - achieved)))/scale


def _pareto_line(H, y, lam, weighted, solver):
    """Eq. (2) at one line, as the slack form of Eq. (B.2)."""
    n = H.shape[1]
    X = cp.Variable((n, n), hermitian=True)
    achieved = cp.real(cp.diag(H @ X @ H.conjugate().T))
    residual = cp.abs(y - achieved)
    if weighted:
        w = np.zeros_like(y)
        np.divide(1.0, y, out=w, where=y > 0)
        residual = cp.multiply(w, residual)
    problem = cp.Problem(cp.Minimize(cp.sum(residual)
                                     + lam*cp.real(cp.trace(X))), [X >> 0])
    status = _solve(problem, solver)
    if X.value is None:
        return np.zeros((n, n), complex), status
    return np.asarray(X.value), status


def _exact_feasible_line(H, y, solver):
    """Eq. (3) as Eq. (B.3): min t s.t. diag(H X H^H) = y, X + tI >= 0.

    Exact control is feasible when the minimum is non-positive.  Returns that
    minimum and the status, not an X -- the X worth having in the feasible case
    is Eq. (4)'s minimum-trace one.
    """
    n = H.shape[1]
    X = cp.Variable((n, n), hermitian=True)
    t = cp.Variable()
    constraints = [cp.real(cp.diag(H @ X @ H.conjugate().T)) == y,
                   X + t*np.eye(n) >> 0]
    problem = cp.Problem(cp.Minimize(t), constraints)
    status = _solve(problem, solver)
    if status not in ('optimal', 'optimal_inaccurate'):
        # 'infeasible' means the equality constraints alone cannot be met, so
        # exact control is impossible; any other non-optimal status means the
        # question was not answered.  Both return +inf, and the status
        # distinguishes them for the caller.
        return np.inf, status
    return float(problem.value), status


def _min_trace_exact_line(H, y, solver):
    """Eq. (4): min Tr(X) s.t. diag(H X H^H) = y, X >= 0.

    Used only where Eq. (3) said exact control is feasible, which is the
    lambda = 0 branch of Algorithm 1.
    """
    n = H.shape[1]
    X = cp.Variable((n, n), hermitian=True)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(X))),
                         [cp.real(cp.diag(H @ X @ H.conjugate().T)) == y,
                          X >> 0])
    status = _solve(problem, solver)
    if X.value is None:
        return None, status
    return np.asarray(X.value), status


def _target_line(H, y, X_hat, lam, solver):
    """Eq. (5) as Eq. (B.4), the overdetermined target.

    Starts from the feasible Z+ X Z+^H of Section 2.2 and improves the diagonal
    agreement with the specification subject to reproducing X_hat exactly.
    """
    n_resp = H.shape[0]
    Z = _regularized_pinv(H[None], lam)[0]
    Y = cp.Variable((n_resp, n_resp), hermitian=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.abs(cp.real(cp.diag(Y)) - y))),
        [Z @ Y @ Z.conjugate().T == X_hat, Y >> 0])
    status = _solve(problem, solver)
    if Y.value is None:
        return None, status
    return np.asarray(Y.value), status


def exact_feasible(specification, transfer_function, lines=None, solver=None,
                   with_margin=False):
    """Can the specified diagonal be met EXACTLY, Eq. (3)?

    This is the real feasibility question, with a full Hermitian drive CPSD --
    not independent_drive_control's sign test, which restricts the drive to its
    diagonal and so answers a much more constrained question.

    Decided by solving Eq. (2) at lambda = 0 and testing the achieved diagonal
    against the specification, NOT by the status or slack of Eq. (B.3); see
    _EXACT_RTOL for the measurement behind that choice.  `with_margin` adds
    Eq. (B.3)'s slack as `margin` for comparison, at the cost of a second solve
    per line.

    Returns `feasible` and `residual` per line, `n_feasible` out of
    `n_tested`, and the solver statuses actually seen.
    """
    _require_cvxpy()
    solver = solver or _solver()
    H = np.asarray(transfer_function)
    y = _spec_diagonal(specification)
    in_band = y.max(axis=1) > 0
    todo = np.nonzero(in_band)[0] if lines is None else np.asarray(lines)

    residual = np.full(H.shape[0], np.nan)
    margin = np.full(H.shape[0], np.nan)
    statuses = {}
    for f in todo:
        X, status = _pareto_line(H[f], y[f], 0.0, False, solver)
        statuses[status] = statuses.get(status, 0) + 1
        if status in ('optimal', 'optimal_inaccurate'):
            residual[f] = _relative_residual(H[f], y[f], _project_psd(X)[0])
        if with_margin:
            margin[f], _ = _exact_feasible_line(H[f], y[f], solver)

    feasible = np.zeros(H.shape[0], dtype=bool)
    with np.errstate(invalid='ignore'):
        feasible[todo] = residual[todo] <= _EXACT_RTOL
    answered = int(np.isfinite(residual[todo]).sum())
    return {'feasible': feasible, 'residual': residual, 'margin': margin,
            'in_band': in_band, 'n_tested': int(len(todo)),
            'n_feasible': int(feasible[todo].sum()), 'statuses': statuses,
            'n_answered': answered}


def pareto_inputs(specification, transfer_function, lam=0.0, weighted=False,
                  exact_branch=True, lines=None, solver=None):
    """Eq. (2), or Eqs. (3)/(4) at lambda = 0: the optimal open-loop drive CPSD.

    `exact_branch` follows Algorithm 1: at lambda = 0, test exact feasibility
    first and use the minimum-trace solution where it holds.  Set it False to
    take Eq. (2) everywhere, which is also what the paper suggests as the
    cheap approximation to lambda = 0 (solve with a small positive lambda).

    `lines` restricts the solve to those frequency indices; everything else is
    returned zero.  Returns the drive CPSD and a diagnostics dict.
    """
    _require_cvxpy()
    solver = solver or _solver()
    H = np.asarray(transfer_function)
    y = _spec_diagonal(specification)
    n_f, n_resp, n_drive = H.shape
    in_band = y.max(axis=1) > 0
    todo = np.nonzero(in_band)[0] if lines is None else np.asarray(lines)

    X = np.zeros((n_f, n_drive, n_drive), complex)
    exact = np.zeros(n_f, dtype=bool)
    worst_eig = np.zeros(n_f)
    rel_residual = np.full(n_f, np.nan)
    failed = []
    use_exact = exact_branch and lam == 0.0
    for f in todo:
        solved, status = _pareto_line(H[f], y[f], lam, weighted, solver)
        if status not in ('optimal', 'optimal_inaccurate'):
            failed.append((int(f), status))
            X[f], worst_eig[f] = _project_psd(solved)
            continue
        solved = _project_psd(solved)[0]
        residual_here = _relative_residual(H[f], y[f], solved)
        if use_exact and residual_here <= _EXACT_RTOL:
            # Algorithm 1's lambda = 0 branch: among the exact solutions, take
            # the minimum-trace one.  Keep it only if it is still exact --
            # Eq. (4) is a different, equality-constrained solve and can come
            # back worse conditioned than the one it is replacing.
            better, status_exact = _min_trace_exact_line(H[f], y[f], solver)
            if better is not None:
                better = _project_psd(better)[0]
                if _relative_residual(H[f], y[f], better) <= _EXACT_RTOL:
                    solved, residual_here = better, _relative_residual(
                        H[f], y[f], better)
            exact[f] = True
        rel_residual[f] = residual_here
        X[f], worst_eig[f] = _project_psd(solved)

    achieved = np.real(np.einsum('fmm->fm',
                                 H @ X @ H.conjugate().transpose(0, 2, 1)))
    residual = np.abs(y - achieved)
    energy = np.real(np.einsum('fjj->f', X))
    return X, {'in_band': in_band, 'solved': todo, 'exact': exact,
               'n_exact': int(exact[todo].sum()), 'lam': lam,
               'relative_residual': rel_residual,
               'residual_per_line': residual.sum(axis=1),
               'input_energy_per_line': energy,
               'total_residual': float(residual[todo].sum()),
               'total_input_energy': float(energy[todo].sum()),
               'worst_relative_eigenvalue': float(worst_eig[todo].min())
               if len(todo) else 0.0,
               'failed_lines': failed}


def pareto_target(specification, transfer_function, inputs, lam=0.0,
                  mode='feasible', refine_underdetermined=True, lines=None,
                  solver=None):
    """Section 2.2: a target CPSD that reproduces `inputs` under H+(lam).

    The case is decided by the shapes, exactly as the paper does:

      n_resp <  n_drive   underdetermined.  Yhat = H X H^H, the predicted
                          response.  With `refine_underdetermined`, Eq. (9)'s
                          update X <- (H+ H) X (H+ H)^H is applied first: it
                          produces an identical output with no more input
                          energy, and corrects an X that is not perfectly
                          optimal.
      n_resp == n_drive   even-determined.  Yhat = Z+ X Z+^H is the unique
                          feasible solution of Eq. (5).
      n_resp >  n_drive   overdetermined.  Z+ X Z+^H is feasible but not
                          generally optimal.  mode='feasible' returns it;
                          mode='optimal' solves Eq. (5) per line from it.

    Returns the target and a dict saying which branch ran.
    """
    H = np.asarray(transfer_function)
    X = np.asarray(inputs)
    y = _spec_diagonal(specification)
    n_f, n_resp, n_drive = H.shape
    in_band = y.max(axis=1) > 0
    todo = np.nonzero(in_band)[0] if lines is None else np.asarray(lines)

    Z = _regularized_pinv(H, lam)
    info = {'case': None, 'mode': mode, 'refined': False, 'failed_lines': []}

    if n_resp < n_drive:
        info['case'] = 'underdetermined'
        if refine_underdetermined:
            P = _regularized_pinv(H, 0.0) @ H
            X = P @ X @ P.conjugate().transpose(0, 2, 1)
            info['refined'] = True
        Y = H @ X @ H.conjugate().transpose(0, 2, 1)
    else:
        info['case'] = ('even-determined' if n_resp == n_drive
                        else 'overdetermined')
        Z_pinv = np.linalg.pinv(Z)
        Y = Z_pinv @ X @ Z_pinv.conjugate().transpose(0, 2, 1)
        if n_resp > n_drive and mode == 'optimal':
            _require_cvxpy()
            solver = solver or _solver()
            for f in todo:
                better, status = _target_line(H[f], y[f], X[f], lam, solver)
                if better is None:
                    info['failed_lines'].append((int(f), status))
                else:
                    Y[f] = better

    worst = 0.0
    for f in todo:
        Y[f], w = _project_psd(Y[f])
        worst = min(worst, w)
    Y[~in_band] = 0.0
    info['worst_relative_eigenvalue'] = float(worst)
    info['inputs'] = X
    return Y, info


def _mean_abs_db(H, X, y, mask):
    """Mean |dB| of the achieved diagonal against the specification.

    Not an objective anywhere in the paper -- it is the metric this study
    scores in, carried alongside so the paper's own selection criterion can be
    judged against it.
    """
    achieved = np.real(np.einsum('fmm->fm',
                                 H @ X @ H.conjugate().transpose(0, 2, 1)))
    ok = mask[:, None] & (y > 0) & (achieved > 0)
    if not ok.any():
        return np.nan
    return float(np.abs(10.0*np.log10(achieved[ok]/y[ok])).mean())


def select_pareto_parameter(lambdas, input_energy, residual, exclude=0.1):
    """Section 3.1: the minimum of log(residuals) + log(inputs).

    The highest `exclude` fraction of each objective is removed from
    consideration first.  Without that, the curve's two ends -- zero residual
    at enormous drive, and zero drive at enormous residual -- always win.
    Returns the chosen lambda and the index into `lambdas`.
    """
    lambdas = np.asarray(lambdas, float)
    inp = np.asarray(input_energy, float)
    res = np.asarray(residual, float)
    keep = np.isfinite(inp) & np.isfinite(res) & (inp > 0) & (res > 0)
    if keep.any():
        keep &= inp < (1.0 - exclude)*inp[keep].max()
    if keep.any():
        keep &= res < (1.0 - exclude)*res[keep].max()
    if not keep.any():
        keep = np.isfinite(inp) & np.isfinite(res) & (inp > 0) & (res > 0)
    if not keep.any():
        return float(lambdas[0]), 0
    score = np.full(lambdas.shape, np.inf)
    score[keep] = np.log(res[keep]) + np.log(inp[keep])
    index = int(np.argmin(score))
    return float(lambdas[index]), index


def pareto_curve(specification, transfer_function, lambdas=None,
                 weighted=False, exact_branch=True, refine=True, lines=None,
                 solver=None, verbose=False, select='paper'):
    """Algorithm 1: the whole curve, the refinement, and the selected lambda.

    Offline.  This is the function to run before a test; the law below takes
    the lambda it reports.  Returns a dict with the curve, `lambda_hat`, and
    `inputs` -- the per-line drive CPSD assembled at lambda_hat, which with
    `refine` is the per-line best over the grid rather than a single grid
    point's solve.

    `mean_abs_db` carries this study's metric at every grid point, which the
    paper's objective does not see.  `select` picks the parameter: 'paper' is
    Section 3.1, 'db' takes the grid point with the lowest mean |dB| instead
    and is a departure -- see the measured reason in VERIFIED.
    """
    _require_cvxpy()
    solver = solver or _solver()
    lambdas = tuple(DEFAULT_LAMBDAS if lambdas is None else lambdas)
    H = np.asarray(transfer_function)
    y = _spec_diagonal(specification)
    in_band = y.max(axis=1) > 0
    todo = np.nonzero(in_band)[0] if lines is None else np.asarray(lines)

    solved_mask = np.zeros(H.shape[0], dtype=bool)
    solved_mask[todo] = True

    solves, energy, residual, db = [], [], [], []
    for lam in lambdas:
        X, info = pareto_inputs(specification, transfer_function, lam,
                                weighted, exact_branch, todo, solver)
        solves.append(X)
        energy.append(info['total_input_energy'])
        residual.append(info['total_residual'])
        db.append(_mean_abs_db(H, X, y, solved_mask))
        if verbose:
            print(f'  lambda {lam:<10.4g} residual {residual[-1]:.6g} '
                  f'energy {energy[-1]:.6g} mean|dB| {db[-1]:.3f} '
                  f'exact {info["n_exact"]}')

    if select == 'db':
        # NOT the paper's criterion.  Section 4.3 names a convex dB surrogate
        # as future work; this is the blunt alternative -- score the grid in the
        # metric we actually care about and take the best point on it.  It
        # abandons the energy objective entirely, so it is not Pareto
        # selection, and it is not the default.
        finite = np.isfinite(db)
        index = int(np.argmin(np.where(finite, db, np.inf)))
        lambda_hat = float(lambdas[index])
    else:
        lambda_hat, index = select_pareto_parameter(lambdas, energy, residual)

    # Section 3.1's refinement: at the selected lambda, take per line whichever
    # grid solve minimizes Eq. (2) for THAT lambda.  Cheap, and the paper notes
    # it is asymptotically optimal as the grid grows.
    X_hat = np.array(solves[index])
    picked = np.full(H.shape[0], index)
    if refine and len(solves) > 1:
        stack = np.stack(solves)
        achieved = np.real(np.einsum('lfmm->lfm', H @ stack
                                     @ H.conjugate().transpose(0, 2, 1)))
        res_line = np.abs(y[None] - achieved)
        if weighted:
            w = np.zeros_like(y)
            np.divide(1.0, y, out=w, where=y > 0)
            res_line = res_line*w[None]
        objective = (res_line.sum(axis=2)
                     + lambda_hat*np.real(np.einsum('lfjj->lf', stack)))
        best = np.argmin(objective, axis=0)
        X_hat[todo] = stack[best[todo], todo]
        picked[todo] = best[todo]

    achieved = np.real(np.einsum('fmm->fm',
                                 H @ X_hat @ H.conjugate().transpose(0, 2, 1)))
    return {'lambdas': np.asarray(lambdas, float),
            'input_energy': np.asarray(energy, float),
            'residual': np.asarray(residual, float),
            'mean_abs_db': np.asarray(db, float),
            'selection': select,
            'lambda_hat': lambda_hat, 'lambda_index': index,
            'inputs': X_hat, 'solves': solves, 'picked_lambda_index': picked,
            'in_band': in_band, 'solved': todo,
            'total_residual': float(np.abs(y - achieved)[todo].sum()),
            'total_input_energy': float(
                np.real(np.einsum('fjj->f', X_hat))[todo].sum())}


def parse_pareto_parameters(extra_parameters):
    """lam, max_drive_coherence, startup_test_level_cap_db, weighted, set_trace"""
    lam, cap, startup_db, weighted, set_trace = 0.0, 1.0, -9.0, False, False
    fields = [p.strip() for p in str(extra_parameters or '').split(',')]
    fields = [p for p in fields if p != '']
    try:
        if len(fields) > 0:
            lam = float(fields[0])
        if len(fields) > 1:
            cap = float(fields[1])
        if len(fields) > 2:
            startup_db = float(fields[2])
        if len(fields) > 3:
            weighted = bool(float(fields[3]))
        if len(fields) > 4:
            set_trace = bool(float(fields[4]))
    except ValueError:
        pass
    return lam, cap, startup_db, weighted, set_trace


def pareto_optimal_solve(specification, transfer_function, lam=0.0,
                         max_drive_coherence=1.0,
                         startup_test_level_cap_db=-9.0, weighted=False,
                         set_trace=False, solver=None):
    """The open-loop drive, with this folder's guards applied in the required
    order: solve, cap coherence, set level if asked, startup ceiling LAST."""
    X, info = pareto_inputs(specification, transfer_function, lam, weighted,
                            True, None, solver)
    y = _spec_diagonal(specification)
    in_band = info['in_band']
    target_trace = y.sum(axis=1)
    X = _cap_drive_coherence(X, max_drive_coherence)
    if set_trace and in_band.any():
        # Deliberately NOT the default: forcing the trace discards exactly the
        # energy/accuracy tradeoff lambda was chosen to strike.
        X[in_band] = _set_predicted_response_trace(
            X[in_band], np.asarray(transfer_function)[in_band],
            target_trace[in_band].astype(complex))
    X = _apply_startup_level_cap_from_trace(X, target_trace, transfer_function,
                                            startup_test_level_cap_db)
    X[~in_band] = 0.0
    return X


def pareto_optimal_control(specification,            # Specifications
                           warning_levels,           # Warning levels
                           abort_levels,             # Abort Levels
                           transfer_function,        # Transfer Functions
                           noise_response_cpsd,      # Noise levels and correlation
                           noise_reference_cpsd,     # from the system identification
                           sysid_response_cpsd,      # Response levels and correlation
                           sysid_reference_cpsd,     # from the system identification
                           multiple_coherence,       # Coherence from the system identification
                           frames,                   # Frames in the CPSD and FRF matrices
                           total_frames,             # Total frames that could be in them
                           extra_parameters='',      # Extra parameters for the control law
                           last_response_cpsd=None,  # Last Control Response
                           last_output_cpsd=None,    # Last Control Excitation
                           ):
    """Coletti, Davis & Schultz's Pareto-optimal open-loop drive.  See the
    module docstring.

    THIS LAW IS OPEN LOOP.  last_response_cpsd is accepted and never read.
    """
    (lam, cap, startup_db, weighted,
     set_trace) = parse_pareto_parameters(extra_parameters)
    return pareto_optimal_solve(specification, transfer_function, lam, cap,
                                startup_db, weighted, set_trace)


def pareto_optimal_generator():
    """Type 1 (generator) form of pareto_optimal_control.

    The solve is a semidefinite program per line over the whole band, so it is
    done ONCE and replayed.  Nothing downstream of the first command depends on
    anything that changes, this law being open loop -- and re-solving every
    cycle would cost seconds per cycle for no benefit.
    """
    output_cpsd = None
    while True:
        (specification, warning_levels, abort_levels, transfer_function,
         noise_response_cpsd, noise_reference_cpsd, sysid_response_cpsd,
         sysid_reference_cpsd, multiple_coherence, frames, total_frames,
         extra_parameters, last_response_cpsd,
         last_output_cpsd) = yield output_cpsd
        if output_cpsd is None:
            (lam, cap, startup_db, weighted,
             set_trace) = parse_pareto_parameters(extra_parameters)
            output_cpsd = pareto_optimal_solve(
                specification, transfer_function, lam, cap, startup_db,
                weighted, set_trace)
