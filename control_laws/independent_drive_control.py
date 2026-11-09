# -*- coding: utf-8 -*-
"""Smallwood's independent drive algorithm, as a Rattlesnake random control law.

THE IDEA
--------
Only the DIAGONAL of the response CPSD is specified.  Every other law in this
folder answers the resulting freedom in the cross terms by inheriting them from
somewhere -- the specification's own off-diagonals through a pseudoinverse, a
measured system-identification response (the buzz construction), or whatever a
minimum-norm or SDP solve happens to produce.  This law answers it by asking a
physical question instead:

    what do INDEPENDENT drives produce, and how little coupling is needed to fix
    the part they cannot?

With uncorrelated drives the drive CPSD is diagonal, so the response
auto-spectra are a REAL linear map of the drive auto-spectra:

    diag(Syy)_i = sum_j |H_ij|^2 diag(Sxx)_j        i.e.   s_yy = |H|^2 s_xx

That is step 3.1.  |H|^2 is elementwise, NOT a matrix product -- the paper's
H_d H_d^* notation.  It is a real, non-negative (n_resp x n_drive) matrix, and
solving it is a completely different conditioning problem from inverting the
complex H that every other law here uses.

If the solution is non-negative, independent drives achieve the specified
auto-spectra EXACTLY and the law is done: the drive CPSD is diag(s_xx), with no
cross terms at all.  That is the best possible answer to an auto-spectrum-only
specification, and no other law in this folder can return it.

That last step holds only where the system is square and consistent, which the
paper assumes and a real rig need not supply.  With more control channels than
drives, s_yy = |H|^2 s_xx is OVERDETERMINED: a non-negative answer is then the
best fit, not a solution, and it can be positive while missing the specified
auto-spectra by many dB.  So the sign test and the specification test are two
different tests, and this law applies both -- non-negativity to choose the
branch nothing needs correcting, and the residual to confirm there is nothing
to correct.  `_EXACT_RTOL` is that second threshold, and the 6 x 12 rig below
is why it exists.

If some s_xx are negative, the specification cannot be met with uncorrelated
drives -- too much interaction is encoded in H.  Smallwood then clips the
negatives to zero, forms the response those clipped independent drives DO
produce, and corrects its diagonal with a real diagonal congruence:

    Syy0    = H diag(s_xx_clipped) H^H                               (3.2)
    s_i     = sqrt(spec_ii / Syy0_ii)                                (3.3)
    Syyhat  = S_c Syy0 S_c                                           (3.4)
    X       = H^+ Syyhat H^+H                                        (3.5)

The cross terms of Syyhat are the ones independent drives generated, carried
through unchanged by the congruence (a real diagonal congruence preserves
coherence magnitude and phase exactly).  So the drives end up coupled "just
enough" to achieve the specified diagonal, and no more.

HOW THIS DIFFERS FROM THE LAWS ALREADY HERE
-------------------------------------------
Three things are genuinely new; one is familiar wearing different clothes.

1. The cross terms come from the MODEL, not from a measurement.  The buzz
   construction (optimal_diagonal_control, buzz_feedback, band_average and the
   forcefinder SPR law with use_buzz) takes coherence and phase from the
   measured system-identification response.  This law derives them from what
   uncorrelated drives would produce through H.  No survey CPSD is needed, so
   there is no diagonal-only fallback to worry about and nothing is inherited
   from an excitation measured at the wrong level.

   The closest existing object is forcefinder's make_buzz_cpsd_from_frf(),
   which builds H H^H -- the response to uncorrelated UNIT drives.  This law is
   that generalized to uncorrelated drives at SOLVED levels.  (forcefinder's
   Rattlesnake adapter never calls that method.)

2. The regularizing principle is maximum independence.  band_average lets the
   response cross terms land wherever the minimum-norm solution puts them;
   optimal_diagonal_control optimizes them subject to a coherence cap.  Here
   they land where independence puts them, and the coupling introduced is only
   what the clipped solve could not supply.  "Couple just enough" is the same
   design goal as max_drive_coherence, reached constructively rather than as a
   constraint.

3. The |H|^2 solve carries a diagnostic no other law here has.  A negative
   s_xx is a PROOF that the specification requires correlated drives at that
   line.  independent_drive_feasibility() below exposes that offline, with no
   run: it is a property of the plant and the specification together.

4. Not new: step 3.4 is a diagonal congruence, the same algebra as
   diagonal_congruence_control's X <- D X D.  Two differences -- it acts on the
   RESPONSE CPSD rather than the drive CPSD, and S_c is closed form rather than
   a Gauss-Newton step against a measurement.

MEASURED, BEFORE YOU EXPECT THE POSITIVE BRANCH
-----------------------------------------------
On the RC plate (48 control accelerometers, 12 drives) against the BARC
specification, over 400 lines spanning 20-4000 Hz, the unconstrained solve had
at least one negative s_xx on EVERY line -- 400 of 400.  Restricting to a
square 12 x 12 subproblem (first twelve accelerometers, and every fourth
accelerometer) gave the same answer: 0 of 400 lines feasible either way.  The
best achievable response from fully independent drives on that rig was a median
2.53 mean |dB| error, worst line 12.71.

The 6 x 12 rig's frame plant (control_laws/tests/frame_plant.py, whose
accelerance reproduces examples/sixdrive12resp's stored analysis FRF to 2.8e-15)
says the same in a smaller and more awkward way.  Its identified layout is 8 x
controls on 6 x drives, rank 5 of 6 at every line, and it is where the two
tests above were measured apart:

  layout            n_feasible    n_exact    best independent    after coupling
  6x8  (the rig)       5 / 181      0 / 181   4.31 median |dB|    2.94 mean |dB|
  10x14 full rank 10   0 / 181      0 / 181   2.52                1.82
  10x14_rankdef  9/10 12 / 181      0 / 181   2.90                1.83

Note the first column against the second.  Five of the rig's lines pass
Smallwood's sign test and none of them meets the specification -- they are in
fact the five WORST lines in the band, 8.8 to 9.4 mean |dB|, all at the top
edge, 980 to 1000 Hz.  Reading n_feasible as "independent drives work here"
would have inverted the conclusion exactly where it mattered most.

The rig is also the clearest case of what rank deficiency does to this law:
with one of six drive directions structurally absent, the non-negative solve
zeroes a median 4 of 6 drives and the answer comes back at 0.835 median
pairwise drive coherence -- nearly fully correlated, from an algorithm whose
premise is independence.  The full-rank 10x14 on the same model sits at 0.347.
Rank, not channel count, is what governs how much coupling this law ends up
issuing.

So on that article the positive branch never fires and the clipping path IS the
algorithm.  Do not assume the exact-independent case is reachable; check it
with independent_drive_feasibility() first.  A rig with better-separated
shakers, or fewer control channels, may behave entirely differently -- which is
presumably the case the method was written for.

VERIFIED
--------
Loaded the way Rattlesnake loads a control law (spec_from_file_location, no
package context) and exercised on 2026-11-08:

  decoupled plant, 6 x 6 diagonal H   max |dB| error 2.9e-15, and every
                                      off-diagonal drive term exactly 0 -- the
                                      exact-independent branch returns an
                                      uncorrelated drive, as it must
  RC plate, 48 x 12, 291 in-band      0 of 291 lines feasible; median 7 of the
                                      12 drives driven to zero by the
                                      non-negative solve
  best independent response           median 2.74 mean |dB|
  after the congruence couples them   2.035 mean |dB| (predicted, open loop),
                                      worst line 33.3
  drive CPSD                          PSD to -3.2e-15 relative eigenvalue
  -9 dB startup ceiling               predicted trace re spec exactly -9.00 dB
  generator form                      identical output to the function form
  nnls versus the paper's clip-once   2.035 against 2.192 mean |dB|
  6 x 12 rig, three layouts           n_exact 0 of 181 on all three, trace
                                      error at most 2.5e-11 dB on every
                                      in-band line -- the regression guard
                                      against taking an overdetermined line as
                                      exact (check_independent_drive.py step 5)
  ceiling gating, 2026-11-09          first command -5.04 dB on its worst
                                      channel with the trace at exactly
                                      -9.00; converged 2.937 mean |dB|, which
                                      is +1.470 over the achievable floor.
                                      Before the gate the law held the -9 dB
                                      ceiling for the whole test and scored
                                      10.471 -- see the comment in
                                      independent_drive_solve

Two of those deserve attention before anyone runs this.  The non-negative solve
is SPARSE -- it zeroes a median 7 of 12 drives -- so the response it builds is
rank 5 or so, and the drive CPSD that comes back out of the pseudoinverse is
correspondingly low rank and strongly correlated (median pairwise drive
coherence 0.564).  An algorithm that starts from independence can end up
issuing a heavily coupled drive, and on this plant it does.  The worst-line
33.3 dB also says the predicted fit is not uniformly good, which is what an
open-loop solve against a rank-deficient map should be expected to give.

THREE IMPLEMENTATION DECISIONS THE PAPER LEAVES OPEN
----------------------------------------------------
NON-NEGATIVITY.  "Set the drive corresponding to negative auto spectrum to
zero" is one pass of what non-negative least squares does properly: clipping
the negatives of an unconstrained solve is not the same as the best
non-negative solution, because the remaining drives should be re-solved once
the clipped ones are held at zero.  scipy.optimize.nnls does the whole thing.
This law uses nnls when scipy is available and falls back to the literal
clip-once rule otherwise; `nnls` in the parameter string forces either.

OVERDETERMINED SYSTEMS.  With 48 responses and 12 drives there is no exact
solution, so there is no sign to test and Smallwood's positivity criterion
needs a fitting criterion underneath it.  Channels at different levels must
count alike or the loudest ones dominate, so each equation is weighted by
1/spec_ii -- band_average's _psd_fit uses the same relative weighting for the
same reason.  `weighted = False` turns it off and fits raw power.

RANK TRUNCATION BREAKS STEP 3.4.  The congruence guarantees the target diagonal
only if H H^+ = I.  Every law here truncates rank, so it does not, and the
achieved diagonal drifts from the target by whatever the discarded directions
carried.  The level is therefore set EXPLICITLY after the solve, with
_set_predicted_response_trace, exactly as the match-trace laws do after a shape
refresh -- never inherited through the pseudoinverse and hoped for.

FEEDBACK
--------
THIS LAW IS OPEN LOOP.  It solves once from the specification and the transfer
function and has no error feedback of any kind; last_response_cpsd is accepted
and never read.  It is the startup solve of a closed-loop law, offered on its
own so its cross terms can be compared against the buzz construction with
everything downstream held identical.  To close the loop, use it as the startup
branch of match_trace_pseudoinverse in place of pinv(H) @ spec @ pinv(H)^H.

PARAMETERS
----------
Comma separated, all optional, defaulted incrementally like every other law in
this module:

    rcond, max_drive_coherence, startup_test_level_cap_db, weighted, nnls

    rcond                       passed to _mt_pinv.  >= 0 is an absolute ratio;
                                < 0 selects rank by singular-value gap with
                                |rcond| as the protect floor.  Default -0.05.
    max_drive_coherence         cap on |X_ij|/sqrt(X_ii X_jj).  Default 1.0,
                                i.e. no cap: the whole point of this law is to
                                produce the least coupling that works, so
                                capping it further is usually the wrong tool.
    startup_test_level_cap_db   ceiling on the predicted response trace
                                relative to the specification's.  Default -9.0,
                                the same guard every law in this module uses on
                                an unfed-back first command.
    weighted                    1 (default) weights each equation by 1/spec_ii.
    nnls                        1 (default) uses non-negative least squares,
                                0 uses the paper's literal clip-once rule.

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
    from scipy.optimize import nnls as _scipy_nnls
except Exception:                                    # scipy is optional here
    _scipy_nnls = None


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
_mt_pinv = _cl._mt_pinv
_cap_drive_coherence = _cl._cap_drive_coherence
_apply_startup_level_cap_from_trace = _cl._apply_startup_level_cap_from_trace
_set_predicted_response_trace = _cl._set_predicted_response_trace
_apply_running_ceiling = _cl._apply_running_ceiling

# Relative residual below which the |H|^2 solve counts as having SOLVED the
# line, not merely as having come out non-negative.  A consistent system
# returns 1e-16 or so, an inconsistent one returns order 1, so anything in
# between is a wide margin; it is deliberately tight because being wrong in
# this direction returns a drive that does not meet the specification.
_EXACT_RTOL = 1e-10


def parse_independent_drive_parameters(extra_parameters):
    """'rcond, max_drive_coherence, startup_test_level_cap_db, weighted, nnls,
    running_ceiling_db'

    running_ceiling_db defaults to 1e9, which is off, exactly as
    control_laws._parse_open_loop_parameters defaults it for the other
    open-loop laws.  It bounds every command AFTER the first; the startup
    cap bounds the first.  See independent_drive_solve for why they differ.
    """
    rcond, cap, startup_db, weighted, use_nnls = -0.05, 1.0, -9.0, True, True
    running_db = 1e9
    if extra_parameters is not None and str(extra_parameters).strip() != '':
        parts = str(extra_parameters).split(',')

        def _get(i, default, cast=float):
            if len(parts) > i and parts[i].strip() != '':
                try:
                    return cast(parts[i])
                except (TypeError, ValueError):
                    return default
            return default

        rcond = _get(0, rcond)
        cap = _get(1, cap)
        startup_db = _get(2, startup_db)
        weighted = bool(_get(3, 1.0))
        use_nnls = bool(_get(4, 1.0))
        running_db = _get(5, running_db)
    return rcond, cap, startup_db, weighted, use_nnls, running_db


def _spec_diagonal(specification):
    """The specified response auto-spectra, real and non-negative."""
    d = np.real(np.einsum('fmm->fm', np.asarray(specification)))
    return np.maximum(d, 0.0)


def _residual(A, b, s):
    """Relative residual of the fit, ||A s - b|| / ||b||."""
    scale = float(np.linalg.norm(b))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.inf
    return float(np.linalg.norm(A @ s - b))/scale


def _solve_line(A, b, use_nnls):
    """Best non-negative s_xx for one line, whether the UNCONSTRAINED solve was
    already non-negative -- Smallwood's own feasibility test -- and the relative
    residual of the fit.

    Non-negativity and solving the system are the SAME question only when the
    system is square and consistent.  With more controls than drives it is
    overdetermined, so a non-negative least-squares answer can still miss the
    specified auto-spectra badly; the caller needs the residual to tell the two
    apart.  See the 6 x 12 rig entry under MEASURED in the module docstring.
    """
    try:
        s_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(A.shape[1]), False, np.inf
    feasible = bool(np.all(s_ls >= 0.0)) and np.all(np.isfinite(s_ls))
    if feasible:
        return s_ls, True, _residual(A, b, s_ls)
    if use_nnls and _scipy_nnls is not None:
        try:
            s_nn = _scipy_nnls(A, b)[0]
            return s_nn, False, _residual(A, b, s_nn)
        except Exception:
            pass
    s_cl = np.maximum(np.nan_to_num(s_ls, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return s_cl, False, _residual(A, b, s_cl)


def independent_drive_solve(specification, transfer_function, rcond=-0.05,
                            max_drive_coherence=1.0,
                            startup_test_level_cap_db=-9.0, weighted=True,
                            use_nnls=True, return_diagnostics=False,
                            apply_ceiling=True, running_ceiling_db=1e9,
                            last_output_cpsd=None, last_response_cpsd=None):
    """The algorithm.  Returns the drive CPSD, and optionally the diagnostics.

    Pure function of the specification and the transfer function: no measured
    response enters anywhere.  Safe to call offline on a saved FRF.
    """
    H = np.asarray(transfer_function)
    s_yy = _spec_diagonal(specification)
    n_f, n_resp, n_drive = H.shape
    A_all = np.abs(H)**2
    in_band = s_yy.max(axis=1) > 0

    s_xx = np.zeros((n_f, n_drive))
    feasible = np.zeros(n_f, dtype=bool)
    residual = np.full(n_f, np.inf)
    for f in np.nonzero(in_band)[0]:
        A, b = A_all[f], s_yy[f]
        if weighted:
            # Each equation weighted by 1/spec_ii so channels at different
            # levels count alike; out-of-band rows (zero target) drop out.
            w = np.zeros(n_resp)
            np.divide(1.0, b, out=w, where=b > 0)
            A, b = A*w[:, None], b*w
        s_xx[f], feasible[f], residual[f] = _solve_line(A, b, use_nnls)

    # Step 3.2: the response those independent drives actually produce.
    X_ind = np.zeros((n_f, n_drive, n_drive), dtype=complex)
    idx = np.arange(n_drive)
    X_ind[:, idx, idx] = s_xx
    Syy0 = H @ X_ind @ H.conjugate().transpose(0, 2, 1)
    d0 = np.real(np.einsum('fmm->fm', Syy0))

    # Step 3.3 / 3.4: the real diagonal congruence that forces the target
    # diagonal.  A real diagonal congruence preserves coherence magnitude and
    # phase exactly, so the independent-drive cross-term structure survives.
    with np.errstate(divide='ignore', invalid='ignore'):
        sc = np.sqrt(s_yy/d0)
    sc[~np.isfinite(sc)] = 0.0
    sc[d0 <= 0] = 0.0
    Syy_hat = Syy0*sc[:, :, None]*sc[:, None, :]

    # Step 3.5.  Where the independent solve was exact there is nothing to
    # couple, so the diagonal drive CPSD is returned untouched -- going through
    # the pseudoinverse there would only reintroduce cross terms the plant does
    # not need.
    tf_pinv = _mt_pinv(H, rcond)
    output = tf_pinv @ Syy_hat @ tf_pinv.conjugate().transpose(0, 2, 1)
    # Non-negative is NOT the same as exact once there are more controls than
    # drives: the system is overdetermined, so lstsq can return an all-positive
    # s_xx that still misses the specified auto-spectra.  Taking this branch on
    # sign alone returns a diagonal drive that does not meet the spec AND skips
    # the explicit level set below, which is how the 6 x 12 rig's five worst
    # lines were made.  Require the fit as well.
    exact = feasible & in_band & (residual <= _EXACT_RTOL)
    if exact.any():
        output[exact] = X_ind[exact]

    output = _cap_drive_coherence(output, max_drive_coherence)
    # Rank truncation means H H^+ != I, so step 3.4's exactness does not
    # survive the pseudoinverse.  Set the level explicitly instead of
    # inheriting it.  After the cap, never before -- see
    # control_laws._apply_startup_level_cap.
    target_trace = s_yy.sum(axis=1)
    live = in_band & ~exact
    if live.any():
        output[live] = _set_predicted_response_trace(
            output[live], H[live], target_trace[live].astype(complex))
    #
    # WHICH CEILING, AND WHY IT IS GATED (fixed 2026-11-09).  Until today the
    # startup cap was applied on EVERY command, so the law held the response at
    # the cap for the whole test instead of only at startup.  Measured on the
    # 6x8 bed: 10.471 mean |dB| against 2.937 with the cap lifted after the
    # first command, and the drive 9 dB low to match.  Every law in
    # control_laws.py gates on `last_output_cpsd is None` and uses
    # _apply_running_ceiling afterwards; this now does the same.
    #
    # Removing the cap instead would not do: uncapped, the first command
    # measured +3.96 dB on its worst channel against the -3 dB the startup
    # criterion asks for.  The first command needs the cap AND later commands
    # need to reach level, which is what gating gives.
    if apply_ceiling and last_output_cpsd is None:
        output = _apply_startup_level_cap_from_trace(
            output, target_trace, H, startup_test_level_cap_db)
    elif apply_ceiling and np.ndim(specification) == 3:
        output = _apply_running_ceiling(output, specification, H,
                                        last_output_cpsd, last_response_cpsd,
                                        running_ceiling_db)
    output[~in_band] = 0.0

    if not return_diagnostics:
        return output
    return output, _diagnostics(A_all, s_yy, s_xx, in_band, feasible, residual,
                                exact, output)


def _diagnostics(A_all, s_yy, s_xx, in_band, feasible, residual, exact,
                 output):
    """What the |H|^2 solve says about the plant and the specification."""
    achieved = np.einsum('fij,fj->fi', A_all, s_xx)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10.0*np.log10(achieved/s_yy)
    db[~np.isfinite(db)] = np.nan
    zeroed = (s_xx <= 0).sum(axis=1)
    diag = np.real(np.einsum('fjj->fj', output))
    denom = np.sqrt(np.maximum(diag[:, :, None]*diag[:, None, :], 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        coh = np.abs(output)/denom
    coh[~np.isfinite(coh)] = 0.0
    # nanmean over an all-NaN out-of-band row warns; compute the mean by hand
    # so a genuine warning elsewhere still gets through.
    valid = np.isfinite(db)
    count = valid.sum(axis=1)
    total = np.where(valid, np.abs(db), 0.0).sum(axis=1)
    mean_abs_db = np.full(db.shape[0], np.nan)
    np.divide(total, count, out=mean_abs_db, where=count > 0)
    mean_abs_db[~in_band] = np.nan
    n_d = output.shape[-1]
    off = ~np.eye(n_d, dtype=bool)
    return {
        'in_band': in_band,
        # `feasible` is Smallwood's sign test: the unconstrained solve came out
        # non-negative.  `exact` additionally requires it to SOLVE the system,
        # which is a different question whenever there are more controls than
        # drives.  Only `exact` lines get the uncorrelated drive returned.
        'feasible': feasible,
        'exact': exact,
        'residual': residual,
        'n_in_band': int(in_band.sum()),
        'n_feasible': int((feasible & in_band).sum()),
        'n_exact': int((exact & in_band).sum()),
        'drives_zeroed': zeroed,
        'independent_error_db': db,
        # Per LINE, full length, NaN out of band -- every field in this dict
        # is indexed the same way so they can be masked together.
        'mean_abs_independent_error_db': mean_abs_db,
        'mean_drive_coherence': coh[:, off].mean(axis=1),
    }


def independent_drive_feasibility(specification, transfer_function,
                                  weighted=True, use_nnls=True):
    """Offline diagnostic: can this specification be met with INDEPENDENT drives?

    Needs no run and no control law -- it is a property of the plant and the
    specification together.  Returns the diagnostics dictionary described in
    _diagnostics.

    Read `n_exact` out of `n_in_band`, not `n_feasible`.  `n_feasible` is
    Smallwood's sign test, and it answers the specification only on a square,
    consistent system; with more controls than drives a line can be
    non-negative and still miss the spec by several dB, so the two counts
    differ and only `n_exact` means "independent drives achieve this".  The
    other field to read is `mean_abs_independent_error_db`, the best a fully
    uncorrelated drive could do at each line.
    """
    _, diag = independent_drive_solve(
        specification, transfer_function, max_drive_coherence=1.0,
        startup_test_level_cap_db=1e6, weighted=weighted, use_nnls=use_nnls,
        return_diagnostics=True)
    return diag


def independent_drive_control(specification,            # Specifications
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
    """Smallwood's independent drive algorithm.  See the module docstring.

    THIS LAW IS OPEN LOOP.  last_response_cpsd is accepted and never read.
    """
    (rcond, cap, startup_db, weighted, use_nnls,
     running_db) = parse_independent_drive_parameters(extra_parameters)
    return independent_drive_solve(specification, transfer_function, rcond, cap,
                                   startup_db, weighted, use_nnls,
                                   running_ceiling_db=running_db,
                                   last_output_cpsd=last_output_cpsd,
                                   last_response_cpsd=last_response_cpsd)


def independent_drive_generator():
    """Type 1 (generator) form of independent_drive_control.

    The solve is a per-line least-squares problem over the whole band, so it is
    done ONCE and replayed, this law being open loop.

    The CEILING, however, is applied per command and not cached with the solve.
    Caching it was the 2026-11-09 bug in a second form: a drive cached WITH the
    startup cap replays at the cap forever, which is how this law came to sit
    9 dB low for a whole test.  So the cached solve is uncapped and each
    command gets the startup cap or the running ceiling as appropriate.
    """
    base = None
    output_cpsd = None
    while True:
        (specification, warning_levels, abort_levels, transfer_function,
         noise_response_cpsd, noise_reference_cpsd, sysid_response_cpsd,
         sysid_reference_cpsd, multiple_coherence, frames, total_frames,
         extra_parameters, last_response_cpsd,
         last_output_cpsd) = yield output_cpsd
        if base is None:
            (rcond, cap, startup_db, weighted, use_nnls,
             running_db) = parse_independent_drive_parameters(extra_parameters)
            base = independent_drive_solve(
                specification, transfer_function, rcond, cap, startup_db,
                weighted, use_nnls, apply_ceiling=False)
            target_trace = _spec_diagonal(specification).sum(axis=1)
        if last_output_cpsd is None:
            output_cpsd = _apply_startup_level_cap_from_trace(
                base, target_trace, transfer_function, startup_db)
        elif np.ndim(specification) == 3:
            output_cpsd = _apply_running_ceiling(
                base, specification, transfer_function, last_output_cpsd,
                last_response_cpsd, running_db)
        else:
            output_cpsd = base
