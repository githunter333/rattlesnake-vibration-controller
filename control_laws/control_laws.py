import numpy as np

def cpsd_coherence(cpsd):
    num = np.abs(cpsd)**2
    den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
           cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
    den[den==0.0] = 1 # Set to 1
    return np.real(num/
                   den)

def cpsd_phase(cpsd):
    return np.angle(cpsd)

def cpsd_from_coh_phs(asd,coh,phs):
    return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])

def cpsd_autospectra(cpsd):
    return np.einsum('ijj->ij',cpsd)

def match_coherence_phase(cpsd_original,cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_original)
    return cpsd_from_coh_phs(asd,coh,phs)

def trace(cpsd):
    return np.einsum('ijj->i',cpsd)

def _cap_drive_coherence(cpsd, max_drive_coherence):
    """Post-process a drive/output CPSD (shape F x N x N) to cap pairwise
    drive-to-drive coherence at max_drive_coherence (0-1): for every bin and
    every drive pair whose coherence exceeds the cap, shrink that cross
    term's magnitude down to the cap (preserving phase and every
    diagonal/auto-spectrum value), then re-project onto the PSD cone (clip
    any negative eigenvalues the per-pair shrink can introduce) so the
    result is still a physically valid CPSD. max_drive_coherence >= 1.0 is a
    no-op (returns cpsd unchanged).

    This is the same methodology validated in examples/sixdrive12resp/code/
    investigate_buzz_coherence_cap.py and used by
    control_laws/optimal_diagonal_control.py's SDP constraint
    (max_drive_coherence, default 0.95 there), generalized here to the
    closed-form pseudoinverse/match-trace/buzz laws in this module -- see
    design doc notes on systematically comparing each closed-form law with
    and without the cap as test level (and nonlinearity) increases.
    """
    if max_drive_coherence >= 1.0:
        return cpsd
    out = cpsd.copy()
    n = out.shape[-1]
    pair_i, pair_j = np.triu_indices(n, k=1)
    for fi in range(out.shape[0]):
        X = out[fi]
        diagX = np.real(np.diag(X))
        modified = False
        for a, b in zip(pair_i, pair_j):
            denom = np.sqrt(max(diagX[a] * diagX[b], 1e-30))
            if denom <= 0:
                continue
            coh_ab = np.abs(X[a, b]) / denom
            if coh_ab > max_drive_coherence:
                limit_mag = max_drive_coherence * denom
                scale = limit_mag / max(np.abs(X[a, b]), 1e-30)
                X[a, b] *= scale
                X[b, a] = np.conj(X[a, b])
                modified = True
        if modified:
            w, v = np.linalg.eigh(X)
            if np.any(w < 0):
                w = np.clip(w, 0, None)
                X = (v * w) @ v.conj().T
            out[fi] = X
    return out

def _parse_rcond_and_cap(extra_parameters):
    """Shared extra_parameters parsing for pseudoinverse_control,
    match_trace_pseudoinverse, and buzz_control: 'rcond' (unchanged,
    backward-compatible single-value usage) or 'rcond,max_drive_coherence'
    (new -- e.g. '1e-15,0.95' caps pairwise drive-to-drive coherence at
    0.95; omit the second value, or leave it >= 1.0, for the original
    uncapped behavior). Matches optimal_diagonal_control.py's
    max_drive_coherence naming/default (0.95) for consistency."""
    parts = extra_parameters.split(',') if extra_parameters else []
    try:
        rcond = float(parts[0]) if len(parts) >= 1 and parts[0].strip() != '' else 1e-15
    except ValueError:
        rcond = 1e-15
    try:
        max_drive_coherence = float(parts[1]) if len(parts) >= 2 and parts[1].strip() != '' else 1.0
    except ValueError:
        max_drive_coherence = 1.0
    return rcond, max_drive_coherence

def _parse_match_trace_parameters(extra_parameters, default_startup_test_level_cap_db=-9.0):
    """extra_parameters parsing for match_trace_pseudoinverse and
    buzz_control (both apply a startup guard on their first, otherwise-
    unguarded pseudoinverse solve -- see each function's own comments for
    how the guard is applied to its particular control loop structure).
    Same
    'rcond' / 'rcond,max_drive_coherence' as _parse_rcond_and_cap, plus an
    optional third value -- 'rcond,max_drive_coherence,startup_test_level_cap_db'
    -- that caps the very first control command (last_output_cpsd is None)
    so it can't imply more than startup_test_level_cap_db dB relative to
    the specification's own trace (0 dB == full spec-match, the same
    reference the steady-state trace_ratio feedback below converges
    toward). Defaults to -9.0 dB if the third value is omitted/empty.

    Added 2026-09-02: on the very first control cycle there is no prior
    measured response to correct against, so match_trace_pseudoinverse
    falls back to a raw, unguarded pseudoinverse solve straight from
    whatever FRF estimate spectral_processing has computed so far --
    which on the very first frame is a single noisy, unaveraged estimate
    under BOTH exponential and linear averaging (there is no minimum-
    frames gate before an FRF gets published and handed to the control
    law). A poorly-conditioned first-frame FRF can make that pseudoinverse
    solve wildly ill-conditioned, commanding an output far beyond
    anything the loop would ever settle at -- observed on real hardware
    as the control level "overshoot by several decades" before the
    trace_ratio feedback (next iteration onward, once there's a real
    measured response) brings it back down. This cap is a hard safety
    ceiling on that one otherwise-unguarded first command; it does not
    change anything about the loop after last_output_cpsd is no longer
    None. Pass a large value (e.g. 100) for the third parameter to
    effectively disable it.
    """
    parts = extra_parameters.split(',') if extra_parameters else []
    try:
        rcond = float(parts[0]) if len(parts) >= 1 and parts[0].strip() != '' else 1e-15
    except ValueError:
        rcond = 1e-15
    try:
        max_drive_coherence = float(parts[1]) if len(parts) >= 2 and parts[1].strip() != '' else 1.0
    except ValueError:
        max_drive_coherence = 1.0
    try:
        startup_cap_db = float(parts[2]) if len(parts) >= 3 and parts[2].strip() != '' else default_startup_test_level_cap_db
    except ValueError:
        startup_cap_db = default_startup_test_level_cap_db
    # 5th value: refresh the drive SHAPE against the live FRF each cycle.
    # DEFAULT OFF, so match_trace_pseudoinverse reproduces runs 01 and 02.
    # Position 4 is deliberately skipped here -- it is running_ceiling_db for
    # pseudoinverse_control and buzz_control, which share this string format
    # (see _parse_open_loop_parameters), and giving one position two meanings
    # would make parameter strings unsafe to copy between laws.
    try:
        refresh_shape = bool(float(parts[4])) if len(parts) >= 5 and parts[4].strip() != '' else False
    except ValueError:
        refresh_shape = False
    return rcond, max_drive_coherence, startup_cap_db, refresh_shape


def _apply_running_ceiling(output, specification, transfer_function,
                           last_output_cpsd, last_response_cpsd,
                           running_ceiling_db):
    """Per-cycle ceiling for the laws with no error feedback.  Bounds the
    response to running_ceiling_db dB above the specification's own trace,
    using the MORE RESTRICTIVE of two estimates, and never raises a line.

      model-predicted  -- tr(H X H^H), the same quantity the startup cap uses.
      measured-anchored -- tr(last_response_cpsd) * (predicted new / predicted
          previous).  The ABSOLUTE level comes from the measurement; only the
          ratio between this command and the last one is taken from the model.

    THE SECOND ESTIMATE IS THE ONE THAT MATTERS, and the reason is worth
    stating plainly.  A purely model-based ceiling is computed with the same
    transfer function the law just used to build the drive, so it is blind in
    exactly the direction that hurts: if the FRF UNDERSTATES the plant, the law
    commands too much drive AND the ceiling agrees that the response will be
    fine.  Measured on run 01's identification with a ceiling of +3.0 dB and an
    FRF understating the plant by 12 dB, both pseudoinverse_control and
    buzz_control sailed to +14.3 dB of true response with the model ceiling
    engaged and reporting no problem.  Anchoring on tr(last_response_cpsd)
    closes that: the level is then whatever the rig actually did.

    Reading last_response_cpsd here does NOT make these laws closed loop.  It
    is used only to scale DOWN; nothing here can raise a command, and the
    control solve itself still never sees the measured response.

    Ordering: must be the last thing applied, for the same reason as
    _apply_startup_level_cap.
    """
    if transfer_function is None:
        return output
    spec_trace = np.real(trace(specification))
    with np.errstate(over='ignore'):
        max_power_ratio = 10.0**(np.float64(running_ceiling_db)/10.0)
    limit = max_power_ratio*spec_trace

    predicted_new = _predicted_response_trace(transfer_function, output)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        scale = np.minimum(1.0, limit/predicted_new)
    scale[~np.isfinite(scale)] = 1.0
    scale[predicted_new <= 0] = 1.0

    if last_response_cpsd is not None and last_output_cpsd is not None:
        measured_prev = np.real(trace(last_response_cpsd))
        predicted_prev = _predicted_response_trace(transfer_function,
                                                   last_output_cpsd)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            estimated = measured_prev*(predicted_new/predicted_prev)
            scale_measured = np.minimum(1.0, limit/estimated)
        usable = (np.isfinite(scale_measured) & (measured_prev > 0)
                  & (predicted_prev > 0) & (predicted_new > 0))
        scale = np.where(usable, np.minimum(scale, scale_measured), scale)

    scale = np.where(spec_trace > 0, scale, 1.0)
    return output*scale[:, np.newaxis, np.newaxis]


def _parse_open_loop_parameters(extra_parameters,
                                default_startup_test_level_cap_db=-9.0,
                                default_running_ceiling_db=1e9):
    """_parse_match_trace_parameters plus a 4th value, running_ceiling_db, for
    the laws that have no error feedback:
    'rcond,max_drive_coherence,startup_test_level_cap_db,running_ceiling_db'.

    WHY THE OPEN-LOOP LAWS NEED A FOURTH NUMBER (2026-09-17).  The startup cap
    is gated on last_output_cpsd is None and so binds on the first command
    only.  For a law with error feedback that is enough: the steady state is a
    correction on the MEASURED response, so a wrong FRF costs convergence, not
    level.  pseudoinverse_control and buzz_control have no such branch -- every
    cycle is pinv(H) @ target @ pinv(H)^H recomputed from scratch, so the drive
    level is a direct function of whatever H arrives.  With "Update Transfer
    Function During Control" on, a bad FRF estimate is an immediate,
    unbounded level change with nothing in the law to stop it.

    running_ceiling_db bounds the predicted response trace on EVERY cycle, in
    dB relative to the specification's own trace.  It DEFAULTS TO OFF (1e9),
    because switching it on changes what these laws do in steady state and the
    twelve-run comparison set was taken without it -- runs 09 and 11 sat at
    +13.55 and +9.38 dB level, which a +3 dB ceiling would have clamped, so a
    default-on ceiling would silently make those runs unreproducible.  Pass a
    4th value (e.g. 3.0) to switch it on.  The startup cap still governs the
    first command either way.
    """
    rcond, max_drive_coherence, startup_cap_db, _ = _parse_match_trace_parameters(
        extra_parameters, default_startup_test_level_cap_db)
    parts = extra_parameters.split(',') if extra_parameters else []
    try:
        running_ceiling_db = (float(parts[3])
                              if len(parts) >= 4 and parts[3].strip() != ''
                              else default_running_ceiling_db)
    except ValueError:
        running_ceiling_db = default_running_ceiling_db
    return rcond, max_drive_coherence, startup_cap_db, running_ceiling_db


def pseudoinverse_control(specification, # Specifications
                          warning_levels, # Warning levels
                          abort_levels, # Abort Levels
                          transfer_function,  # Transfer Functions
                          noise_response_cpsd,  # Noise levels and correlation 
                          noise_reference_cpsd, # from the system identification
                          sysid_response_cpsd,  # Response levels and correlation
                          sysid_reference_cpsd, # from the system identification
                          multiple_coherence, # Coherence from the system identification
                          frames, # Number of frames in the CPSD and FRF matrices
                          total_frames, # Total frames that could be in the CPSD and FRF matrices
                          extra_parameters = '', # Extra parameters for the control law
                          last_response_cpsd = None, # Last Control Response for Error Correction
                          last_output_cpsd = None, # Last Control Excitation for Drive-based control
                          ):
    """
    A control law that simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the specification by that inverse via
    the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
        Format: 'rcond', 'rcond,max_drive_coherence',
        'rcond,max_drive_coherence,startup_test_level_cap_db', or
        'rcond,max_drive_coherence,startup_test_level_cap_db,running_ceiling_db'
        -- see _parse_open_loop_parameters. The third value caps the very
        first control command at startup_test_level_cap_db dB relative to
        the specification (0 dB = full spec-match), default -9.0; the fourth
        caps EVERY LATER command at running_ceiling_db dB, default +3.0,
        which is what bounds this open-loop law when the FRF is being
        updated during control.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    

    Notes
    -----
    THIS LAW IS OPEN LOOP.  It has no error feedback of any kind:
    ``last_response_cpsd`` is accepted and never referenced in the body, and
    ``last_output_cpsd`` is read only as a first-call sentinel where a startup
    clamp exists.  The solve is recomputed identically every cycle and the law
    cannot converge toward the specification -- it lands wherever the current
    FRF estimate puts it.  The only quantity that changes between calls is the
    transfer function, so with "Update Transfer Function During Control" on the
    law adapts to a changing plant model, not to its own error.  Verified by
    inspection 2026-09-17.

    """
    (rcond, max_drive_coherence, startup_test_level_cap_db,
     running_ceiling_db) = _parse_open_loop_parameters(extra_parameters)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    output = _cap_drive_coherence(output, max_drive_coherence)
    # Startup ceiling (added 2026-09-17), the same guard match_trace_pseudo-
    # inverse and buzz_control have carried since 2026-09-02: the very first
    # command is a raw pseudoinverse solve off whatever FRF estimate exists
    # so far, with nothing measured yet to check it against.  Goes LAST --
    # see _apply_startup_level_cap.
    #
    # CAVEAT, and it is a real one: this law is OPEN LOOP, so the ceiling
    # binds on the first command only and cycle 2 goes straight to the
    # uncapped solve.  It is a guard on the one command issued before anyone
    # has seen the rig respond, not a level limit for the run.  (Same
    # limitation as buzz_control's; the match_trace family does not have it
    # because its steady state converges on measured error.)
    #
    # RUNNING CEILING (2026-09-17): the startup cap governs the first command;
    # running_ceiling_db governs every command after it.  This law is open
    # loop, so with the FRF being updated live its drive level follows H
    # directly and nothing else bounds it -- see _parse_open_loop_parameters.
    if last_output_cpsd is None:
        output = _apply_startup_level_cap(output, specification,
                                          transfer_function,
                                          startup_test_level_cap_db)
    else:
        output = _apply_running_ceiling(output, specification,
                                        transfer_function, last_output_cpsd,
                                        last_response_cpsd,
                                        running_ceiling_db)
    return output

def _apply_startup_level_cap(output, specification, transfer_function,
                             startup_test_level_cap_db):
    """Scale each frequency line so the PREDICTED response trace sits no more
    than startup_test_level_cap_db dB above the specification's own trace.

    MUST BE THE LAST THING APPLIED TO THE OUTPUT. This is a safety ceiling on
    the one otherwise-unguarded first command, and anything that runs after it
    can lift the level straight back through it. Until 2026-09-17 both callers
    ran _cap_drive_coherence afterwards, which did exactly that: with the
    startup clamp correct and the coherence cap at 0.95, the first command
    measured +6.72 dB re spec for match_trace_pseudoinverse and -0.12 dB for
    buzz_control, with worst single lines at +22.41 and +16.53 dB -- against a
    ceiling of -9.00 dB. Uncapped, both hit -9.00 dB exactly. The cap shrinks
    off-diagonal drive terms while preserving the diagonal, which destroys the
    inter-drive cancellation the pseudoinverse solve relies on, so the same
    drive produces far more response; scaling before that happens bounds the
    wrong quantity.
    """
    return _apply_startup_level_cap_from_trace(
        output, np.real(trace(specification)), transfer_function,
        startup_test_level_cap_db)


def _apply_startup_level_cap_from_trace(output, spec_trace, transfer_function,
                                        startup_test_level_cap_db):
    """_apply_startup_level_cap for callers that hold the target as a per-line
    trace (or a target DIAGONAL summed over channels) rather than a full CPSD
    matrix -- optimal_diagonal_control and its _fast subclass keep only
    y_diag_target.  Same ceiling, same must-be-last rule."""
    if output is None or transfer_function is None:
        return output
    spec_trace = np.real(np.asarray(spec_trace))
    predicted_response = (transfer_function @ output
                          @ transfer_function.conjugate().transpose(0, 2, 1))
    output_trace = np.real(trace(predicted_response))
    # np.float64 rather than the Python float: a deliberately huge ceiling
    # (the documented way to disable one, e.g. 1e6 dB) overflows Python's
    # float pow with OverflowError, where numpy gives +inf and the
    # np.minimum below then correctly leaves every line unscaled.
    with np.errstate(over='ignore'):
        max_power_ratio = 10.0**(np.float64(startup_test_level_cap_db)/10.0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        scale = np.minimum(1.0, max_power_ratio*spec_trace/output_trace)
    scale[~np.isfinite(scale)] = 1.0
    scale[output_trace <= 0] = 1.0
    return output*scale[:, np.newaxis, np.newaxis]


def _predicted_response_trace(transfer_function, cpsd):
    """Per-frequency-line trace of the response a drive CPSD is predicted to
    produce through transfer_function: real(tr(H X H^H))."""
    return np.real(trace(transfer_function@cpsd
                         @transfer_function.conjugate().transpose(0, 2, 1)))


def _refresh_drive_shape(specification, transfer_function, rcond):
    """The drive SHAPE the CURRENT transfer function calls for -- the raw
    pseudoinverse solve, with no level meaning attached.  The caller caps it
    and then sets its level explicitly with _set_predicted_response_trace.

    Why this exists (2026-09-17).  match_trace_pseudoinverse and
    match_trace_pseudoinverse_pi update the drive as a per-frequency-line real
    scalar on the previous command: output = last_output_cpsd * correction.
    The transfer function therefore entered ONCE, in the startup pseudoinverse,
    and never again -- so "Update Transfer Function During Control" was a
    complete no-op for both laws.  They tracked LEVEL against a re-estimated
    plant while holding a drive SHAPE synthesized from the plant as it looked
    before the test started.

    NOT used by match_trace_pi_resolve or match_trace_resolve, whose phase-1
    level-only hold is deliberate and whose phase 2 re-solves against the live
    H already.
    """
    if transfer_function is None:
        return None
    tf_pinv = np.linalg.pinv(transfer_function, rcond)
    return tf_pinv@specification@tf_pinv.conjugate().transpose(0, 2, 1)


def _set_predicted_response_trace(output, transfer_function, target_trace):
    """Scale each frequency line of `output` so its predicted response trace
    equals `target_trace`.  MUST run after the coherence cap, not before.

    This is what makes a refreshed drive shape safe to substitute for the
    previous command.  The obvious implementation -- rescale the new shape to
    the old command's level, then cap -- looks equivalent and is not:
    _cap_drive_coherence re-projects onto the PSD cone after shrinking cross
    terms, which moves the level by an amount that depends on the bin's
    conditioning, and is not idempotent.  Inheriting a level through it and
    then capping again compounds that shift every cycle.  Measured on run 01's
    identification with the cap at 0.95, one ill-conditioned line (bin 134)
    went +0.33 dB at cycle 5 and +28.44 dB at cycle 6, dragging the aggregate
    to +2.49 dB on a law that had been sitting at +0.0001 dB.

    Setting the level from the FINAL array removes the question entirely.  It
    also makes the constant-FRF identity exact rather than approximate: with H
    unchanged the capped solve is the same array every cycle, so the scale
    recovered here reproduces last_output_cpsd*correction to the bit.

    target_trace <= 0 silences the line (what the callers' own correction of 0
    did before).  A line whose predicted trace is non-positive or non-finite
    cannot be scaled and is left as it is.
    """
    current = _predicted_response_trace(transfer_function, output)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.asarray(target_trace)/current
    scale = np.where(np.asarray(target_trace) <= 0, 0.0, scale)
    unscalable = (~np.isfinite(scale)) | (current <= 0) | (~np.isfinite(current))
    scale = np.where(unscalable, 1.0, scale)
    return output*scale[:, np.newaxis, np.newaxis]


def match_trace_pseudoinverse(specification, # Specifications
                              warning_levels, # Warning levels
                              abort_levels, # Abort Levels
                              transfer_function,  # Transfer Functions
                              noise_response_cpsd,  # Noise levels and correlation 
                              noise_reference_cpsd, # from the system identification
                              sysid_response_cpsd,  # Response levels and correlation
                              sysid_reference_cpsd, # from the system identification
                              multiple_coherence, # Coherence from the system identification
                              frames, # Number of frames in the CPSD and FRF matrices
                              total_frames, # Total frames that could be in the CPSD and FRF matrices
                              extra_parameters = '', # Extra parameters for the control law
                              last_response_cpsd = None, # Last Control Response for Error Correction
                              last_output_cpsd = None, # Last Control Excitation for Drive-based control
                              ):
    """
    A control law that initially performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.  On subsequent iterations, it will scale
    the output at each frequency line up or down depending on if the frequency line
    is on average higher or low.  This is equivalent to matching the "trace" (sum of
    the diagonal) of the CPSD specification in a closed-loop fashion.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
        Format: 'rcond', 'rcond,max_drive_coherence', or
        'rcond,max_drive_coherence,startup_test_level_cap_db'. The third
        value caps the very first control command at startup_test_level_cap_db
        dB relative to the specification (0 dB = full spec-match); it
        defaults to -9.0 dB if omitted -- see _parse_match_trace_parameters.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    (rcond, max_drive_coherence, startup_test_level_cap_db,
     refresh_shape) = _parse_match_trace_parameters(extra_parameters)
    # If it's the first time through, do the actual control
    apply_startup_cap = False
    target_response_trace = None
    if last_output_cpsd is None:
        # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function,rcond)
        # Return the least squares solution for the new output CPSD
        output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
        # Startup guard (added 2026-09-02): this branch has no prior
        # measured response to correct against, so it's an unguarded raw
        # pseudoinverse solve on whatever FRF estimate is available so far
        # -- typically a single noisy, unaveraged first frame. Clamp the
        # per-frequency-line output trace (power) so it can't imply more
        # than startup_test_level_cap_db dB relative to the specification's
        # own trace, as a hard ceiling on this one otherwise-unguarded
        # command. Steady-state operation (the trace_ratio branch below,
        # once there's a real measured response) is untouched.
        # The clamp is on the PREDICTED RESPONSE relative to the
        # specification, so both traces must be response-domain quantities.
        # Until 2026-09-17 this compared tr(specification) against
        # tr(output) -- and `output` is the DRIVE CPSD, in V^2. That ratio
        # is not dimensionless: it carries units of (response/volt)^2, so
        # its value tracked the plant gain and the response channels' units
        # rather than the overshoot it was meant to bound. Measured on run
        # 01's identification, the raw solve sat -2.04 dB re spec, the
        # clamp as written took it to a median -18.88 dB, and the clamp as
        # documented takes it to exactly -9.00 dB. Roughly 10 dB too
        # aggressive, and unit-dependent on top of that.
        apply_startup_cap = True
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        # Refresh the drive SHAPE against the CURRENT transfer function, and
        # carry the level forward explicitly rather than by inheritance -- see
        # _refresh_drive_shape and _set_predicted_response_trace.  Without this
        # the law never read transfer_function again after the startup solve,
        # so "Update Transfer Function During Control" did nothing at all.
        # Exactly equivalent to output = last_output_cpsd*trace_ratio whenever
        # the FRF is not being updated.
        shape = (_refresh_drive_shape(specification, transfer_function, rcond)
                 if refresh_shape else None)
        if shape is None:
            output = last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
            target_response_trace = None
        else:
            output = shape
            target_response_trace = trace_ratio*_predicted_response_trace(
                transfer_function, last_output_cpsd)
    # Note: a uniform per-bin real scalar (the trace_ratio branch, and the
    # startup guard above) leaves pairwise coherence ratios unchanged, so
    # this cap is only ever "doing work" on the raw pseudoinverse itself --
    # but it's applied unconditionally here too so a mid-test law switch
    # (or any other path that reaches this point with an uncapped
    # last_output_cpsd) can't silently skip the cap.
    output = _cap_drive_coherence(output, max_drive_coherence)
    # Level is set AFTER the cap, never inherited through it -- see
    # _set_predicted_response_trace.
    if not apply_startup_cap and target_response_trace is not None:
        output = _set_predicted_response_trace(output, transfer_function,
                                               target_response_trace)
    # The startup ceiling goes LAST -- see _apply_startup_level_cap.
    if apply_startup_cap:
        output = _apply_startup_level_cap(output, specification,
                                          transfer_function,
                                          startup_test_level_cap_db)
    return output


def _parse_match_trace_pi_parameters(extra_parameters, default_startup_test_level_cap_db=-9.0):
    """extra_parameters parsing for match_trace_pseudoinverse_pi: the same
    'rcond' / 'rcond,max_drive_coherence' / 'rcond,max_drive_coherence,
    startup_test_level_cap_db' as _parse_match_trace_parameters, plus two
    more optional values -- 'rcond,max_drive_coherence,startup_test_level_
    cap_db,Kp,Ki' -- for the proportional and integral gains of the PI
    controller below. Kp defaults to 0.0 and Ki to 1.0, which makes the
    steady-state update mathematically identical to match_trace_pseudo-
    inverse's own trace_ratio update (see match_trace_pseudoinverse_pi's
    docstring), so leaving these two off entirely reproduces today's
    match_trace_pseudoinverse behavior exactly, cap and all.

    Two more optional values after that -- 'rcond,max_drive_coherence,
    startup_test_level_cap_db,Kp,Ki,resonance_sensitivity,max_step_db' --
    control per-frequency-line adaptive gain and a global safety cap (see
    match_trace_pseudoinverse_pi's docstring). Both default to 0.0 --
    fully OFF -- so leaving them off entirely changes nothing about the
    Kp/Ki behavior above.

    NOTE (2026-09-04): the 7th value used to be resonance_window, a
    manually-tuned bin count for a fixed-width local-median baseline.
    That's been replaced by a self-scaling half-power-bandwidth baseline
    that needs no window to tune at all (see
    _half_power_resonance_baseline) -- an old parameter string that
    specified a 7th value will now have it parsed as max_step_db (a dB
    figure) instead of a bin count, which is a real behavior change if
    you reuse an old string verbatim.
    """
    parts = extra_parameters.split(',') if extra_parameters else []
    def _get(i, default):
        try:
            return float(parts[i]) if len(parts) > i and parts[i].strip() != '' else default
        except ValueError:
            return default
    rcond = _get(0, 1e-15)
    max_drive_coherence = _get(1, 1.0)
    startup_cap_db = _get(2, default_startup_test_level_cap_db)
    Kp = _get(3, 0.0)
    Ki = _get(4, 1.0)
    resonance_sensitivity = _get(5, 0.0)
    max_step_db = _get(6, 0.0)
    # 8th value: refresh the drive SHAPE against the live FRF each cycle.
    # DEFAULT OFF, matching match_trace_pseudoinverse.
    refresh_shape = bool(_get(7, 0.0))
    return (rcond, max_drive_coherence, startup_cap_db, Kp, Ki,
            resonance_sensitivity, max_step_db, refresh_shape)


def _find_local_maxima(x):
    """Indices of strict interior local maxima in 1-D array x (x[i] >
    both neighbors). Plateaus aren't flagged -- keeps this simple and
    avoids pathological repeats on near-flat data. Pure numpy.
    """
    if len(x) < 3:
        return np.array([], dtype=int)
    return np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) + 1


def _half_power_resonance_baseline(x, min_peak_ratio=1.15, max_half_width=None):
    """Self-scaling 'what would this line look like if it weren't sitting
    on a resonant peak' baseline for a per-frequency-line magnitude array
    x (e.g. Frobenius norm of transfer_function). Replaces an earlier
    fixed-bin-count sliding-window version (2026-09-03): live testing
    found a 21-bin window badly understated peakiness at a real resonance
    that turned out to be 40-60 Hz wide, and widening the window as a
    fixed number required re-tuning per system/per mode. This version
    instead measures each detected peak's own half-power (-3dB, i.e.
    1/sqrt(2) amplitude) width directly from the data, so it self-scales
    to a narrow high-Q resonance or a broad well-damped one without any
    window-size parameter at all.

    Algorithm:
      1. Find every interior local maximum in x (_find_local_maxima), and
         for each, its flanking local minima (walking outward from the
         peak in both directions while x keeps not-increasing).
      2. Keep only maxima at least min_peak_ratio times their lower
         flanking local minimum -- filters out noise-level bumps that
         aren't real resonant humps. min_peak_ratio is an internal
         robustness constant, not meant to be user-tuned.
      3. Process surviving peaks tallest-first (so a smaller shoulder
         peak's zone can't steal bins a taller neighboring peak needs),
         and for each, walk outward from the peak in both directions
         until x falls to peak_value/sqrt(2) (bounded by the flanking
         local minima found in step 1, a taller peak's already-claimed
         zone, or max_half_width bins -- a generous hardcoded safety cap,
         not user-tunable, that only matters for a pathological
         non-decaying curve) -- this defines how many bins are "in the
         resonance" and get de-peaked.
      4. Baseline = x itself outside every peak's zone; inside a zone,
         linear interpolation between the two FLANKING LOCAL MINIMA
         values from step 1 (not the near-threshold values right at the
         zone's own edge, which are by construction always close to
         peak_value/sqrt(2) regardless of how prominent the peak truly
         is -- anchoring there would cap peakiness near sqrt(2) for
         every resonance no matter how sharp, defeating the point).
         Interpolating using the true background level on either side
         instead lets peakiness reflect the actual peak-to-background
         ratio. At an array boundary with no interior local minimum, the
         single available edge value is held flat.

    Pure numpy (no scipy dependency), so this stays self-contained to
    match_trace_pseudoinverse_pi and can't affect any other control
    law's loadability.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    baseline = x.copy()
    if n < 5:
        return baseline
    if max_half_width is None:
        max_half_width = max(10, n // 2)

    claimed = np.zeros(n, dtype=bool)
    maxima = _find_local_maxima(x)
    if maxima.size == 0:
        return baseline

    peak_info = []  # (peak index, flanking left-min index, flanking right-min index)
    for i in maxima:
        left_min = i
        while left_min > 0 and x[left_min - 1] <= x[left_min]:
            left_min -= 1
        right_min = i
        while right_min < n - 1 and x[right_min + 1] <= x[right_min]:
            right_min += 1
        lower_flank = min(x[left_min], x[right_min])
        if lower_flank <= 0 or x[i]/lower_flank >= min_peak_ratio:
            peak_info.append((i, left_min, right_min))

    # Tallest first, so overlapping zones resolve in favor of the taller peak.
    peak_info.sort(key=lambda t: -x[t[0]])

    sqrt2 = np.sqrt(2.0)
    for i, left_min, right_min in peak_info:
        if claimed[i]:
            continue
        threshold = x[i]/sqrt2

        left = i
        steps = 0
        while (left > left_min and not claimed[left - 1] and x[left - 1] >= threshold
               and steps < max_half_width):
            left -= 1
            steps += 1

        right = i
        steps = 0
        while (right < right_min and not claimed[right + 1] and x[right + 1] >= threshold
               and steps < max_half_width):
            right += 1
            steps += 1

        claimed[left:right + 1] = True
        left_edge_val = x[left_min]
        right_edge_val = x[right_min]
        span = right - left
        baseline[left:right + 1] = (np.linspace(left_edge_val, right_edge_val, span + 1)
                                     if span > 0 else np.array([left_edge_val]))

    return baseline


class match_trace_pseudoinverse_pi:
    """A PI-controller variant of match_trace_pseudoinverse.

    match_trace_pseudoinverse's steady-state update -- output =
    last_output_cpsd * trace(specification)/trace(last_response_cpsd) -- is,
    in log-power terms, a pure discrete INTEGRATOR (gain 1) on the log-power
    error at each frequency line: writing e(k) = log(trace(spec)) -
    log(trace(measured response(k))), that update is exactly log(output(k))
    = log(output(k-1)) + e(k-1). There is no proportional term, and the
    error it's integrating comes through whatever CPSD averaging is
    configured for the control phase (a first-order lag for Exponential
    averaging, a boxcar/moving-average lag for Linear) -- a pure integrator
    fed a lagged error is a classic slow-to-settle, overshoot-prone
    configuration.

    Verified numerically (2026-09-03, simplified single-frequency-line
    simulation, see loopsim/pid.py in that session's scratch work): at a
    representative Exponential averaging coefficient of 0.10, the pure-
    integral law took ~53 control cycles and ~103% peak overshoot to settle
    within 5% of a step change; adding a proportional term on top of a
    reduced integral gain (Kp~1.0, Ki~0.5-0.7) settled in 10-30 cycles with
    5-25% overshoot instead, consistently across every averaging
    coefficient tried, and held up just as well with realistic per-frame
    measurement noise added. A derivative term never helped in that search
    -- likely because it mainly amplifies exactly the frame-to-frame noise
    the CPSD averaging exists to suppress -- so this implements PI only,
    not full PID.

    Update per frequency line, per cycle (velocity/incremental form, so
    Ki's contribution doesn't get double-counted against what's already
    baked into last_output_cpsd from prior cycles):
        e(k)   = log(trace(specification)) - log(trace(last_response_cpsd))
        output(k) = last_output_cpsd * exp(Ki*e(k) + Kp*(e(k) - e(k-1)))
    with e(k-1) held as internal state (self.prev_error) between calls --
    this is why this control law needs to be loaded as a Class (not a
    Function): a Function-type control law only gets last_response_cpsd/
    last_output_cpsd back from the environment, which is enough to recover
    e(k) fresh each cycle but not e(k-1), since last_output_cpsd already
    has any prior P contribution folded in and can't be un-mixed from the
    I contribution again.

    Kp=0.0, Ki=1.0 (the defaults) makes this mathematically identical to
    match_trace_pseudoinverse's own steady-state update, and this class
    reuses the exact same startup guard (raw pseudoinverse solve on the
    very first call, trace-capped at startup_test_level_cap_db dB) and
    drive-coherence cap.

    Optional adaptive per-frequency-line gain (resonance_sensitivity,
    off/default by default): live testing (2026-09-03) on the real
    hardware showed that a single global (Kp,Ki) pair that's fast and
    stable on flat/non-resonant lines can badly overshoot at a resonance
    (observed: Kp=0.1,Ki=1.0 drove one resonant line to ~100x/~20dB over
    spec while every flat line converged cleanly) -- a resonance carries
    real mechanical phase lag on top of whatever the CPSD averaging
    contributes, which a uniform gain can't account for. When
    resonance_sensitivity > 0, each frequency line's FRF magnitude
    (Frobenius norm of transfer_function at that line) is compared each
    cycle against a local baseline computed by
    _half_power_resonance_baseline -- a proxy for "what this line would
    look like if it weren't a resonant peak", built by measuring each
    detected peak's own half-power (-3dB) width directly from the data
    rather than assuming a fixed neighborhood size (see that function's
    docstring; this replaced an earlier fixed-bin-count sliding-median
    version on 2026-09-04 after live testing showed a 21-bin window badly
    understated peakiness at a resonance that turned out to be 40-60 Hz
    wide, and that a wider window had to be hand-picked per system to fix
    it -- the half-power version self-scales to each peak's own measured
    width instead, with nothing to tune per system). Both the Kp and Ki
    terms are scaled down together at lines whose FRF magnitude stands
    out above that baseline:
        peakiness   = line_gain / half_power_resonance_baseline(line_gain)
        gain_scale  = 1 / (1 + resonance_sensitivity * max(0, peakiness-1))
    Flat lines (peakiness <= 1) are unaffected (gain_scale = 1); a line
    sitting right on a sharp resonance peak gets its Kp and Ki both
    divided down by however much resonance_sensitivity and its peakiness
    call for. resonance_sensitivity=0.0 (the default) makes gain_scale
    exactly 1.0 everywhere -- fully backward compatible with the
    fixed-gain behavior above.

    Optional global safety cap (max_step_db, off/default by default): a
    hard ceiling on the log-power correction applied to any single
    specified line in one cycle, independent of Kp, Ki, or the resonance
    gain scaling above -- "never let any specified line's drive change by
    more than max_step_db dB in one control cycle", applied after
    resonance-adaptive scaling as a final backstop. Unlike
    resonance_sensitivity, it needs no knowledge of the FRF shape at all
    and so needs no per-system tuning either; the tradeoff is that it
    caps every specified line uniformly (flat lines included), not just
    resonant ones, so it trades a bit of best-case settling speed for a
    universal bound on how bad any single bad cycle can get.
    max_step_db=0.0 (the default) leaves corrections uncapped -- fully
    backward compatible.

    Frequency lines where the specification (or measured response) makes
    the correction undefined -- e.g. an unspecified line outside the
    control band, where trace(specification) is 0 or NaN -- are forced to
    zero drive, exactly like match_trace_pseudoinverse, and their internal
    P/I state is reset rather than left to accumulate stale history so a
    line re-entering the control band later doesn't inherit a spurious
    proportional kick from however long it was masked.

    extra_parameters format: 'rcond', 'rcond,max_drive_coherence',
    'rcond,max_drive_coherence,startup_test_level_cap_db',
    'rcond,max_drive_coherence,startup_test_level_cap_db,Kp,Ki', or
    'rcond,max_drive_coherence,startup_test_level_cap_db,Kp,Ki,
    resonance_sensitivity,max_step_db'. See
    _parse_match_trace_pi_parameters (note the 7th value's meaning
    changed 2026-09-04, from a bin-count window to a dB step cap).
    """

    def __init__(self,
                 specification, # Specifications
                 warning_levels, # Warning levels
                 abort_levels, # Abort Levels
                 extra_parameters, # Extra parameters for the control law
                 transfer_function = None, # Transfer Functions
                 noise_response_cpsd = None, # Noise levels and correlation
                 noise_reference_cpsd = None, # from the system identification
                 sysid_response_cpsd = None, # Response levels and correlation
                 sysid_reference_cpsd = None, # from the system identification
                 multiple_coherence = None, # Coherence from the system identification
                 frames = None, # Number of frames in the CPSD and FRF matrices
                 total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                 last_response_cpsd = None, # Last Control Response for Error Correction
                 last_output_cpsd = None, # Last Control Excitation for Drive-based control
                 ):
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        (self.rcond, self.max_drive_coherence, self.startup_test_level_cap_db,
         self.Kp, self.Ki, self.resonance_sensitivity, self.max_step_db,
         self.refresh_shape) = _parse_match_trace_pi_parameters(extra_parameters)
        # Allocated (to the per-frequency-line shape) on the first call --
        # we don't know the frequency-line count until then.
        self.prev_error = None

    def system_id_update(self,
                         transfer_function = None, # Transfer Functions
                         noise_response_cpsd = None, # Noise levels and correlation
                         noise_reference_cpsd = None, # from the system identification
                         sysid_response_cpsd = None, # Response levels and correlation
                         sysid_reference_cpsd = None, # from the system identification
                         multiple_coherence = None, # Coherence from the system identification
                         frames = None, # Number of frames in the CPSD and FRF matrices
                         total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # match_trace_pseudoinverse doesn't use the system-ID-phase CPSDs
        # for anything (unlike buzz_control's coherence/phase substitution),
        # so neither does this PI variant -- nothing to do here.
        pass

    def control(self,
                transfer_function = None, # Transfer Functions
                multiple_coherence = None, # Coherence from the system identification
                frames = None, # Number of frames in the CPSD and FRF matrices
                total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                last_response_cpsd = None, # Last Control Response for Error Correction
                last_output_cpsd = None, # Last Control Excitation for Drive-based control
                ) -> np.ndarray:
        apply_startup_cap = False
        target_response_trace = None
        if last_output_cpsd is None:
            # Same startup guard as match_trace_pseudoinverse: no prior
            # measured response to correct against yet, so this is a raw,
            # otherwise-unguarded pseudoinverse solve off whatever FRF
            # estimate is available so far -- clamp its trace to
            # startup_test_level_cap_db dB relative to the specification.
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
            # The startup ceiling is computed on the PREDICTED RESPONSE and
            # applied LAST, after _cap_drive_coherence -- see
            # _apply_startup_level_cap for both bugs this replaces (a drive-
            # domain trace ratio that was not dimensionless, and a coherence
            # cap that ran afterwards and lifted the level back through the
            # ceiling).
            apply_startup_cap = True
            self.prev_error = np.zeros(output.shape[0])
        else:
            spec_trace = np.real(trace(self.specification))
            resp_trace = np.real(trace(last_response_cpsd))
            if self.prev_error is None or self.prev_error.shape[0] != spec_trace.shape[0]:
                self.prev_error = np.zeros_like(spec_trace)
            # BUGFIX (2026-09-03, found on a live run): there are two
            # different "can't correct this line normally" cases, and
            # conflating them (the original version of this method did)
            # causes a real, observed failure. A frequency line outside the
            # control band has spec_trace <= 0 EVERY cycle -- a static
            # property of the specification -- and should be silenced
            # (output=0), matching match_trace_pseudoinverse's own
            # behavior there. But a frequency line that IS in the spec and
            # just happens to measure ~0 response THIS cycle (plausible
            # during a rough transient, especially with Kp added) must NOT
            # be forced to output=0: the update is output(k) =
            # output(k-1)*correction, so any line ever driven to exactly 0
            # can never recover on its own (0 times any finite correction
            # is still 0) -- a permanent lock, not a recoverable dip. Live
            # symptom: a PI run's rough startup transient drove response
            # error to ~30 dB below spec and it stayed pinned there for the
            # rest of the run, never working back up. (match_trace_
            # pseudoinverse itself is not immune to the same structural
            # issue -- trace(specification)/trace(last_response_cpsd) with
            # a complex-zero response trace produces inf+nanj, and
            # np.isnan() on that is True because of the NaN component, so
            # its isnan-only mask zeroes the line too -- this fix does NOT
            # change match_trace_pseudoinverse itself, only avoids the same
            # trap here.)
            unspecified = ~np.isfinite(spec_trace) | (spec_trace <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = spec_trace/resp_trace
            transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
            normal = ~unspecified & ~transient_bad

            # Adaptive per-frequency-line gain scaling (2026-09-03, live
            # testing found a single global Kp/Ki can't be both fast on
            # flat lines and stable on a resonant one -- a resonance
            # carries real dynamical phase lag on top of whatever the
            # CPSD averaging contributes, so it has a much lower "safe
            # gain" ceiling). Off by default (resonance_sensitivity=0.0),
            # in which case gain_scale is exactly 1.0 everywhere and this
            # reduces to the fixed-gain behavior above, unchanged.
            if self.resonance_sensitivity > 0 and transfer_function is not None:
                line_gain = np.linalg.norm(transfer_function, axis=(1, 2))
                baseline = _half_power_resonance_baseline(line_gain)
                peakiness = line_gain/np.maximum(baseline, np.finfo(float).tiny)
                gain_scale = 1.0/(1.0 + self.resonance_sensitivity
                                   *np.maximum(0.0, peakiness - 1.0))
            else:
                gain_scale = np.ones_like(spec_trace)

            log_error = np.zeros_like(spec_trace)
            log_error[normal] = np.log(raw_ratio[normal])
            correction = np.zeros_like(spec_trace)
            correction[normal] = gain_scale[normal]*(self.Ki*log_error[normal]
                                   + self.Kp*(log_error[normal] - self.prev_error[normal]))

            # Global safety cap (2026-09-04), independent of resonance
            # gain scaling: needs no FRF-shape knowledge and so needs no
            # per-system tuning, unlike resonance_sensitivity -- a flat
            # ceiling on how much any specified line's drive can move in
            # one cycle. max_step_db=0.0 (default) leaves this uncapped.
            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
            # transient_bad lines get correction=0 (held at last_output_cpsd,
            # not zeroed) via the zeros-initialized correction array above;
            # their prev_error is left untouched below so a later good
            # measurement compares against the last KNOWN-good error rather
            # than a corrupted one.
            # Refresh the drive SHAPE against the CURRENT transfer function,
            # carrying the level forward explicitly -- see
            # _refresh_drive_shape and _set_predicted_response_trace.  Exactly
            # equivalent to output = last_output_cpsd*exp(correction) whenever
            # the FRF is not being updated.
            shape = (_refresh_drive_shape(self.specification, transfer_function,
                                          self.rcond)
                     if self.refresh_shape else None)
            if shape is None:
                output = last_output_cpsd*np.exp(correction)[:,np.newaxis,np.newaxis]
                target_response_trace = None
            else:
                output = shape
                target_response_trace = (
                    np.exp(correction)*_predicted_response_trace(
                        transfer_function, last_output_cpsd))
                target_response_trace[unspecified] = 0.0
            output[unspecified] = 0.0
            new_prev_error = self.prev_error.copy()
            new_prev_error[normal] = log_error[normal]
            self.prev_error = new_prev_error

            # Safety-net floor, independent of the above: even on a
            # "normal" cycle, a large correction (Kp's proportional kick
            # especially) can in principle push a specified line's output
            # trace down near/at machine-epsilon, which is just as
            # unrecoverable as an explicit zero once it gets there. Clamp
            # every SPECIFIED line's output trace to at least a tiny
            # fraction of the largest spec target in the band -- far below
            # anything physically meaningful, but enough to keep the
            # multiplicative recursion able to correct itself back up
            # instead of getting stuck.
            # Per-line floor (fixed 2026-09-06): relative to THIS line's
            # own spec target, not the single loudest line in the band. The
            # original global-max floor (1e-8 * max spec_trace across the
            # whole spectrum) forced every line's output up to the SAME
            # absolute value regardless of how quiet that particular line's
            # own target is -- on a spectrum with wide dynamic range across
            # lines (e.g. loudest line's target ~1e8-1e9x the quietest)
            # this spuriously inflated legitimately-quiet, correctly-
            # converging lines (often exactly the high-plant-gain lines
            # where a small drive is the CORRECT answer), overshooting them
            # instead of rescuing anything -- measured as a persistent
            # several-dB RMS regression vs. an unfloored update on a real
            # captured FRF with ~1e8 dynamic range across lines. Scaling
            # the floor to each line's own spec_trace keeps the same
            # "can't get stuck at zero" protection without imposing a
            # spectrum-wide floor on every other line.
            output_trace = np.real(trace(output))
            floor = 1e-8*spec_trace
            needs_floor = (~unspecified) & (output_trace < floor)
            if np.any(needs_floor):
                scale = np.ones_like(output_trace)
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale[needs_floor] = floor[needs_floor]/np.maximum(output_trace[needs_floor], 1e-300)
                scale[~np.isfinite(scale)] = 1.0
                output = output*scale[:,np.newaxis,np.newaxis]
        output = _cap_drive_coherence(output, self.max_drive_coherence)
        # Level is set AFTER the cap, never inherited through it -- see
        # _set_predicted_response_trace.
        if not apply_startup_cap and target_response_trace is not None:
            output = _set_predicted_response_trace(output, transfer_function,
                                                   target_response_trace)
        # The startup ceiling goes LAST -- see _apply_startup_level_cap.
        if apply_startup_cap:
            output = _apply_startup_level_cap(output, self.specification,
                                              transfer_function,
                                              self.startup_test_level_cap_db)
        return output


def buzz_control(specification, # Specifications
                 warning_levels, # Warning levels
                 abort_levels, # Abort Levels
                 transfer_function,  # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 extra_parameters = '', # Extra parameters for the control law
                 last_response_cpsd = None, # Last Control Response for Error Correction
                 last_output_cpsd = None, # Last Control Excitation for Drive-based control
                 ):
    """
    A control law that updates the coherence and phase of the specification
    with the coherence and phase derived from the system identification phase.
    It then simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
        Format: 'rcond', 'rcond,max_drive_coherence',
        'rcond,max_drive_coherence,startup_test_level_cap_db', or
        'rcond,max_drive_coherence,startup_test_level_cap_db,running_ceiling_db'
        -- see _parse_open_loop_parameters. The third value caps the very
        first control command at startup_test_level_cap_db dB relative to
        the specification (0 dB = full spec-match), default -9.0; the fourth
        caps EVERY LATER command at running_ceiling_db dB, default +3.0,
        which is what bounds this open-loop law when the FRF is being
        updated during control.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    

    Notes
    -----
    THIS LAW IS OPEN LOOP.  It has no error feedback of any kind:
    ``last_response_cpsd`` is accepted and never referenced in the body, and
    ``last_output_cpsd`` is read only as a first-call sentinel where a startup
    clamp exists.  The solve is recomputed identically every cycle and the law
    cannot converge toward the specification -- it lands wherever the current
    FRF estimate puts it.  The only quantity that changes between calls is the
    transfer function, so with "Update Transfer Function During Control" on the
    law adapts to a changing plant model, not to its own error.  Verified by
    inspection 2026-09-17.

    """
    (rcond, max_drive_coherence, startup_test_level_cap_db,
     running_ceiling_db) = _parse_open_loop_parameters(extra_parameters)
    # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)
    apply_startup_cap = False
    if last_output_cpsd is None:
        # Startup guard (added 2026-09-02, mirroring match_trace_pseudoinverse's):
        # unlike match_trace_pseudoinverse, buzz_control has no separate
        # steady-state branch at all -- it always recomputes this same raw
        # pseudoinverse solve every single cycle, using whatever FRF is
        # currently published as control_frf (see
        # random_vibration_sys_id_data_analysis.py's FRF-seeding logic,
        # which now governs what transfer_function actually is here on the
        # very first call too). So there's no way to guard "steady state"
        # differently from "startup" the way match_trace_pseudoinverse does
        # -- instead, only the very first call (last_output_cpsd is None,
        # i.e. no prior measured response/output exists yet) gets clamped
        # here; every later call is completely unaffected, so
        # buzz_control's normal steady-state behavior is unchanged from
        # before this fix. Clamp the per-frequency-line output trace
        # (power) so it can't imply more than startup_test_level_cap_db dB
        # relative to the specification's own trace.
        # The clamp is on the PREDICTED RESPONSE relative to the
        # specification, so both traces must be response-domain quantities.
        # Until 2026-09-17 this compared tr(specification) against
        # tr(output) -- and `output` is the DRIVE CPSD, in V^2. That ratio
        # is not dimensionless: it carries units of (response/volt)^2, so
        # its value tracked the plant gain and the response channels' units
        # rather than the overshoot it was meant to bound. Measured on run
        # 01's identification, the raw solve sat -2.04 dB re spec, the
        # clamp as written took it to a median -18.88 dB, and the clamp as
        # documented takes it to exactly -9.00 dB. Roughly 10 dB too
        # aggressive, and unit-dependent on top of that.
        apply_startup_cap = True
    output = _cap_drive_coherence(output, max_drive_coherence)
    # The ceiling goes LAST -- see _apply_startup_level_cap.
    #
    # RUNNING CEILING (2026-09-17): the startup cap governs the first command;
    # running_ceiling_db governs every command after it.  This law is open
    # loop -- it has no steady-state branch at all, every cycle is the same
    # raw pseudoinverse solve against whatever H arrives -- so with the FRF
    # being updated live nothing else bounds its level.  See
    # _parse_open_loop_parameters.
    if apply_startup_cap:
        output = _apply_startup_level_cap(output, specification,
                                          transfer_function,
                                          startup_test_level_cap_db)
    else:
        output = _apply_running_ceiling(output, specification,
                                        transfer_function, last_output_cpsd,
                                        last_response_cpsd,
                                        running_ceiling_db)
    return output

def buzz_control_generator():
    output_cpsd = None
    modified_spec = None
    while True:
        (specification, # Specifications
         warning_levels, # Warning levels
         abort_levels, # Abort Levels
         transfer_function,  # Transfer Functions
         noise_response_cpsd,  # Noise levels and correlation 
         noise_reference_cpsd, # from the system identification
         sysid_response_cpsd,  # Response levels and correlation
         sysid_reference_cpsd, # from the system identification
         multiple_coherence, # Coherence from the system identification
         frames, # Number of frames in the CPSD and FRF matrices
         total_frames, # Total frames that could be in the CPSD and FRF matrices
         extra_parameters, # Extra parameters for the control law
         last_response_cpsd, # Last Control Response for Error Correction
         last_output_cpsd, # Last Control Excitation for Drive-based control
            ) = yield output_cpsd
        # Only comput the modified spec if it hasn't been yet.
        if modified_spec is None:
            modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
         # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Assign the output_cpsd so it is yielded next time through the loop
        output_cpsd = tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)

class buzz_control_class:
    def __init__(self,
                 specification : np.ndarray, # Specifications
                 warning_levels  : np.ndarray, # Warning levels
                 abort_levels  : np.ndarray, # Abort Levels
                 extra_parameters : str, # Extra parameters for the control law
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                 noise_reference_cpsd : np.ndarray = None, # from the system identification
                 sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                 sysid_reference_cpsd : np.ndarray = None, # from the system identification
                 multiple_coherence : np.ndarray = None, # Coherence from the system identification
                 frames = None, # Number of frames in the CPSD and FRF matrices
                 total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                 last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last Control Excitation for Drive-based control
                 ):
        # Store the specification to the class
        if sysid_response_cpsd is None: # If it's the first time through we won't have a buzz test yet
            self.specification = specification
        else: # Otherwise we can compute the modified spec right away
            self.specification = self.match_coherence_phase(specification, sysid_response_cpsd)
            
    def system_id_update(self,
                         transfer_function : np.ndarray = None,  # Transfer Functions
                         noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                         noise_reference_cpsd : np.ndarray = None, # from the system identification
                         sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                         sysid_reference_cpsd : np.ndarray = None, # from the system identification
                         multiple_coherence : np.ndarray = None, # Coherence from the system identification
                         frames = None, # Number of frames in the CPSD and FRF matrices
                         total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Update the specification with the buzz_cpsd
        self.specification = self.match_coherence_phase(self.specification,sysid_response_cpsd)

    def control(self,
                transfer_function : np.ndarray = None,  # Transfer Functions
                multiple_coherence : np.ndarray = None, # Coherence from the system identification
                frames = None, # Number of frames in the CPSD and FRF matrices
                total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        # Perform the control
        tf_pinv = np.linalg.pinv(transfer_function)
        return tf_pinv @ self.specification @ tf_pinv.conjugate().transpose(0,2,1)
        
    def cpsd_coherence(self,cpsd):
        num = np.abs(cpsd)**2
        den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
               cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
        den[den==0.0] = 1 # Set to 1
        return np.real(num/
                       den)
    
    def cpsd_phase(self,cpsd):
        return np.angle(cpsd)
    
    def cpsd_from_coh_phs(self,asd,coh,phs):
        return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])
    
    def cpsd_autospectra(self,cpsd):
        return np.einsum('ijj->ij',cpsd)
    
    def match_coherence_phase(self,cpsd_original,cpsd_to_match):
        coh = self.cpsd_coherence(cpsd_to_match)
        phs = self.cpsd_phase(cpsd_to_match)
        asd = self.cpsd_autospectra(cpsd_original)
        return self.cpsd_from_coh_phs(asd,coh,phs)


def _parse_buzz_feedback_parameters(extra_parameters, default_startup_test_level_cap_db=-9.0):
    """extra_parameters format for buzz_feedback: 'rcond',
    'rcond,max_drive_coherence', 'rcond,max_drive_coherence,
    startup_test_level_cap_db', 'rcond,max_drive_coherence,
    startup_test_level_cap_db,Ki', 'rcond,max_drive_coherence,
    startup_test_level_cap_db,Ki,max_step_db', or
    'rcond,max_drive_coherence,startup_test_level_cap_db,Ki,max_step_db,
    min_correction_frames'. Same meaning as _parse_match_trace_parameters
    for the first three (max_drive_coherence, startup_test_level_cap_db),
    but NOT the same rcond default -- see below. Ki defaults to 1.0 --
    full per-channel integral correction each cycle, matching
    match_trace_pseudoinverse's own default closed-loop gain -- see
    buzz_feedback's docstring for what Ki actually multiplies here (a
    per-response-channel log-diagonal error, not one aggregate per-line
    trace ratio). max_step_db defaults to 0.0 (uncapped, matching
    match_trace_pseudoinverse_pi's own default for the same parameter) --
    see buzz_feedback's docstring for why leaving it uncapped is NOT
    recommended here despite that default (max_step_db now also rate-
    limits the overall output trace per cycle, not just target_diag --
    see the class docstring's 2026-09-04 update). min_correction_frames
    defaults to 0 (disabled) -- see buzz_feedback's docstring for what it
    gates and why, and why it's off by default despite that same
    recommendation.

    rcond default tightened 1e-15 -> 1e-5 (2026-09-04, live blowup
    investigation): rcond=1e-15 (the value shared with
    _parse_match_trace_parameters) leaves np.linalg.pinv essentially
    untruncated -- any singular value more than 1e-15 times the largest
    is kept and inverted. Checked directly against the real 8-response/
    6-drive rig's captured FRF that day: several frequency lines (980,
    972, 145, ~108-110 Hz) had condition numbers of 1000-7000, i.e. a
    smallest singular value only ~1e-4 to ~1e-3 of the largest -- still
    comfortably above 1e-15 (or even 1e-5; ratios that close to 1e-4
    aren't touched until rcond is raised past roughly 1e-4 to 1e-3 for
    THIS rig), so raising the default alone does not fully neutralize
    those specific lines -- it only removes the truly-negligible tail
    below 1e-5, which 1e-15 was letting through for no benefit. Treat
    1e-5 as a safer floor, not a guarantee: if a specific rig's FRF has
    worse conditioning than this one did, a larger rcond (1e-3 or more)
    may be needed to actually truncate its worst lines -- check with
    np.linalg.svd(H) directly rather than assuming.
    """
    parts = extra_parameters.split(',') if extra_parameters else []
    def _get(i, default):
        try:
            return float(parts[i]) if len(parts) > i and parts[i].strip() != '' else default
        except ValueError:
            return default
    rcond = _get(0, 1e-5)
    max_drive_coherence = _get(1, 1.0)
    startup_cap_db = _get(2, default_startup_test_level_cap_db)
    Ki = _get(3, 1.0)
    max_step_db = _get(4, 0.0)
    min_correction_frames = _get(5, 0.0)
    return rcond, max_drive_coherence, startup_cap_db, Ki, max_step_db, min_correction_frames


class buzz_feedback:
    """A closed-loop variant of buzz_control (2026-09-04).

    Like buzz_control, this seeds the target response CPSD's cross-terms
    (coherence and phase) from a real measurement rather than assuming
    diagonal/uncorrelated behavior -- system ID for the very first
    command. Unlike buzz_control, the cross-terms are then REFRESHED from
    the live last_response_cpsd every cycle after that, so the assumed
    correlation structure self-corrects as control progresses instead of
    staying frozen at whatever the system-ID-level excitation happened to
    show (relevant especially on a nonlinear system, where coherence/phase
    measured at system-ID amplitude need not match coherence/phase at full
    test level).

    More importantly, this version actually closes the loop on tracking
    error, which buzz_control does not: buzz_control_class.control()
    computes one open-loop pseudoinverse solve from the system-ID-informed
    target and then just replays that exact same solve every cycle for the
    rest of the test -- last_response_cpsd is accepted as a parameter but
    never referenced. Any error in that one-shot solve (most likely right
    at a resonance, where the transfer function is worst-conditioned)
    becomes a permanent, uncorrected steady-state error. Live testing
    (2026-09-04) confirmed exactly this failure mode: buzz_control_class
    showed no transient overshoot but settled with many lines persistently
    far off spec.

    Design: rather than match_trace_pseudoinverse's approach of scaling
    the previous DRIVE (output) CPSD by a single per-frequency-line real
    scalar -- which freezes whatever cross-terms the very first raw,
    non-coherence/phase-matched pseudoinverse solve happened to produce,
    and moves every channel at a line by the same ratio even when their
    individual errors differ -- this recomputes the ENTIRE target response
    CPSD fresh each cycle from a per-CHANNEL corrected diagonal plus the
    latest measured coherence/phase, then re-solves the pseudoinverse from
    scratch:
        e_i(f)          = log(specification_ii(f) / last_response_cpsd_ii(f))
        target_diag_i(f) <- target_diag_i(f) * exp(Ki * e_i(f))
        Syy_target(f)   = cpsd_from_coh_phs(target_diag(f),
                                             coherence(last_response_cpsd(f)),
                                             phase(last_response_cpsd(f)))
        output(f)       = H+(f) @ Syy_target(f) @ H+(f)^H
    target_diag is *state* (self.target_diag), persisted between calls and
    initialized from the specification's own diagonal; the pseudoinverse
    solve itself is completely fresh every cycle, not an incremental
    rescaling of the previous drive -- so the result is always a properly
    formed target CPSD built from the current best estimate of achievable
    cross-correlation, rather than an accumulation of scalar tweaks to
    whatever the very first cycle happened to produce. A useful side
    effect of recomputing fresh each cycle rather than multiplying the
    previous OUTPUT: there's no way for the drive to get permanently
    "stuck" at a degenerate value the way a purely multiplicative update
    can (see match_trace_pseudoinverse_pi's floor-safety-net comments) --
    each cycle's output is a fresh linear function of target_diag, which
    is itself guarded against corruption below.

    Ki=1.0 (the default) is full correction each cycle, matching
    match_trace_pseudoinverse's own default gain, just applied per
    response channel here instead of via one aggregate per-line trace
    ratio.

    This is a deliberate first cut, using integral action (Ki) only -- Kp
    (proportional/derivative-style damping) and resonance-aware gain
    scheduling both exist for match_trace_pseudoinverse_pi (see that
    class) and could be added here the same way later if a fixed Ki alone
    proves too aggressive near a resonance; deliberately left out for now
    to keep this first version simple.

    Optional global safety cap (max_step_db, added 2026-09-04, off/default
    by default -- see below for why you should set it anyway): live
    testing the Ki-only version (both at Ki=1.0 and, when that blew up, at
    Ki=0.1) found that Ki alone cannot make this design safe, at ANY
    setting, because the failure mode isn't gradual windup -- it's a
    single-cycle overcorrection. The very first real correction cycle
    (the second control() call) compares against a response measured off
    the deliberately quiet startup-capped command; several channels can
    still be sitting near the noise floor at that point, making
    log(specification/achieved) enormous (tens of nepers) for perfectly
    ordinary reasons, not a fault condition. Even a "conservative" Ki
    multiplies that huge error by a smaller fraction, but a huge number
    times a small fraction can still be huge -- Ki rations how much of a
    bad error gets applied, it does not cap how bad any single error can
    be. Confirmed live (2026-09-04): response error pegged around
    800 dB (~1e80 power ratio), drive collapsed to 0V, at Ki=0.1 -- not a
    slow drift, a first-cycle blowup. max_step_db fixes this the same way
    match_trace_pseudoinverse_pi does: a hard ceiling, independent of Ki,
    on how far log(target_diag) can move for any one channel in one
    cycle, applied per (frequency line, response channel) here rather
    than match_trace_pseudoinverse_pi's per-line scalar:
        correction_i(f)   = Ki * e_i(f)
        correction_i(f)   = clip(correction_i(f), -max_step_nat, max_step_nat)
        target_diag_i(f) <- target_diag_i(f) * exp(correction_i(f))
    where max_step_nat = max_step_db*ln(10)/10. max_step_db=0.0 (the
    default, matching match_trace_pseudoinverse_pi's own convention for
    backward compatibility) leaves this uncapped -- given the live
    failure above, do not actually run with it left at 0; a starting
    point in the 3-8 dB range (match_trace_pseudoinverse_pi was tuned
    successfully at 8 dB on this same rig) is a reasonable first guess,
    not a validated recommendation.

    Optional minimum-frames gate (min_correction_frames, added
    2026-09-04, off/default by default): complementary to max_step_db,
    not a substitute for it. max_step_db bounds how big a correction can
    be once it fires; min_correction_frames instead controls whether a
    correction fires at all yet, by checking the live frame count already
    passed into control() as `frames` (Rattlesnake's own
    self.frames_computed, incremented once per raw measured frame
    regardless of averaging type) against this threshold. While
    frames < min_correction_frames, the entire per-cycle update --
    target_diag's Ki/log-error correction AND the pseudoinverse re-solve
    that follows it -- is skipped, and the drive is held exactly at
    last_output_cpsd instead. This mirrors a guard Rattlesnake's own
    startup code already applies on the FRF side
    (min_live_frf_frames_before_replacing_sysid_seed = 2 in
    random_vibration_sys_id_data_analysis.py, which holds the transfer
    function at its system-ID-seeded value rather than trust an
    immediately-noisy live estimate) but that guard does not extend to
    last_response_cpsd, which every control law -- buzz_feedback
    included -- otherwise receives unconditionally from the very first
    live cycle. Both Linear and Exponential averaging (the two types this
    codebase supports) start from a single raw, unaveraged frame on that
    first cycle, so a low min_correction_frames (say, 1-3) buys little;
    the user-suggested 4-8 is a reasonable starting range, not a
    validated one. min_correction_frames=0 (the default) disables this
    gate entirely, matching max_step_db's off-by-default convention --
    given the live failure documented above, running with EITHER
    max_step_db or min_correction_frames (or both) is recommended over
    running with neither.

    Output-trace rate limiter (2026-09-04, same live-blowup investigation
    as above, added to max_step_db rather than as a separate parameter):
    live testing with min_correction_frames=4, max_step_db=3 still blew
    up on the very first cycle the gate released -- RMS output jumped
    from ~0.13 to ~6.6e13 in that one cycle (confirmed directly from
    Rattlesnake.log timestamps), even though target_diag is mathematically
    incapable of moving by more than a factor of exp(max_step_db*ln(10)/10)
    per cycle. The gap: this closed-loop branch does not do a bounded
    multiplicative update on the PREVIOUS drive the way
    match_trace_pseudoinverse_pi does (output(k) =
    output(k-1)*exp(bounded correction)) -- it rebuilds the full target
    CPSD from live coherence/phase and re-solves the pseudoinverse from
    scratch every cycle, so a bounded diagonal step does not imply a
    bounded output; a biased coherence estimate (a real, well-known
    effect -- coherence estimated from few frames is biased toward 1,
    maximally so at exactly 1 frame) combined with an ill-conditioned FRF
    direction can turn a small target_diag move into an arbitrary output.
    max_step_db, when > 0, now also rate-limits the RESULT directly:
    after the pseudoinverse re-solve, output's trace (power) is clamped
    to at most exp(max_step_db*ln(10)/10) times last cycle's own output
    trace, regardless of what caused that cycle's solve to be large. This
    is relative to the previous drive, not the specification, so it does
    not fight the environment's own test-level ramp -- it only limits how
    fast the drive can move, not how high it can ultimately go once
    genuinely converging.

    Channels/lines the specification doesn't cover (specification_ii <= 0
    or non-finite) have their target_diag forced to zero, matching
    match_trace_pseudoinverse's own behavior for out-of-band lines. A
    channel that measures an exact/near-zero or non-finite response on
    some cycle (plausible during a rough transient) holds its previous
    target_diag state unchanged rather than let log(spec/~0) blow the
    state up to inf/NaN -- mirroring match_trace_pseudoinverse_pi's
    analogous transient-bad guard.

    extra_parameters format: 'rcond', 'rcond,max_drive_coherence',
    'rcond,max_drive_coherence,startup_test_level_cap_db',
    'rcond,max_drive_coherence,startup_test_level_cap_db,Ki',
    'rcond,max_drive_coherence,startup_test_level_cap_db,Ki,max_step_db',
    or 'rcond,max_drive_coherence,startup_test_level_cap_db,Ki,
    max_step_db,min_correction_frames'. See
    _parse_buzz_feedback_parameters.
    """

    def __init__(self,
                 specification, # Specifications
                 warning_levels, # Warning levels
                 abort_levels, # Abort Levels
                 extra_parameters, # Extra parameters for the control law
                 transfer_function = None, # Transfer Functions
                 noise_response_cpsd = None, # Noise levels and correlation
                 noise_reference_cpsd = None, # from the system identification
                 sysid_response_cpsd = None, # Response levels and correlation
                 sysid_reference_cpsd = None, # from the system identification
                 multiple_coherence = None, # Coherence from the system identification
                 frames = None, # Number of frames in the CPSD and FRF matrices
                 total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                 last_response_cpsd = None, # Last Control Response for Error Correction
                 last_output_cpsd = None, # Last Control Excitation for Drive-based control
                 ):
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        (self.rcond, self.max_drive_coherence, self.startup_test_level_cap_db,
         self.Ki, self.max_step_db,
         self.min_correction_frames) = _parse_buzz_feedback_parameters(extra_parameters)
        # Per-response-channel target diagonal: the persistent integral
        # state this law corrects cycle to cycle. Starts at the
        # specification's own autospectra, exactly like buzz_control's
        # modified_spec does before any buzz-phase coherence/phase
        # substitution.
        self.target_diag = np.real(cpsd_autospectra(specification)).copy()
        # Initial target CPSD (specification's diagonal + system-ID-
        # measured coherence/phase, same as buzz_control) -- used only for
        # the very first, startup-guarded command, before any real
        # response measurement exists to refresh the cross-terms from.
        if sysid_response_cpsd is None:
            self.modified_spec = specification
        else:
            self.modified_spec = match_coherence_phase(specification, sysid_response_cpsd)

    def system_id_update(self,
                         transfer_function = None, # Transfer Functions
                         noise_response_cpsd = None, # Noise levels and correlation
                         noise_reference_cpsd = None, # from the system identification
                         sysid_response_cpsd = None, # Response levels and correlation
                         sysid_reference_cpsd = None, # from the system identification
                         multiple_coherence = None, # Coherence from the system identification
                         frames = None, # Number of frames in the CPSD and FRF matrices
                         total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Same as buzz_control_class: once real system-ID data is
        # available, (re-)seed the coherence/phase used for the very first
        # control command from it.
        self.modified_spec = match_coherence_phase(self.specification, sysid_response_cpsd)

    def control(self,
                transfer_function = None, # Transfer Functions
                multiple_coherence = None, # Coherence from the system identification
                frames = None, # Number of frames in the CPSD and FRF matrices
                total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                last_response_cpsd = None, # Last Control Response for Error Correction
                last_output_cpsd = None, # Last Control Excitation for Drive-based control
                ) -> np.ndarray:
        apply_startup_cap = False
        if last_output_cpsd is None:
            # Startup guard, same structure as match_trace_pseudoinverse /
            # match_trace_pseudoinverse_pi: no real measured response yet,
            # so this is a raw, otherwise-unguarded pseudoinverse solve off
            # the system-ID-informed target -- clamp its trace to
            # startup_test_level_cap_db dB relative to the specification.
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@self.modified_spec@tf_pinv.conjugate().transpose(0,2,1)
            # The startup ceiling is computed on the PREDICTED RESPONSE and
            # applied LAST, after _cap_drive_coherence -- see
            # _apply_startup_level_cap for both bugs this replaces (a drive-
            # domain trace ratio that was not dimensionless, and a coherence
            # cap that ran afterwards and lifted the level back through the
            # ceiling).
            apply_startup_cap = True
        else:
            if (self.min_correction_frames > 0
                    and (frames is None or frames < self.min_correction_frames)):
                # Not enough live frames yet to trust last_response_cpsd for
                # a correction (2026-09-04 request) -- same idea as
                # min_live_frf_frames_before_replacing_sysid_seed in
                # random_vibration_sys_id_data_analysis.py, which gives the
                # FRF estimate this kind of grace period but never extended
                # it to the response CPSD any control law (including this
                # one) otherwise receives unconditionally every cycle.
                # Complementary to max_step_db, not a replacement for it:
                # max_step_db bounds how large a correction that DOES fire
                # can be; this skips the correction (and the target_diag
                # update behind it) entirely while frames is still too thin
                # to trust, holding the drive exactly as it was last cycle.
                output = last_output_cpsd
            else:
                spec_diag = np.real(cpsd_autospectra(self.specification))
                achieved_diag = np.real(cpsd_autospectra(last_response_cpsd))

                # Same two-case split as match_trace_pseudoinverse_pi: a line
                # outside the specified band (spec_diag <= 0, a static
                # property of the specification) is silenced; a specified
                # channel that merely measured ~0 or a non-finite response
                # THIS cycle (plausible during a rough transient) must NOT be
                # silenced -- that would be unrecoverable since target_diag is
                # multiplicative -- so its target_diag state is simply held
                # unchanged instead.
                unspecified = ~np.isfinite(spec_diag) | (spec_diag <= 0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    raw_ratio = spec_diag/achieved_diag
                transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
                normal = ~unspecified & ~transient_bad

                log_error = np.zeros_like(spec_diag)
                log_error[normal] = np.log(raw_ratio[normal])
                correction = np.zeros_like(spec_diag)
                correction[normal] = self.Ki*log_error[normal]
                # Global safety cap (2026-09-04, added after a live blowup --
                # see class docstring): caps the per-(line, channel) log-power
                # step independent of Ki, so a single cycle's error -- however
                # large -- can never move target_diag by more than
                # max_step_db in one shot. max_step_db<=0 leaves this uncapped.
                if self.max_step_db > 0:
                    max_step_nat = self.max_step_db*np.log(10.0)/10.0
                    correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
                self.target_diag[normal] = self.target_diag[normal]*np.exp(correction[normal])
                self.target_diag[unspecified] = 0.0
                # transient_bad entries: target_diag left untouched above.

                # Cross-terms refreshed from the live measured response every
                # cycle (2026-09-04 request) -- unlike buzz_control, which
                # only ever uses the system-ID snapshot from __init__/
                # system_id_update, this lets the assumed correlation
                # structure track the real system as control progresses.
                coh = cpsd_coherence(last_response_cpsd)
                phs = cpsd_phase(last_response_cpsd)
                syy_target = cpsd_from_coh_phs(self.target_diag, coh, phs)

                tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
                output = tf_pinv@syy_target@tf_pinv.conjugate().transpose(0,2,1)

                # Output-trace rate limiter (2026-09-04, added after a live
                # blowup -- see class docstring): the target_diag cap above
                # only bounds the DIAGONAL going into this re-solve, not the
                # re-solved output itself -- the coherence/phase rebuild and
                # fresh pseudoinverse each cycle can turn a small, bounded
                # target_diag step into an arbitrarily large output if that
                # cycle's live coherence estimate (biased by a low frame
                # count) or the FRF's conditioning make the re-solve
                # ill-behaved. This reuses max_step_db as a second, direct
                # cap: whatever the fresh solve computed, its trace (power)
                # is not allowed to exceed exp(max_step_nat) times the
                # PREVIOUS cycle's own output trace. Unlike the startup
                # guard's cap (relative to the specification, meant only
                # for the deliberately-quiet first command), this is
                # relative to last cycle's already-in-use drive level, so it
                # does not fight the environment's own test-level ramp --
                # it only limits how fast the drive can move, not how high
                # it can ultimately go. max_step_db<=0 leaves this uncapped,
                # same as the target_diag cap above.
                if self.max_step_db > 0:
                    max_step_nat = self.max_step_db*np.log(10.0)/10.0
                    max_trace_ratio = np.exp(max_step_nat)
                    prev_trace = np.real(trace(last_output_cpsd))
                    out_trace = np.real(trace(output))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        trace_scale = np.minimum(1.0, max_trace_ratio*prev_trace/out_trace)
                    trace_scale[~np.isfinite(trace_scale)] = 1.0
                    trace_scale[out_trace <= 0] = 1.0
                    trace_scale[prev_trace <= 0] = 1.0
                    output = output*trace_scale[:,np.newaxis,np.newaxis]
        output = _cap_drive_coherence(output, self.max_drive_coherence)
        # The startup ceiling goes LAST -- see _apply_startup_level_cap.
        if apply_startup_cap:
            output = _apply_startup_level_cap(output, self.specification,
                                              transfer_function,
                                              self.startup_test_level_cap_db)
        return output


def _parse_match_trace_pi_resolve_parameters(extra_parameters, default_startup_test_level_cap_db=-9.0):
    """extra_parameters parsing for match_trace_pi_resolve (added
    2026-09-05). Format, all optional and defaulted incrementally exactly
    like the other control laws in this module (provide a prefix, the
    rest default):

        rcond, max_drive_coherence, startup_test_level_cap_db,
        Kp, Ki, max_step_db,
        Ki_resolve, ceiling_db,
        stability_tol, stability_hold_cycles, min_hold_cycles,
        max_cycles_before_resolve, avg_alpha, refreeze_enabled

    rcond (default 1e-3, NOT 1e-15): tightened from the historical
    default used elsewhere in this module. Verified 2026-09-04/05 against
    the real captured FRF (examples/sixdrive12resp/results/cva_captures/
    latest_h1_sysid_capture.npz, 8 responses x 6 drives): a full-band SVD
    sweep found 579 of 2049 lines have condition number > 1000 (not just
    the handful of resonance peaks spot-checked earlier this session) --
    with more response channels than drive channels, this system is
    poorly conditioned across a much wider band than a peak-picking check
    would suggest. rcond=1e-3 truncates all of those; see the class
    docstring for the measured cost of doing so.

    max_drive_coherence (default 1.0): unchanged meaning from every other
    law in this module.

    startup_test_level_cap_db (default -9.0): unchanged meaning; applies
    only to the very first command, identical to match_trace_pseudo-
    inverse_pi's own startup guard, which this law reuses verbatim.

    Kp, Ki (defaults 0.3, 0.5): Phase 1 (see class docstring) PI gains on
    log(trace(spec)/trace(response)) -- identical math and identical
    defaults to match_trace_pseudoinverse_pi's own recommended tuning.

    max_step_db (default 3.0, NOT 0.0/off): a single safety cap shared by
    every bounded step in this law -- Phase 1's per-line trace
    correction, Phase 2's per-channel diagonal correction, and Phase 2's
    output-trace rate limiter (see buzz_feedback's docstring/history for
    why the output-trace limiter is needed in addition to a diagonal-only
    cap: without it, a bounded diagonal step does not bound the actual
    re-solved output). This defaults ON here -- every other law in this
    module defaults it OFF, for backward compatibility with pre-existing
    tuned setups -- because this is a new law with no such history, and
    every real blowup diagnosed this session happened with it disabled.

    Ki_resolve (default 0.5): Phase 2's own integral gain on
    log(spec_diag/achieved_diag), independent of Phase 1's Ki -- Phase 2
    corrects a fundamentally different, per-channel quantity against a
    frozen cross-term structure, so there's no reason it should have to
    share a gain with Phase 1's scalar trace correction.

    ceiling_db (default 30.0): Phase 2 anti-windup -- target_diag is
    clamped to never exceed spec_diag * 10**(ceiling_db/10), independent
    of max_step_db's per-cycle RATE limit (a bounded rate does not bound
    an asymptotic value over unlimited cycles -- see this session's
    diag_windup.py reproduction, which is exactly what motivated this).
    Verified 2026-09-05 in simulation against the real FRF: this ceiling
    engaged on 0.04% of (line, channel) cells at convergence -- a
    backstop for a genuinely unfixable channel, not something that costs
    accuracy under normal operation.

    stability_tol, stability_hold_cycles (defaults 0.02, 5): Phase 1 ->
    Phase 2 transition (and later re-freezes) trigger once the cycle-to-
    cycle change in an exponentially-averaged coherence estimate
    (avg_alpha below) stays under stability_tol for stability_hold_cycles
    consecutive cycles. These are the two parameters most likely to need
    retuning per system -- there's no way to derive a universally correct
    threshold, only a validated starting point (see class docstring).

    min_hold_cycles (default 10): floor before the very FIRST resolve is
    eligible, and also the cooldown floor before any later re-freeze --
    prevents acting on a stability read that's only stable because
    nothing has happened yet.

    max_cycles_before_resolve (default 100): fallback ceiling -- forces
    the first resolve even if the stability check never latches (e.g.
    stability_tol tuned too tight for this system's real noise floor), so
    this law can't get stuck running Phase 1 forever. Also doubles as the
    knob for "just use a fixed cycle count instead of auto-detection":
    set stability_hold_cycles=1 and a very loose stability_tol (e.g.
    10.0) so the auto-check trivially passes immediately, and the resolve
    then fires as soon as min_hold_cycles/max_cycles_before_resolve is
    reached -- effectively fixed-cycle behavior using the same machinery.

    avg_alpha (default 0.1): exponential-averaging coefficient (~10-cycle
    effective window) for the coherence tracker that feeds the stability
    check AND becomes coh_frozen/phs_frozen at the moment of a resolve.
    Separate from whatever CPSD averaging the environment itself is
    configured with.

    refreeze_enabled (default 1.0, i.e. True): after the first resolve,
    keep monitoring the same stability check; if the coherence estimate
    moves away from stable (delta > stability_tol at least once -- e.g.
    from a test-level change altering the real cross-coupling through
    amplitude-dependent nonlinearity) and then re-settles, re-freeze
    coh_frozen/phs_frozen from the new stable estimate rather than
    continuing to correct against a stale snapshot. Set to 0.0 to freeze
    exactly once and never again.
    """
    parts = extra_parameters.split(',') if extra_parameters else []
    def _get(i, default):
        try:
            return float(parts[i]) if len(parts) > i and parts[i].strip() != '' else default
        except ValueError:
            return default
    rcond = _get(0, 1e-3)
    max_drive_coherence = _get(1, 1.0)
    startup_cap_db = _get(2, default_startup_test_level_cap_db)
    Kp = _get(3, 0.3)
    Ki = _get(4, 0.5)
    max_step_db = _get(5, 3.0)
    Ki_resolve = _get(6, 0.5)
    ceiling_db = _get(7, 30.0)
    stability_tol = _get(8, 0.02)
    stability_hold_cycles = int(_get(9, 5))
    min_hold_cycles = int(_get(10, 10))
    max_cycles_before_resolve = int(_get(11, 100))
    avg_alpha = _get(12, 0.1)
    refreeze_enabled = _get(13, 1.0) > 0
    return (rcond, max_drive_coherence, startup_cap_db, Kp, Ki, max_step_db,
            Ki_resolve, ceiling_db, stability_tol, stability_hold_cycles,
            min_hold_cycles, max_cycles_before_resolve, avg_alpha, refreeze_enabled)


class match_trace_pi_resolve:
    """Two-phase control law (added 2026-09-05) combining
    match_trace_pseudoinverse_pi's safe, bounded trace-based PI
    correction with a one-time (or periodic) full per-channel resolve
    once the measured coherence/phase have stabilized -- built to answer
    a direct question raised this session: "if match_trace_pseudoinverse
    _pi has equalized to its best response, is there a way to decrease
    the error?"

    Why there's error left for match_trace_pseudoinverse_pi to leave
    behind: that law corrects a SINGLE SCALAR per frequency line --
    log(trace(spec)/trace(response)) -- and multiplies the ENTIRE
    previous drive matrix (diagonal and cross-terms alike) by
    exp(correction). Total power at each line converges to spec, but the
    cross-term SHAPE is whatever the very first startup pseudoinverse
    solve produced (pinv(H) @ specification @ pinv(H)^H, using the
    specification's own assumed coherence) and is never touched again --
    if the plant's real cross-coupling differs from what the spec
    assumed (it generally will; a real structure's coupling isn't
    dictated by the test spec), individual channels can sit off-target
    indefinitely even while total power is exactly right.

    Phase 1 (below, until a resolve triggers) is match_trace_pseudo-
    inverse_pi's exact update law, verbatim -- same math, same defaults,
    same startup guard, same near-zero-lock floor safety net. It exists
    to get the loop to a safe, converged operating point using ONLY the
    well-tested trace-scalar correction, while passively building up an
    exponentially-averaged estimate of the measured response CPSD (see
    avg_alpha) for later use.

    Phase 2 triggers once that averaged coherence estimate has stopped
    changing cycle-to-cycle (stability_tol/stability_hold_cycles) -- not
    on a fixed frame count -- specifically because live coherence
    estimated from only a few frames is a known-biased estimator (a
    single frame's own outer product has coherence exactly 1 between
    every channel pair, always; see this session's live blowup #2,
    caused by exactly this). At that point it freezes the averaged
    coherence/phase (coh_frozen/phs_frozen) and switches to a bounded,
    ceiling-clamped PER-CHANNEL diagonal correction (target_diag_i *=
    exp(clip(Ki_resolve*log(spec_i/achieved_i)))) rebuilt against that
    FROZEN cross-term structure each cycle via cpsd_from_coh_phs, with
    the same output-trace rate limiter buzz_feedback uses (a bounded
    diagonal step does not bound the re-solved output on its own) and an
    absolute ceiling on target_diag (ceiling_db) that buzz_feedback never
    got -- the anti-windup fix identified but not implemented there (see
    diag_windup.py from this session). Cross-terms are only ever
    refreshed at a resolve event, never rebuilt from a fresh live
    snapshot mid-cycle -- this is the structural difference from
    buzz_feedback that makes Phase 2 safe to run continuously.

    If refreeze_enabled (default on), the same stability check keeps
    running after the first resolve: a genuine disturbance (e.g. the
    environment ramping test level, which can shift a real structure's
    amplitude-dependent coupling) will push the tracked coherence away
    from stable, and once it settles again -- at whatever new structure
    applies -- coh_frozen/phs_frozen are refreshed. The per-channel
    diagonal correction itself (Phase 2's Ki_resolve term) is never
    suspended during this -- it keeps tracking spec_i/achieved_i every
    cycle regardless of freeze state, so ordinary level changes are
    handled by that alone; refreeze exists for when the CROSS-TERM shape
    itself needs to change, not the levels.

    Verified 2026-09-05 against the real captured FRF (examples/
    sixdrive12resp/results/cva_captures/latest_h1_sysid_capture.npz) with
    a synthetic diagonal-only specification (each channel's level taken
    from replaying the real system-ID drive through the real FRF, but
    with NO cross-terms specified -- the common real-world case, and
    deliberately not what this 8-response/6-drive system will naturally
    produce, so the reachability gap this law targets is genuine, not
    contrived): Phase 1 alone converged to 12.3 dB RMS per-channel error
    (max 28.8 dB, 1555 of 2049 lines off by >3 dB) despite total power
    matching spec almost exactly throughout. Adding Phase 2 dropped that
    to 1.4 dB RMS (max 17.1 dB, 323 of 2049 lines >3 dB) -- roughly a 9x
    RMS reduction, improving 79% of all (line, channel) error cells.
    Phase 2 plateaus by ~10-20 cycles (not still improving at 40), and
    the ceiling clamp engaged on only 0.04% of cells -- the 1.4 dB
    residual is a real floor (rcond truncation on the worst-conditioned
    579 lines, plus measurement noise), not the anti-windup clamp costing
    accuracy. Separately verified the coherence-bias claim numerically: a
    4-frame estimate (matching the live blowup's frame count) differs
    from a 3000-frame near-ground-truth reference by 0.235 mean absolute
    coherence; a 640-frame running average (what Phase 2 actually uses)
    differs by 0.009 -- about 26x tighter, which is why waiting for
    stability before resolving is safe where reconstructing every cycle
    was not.

    extra_parameters format and every default: see
    _parse_match_trace_pi_resolve_parameters.
    """

    def __init__(self,
                 specification,
                 warning_levels,
                 abort_levels,
                 extra_parameters,
                 transfer_function=None,
                 noise_response_cpsd=None,
                 noise_reference_cpsd=None,
                 sysid_response_cpsd=None,
                 sysid_reference_cpsd=None,
                 multiple_coherence=None,
                 frames=None,
                 total_frames=None,
                 last_response_cpsd=None,
                 last_output_cpsd=None,
                 ):
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        (self.rcond, self.max_drive_coherence, self.startup_test_level_cap_db,
         self.Kp, self.Ki, self.max_step_db,
         self.Ki_resolve, self.ceiling_db,
         self.stability_tol, self.stability_hold_cycles, self.min_hold_cycles,
         self.max_cycles_before_resolve, self.avg_alpha,
         self.refreeze_enabled) = _parse_match_trace_pi_resolve_parameters(extra_parameters)
        # All allocated to the per-frequency-line shape on the first real
        # call, once we know the frequency-line count.
        self.prev_pi_error = None
        self.avg_response = None
        self.coh_prev = None
        self.stable_count = 0
        self.cycles_since_start = 0
        self.cycles_since_last_resolve = 0
        self.has_resolved = False
        self.instability_seen_since_freeze = True
        self.coh_frozen = None
        self.phs_frozen = None
        self.target_diag = None

    def system_id_update(self,
                         transfer_function=None,
                         noise_response_cpsd=None,
                         noise_reference_cpsd=None,
                         sysid_response_cpsd=None,
                         sysid_reference_cpsd=None,
                         multiple_coherence=None,
                         frames=None,
                         total_frames=None,
                         ):
        # Like match_trace_pseudoinverse_pi, this law doesn't use the
        # system-ID-phase CPSDs for anything -- Phase 1's startup guard is
        # a raw pseudoinverse-of-specification solve, and Phase 2 only
        # ever seeds its cross-term structure from LIVE, stability-gated
        # coherence -- so there's nothing to do here.
        pass

    def control(self,
                transfer_function=None,
                multiple_coherence=None,
                frames=None,
                total_frames=None,
                last_response_cpsd=None,
                last_output_cpsd=None,
                ) -> np.ndarray:
        if last_output_cpsd is None:
            # Startup guard: identical to match_trace_pseudoinverse_pi's.
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
            # The startup ceiling is computed on the PREDICTED RESPONSE and
            # applied LAST, after _cap_drive_coherence -- see
            # _apply_startup_level_cap for both bugs this replaces (a drive-
            # domain trace ratio that was not dimensionless, and a coherence
            # cap that ran afterwards and lifted the level back through the
            # ceiling).
            # Reset all persistent state for a fresh run.
            self.prev_pi_error = np.zeros(output.shape[0])
            self.avg_response = None
            self.coh_prev = None
            self.stable_count = 0
            self.cycles_since_start = 0
            self.cycles_since_last_resolve = 0
            self.has_resolved = False
            self.instability_seen_since_freeze = True
            self.coh_frozen = None
            self.phs_frozen = None
            self.target_diag = None
            output = _cap_drive_coherence(output, self.max_drive_coherence)
            # The startup ceiling goes LAST -- see _apply_startup_level_cap.
            output = _apply_startup_level_cap(output, self.specification,
                                              transfer_function,
                                              self.startup_test_level_cap_db)
            return output

        # ------------------------------------------------------------
        # Coherence/stability tracker: runs every cycle regardless of
        # phase, off the environment's own last_response_cpsd (never off
        # the possibly-stale frozen snapshot).
        # ------------------------------------------------------------
        if (self.avg_response is None
                or self.avg_response.shape != last_response_cpsd.shape):
            self.avg_response = last_response_cpsd.copy()
        else:
            self.avg_response = (self.avg_alpha*last_response_cpsd
                                  + (1.0 - self.avg_alpha)*self.avg_response)
        coh_now = cpsd_coherence(self.avg_response)
        n_ch = coh_now.shape[-1]
        off_diag = ~np.eye(n_ch, dtype=bool)
        if self.coh_prev is not None and self.coh_prev.shape == coh_now.shape:
            delta = np.mean(np.abs(coh_now[:, off_diag] - self.coh_prev[:, off_diag]))
        else:
            delta = np.inf
        self.coh_prev = coh_now
        self.cycles_since_start += 1
        self.cycles_since_last_resolve += 1
        if delta > self.stability_tol:
            self.stable_count = 0
            self.instability_seen_since_freeze = True
        else:
            self.stable_count += 1

        ready_first = (not self.has_resolved) and (
            self.cycles_since_start >= self.min_hold_cycles
            and (self.stable_count >= self.stability_hold_cycles
                 or self.cycles_since_start >= self.max_cycles_before_resolve))
        ready_refreeze = (self.has_resolved and self.refreeze_enabled
                           and self.instability_seen_since_freeze
                           and self.cycles_since_last_resolve >= self.min_hold_cycles
                           and (self.stable_count >= self.stability_hold_cycles
                                or self.cycles_since_last_resolve >= self.max_cycles_before_resolve))

        if ready_first or ready_refreeze:
            self.coh_frozen = cpsd_coherence(self.avg_response)
            self.phs_frozen = cpsd_phase(self.avg_response)
            if self.target_diag is None:
                self.target_diag = np.real(cpsd_autospectra(self.avg_response)).copy()
            self.has_resolved = True
            self.cycles_since_last_resolve = 0
            self.instability_seen_since_freeze = False
            self.stable_count = 0

        if not self.has_resolved:
            # ---- Phase 1: match_trace_pseudoinverse_pi's own update, verbatim. ----
            spec_trace = np.real(trace(self.specification))
            resp_trace = np.real(trace(last_response_cpsd))
            if self.prev_pi_error is None or self.prev_pi_error.shape[0] != spec_trace.shape[0]:
                self.prev_pi_error = np.zeros_like(spec_trace)
            unspecified = ~np.isfinite(spec_trace) | (spec_trace <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = spec_trace/resp_trace
            transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
            normal = ~unspecified & ~transient_bad
            log_error = np.zeros_like(spec_trace)
            log_error[normal] = np.log(raw_ratio[normal])
            correction = np.zeros_like(spec_trace)
            correction[normal] = (self.Ki*log_error[normal]
                                   + self.Kp*(log_error[normal] - self.prev_pi_error[normal]))
            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
            output = last_output_cpsd*np.exp(correction)[:,np.newaxis,np.newaxis]
            output[unspecified] = 0.0
            new_prev_error = self.prev_pi_error.copy()
            new_prev_error[normal] = log_error[normal]
            self.prev_pi_error = new_prev_error

            # Per-line floor (fixed 2026-09-06): relative to THIS line's
            # own spec target, not the single loudest line in the band. The
            # original global-max floor (1e-8 * max spec_trace across the
            # whole spectrum) forced every line's output up to the SAME
            # absolute value regardless of how quiet that particular line's
            # own target is -- on a spectrum with wide dynamic range across
            # lines (e.g. loudest line's target ~1e8-1e9x the quietest)
            # this spuriously inflated legitimately-quiet, correctly-
            # converging lines (often exactly the high-plant-gain lines
            # where a small drive is the CORRECT answer), overshooting them
            # instead of rescuing anything -- measured as a persistent
            # several-dB RMS regression vs. an unfloored update on a real
            # captured FRF with ~1e8 dynamic range across lines. Scaling
            # the floor to each line's own spec_trace keeps the same
            # "can't get stuck at zero" protection without imposing a
            # spectrum-wide floor on every other line.
            output_trace = np.real(trace(output))
            floor = 1e-8*spec_trace
            needs_floor = (~unspecified) & (output_trace < floor)
            if np.any(needs_floor):
                scale = np.ones_like(output_trace)
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale[needs_floor] = floor[needs_floor]/np.maximum(output_trace[needs_floor], 1e-300)
                scale[~np.isfinite(scale)] = 1.0
                output = output*scale[:,np.newaxis,np.newaxis]
        else:
            # ---- Phase 2: bounded, ceiling-clamped per-channel diagonal
            #      correction against the FROZEN cross-term structure. ----
            spec_diag = np.real(cpsd_autospectra(self.specification))
            achieved_diag = np.real(cpsd_autospectra(last_response_cpsd))
            unspecified = ~np.isfinite(spec_diag) | (spec_diag <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = spec_diag/achieved_diag
            transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
            normal = ~unspecified & ~transient_bad
            log_error = np.zeros_like(spec_diag)
            log_error[normal] = np.log(raw_ratio[normal])
            correction = np.zeros_like(spec_diag)
            correction[normal] = self.Ki_resolve*log_error[normal]
            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
            ceiling = spec_diag*10.0**(self.ceiling_db/10.0)
            self.target_diag[normal] = np.minimum(
                self.target_diag[normal]*np.exp(correction[normal]), ceiling[normal])

            # Per-(line, channel) floor (fixed 2026-09-06): same reasoning
            # as the Phase-1 floor fix -- relative to each entry's own
            # spec_diag target, not the single loudest (line, channel)
            # entry in the whole band.
            diag_floor = 1e-8*spec_diag
            self.target_diag[normal] = np.maximum(self.target_diag[normal], diag_floor[normal])
            self.target_diag[unspecified] = 0.0
            # transient_bad entries: target_diag left untouched, same
            # reasoning as buzz_feedback -- don't trust this cycle's
            # measurement enough to move the integrator either way.

            S_target = cpsd_from_coh_phs(self.target_diag, self.coh_frozen, self.phs_frozen)
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@S_target@tf_pinv.conjugate().transpose(0,2,1)

            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                max_trace_ratio = np.exp(max_step_nat)
                prev_trace = np.real(trace(last_output_cpsd))
                out_trace = np.real(trace(output))
                with np.errstate(divide='ignore', invalid='ignore'):
                    trace_scale = np.minimum(1.0, max_trace_ratio*prev_trace/out_trace)
                trace_scale[~np.isfinite(trace_scale)] = 1.0
                trace_scale[out_trace <= 0] = 1.0
                trace_scale[prev_trace <= 0] = 1.0
                output = output*trace_scale[:,np.newaxis,np.newaxis]

        return _cap_drive_coherence(output, self.max_drive_coherence)

def _parse_match_trace_resolve_parameters(extra_parameters, default_startup_test_level_cap_db=-9.0):
    """extra_parameters parsing for match_trace_resolve (added
    2026-09-05, same day as match_trace_pi_resolve). Format, all optional
    and defaulted incrementally exactly like every other control law in
    this module (provide a prefix, the rest default):

        rcond, max_drive_coherence, startup_test_level_cap_db,
        max_step_db,
        Ki_resolve, ceiling_db,
        frf_stability_tol, stability_hold_cycles, min_hold_cycles,
        max_cycles_before_resolve, avg_alpha, refreeze_enabled

    This is match_trace_pi_resolve with two changes, both requested
    directly after live use of that law found its 14-parameter Kp/Ki/
    Ki_resolve/max_step_db interaction confusing (and, combined with an
    unclamped rcond, caused a real overshoot on hardware on 2026-09-05):

    1. Phase 1 is match_trace_pseudoinverse's own plain trace_ratio
       update (log_error = log(trace(spec)/trace(response)), output =
       last_output_cpsd*exp(log_error)) -- NOT match_trace_pseudoinverse
       _pi's PI update. No Kp, no Ki: two fewer parameters, and no
       proportional/integral interaction to mistune. max_step_db is
       still applied to Phase 1 (clamping the per-line log correction,
       identical mechanics to match_trace_pseudoinverse_pi's own cap) --
       this is the "safety clamp" added on top of the plain law, which
       had none before.

    2. The Phase 1 -> Phase 2 (and later re-freeze) trigger is FRF
       stability, not response-coherence stability. Rattlesnake's live
       control loop keeps re-estimating transfer_function every cycle as
       more frames accumulate (H1/H2/H3/HV averaging) -- the very first
       cycles' FRF is the noisiest, and it is what the Phase 2 resolve's
       pinv(transfer_function) is built from. Freezing cross-term
       structure off a still-moving FRF estimate freezes it against a
       moving target; gating on "has the incoming transfer_function
       itself stopped changing" ties the trigger to the thing the resolve
       actually depends on, and doubles as the same signal for a later
       test-level-driven re-freeze (a real amplitude-dependent
       nonlinearity shifts the live FRF estimate, not just the response
       coherence). FRF stability is measured each cycle as a single
       global Frobenius-norm-relative change,
       ||transfer_function - previous_transfer_function||_F /
       ||previous_transfer_function||_F, compared against
       frf_stability_tol.

    Phase 2 itself (the per-channel diagonal correction against a frozen
    coherence/phase, Ki_resolve/ceiling_db/the output-trace rate
    limiter) is unchanged from match_trace_pi_resolve -- see that
    class's docstring for the full mechanics and the real-FRF
    verification numbers, which apply here identically since Phase 2's
    code is shared logic, not reimplemented.

    rcond (default 1e-3, NOT 1e-15): same rationale and same measured
    conditioning sweep as match_trace_pi_resolve (579 of 2049 lines on
    the real captured 8-response/6-drive FRF have condition number >
    1000). Do not leave this blank on this law; unlike plain
    match_trace_pseudoinverse (whose own historical default is 1e-15,
    i.e. no truncation), this law defaults to the validated 1e-3.

    max_drive_coherence (default 1.0), startup_test_level_cap_db
    (default -9.0): unchanged meaning from every other law here.

    max_step_db (default 3.0, NOT 0.0/off): shared safety cap for Phase
    1's per-line trace correction, Phase 2's per-channel diagonal
    correction, and Phase 2's output-trace rate limiter -- identical
    role to match_trace_pi_resolve's own max_step_db. Defaults ON here
    for the same reason: this is a new law, and the one live blowup
    diagnosed this session happened with an equivalent cap effectively
    disabled (rcond loosened to 1e-4 at the same time as raising gains).

    Ki_resolve (default 0.5), ceiling_db (default 30.0): Phase 2's
    integral gain and anti-windup ceiling, identical meaning and
    defaults to match_trace_pi_resolve.

    frf_stability_tol (default 0.02): the Frobenius-norm-relative
    cycle-to-cycle FRF change below which the FRF counts as "not moving
    this cycle" -- see mechanism description above. Along with
    stability_hold_cycles, this is the parameter most likely to need
    retuning per system's own live-averaging noise floor.

    stability_hold_cycles (default 5): consecutive stable cycles
    required before a resolve is eligible.

    min_hold_cycles (default 10): floor before the very first resolve is
    eligible, and cooldown floor before any later re-freeze -- prevents
    acting on a stability read that's only stable because nothing has
    happened yet.

    max_cycles_before_resolve (default 100): fallback ceiling forcing
    the first resolve even if FRF stability never latches, so this law
    cannot get stuck in Phase 1 forever. Also doubles as the "just use a
    fixed cycle count" knob: set stability_hold_cycles=1 and a very
    loose frf_stability_tol (e.g. 10.0) so the check trivially passes,
    and the resolve fires at min_hold_cycles/max_cycles_before_resolve
    on a fixed schedule instead.

    avg_alpha (default 0.1): exponential-averaging coefficient (~10-cycle
    effective window) for the response-CPSD average that Phase 2's
    coh_frozen/phs_frozen are drawn from at the moment of a resolve --
    unrelated to the FRF-stability trigger itself, which looks at
    transfer_function directly rather than this average.

    refreeze_enabled (default 1.0, i.e. True): after the first resolve,
    keep watching for the FRF to move away from stable and then
    re-settle (e.g. a test-level change shifting a real amplitude-
    dependent nonlinearity), and re-freeze coh_frozen/phs_frozen from
    the newly stable operating point when it does. Set to 0.0 to freeze
    exactly once.
    """
    parts = extra_parameters.split(',') if extra_parameters else []
    def _get(i, default):
        try:
            return float(parts[i]) if len(parts) > i and parts[i].strip() != '' else default
        except ValueError:
            return default
    rcond = _get(0, 1e-3)
    max_drive_coherence = _get(1, 1.0)
    startup_cap_db = _get(2, default_startup_test_level_cap_db)
    max_step_db = _get(3, 3.0)
    Ki_resolve = _get(4, 0.5)
    ceiling_db = _get(5, 30.0)
    frf_stability_tol = _get(6, 0.02)
    stability_hold_cycles = int(_get(7, 5))
    min_hold_cycles = int(_get(8, 10))
    max_cycles_before_resolve = int(_get(9, 100))
    avg_alpha = _get(10, 0.1)
    refreeze_enabled = _get(11, 1.0) > 0
    return (rcond, max_drive_coherence, startup_cap_db, max_step_db,
            Ki_resolve, ceiling_db, frf_stability_tol, stability_hold_cycles,
            min_hold_cycles, max_cycles_before_resolve, avg_alpha, refreeze_enabled)


class match_trace_resolve:
    """match_trace_pseudoinverse (the plain, non-PI trace-matching law)
    plus match_trace_pi_resolve's Phase 2 per-channel resolve, with the
    Phase 1 -> Phase 2 trigger changed to FRF stability instead of
    response-coherence stability. Added 2026-09-05, directly after live
    hardware use of match_trace_pi_resolve found its 14-parameter
    Kp/Ki/Ki_resolve/max_step_db surface confusing to tune, and after a
    real overshoot caused by loosening rcond and raising gains together.
    See _parse_match_trace_resolve_parameters for the full parameter
    list and every default's rationale.

    Phase 1 (until a resolve triggers): match_trace_pseudoinverse's own
    steady-state update, output = last_output_cpsd *
    exp(clip(log(trace(spec)/trace(response)), +-max_step_db)) -- no
    Kp, no Ki. This is deliberately the simplest possible correction:
    match_trace_pi_resolve's Kp/Ki PI gains are dropped entirely, and
    the only addition relative to plain match_trace_pseudoinverse is the
    max_step_db safety clamp (previously match_trace_pseudoinverse had
    no per-cycle step limit at all) and the same near-zero-lock floor
    safety net match_trace_pseudoinverse_pi uses. Startup guard (the
    very first call, last_output_cpsd is None) is the same raw
    pseudoinverse-of-specification solve, capped at
    startup_test_level_cap_db dB, that every law in this module uses.

    Trigger: unlike match_trace_pi_resolve (which gates on the measured
    RESPONSE coherence settling), this law gates the resolve on the
    incoming transfer_function argument itself settling -- Rattlesnake's
    live control loop keeps re-estimating the FRF every cycle as more
    frames accumulate (H1/H2/H3/HV averaging), and Phase 2's resolve is
    built directly from pinv(transfer_function); resolving before that
    estimate has itself converged freezes cross-term structure against a
    moving target regardless of how good the response-coherence estimate
    is. FRF stability each cycle is a single global Frobenius-norm-
    relative change, ||H_now - H_prev||_F / ||H_prev||_F, compared
    against frf_stability_tol; stability_hold_cycles consecutive stable
    cycles (plus the min_hold_cycles floor, plus the
    max_cycles_before_resolve fallback ceiling) triggers a resolve
    exactly as in match_trace_pi_resolve, just off this signal instead.
    If refreeze_enabled, the same FRF-stability check continues after
    the first resolve and re-freezes coh_frozen/phs_frozen if the FRF
    moves away from stable (e.g. a test-level change shifting a real
    amplitude-dependent nonlinearity) and then re-settles.

    The response-CPSD average (avg_response, avg_alpha) is still tracked
    every cycle -- it is what coh_frozen/phs_frozen (the actual
    cross-term coherence and phase applied at a resolve) are drawn from
    at the moment the FRF-stability trigger fires. The FRF-stability
    check and the response-coherence average are two separate signals:
    one decides WHEN to resolve, the other decides WHAT to resolve to.

    Phase 2 (once resolved) is match_trace_pi_resolve's Phase 2 verbatim
    -- bounded, ceiling-clamped per-channel diagonal correction
    (Ki_resolve, ceiling_db) against the frozen coherence/phase,
    rebuilt via cpsd_from_coh_phs and pinv(transfer_function) every
    cycle, with the same output-trace rate limiter and diagonal floor.
    See match_trace_pi_resolve's docstring for the full real-FRF
    verification numbers for this half of the law -- that code path is
    shared, not reimplemented, so those results apply here identically
    once resolved.
    """

    def __init__(self,
                 specification,
                 warning_levels,
                 abort_levels,
                 extra_parameters,
                 transfer_function=None,
                 noise_response_cpsd=None,
                 noise_reference_cpsd=None,
                 sysid_response_cpsd=None,
                 sysid_reference_cpsd=None,
                 multiple_coherence=None,
                 frames=None,
                 total_frames=None,
                 last_response_cpsd=None,
                 last_output_cpsd=None,
                 ):
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        (self.rcond, self.max_drive_coherence, self.startup_test_level_cap_db,
         self.max_step_db, self.Ki_resolve, self.ceiling_db,
         self.frf_stability_tol, self.stability_hold_cycles, self.min_hold_cycles,
         self.max_cycles_before_resolve, self.avg_alpha,
         self.refreeze_enabled) = _parse_match_trace_resolve_parameters(extra_parameters)
        self.prev_error = None
        self.avg_response = None
        self.frf_prev = None
        self.stable_count = 0
        self.cycles_since_start = 0
        self.cycles_since_last_resolve = 0
        self.has_resolved = False
        self.instability_seen_since_freeze = True
        self.coh_frozen = None
        self.phs_frozen = None
        self.target_diag = None

    def system_id_update(self,
                         transfer_function=None,
                         noise_response_cpsd=None,
                         noise_reference_cpsd=None,
                         sysid_response_cpsd=None,
                         sysid_reference_cpsd=None,
                         multiple_coherence=None,
                         frames=None,
                         total_frames=None,
                         ):
        # No use for the system-ID-phase CPSDs, same as match_trace_pi_resolve.
        pass

    def control(self,
                transfer_function=None,
                multiple_coherence=None,
                frames=None,
                total_frames=None,
                last_response_cpsd=None,
                last_output_cpsd=None,
                ) -> np.ndarray:
        if last_output_cpsd is None:
            # Startup guard: identical to every other law in this module.
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
            # The startup ceiling is computed on the PREDICTED RESPONSE and
            # applied LAST, after _cap_drive_coherence -- see
            # _apply_startup_level_cap for both bugs this replaces (a drive-
            # domain trace ratio that was not dimensionless, and a coherence
            # cap that ran afterwards and lifted the level back through the
            # ceiling).
            # Reset all persistent state for a fresh run.
            self.prev_error = np.zeros(output.shape[0])
            self.avg_response = None
            self.frf_prev = transfer_function.copy() if transfer_function is not None else None
            self.stable_count = 0
            self.cycles_since_start = 0
            self.cycles_since_last_resolve = 0
            self.has_resolved = False
            self.instability_seen_since_freeze = True
            self.coh_frozen = None
            self.phs_frozen = None
            self.target_diag = None
            output = _cap_drive_coherence(output, self.max_drive_coherence)
            # The startup ceiling goes LAST -- see _apply_startup_level_cap.
            output = _apply_startup_level_cap(output, self.specification,
                                              transfer_function,
                                              self.startup_test_level_cap_db)
            return output

        # ------------------------------------------------------------
        # FRF-stability tracker (decides WHEN to resolve): runs every
        # cycle regardless of phase, off the live transfer_function
        # argument.
        # ------------------------------------------------------------
        if (transfer_function is not None and self.frf_prev is not None
                and self.frf_prev.shape == transfer_function.shape):
            prev_norm = np.linalg.norm(self.frf_prev)
            if prev_norm > 0:
                frf_delta = np.linalg.norm(transfer_function - self.frf_prev)/prev_norm
            else:
                frf_delta = np.inf
        else:
            frf_delta = np.inf
        if transfer_function is not None:
            self.frf_prev = transfer_function.copy()

        # ------------------------------------------------------------
        # Response-CPSD average (decides WHAT to resolve to): runs every
        # cycle regardless of phase, off the environment's own
        # last_response_cpsd.
        # ------------------------------------------------------------
        if (self.avg_response is None
                or self.avg_response.shape != last_response_cpsd.shape):
            self.avg_response = last_response_cpsd.copy()
        else:
            self.avg_response = (self.avg_alpha*last_response_cpsd
                                  + (1.0 - self.avg_alpha)*self.avg_response)

        self.cycles_since_start += 1
        self.cycles_since_last_resolve += 1
        if frf_delta > self.frf_stability_tol:
            self.stable_count = 0
            self.instability_seen_since_freeze = True
        else:
            self.stable_count += 1

        ready_first = (not self.has_resolved) and (
            self.cycles_since_start >= self.min_hold_cycles
            and (self.stable_count >= self.stability_hold_cycles
                 or self.cycles_since_start >= self.max_cycles_before_resolve))
        ready_refreeze = (self.has_resolved and self.refreeze_enabled
                           and self.instability_seen_since_freeze
                           and self.cycles_since_last_resolve >= self.min_hold_cycles
                           and (self.stable_count >= self.stability_hold_cycles
                                or self.cycles_since_last_resolve >= self.max_cycles_before_resolve))

        if ready_first or ready_refreeze:
            self.coh_frozen = cpsd_coherence(self.avg_response)
            self.phs_frozen = cpsd_phase(self.avg_response)
            if self.target_diag is None:
                self.target_diag = np.real(cpsd_autospectra(self.avg_response)).copy()
            self.has_resolved = True
            self.cycles_since_last_resolve = 0
            self.instability_seen_since_freeze = False
            self.stable_count = 0

        if not self.has_resolved:
            # ---- Phase 1: match_trace_pseudoinverse's plain trace_ratio
            #      update, plus the max_step_db safety clamp and the
            #      near-zero-lock floor. ----
            spec_trace = np.real(trace(self.specification))
            resp_trace = np.real(trace(last_response_cpsd))
            if self.prev_error is None or self.prev_error.shape[0] != spec_trace.shape[0]:
                self.prev_error = np.zeros_like(spec_trace)
            unspecified = ~np.isfinite(spec_trace) | (spec_trace <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = spec_trace/resp_trace
            transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
            normal = ~unspecified & ~transient_bad
            log_error = np.zeros_like(spec_trace)
            log_error[normal] = np.log(raw_ratio[normal])
            correction = np.zeros_like(spec_trace)
            correction[normal] = log_error[normal]
            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
            output = last_output_cpsd*np.exp(correction)[:,np.newaxis,np.newaxis]
            output[unspecified] = 0.0
            new_prev_error = self.prev_error.copy()
            new_prev_error[normal] = log_error[normal]
            self.prev_error = new_prev_error

            # Per-line floor (fixed 2026-09-06): relative to THIS line's
            # own spec target, not the single loudest line in the band. The
            # original global-max floor (1e-8 * max spec_trace across the
            # whole spectrum) forced every line's output up to the SAME
            # absolute value regardless of how quiet that particular line's
            # own target is -- on a spectrum with wide dynamic range across
            # lines (e.g. loudest line's target ~1e8-1e9x the quietest)
            # this spuriously inflated legitimately-quiet, correctly-
            # converging lines (often exactly the high-plant-gain lines
            # where a small drive is the CORRECT answer), overshooting them
            # instead of rescuing anything -- measured as a persistent
            # several-dB RMS regression vs. an unfloored update on a real
            # captured FRF with ~1e8 dynamic range across lines. Scaling
            # the floor to each line's own spec_trace keeps the same
            # "can't get stuck at zero" protection without imposing a
            # spectrum-wide floor on every other line.
            output_trace = np.real(trace(output))
            floor = 1e-8*spec_trace
            needs_floor = (~unspecified) & (output_trace < floor)
            if np.any(needs_floor):
                scale = np.ones_like(output_trace)
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale[needs_floor] = floor[needs_floor]/np.maximum(output_trace[needs_floor], 1e-300)
                scale[~np.isfinite(scale)] = 1.0
                output = output*scale[:,np.newaxis,np.newaxis]
        else:
            # ---- Phase 2: identical to match_trace_pi_resolve's Phase 2. ----
            spec_diag = np.real(cpsd_autospectra(self.specification))
            achieved_diag = np.real(cpsd_autospectra(last_response_cpsd))
            unspecified = ~np.isfinite(spec_diag) | (spec_diag <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = spec_diag/achieved_diag
            transient_bad = (~unspecified) & (~np.isfinite(raw_ratio) | (raw_ratio <= 0))
            normal = ~unspecified & ~transient_bad
            log_error = np.zeros_like(spec_diag)
            log_error[normal] = np.log(raw_ratio[normal])
            correction = np.zeros_like(spec_diag)
            correction[normal] = self.Ki_resolve*log_error[normal]
            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                correction[normal] = np.clip(correction[normal], -max_step_nat, max_step_nat)
            ceiling = spec_diag*10.0**(self.ceiling_db/10.0)
            self.target_diag[normal] = np.minimum(
                self.target_diag[normal]*np.exp(correction[normal]), ceiling[normal])

            # Per-(line, channel) floor (fixed 2026-09-06): same reasoning
            # as the Phase-1 floor fix -- relative to each entry's own
            # spec_diag target, not the single loudest (line, channel)
            # entry in the whole band.
            diag_floor = 1e-8*spec_diag
            self.target_diag[normal] = np.maximum(self.target_diag[normal], diag_floor[normal])
            self.target_diag[unspecified] = 0.0

            S_target = cpsd_from_coh_phs(self.target_diag, self.coh_frozen, self.phs_frozen)
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@S_target@tf_pinv.conjugate().transpose(0,2,1)

            if self.max_step_db > 0:
                max_step_nat = self.max_step_db*np.log(10.0)/10.0
                max_trace_ratio = np.exp(max_step_nat)
                prev_trace = np.real(trace(last_output_cpsd))
                out_trace = np.real(trace(output))
                with np.errstate(divide='ignore', invalid='ignore'):
                    trace_scale = np.minimum(1.0, max_trace_ratio*prev_trace/out_trace)
                trace_scale[~np.isfinite(trace_scale)] = 1.0
                trace_scale[out_trace <= 0] = 1.0
                trace_scale[prev_trace <= 0] = 1.0
                output = output*trace_scale[:,np.newaxis,np.newaxis]

        return _cap_drive_coherence(output, self.max_drive_coherence)
