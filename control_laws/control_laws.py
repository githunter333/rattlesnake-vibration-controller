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
    return rcond, max_drive_coherence, startup_cap_db

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
    rcond, max_drive_coherence = _parse_rcond_and_cap(extra_parameters)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    output = _cap_drive_coherence(output, max_drive_coherence)
    return output

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
    rcond, max_drive_coherence, startup_test_level_cap_db = _parse_match_trace_parameters(extra_parameters)
    # If it's the first time through, do the actual control
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
        spec_trace = np.real(trace(specification))
        output_trace = np.real(trace(output))
        max_power_ratio = 10.0**(startup_test_level_cap_db/10.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(1.0, max_power_ratio*spec_trace/output_trace)
        scale[~np.isfinite(scale)] = 1.0
        scale[output_trace <= 0] = 1.0
        output = output*scale[:,np.newaxis,np.newaxis]
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        output =  last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
    # Note: a uniform per-bin real scalar (the trace_ratio branch, and the
    # startup guard above) leaves pairwise coherence ratios unchanged, so
    # this cap is only ever "doing work" on the raw pseudoinverse itself --
    # but it's applied unconditionally here too so a mid-test law switch
    # (or any other path that reaches this point with an uncapped
    # last_output_cpsd) can't silently skip the cap.
    output = _cap_drive_coherence(output, max_drive_coherence)
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
        Format: 'rcond', 'rcond,max_drive_coherence', or
        'rcond,max_drive_coherence,startup_test_level_cap_db' (shared with
        match_trace_pseudoinverse -- see _parse_match_trace_parameters). The
        third value caps only the very first control command at
        startup_test_level_cap_db dB relative to the specification (0 dB =
        full spec-match); it defaults to -9.0 dB if omitted.
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
    rcond, max_drive_coherence, startup_test_level_cap_db = _parse_match_trace_parameters(extra_parameters)
    # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)
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
        spec_trace = np.real(trace(specification))
        output_trace = np.real(trace(output))
        max_power_ratio = 10.0**(startup_test_level_cap_db/10.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(1.0, max_power_ratio*spec_trace/output_trace)
        scale[~np.isfinite(scale)] = 1.0
        scale[output_trace <= 0] = 1.0
        output = output*scale[:,np.newaxis,np.newaxis]
    return _cap_drive_coherence(output, max_drive_coherence)

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
