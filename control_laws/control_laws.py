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
    return rcond, max_drive_coherence, startup_cap_db, Kp, Ki, resonance_sensitivity, max_step_db


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
         self.Kp, self.Ki, self.resonance_sensitivity,
         self.max_step_db) = _parse_match_trace_pi_parameters(extra_parameters)
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
        if last_output_cpsd is None:
            # Same startup guard as match_trace_pseudoinverse: no prior
            # measured response to correct against yet, so this is a raw,
            # otherwise-unguarded pseudoinverse solve off whatever FRF
            # estimate is available so far -- clamp its trace to
            # startup_test_level_cap_db dB relative to the specification.
            tf_pinv = np.linalg.pinv(transfer_function, self.rcond)
            output = tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
            spec_trace = np.real(trace(self.specification))
            output_trace = np.real(trace(output))
            max_power_ratio = 10.0**(self.startup_test_level_cap_db/10.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.minimum(1.0, max_power_ratio*spec_trace/output_trace)
            scale[~np.isfinite(scale)] = 1.0
            scale[output_trace <= 0] = 1.0
            output = output*scale[:,np.newaxis,np.newaxis]
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
            output = last_output_cpsd*np.exp(correction)[:,np.newaxis,np.newaxis]
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
            finite_positive_spec = spec_trace[np.isfinite(spec_trace) & (spec_trace > 0)]
            spec_scale = np.max(finite_positive_spec) if finite_positive_spec.size else 0.0
            if spec_scale > 0:
                floor = 1e-8*spec_scale
                output_trace = np.real(trace(output))
                needs_floor = (~unspecified) & (output_trace < floor)
                if np.any(needs_floor):
                    scale = np.ones_like(output_trace)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        scale[needs_floor] = floor/np.maximum(output_trace[needs_floor], 1e-300)
                    scale[~np.isfinite(scale)] = 1.0
                    output = output*scale[:,np.newaxis,np.newaxis]
        return _cap_drive_coherence(output, self.max_drive_coherence)


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
