"""
Band-Average Control Law for the Rattlesnake Vibration Controller, 2026-11-04.

A new version of this law gets a new date in its file name.

This version chooses the regularization threshold kappa band by band, and its
sweep scores the drive the law outputs, PSD re-fit included. By default
(kappa_rule 'front') the bands are chosen together, so drive goes to the bands
where it lowers the error most; see kappa_rule below. A single kappa for every
band, the knee of the whole spectrum's mean error, was set on the RC plate under
the BARC specification by its worst-conditioned bands (3162) and spent drive
energy in every other band for no gain.

Solves the MIMO random-vibration inverse in fractional-octave (band) space
rather than line-by-line: for each band it finds a single flat-within-band
correlated drive CPSD whose band-averaged response matches the band-averaged
specification, using the band cross-power sensitivity
    C[p, i, j, b] = < H[p,i] * conj(H[p,j]) >_b ,
which is exact for flat-within-band drives:
    < G_xx[p, p] >_b = sum_{i,j} C[p,i,j,b] * P_b[i,j].
The per-band minimum-norm solution is the least-energy drive that meets the
band-averaged specification, and because the FRF is averaged over each band
BEFORE it is inverted, identification noise is not amplified.

The octave band is a DEFINED frequency band, whole. Bands are ANSI S1.11
base-ten (ratio 10**0.3, anchored at 1 kHz); each narrowband line owns a
df-wide bin, and a band's value is the integrated bin content over the
band's FULL width -- so a band the specification only partly covers reads
low, and the value does not depend on the analysis grid. This matches the
visualdynamics octave definition (core.octave.resample) to machine
precision. The EXCITATION, however, is confined to the specification's own
bandwidth: a line where no control channel specifies energy gets no drive.
A partially covered edge band still receives its whole band-energy target,
delivered inside the covered lines, which is the same thing as matching the
specification's own level over the part of the band it actually covers.

Regularization is automatic. At the first solve the law sweeps its
scale-invariant condition-number threshold kappa over 1 to 100 000 (10 to 100 000
for kappa_rule 'knee'), runs the solve the law outputs at each one, PSD re-fit
included, and predicts each band's control error and drive energy through its own
band-space model. It freezes one kappa per band by kappa_rule. The sweep takes a few
seconds, about 9 s at 8193 lines with 48 controls and 12 drives; it never runs inside
the control loop, and the selection is printed to the run log. ``drive_reduction``
gives up a chosen fraction of the selected solution's predicted drive energy by
regularizing harder, at a known compliance cost, when shakers or the article must
be protected. ``condition_number`` fixes kappa instead, and ``regularization`` can
turn it off ('none').

Drive RMS limit (``drive_rms_limit``, default 1.5 V RMS per drive). The RMS
of every drive is known from the drive CPSD before it is output. When a drive
is over its limit the law steps down the sweep's error-energy front (per-band
kappa values), scales any drive still over to its limit (keeping coherence and
phase), and outputs the point with the smallest predicted band error: a drive
that is large only in its weak directions is met by a lower point on the front,
a real shortfall by scaling. The limit is met in the solve and the signal is never clipped. It also
protects the loop: with no regularization, the per-channel gain correction can
ask the solve for a target it can only reach through near-zero singular
directions, and the limit stops that drive before it is output. Set it to the
amplifier or shaker rating; 'none' turns it off.

Closed-loop transfer-function updates (``use_tf_update``). The controller
re-estimates the FRF during control as H1 = G_xf G_ff^g from the measured
drive CPSD G_ff. A correlated minimum-norm drive is deliberately near-rank-1
within each band, so G_ff would be near-singular and the estimate garbage in
exactly the directions the solve needs. The law therefore CONDITIONS every
drive it emits: a per-band diagonal load d_b I with
    d_b = max( lambda_max(P_b)/kappa_d , alpha tr(P_b)/n_drive ),
capped at half the spec-feasible level, bounds the drive CPSD condition number
at kappa_d + 1 and floors every drive's autopower; because the load's band
response is exactly d_b a_b with a_b[p] = sum_i C[p,i,i,b], the correlated
part is re-solved against the deflated target spec_b - d_b a_b, so compliance
is unaffected. The re-solved drive can be larger than the first one, so d_b
is re-sized from it and the re-solve repeated until d_b stops growing. Where
the cap binds, the bound does not hold; ``hold_capped_bands`` leaves those
bands out of the FRF update. Incoming transfer functions are accepted only when they are
finite and the controller's CPSD buffer is full (frames >= total_frames), and
accepted updates are blended cumulatively, each weighted as one new frame of
information, so the sensitivity keeps improving and never forgets the system
identification. While updates are rejected, the last commanded drive is held.
With ``min_coherence`` set, a control channel's lines whose multiple coherence
(the controller's running estimate) is below it are left out of that channel's
rows of the update, the rest standing in for the band; a row with less than
half its band's width coherent keeps its previous value. Each accepted update
prints one line: the commanded and the measured drive CPSD condition number
(the measured one per line, median over each band's lines; the median band and
the worst band), the bands commanded over the kappa_d + 1 bound, the bands
where the load was capped, and, with the gate or the hold on, what they held.

Response limiting (``limit_channel_indices``). Extra response channels can ride
along in the profile as additional "control" channels -- their specification,
FRF, and streamed data all flow through Rattlesnake's existing plumbing with no
controller changes. ``limit_channel_indices`` names which of those channels are
EXCLUDED from the control solution and treated as narrowband response limits
instead: the named channels' specification diagonals become limit ASDs (NaN or
<= 0 at a line means no limit at that line), and the drive CPSD is scaled down,
all drives together, at every frequency line where a limit would be exceeded.
Limiting is prediction + measured trim:
  * PREDICTION -- at every solve, each limit channel's response is predicted
    through the FRF (y_l = h_l P h_l^H per line) and the drive is scaled where
    y exceeds the limit, so an over-limit drive is never commanded. In closed
    loop the limit rows of the FRF are blended with the same weight as the band
    sensitivity, so the prediction tracks the structure as measured.
  * MEASURED TRIM -- every frame, the measured limit-channel ASD (Rattlesnake
    provides the measured control CPSD each frame, open- and closed-loop) is
    compared to the limit; where it exceeds, a persistent per-line trim steps
    down by half the exceedance in dB (damped, because the measurement is a
    lagged multi-frame average), and where it is back under, the trim relaxes
    up by 0.1 dB per frame toward 1. The trim rides on top of the solved drive.
Warning/abort tolerances: the law accepts ``warning_levels``/``abort_levels``
for interface compatibility and ignores them for EVERY channel, limit channels
included (tolerances are evaluation-side, not control-side). They need not be
defined for limit channels -- Rattlesnake normalizes missing tolerance keys to
all-NaN, and NaN tolerances never warn or abort.

Controlling to the cross terms (``control_to_cross_terms``). By default the law
targets only the specification's diagonal (the band-averaged ASDs); the spec's
cross terms -- the coherence and phase it prescribes between control channels --
are not targeted, and the response cross-spectra land wherever the minimum-norm
solution puts them. With the option on, every channel pair whose specification
cross-spectrum is nonzero anywhere contributes one extra complex row per band,
    < G_xx[p, q] >_b = sum_{i,j} < H[p,i] * conj(H[q,j]) >_b * P_b[i,j],
with the target being the complex band average of the spec's cross term (one
coherence-and-phase per band). The augmented system is generally OVERDETERMINED
(each pair adds two real constraints), so the solve becomes a least-squares
compromise with ASD and cross rows equally weighted: the cross terms are
approached, not met, and the ASD fit degrades in exchange. The automatic knee
sweep still scores candidates on the ASD rows only. Expect a slower solve (the
per-band system gains one row per pair) and, on a noisy identification, better
cross-term PHASING but not necessarily better cross-term magnitudes -- turn it
on only when the relative phasing between points is itself a requirement.

Buzz cross terms (``use_buzz``). In practice the cross terms a MIMO test controls to
are often taken from the system identification (the "buzz test") instead of the
specification: the specification's ASDs are kept, and the coherence and phase
between every pair of control channels are those of the response CPSD measured
during system identification -- Rattlesnake's ``buzz_control`` and forcefinder's
``use_buzz`` do the same for their per-line inverses. With ``use_buzz:True`` the law
builds that modified specification from the ``sysid_response_cpsd`` Rattlesnake
passes it and controls to its cross terms as above, over every channel pair
(``control_to_cross_terms`` is implied).

Pure numpy; no other imports, so it runs on any machine. The same file works on
Rattlesnake's main and dev branches, which call a control law the same way: select
``band_average_generator`` (generator) or the class ``band_average`` (class); both
give the same drive frame for frame. The defaults are the configuration the paper
recommends (automatic threshold, constrained band fit, gain correction, channel
check); add ``use_tf_update:True`` and turn on the controller's transfer-function
updates when the article may change during the test.

extra_parameters (newline-delimited "name:value"):
    df                     : float, REQUIRED. Frequency resolution [Hz].
    oct_order              : int, default 6. 1/N-octave bands (6 = 1/6-octave).
    kappa_rule             : 'front' (default) | 'knee' | 'lcurve'. How each band's
                             kappa is chosen from the sweep. 'knee': the smallest
                             kappa within 0.1 dB of the band's best error. 'front':
                             the bands are chosen together -- each band takes the
                             kappa minimizing n_b err_b + mu energy_b, one mu for
                             all, which traces the least drive energy for each
                             mean error; the knee of that front (least energy
                             within 0.1 dB of its best error) is frozen. 'lcurve':
                             the corner of each band's L-curve (log residual
                             against log drive size, greatest curvature), as
                             forcefinder's auto_tikhonov_by_l_curve does per line.
    drive_reduction        : float, default 0. Fraction of the automatic knee
                             solution's predicted drive energy to give up by
                             regularizing harder (e.g. 0.3 = spend 30% less),
                             at a known compliance cost.
    use_tf_update          : bool, default False. Re-solve from the
                             controller's running FRF estimate whenever a
                             trustworthy update arrives (guarded, blended,
                             conditioned -- see above).
    drive_condition_number : float, default 100. kappa_d: per-band bound on
                             the emitted drive CPSD condition number
                             (cond <= kappa_d + 1).
    min_drive_fraction     : float, default 0.02. alpha: minimum per-drive
                             autopower as a fraction of the band-average
                             drive power.
    hold_capped_bands      : bool, default False. Leave a band out of the FRF
                             update when its conditioning load hit the cap in
                             the last solve (its drive is then not bounded at
                             kappa_d + 1); the band keeps its previous
                             sensitivity.
    min_coherence          : float, optional (off by default). Multiple
                             coherence below which a channel's line is left out
                             of the FRF update (see above).
    tf_update_blend        : 'cumulative' (default) | float. 'cumulative'
                             weights each accepted update as one new frame of
                             information (A <- (N A + A_new)/(N+1), N growing
                             from the system-identification average count). A
                             float beta gives exponential forgetting
                             (A <- (1-beta) A + beta A_new) for structures
                             whose FRF drifts during the test.
    n_e_seed               : float, optional. Override the cumulative-blend
                             seed count (the frames of trust the initial
                             sensitivity carries); lower it when the initial
                             identification is distrusted.
    limit_channel_indices  : comma-separated ints, optional. 1-BASED channel
                             numbers (matching the Rattlesnake channel-table
                             rows) to exclude from the control solution and
                             treat as narrowband response limits; their
                             specification diagonals are the limit ASDs (NaN
                             or <= 0 at a line = no limit at that line). See
                             the response-limiting note above.
    condition_number       : float, optional. An explicit threshold kappa; bypasses the
                             automatic knee selection (drive_reduction is then ignored).
    regularization         : 'auto' (default) | 'none'. 'auto' selects the
                             condition-number threshold at the knee (above).
                             'none' solves each band with the plain
                             pseudoinverse -- every singular value above the
                             numerical floor inverted, no Tikhonov damping, no
                             sweep, drive_reduction ignored -- for a test whose
                             drive space is already well conditioned by design,
                             e.g. 6x12 control through an output transformation
                             to the six rigid-body motions.
    constrained_solve      : bool, default True. The per-band fit is a
                             Hermitian matrix that is not always a realizable
                             (PSD) drive; its negative eigenvalues are dropped.
                             Where that moves a channel's predicted band
                             response by more than 0.1 dB, the band is re-fit
                             over PSD drives directly (P = L L^H, minimum-norm
                             Gauss-Newton, a few ms per solve). Matters most
                             through an output transformation (6x12), where
                             projecting overshot bands by up to +21 dB; False
                             restores fit-then-project. Not applied with
                             control_to_cross_terms.
    gain_correction        : bool, default True. Per-band, per-channel error
                             feedback from the measured control responses:
                             each frame, the measured band response over the
                             model's prediction from the measured drive (same
                             CPSD window) is the model's error; its inverse,
                             smoothed and capped at +/-6 dB, multiplies that
                             band's target, and the band is re-solved when a
                             multiplier moves > 0.05 dB. Works open or closed
                             loop; corrects level within a few frames where
                             the FRF update corrects the model slowly. Leans
                             on a good system identification: it adjusts
                             levels, never the model. False restores the
                             hold-the-drive open loop.
    gain_correction_rate   : float, default 0.25. Exponential smoothing rate
                             of the gain correction per frame (1 = no
                             smoothing).
    channel_check_db       : float, default 20; 'none' turns it off. Each
                             frame, every control channel's measured response
                             over the whole specification band is compared to
                             the model's prediction from the measured drive; a
                             channel more than this many dB below or above its
                             prediction (a failed, disconnected or saturated
                             gauge) is left out of both feedback paths while it
                             stays that way: its gain-correction multipliers go
                             back to 1 and its rows of the band sensitivity keep
                             their previous value instead of taking the new FRF
                             estimate. The channel is still controlled through
                             the model, and a message names it when it is
                             excluded and when it recovers. A healthy channel is
                             never affected, so the drive is unchanged.
    drive_rms_limit        : float or comma-separated floats, default 1.5. RMS
                             limit on each drive [the drive's units, V for a
                             voltage output], one value for all drives or one
                             per drive; 'none' turns it off. Set it to the
                             amplifier or shaker rating. Held in the solve,
                             never by clipping. With kappa_rule 'front' (the
                             default) each point on the sweep's error-energy
                             front, up to the knee or the drive_reduction
                             point, is solved, any drive over its limit is
                             scaled down to it (a per-drive congruence, so
                             coherence and phase are kept), and the point with
                             the smallest predicted band error is output; each
                             frame tries last frame's point and its neighbours
                             (three solves). With the other rules the band
                             solves are repeated over a grid of single
                             thresholds instead. Regularization removes the
                             weak, drive-hungry directions first; scaling
                             spreads a true shortfall. If the knee already fits
                             under the limit, the limit changes nothing. Applies to every solve, including
                             gain-correction re-solves and closed-loop updates.
                             Through an output transformation (6x12) the law
                             sees the virtual drives, so the limit applies to
                             those.
    control_to_cross_terms : bool, default False. Also target the
                             specification's cross terms (coherence and phase
                             between control channels) as a least-squares
                             secondary objective. See the cross-term note
                             above for what it buys and what it costs.
    use_buzz               : bool, default False. Take the cross terms from the
                             system identification's response CPSD (coherence
                             and phase), keeping the specification's ASDs, and
                             control to them over every channel pair. Implies
                             control_to_cross_terms.
"""

import numpy as np


# ---------------------------------------------------------------------------
# module-level helpers (no state)
# ---------------------------------------------------------------------------
def _psd_fit(C, t, P0, max_iter=30, tol_db=1e-3):
    """Fit the band response Re(sum C_p * P) to the targets t over PSD drives P:
    minimize sum_p ((r_p - t_p) / t_p)^2 over the rows with t_p > 0.

    Relative (1/t) weighting so channels at different levels count alike. Where
    no PSD drive meets every target, a linear residual prefers under-driving a
    channel to over-driving one; a residual in dB was tried and spreads the
    shortfall instead, by overshooting (+10 dB bands where this fit has none)
    and scoring no better on band error, bands out of tolerance, or channel RMS
    -- an over-test being the more dangerous failure, the linear fit is kept.
    P is factored as L L^H, so it is PSD by construction and its trace (the
    drive energy) is ||L||_F^2. Damped minimum-norm Gauss-Newton on L: each step
    is dL = J^T (J J^T + mu I)^-1 res, one n_ctrl x n_ctrl solve, and the
    minimum-norm step keeps ||L||, the energy, small. Warm-started from P0;
    returns P0's factor untouched if no step reduces the residual.

    C : (n_ctrl, n, n) band cross-power sensitivity.  t : (n_ctrl,) targets.
    """
    m = t > 0
    C = C[m] / t[m][:, None, None]
    n_rows, n, _ = C.shape
    target = np.ones(n_rows)
    w, V = np.linalg.eigh(0.5 * (P0 + P0.conj().T))
    L = V * np.sqrt(np.clip(w, 0.0, None))
    if not np.any(L):
        L = V * np.sqrt(max(w.max(), 1e-30))
    Cr = C.reshape(n_rows, n * n)
    eye = np.eye(n_rows)

    def response(Lx):
        return np.real(Cr @ (Lx @ Lx.conj().T).reshape(-1))

    r = response(L)
    mu = None
    for _ in range(max_iter):
        res = target - r
        G = C @ L.conj()                          # dr_p = 2 Re sum(dL * G_p)
        J = np.concatenate([2.0 * G.real.reshape(n_rows, -1),
                            -2.0 * G.imag.reshape(n_rows, -1)], axis=1)
        JJ = J @ J.T
        scale = np.trace(JJ) / n_rows
        if mu is None:
            mu = 1e-6 * scale
        cost = res @ res
        while True:
            d = J.T @ np.linalg.solve(JJ + mu * eye, res)
            Ln = L + (d[:n * n] + 1j * d[n * n:]).reshape(n, n)
            rn = response(Ln)
            if (target - rn) @ (target - rn) < cost:
                mu = max(mu / 3.0, 1e-12 * scale)
                break
            mu *= 4.0
            if mu > 1e6 * scale:                  # no descent left: converged or stuck
                return L @ L.conj().T
        moved = np.max(np.abs(10.0 * np.log10(np.clip(rn, 1e-300, None)
                                              / np.clip(r, 1e-300, None))))
        L, r = Ln, rn
        if moved < tol_db:
            break
    return L @ L.conj().T


def _l_curve_corner(t, residual, penalty):
    """Index of the L-curve corner: the point of greatest curvature of
    (log residual, log penalty) along the regularization parameter t = log kappa.
    Kappa rising moves the curve left (residual falls) then up (drive grows), a
    clockwise turn, so the corner is the most negative signed curvature."""
    tiny = np.finfo(float).tiny
    rho = np.log(np.maximum(residual, tiny))
    nu = np.log(np.maximum(penalty, tiny))
    r1, n1 = np.gradient(rho, t), np.gradient(nu, t)
    r2, n2 = np.gradient(r1, t), np.gradient(n1, t)
    curvature = (r1 * n2 - n1 * r2) / np.maximum((r1 ** 2 + n1 ** 2) ** 1.5, tiny)
    return int(np.argmin(curvature))


def nth_octave_freqs(freq, oct_order):
    """Lower/upper edges of the 1/N-octave bands overlapping ``freq`` (ANSI S1.11,
    base-10 octave ratio 10**0.3, 1 kHz reference)."""
    freq = np.asarray(freq, dtype=float).flatten()
    oct_ratio = 10.0 ** 0.3
    f_ref = 1000.0
    bands = np.arange(-200, 201)
    powr = [(2 * bands + 1) / (2 * oct_order), bands / oct_order]
    band_centers = f_ref * oct_ratio ** powr[oct_order % 2]
    band_lb = (band_centers * oct_ratio ** (-1 / (2 * oct_order))).round(12)
    band_ub = (band_centers * oct_ratio ** (1 / (2 * oct_order))).round(12)
    keep = (max(freq.min(), 10.0) < band_ub) * (freq.max() > band_lb)
    return band_lb[keep], band_ub[keep]


class band_average:
    """Band-average (octave-space) control law: automatic knee-finding
    regularization, correlated per-band solves, and conditioned closed-loop
    transfer-function updates.

    This class matches the Rattlesnake Type-2 "Class" interface (``__init__`` /
    ``system_id_update`` / ``control``); deploy it through the module-level
    ``band_average_generator`` (Type 1), which front-loads the band sensitivity
    once and reuses it every frame. Used standalone, construct it and pass
    ``specification=`` (and the rest) by keyword.
    """

    def __init__(self,
                 specification=None,        # (n_freq, n_ctrl, n_ctrl) complex CPSD target
                 warning_levels=None,       # accepted for interface compatibility, unused
                 abort_levels=None,         # accepted for interface compatibility, unused
                 extra_parameters='',       # newline-delimited name:value pairs
                 transfer_function=None,    # (n_freq, n_ctrl, n_drive)
                 noise_response_cpsd=None,
                 noise_reference_cpsd=None,
                 sysid_response_cpsd=None,
                 sysid_reference_cpsd=None,
                 multiple_coherence=None,
                 frames=None,
                 total_frames=None,
                 last_response_cpsd=None,
                 last_output_cpsd=None):
        # --- parse extra_parameters -------------------------------------------------
        p = {}
        for line in extra_parameters.splitlines():
            line = line.strip()
            if ':' in line:
                name, value = line.split(':', 1)
                p[name.strip().lower()] = value.strip()

        def _as_bool(s, default):
            if s is None:
                return default
            return s.strip().lower() in ('1', 'true', 'yes', 'on')

        self.df = float(p['df'])
        self.oct_order = int(p.get('oct_order', 6))
        self.drive_reduction = float(p.get('drive_reduction', 0.0))
        self.kappa_rule = p.get('kappa_rule', 'front').strip().lower()
        if self.kappa_rule not in ('knee', 'front', 'lcurve'):
            raise ValueError(f"kappa_rule must be 'knee', 'front' or 'lcurve', not {self.kappa_rule!r}")
        self.regularization = p.get('regularization', 'auto').strip().lower()
        if self.regularization not in ('auto', 'none'):
            raise ValueError("regularization must be 'auto' or 'none', "
                             f"not {self.regularization!r}")
        self.use_tf_update = _as_bool(p.get('use_tf_update'), False)
        self.drive_cond_number = float(p.get('drive_condition_number', 100.0))
        self.min_drive_fraction = float(p.get('min_drive_fraction', 0.02))
        self.hold_capped_bands = _as_bool(p.get('hold_capped_bands'), False)
        mc = p.get('min_coherence', 'none').strip().lower()
        self.min_coherence = None if mc in ('none', 'off', '') else float(mc)
        blend = p.get('tf_update_blend', 'cumulative').lower()
        self.tf_update_blend = blend if blend == 'cumulative' else float(blend)
        self.n_e_seed = p.get('n_e_seed')
        self.n_e_seed = float(self.n_e_seed) if self.n_e_seed is not None else None
        self.use_buzz = _as_bool(p.get('use_buzz'), False)
        self.control_to_cross_terms = self.use_buzz or _as_bool(p.get('control_to_cross_terms'), False)
        self.constrained_solve = _as_bool(p.get('constrained_solve'), True)
        self.gain_correction = _as_bool(p.get('gain_correction'), True)
        self.gain_correction_rate = float(p.get('gain_correction_rate', 0.25))
        cc = p.get('channel_check_db', '20').strip().lower()
        self.channel_check_db = None if cc in ('none', 'off', 'inf', '') else float(cc)
        rms = p.get('drive_rms_limit', '1.5').strip().lower()
        self.drive_rms_limit = (None if rms in ('none', 'off', 'inf', '')
                                else np.array([float(v) for v in rms.split(',')]))
        self._kappa_rms = None         # threshold the RMS limit last required
        self._front = None             # per-band thresholds along the front, least drive first (RMS limit)
        self._front_i = None           # the point on it the RMS limit last required
        lci = p.get('limit_channel_indices')
        self._limit_idx = (np.array([int(v) - 1 for v in lci.split(',')], dtype=int)
                           if lci else np.array([], dtype=int))     # 1-based in, 0-based inside
        # resolved to the knee at the first solve, then frozen -- unless given explicitly
        self.cond_number = float(p['condition_number']) if 'condition_number' in p else 'auto'
        self.auto_kappa = None         # sweep record, for inspection
        self.band_kappa = None         # per-band thresholds from the automatic sweep

        # --- specification + frequency axis ----------------------------------------
        # limit channels are split out FIRST: their spec diagonals become the limit
        # ASDs (NaN-preserving; NaN never reaches the solve or the emitted drive) and
        # everything downstream sees the control-only specification
        if self._limit_idx.size:
            self._ctrl_idx = np.setdiff1d(np.arange(specification.shape[1]), self._limit_idx)
            self._limit_asd = np.real(specification[:, self._limit_idx, self._limit_idx])
            specification = specification[:, self._ctrl_idx][:, :, self._ctrl_idx]
        self._H_lim = None             # limit-row FRF (n_freq, n_lim, n_drive), blended like the sensitivity
        self._limit_trim = None        # per-line measured trim t(f), lazily initialized to 1
        self.specification = specification
        n_freq = specification.shape[0]
        self.frequencies = self.df * np.arange(n_freq)

        # --- narrow <-> octave transforms (bin-overlap, energy-exact) --------------
        band_lb, band_ub = nth_octave_freqs(
            np.array([self.frequencies[1], self.frequencies[-1]]), self.oct_order)
        self.band_lb, self.band_ub = band_lb, band_ub
        self.n_bands = band_lb.size
        self.band_widths = band_ub - band_lb
        b_of = np.searchsorted(band_lb, self.frequencies, side='right') - 1
        inb = ((b_of >= 0)
               & (self.frequencies < band_ub[np.clip(b_of, 0, band_ub.size - 1)])
               & (self.frequencies >= band_lb[0]))
        self._b_of, self._inb = b_of, inb

        # each narrowband line owns a df-wide bin; a band's value is the integrated
        # bin content over the band's FULL width. A partially covered band therefore
        # reads low, and the value does not depend on the analysis grid -- the same
        # definition as visualdynamics core.octave.resample, to machine precision.
        left = self.frequencies - 0.5 * self.df
        right = self.frequencies + 0.5 * self.df
        self._overlap = np.clip(                      # (n_bands, n_freq): Hz of bin k inside band b
            np.minimum(right[None, :], band_ub[:, None])
            - np.maximum(left[None, :], band_lb[:, None]), 0.0, None)
        self.T_narrow_to_octave = self._overlap / self.band_widths[:, None]
        M = np.zeros((self.n_bands, n_freq))          # flat-in-band broadcast, by bin centre
        M[b_of[inb], np.where(inb)[0]] = 1.0
        self.T_octave_to_narrow = M.T                 # (n_freq, n_bands)

        # octave-band specification ASDs: (n_bands, n_ctrl), via matrix multiply
        spec_asd = np.real(np.diagonal(specification, axis1=1, axis2=2))   # (n_freq, n_ctrl)
        self.spec_asd_octave = self.T_narrow_to_octave @ spec_asd          # (n_bands, n_ctrl)
        # excitation is confined to the specification's own bandwidth: a line where
        # no control channel specifies energy gets NO drive. A partially covered
        # edge band still delivers its whole band-energy target, inside the covered
        # lines, which lands the drive at the specification's own level there.
        self._spec_lines = np.any(spec_asd > 0, axis=1)

        # cross-term control: target pairs = spec off-diagonals that are nonzero anywhere;
        # per-band complex targets are the band averages of the spec's cross terms (one
        # coherence-and-phase per band -- the same operator as the ASDs, on complex values)
        self._cross_pairs = None
        self._cross_spec_octave = None
        self._c_cross_octave = None
        self._control_spec = specification
        if self.control_to_cross_terms and not self.use_buzz:
            self._set_cross_targets(specification)
        elif self.use_buzz and sysid_response_cpsd is not None:
            self._set_buzz(sysid_response_cpsd)

        # --- octave-band cross-power sensitivity (needs the transfer function) -----
        # Built lazily on the first control()/system_id_update() rather than here, so it is
        # computed exactly once even when constructed with a transfer function AND updated
        # with the same one.
        self._sensitivity = None       # list of (U, S, Vh, sel) per band, or None entries
        self._n_drive = None
        self._transfer_function = transfer_function   # remembered; sensitivity built on first use
        # closed-loop state (populated only when use_tf_update is on)
        self._A_bands = None           # per-band raw sensitivity matrices -- the blend state
        self._a_octave = None          # per-band diagonal-load response vectors a_b (n_ctrl,)
        self._last_drive = None        # last emitted drive, held while updates are rejected
        self._gain = None              # (n_bands, n_ctrl) target multipliers (gain correction)
        self._gain_solved = None       # the multipliers the current drive was solved with
        self._excluded = None          # (n_ctrl,) control channels failing the channel check
        self._n_avg = None             # effective average count behind the sensitivity
        self._held_bands = 0
        self._load_capped = None       # (n_bands,) the conditioning load hit its cap in the last solve
        self._held_rows = 0.0          # fraction of rows the coherence gate held in the last update
        self.update_log = []           # one dict per accepted update (the printed diagnostics)
                                       # (cumulative blending; grows by 1 per accepted update)

    # ---------------------------------------------------------------------------
    def _set_cross_targets(self, specification):
        """Cross-term targets: every channel pair whose cross-spectrum is nonzero
        anywhere, as the complex band averages of that cross-spectrum."""
        n_ctrl = specification.shape[1]
        pairs = [(pp, qq) for pp in range(n_ctrl) for qq in range(pp + 1, n_ctrl)
                 if np.any(np.abs(specification[:, pp, qq]) > 0)]
        self._cross_pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        if self._cross_pairs.shape[0]:
            cross_spec = specification[:, self._cross_pairs[:, 0], self._cross_pairs[:, 1]]
            self._cross_spec_octave = self.T_narrow_to_octave @ cross_spec   # (n_bands, n_pairs) complex
            self._c_cross_octave = [None] * self.n_bands
        else:
            self._cross_pairs = None

    def _set_buzz(self, sysid_response_cpsd):
        """The specification's ASDs with the coherence and phase of the system
        identification's response CPSD (the buzz test), as Rattlesnake's
        buzz_control builds it; its cross terms become the cross-term targets."""
        spec = self._control_spec
        buzz = np.asarray(sysid_response_cpsd)
        if self._limit_idx.size:
            buzz = buzz[:, self._ctrl_idx][:, :, self._ctrl_idx]
        asd = np.real(np.diagonal(spec, axis1=1, axis2=2))
        bd = np.real(np.diagonal(buzz, axis1=1, axis2=2))
        den = bd[:, :, None] * bd[:, None, :]
        coh = np.divide(np.abs(buzz) ** 2, den, out=np.zeros(buzz.shape), where=den > 0)
        modified = np.exp(1j * np.angle(buzz)) * np.sqrt(coh * asd[:, :, None] * asd[:, None, :])
        self._set_cross_targets(modified)

    def _build_sensitivity(self, transfer_function, blend=False, coherence=None):
        """Build and SVD-factor the per-band cross-power sensitivity from the
        transfer function ``H`` of shape (n_freq, n_ctrl, n_drive).

        With ``blend=True`` (closed-loop transfer-function updates) each band's
        new sensitivity is averaged into the previous one before factoring.
        The default 'cumulative' mode weights the update as one new frame of
        information, A <- (N A_prev + A_new)/(N + 1) with N the effective
        average count (consecutive controller estimates share all but one frame
        of their buffer, so one frame is exactly what each update adds); the
        estimate keeps improving on a stationary structure and never forgets
        the system-identification data.  A float ``tf_update_blend`` gives
        exponential forgetting instead, A <- (1 - beta) A_prev + beta A_new.
        The diagonal-load response vector a_b (the band response of a unit
        uncorrelated drive) is cached from the blended sensitivity so the
        conditioning compensation matches the solve.  ``coherence`` (n_freq,
        n_ctrl), with ``min_coherence`` set, gates the update's lines per row.
        """
        H = np.asarray(transfer_function)
        do_blend = blend and self.use_tf_update and self._A_bands is not None
        if do_blend:
            if self.tf_update_blend == 'cumulative':
                if self._n_avg is None:
                    self._n_avg = 50.0
                w = 1.0 / (self._n_avg + 1.0)
                self._n_avg += 1.0
            else:
                w = self.tf_update_blend
        if self._limit_idx.size:                     # limit rows feed the prediction, not the solve
            H_lim = H[:, self._limit_idx, :]
            self._H_lim = ((1.0 - w) * self._H_lim + w * H_lim
                           if do_blend and self._H_lim is not None else H_lim)
            H = H[:, self._ctrl_idx, :]
        gate = do_blend and coherence is not None and self.min_coherence is not None
        if gate:
            coherence = np.real(np.asarray(coherence))
            if self._limit_idx.size:
                coherence = coherence[:, self._ctrl_idx]
            held = total = 0
        n_freq, n_ctrl, n_drive = H.shape
        self._n_drive = n_drive
        if self._A_bands is None:
            self._A_bands = [None] * self.n_bands
        if self._a_octave is None:
            self._a_octave = [None] * self.n_bands
        sens = []
        held_bands = 0
        for b in range(self.n_bands):
            # drive lines: bin centre in this band AND inside the spec's bandwidth
            sel = self._inb & (self._b_of == b) & self._spec_lines
            if not sel.any():
                sens.append(None)
                continue
            wk = self._overlap[b, sel]                            # Hz each drive line's bin covers
            Wb = self.band_widths[b]
            # cross-power C[p, i, j] = integral of H[p,i] conj(H[p,j]) over the
            # drive lines' bin content, over the band's FULL width -- so the solve
            # matches whole-band energy while emitting only inside the spec band
            Hp = np.moveaxis(H[sel], 0, -1)                       # (n_ctrl, n_drive, n_lines)
            if gate:
                ok = (coherence[sel] >= self.min_coherence).T     # (n_ctrl, n_lines)
                wp = wk * ok                                      # each row's coherent lines
                frac = wp.sum(axis=1) / wk.sum()
                wp = wp / np.where(frac > 0, frac, 1.0)[:, None]  # stand in for the whole band
            else:
                wp = np.broadcast_to(wk, (n_ctrl, wk.size))
            Cb = ((Hp * wp[:, None, :]) @ np.conj(np.swapaxes(Hp, -1, -2))) / Wb   # (n_ctrl, n_drive, n_drive)
            A = Cb.reshape(n_ctrl, n_drive * n_drive)             # (n_ctrl, n_drive**2)
            if self._cross_pairs is not None:
                # augment with one complex row per targeted response pair:
                # C_pq[i,j] = the band-integrated cross sensitivity, same weighting
                i0, i1 = self._cross_pairs[:, 0], self._cross_pairs[:, 1]
                Hq = Hp[i1]                                       # (n_pairs, n_drive, n_lines)
                Hpp = Hp[i0]
                if gate:
                    wx = wk * ok[i0] * ok[i1]
                    fx = wx.sum(axis=1) / wk.sum()
                    wx = wx / np.where(fx > 0, fx, 1.0)[:, None]
                else:
                    wx = np.broadcast_to(wk, (i0.size, wk.size))
                Cx = ((Hpp * wx[:, None, :]) @ np.conj(np.swapaxes(Hq, -1, -2))) / Wb   # (n_pairs, n_drive, n_drive)
                A = np.concatenate([A, Cx.reshape(self._cross_pairs.shape[0], n_drive * n_drive)], axis=0)
            if (do_blend and self.hold_capped_bands and self._load_capped is not None
                    and self._load_capped[b] and self._A_bands[b] is not None):
                A = self._A_bands[b]                               # drive not conditioned: hold the band
                held_bands += 1
            elif do_blend and self._A_bands[b] is not None:
                A_prev = self._A_bands[b]
                A = (1.0 - w) * A_prev + w * A
                keep = np.zeros(A.shape[0], dtype=bool)
                if self._excluded is not None and self._excluded.any():
                    keep[:n_ctrl] = self._excluded                 # a failed channel's rows keep their value
                    if self._cross_pairs is not None:
                        keep[n_ctrl:] = np.isin(self._cross_pairs, np.flatnonzero(self._excluded)).any(axis=1)
                if gate:                                           # too little coherent: keep the old row
                    keep[:n_ctrl] |= frac < 0.5
                    if self._cross_pairs is not None:
                        keep[n_ctrl:] |= fx < 0.5
                    held += int(keep.sum())
                    total += keep.size
                A[keep] = A_prev[keep]
            self._A_bands[b] = A
            self._a_octave[b] = np.real(                          # a_b[p] = sum_i C[p,i,i,b]
                np.einsum('pii->p', A[:n_ctrl].reshape(n_ctrl, n_drive, n_drive)))
            if self._cross_pairs is not None:
                # the diagonal load's response on the cross rows: c_b[pq] = sum_i C_pq[i,i]
                self._c_cross_octave[b] = np.einsum(
                    'kii->k', A[n_ctrl:].reshape(-1, n_drive, n_drive))
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            sens.append((U, S, Vh, sel))
        self._sensitivity = sens
        self._held_bands = held_bands
        if gate:
            self._held_rows = held / max(total, 1)

    def _resolve_auto_kappa(self):
        """Front-loaded automatic regularization, one kappa per band.

        Sweep kappa over the studied range; at each candidate, solve every band
        (PSD re-fit included) and record each band's predicted mean |dB| error
        over its specified channels and its drive energy, through the band-space
        forward model r_b = A_b vec(P_b). A band's solve depends on its own kappa
        alone, so the bands are chosen independently. Freeze, per band:

        * the KNEE: the smallest kappa whose predicted error in that band is
          within 0.1 dB of the band's best over the sweep;
        * with ``drive_reduction`` r > 0, the largest kappa at or below the band's
          knee whose energy is at most (1 - r) times the knee's.

        The frozen thresholds are reused by every subsequent re-solve; the sweep
        never runs inside the loop. The record is kept on ``self.auto_kappa``.
        """
        # the L-curve and the front need the whole span, from the strongest regularization
        # (kappa 1, the band's largest singular value) up, as forcefinder's search runs from its
        # largest singular value down
        wide = self.kappa_rule != 'knee'
        grid = np.logspace(0.0 if wide else 1.0, 5.0, 81 if wide else 65)
        n_bands = len(self._sensitivity)
        err = np.full((grid.size, n_bands), np.nan)
        energy = np.zeros((grid.size, n_bands))
        residual = np.zeros((grid.size, n_bands))
        penalty = np.zeros((grid.size, n_bands))
        power = np.zeros((grid.size, n_bands, self._n_drive))   # each drive's power in each band
        solved = [None] * grid.size                              # every band's drive at each kappa
        for k, kappa in enumerate(grid):
            output = self._solve_bands(kappa=kappa)
            solved[k] = output
            for b, entry in enumerate(self._sensitivity):
                if entry is None:
                    continue
                sel = entry[3]
                spec_b = self.spec_asd_octave[b]
                Pb = output[sel][0]
                r_b = np.real(self._A_bands[b][:spec_b.size] @ Pb.reshape(-1))   # ASD rows only
                m = spec_b > 0
                if m.any():
                    ratio = np.clip(r_b[m] / spec_b[m], 1e-6, None)
                    err[k, b] = np.abs(10.0 * np.log10(ratio)).mean()
                energy[k, b] = np.real(np.trace(Pb)) * self._overlap[b, sel].sum()
                residual[k, b] = np.sum((r_b - spec_b) ** 2)
                penalty[k, b] = np.sum(np.abs(Pb) ** 2)
                power[k, b] = np.real(np.diag(Pb)) * sel.sum() * self.df
        scored = ~np.isnan(err).all(axis=0)
        if not scored.any():
            self.cond_number = 300.0
            return
        knee = np.zeros(n_bands, dtype=int)
        chosen = np.zeros(n_bands, dtype=int)
        if self.kappa_rule == 'front':
            b_ok = np.flatnonzero(scored)
            n_b = np.array([(self.spec_asd_octave[b] > 0).sum() for b in b_ok])
            e_b = np.nan_to_num(err[:, b_ok], nan=np.inf)
            w_b = n_b * e_b
            # one multiplier mu for every band: each band then spends drive where it buys the most
            # error, and sweeping mu traces the least total energy for each mean error
            mu_ref = np.sum(w_b[0][np.isfinite(w_b[0])]) / max(np.sum(energy[-1, b_ok]), 1e-300)
            picks, errs, energies = [], [], []
            for mu in mu_ref * np.logspace(-6.0, 6.0, 241):
                idx = np.argmin(w_b + mu * energy[:, b_ok], axis=0)
                picks.append(idx)
                errs.append(np.sum(w_b[idx, np.arange(b_ok.size)]) / n_b.sum())
                energies.append(np.sum(energy[idx, b_ok]))
            errs, energies = np.array(errs), np.array(energies)
            ok = errs <= errs.min() + 0.1
            front_i = int(np.flatnonzero(ok)[np.argmin(energies[ok])])
            sel_i = front_i
            if self.drive_reduction > 0:
                cheaper = np.flatnonzero(energies <= (1.0 - self.drive_reduction) * energies[front_i])
                if cheaper.size:
                    sel_i = int(cheaper[np.argmin(errs[cheaper])])
            if self.drive_rms_limit is not None:
                # every front point up to the selection, least drive first; each is scaled to the
                # limit (only the drives over it) and the one with the least predicted error is used
                front_k = np.array(picks)                                  # (n_mu, n_ok)
                below = np.flatnonzero(energies <= energies[sel_i])
                below = below[np.argsort(energies[below], kind='stable')]
                front, pred = [], []
                for i in below:
                    v = np.full(n_bands, grid[0])
                    v[b_ok] = grid[front_k[i]]
                    if front and np.array_equal(v, front[-1]):
                        continue
                    front.append(v)
                    out = np.zeros_like(solved[0])
                    for jb, b in enumerate(b_ok):
                        sel = self._sensitivity[b][3]
                        out[sel] = solved[front_k[i, jb]][sel]
                    pred.append(self._predicted_error(self._limit_scaled(out)[0]))
                self._front = front
                self._front_i = int(np.argmin(pred))
                chosen_v = front[self._front_i]
            knee[b_ok] = picks[front_i]
            chosen[b_ok] = picks[sel_i]
            if self._front is not None:
                chosen[b_ok] = np.searchsorted(grid, chosen_v[b_ok])
        else:
            for b in np.flatnonzero(scored):
                if self.kappa_rule == 'lcurve':
                    knee[b] = _l_curve_corner(np.log(grid), residual[:, b], penalty[:, b])
                else:
                    knee[b] = int(np.argmax(err[:, b] <= np.nanmin(err[:, b]) + 0.1))
                chosen[b] = knee[b]
                if self.drive_reduction > 0:
                    below = [j for j in range(knee[b] + 1) if energy[j, b] <= (1.0 - self.drive_reduction) * energy[knee[b], b]]
                    chosen[b] = below[-1] if below else 0
        self.band_kappa = grid[chosen]
        self.cond_number = float(self.band_kappa[scored].max())   # the hardest band, for the RMS-limit grid
        self.auto_kappa = {'grid': grid, 'error_db': err, 'energy': energy, 'residual': residual,
                           'penalty': penalty, 'knee': grid[knee], 'selected': self.band_kappa,
                           'scored': scored}
        kb = self.band_kappa[scored]
        print(f"band_average: condition_number auto per band ({self.kappa_rule}) -> median {np.median(kb):.0f}, "
              f"range {kb.min():.0f}..{kb.max():.0f} over {kb.size} bands "
              f"(drive_reduction {self.drive_reduction:g})")

    def _diag_load(self, lmax, tr, spec_b, a_b):
        """Per-band diagonal load d_b for closed-loop drive conditioning:
        bounds the drive CPSD condition number (lambda_max / kappa_d) and floors
        every drive's autopower (alpha * band-average drive power), capped at
        half the spec-feasible level so the load alone cannot over-drive the
        specification in a quiet band."""
        d = max(lmax / self.drive_cond_number,
                self.min_drive_fraction * tr / self._n_drive)
        capped = False
        m = (spec_b > 0) & (a_b > 0)
        if m.any():
            cap = 0.5 * np.min(spec_b[m] / a_b[m])
            capped = cap < d
            d = min(d, cap)
        return d, capped

    def _log_conditioning(self, drive_cpsd):
        """Prints and records how well conditioned the drive was for identification: the
        commanded drive (what the load bounds) and the measured drive CPSD, the controller's
        same window as the update just taken. The measured one is a 20-frame estimate, and a
        12 x 12 CPSD estimated from 20 averages is far worse conditioned than the one it estimates."""
        x = np.asarray(drive_cpsd)
        med = np.full(len(self._sensitivity), np.nan)
        cmd = np.full(len(self._sensitivity), np.nan)
        for b, entry in enumerate(self._sensitivity):
            if entry is None:
                continue
            sel = entry[-1]
            ev = np.linalg.eigvalsh(x[sel])                       # (n_lines, n_drive), ascending
            ec = np.linalg.eigvalsh(self._last_drive[np.flatnonzero(sel)[0]])   # flat in the band
            with np.errstate(divide='ignore', invalid='ignore'):
                med[b] = np.median(ev[:, -1] / np.maximum(ev[:, 0], 1e-300 * ev[:, -1].max()))
                cmd[b] = ec[-1] / max(ec[0], 1e-300 * ec[-1])
        ok = np.isfinite(med)
        worst = int(np.nanargmax(med))
        rec = {'update': len(self.update_log) + 1, 'n_avg': self._n_avg,
               'commanded_median': float(np.median(cmd[ok])), 'commanded_worst': float(np.max(cmd[ok])),
               'cond_median': float(np.median(med[ok])), 'cond_worst': float(med[worst]),
               'worst_band_hz': float(np.sqrt(self.band_lb[worst] * self.band_ub[worst])),
               'over_bound': int((cmd[ok] > self.drive_cond_number + 1).sum()), 'bands': int(ok.sum()),
               'load_capped': int(self._load_capped.sum()), 'held_rows': self._held_rows,
               'held_bands': self._held_bands}
        self.update_log.append(rec)
        print(f"band_average: update {rec['update']}: commanded drive cond median "
              f"{rec['commanded_median']:.0f}, worst {rec['commanded_worst']:.0f}; measured median "
              f"{rec['cond_median']:.0f}, worst {rec['cond_worst']:.0f} at {rec['worst_band_hz']:.0f} Hz; "
              f"{rec['over_bound']}/{rec['bands']} bands commanded over {self.drive_cond_number + 1:g}, "
              f"load capped in {rec['load_capped']}"
              + (f", held {rec['held_bands']} capped bands" if self.hold_capped_bands else '')
              + (f", coherence gate held {100 * rec['held_rows']:.0f} % of rows"
                 if self.min_coherence is not None else ''))

    def _apply_limits(self, output):
        """Prediction-based response limiting: scale the drive CPSD down, all
        drives together, at every line where a limit channel's predicted
        response y_l = h_l P h_l^H exceeds its limit ASD. NaN or <= 0 limit
        values constrain nothing. Uniform scaling preserves PSD-ness and the
        drive CPSD's condition number, so the closed-loop conditioning
        guarantee survives."""
        y = np.real(np.einsum('flj,fjk,flk->fl',
                              self._H_lim, output, np.conj(self._H_lim)))
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where((self._limit_asd > 0) & (y > 0),
                             self._limit_asd / y, np.inf)
        s = np.clip(ratio.min(axis=1), None, 1.0)
        return output * s[:, None, None]

    def _update_trim(self, last_response_cpsd):
        """Measured-response trim: where the measured limit-channel ASD exceeds
        its limit, step the persistent per-line trim down by HALF the exceedance
        in dB -- deliberately damped, because the measurement is a lagged
        multi-frame average that still contains pre-trim frames -- and where it
        is back under the limit, relax the trim up by 0.1 dB per frame toward 1
        so noise-driven over-trim recovers instead of ratcheting to zero."""
        measured = np.real(np.asarray(last_response_cpsd)[:, self._limit_idx, self._limit_idx])
        if self._limit_trim is None:
            self._limit_trim = np.ones(measured.shape[0])
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where((self._limit_asd > 0) & (measured > 0),
                             self._limit_asd / measured, np.inf)
        r = ratio.min(axis=1)
        over = r < 1.0
        self._limit_trim[over] *= np.sqrt(r[over])                # half the exceedance in dB
        self._limit_trim[~over] = np.minimum(                     # +0.1 dB/frame back toward 1
            1.0, self._limit_trim[~over] * 10 ** 0.01)

    def _trimmed(self, drive):
        """The commanded drive: the solved (prediction-limited) drive with the
        measured trim riding on top."""
        if self._limit_trim is not None and np.any(self._limit_trim < 1.0):
            return drive * self._limit_trim[:, None, None]
        return drive

    def _target(self, b):
        """Band b's ASD targets: the specification, times the gain-correction
        multipliers when those are active."""
        if self._gain is None:
            return self.spec_asd_octave[b]
        return self.spec_asd_octave[b] * self._gain[b]

    def _measured_predicted(self, last_response_cpsd, last_output_cpsd):
        """The measured band response of every control channel and the model's
        prediction from the measured drive, both from the controller's same CPSD
        window: two (n_bands, n_ctrl) arrays, zero where a band has no drive."""
        n_ctrl = self.spec_asd_octave.shape[1]
        y = np.real(np.diagonal(np.asarray(last_response_cpsd), axis1=1, axis2=2))
        if self._limit_idx.size:
            y = y[:, self._ctrl_idx]
        measured = self.T_narrow_to_octave @ y[:self.T_narrow_to_octave.shape[1]]
        predicted = np.zeros_like(measured)
        x = np.asarray(last_output_cpsd)
        for b, entry in enumerate(self._sensitivity):
            if entry is None:
                continue
            sel = entry[-1]
            wk = self._overlap[b, sel]
            xbar = np.tensordot(wk, x[sel], axes=(0, 0)) / wk.sum()     # band-mean measured drive
            predicted[b] = np.real(self._A_bands[b][:n_ctrl] @ xbar.reshape(-1))
        return measured, predicted

    def _check_channels(self, measured, predicted):
        """Flag control channels whose measured response over the specification band
        is more than channel_check_db below or above the model's prediction; report
        each change once."""
        n_ctrl = measured.shape[1]
        if self._excluded is None:
            self._excluded = np.zeros(n_ctrl, dtype=bool)
        if self.channel_check_db is None:
            return
        m = self.spec_asd_octave > 0
        meas = np.where(m, measured, 0.0).sum(axis=0)
        pred = np.where(m, predicted, 0.0).sum(axis=0)
        ok = pred > 0
        with np.errstate(divide='ignore'):
            level = np.where(ok, 10.0 * np.log10(np.maximum(meas, 1e-300) / np.where(ok, pred, 1.0)), 0.0)
        bad = ok & (np.abs(level) > self.channel_check_db)
        number = (self._ctrl_idx if self._limit_idx.size else np.arange(n_ctrl)) + 1   # 1-based, as the table
        for p in np.flatnonzero(bad & ~self._excluded):
            print(f"band_average: control channel {number[p]} reads {level[p]:+.0f} dB from its "
                  f"prediction; excluded from the gain correction and the FRF update")
        for p in np.flatnonzero(~bad & self._excluded):
            print(f"band_average: control channel {number[p]} back within "
                  f"{self.channel_check_db:g} dB of its prediction; used again")
        self._excluded = bad

    def _update_gain(self, last_response_cpsd, last_output_cpsd, mp=None):
        """Per-band, per-channel gain correction. The ratio of the MEASURED band
        response to the response the law's model predicts from the MEASURED drive
        (both from the controller's same CPSD window) is the model's error for
        that channel and band. Its inverse, exponentially smoothed at
        gain_correction_rate and capped at +/-6 dB, multiplies that band's target.
        Measured-over-predicted rather than measured-over-spec, so intentional
        drive changes (the limit trim, the conditioning load) are in the
        prediction and never fought, and so the estimate does not depend on the
        correction itself -- no integrator winding up on a lagged measurement.
        Returns True when a multiplier moved enough (> 0.05 dB) to re-solve."""
        n_ctrl = self.spec_asd_octave.shape[1]
        if self._gain is None:
            self._gain = np.ones((self.n_bands, n_ctrl))
            self._gain_solved = self._gain.copy()
        measured, predicted = mp if mp is not None else self._measured_predicted(last_response_cpsd,
                                                                                 last_output_cpsd)
        cap = np.log(10 ** 0.6)
        lg = np.log(self._gain)
        for b, entry in enumerate(self._sensitivity):
            if entry is None:
                continue
            ok = (self.spec_asd_octave[b] > 0) & (predicted[b] > 0) & (measured[b] > 0)
            lg[b, ok] = ((1.0 - self.gain_correction_rate) * lg[b, ok]
                         + self.gain_correction_rate * np.log(predicted[b, ok] / measured[b, ok]))
        if self._excluded is not None and self._excluded.any():
            lg[:, self._excluded] = 0.0                       # a failed channel's targets stay as specified
        self._gain = np.exp(np.clip(lg, -cap, cap))
        moved = np.max(np.abs(10.0 * np.log10(self._gain / self._gain_solved)))
        if moved > 0.05:
            self._gain_solved = self._gain.copy()
            return True
        return False

    def _realizable(self, b, P_fit, P_proj, w):
        """The realizable (PSD) drive for band b. The regularized fit P_fit is
        Hermitian but not always PSD; P_proj drops its negative eigenvalues,
        which is exact when there are none and harmless when they are small.
        Where dropping them moves some channel's predicted band response by more
        than 0.1 dB, re-fit that response over PSD drives instead (_psd_fit):
        the regularization still decides WHAT response is asked for, the
        constraint only removes the unrealizable part. Returns (P, eigenvalues)."""
        if (not self.constrained_solve or self._cross_spec_octave is not None
                or w.min() >= 0.0):
            return P_proj, w
        n_ctrl = self.spec_asd_octave.shape[1]
        A = self._A_bands[b][:n_ctrl]
        r_fit = np.real(A @ P_fit.reshape(-1))
        r_prj = np.real(A @ P_proj.reshape(-1))
        m = r_fit > 0
        if not m.any():
            return P_proj, w
        gap = np.max(np.abs(10.0 * np.log10(np.clip(r_prj[m], 1e-300, None) / r_fit[m])))
        if gap <= 0.1:
            return P_proj, w
        n = P_fit.shape[0]
        # always from this solve's own projection: a warm start from the previous frame's
        # constrained drive let the drive energy ratchet up over repeated re-solves (the
        # fit takes minimum-norm steps from where it starts, not from zero), x1.8 in a
        # streamed run with the gain correction and x4.6 with a noisy identification (2026-09-27)
        P = _psd_fit(A.reshape(n_ctrl, n, n), np.clip(r_fit, 0.0, None), P_proj)
        P = 0.5 * (P + P.conj().T)
        return P, np.linalg.eigvalsh(P)

    def _solve_octave(self):
        """Octave-space solve -> flat-in-band correlated drive CPSD of shape
        (n_freq, n_drive, n_drive).

        When ``use_tf_update`` is on, each band's drive is conditioned for
        trustworthy closed-loop identification: a diagonal load d_b I is added
        (bounding the drive CPSD condition number at kappa_d + 1 and flooring
        every drive's autopower) and the correlated part is re-solved against
        the deflated target spec_b - d_b a_b, so the total band response still
        meets the specification."""
        if self.regularization == 'auto' and self.cond_number == 'auto':
            self._resolve_auto_kappa()                    # once; frozen for all re-solves
        if self.drive_rms_limit is None:
            output = self._solve_bands()
        else:
            output = self._rms_limited()
        if self._limit_idx.size:
            output = self._apply_limits(output)
        return output

    def _drive_rms(self, output):
        return np.sqrt(np.real(np.einsum('fii->i', output)) * self.df)

    def _rms_limited(self):
        """Solve under the per-drive RMS limit (see drive_rms_limit). When the
        configured solve puts a drive over its limit, try a grid of thresholds
        from kappa = 1 up to the configured one; at each, scale any drive still
        over its limit down to it (a per-drive congruence, which keeps
        coherence and phase), predict the band response through the band
        sensitivity, and keep the candidate with the smallest mean band error
        against the current targets. A drive that only needs its weak,
        drive-hungry directions removed is met by regularization alone; a
        specification that needs more drive than allowed is met by scaling;
        most cases are a mix. Searched afresh each time the limit binds, so
        the threshold can rise again when the targets fall (reusing the last
        threshold that held only ever lowered it, and the error grew frame by
        frame under the gain correction)."""
        limit = self.drive_rms_limit
        if self._front is not None:
            return self._rms_limited_front()
        output = self._solve_bands()
        if np.all(self._drive_rms(output) <= limit):
            self._kappa_rms = None
            return output
        if self.regularization == 'auto' or self.cond_number != 'auto':
            hi = float(self.cond_number)
        else:
            hi = 1e8                                     # 'none': above any useful kappa
        grid = list(np.logspace(0.0, np.log10(hi), int(min(11, max(3, 2 * np.ceil(np.log10(hi)) + 1)))))
        if self._kappa_rms is not None and self._kappa_rms not in grid:
            grid.append(self._kappa_rms)                 # last frame's choice competes too
        best = None
        for kappa in grid:
            out, sc = self._limit_scaled(self._solve_bands(kappa))
            e = self._predicted_error(out)
            if best is None or e < best[0]:
                best = (e, kappa, out, sc)
        err, kappa, out, sc = best
        if self._kappa_rms is None or abs(np.log(kappa / self._kappa_rms)) > 0.1 or np.any(sc < 1.0):
            scaled = (f", drives scaled by {np.round(20 * np.log10(sc), 2)} dB"
                      if np.any(sc < 0.999) else "")
            print(f"band_average: drive RMS limit -> condition_number {kappa:.3g}{scaled} "
                  f"(predicted mean band error {err:.2f} dB)")
        self._kappa_rms = kappa
        return out

    def _limit_scaled(self, output):
        """Scale each drive over its RMS limit down to it (a per-drive congruence, so
        coherence and phase are kept); returns the scaled drive and the scale factors."""
        r = self._drive_rms(output)
        sc = np.minimum(1.0, self.drive_rms_limit / np.where(r > 0, r, 1.0))
        return output * sc[None, :, None] * sc[None, None, :], sc

    def _predicted_error(self, output):
        """Mean |dB| band error of a drive against the current targets, predicted
        through the band sensitivity."""
        n_ctrl = self.spec_asd_octave.shape[1]
        e, n = 0.0, 0
        for b, entry in enumerate(self._sensitivity):
            if entry is None:
                continue
            t = self._target(b)
            m = t > 0
            if not m.any():
                continue
            pred = np.real(self._A_bands[b][:n_ctrl] @ output[entry[-1]][0].reshape(-1))
            e += np.abs(10.0 * np.log10(np.clip(pred[m], 1e-300, None) / t[m])).sum()
            n += int(m.sum())
        return e / max(n, 1)

    def _rms_limited_front(self):
        """The RMS limit with the 'front' rule. The sweep traced the least-drive
        front of band error against drive energy (per-band thresholds); each
        point on it is solved, any drive over its limit is scaled down to it,
        and the point with the least predicted band error is output. A frame
        tries last frame's point and its two neighbours (three solves) and moves
        at most one step, so the choice follows the gain correction and the FRF
        updates without a search every frame."""
        n = len(self._front)
        j0 = self._front_i
        best = None
        for j in (j0 - 1, j0, j0 + 1):
            if not 0 <= j < n:
                continue
            out, sc = self._limit_scaled(self._solve_bands(self._front[j]))
            e = self._predicted_error(out)
            if best is None or e < best[0]:
                best = (e, j, out, sc)
        e, j, out, sc = best
        if j != j0 or self._kappa_rms is None:
            scaled = (f", drives scaled by {np.round(20 * np.log10(sc), 2)} dB"
                      if np.any(sc < 0.999) else "")
            print(f"band_average: drive RMS limit -> front point {j + 1} of {n} "
                  f"(predicted band error {e:.2f} dB, drive {np.real(np.einsum('fii->', out)) * self.df:.3g}){scaled}")
        self._front_i, self._kappa_rms = j, j
        return out

    def _solve_bands(self, kappa=None):
        """The per-band solves with the configured regularization, or, when
        kappa is given, with the condition-number threshold kappa."""
        n_freq = self.specification.shape[0]
        n_drive = self._n_drive
        output = np.zeros((n_freq, n_drive, n_drive), dtype=complex)
        self._load_capped = np.zeros(len(self._sensitivity), dtype=bool)
        for b, entry in enumerate(self._sensitivity):
            if entry is None:
                continue
            U, S, Vh, sel = entry
            t_b = self._target(b)
            if self._cross_spec_octave is not None:
                # augmented target: band-averaged ASDs + complex band-averaged cross terms
                t_b = np.concatenate([t_b.astype(complex), self._cross_spec_octave[b]])
            g = U.conj().T @ t_b                              # project target onto left sing. vecs
            if kappa is None:
                kappa_b = self.band_kappa[b] if self.band_kappa is not None else None
            else:
                kappa_b = kappa[b] if np.ndim(kappa) else kappa
            if kappa_b is not None:
                lam = (S.max() / kappa_b) ** 2 if S.size and S.max() > 0 else 0.0
                filt = S / (S ** 2 + lam)
            elif self.regularization == 'none':
                # plain pseudoinverse, numpy's convention: invert every singular value
                # above the numerical floor, drop the rest -- no Tikhonov damping
                floor = np.finfo(float).eps * max(len(U), Vh.shape[1]) * (S.max() if S.size else 0.0)
                filt = np.divide(1.0, S, out=np.zeros_like(S), where=S > floor)
            else:
                lam = (S.max() / self.cond_number) ** 2 if S.size and S.max() > 0 else 0.0
                filt = S / (S ** 2 + lam)
            vecP = Vh.conj().T @ (filt * g)                   # regularized min-norm P_b
            Pb = vecP.reshape(n_drive, n_drive)
            Pb = 0.5 * (Pb + Pb.conj().T)                     # Hermitian
            w, V = np.linalg.eigh(Pb)
            Pfit = Pb
            Pb = (V * np.clip(w.real, 0.0, None)) @ V.conj().T  # nearest PSD (drop negatives)
            if not self.use_tf_update:                    # (closed loop: applied to the re-solve)
                Pb, w = self._realizable(b, Pfit, Pb, w)
            if self.use_tf_update:
                spec_b, a_b = self._target(b), self._a_octave[b]
                wpos = np.clip(w.real, 0.0, None)
                d, self._load_capped[b] = self._diag_load(wpos.max(initial=0.0), wpos.sum(), spec_b, a_b)
                if d > 0:
                    # re-solve against the deflated target (same lambda -- pass-1 filter); the
                    # re-solved drive can be larger than the pass-1 one, so the load is re-sized
                    # from it and the re-solve repeated until the load stops growing
                    for it in range(10):
                        t2 = np.clip(spec_b - d * a_b, 0.0, None)
                        if self._cross_spec_octave is not None:
                            # the load's cross response deflates the complex cross targets (no clip)
                            t2 = np.concatenate([t2.astype(complex),
                                                 self._cross_spec_octave[b] - d * self._c_cross_octave[b]])
                        g2 = U.conj().T @ t2
                        P2 = (Vh.conj().T @ (filt * g2)).reshape(n_drive, n_drive)
                        P2 = 0.5 * (P2 + P2.conj().T)
                        w2, V2 = np.linalg.eigh(P2)
                        Pb, _ = self._realizable(b, P2, (V2 * np.clip(w2.real, 0.0, None)) @ V2.conj().T, w2)
                        e2 = np.clip(np.linalg.eigvalsh(Pb), 0.0, None)
                        d2, capped = self._diag_load(e2.max(initial=0.0), e2.sum(), spec_b, a_b)
                        if d2 <= d * 1.001 or it == 9:
                            break                                 # Pb is the re-solve at this d
                        d, self._load_capped[b] = d2, capped
                    Pb = Pb + d * np.eye(n_drive)             # the conditioning diagonal load
                else:
                    Pb, _ = self._realizable(b, Pfit, Pb, w)
            output[sel] = Pb                                      # flat within the band
        return output

    # ---------------------------------------------------------------------------
    def system_id_update(self,
                         transfer_function=None,
                         noise_response_cpsd=None,
                         noise_reference_cpsd=None,
                         sysid_response_cpsd=None,
                         sysid_reference_cpsd=None,
                         multiple_coherence=None,
                         frames=None,
                         total_frames=None):
        """Rebuild the octave-band cross-power sensitivity when a new transfer
        function arrives.  All other arguments are accepted for interface
        compatibility but unused."""
        if self.use_buzz and self._cross_pairs is None and sysid_response_cpsd is not None:
            self._set_buzz(sysid_response_cpsd)
        if transfer_function is not None:
            self._build_sensitivity(transfer_function)

    def control(self,
                transfer_function=None,
                multiple_coherence=None,
                frames=None,
                total_frames=None,
                last_response_cpsd=None,
                last_output_cpsd=None):
        """Return the next drive CPSD, shape (n_freq, n_drive, n_drive).

        First call: builds the band sensitivity from the system-identification
        transfer function, resolves the automatic regularization (the knee
        sweep, once), and solves the octave-space inverse for a flat-in-band
        correlated drive.

        Subsequent calls:
          * if ``use_tf_update`` and a trustworthy transfer function is supplied
            (all-finite, and the controller's CPSD buffer is full:
            ``frames >= total_frames``) -> blend it into the sensitivity and
            re-solve with a conditioned drive;
          * else -> hold the last commanded drive.
        """
        if self._sensitivity is None:
            if self.use_buzz and self._cross_pairs is None:
                raise RuntimeError('use_buzz needs the system identification response CPSD '
                                   '(sysid_response_cpsd) before control can run.')
            build_tf = transfer_function if transfer_function is not None else self._transfer_function
            if build_tf is None:
                raise RuntimeError(
                    'A transfer function must be provided before control can run.')
            self._build_sensitivity(build_tf)
            if self.n_e_seed is not None:   # explicit trust in the initial sensitivity
                self._n_avg = self.n_e_seed
            elif frames is not None:        # else seed the effective average count from
                self._n_avg = float(frames)  # the system-identification estimate

        if last_output_cpsd is None or last_response_cpsd is None:
            self._last_drive = self._solve_octave()
            return self._last_drive

        if self._limit_idx.size:
            self._update_trim(last_response_cpsd)
        mp = None
        if self.gain_correction or self.use_tf_update:        # both feedback paths use the check
            mp = self._measured_predicted(last_response_cpsd, last_output_cpsd)
            self._check_channels(*mp)
        resolve = (self.gain_correction
                   and self._update_gain(last_response_cpsd, last_output_cpsd, mp))

        if (self.use_tf_update
                and transfer_function is not None
                and (frames is None or total_frames is None
                     or frames >= total_frames)
                and np.all(np.isfinite(transfer_function))):
            self._build_sensitivity(transfer_function, blend=True, coherence=multiple_coherence)
            self._last_drive = self._solve_octave()
            self._log_conditioning(last_output_cpsd)
            return self._trimmed(self._last_drive)

        if resolve:                          # the gain correction moved: re-solve, same model
            self._last_drive = self._solve_octave()
        if self._last_drive is not None:     # hold the COMMANDED drive (not the measured echo)
            return self._trimmed(self._last_drive)
        return last_output_cpsd


def band_average_generator():
    """Rattlesnake Type-1 (generator) control-law entry point -- the stateful, compute-once form.

    Rattlesnake creates the generator, advances it once (``next``), then ``send``s the per-frame
    data tuple each frame and receives the drive CPSD back. Because the generator's locals persist
    across frames, the ``band_average`` law -- and its octave-band cross-power sensitivity -- is
    **built only once** (on the first frame, which carries the system-ID transfer function) and
    reused for every subsequent frame.  Set ``use_tf_update`` in ``extra_parameters`` to re-solve
    from the controller's running FRF estimate; ``frames`` / ``total_frames`` are forwarded so the
    update guard can reject estimates taken before the controller's CPSD buffer is full.
    """
    law = None
    data = yield                                   # next() lands here; first send() supplies data
    while True:
        (specification, warning_levels, abort_levels, transfer_function,
         noise_response_cpsd, noise_reference_cpsd, sysid_response_cpsd,
         sysid_reference_cpsd, multiple_coherence, frames, total_frames,
         extra_parameters, last_response_cpsd, last_output_cpsd) = data
        if law is None:                            # front-load: build the law + sensitivity once
            law = band_average(specification=specification, extra_parameters=extra_parameters,
                               transfer_function=transfer_function,
                               sysid_response_cpsd=sysid_response_cpsd)
        drive = law.control(transfer_function=transfer_function, multiple_coherence=multiple_coherence,
                            frames=frames, total_frames=total_frames,
                            last_response_cpsd=last_response_cpsd,
                            last_output_cpsd=last_output_cpsd)
        data = yield drive
