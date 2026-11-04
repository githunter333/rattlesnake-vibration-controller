"""
optimal_diagonal_control_fast on the 12-drive RC plate, closed loop, offline.

WHAT THIS IS
------------
The rattlesnake-band-average repository's RC plate study (studies/rc_plate)
streams a 12-drive / 48-control synthetic test through Rattlesnake: the
removable component on a fixture plate, driven by 12 forces in lbf and
controlled at 48 accelerometers in g to the BARC response specification,
20-4000 Hz, at 16384 Hz with 1 Hz lines.  One row of that study's results
table is `nl8n_hu_odf_tfu` -- this law, with the controller's transfer-function
update on.

This script runs the SAME plant and the SAME specification through the SAME law
offline, in the closed loop that control_laws/tests/compare_optdiag_frame.py
uses, so the law's behaviour can be read without a 200-second real-time stream:
the response is "measured" from n_frames Gaussian frames (coherence_sweep.measure)
and handed back to control(), as Rattlesnake would.

It does not modify either repository.  It reads four files from the
band-average checkout and builds the plant and the specification itself.

WHICH LAW IS "CURRENT"
----------------------
control_laws/optimal_diagonal_control_fast.py on branch control-laws-on-dev.
`optimal_diagonal_control.py` is the SDP parent this subclass overrides the
per-bin solve of, not an older generation; temp/optimal_diagona_controlpy.py is
a stale scratch copy.  The band-average study's copy in
control_laws/hunter_9da1564/ is byte-identical to this file at
control-laws-on-dev, so the law run here is the law that produced that study's
`nl8n_hu_odf_tfu` row.

WHERE THE PLANT AND SPEC COME FROM
----------------------------------
The band-average study's two large inputs (`rc_plate_frf.npz` and
`barc_response_spec.npz`) are GitHub Release downloads and are not in that
checkout.  Both are rebuilt here from files that ARE in it, and each
reconstruction is checked against a number that repository documents:

plant H (48 x 12, g/lbf)
    H(f) = C (j w I - A)^-1 B + D from `rc_plate_state_space.npz`, keeping the
    48 rows whose `channel_unit` is 'g' (the first 12 rows are the force
    passthrough Rattlesnake requires, identity in D).  This is exactly the
    plant that study streams, not an approximation of it.
    CHECK: a modal sum over `sdynpy_models/rc_plate_test_shapes.npy` (rigid
    modes lifted to 1 Hz, the 31 modes below 8192 Hz, 1 % damping) reproduces
    it to a median relative error of 8e-16 over 20-4000 Hz.  That repository's
    README claims 5e-15 against the Release FRF, so the same convention is in
    use here.

specification (48 x 48 CPSD, g^2/Hz)
    The BARC's response to its 12 uncorrelated drive forces, flat at
    1e-3 lbf^2/Hz over 20-4000 Hz: S = H_barc G_ff H_barc^H with H_barc the
    modal accelerance through `sdynpy_models/barc_{response,force}_shapes.npy`
    (59 modes, 1 % damping).  This is what make_barc_vdyn.py computes in Visual
    Dynamics.
    CHECK: plot_spec.py in that repository records "101Z+ is near the median
    channel RMS (18.5 g of 7.7 to 45.5 g)".  This reconstruction gives 101Z+ at
    18.46 g and a range of 7.66 to 45.53 g.
    Warning and abort limits are +/-3 dB and +/-6 dB on each ASD, NaN on the
    cross terms, as make_barc_vdyn.py sets them.

Run settings are read off the study's own workbook
(`runs/rc_plate_nl8n_hu_odf_tfu.xlsx`): 20 frames in the CPSD, 50 system-ID
averages, 4 lbf RMS per drive of flat system-ID excitation, empty Control
Parameters (every default), 1e-5 g^2/Hz white response noise from
`rc_plate_noise_state_space.npz`.

THE FRF HANDED TO THE LAW IS MEASURED, NOT EXACT -- THIS MATTERS MOST
---------------------------------------------------------------------
compare_optdiag_frame.py hands the law the frame plant's exact analytic FRF,
because that plant is well enough conditioned that identification noise does
not change the answer.  On the RC plate it changes everything, so this script
simulates the identification the study ran (`identify()`: 50 averages of flat
uncorrelated random at 4 lbf RMS per drive, H1 estimator, 1e-5 g^2/Hz response
noise) and hands the law that H1 estimate, its Gyy, its Gvv and its multiple
coherence -- what Rattlesnake would hand it.  Scoring is always against the
exact plant.

  exact H           sigma_min/sigma_max over the band: median 5.2e-05, min 6.9e-08
  H1 estimate       median 5.1e-03, min 1.4e-04

The identification noise floors the smallest singular values by two orders of
magnitude, and that floor, not the structure, is what bounds the drive a law
can ask for.  The check that settles it: Rattlesnake's `pseudoinverse_control`
takes rcond = 1e-15 (no truncation), and the study's `n_pinv_noclip` row
records drive_lbf2 = 8.36e4.  On this reconstruction, pinv at rcond 1e-15 of
the H1 estimate gives 8.54e4 -- within 1 % -- while pinv of the exact H gives
6.65e10, six orders out.  `--frf exact` runs the exact-H case for comparison;
it is not what the rig does.

WHAT IS REPORTED
----------------
  err            RMS dB error of the TRUE response ASDs against the spec ASDs
                 over the specified lines, after each cycle -- narrowband, so
                 harsher than any banded score
  band_db        mean |error| in the study's 1/6-octave bands, which is exactly
                 the `band_db` of its runs/scores.jsonl: 1.6-3.5 dB for every
                 law in that table except this one, whose streamed row reads
                 155 dB
  floor          the best any law could do on the exact plant (achievable_
                 response.achievable_diagonal, on a sample of lines) -- 48
                 controls on 12 drives, so part of the spec is unreachable by
                 construction and `err` must be read against this, not 0 dB
  solved         specified lines the scheduler has refined off the buzz baseline
  drive_lbf2     sum over drives of the per-drive mean-square force, lbf^2 --
                 the same number as `drive_lbf2` in the study's runs/scores.jsonl,
                 so this row can be put beside that table directly
  drive |coh|    median pairwise drive coherence magnitude

Usage
-----
    PYTHONPATH=src python control_laws/tests/optdiag_on_rc_plate.py \
        [--band-average DIR] [--df HZ] [--cycles N] [--frames N]
        [--config default|fullband|both] [--frf measured|exact] [--tf-update]
        [--floor-lines N] [--no-floor]

    defaults: --df 1.0 --cycles 15 --frames 20 --config both --frf measured
              --floor-lines 40

`--df` thins the 1 Hz line grid by an integer factor (2 gives 2 Hz lines); the
law sees fewer lines and the whole thing runs in a quarter of the memory.  1.0
is the study's own grid.

`--tf-update` re-estimates the FRF from the CONTROL DRIVE every cycle, which is
what Rattlesnake's "Update System ID During Control" does (see
reidentify_from_control) and what the study's `_tfu` rows ran.  It also
exercises the fast law's safety rule: once H moves by more than
frf_update_threshold in relative Frobenius norm, every solve for the rest of
that call goes to the coherence-capped SDP instead of the unconstrained
factored solve.
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(LAW_DIR)
sys.path.insert(0, HERE)
if os.path.isdir(os.path.join(REPO, 'src')):
    sys.path.insert(0, os.path.join(REPO, 'src'))
import coherence_sweep as cs                           # noqa: E402
from rattlesnake.utilities import load_control_module   # noqa: E402

#: the band-average checkout holding studies/rc_plate
DEFAULT_BAND_AVERAGE = os.path.expanduser('~/Code/python/rattlesnake-band-average')

G = 9.80665                      # m/s^2 per g
LBF = 4.4482216152605            # N per lbf
#: make_barc_vdyn.py: the BARC's drive force CPSD, lbf^2/Hz, flat over FORCE_BAND
FORCE_LEVEL = 1e-3
FORCE_BAND = (20.0, 4000.0)
#: make_barc_vdyn.py: limits about each ASD, dB
WARNING_DB, ABORT_DB = 3.0, 6.0
#: 1 Hz lines to 4096 Hz -- the specification's own grid (FORCE_DF, FORCE_TOP)
DF_BASE, F_TOP = 1.0, 4096.0
#: make_noisy_plant.py, as the nl8n_ runs used it: white response ASD, g^2/Hz
RESPONSE_NOISE_ASD = 1e-5
#: the study workbook's "System ID Level": flat random, lbf RMS per drive, and
#: its "System ID Averages" and identified bandwidth
SYSID_RMS_LBF = 4.0
SYSID_AVERAGES = 50
SYSID_BANDWIDTH = 8192.0

#: the study's scoring bands (score_softening_runs.py): 1/6-octave centres whose
#: full width lies inside 20-4000 Hz
OCTAVE_FRACTION = 6


def octave_masks(f):
    """The study's 1/6-octave band masks over 20-4000 Hz.

    A band narrower than the line spacing holds no lines; it is dropped, so a
    coarse `--df` scores fewer bands rather than returning NaN.  On the study's
    own 1 Hz grid every band is populated.
    """
    fc = 1000 * 2.0 ** (np.arange(-60, 13) / OCTAVE_FRACTION)
    edge = 2 ** (1 / (2 * OCTAVE_FRACTION))
    fc = fc[(fc >= FORCE_BAND[0] * edge) & (fc <= FORCE_BAND[1] / edge)]
    masks = [(f >= c / edge) & (f < c * edge) for c in fc]
    return [m for m in masks if m.any()]


def band_error(response_asd, spec_asd, masks):
    """10 log10(mean band power / mean band spec), (n_bands, n_channels) dB --
    score_softening_runs.band_err, which the study's `band_db` averages |.| of."""
    return np.array([10 * np.log10(response_asd[m].mean(0) / spec_asd[m].mean(0))
                     for m in masks])


#: the two budgets of compare_optdiag_frame.py, in the named parameter form
CONFIGS = {
    'default': ('defaults (20 lines/call)', ''),
    'fullband': ('full band (400 lines, 500 ms)',
                 'max_bins_per_update: 400\nmax_solve_ms_per_update: 500'),
}


# --------------------------------------------------------------------------
# the plant
# --------------------------------------------------------------------------
def _dof_name(record):
    """'101Z+' from an sdynpy coordinate record (direction sign = sense)."""
    d = int(record['direction'])
    return f"{int(record['node'])}{'XYZ'[abs(d) - 1]}{'+' if d > 0 else '-'}"


def _modal_accelerance(f, phi_response, phi_force, frequency, damping):
    """Accelerance (len(f), n_response, n_force) in g/lbf from mass-normalized
    shapes in 1/sqrt(kg) and modal frequencies in Hz.

    -w^2 / (wr^2 - w^2 + 2 j zeta wr w) per mode; a rigid mode (wr = 0) gives
    the unit mass line, +1, and f = 0 gives 0.
    """
    wr = 2 * np.pi * np.asarray(frequency, float)
    zr = np.asarray(damping, float)
    out = np.zeros((len(f), phi_response.shape[1], phi_force.shape[1]), complex)
    for k, fk in enumerate(f):
        w = 2 * np.pi * fk
        if w == 0.0:
            continue
        gain = -w ** 2 / (wr ** 2 - w ** 2 + 2j * zr * wr * w)
        out[k] = (phi_response.T * gain) @ phi_force
    return out * (LBF / G)


def state_space_frf(path, f):
    """(len(f), 48, 12) g/lbf from a study state-space file, plus its channel names.

    The file's C and D carry 60 rows: the 12 force passthrough rows Rattlesnake
    requires (unit 'lbf', identity in D) and the 48 accelerometers (unit 'g').
    Only the accelerometer rows are control responses.
    """
    z = np.load(path, allow_pickle=True)
    A, B, C, D = z['A'], z['B'], z['C'], z['D']
    unit = list(map(str, z['channel_unit']))
    dof = list(map(str, z['channel_dof']))
    response = [i for i, u in enumerate(unit) if u == 'g']
    force = [i for i, u in enumerate(unit) if u == 'lbf']
    assert [dof[i] for i in force] == list(map(str, z['input_dof'])), \
        'force passthrough rows are not in drive order'
    Cr, Dr = C[response], D[response]
    eye = np.eye(A.shape[0])
    H = np.empty((len(f), len(response), B.shape[1]), complex)
    for k, fk in enumerate(f):
        H[k] = Cr @ np.linalg.solve(2j * np.pi * fk * eye - A, B) + Dr
    return H, [dof[i] for i in response], list(map(str, z['input_dof']))


def check_state_space_frf(band_average, f, H, band):
    """Median and max relative error of `H` against a modal sum over the study's
    sdynpy shapes -- the check the band-average README documents at 5e-15."""
    models = os.path.join(band_average, 'studies', 'rc_plate', 'sdynpy_models')
    shapes = np.load(os.path.join(models, 'rc_plate_test_shapes.npy'), allow_pickle=True)
    z = np.load(os.path.join(band_average, 'studies', 'rc_plate', 'rc_plate_state_space.npz'),
                allow_pickle=True)
    unit = list(map(str, z['channel_unit']))
    names = [_dof_name(c) for c in shapes['coordinate'][0]]
    assert names == list(map(str, z['channel_dof'])), 'shape and channel DOFs differ'
    response = [i for i, u in enumerate(unit) if u == 'g']
    force = [i for i, u in enumerate(unit) if u == 'lbf']
    # the state-space plants lift the rigid modes to 1 Hz and keep the modes
    # below 8192 Hz (band-average studies/rc_plate/README.md)
    frequency = np.where(shapes['frequency'] < 1.0, 1.0, shapes['frequency'])
    keep = frequency < 8192.0
    Hm = _modal_accelerance(f, shapes['shape_matrix'][keep][:, response],
                            shapes['shape_matrix'][keep][:, force],
                            frequency[keep], shapes['damping'][keep])
    rel = np.abs(H[band] - Hm[band]) / np.maximum(np.abs(H[band]), 1e-300)
    return float(np.median(rel)), float(rel.max())


def barc_specification(band_average, f, control_dofs):
    """(len(f), 48, 48) BARC response specification CPSD in g^2/Hz.

    S(f) = H_barc(f) G_ff H_barc(f)^H, G_ff = FORCE_LEVEL * I over FORCE_BAND:
    what make_barc_vdyn.py computes in Visual Dynamics and exports as
    barc_response_spec.npz.  Drive-column signs cancel in the outer product, so
    the drive DOF senses do not matter here.
    """
    models = os.path.join(band_average, 'studies', 'rc_plate', 'sdynpy_models')
    response = np.load(os.path.join(models, 'barc_response_shapes.npy'), allow_pickle=True)
    force = np.load(os.path.join(models, 'barc_force_shapes.npy'), allow_pickle=True)
    names = [_dof_name(c) for c in response['coordinate'][0]]
    assert names == list(control_dofs), \
        'the BARC specification DOFs are not the plate control DOFs, in order'
    assert np.allclose(response['frequency'], force['frequency'])
    Hb = _modal_accelerance(f, response['shape_matrix'], force['shape_matrix'],
                            response['frequency'], response['damping'])
    level = np.where((f >= FORCE_BAND[0]) & (f <= FORCE_BAND[1]), FORCE_LEVEL, 0.0)
    spec = np.zeros((len(f), Hb.shape[1], Hb.shape[1]), complex)
    inb = np.flatnonzero(level > 0)
    for k in inb:
        spec[k] = level[k] * (Hb[k] @ Hb[k].conj().T)
    return spec


def rc_plate_plant(band_average=DEFAULT_BAND_AVERAGE, df=DF_BASE, plant='rc_plate_state_space.npz'):
    """Plant dict with compare_optdiag_frame's keys, for the 12-drive RC plate."""
    study = os.path.join(band_average, 'studies', 'rc_plate')
    step = int(round(df / DF_BASE))
    if abs(step * DF_BASE - df) > 1e-9 or step < 1:
        raise ValueError(f'df must be a positive multiple of {DF_BASE:g} Hz, not {df}')
    f = np.arange(0.0, F_TOP + DF_BASE / 2, DF_BASE)[::step]
    H, control_dofs, drive_dofs = state_space_frf(os.path.join(study, plant), f)
    spec = barc_specification(band_average, f, control_dofs)
    n_freq, n_resp, n_drive = H.shape

    asd = np.real(np.einsum('fcc->fc', spec))
    band = asd.max(axis=1) > 0
    warn = np.where(band[None, :, None], asd[None] * 10 ** (np.array([-1.0, 1.0])[:, None, None]
                                                            * WARNING_DB / 10), np.nan)
    abort = np.where(band[None, :, None], asd[None] * 10 ** (np.array([-1.0, 1.0])[:, None, None]
                                                             * ABORT_DB / 10), np.nan)
    # white response noise on every control channel; the drive channels are
    # force transducers teed back from the commanded signal, so their own noise
    # is well below the drive and is carried at the same ASD for symmetry
    noise = RESPONSE_NOISE_ASD
    gxx_noise = np.tile(noise * np.eye(n_resp), (n_freq, 1, 1)).astype(complex)
    gff_noise = np.tile(noise * np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    #: flat uncorrelated system-ID drive, SYSID_RMS_LBF per drive spread over
    #: the identified bandwidth (the workbook's 0-8192 Hz), lbf^2/Hz
    sysid_asd = SYSID_RMS_LBF ** 2 / SYSID_BANDWIDTH
    return dict(spec=spec, warn=warn, abort=abort, H=H, gxx_noise=gxx_noise,
                gff_noise=gff_noise, sysid_asd=sysid_asd,
                band=band, f=f, df=f[1] - f[0], spec_asd=asd,
                controls=control_dofs, drives=drive_dofs, plant=plant,
                band_average=band_average)


def identify(p, rng, n_avg=SYSID_AVERAGES):
    """One system identification, as Rattlesnake runs it: `n_avg` frames of flat
    uncorrelated random at SYSID_RMS_LBF per drive, H1 estimator, response noise.

    Returns what the controller hands a law: the H1 estimate, the measured
    response and reference CPSDs, and the multiple coherence.  `n_avg = 0`
    returns the exact FRF and its expected CPSDs (no identification noise) --
    what compare_optdiag_frame.py does, and NOT what the rig does.
    """
    H = p['H']
    n_freq, n_resp, n_drive = H.shape
    noise_diag = np.real(np.einsum('fii->fi', p['gxx_noise']))
    gvv_exact = np.tile(p['sysid_asd'] * np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    if n_avg == 0:
        gyy = H @ gvv_exact @ H.conj().transpose(0, 2, 1) + p['gxx_noise']
        sig = np.real(np.einsum('fmm->fm', gyy)) - noise_diag
        return dict(H=H, gxx_sysid=gyy, gff_sysid=gvv_exact,
                    coh=sig / np.maximum(sig + noise_diag, 1e-300))
    scale = np.sqrt(p['sysid_asd'])
    z = (rng.standard_normal((n_freq, n_drive, n_avg))
         + 1j * rng.standard_normal((n_freq, n_drive, n_avg))) / np.sqrt(2)
    v = scale * z
    e = (rng.standard_normal((n_freq, n_resp, n_avg))
         + 1j * rng.standard_normal((n_freq, n_resp, n_avg))) / np.sqrt(2)
    y = H @ v + np.sqrt(noise_diag)[:, :, None] * e
    gvv = v @ v.conj().transpose(0, 2, 1) / n_avg
    gyv = y @ v.conj().transpose(0, 2, 1) / n_avg
    gyy = y @ y.conj().transpose(0, 2, 1) / n_avg
    gvv_inv = np.linalg.inv(gvv)
    H1 = gyv @ gvv_inv
    coherent = np.real(np.einsum('fmn,fnk,fmk->fm', gyv, gvv_inv, gyv.conj()))
    coh = coherent / np.maximum(np.real(np.einsum('fmm->fm', gyy)), 1e-300)
    return dict(H=H1, gxx_sysid=gyy, gff_sysid=gvv, coh=np.clip(coh, 0.0, 1.0))


def spread(H, band):
    """sigma_min/sigma_max over the specified lines: median, 10th pct, min."""
    s = np.linalg.svd(H[band], compute_uv=False)
    r = s[:, -1] / s[:, 0]
    return float(np.median(r)), float(np.percentile(r, 10)), float(r.min())


def summary(p):
    s = np.linalg.svd(p['H'][p['band']], compute_uv=False)
    ratio = s[:, -1] / s[:, 0]
    rank = np.sum(s / s[:, :1] > 1e-8, axis=1)
    rms = np.sqrt(p['spec_asd'][p['band']].sum(axis=0) * p['df'])
    return (f"{p['plant']}: {len(p['controls'])} controls x {len(p['drives'])} drives, "
            f"{p['f'].size} lines at {p['df']:g} Hz, {int(p['band'].sum())} specified "
            f"({p['f'][p['band']].min():g}-{p['f'][p['band']].max():g} Hz)\n"
            f"  in band: rank {np.median(rank):.0f} (min {rank.min()}), "
            f"sigma_min/sigma_max median {np.median(ratio):.1e}, "
            f"10th pct {np.percentile(ratio, 10):.1e}, min {ratio.min():.1e}\n"
            f"  specification: channel RMS {rms.min():.2f} to {rms.max():.2f} g "
            f"(median {np.median(rms):.2f}), trace "
            f"{np.sqrt(p['spec_asd'][p['band']].sum() * p['df']):.1f} g RMS")


# --------------------------------------------------------------------------
# the closed loop -- the same one as compare_optdiag_frame.run
# --------------------------------------------------------------------------
def run(p, cls, params, n_cycles, n_frames, n_avg=SYSID_AVERAGES, tf_update=False, seed=1):
    """One closed-loop run.  The law sees the identified FRF; scoring is against
    the exact plant.  With `tf_update`, the FRF is re-identified every cycle."""
    rng = np.random.default_rng(seed)
    H_true, sel = p['H'], p['band']
    noise_diag = np.real(np.einsum('fii->fi', p['gxx_noise']))
    s_d = p['spec_asd'][sel]
    nf = n_frames if n_frames > 0 else 1000
    sysid = identify(p, rng, n_avg)
    Hm = sysid['H']
    args = (p['gxx_noise'], p['gff_noise'], sysid['gxx_sysid'], sysid['gff_sysid'])

    def err(X):
        y = np.real(np.einsum('fmn,fnk,fmk->fm', H_true, X, H_true.conj()))[sel]
        return 10 * np.log10(np.maximum(y, 1e-300) / s_d)

    def n_solves(law):
        return getattr(law, 'n_fast_solves', 0) + getattr(law, 'n_safe_solves', 0)

    def refined(law):
        return getattr(law, 'sdp_refined', np.zeros(p['f'].size, dtype=bool))

    errs, ms, lines, solved, drive_lbf2 = [], [], [], [], []
    with contextlib.redirect_stdout(io.StringIO()) as log:
        law = cls(p['spec'], p['warn'], p['abort'], params, None, *args, None, 0, nf, None, None)
        t0 = time.perf_counter()
        law.system_id_update(Hm, *args, sysid['coh'], nf, nf)
        X = law.control(Hm, sysid['coh'], nf, nf, None, None)
        t_init = time.perf_counter() - t0
        errs.append(err(X))
        solved.append(int(np.sum(refined(law)[sel])))
        drive_lbf2.append(_drive_lbf2(X, sel, p['df']))
        for _ in range(n_cycles):
            Gyy, Gxx, coh = cs.measure(H_true, X, noise_diag, n_frames, rng)
            if tf_update:
                sysid = reidentify_from_control(p, X, nf, rng)
                Hm = sysid['H']
                law.system_id_update(Hm, p['gxx_noise'], p['gff_noise'],
                                     sysid['gxx_sysid'], sysid['gff_sysid'],
                                     sysid['coh'], nf, nf)
                coh = sysid['coh']
            s0 = n_solves(law)
            t0 = time.perf_counter()
            X = law.control(Hm, coh, nf, nf, Gyy, Gxx)
            ms.append(1e3 * (time.perf_counter() - t0))
            lines.append(n_solves(law) - s0)
            errs.append(err(X))
            solved.append(int(np.sum(refined(law)[sel])))
            drive_lbf2.append(_drive_lbf2(X, sel, p['df']))
    d = np.real(np.einsum('fii->fi', X))[sel]
    c = np.abs(X[sel]) / np.sqrt(np.maximum(d[:, :, None] * d[:, None, :], 1e-300))
    iu = np.triu_indices(X.shape[1], 1)
    # effective rank of the commanded drive: a rank-1 X is 12 perfectly
    # correlated drives, which no H1/H2 estimator can re-identify through
    ev = np.linalg.eigvalsh(X[sel])[:, ::-1]
    rank = np.sum(ev > ev[:, :1] * 1e-6, axis=1)
    # the study's own score: mean |error| in 1/6-octave bands (its `band_db`)
    y_full = np.real(np.einsum('fmn,fnk,fmk->fm', H_true, X, H_true.conj()))
    be = band_error(y_full, p['spec_asd'], octave_masks(p['f']))
    return dict(X=X, errs=np.array(errs), solved=np.array(solved), ms=np.array(ms),
                band_db=float(np.nanmean(np.abs(be))),
                drive_rank=(float(np.median(rank)), int(rank.min()), int(rank.max())),
                lines=np.array(lines), t_init=t_init, H_meas=Hm,
                drive_lbf2=np.array(drive_lbf2),
                drive_db=10 * np.log10(np.mean(np.real(np.einsum('fii->f', X))[sel])),
                drive_rms=np.sqrt(np.maximum(d, 0).sum(axis=0) * p['df']),
                n_fast=getattr(law, 'n_fast_solves', 0), n_safe=getattr(law, 'n_safe_solves', 0),
                coh=float(np.median(c[:, iu[0], iu[1]])), log=log.getvalue())


def reidentify_from_control(p, X, n_frames, rng):
    """Re-estimate the FRF from the CONTROL drive, as Rattlesnake does when
    "Update System ID During Control" is on.

    random_vibration_sys_id_data_analysis.py hands a type-2 law `self.control_frf`
    in that case -- the FRF computed from the control data, with the commanded
    drive as the reference -- not a separate system-ID sweep.  The estimator is
    spectral_processing.py's H1: Gyv @ pinv(Gvv, rcond=1e-12, hermitian=True).

    This is the step that makes a rank-deficient drive self-defeating: pinv of a
    singular reference CPSD truncates to the drive's own rank, so the columns of
    the new FRF that the drive did not excite come back wrong, and the next solve
    is made against them.
    """
    H = p['H']
    n_freq, n_resp, n_drive = H.shape
    noise_diag = np.real(np.einsum('fii->fi', p['gxx_noise']))
    Xh = (X + X.conj().transpose(0, 2, 1)) / 2
    lam, V = np.linalg.eigh(Xh)
    L = V * np.sqrt(np.clip(lam, 0, None))[:, None, :]
    z = (rng.standard_normal((n_freq, n_drive, n_frames))
         + 1j * rng.standard_normal((n_freq, n_drive, n_frames))) / np.sqrt(2)
    v = L @ z
    e = (rng.standard_normal((n_freq, n_resp, n_frames))
         + 1j * rng.standard_normal((n_freq, n_resp, n_frames))) / np.sqrt(2)
    y = H @ v + np.sqrt(noise_diag)[:, :, None] * e
    gvv = v @ v.conj().transpose(0, 2, 1) / n_frames
    gyv = y @ v.conj().transpose(0, 2, 1) / n_frames
    gyy = y @ y.conj().transpose(0, 2, 1) / n_frames
    gvv_pinv = np.linalg.pinv(gvv, rcond=1e-12, hermitian=True)
    coherent = np.real(np.einsum('fmn,fnk,fmk->fm', gyv, gvv_pinv, gyv.conj()))
    coh = coherent / np.maximum(np.real(np.einsum('fmm->fm', gyy)), 1e-300)
    return dict(H=gyv @ gvv_pinv, gxx_sysid=gyy, gff_sysid=gvv,
                coh=np.clip(coh, 0.0, 1.0))


def _drive_lbf2(X, sel, df):
    """sum over drives of per-drive mean square force, lbf^2 -- the study's
    `drive_lbf2` in runs/scores.jsonl (score_softening_runs.py: sum(drms**2))."""
    d = np.real(np.einsum('fii->fi', X))[sel]
    return float((np.maximum(d, 0).sum(axis=0) * df).sum())


def floor(p, n_lines, seed=0):
    """Best achievable RMS and peak dB error on a sample of specified lines."""
    ar = load_control_module(os.path.join(LAW_DIR, 'achievable_response.py'))
    idx = np.flatnonzero(p['band'])
    idx = idx[np.linspace(0, idx.size - 1, min(n_lines, idx.size)).round().astype(int)]
    with contextlib.redirect_stdout(io.StringIO()):
        out = ar.achievable_diagonal(p['H'], p['spec_asd'], line_indices=idx, seed=seed)
    rms = np.asarray(out['best_rms_db'], float)[idx]
    peak = np.asarray(out['best_minimax_db'], float)[idx]
    trace = np.asarray(out['drive_trace'], float)[idx]
    good = np.isfinite(rms)
    return dict(n=int(good.sum()), rms_median=float(np.median(rms[good])),
                rms_max=float(rms[good].max()), peak_median=float(np.median(peak[good])),
                trace_db=float(10 * np.log10(np.median(trace[good]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-average', default=DEFAULT_BAND_AVERAGE,
                    help='the rattlesnake-band-average checkout (default: %(default)s)')
    ap.add_argument('--df', type=float, default=DF_BASE,
                    help='line spacing in Hz, a multiple of 1 Hz (default: %(default)s)')
    ap.add_argument('--cycles', type=int, default=15)
    ap.add_argument('--frames', type=int, default=20,
                    help='frames per CPSD estimate; 0 = exact CPSDs (default: %(default)s)')
    ap.add_argument('--config', default='both', choices=['default', 'fullband', 'both'])
    ap.add_argument('--frf', default='measured', choices=['measured', 'exact'],
                    help="FRF handed to the law: 'measured' simulates the study's "
                         "50-average H1 identification (what the rig does); 'exact' "
                         'hands it the analytic plant (default: %(default)s)')
    ap.add_argument('--tf-update', action='store_true',
                    help="re-estimate the FRF from the control drive every cycle, as "
                         "Rattlesnake's 'Update System ID During Control' does and as the "
                         "study's _tfu rows did")
    ap.add_argument('--plant', default='rc_plate_state_space.npz',
                    help='state-space file in studies/rc_plate (default: %(default)s)')
    ap.add_argument('--floor-lines', type=int, default=40,
                    help='specified lines to compute the achievable floor on (default: %(default)s)')
    ap.add_argument('--no-floor', action='store_true')
    a = ap.parse_args()

    p = rc_plate_plant(a.band_average, a.df, a.plant)
    print(summary(p))
    med, mx = check_state_space_frf(a.band_average, p['f'], p['H'], p['band'])
    print(f'  plant check: state-space H vs modal sum, relative error '
          f'median {med:.1e}, max {mx:.1e}')
    rms = np.sqrt(p['spec_asd'][p['band']].sum(axis=0) * p['df'])
    k = list(p['controls']).index('101Z+')
    print(f"  spec check: 101Z+ {rms[k]:.2f} g RMS, range {rms.min():.2f}-{rms.max():.2f} g "
          f"(band-average plot_spec.py records 18.5 g of 7.7 to 45.5)")

    n_avg = SYSID_AVERAGES if a.frf == 'measured' else 0
    sysid = identify(p, np.random.default_rng(1), n_avg)
    for tag, Hx in (('exact plant      ', p['H']), (f'FRF to the law ({a.frf[:4]})', sysid['H'])):
        med, pct10, mn = spread(Hx, p['band'])
        print(f'  {tag}: sigma_min/sigma_max median {med:.1e}, 10th pct {pct10:.1e}, min {mn:.1e}')
    if n_avg:
        mc = sysid['coh'][p['band']]
        print(f"  identification: {n_avg} averages at {SYSID_RMS_LBF:g} lbf RMS per drive, "
              f"multiple coherence median {np.median(mc):.3f}, 5th pct {np.percentile(mc, 5):.3f}")
    pinv = np.linalg.pinv(sysid['H'], rcond=1e-15)
    Xp = pinv @ p['spec'] @ pinv.conj().transpose(0, 2, 1)
    print(f"  calibration: pseudoinverse_control (rcond 1e-15) on this FRF gives "
          f"drive_lbf2 {_drive_lbf2(Xp, p['band'], p['df']):.3e}; the study's "
          f"n_pinv_noclip row records 8.356e+04")

    print(f"\n{a.cycles} cycles, " + (f'{a.frames}-frame CPSD estimates' if a.frames
                                      else 'exact CPSDs')
          + (', FRF re-identified every cycle' if a.tf_update else ', FRF held fixed'))

    if not a.no_floor:
        fl = floor(p, a.floor_lines)
        print(f"\nbest achievable on this plant ({fl['n']} lines sampled over the band): "
              f"RMS {fl['rms_median']:.2f} dB median, {fl['rms_max']:.2f} dB worst; "
              f"peak {fl['peak_median']:.2f} dB median; drive trace "
              f"{fl['trace_db']:+.1f} dB re 1 lbf^2 median")
        print('  48 controls on 12 drives: this is the floor every law here is measured against.')

    with contextlib.redirect_stdout(io.StringIO()):
        module = load_control_module(os.path.join(LAW_DIR, 'optimal_diagonal_control_fast.py'))
    cls = module.optimal_diagonal_control_fast

    names = ['default', 'fullband'] if a.config == 'both' else [a.config]
    print(f"\n{'':32s} {'err':>6s} {'band_db':>8s}  {'solved':>13s}  {'ms/call':>14s}  "
          f"{'lines':>6s}  {'drive_lbf2':>11s}  {'drive RMS lbf':>21s}  {'fast/':>11s}  |coh|")
    print(f"{'':32s} {'(dB)':>6s} {'(dB)':>8s}  {'of specified':>13s}  {'median / max':>14s}  "
          f"{'/call':>6s}  {'(study 8e4)':>11s}  {'min / median / max':>21s}  "
          f"{'SDP solves':>11s}  median  drive rank (of 12)")
    results = {}
    for name in names:
        label, params = CONFIGS[name]
        r = results[name] = run(p, cls, params, a.cycles, a.frames, n_avg, a.tf_update)
        e = float(np.sqrt(np.mean(r['errs'][-1] ** 2)))
        dr = r['drive_rms']
        print(f"{label:32s} {e:6.2f} {r['band_db']:8.2f}  "
              f"{r['solved'][-1]:6d}/{int(p['band'].sum()):<6d}  "
              f"{np.median(r['ms']):6.0f} /{r['ms'].max():6.0f}  {np.median(r['lines']):6.0f}  "
              f"{r['drive_lbf2'][-1]:11.3e}  {dr.min():6.1f} /{np.median(dr):6.1f} /{dr.max():6.1f}  "
              f"{r['n_fast']:5d}/{r['n_safe']:<5d}  {r['coh']:6.2f}  "
              f"median {r['drive_rank'][0]:.0f}, {r['drive_rank'][1]}-{r['drive_rank'][2]}",
              flush=True)

    print('\nerr (dB) by cycle (cycle 0 = the first control call, startup-capped):')
    print(f"{'cycle':>32s} " + ' '.join(f'{k:>9d}' for k in range(a.cycles + 1)))
    for name in names:
        r = results[name]
        print(f'{CONFIGS[name][0]:32s} ' +
              ' '.join(f"{np.sqrt(np.mean(e ** 2)):9.2f}" for e in r['errs']))
    print('\ndrive_lbf2 by cycle (the study records 3e3 for auto-Tikhonov, 8e4 for pinv):')
    print(f"{'cycle':>32s} " + ' '.join(f'{k:>9d}' for k in range(a.cycles + 1)))
    for name in names:
        print(f'{CONFIGS[name][0]:32s} ' +
              ' '.join(f'{v:9.2e}' for v in results[name]['drive_lbf2']))

    print('\nper-channel final error, worst and best 6 channels (full-band budget if run):')
    r = results[names[-1]]
    per = np.sqrt(np.mean(r['errs'][-1] ** 2, axis=0))
    order = np.argsort(per)
    for tag, sl in (('worst', order[::-1][:6]), ('best ', order[:6])):
        print(f"  {tag}: " + '  '.join(f"{p['controls'][i]} {per[i]:+6.2f}" for i in sl))
    over = np.sum(per > 3.0)
    print(f"  {over} of {len(per)} channels over 3 dB RMS; "
          f"{np.sum(per > 6.0)} over 6 dB")


if __name__ == '__main__':
    main()
