"""The response-error trim added 2026-09-21 -- the second loop.

Until now optimal_diagonal_control was closed on the identified plant model
(it re-solves against the live FRF every cycle) and open on the response
error (last_response_cpsd was accepted and never read).  Anything the FRF
did not explain -- H1 bias, extraneous input, drive clipping, a
nonlinearity that never appears in a linear FRF -- stood uncorrected,
because the law never compared what it measured with what was asked for.

TEST 1  REGRESSION.  Gain 0 is the default.  The command must be BIT
        identical to the committed code, on real run-29 data.  This is the
        test that has to hold: if it fails, every archived run is in doubt.
TEST 2  CONVERGENCE.  Against a plant that is a known 2.0 dB hotter than
        the model the law was handed, the measured response must come onto
        specification and the trim must settle near -2.0 dB.
TEST 3  NO WINDUP.  At channel-bins the plant CANNOT reach, the error never
        clears.  An ordinary integrator would run to its clamp demanding
        drive that buys nothing.  The solver-agreement gate must hold the
        trim at exactly 0 dB there while the reachable channels correct.
TEST 4  LEVEL-CHANGE FREEZE.  The data collector divides each frame by the
        CURRENT test level, so a frame spanning a ramp is normalized by the
        wrong number.  set_test_level_db must stand the trim down.
"""
import sys, os, numpy as np, netCDF4 as nc4, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
HEAD = os.path.join(os.path.expanduser('~'), 'work', 'head')   # git-HEAD copies
RUNS = os.path.join(ROOT, 'examples', 'sixdrive12resp', 'results', 'runs')

d = nc4.Dataset(os.path.join(RUNS, 'run29_optdiagfast_updateon_gatefix_sysid.nc4'))
g = d['Frame 6x12 Random']
spec = np.asarray(g['specification_cpsd_matrix_real'][:] + 1j*g['specification_cpsd_matrix_imag'][:])
H    = np.asarray(g['frf_data_real'][:] + 1j*g['frf_data_imag'][:])
ctrl = np.asarray(g['control_channel_indices'][:]).astype(int)
sr   = np.asarray(g['response_cpsd_real'][:] + 1j*g['response_cpsd_imag'][:])
rr   = np.asarray(g['reference_cpsd_real'][:] + 1j*g['reference_cpsd_imag'][:])
nr   = np.asarray(g['response_noise_cpsd_real'][:] + 1j*g['response_noise_cpsd_imag'][:])
nf   = np.asarray(g['reference_noise_cpsd_real'][:] + 1j*g['reference_noise_cpsd_imag'][:])
coh  = np.asarray(g['frf_coherence'][:]); d.close()
H = H[:, ctrl, :]; sr = sr[:, ctrl][:, :, ctrl]; nr = nr[:, ctrl][:, :, ctrl]
if spec.shape[1] != H.shape[1]: spec = spec[:, ctrl][:, :, ctrl]
coh = coh[:, ctrl] if coh.ndim > 1 else coh

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m

live = load(os.path.join(ROOT, 'control_laws', 'optimal_diagonal_control_fast.py'), 'odcf_live')
base = load(os.path.join(HEAD, 'control_laws', 'optimal_diagonal_control_fast.py'), 'odcf_head')

def build(mod, params, sp=spec, hh=H, ch=coh, s_r=sr, n_r=nr):
    w = np.zeros(sp.shape[:2]); a = np.zeros(sp.shape[:2])
    return mod.optimal_diagonal_control_fast(sp, w, a, params, hh, n_r, nf, s_r, rr, ch, 100, 100, None, None)

print('=' * 74)
print('TEST 1 -- gain 0 (the default): bit-identical to the committed code')
print('=' * 74)
P10 = '1e-6,0.5,20,1.0,0.95,6'          # ten fields or fewer: no trim named at all
lo = build(live, P10); hd = build(base, P10)
for call in range(4):
    o_lo = lo.control(H, coh, 101, 101, None, None)
    o_hd = hd.control(H, coh, 101, 101, None, None)
delta = float(np.max(np.abs(o_lo - o_hd)))
print(f'  y_trim is None                {lo.y_trim is None}')
print(f'  _p_commanded is None          {lo._p_commanded is None}')
print(f'  max |new - committed|         {delta:.3e}')
assert lo.y_trim is None, 'trim state allocated while the trim is off'
assert lo._p_commanded is None, 'prediction cached while the trim is off'
assert delta == 0.0, 'THE TRIM CHANGED THE COMMAND WITH THE GAIN AT ZERO'
print('  PASS -- the default path is untouched')

# --- a small in-band subset, so the closed-loop sims run in seconds --------
diag = np.real(spec.diagonal(axis1=1, axis2=2))
inb = np.where(diag.max(axis=1) > 0)[0]
oob = np.where(diag.max(axis=1) <= 0)[0]
sel = np.concatenate([inb[300:420], oob[:20]])          # 120 in-band + 20 out-of-band
S, HS, CS, SRS, NRS = spec[sel], H[sel], coh[sel], sr[sel], nr[sel]
nfS, rrS = nf[sel], rr[sel]
inb_s = np.real(S.diagonal(axis1=1, axis2=2)).max(axis=1) > 0

def build_s(mod, params, sp, hm=None):
    hm = HS if hm is None else hm
    w = np.zeros(sp.shape[:2]); a = np.zeros(sp.shape[:2])
    return mod.optimal_diagonal_control_fast(sp, w, a, params, hm, NRS, nfS, SRS, rrS, CS, 100, 100, None, None)

def run_loop(law, Htrue, n_calls, sp=None, hm=None):
    """Close the loop the way Rattlesnake does: command, measure, feed back."""
    sp = S if sp is None else sp
    hm = HS if hm is None else hm
    specd = np.real(sp.diagonal(axis1=1, axis2=2))
    z_prev = None; out_prev = None; z = None
    for k in range(n_calls):
        out = law.control(hm, CS, 101, 101, z_prev, out_prev)
        Y = np.einsum('fmn,fnk,flk->fml', Htrue, out, Htrue.conj())
        z = np.maximum(np.real(np.einsum('fmm->fm', Y)), 0.0)
        z_prev = Y; out_prev = out
    return z, specd


def scored(law, z, specd):
    """The population the trim is RESPONSIBLE for, and its error.

    Not every channel-bin: this frame is rank 5 with 6 drives, so at a
    structural null the law lands short of specification whatever is asked
    of it, and pooling those in measures the plant's conditioning, not the
    loop.  The trim's remit is exactly the set where the solver reached
    what it aimed at -- the same gate the trim itself applies -- because
    only there is the gap between predicted and measured attributable to
    the model rather than to achievability.
    """
    Y = np.einsum('fmn,fnk,flk->fml', law._H_last_used, law.output_cpsd,
                  law._H_last_used.conj())
    p = np.maximum(np.real(np.einsum('fmm->fm', Y)), 0.0)
    teff = specd if law.y_trim is None else specd*law.y_trim
    ok = inb_s[:, None] & (p > 0) & (teff > 0) & (z > 0) & (specd > 0)
    gate = np.zeros_like(p, dtype=bool)
    gate[ok] = np.abs(10*np.log10(p[ok]/teff[ok])) <= law.error_threshold_db
    e = 10*np.log10(z[gate]/specd[gate])
    return gate.sum(), float(np.sqrt(np.mean(e**2)))

print()
print('=' * 74)
print('TEST 2 -- a plant 2.0 dB hotter than the model: does it come back?')
print('=' * 74)
BIAS_DB = 2.0
Htrue = HS * np.sqrt(10.0**(BIAS_DB/10.0))
PT = '1e-6,0.5,20,1.0,0.95,6,-9.0,0,2,2.0,0.5,3.0,0.5,0.5'   # gain 0.5
law_on = build_s(live, PT, S)
z_on, specd = run_loop(law_on, Htrue, 30)
law_off = build_s(live, '1e-6,0.5,20,1.0,0.95,6,-9.0,0,2,2.0', S)
z_off, _ = run_loop(law_off, Htrue, 30)
n_off, e_off = scored(law_off, z_off, specd)
n_on,  e_on  = scored(law_on,  z_on,  specd)
t_db = 10*np.log10(law_on.y_trim[inb_s])
DEAD = law_on.response_trim_deadband_db
print(f'  channel-bins the solver reached   {n_off} (off) / {n_on} (on) '
      f'of {int(inb_s.sum())*S.shape[1]} in band')
print(f'  measured rms error there, OFF     {e_off:.2f} dB   <- the bias, uncorrected')
print(f'  measured rms error there, ON      {e_on:.2f} dB')
print(f'  trim settled at                   median {np.median(t_db):+.2f} dB '
      f'(expected {-BIAS_DB:+.2f}, deadband {DEAD:g})')
print(f'  trim updates                      {law_on.n_trim_updates}')
assert abs(e_off - BIAS_DB) < 0.5, (
    f'with the trim off the error on the reached set should BE the bias, got {e_off:.2f}')
assert e_on <= DEAD + 0.2, f'trim failed to bring the response onto spec ({e_on:.2f} dB)'
assert e_on < e_off - 1.0, 'trim did not improve on the uncorrected case'
assert abs(np.median(t_db) + BIAS_DB) <= DEAD + 0.2, 'trim did not settle near -bias'
print('  PASS -- error the FRF cannot explain is now corrected, to the deadband')

print()
print('=' * 74)
print('TEST 3 -- unreachable channel-bins: the gate must stop the wind-up')
print('=' * 74)
# A PROVABLY unreachable target, not merely an expensive one.  Give two
# response channels the SAME row of H -- one drive combination moves both
# identically -- and then ask for targets 10 dB apart.  No X of any size
# satisfies both, so the solver's own predicted error stays large at both
# for ever and the gate must exclude them permanently.  (The first version
# of this test just raised one channel's specification by 20 dB, which the
# solver could partly reach by pouring in drive; the trim crept to 1.44 dB
# and then stopped on its own.  Correct behaviour, but it was testing the
# gate closing late rather than the gate holding.)
S2 = S.copy(); H2 = HS.copy()
hot_bins = np.where(inb_s)[0][:30]
CH_A, CH_B = 3, 4
H2[np.ix_(hot_bins, [CH_B])] = H2[np.ix_(hot_bins, [CH_A])]
for f in hot_bins:
    S2[f, CH_B, CH_B] = S2[f, CH_A, CH_A]*10.0        # identical rows, 10 dB apart
H2true = H2*np.sqrt(10.0**(BIAS_DB/10.0))
law_w = build_s(live, PT, S2, hm=H2)
run_loop(law_w, H2true, 30, sp=S2, hm=H2)
tw = 10*np.log10(law_w.y_trim)
hot = tw[np.ix_(hot_bins, [CH_A, CH_B])]
cold_bins = np.where(inb_s)[0][40:]
cold = tw[cold_bins][:, [0, 1, 2, 5]]
Yw = np.einsum('fmn,fnk,flk->fml', law_w._H_last_used, law_w.output_cpsd,
               law_w._H_last_used.conj())
pw = np.maximum(np.real(np.einsum('fmm->fm', Yw)), 1e-30)
sw = np.real(S2.diagonal(axis1=1, axis2=2))
short = 10*np.log10(pw[np.ix_(hot_bins, [CH_A, CH_B])] /
                    np.maximum(sw[np.ix_(hot_bins, [CH_A, CH_B])], 1e-30))
# Both channels see the SAME row of H, so the solver splits the difference:
# it lands about +5 dB on one and -5 dB on the other.  The signed median of
# that pooled set is ~0 and says nothing -- score the magnitude.
print(f'  solver shortfall there            {np.median(short[:, 0]):+.2f} / '
      f'{np.median(short[:, 1]):+.2f} dB on the two channels, '
      f'median |{np.median(np.abs(short)):.2f}| (gate is +/-{law_w.error_threshold_db:g})')
print(f'  trim on the conflicting channels  max |{np.abs(hot).max():.4f}| dB '
      f'over {hot.size} channel-bins')
print(f'  trim on the reachable channels    median {np.median(cold):+.2f} dB')
print(f'  clamp is                          +/-{law_w.response_trim_limit_db:g} dB')
assert np.median(np.abs(short)) > law_w.error_threshold_db, (
    'the construction is not actually unreachable -- test is not testing the gate')
assert np.all(hot == 0.0), 'THE TRIM WOUND UP AT BINS THE PLANT CANNOT REACH'
assert np.median(cold) < -1.0, 'reachable channels should have corrected'
print('  PASS -- achievability shortfall is not mistaken for model error')

print()
print('=' * 74)
print('TEST 4 -- the trim stands down across a test-level change')
print('=' * 74)
law_l = build_s(live, PT, S)
run_loop(law_l, Htrue, 8)
n_before = law_l.n_trim_updates
law_l.set_test_level_db(-12.0); law_l.set_test_level_db(0.0)
print(f'  frozen calls armed                {law_l._trim_frozen_calls}')
assert law_l._trim_frozen_calls == 3
Y = np.einsum('fmn,fnk,flk->fml', Htrue, law_l.output_cpsd, Htrue.conj())
for k in range(3):
    law_l.control(HS, CS, 101, 101, Y, law_l.output_cpsd)
print(f'  trim updates during the freeze    {law_l.n_trim_updates - n_before}')
assert law_l.n_trim_updates == n_before, 'trim updated on a level-ramp frame'
law_l.control(HS, CS, 101, 101, Y, law_l.output_cpsd)
print(f'  trim updates after the freeze     {law_l.n_trim_updates - n_before}')
assert law_l.n_trim_updates > n_before, 'trim never resumed'
print('  PASS -- ramp frames are not fed to the integrator')

print()
print('=' * 74)
print('ALL FOUR PASS')
print('=' * 74)
