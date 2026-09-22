"""The gated drive-shape refresh added to match_trace_pseudoinverse, 2026-09-22.

Before this, with "Update Transfer Function During Control" ON, the law read
the FRF exactly once -- in the cycle-1 pseudoinverse -- and never again.
Steady state was output = last_output * trace_ratio, a per-line real scalar.
It tracked LEVEL against a re-estimated plant while holding a drive SHAPE
synthesised from the plant as it looked before the test started.
refresh_shape (field 4) closed that, but only by re-solving every line every
cycle -- and run 35 measured the in-band FRF moving a median 0.185 per cycle
with a model mismatch of just 0.060 dB rms, so almost all of that movement is
estimator noise. Re-solving on it is re-solving on noise.

Field 5, refresh_threshold, gates the refresh per line on relative Frobenius
change since that LINE's last solve. Field 6, frf_diag, records the per-line
movement distribution so the threshold can be set from measurement.

TEST 1  REGRESSION. The two-field string every archived run used must be
        bit-identical to the previous commit, over several cycles.
TEST 2  threshold 0 == the old refresh_shape, bit for bit, so an existing
        five-field string is unchanged.
TEST 3  PARTIAL REFRESH. The one that can corrupt silently: lines below the
        threshold must come out EXACTLY as the no-refresh path would leave
        them, while lines above it take the new shape.
TEST 4  The diagnostics must not touch the command.
TEST 5  The gate must actually fire on drift and stay shut on noise.
"""
import sys, os, numpy as np, netCDF4, importlib.util, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
REF = os.path.join(os.path.expanduser('~'), 'work', 'head2', 'control_laws')

f = sorted(glob.glob(os.path.join(ROOT, 'examples/sixdrive12resp/results/runs/run33*_sysid.nc4')))[0]
ds = netCDF4.Dataset(f); g = ds['Frame 6x12 Random']
S = np.array(g['specification_cpsd_matrix_real'][:]) + 1j*np.array(g['specification_cpsd_matrix_imag'][:])
H0 = np.array(g['frf_data_real'][:]) + 1j*np.array(g['frf_data_imag'][:])
ctrl = np.array(g['control_channel_indices'][:]).astype(int)
fr = np.array(g['specification_frequency_lines'][:], float); ds.close()
if H0.shape[1] != S.shape[1]: H0 = H0[:, ctrl, :]
# a small in-band slice keeps every test in seconds
sd = np.real(np.einsum('fmm->fm', S))
inb = np.where((sd.max(axis=1) > 0) & (fr >= 100) & (fr <= 1000))[0][:150]
S, H0 = S[inb], H0[inb]
M, N = S.shape[1], H0.shape[2]
W = np.zeros(S.shape[:2])

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m

live = load(os.path.join(ROOT, 'control_laws', 'control_laws.py'), 'cl_live')
base = load(os.path.join(REF, 'control_laws.py'), 'cl_head')

def call(mod, params, H, last_resp, last_out):
    return mod.match_trace_pseudoinverse(S, W, W, H, None, None, None, None,
                                         None, 20, 20, params, last_resp, last_out)

def loop(mod, params, n, Hseq, Htrue=None):
    """Close the loop: command, measure through Htrue, feed back."""
    resp = out = None
    for k in range(n):
        H = Hseq(k)
        out = call(mod, params, H, resp, out)
        Ht = Htrue(k) if Htrue is not None else H
        resp = np.einsum('fmn,fnk,flk->fml', Ht, out, Ht.conj())
    return out

static = lambda k: H0
rng = np.random.default_rng(11)
NOISE = [ (1 + 0.02*rng.standard_normal(H0.shape)) for _ in range(40) ]
noisy = lambda k: H0*NOISE[k % len(NOISE)]
def gain_only(k):                     # per-line GAIN drift: the level loop
    return H0*(1 + 0.05*k)             # absorbs this, no re-solve needed
def shape_drift(k):                   # relative change ACROSS the drives
    Hs = H0.copy(); Hs[:, :, 0] *= (1 + 0.08*k); return Hs

def loop_count(mod, params, n, Hseq):
    """Run the loop and total up how many line-refreshes happened."""
    resp = out = None; total = 0
    for k in range(n):
        out = call(mod, params, Hseq(k), resp, out)
        total += mod._MT_FRF_STATE.get('last_mask_count', 0)
        Ht = Hseq(k)
        resp = np.einsum('fmn,fnk,flk->fml', Ht, out, Ht.conj())
    return total

print('='*74); print('TEST 1 -- the archived two-field string is untouched'); print('='*74)
a = loop(live, '1e-3,0.95', 6, static); b = loop(base, '1e-3,0.95', 6, static)
d = float(np.max(np.abs(a-b)))
print(f'  max |new - previous commit|   {d:.3e}')
assert d == 0.0, 'THE DEFAULT PATH CHANGED'
print('  PASS')

print(); print('='*74); print('TEST 2 -- threshold 0 reproduces the old refresh_shape'); print('='*74)
a = loop(live, '1e-3,0.95,-9.0,,1,0', 6, noisy); b = loop(base, '1e-3,0.95,-9.0,,1', 6, noisy)
d = float(np.max(np.abs(a-b)))
print(f'  max |gated(0) - ungated|      {d:.3e}')
assert d == 0.0, 'threshold 0 is not the old behaviour'
print('  PASS')

print(); print('='*74); print('TEST 3 -- partial refresh leaves the other lines exactly alone'); print('='*74)
# A genuine SHAPE change on the first 40 lines only, so the gate splits the
# band.  It has to be a shape change: a pure per-line gain change is absorbed
# exactly by trace_ratio, and refreshing those lines moves the command by
# 7e-11 -- which is what the first version of this test accidentally measured.
def split(k):
    Hs = H0.copy(); Hs[:40, :, 0] *= (1 + 0.20*k)
    return Hs
gated = loop(live, '1e-3,0.95,-9.0,,1,0.10', 5, split)
noref = loop(live, '1e-3,0.95', 5, split)
lo = float(np.max(np.abs(gated[40:] - noref[40:])))
hi = float(np.max(np.abs(gated[:40] - noref[:40])))
print(f'  lines BELOW threshold: max |gated - no-refresh|  {lo:.3e}   (must be 0)')
print(f'  lines ABOVE threshold: max |gated - no-refresh|  {hi:.3e}   (must be > 0)')
assert lo == 0.0, 'UNREFRESHED LINES WERE DISTURBED'
assert hi > 0.0, 'the refreshed lines did not actually change'
print('  PASS')

print(); print('='*74); print('TEST 4 -- diagnostics do not touch the command'); print('='*74)
a = loop(live, '1e-3,0.95,-9.0,,1,0.10,1', 4, split)
b = loop(live, '1e-3,0.95,-9.0,,1,0.10,0', 4, split)
d = float(np.max(np.abs(a-b)))
print(f'  max |diag on - diag off|      {d:.3e}')
assert d == 0.0, 'the instrumentation changed the output'
print('  PASS')

print(); print('='*74); print('TEST 5 -- fires on drift, shut on noise'); print('='*74)
# Totals across the run, not the last cycle: once a line re-solves, its
# reference moves with it, so the last cycle can legitimately be quiet.
for name, seq, expect in [('2% per-cycle noise', noisy, 'none'),
                          ('5% per-cycle GAIN drift', gain_only, 'none'),
                          ('8% per-cycle SHAPE drift', shape_drift, 'some')]:
    t = loop_count(live, '1e-3,0.95,-9.0,,1,0.10', 10, seq)
    print(f'  {name:26s} total line-refreshes over 10 cycles: {t:5d}')
    if expect == 'none':
        assert t == 0, f'the gate fired on {name}'
    else:
        assert t > 0, 'the gate never fired on a real shape change'
print('  PASS -- fires on shape change, ignores both noise and pure gain drift')

print(); print('='*74); print('ALL FIVE PASS'); print('='*74)
