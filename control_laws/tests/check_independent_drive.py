"""Checks for independent_drive_control (Smallwood's independent drive algorithm).

Loads the law exactly as Rattlesnake does -- spec_from_file_location, no package
context -- then exercises it on two plants:

  1. a decoupled 6 x 6 plant, where independent drives MUST be exact and the
     returned drive CPSD must have no off-diagonal terms at all
  2. the RC plate (48 control accelerometers, 12 drives) against the BARC
     specification, where the exact branch never fires

and checks the PSD property, the startup ceiling, the generator form against the
function form, and nnls against the paper's literal clip-once rule.

Needs the rc_plate study data; run it with an environment that has netCDF4 and
scipy, e.g.

    /opt/anaconda3/envs/rattlesnake/bin/python3 check_independent_drive.py

The numbers it prints are the ones quoted in the law's module docstring.
"""
import importlib.util, os
import numpy as np

LAW = '/Users/nhunterjr/Code/python/rattlesnake-vibration-controller/control_laws/independent_drive_control.py'
spec_ = importlib.util.spec_from_file_location('independent_drive_control', LAW)
idc = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(idc)
print('loaded with no package context: OK')

# ---------- 1. a plant where independence MUST work: diagonal H ----------
n_f, n = 40, 6
rng = np.random.default_rng(0)
H = np.zeros((n_f, n, n), complex)
for f in range(n_f):
    H[f] = np.diag(rng.uniform(0.5, 2.0, n)*np.exp(1j*rng.uniform(0, 2*np.pi, n)))
tgt = rng.uniform(1.0, 4.0, (n_f, n))
S = np.zeros((n_f, n, n), complex)
S[:, np.arange(n), np.arange(n)] = tgt
X = idc.independent_drive_solve(S, H, startup_test_level_cap_db=1e6)
ach = np.real(np.einsum('fmm->fm', H @ X @ H.conj().transpose(0, 2, 1)))
offdiag = np.abs(X[:, ~np.eye(n, dtype=bool)]).max()
print(f'\ndecoupled plant: max |dB| error {np.abs(10*np.log10(ach/tgt)).max():.2e}, '
      f'max |off-diagonal drive| {offdiag:.2e}  (should both be ~0)')

# ---------- 2. the RC plate ----------
D = '/Users/nhunterjr/Code/python/rattlesnake-band-average/studies/rc_plate/'
z = np.load(D + 'rc_plate_state_space.npz')
A_ss, B, C, Dm = z['A'], z['B'], z['C'], z['D']
ctrl = np.where(z['channel_unit'] == 'g')[0]
sp = np.load(D + 'barc_response_spec.npz')
f_all, cpsd = sp['f'], sp['cpsd']
keep = np.linspace(0, len(f_all)-1, 300).astype(int)
fk, Sk = f_all[keep], cpsd[keep]
I = np.eye(A_ss.shape[0])
H = np.empty((len(fk), len(ctrl), B.shape[1]), complex)
for i, fi in enumerate(fk):
    H[i] = (C @ np.linalg.solve(1j*2*np.pi*max(fi, 1e-6)*I - A_ss, B) + Dm)[ctrl]

X, diag = idc.independent_drive_solve(Sk, H, startup_test_level_cap_db=1e6,
                                      return_diagnostics=True)
ib = diag['in_band']
tgt = np.real(np.einsum('fmm->fm', Sk))
ach = np.real(np.einsum('fmm->fm', H @ X @ H.conj().transpose(0, 2, 1)))
ok = ib[:, None] & (tgt > 0) & (ach > 0)
db = np.abs(10*np.log10(ach[ok]/tgt[ok]))
print(f'\nRC plate 48x12, {ib.sum()} in-band lines of {len(fk)}')
print(f'  feasible with independent drives : {diag["n_feasible"]} / {diag["n_in_band"]}')
print(f'  drives zeroed, median            : {np.median(diag["drives_zeroed"][ib]):.0f} of 12')
print(f'  best INDEPENDENT response, mean |dB| median : '
      f'{np.median(diag["mean_abs_independent_error_db"][ib]):.2f}')
print(f'  AFTER coupling, achieved vs spec mean |dB|  : {db.mean():.3f}  max {db.max():.3f}')
print(f'  mean drive coherence of the answer, median  : '
      f'{np.median(diag["mean_drive_coherence"][ib]):.3f}')

# PSD check
w = np.linalg.eigvalsh(0.5*(X[ib] + X[ib].conj().transpose(0, 2, 1)))
scale = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-300)
print(f'  most negative eigenvalue, relative          : {(w/scale).min():.2e}  (PSD if ~0)')

# ---------- 3. trace level honoured, and the generator path ----------
X2 = idc.independent_drive_solve(Sk, H, startup_test_level_cap_db=-9.0)
t_spec = tgt.sum(axis=1)
t_got = np.real(np.einsum('fmm->fm', H @ X2 @ H.conj().transpose(0, 2, 1))).sum(axis=1)
r = 10*np.log10(t_got[ib]/t_spec[ib])
print(f'\nwith the -9 dB startup ceiling: predicted trace re spec, '
      f'max {r.max():.2f} dB (must be <= -9.00)')

gen = idc.independent_drive_generator()
next(gen)
drive = gen.send((Sk, None, None, H, None, None, None, None, None, 1, 1,
                  '-0.05,1.0,-9.0', None, None))
print(f'generator agrees with the function form: '
      f'{np.allclose(drive, X2, equal_nan=True)}')

# ---------- 4. the clip-once rule versus nnls ----------
Xc = idc.independent_drive_solve(Sk, H, startup_test_level_cap_db=1e6, use_nnls=False)
achc = np.real(np.einsum('fmm->fm', H @ Xc @ H.conj().transpose(0, 2, 1)))
okc = ib[:, None] & (tgt > 0) & (achc > 0)
print(f"\npaper's clip-once rule: mean |dB| {np.abs(10*np.log10(achc[okc]/tgt[okc])).mean():.3f} "
      f"vs nnls {db.mean():.3f}")
