"""Checks for pareto_target_control (Coletti, Davis & Schultz 2026).

Loads the module exactly as Rattlesnake loads a control law -- spec_from_file_
location, no package context -- and then:

  1. a decoupled 6 x 6 plant, where exact control must be feasible and the
     minimum-trace solution must come out diagonal
  2. EXACT feasibility against channel count on the frame model, plus the RC
     plate.  This is where the paper's n_resp <= n_drive^2 parameter count is
     shown to be necessary and not sufficient: the practical rule is
     n_resp <= rank(H)
  3. the Pareto curve on the rig, its automatic lambda, and whether it
     dominates the independent drives method the paper benchmarks against
  4. the three target-definition cases of Section 2.2, each checked against the
     constraint it is supposed to satisfy
  5. the dB caveat of Section 4.3, measured
  6. the law form, the generator, and the startup ceiling

    /opt/anaconda3/envs/rattlesnake/bin/python3 check_pareto_target.py

The numbers it prints are the ones quoted in the module docstring.
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from frame_plant import frame_plant                                # noqa: E402


def load(name):
    path = os.path.join(CL, name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(name)[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ptc = load('pareto_target_control.py')
idc = load('independent_drive_control.py')
print('loaded with no package context: OK  (solver %s)' % ptc._solver())


def diag_of(M):
    return np.real(np.einsum('...mm->...m', M))


def response(H, X):
    return H @ X @ H.conjugate().transpose(0, 2, 1)


def mean_abs_db(H, X, y, mask):
    ach = diag_of(response(H, X))
    ok = mask[:, None] & (y > 0) & (ach > 0)
    return float(np.abs(10*np.log10(ach[ok]/y[ok])).mean())


def worst_eig(X, mask):
    w = np.linalg.eigvalsh(0.5*(X[mask] + X[mask].conjugate().transpose(0, 2, 1)))
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-300)
    return float((w/scale).min())


def idm_target(H, y):
    """The paper's Eq. (10): the specification diagonal with the correlation and
    phase of H H^H, the response to uncorrelated UNIT drives.

    This is the paper's IDM baseline, and it is the buzz construction.  Note it
    is NOT the same object as independent_drive_control, which is Smallwood's
    later variant: there the independent drive LEVELS are solved from |H|^2
    first, so the cross terms come from uncorrelated drives at solved levels
    rather than at unit level.
    """
    B = H @ H.conjugate().transpose(0, 2, 1)
    d = np.sqrt(np.maximum(diag_of(B), 0.0))
    denom = d[:, :, None]*d[:, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = np.where(denom > 0, B/denom, 0.0)
    amp = np.sqrt(np.maximum(y, 0.0))
    Y = unit*(amp[:, :, None]*amp[:, None, :])
    idx = np.arange(H.shape[1])
    Y[:, idx, idx] = y
    return Y


# ---------- 1. a decoupled plant: exact control must be feasible ----------
n = 6
H1 = (np.eye(n)*np.linspace(1.0, 3.0, n))[None].astype(complex)
S1 = np.zeros((1, n, n), complex)
y1 = np.linspace(1.0, 4.0, n)*1e-3
S1[0, np.arange(n), np.arange(n)] = y1

feas1 = ptc.exact_feasible(S1, H1)
X1, i1 = ptc.pareto_inputs(S1, H1, lam=0.0)
off1 = np.abs(X1[0][~np.eye(n, dtype=bool)]).max()
print('\ndecoupled 6x6: exact feasible %d/%d, taken by the exact branch %d/%d'
      % (feas1['n_feasible'], feas1['n_tested'], i1['n_exact'], 1))
print('  max |dB| error %.2e, max |off-diagonal drive| %.2e  (both ~0)'
      % (mean_abs_db(H1, X1, y1[None], np.ones(1, bool)), off1))

# ---------- 2. exact feasibility where independent drives failed ----------
print('\nEXACT feasibility with a FULL Hermitian drive CPSD, Eq. (3)')
print('  the paper\'s count says feasible if n_resp <= n_drive^2; measured:')

rig = frame_plant('6x8')
Hr, Sr = rig['H'], rig['spec']
yr, mr = ptc._spec_diagonal(Sr), rig['band']
lines_r = np.nonzero(mr)[0]


def sub_layout(plant, n_resp):
    y = ptc._spec_diagonal(plant['spec'])[:, :n_resp]
    S = np.zeros((plant['H'].shape[0], n_resp, n_resp), complex)
    S[:, np.arange(n_resp), np.arange(n_resp)] = y
    return S, plant['H'][:, :n_resp, :], y


for n_resp in (4, 5, 6, 8):
    Ss, Hs, ys = sub_layout(rig, n_resp)
    rank = int(np.linalg.matrix_rank(Hs[lines_r[0]]))
    t0 = time.time()
    fs = ptc.exact_feasible(Ss, Hs)
    rr = fs['residual'][fs['in_band']]
    rr = rr[np.isfinite(rr)]
    print('  %d resp x 6 drives, rank %d (%d <= 36): %3d / %3d feasible; '
          'relative residual median %.1e min %.1e   [%.1f s]'
          % (n_resp, rank, n_resp, fs['n_feasible'], fs['n_tested'],
             np.median(rr), rr.min(), time.time() - t0))
print('  -> the rig is the 8-response row: the parameter count predicted '
      'feasible, it is not.')
print('  -> independent_drive_control found 0/181 here too, so on this rig '
      'that is NOT')
print('     an artifact of its diagonal-only drive.')

D = '/Users/nhunterjr/Code/python/rattlesnake-band-average/studies/rc_plate/'
z = np.load(D + 'rc_plate_state_space.npz')
A_ss, B_ss, C_ss, D_ss = z['A'], z['B'], z['C'], z['D']
ctrl = np.where(z['channel_unit'] == 'g')[0]
sp = np.load(D + 'barc_response_spec.npz')
f_all, cpsd = sp['f'], sp['cpsd']
keep = np.linspace(0, len(f_all) - 1, 300).astype(int)
fk, Sp = f_all[keep], cpsd[keep]
Ieye = np.eye(A_ss.shape[0])
Hp = np.empty((len(fk), len(ctrl), B_ss.shape[1]), complex)
for i, fi in enumerate(fk):
    Hp[i] = (C_ss @ np.linalg.solve(1j*2*np.pi*max(fi, 1e-6)*Ieye - A_ss, B_ss)
             + D_ss)[ctrl]
yp = ptc._spec_diagonal(Sp)
mp = yp.max(axis=1) > 0
sub = np.nonzero(mp)[0][::6]
t0 = time.time()
fp = ptc.exact_feasible(Sp, Hp, lines=sub)
rrp = fp['residual'][sub]
rrp = rrp[np.isfinite(rrp)]
print('  RC plate 48 resp x 12 drives, rank %d (48 <= 144): %3d / %3d '
      'feasible; relative residual median %.1e min %.1e   [%.1f s, every 6th '
      'line]'
      % (int(np.linalg.matrix_rank(Hp[sub[0]])), fp['n_feasible'],
         fp['n_tested'], np.median(rrp), rrp.min(), time.time() - t0))
print('  statuses seen: %s' % fp['statuses'])

# ---------- 3. the Pareto curve on the rig, against the IDM ----------
t0 = time.time()
curve = ptc.pareto_curve(Sr, Hr)
print('\nPareto curve on the rig, %d lambdas x %d lines  [%.1f s]'
      % (len(curve['lambdas']), len(lines_r), time.time() - t0))
db_lam, db_index = ptc.select_pareto_parameter(
    curve['lambdas'], curve['input_energy'], curve['residual'])
best_db = int(np.nanargmin(curve['mean_abs_db']))
for i, (lam, inp, res, mdb) in enumerate(zip(
        curve['lambdas'], curve['input_energy'], curve['residual'],
        curve['mean_abs_db'])):
    mark = ''
    if i == curve['lambda_index']:
        mark += "  <- Section 3.1 picks this"
    if i == best_db:
        mark += '  <- lowest mean |dB|'
    print('  lambda %-9.4g residual %-10.6g input energy %-12.6g '
          'mean |dB| %7.3f%s' % (lam, res, inp, mdb, mark))
print('  the paper\'s criterion and our dB metric disagree by %.1f dB here; '
      'the energy axis'
      % abs(curve['mean_abs_db'][curve['lambda_index']]
            - curve['mean_abs_db'][best_db]))
print('  spans %.0e and the residual axis only %.1fx, so log(res)+log(inp) is '
      'dominated by energy.'
      % (curve['input_energy'].max()/max(curve['input_energy'].min(), 1e-300),
         curve['residual'].max()/curve['residual'].min()))

Yi = idm_target(Hr, yr)
idm_inp, idm_res = [], []
for lam in curve['lambdas']:
    Zl = ptc._regularized_pinv(Hr, lam)
    Xi = Zl @ Yi @ Zl.conjugate().transpose(0, 2, 1)
    idm_inp.append(float(np.real(np.einsum('fjj->f', Xi))[mr].sum()))
    idm_res.append(float(np.abs(yr - diag_of(response(Hr, Xi)))[mr].sum()))
idm_inp, idm_res = np.asarray(idm_inp), np.asarray(idm_res)

pr_inp, pr_res = curve['input_energy'], curve['residual']
dominated = sum(bool(np.any((pr_inp <= i_e*(1 + 1e-9))
                            & (pr_res <= r_e*(1 + 1e-9))
                            & ((pr_inp < i_e) | (pr_res < r_e))))
                for i_e, r_e in zip(idm_inp, idm_res))
print('  IDM points strictly dominated by a Pareto point: %d / %d'
      % (dominated, len(idm_inp)))
lam_i, _ = ptc.select_pareto_parameter(curve['lambdas'], idm_inp, idm_res)
print('  at its own selected lambda, IDM residual %.6g energy %.6g'
      % (idm_res[list(curve['lambdas']).index(lam_i)],
         idm_inp[list(curve['lambdas']).index(lam_i)]))
print('  Pareto at its selected lambda, residual %.6g energy %.6g'
      % (pr_res[curve['lambda_index']], pr_inp[curve['lambda_index']]))

# select='db' is the departure of VERIFIED item 2; exercise it on a short grid.
small = (0.0, 1.0, 100.0)
c_db = ptc.pareto_curve(Sr, Hr, lambdas=small, select='db')
c_paper = ptc.pareto_curve(Sr, Hr, lambdas=small, select='paper')
print("  select='db' picks lambda %-9.4g (mean |dB| %.3f); 'paper' picks "
      "%-9.4g (mean |dB| %.3f)"
      % (c_db['lambda_hat'], c_db['mean_abs_db'][c_db['lambda_index']],
         c_paper['lambda_hat'],
         c_paper['mean_abs_db'][c_paper['lambda_index']]))
assert (c_db['mean_abs_db'][c_db['lambda_index']]
        <= c_paper['mean_abs_db'][c_paper['lambda_index']] + 1e-9), \
    "select='db' did not pick the lowest dB point"

# ---------- 4. the three target cases of Section 2.2 ----------
print('\ntarget definition, Section 2.2')
lam_hat = curve['lambda_hat']
Xr = curve['inputs']

def target_db(Y, mask):
    return float(np.abs(10*np.log10(np.maximum(diag_of(Y), 1e-300)
                                    / np.maximum(yr, 1e-300)))[mask].mean())


# The cheap target Z+ X Z+^H inverts a REGULARIZED pseudoinverse, so how usable
# it is depends entirely on lambda: at large lambda the factor
# sigma/(sigma^2+lambda) is tiny in the weak directions and Z+ is correspondingly
# enormous.  Measured at both ends.
for lam_try, label in ((0.0, 'lambda 0        '),
                       (lam_hat, 'lambda %-9.4g' % lam_hat)):
    Xt, _ = ptc.pareto_inputs(Sr, Hr, lam=lam_try)
    Yt, _ = ptc.pareto_target(Sr, Hr, Xt, lam_try, mode='feasible')
    Zt = ptc._regularized_pinv(Hr, lam_try)
    bk = Zt @ Yt @ Zt.conjugate().transpose(0, 2, 1)
    print('  overdetermined 8x6, feasible mode, %s: constraint %.2e, '
          'target diagonal mean |dB| %.3f, PSD %.2e'
          % (label, np.abs(bk[mr] - Xt[mr]).max()
             / max(np.abs(Xt[mr]).max(), 1e-300),
             target_db(Yt, mr), worst_eig(Yt, mr)))

Yf, inf_f = ptc.pareto_target(Sr, Hr, Xr, lam_hat, mode='feasible')
Zl = ptc._regularized_pinv(Hr, lam_hat)
back = Zl @ Yf @ Zl.conjugate().transpose(0, 2, 1)

few = lines_r[::20]
t0 = time.time()
Yo, inf_o = ptc.pareto_target(Sr, Hr, Xr, lam_hat, mode='optimal', lines=few)
back_o = (Zl[few] @ Yo[few] @ Zl[few].conjugate().transpose(0, 2, 1))
print('  overdetermined 8x6, optimal mode (Eq. 5, %d lines) [%.1f s]'
      % (len(few), time.time() - t0))
print('    constraint |Z Y Z^H - X| / max|X| = %.2e, diagonal mean |dB| %.3f'
      % (np.abs(back_o - Xr[few]).max()/max(np.abs(Xr[few]).max(), 1e-300),
         float(np.abs(10*np.log10(np.maximum(diag_of(Yo[few]), 1e-300)
                                  / np.maximum(yr[few], 1e-300))).mean())))
print('    feasible mode on the same lines, diagonal mean |dB| %.3f'
      % float(np.abs(10*np.log10(np.maximum(diag_of(Yf[few]), 1e-300)
                                 / np.maximum(yr[few], 1e-300))).mean()))

even = frame_plant('6x8')
He = even['H'][:, :6, :]
Se = np.zeros_like(even['spec'][:, :6, :6])
ye = ptc._spec_diagonal(even['spec'])[:, :6]
Se[:, np.arange(6), np.arange(6)] = ye
me = even['band']
Xe, _ = ptc.pareto_inputs(Se, He, lam=1e-3)
Ye, inf_e = ptc.pareto_target(Se, He, Xe, 1e-3)
Ze = ptc._regularized_pinv(He, 1e-3)
backe = Ze @ Ye @ Ze.conjugate().transpose(0, 2, 1)
print('  even-determined 6x6 (%s): max |Z Y Z^H - X| / max|X| = %.2e'
      % (inf_e['case'],
         np.abs(backe[me] - Xe[me]).max()/max(np.abs(Xe[me]).max(), 1e-300)))

Hu = even['H'][:, :4, :]
Su = np.zeros_like(even['spec'][:, :4, :4])
yu = ptc._spec_diagonal(even['spec'])[:, :4]
Su[:, np.arange(4), np.arange(4)] = yu
Xu, _ = ptc.pareto_inputs(Su, Hu, lam=1e-3)
Yu, inf_u = ptc.pareto_target(Su, Hu, Xu, 1e-3)
e_before = float(np.real(np.einsum('fjj->f', Xu))[me].sum())
e_after = float(np.real(np.einsum('fjj->f', inf_u['inputs']))[me].sum())
out_same = np.abs(diag_of(response(Hu, inf_u['inputs']))[me]
                  - diag_of(response(Hu, Xu))[me]).max()
print('  underdetermined 4x6 (%s), Eq. (9) refinement applied %s'
      % (inf_u['case'], inf_u['refined']))
print('    input energy %.6g -> %.6g (must not rise), '
      'output diagonal unchanged to %.2e'
      % (e_before, e_after, out_same))

# ---------- 5. the Section 4.3 dB caveat, measured ----------
print('\nSection 4.3: the objective is an ABSOLUTE residual, this study scores dB')
for lam in (0.0, curve['lambda_hat']):
    Xl, _ = ptc.pareto_inputs(Sr, Hr, lam=lam)
    achl = diag_of(response(Hr, Xl))
    ok = mr[:, None] & (yr > 0)
    near_zero = int((achl[ok] < 1e-3*yr[ok]).sum())
    print('  lambda %-9.4g mean |dB| %7.3f   bins driven below -30 dB of spec: '
          '%d of %d' % (lam, mean_abs_db(Hr, Xl, yr, mr), near_zero,
                        int(ok.sum())))
assert curve['mean_abs_db'][best_db] <= curve['mean_abs_db'][
    curve['lambda_index']] + 1e-9, 'dB column is not consistent with selection'

Xw, _ = ptc.pareto_inputs(Sr, Hr, lam=curve['lambda_hat'], weighted=True)
print('  weighted=1 (our departure) at the same lambda, mean |dB| %7.3f'
      % mean_abs_db(Hr, Xw, yr, mr))
Xid = idc.independent_drive_solve(Sr, Hr, startup_test_level_cap_db=1e6)
print('  independent_drive_control for reference,     mean |dB| %7.3f'
      % mean_abs_db(Hr, Xid, yr, mr))

# ---------- 6. the law, the generator, the ceiling ----------
print('\nthe law form')
Xlaw = ptc.pareto_optimal_control(Sr, None, None, Hr, None, None, None, None,
                                  None, 1, 1, '0.0,1.0,1e6')
print('  PSD of the returned drive CPSD: %.2e' % worst_eig(Xlaw, mr))
Xcap = ptc.pareto_optimal_control(Sr, None, None, Hr, None, None, None, None,
                                  None, 1, 1, '0.0,1.0,-9.0')
tr = 10*np.log10(np.real(np.einsum('fmm->f', response(Hr, Xcap)))[mr]
                 / yr.sum(axis=1)[mr])
print('  with the -9 dB startup ceiling: predicted trace re spec, max %.2f dB'
      ' (must be <= -9.00)' % tr.max())
gen = ptc.pareto_optimal_generator()
next(gen)
Xgen = gen.send((Sr, None, None, Hr, None, None, None, None, None, 1, 1,
                 '0.0,1.0,-9.0', None, None))
print('  generator agrees with the function form: %s'
      % np.allclose(Xgen, Xcap, equal_nan=True))

assert worst_eig(Xlaw, mr) > -1e-9, 'returned drive CPSD is not PSD'
assert tr.max() <= -9.0 + 1e-6, 'startup ceiling not honoured'
assert np.allclose(Xgen, Xcap, equal_nan=True), 'generator disagrees'
assert e_after <= e_before*(1 + 1e-9), 'Eq. (9) raised the input energy'
print('\nall assertions passed')
