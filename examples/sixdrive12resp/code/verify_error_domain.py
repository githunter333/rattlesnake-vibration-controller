"""Score the optimal-diagonal SOLVERS in the LINEAR and dB error domains
against the reachability floor, on run 14's own identification, reporting
DRIVE DEMAND alongside accuracy.

Calls _solve_one_bin directly rather than driving the control loop: the bin
budget and refinement order are identical between domains, so the solver is
the only thing that differs, and this isolates it in seconds rather than tens
of minutes.

Both columns matter. The linear objective caps what abandoning a channel costs
at that channel's own target squared, so on a plant with unequal row gain it
sells the hardest channels. The dB objective removes that ceiling -- and with
it, any limit on the drive it will spend chasing a barely-reachable channel,
which is what drive_rcond exists to bound. Read accuracy and drive together.

    DECIM=15 python verify_error_domain.py      # every 15th in-band line
"""
import importlib.util as u, numpy as np, netCDF4 as nc4, io, contextlib, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
def load(n, rel):
    s = u.spec_from_file_location(n, os.path.join(ROOT, rel))
    m = u.module_from_spec(s); s.loader.exec_module(m); return m
ach  = load('ar',   'control_laws/achievable_response.py').achievable_diagonal
Base = load('odc',  'control_laws/optimal_diagonal_control.py').optimal_diagonal_control
Fast = load('odcf', 'control_laws/optimal_diagonal_control_fast.py').optimal_diagonal_control_fast

RUN = os.path.join(ROOT, 'examples/sixdrive12resp/results/runs/'
                         'run14_optdiagfast_updateoff_sysid.nc4')
d = nc4.Dataset(RUN); g = d[[x for x in d.groups if x != 'channels'][0]]
spec = np.array(g['specification_cpsd_matrix_real'][:]).astype(complex)
H = np.array(g['frf_data_real'][:]) + 1j*np.array(g['frf_data_imag'][:])
ctrl = np.array(g['control_channel_indices'][:]).astype(int)
f = np.array(g['specification_frequency_lines'][:]); d.close()
H = H[:, ctrl, :]
if spec.shape[1] != 8: spec = spec[:, ctrl][:, :, ctrl]
tgt = np.real(np.einsum('fmm->fm', spec))
band = (tgt.max(axis=1) > 0) & (f >= 100) & (f <= 1000)
z = np.zeros(spec.shape[:2])
NAMES = ['7X+','8X+','9X+','10X+','11X+','12X+','13X+','14X+']
DECIM = int(os.environ.get('DECIM', '15'))
idx = np.flatnonzero(band)[::DECIM]

fl = ach(H, tgt, line_indices=idx, restrict_rcond=1e-3, progress=None)
sol = np.flatnonzero(np.asarray(fl['solved'], bool))
with np.errstate(divide='ignore', invalid='ignore'):
    _e = 10*np.log10(np.asarray(fl['achieved'])[sol]/tgt[sol])
EF = np.where(np.isfinite(_e), _e, 0.0)

def score(Law, domain, fast_path):
    params = f'1e-6,0.05,20,1.0,0.95,6,-9.0,{0 if domain=="linear" else 1}'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        law = Law(spec, z, z, params, None, None, None, None, None, None,
                  None, None, None, None)
        if fast_path:
            law._h_changed_this_call = False        # force the BM path
        E = []; tr = []; t = time.time()
        for i in sol:
            X = law._solve_one_bin(H[i], tgt[i])
            a = np.real(np.einsum('mn,nk,mk->m', H[i], X, H[i].conj()))
            with np.errstate(divide='ignore', invalid='ignore'):
                e = 10*np.log10(a/tgt[i])
            E.append(np.where(np.isfinite(e), e, 0.0))
            tr.append(float(np.real(np.trace(X))))
        ms = (time.time()-t)/len(sol)*1000
    return np.array(E), np.array(tr), ms

print(f'run 14 FRF, {sol.size} lines (every {DECIM}th in band), floor restrict_rcond=1e-3')
print(f'{"solver":34s} ' + ' '.join(f'{n:>6s}' for n in NAMES) +
      f' {"POOL":>6s} {"ms/bin":>7s} {"drive":>8s} {"worst":>7s}')
print(f'{"reachability floor":34s} ' +
      ' '.join(f'{np.sqrt(np.mean(EF[:,m]**2)):6.2f}' for m in range(8)) +
      f' {np.sqrt(np.mean(EF**2)):6.2f} {"--":>7s} {"--":>8s} {"--":>7s}')
for Law, fastp, lname in ((Base, False, 'optimal_diagonal      SDP'),
                          (Fast, True,  'optimal_diagonal_fast  BM')):
    ref = None
    for dom in ('linear', 'db'):
        E, tr, ms = score(Law, dom, fastp)
        if ref is None:
            ref = tr
        rel = 10*np.log10(np.maximum(tr, 1e-300)/np.maximum(ref, 1e-300))
        print(f'{lname+" ["+dom+"]":34s} ' +
              ' '.join(f'{np.sqrt(np.mean(E[:,m]**2)):6.2f}' for m in range(8)) +
              f' {np.sqrt(np.mean(E**2)):6.2f} {ms:7.1f} {rel.mean():+8.2f} {rel.max():+7.2f}')
print()
print('drive columns are dB relative to that law\'s own linear-objective solve.')
