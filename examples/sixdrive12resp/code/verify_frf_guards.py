"""The two live-FRF guards added 2026-09-19 after run 30.

GUARD 1 (divergence): a live FRF that has walked away from the system
identification by more than _FRF_DIVERGENCE_LIMIT is not a plant model.
Run 30 reached since_initial median 39834, max 500217, and the law solved
against it and asked for 1.8e5 V.

GUARD 2 (deadlock escape): run 30's end state was absorbing -- near-zero
drive, so no excitation, so no FRF movement, so nothing scheduled, so the
same dead solution forever (70+ calls, bit-identical).

The third test is the one that matters most: on a HEALTHY FRF sequence
neither guard may fire, and the control output must be bit-identical to the
code before the guards went in.
"""
import sys, numpy as np, netCDF4 as nc4, importlib.util, shutil, os
ROOT='/sessions/rcw-01hncq3istbjnstukh3lgy2c/mnt/nhunterjr--Code--python--rattlesnake-vibration-controller'
sys.path.insert(0,ROOT)
RUNS=f'{ROOT}/examples/sixdrive12resp/results/runs'

d=nc4.Dataset(f'{RUNS}/run29_optdiagfast_updateon_gatefix_sysid.nc4'); g=d['Frame 6x12 Random']
spec=np.asarray(g['specification_cpsd_matrix_real'][:]+1j*g['specification_cpsd_matrix_imag'][:])
H=np.asarray(g['frf_data_real'][:]+1j*g['frf_data_imag'][:])
ctrl=np.asarray(g['control_channel_indices'][:]).astype(int)
sr=np.asarray(g['response_cpsd_real'][:]+1j*g['response_cpsd_imag'][:])
rr=np.asarray(g['reference_cpsd_real'][:]+1j*g['reference_cpsd_imag'][:])
nr=np.asarray(g['response_noise_cpsd_real'][:]+1j*g['response_noise_cpsd_imag'][:])
nf=np.asarray(g['reference_noise_cpsd_real'][:]+1j*g['reference_noise_cpsd_imag'][:])
coh=np.asarray(g['frf_coherence'][:]); d.close()
H=H[:,ctrl,:]; sr=sr[:,ctrl][:,:,ctrl]; nr=nr[:,ctrl][:,:,ctrl]
if spec.shape[1]!=H.shape[1]: spec=spec[:,ctrl][:,:,ctrl]
coh=coh[:,ctrl] if coh.ndim>1 else coh
w=np.zeros(spec.shape[:2]); a=np.zeros(spec.shape[:2])
PARAMS='1e-6,0.5,20,1.0,0.95,6'
rng=np.random.default_rng(7)

def build(mod):
    return mod.optimal_diagonal_control_fast(spec,w,a,PARAMS,H,nr,nf,sr,rr,coh,100,100,None,None)

def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m; sp.loader.exec_module(m); return m

fast=load(f'{ROOT}/control_laws/optimal_diagonal_control_fast.py','odcf_live')
import control_laws.optimal_diagonal_control as _odc
ESCAPE_CALLS=_odc._DEADLOCK_ESCAPE_CALLS
LIMIT=_odc._FRF_DIVERGENCE_LIMIT

print('='*74)
print('TEST 1 -- divergence guard rejects the bins that ran away, and only those')
print('='*74)
law=build(fast)
# Corrupt 40 IN-BAND bins.  The guard scores in-band only (fixed after run 31
# fired it on ~1050 out-of-band bins per call while controlling perfectly),
# so the test has to corrupt bins the guard actually looks at.
inb=np.where(np.asarray(spec.real.diagonal(axis1=1,axis2=2)).max(axis=1)>0)[0]
bad=np.zeros(H.shape[0],dtype=bool); bad[inb[200:240]]=True     # 40 in-band bins
Hbad=H.copy(); Hbad[bad]*=22325.0                          # run 30's measured factor
law.control(Hbad,coh,101,101,None,None)
print(f'  in-band bins             {inb.size}   (out of {H.shape[0]})')
print(f'  bins corrupted {int(bad.sum())}   bins rejected {law._frf_rejected}')
assert law._frf_rejected==int(bad.sum()), 'guard did not reject exactly the corrupted bins'
out=law.output_cpsd
drive=float(np.sum(np.real(np.einsum('fnn->fn',out))))
print(f'  total commanded drive {drive:.4e}  (finite: {np.isfinite(drive)})')
assert np.isfinite(drive)

print()
print('='*74)
print('TEST 2 -- deadlock escape fires, and only after the run really is stuck')
print('='*74)
law=build(fast)
law.control(H,coh,101,101,None,None)

# Reproduce run 30's END STATE faithfully, which is NOT "output is zero".
# It is a solution the law considers self-consistent -- every bin refined,
# nothing measured as degraded, the FRF static -- that nevertheless commands
# almost no drive.  err_db_cache = inf is what makes step 3 find nothing
# stale (err > inf + threshold is never true), which is exactly the
# stale_resolved=0 seen from call #109 onward.
law.output_cpsd *= 1e-6
law.sdp_refined[:]=True
law.err_db_cache[:]=np.inf
fired=None
for i in range(1,16):
    law._refine_batch(H)
    if law._n_escapes and fired is None: fired=i
print(f'  escape fired on starved call #{fired} (threshold {ESCAPE_CALLS})')
recovered=float(np.sum(np.real(np.einsum('fnn->fn',law.output_cpsd))))
print(f'  drive after escape {recovered:.4e}')
assert fired==ESCAPE_CALLS, f'escape fired on call {fired}, expected {ESCAPE_CALLS}'
assert recovered>0, 'escape did not restore a live solution'

print()
print('='*74)
print('TEST 3 -- REGRESSION: on a healthy FRF neither guard fires, and the')
print('          output is bit-identical to the pre-guard code')
print('='*74)
# Copy the WHOLE package: these modules load their siblings by file path
# (Rattlesnake loads control scripts standalone, so relative imports fail),
# so a two-file copy cannot resolve control_laws.py next to it.
ref_dir='/tmp/preguard'
if os.path.isdir(ref_dir): shutil.rmtree(ref_dir)
shutil.copytree(f'{ROOT}/control_laws', ref_dir)
shutil.copy('/tmp/odc_backup.py', f'{ref_dir}/optimal_diagonal_control.py')
old=load(f'{ref_dir}/optimal_diagonal_control_fast.py','odcf_old')

def sequence(mod):
    law=mod.optimal_diagonal_control_fast(spec,w,a,PARAMS,H,nr,nf,sr,rr,coh,100,100,None,None)
    r=np.random.default_rng(7)
    outs=[]
    for k in range(6):
        n=r.standard_normal(H.shape)+1j*r.standard_normal(H.shape)
        n*=0.03*np.linalg.norm(H)/np.linalg.norm(n)     # ordinary averaging jitter
        law.control(H+n,coh,101+k,101+k,None,None)
        outs.append(law.output_cpsd.copy())
    return law,outs

law_new,new_outs=sequence(fast)
law_old,old_outs=sequence(old)
diff=max(float(np.max(np.abs(a1-a2))) for a1,a2 in zip(new_outs,old_outs))
print(f'  guard rejections over the sequence : {law_new._frf_rejected}')
print(f'  deadlock escapes over the sequence : {law_new._n_escapes}')
print(f'  max |new - old| output_cpsd        : {diff:.3e}')
assert law_new._frf_rejected==0 and law_new._n_escapes==0, 'a guard fired on healthy data'
assert diff==0.0, 'guards changed the healthy-case answer'
print()
print('ALL THREE PASS')
