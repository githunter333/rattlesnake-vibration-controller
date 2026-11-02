"""optimal_diagonal_control_fast must keep the fast path under ordinary FRF
averaging jitter and demote to the SDP when the plant model really moves."""
import sys, numpy as np, netCDF4 as nc4
ROOT='/sessions/rcw-01hncq3istbjnstukh3lgy2c/mnt/nhunterjr--Code--python--rattlesnake-vibration-controller'
sys.path.insert(0,ROOT)
from control_laws.optimal_diagonal_control_fast import optimal_diagonal_control_fast
RUNS=f'{ROOT}/examples/sixdrive12resp/results/runs'
d=nc4.Dataset(f'{RUNS}/run01_matchtrace_capon_sysid.nc4'); g=d['Frame 6x12 Random']
spec=g['specification_cpsd_matrix_real'][:]+1j*g['specification_cpsd_matrix_imag'][:]
H=g['frf_data_real'][:]+1j*g['frf_data_imag'][:]
ctrl=np.asarray(g['control_channel_indices'][:]).astype(int)
sr=g['response_cpsd_real'][:]+1j*g['response_cpsd_imag'][:]
rr=g['reference_cpsd_real'][:]+1j*g['reference_cpsd_imag'][:]
nr=g['response_noise_cpsd_real'][:]+1j*g['response_noise_cpsd_imag'][:]
nf=g['reference_noise_cpsd_real'][:]+1j*g['reference_noise_cpsd_imag'][:]
coh=np.asarray(g['frf_coherence'][:]); d.close()
H=np.asarray(H)[:,ctrl,:]; sr=np.asarray(sr)[:,ctrl][:,:,ctrl]; nr=np.asarray(nr)[:,ctrl][:,:,ctrl]
spec=np.asarray(spec)
if spec.shape[1]!=H.shape[1]: spec=spec[:,ctrl][:,:,ctrl]
coh=coh[:,ctrl] if coh.ndim>1 else coh
w=np.zeros(spec.shape[:2]); a=np.zeros(spec.shape[:2])
rng=np.random.default_rng(1)
def perturb(frac):
    n=rng.standard_normal(H.shape)+1j*rng.standard_normal(H.shape)
    n*= frac*np.linalg.norm(H)/np.linalg.norm(n)
    return H+n
for frac,label in ((0.0,'identical    '),(0.01,'1% jitter    '),
                   (0.04,'4% jitter    '),(0.20,'20% real move')):
    law=optimal_diagonal_control_fast(spec,w,a,'1e-6,0.05,4,1.0,0.95,4,-9.0',
                                      H,nr,nf,sr,rr,coh,100,100,None,None)
    f0,s0=law.n_fast_solves,law.n_safe_solves
    law.control(perturb(frac) if frac else H,coh,101,101,None,None)
    df,ds=law.n_fast_solves-f0,law.n_safe_solves-s0
    path='FAST' if df>0 and ds==0 else ('SDP ' if ds>0 and df==0 else 'mixed')
    print(f'FRF {label}  ->  {path}   (fast {df}, sdp {ds})')
