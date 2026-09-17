"""Every law with a startup level cap must clamp its FIRST command to
startup_test_level_cap_db re spec -- with the drive-coherence cap on AND off."""
import sys, numpy as np, netCDF4 as nc4
ROOT='/sessions/rcw-01hncq3istbjnstukh3lgy2c/mnt/nhunterjr--Code--python--rattlesnake-vibration-controller'
sys.path.insert(0, ROOT)
from control_laws import control_laws as cl
from control_laws.diagonal_congruence_control import match_diagonal_congruence
from control_laws.optimal_diagonal_control import optimal_diagonal_control
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
CAP=-9.0
st=np.real(np.trace(spec,axis1=1,axis2=2)); m=st>0

def score(out):
    pred=H@out@H.conj().transpose(0,2,1)
    ot=np.real(np.trace(pred,axis1=1,axis2=2))
    agg=10*np.log10(ot[m].sum()/st[m].sum()); worst=(10*np.log10(np.maximum(ot[m],1e-300)/st[m])).max()
    ok='OK  ' if worst<=CAP+1e-6 else 'FAIL'
    return f'{ok} aggregate {agg:+7.2f} dB   worst line {worst:+7.2f} dB'

def row(name,label,out): print(f'{name:30s} {label}  {score(out)}')

FN=((cl.pseudoinverse_control,'pseudoinverse_control'),
    (cl.match_trace_pseudoinverse,'match_trace_pseudoinverse'),
    (cl.buzz_control,'buzz_control'))
for law,name in FN:
    for cap,label in ((f'1e-3,1.0,{CAP}','coh off '),(f'1e-3,0.95,{CAP}','coh 0.95')):
        row(name,label,law(spec,w,a,H,nr,nf,sr,rr,coh,100,100,cap,None,None))

CLS=((cl.match_trace_pseudoinverse_pi,'match_trace_pseudoinverse_pi'),
     (cl.buzz_feedback,'buzz_feedback'),
     (cl.match_trace_pi_resolve,'match_trace_pi_resolve'),
     (cl.match_trace_resolve,'match_trace_resolve'),
     (match_diagonal_congruence,'match_diagonal_congruence'))
for C,name in CLS:
    for cap,label in ((f'1e-3,1.0,{CAP}','coh off '),(f'1e-3,0.95,{CAP}','coh 0.95')):
        o=C(spec,w,a,cap,H,nr,nf,sr,rr,coh,100,100,None,None)
        row(name,label,o.control(H,coh,100,100,None,None))

for C,name in ((optimal_diagonal_control,'optimal_diagonal_control'),
               (optimal_diagonal_control_fast,'optimal_diagonal_control_fast')):
    for cap,label in ((f'1e-6,0.05,4,1.0,1.0,4,{CAP}','coh off '),
                      (f'1e-6,0.05,4,1.0,0.95,4,{CAP}','coh 0.95')):
        o=C(spec,w,a,cap,H,nr,nf,sr,rr,coh,100,100,None,None)
        row(name,label,o.control(H,coh,100,100,None,None))
