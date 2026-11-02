"""FRF-update behaviour.
(a) a constant FRF must reproduce the pre-change law exactly
(b) a CHANGED FRF must now reach the laws that used to ignore it
(c) the open-loop laws must not drive through the running ceiling
"""
import sys, numpy as np, netCDF4 as nc4
ROOT='/sessions/rcw-01hncq3istbjnstukh3lgy2c/mnt/nhunterjr--Code--python--rattlesnake-vibration-controller'
sys.path.insert(0, ROOT)
from control_laws import control_laws as cl

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
st=np.real(np.trace(spec,axis1=1,axis2=2)); m=st>0
NC=6

# A STRUCTURAL FRF change -- per-DRIVE-column gains, so it is NOT a per-line
# scalar the laws are already invariant to.
col=np.array([1.0,0.6,1.5,0.8,1.25,0.9])[:H.shape[2]]
H2=H*col[None,None,:]

def lvl(out,Hu):
    p=Hu@out@Hu.conj().transpose(0,2,1)
    return 10*np.log10(np.real(np.trace(p,axis1=1,axis2=2))[m].sum()/st[m].sum())

def run(law,cap,Hs,H_true=None):
    H_true = H if H_true is None else H_true
    out=None; resp=None; hist=[]
    for Hc in Hs:
        out=law(spec,w,a,Hc,nr,nf,sr,rr,coh,100,100,cap,resp,out)
        resp=H_true@out@H_true.conj().transpose(0,2,1)
        hist.append(lvl(out,H_true))
    return np.array(hist),out

CAP='1e-3,0.95,-9.0'
print('--- (a) constant FRF: new law vs refresh disabled ---')
real_refresh = cl._refresh_drive_shape
for law,name in ((cl.match_trace_pseudoinverse,'match_trace_pseudoinverse'),):
    h_new,o_new = run(law,CAP,[H]*NC)
    cl._refresh_drive_shape = lambda *a, **k: None        # the pre-change law
    h_old,o_old = run(law,CAP,[H]*NC)
    cl._refresh_drive_shape = real_refresh
    worst = np.max(np.abs(h_new-h_old))
    rel = np.linalg.norm(o_new-o_old)/np.linalg.norm(o_old)
    print(f'{name:28s} level path identical to {worst:.2e} dB, drive to {rel:.2e} relative')
    print(f'{"":28s} levels {np.array2string(h_new,precision=4,floatmode="fixed")}')

print()
print('--- (b) structurally CHANGED FRF from cycle 4 (plant itself unchanged) ---')
for law,name in ((cl.match_trace_pseudoinverse,'match_trace_pseudoinverse'),):
    _,o_upd = run(law,CAP,[H,H,H,H2,H2,H2])
    _,o_ref = run(law,CAP,[H]*NC)
    rel = np.linalg.norm(o_upd-o_ref)/np.linalg.norm(o_ref)
    print(f'{name:28s} drive differs by {rel:.3e} relative  '
          f'({"REACHES the law" if rel>1e-9 else "NO EFFECT -- still ignored"})')

print()
print('--- (c) running ceiling, law TOLD the plant is 12 dB stronger than it is ---')
H_bad=H*10**(12/20)
for law,name in ((cl.pseudoinverse_control,'pseudoinverse_control'),
                 (cl.buzz_control,'buzz_control')):
    for cap,label in (('1e-3,0.95,-9.0,1e6','ceiling off'),
                      ('1e-3,0.95,-9.0,3.0','ceiling +3 ')):
        h,_=run(law,cap,[H,H_bad,H_bad,H_bad])
        if label.startswith('ceiling +'):
            v='OK  ' if h[1:].max()<=3.0+1e-6 else 'FAIL'
        else:
            v='    '
        print(f'{name:22s} {label}  levels {np.array2string(h,precision=2,floatmode="fixed")}  {v}')

print()
print('--- (c2) measurement-anchored: FRF says the plant is 12 dB WEAKER ---')
H_weak=H*10**(-12/20)
for law,name in ((cl.pseudoinverse_control,'pseudoinverse_control'),
                 (cl.buzz_control,'buzz_control')):
    h,_=run(law,'1e-3,0.95,-9.0,3.0',[H,H_weak,H_weak,H_weak])
    print(f'{name:22s} ceiling +3   TRUE levels {np.array2string(h,precision=2,floatmode="fixed")}'
          f'   {"OK" if h[1:].max()<=3.0+1e-6 else "FAIL -- ceiling blind in this direction"}')
