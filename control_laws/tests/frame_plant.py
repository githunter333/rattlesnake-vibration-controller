"""
Synthetic test plants built from the 18-node frame model the 6x12 SDynPy
runs use (examples/sixdrive12resp, sdynpy_frame6x12_system_nonlinear.npz,
linear part; copied to tests/data/frame18_model.npz).

THE MODEL
---------
A 2D pin-jointed lattice, 3 rows x 6 nodes, x and y at every node (36 DOFs):
    bottom row  nodes  1-6     (where the shakers are)
    middle row  nodes  7-12
    top row     nodes 13-18
Free-free: 3 rigid-body modes, then 33 elastic modes from 120 to 900 Hz,
2-6 % damping.  The FRF is accelerance, -w^2 (K - w^2 M + j w C)^-1, which
reproduces examples/sixdrive12resp/results/analysis/frf_frame6x12_H.npz
exactly.

LAYOUTS
-------
'6x8'    the 6x12 rig case: drives 1-6 x, controls 7-14 x.  RANK 5, not 6,
         at every line: node 6's x DOF is joined only to node 5's x DOF, so a
         force at 6x looks, from anywhere but nodes 5-6, like a frequency-
         dependent force at 5x.  Any layout driving both 5x and 6x without a
         control at either loses one rank.

'10x14'  (2026-10-31) the harder case: 10 drives, 14 controls, FULL RANK 10
         at every line.
         drives    bottom row x and y at nodes 1, 2, 3, 6, and y at 4, 5
                   -- the 10 of the 12 bottom-row DOFs that maximize the
                   median log volume of the singular values seen from the
                   upper rows (exhaustive over the 66 choices; leaving out
                   4x and 5x avoids the 5x/6x pair above)
         controls  middle row x and y at nodes 7-12, plus 13x and 14x --
                   the rig case's 8 x controls plus the 6 middle-row y DOFs
                   chosen greedily on the same measure
         sigma_min/sigma_max over 100-1000 Hz: median 2.5e-2, 10th
         percentile 2.4e-3, minimum 6.1e-4.  14 controls on 10 drives, so
         part of any 14-channel specification is unreachable.

'10x14_rankdef'  the 6 rig drives (1-6 x) plus 4 bottom-row y drives (1, 2, 3,
         6), same 14 controls: RANK 9 of 10, the 5x/6x pair kept on purpose
         for testing how a law handles a structurally redundant drive.

Usage:
    from frame_plant import frame_plant
    p = frame_plant('10x14')        # same keys as check_control_laws_on_dev's
                                    # synthetic_plant, plus f, df, drives, controls
    python control_laws/tests/frame_plant.py      # prints a summary of each layout
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'data', 'frame18_model.npz')

LAYOUTS = {
    '6x8': dict(
        drives=[(n, 1) for n in range(1, 7)],
        controls=[(n, 1) for n in range(7, 15)]),
    '10x14': dict(
        drives=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 2), (5, 2), (6, 1), (6, 2)],
        controls=[(n, d) for n in range(7, 13) for d in (1, 2)] + [(13, 1), (14, 1)]),
    '10x14_rankdef': dict(
        drives=[(n, 1) for n in range(1, 7)] + [(1, 2), (2, 2), (3, 2), (6, 2)],
        controls=[(n, d) for n in range(7, 13) for d in (1, 2)] + [(13, 1), (14, 1)]),
}


def load_model(path=MODEL):
    d = np.load(path)
    dofs = [(int(c['node']), int(c['direction'])) for c in d['coordinate']]
    return d['mass'], d['stiffness'], d['damping'], dofs


def accelerance(f, rows, cols, path=MODEL):
    """Accelerance FRF (n_freq, len(rows), len(cols)); zero at f = 0."""
    M, K, C, dofs = load_model(path)
    ix = {x: i for i, x in enumerate(dofs)}
    r = [ix[x] for x in rows]
    c = [ix[x] for x in cols]
    H = np.zeros((len(f), len(r), len(c)), complex)
    for k, fk in enumerate(f):
        w = 2*np.pi*fk
        if w == 0:
            continue
        # solve for the drive columns only: (K - w^2 M + j w C) X = I[:, c]
        rhs = np.zeros((len(dofs), len(c)))
        rhs[c, np.arange(len(c))] = 1.0
        X = np.linalg.solve(K - w**2*M + 1j*w*C, rhs)
        H[k] = -w**2*X[r]
    return H


def frame_plant(layout='10x14', df=5.0, fmax=1280.0, band=(100.0, 1000.0), level=1e-3,
                noise_db=-60.0, path=MODEL):
    """Plant dict for the offline law checks.

    Spec: flat `level` on every control autospectrum over `band`, no cross
    terms -- the same specification as flat_spec_frame6x12.mat.  The noise
    CPSDs sit `noise_db` below the spec; the system-ID drive is white,
    uncorrelated, and scaled so the mean in-band response matches the spec.
    """
    lay = LAYOUTS[layout]
    drives, controls = lay['drives'], lay['controls']
    f = np.arange(int(round(fmax/df)) + 1)*df
    H = accelerance(f, controls, drives, path)
    n_freq, n_resp, n_drive = H.shape
    inb = (f >= band[0]) & (f <= band[1])
    spec = np.zeros((n_freq, n_resp, n_resp), complex)
    spec[inb] = level*np.eye(n_resp)
    noise = level*10**(noise_db/10)
    gxx_noise = np.tile(noise*np.eye(n_resp), (n_freq, 1, 1)).astype(complex)
    gff_noise = np.tile(noise*np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    resp_unit = np.mean(np.real(np.einsum('fmi,fmi->fm', H[inb], H[inb].conj())))
    sysid_level = level/max(resp_unit, 1e-300)
    gff_sysid = np.tile(sysid_level*np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    gxx_sysid = H @ gff_sysid @ H.conj().transpose(0, 2, 1) + gxx_noise
    warn = np.full((2, n_freq, n_resp), np.nan)
    abort = np.full((2, n_freq, n_resp), np.nan)
    coh = np.ones((n_freq, n_resp))
    return dict(spec=spec, warn=warn, abort=abort, H=H, gxx_noise=gxx_noise,
                gff_noise=gff_noise, gxx_sysid=gxx_sysid, gff_sysid=gff_sysid,
                coh=coh, band=inb, f=f, df=df, drives=drives, controls=controls,
                layout=layout)


def summary(p):
    s = np.linalg.svd(p['H'][p['band']], compute_uv=False)
    r = s[:, -1]/s[:, 0]
    rank = np.sum(s/s[:, :1] > 1e-8, axis=1)
    return (f"{p['layout']:14s} {len(p['controls'])} controls x {len(p['drives'])} drives, "
            f"{p['f'].size} lines at {p['df']:g} Hz; in band: rank {np.median(rank):.0f} "
            f"(min {rank.min()}), sigma_min/sigma_max median {np.median(r):.1e}, "
            f"10th pct {np.percentile(r, 10):.1e}, min {r.min():.1e}")


if __name__ == '__main__':
    for name in LAYOUTS:
        print(summary(frame_plant(name)))
