"""
Offline check that the control laws in control_laws/ load and run under the
Rattlesnake 4.x (dev) random-vibration loader, without hardware or the GUI.

It reproduces, step for step, what dev does with a control script:
  1. load_control_module(path)                     (rattlesnake.utilities)
  2. the "Control Python Function" dropdown filter  (random_vibration_sys_id_ui.py)
  3. the function-type detection                    (random_vibration_sys_id_environment.py)
  4. construction / system_id_update / control calls, with the same argument
     order as random_vibration_sys_id_data_analysis.py, for a control
     prediction followed by several closed-loop control cycles
  5. set_test_level_db forwarding, for laws that define it

The plant is a synthetic 8-response x 6-drive modal FRF.  This checks that
each law runs and returns a sane drive CPSD; it says nothing about how well
any law controls.

Usage:  python control_laws/tests/check_control_laws_on_dev.py [law_name ...]
"""
import inspect
import os
import sys
import time
import traceback

import numpy as np

from rattlesnake.utilities import load_control_module

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
RV_SCRIPTS = [
    "control_laws.py",
    "optimal_diagonal_control.py",
    "optimal_diagonal_control_fast.py",
    "octave_band_switching_control.py",
    "diagonal_congruence_control.py",
    "broyden_congruence_control.py",
]
N_CYCLES = 4


def dropdown_candidates(module):
    """Same filter random_vibration_sys_id_ui.py uses for the dropdown."""
    out = []
    for name, obj in inspect.getmembers(module):
        ok = (
            (inspect.isfunction(obj) and len(inspect.signature(obj).parameters) >= 12)
            or inspect.isgeneratorfunction(obj)
            or (
                inspect.isclass(obj)
                and all(
                    m in obj.__dict__
                    and not getattr(obj.__dict__[m], "__isabstractmethod__", False)
                    for m in ["system_id_update", "control"]
                )
            )
        )
        if ok:
            out.append((name, obj))
    return out


def function_type(obj):
    """Same detection random_vibration_sys_id_environment.py uses (0 fn, 1 gen, 2 class)."""
    if inspect.isgeneratorfunction(obj):
        return 1
    if inspect.isclass(obj):
        return 2
    return 0


def synthetic_plant(n_freq=401, n_resp=8, n_drive=6, df=1.0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.arange(n_freq) * df
    w = 2 * np.pi * f
    n_modes = 10
    wn = 2 * np.pi * np.linspace(40, 340, n_modes)
    zeta = 0.02
    phi_r = rng.standard_normal((n_resp, n_modes))
    phi_d = rng.standard_normal((n_drive, n_modes))
    H = np.zeros((n_freq, n_resp, n_drive), complex)
    for k in range(n_modes):
        den = wn[k] ** 2 - w ** 2 + 2j * zeta * wn[k] * w
        H += (np.outer(phi_r[:, k], phi_d[:, k])[None] * (wn[k] ** 2 / den)[:, None, None])
    H += 0.01 * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    # in-band 20-380 Hz, flat-ish diagonal spec with some correlation
    spec = np.zeros((n_freq, n_resp, n_resp), complex)
    band = (f >= 20) & (f <= 380)
    base = 0.5 * np.eye(n_resp) + 0.5 / n_resp
    spec[band] = base * 1e-2
    coh = np.ones((n_freq, n_resp))
    gxx_noise = np.tile(1e-8 * np.eye(n_resp), (n_freq, 1, 1)).astype(complex)
    gff_noise = np.tile(1e-8 * np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    gff_sysid = np.tile(1e-2 * np.eye(n_drive), (n_freq, 1, 1)).astype(complex)
    gxx_sysid = H @ gff_sysid @ H.conj().transpose(0, 2, 1)
    warn = np.full((2, n_freq, n_resp), np.nan)
    abort = np.full((2, n_freq, n_resp), np.nan)
    return dict(spec=spec, warn=warn, abort=abort, H=H, gxx_noise=gxx_noise,
                gff_noise=gff_noise, gxx_sysid=gxx_sysid, gff_sysid=gff_sysid,
                coh=coh, band=band)


def check_output(name, out, p, stage):
    n_freq, _, n_drive = p["H"].shape
    out = np.asarray(out)
    if out.shape != (n_freq, n_drive, n_drive):
        raise ValueError(f"{stage}: output shape {out.shape}, expected {(n_freq, n_drive, n_drive)}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{stage}: output has NaN/Inf")
    herm = np.max(np.abs(out - out.conj().transpose(0, 2, 1))) / (np.max(np.abs(out)) + 1e-300)
    eig_min = np.min(np.linalg.eigvalsh((out + out.conj().transpose(0, 2, 1)) / 2)[p["band"]])
    eig_max = np.max(np.linalg.eigvalsh((out + out.conj().transpose(0, 2, 1)) / 2)[p["band"]])
    Y = p["H"] @ out @ p["H"].conj().transpose(0, 2, 1)
    ydiag = np.real(np.einsum("fii->fi", Y))[p["band"]]
    sdiag = np.real(np.einsum("fii->fi", p["spec"]))[p["band"]]
    err_db = 10 * np.log10(np.maximum(ydiag, 1e-300) / sdiag)
    return (f"herm {herm:.1e}, min eig/max {eig_min / (eig_max + 1e-300):+.1e}, "
            f"predicted diag error rms {np.sqrt(np.mean(err_db ** 2)):.2f} dB")


def run_law(script, name, obj, p, params=""):
    ftype = function_type(obj)
    spec, warn, abort, H = p["spec"], p["warn"], p["abort"], p["H"]
    frames = total = 20
    notes = []
    t0 = time.perf_counter()
    if ftype == 2:  # class: same positional order as data analysis
        law = obj(spec, warn, abort, params, None, p["gxx_noise"], p["gff_noise"],
                  p["gxx_sysid"], p["gff_sysid"], None, 0, total, None, None)
        if hasattr(law, "set_test_level_db"):
            law.set_test_level_db(-6.0)
            notes.append("set_test_level_db ok")
        law.system_id_update(H, p["gxx_noise"], p["gff_noise"], p["gxx_sysid"],
                             p["gff_sysid"], p["coh"], frames, total)
        out = law.control(H, p["coh"], frames, total, None, None)
        notes.append("prediction: " + check_output(name, out, p, "prediction"))
        last_out = out
        for k in range(N_CYCLES):
            if k == 2 and hasattr(law, "set_test_level_db"):
                law.set_test_level_db(0.0)
            last_resp = H @ last_out @ H.conj().transpose(0, 2, 1)
            out = law.control(H, p["coh"], frames, total, last_resp, last_out)
            last_out = out
    elif ftype == 1:  # generator
        gen = obj()
        next(gen)
        out = gen.send((spec, warn, abort, H, p["gxx_noise"], p["gff_noise"], p["gxx_sysid"],
                        p["gff_sysid"], p["coh"], frames, total, params, None, None))
        notes.append("prediction: " + check_output(name, out, p, "prediction"))
        last_out = out
        for _ in range(N_CYCLES):
            last_resp = H @ last_out @ H.conj().transpose(0, 2, 1)
            out = gen.send((spec, warn, abort, H, p["gxx_noise"], p["gff_noise"], p["gxx_sysid"],
                            p["gff_sysid"], p["coh"], frames, total, params, last_resp, last_out))
            last_out = out
    else:  # plain function
        out = obj(spec, warn, abort, H, p["gxx_noise"], p["gff_noise"], p["gxx_sysid"],
                  p["gff_sysid"], p["coh"], frames, total, params, None, None)
        notes.append("prediction: " + check_output(name, out, p, "prediction"))
        last_out = out
        for _ in range(N_CYCLES):
            last_resp = H @ last_out @ H.conj().transpose(0, 2, 1)
            out = obj(spec, warn, abort, H, p["gxx_noise"], p["gff_noise"], p["gxx_sysid"],
                      p["gff_sysid"], p["coh"], frames, total, params, last_resp, last_out)
            last_out = out
    notes.append(f"after {N_CYCLES} cycles: " + check_output(name, out, p, "control"))
    return ["function", "generator", "class"][ftype], time.perf_counter() - t0, notes


def main(only=None):
    p = synthetic_plant()
    results = []
    for script in RV_SCRIPTS:
        path = os.path.join(LAW_DIR, script)
        print(f"\n=== {script}")
        try:
            module = load_control_module(path)
        except Exception:
            print("  LOAD FAILED\n" + traceback.format_exc())
            results.append((script, "<load>", "FAIL"))
            continue
        cands = dropdown_candidates(module)
        print(f"  dropdown shows: {[c[0] for c in cands]}")
        for name, obj in cands:
            if only and name not in only:
                continue
            # Only test laws defined in this file, not ones it imported.
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            try:
                kind, dt, notes = run_law(script, name, obj, p)
                print(f"  PASS {name} [{kind}] {dt:.1f}s")
                for n in notes:
                    print(f"       {n}")
                results.append((script, name, "PASS"))
            except Exception:
                print(f"  FAIL {name}\n" + "".join("       " + l for l in traceback.format_exc().splitlines(True)))
                results.append((script, name, "FAIL"))
    print("\n=== summary")
    for r in results:
        print(f"  {r[2]:4s}  {r[0]}: {r[1]}")
    return 0 if all(r[2] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
