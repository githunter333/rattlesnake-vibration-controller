#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_h1_snapshot.py

Companion to the H1_CAPTURE_FRF hook in spectral_processing.py. That hook
overwrites examples/sixdrive12resp/results/cva_captures/latest_h1_sysid_capture.npz
on every H1/H2/H3/HV FRF computation -- during ACTIVE CONTROL, that fires
continuously and the FRF it captures is match_trace_pseudoinverse's own
exponentially-averaged (coefficient 0.10) running estimate, carrying
whatever convergence history that control run has accumulated. That makes
cross-run comparisons unreliable (verified 2026-09-02: the same "mode 26"
peak looked essentially flat across five different capped/uncapped/level
combinations pulled from *live control* frf_data, when drive voltage
differed by 10x between them -- the averaging state, not the plant, was
dominating what got read).

The fix is procedural, not code: because latest_h1_sysid_capture.npz is a
*singleton* that gets overwritten on every fit, getting an independent read
per condition means, for each condition you want to compare:
  1. Fully STOP and RESTART the environment (a real restart resets
     spectral_processing's averaging buffers -- see set_parameters()'s
     reshape_arrays logic -- unlike ADJUST_TEST_LEVEL, which does not).
  2. Let it settle (with exponential_averaging_coefficient=0.10, budget
     ~30-50 frames before treating the average as representative).
  3. IMMEDIATELY copy latest_h1_sysid_capture.npz to a condition-specific
     filename (the accompanying capture_h1_snapshot.py does this) before
     the next control cycle's fit overwrites it.

This script then reads one or more of those labeled snapshots and reports
per-mode peak frequency the same way as the earlier ad-hoc FRF-shift
checks, but from a genuinely independent measurement per condition instead
of one continuously-evolving live-control estimate.

Usage:
    python analyze_h1_snapshot.py snap1.npz [snap2.npz ...] \
        --baseline ../../results/sdynpy_frame6x12_system_nonlinear_allmodes.npz \
        --modes 1 2 3 4 7 9 21 24 26 31 \
        --label snap1.npz=uncapped_-12dB [...]

--modes defaults to the same watch list used throughout this investigation
(7 is the built-in no-nonlinearity negative control; the others span low/
mid/high calibrated-shift modes).
"""

import argparse
import numpy as np


def load_snapshot(path):
    d = np.load(path, allow_pickle=True)
    freqs = d['frequencies']
    frf = d['frf']  # (n_lines, n_response, n_reference) typically
    estimator = str(d['estimator']) if 'estimator' in d else '?'
    frames = d['frames'] if 'frames' in d else None
    sample_rate = float(d['sample_rate']) if 'sample_rate' in d else None
    wall_time = float(d['capture_wall_time']) if 'capture_wall_time' in d else None
    return freqs, frf, estimator, frames, sample_rate, wall_time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--baseline', required=True,
                     help='path to sdynpy_frame6x12_system_nonlinear_allmodes.npz (for baseline mode freqs + has_nl flags)')
    ap.add_argument('--modes', nargs='+', type=int, default=[1, 2, 3, 4, 7, 9, 21, 24, 26, 31])
    ap.add_argument('--label', action='append', default=[], metavar='FILE=NAME')
    ap.add_argument('--window-lo', type=float, default=40.0, help='Hz below baseline to search (default 40, since all calibrated targets are softening)')
    ap.add_argument('--window-hi', type=float, default=10.0, help='Hz above baseline to search (default 10, small margin in case of noise)')
    args = ap.parse_args()

    labels = {}
    for item in args.label:
        f, name = item.split('=', 1)
        labels[f] = name

    bd = np.load(args.baseline, allow_pickle=True)
    base_freqs = bd['nl_target_freqs_hz']
    k3s = bd['nl_k3s']; c2s = bd['nl_c2s']
    has_nl = ~((k3s == 0.0) & (c2s == 0.0))

    snaps = []
    for f in args.files:
        label = labels.get(f, f.rsplit('/', 1)[-1])
        freqs, frf, estimator, frames, sr, wt = load_snapshot(f)
        mag = np.sqrt(np.sum(np.abs(frf) ** 2, axis=tuple(range(1, frf.ndim))))
        snaps.append((label, freqs, mag, estimator, frames, sr, wt))
        print(f"{label:>20}: estimator={estimator} frames={frames} sample_rate={sr} capture_wall_time={wt}")
    print()

    hdr = f"{'mode':>4} {'base_Hz':>9} {'has_nl':>7} " + " ".join(f"{s[0]:>16}" for s in snaps)
    print(hdr)
    print('-' * len(hdr))
    for m in args.modes:
        f0 = base_freqs[m - 1]
        row = []
        for label, freqs, mag, *_ in snaps:
            win = (freqs >= f0 - args.window_lo) & (freqs <= f0 + args.window_hi)
            if not np.any(win):
                row.append(float('nan'))
                continue
            fw = freqs[win]; mw = mag[win]
            row.append(fw[np.argmax(mw)])
        print(f"{m:>4} {f0:>9.2f} {str(bool(has_nl[m-1])):>7} " + " ".join(f"{v:>16.2f}" for v in row))


if __name__ == '__main__':
    main()
