#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_h1_snapshot.py

Copies the current examples/sixdrive12resp/results/cva_captures/
latest_h1_sysid_capture.npz to a condition-labeled filename, for building
up a set of independent per-condition H1 FRF snapshots (see
analyze_h1_snapshot.py's docstring for why this needs to be a fresh
restart + settle + immediate-copy per condition, not just repeated reads
of the same live/continuously-updating file).

Checks the capture's capture_wall_time against current time and warns
(does not block) if the file looks stale -- a stale snapshot most likely
means the environment wasn't actually running/fitting when this was
called (e.g. control stopped, or spectral processing not producing new
H1 fits for some other reason), so the "independent measurement" you
think you're capturing might actually just be leftover from a previous
run.

Usage:
    python capture_h1_snapshot.py OUTPUT_LABEL.npz [--max-age-sec 30]

    (run from examples/sixdrive12resp/code/, or pass --src to point at a
    non-default source path)
"""
import argparse
import os
import shutil
import time

import numpy as np

DEFAULT_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
    'cva_captures', 'latest_h1_sysid_capture.npz'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', help='destination .npz path/filename for this condition\'s labeled snapshot')
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--max-age-sec', type=float, default=30.0,
                     help='warn if capture_wall_time is older than this many seconds (default 30)')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"ERROR: source file not found: {args.src}\n"
                          f"(has H1_CAPTURE_FRF ever fired? check spectral_processing.py's flag is True "
                          f"and that an H1/H2/H3/HV fit has actually completed)")

    d = np.load(args.src, allow_pickle=True)
    wall_time = float(d['capture_wall_time']) if 'capture_wall_time' in d else None
    age = (time.time() - wall_time) if wall_time is not None else None
    if age is None:
        print("WARNING: capture has no capture_wall_time field -- can't check freshness.")
    elif age > args.max_age_sec:
        print(f"WARNING: capture is {age:.1f}s old (> {args.max_age_sec}s threshold) -- "
              f"this may be a stale leftover from a previous run, not a fresh fit from the "
              f"condition you just settled at. Copying anyway, but double-check before trusting it.")
    else:
        print(f"Capture is {age:.1f}s old -- looks fresh.")

    shutil.copy2(args.src, args.output)
    print(f"Copied {args.src} -> {args.output}")


if __name__ == '__main__':
    main()
