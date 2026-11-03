"""
Writes control_laws/optimal_diagonal_parameters.md from the parameter tables in
the three laws, so the reference cannot drift from the code.

    python control_laws/tests/make_parameter_reference.py            write it
    python control_laws/tests/make_parameter_reference.py --check    exit 1 if it is out of date

check_named_params.py runs the check too.  Run in the 'rattlesnake' conda env.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_optdiag as gopt                       # noqa: E402

OUT = os.path.join(gopt.LAW_DIR, 'optimal_diagonal_parameters.md')

HEADER = """\
# Parameters of the optimal diagonal laws

*Generated from the laws' parameter tables by `control_laws/tests/make_parameter_reference.py`;
do not edit by hand.*

Three laws read these: `optimal_diagonal_control` (the SDP law),
`optimal_diagonal_control_fast`, and `octave_band_switching_control`.  Each
subclass accepts everything its parent does, plus the parameters listed under it.
Put them in the environment's **Control Parameters** box (or cell 14 of the
environment sheet of a profile).

## Two forms

**Named (use this).**  One `name: value` per line, or separated by `;`.  Names are
case-insensitive, `#` starts a comment, an empty value means "the default".

```
max_bins_per_update: 10
max_drive_coherence: 0.8
response_trim_gain: 1.0      # trim on
```

A name the law does not have, or a value it cannot read, stops the test from
starting with a message naming the line and the nearest valid name.

**Comma-separated (the original form).**  Positional, by the *field* number in the
tables below; an empty field means "the default"; a field that cannot be read
keeps its default with a warning and the later fields are still read.  Every
archived profile uses this form, and it keeps working.

```
1e-6,0.5,20,1.0,0.95,6,-9.0,1,2,1e-2
```

Field numbers are 1-based.  The two forms must not be mixed in one string.

## Choosing the time budget

`max_solve_ms_per_update` defaults to 0 (off): only `max_bins_per_update` limits the work
per call.  Every behaviour change in these laws ships off by default, so archived profiles
reproduce exactly.  But the line count alone does not bound a call.  At 10 drives x 14
controls the SDP law's default 20 lines per call took 1.1 to 1.5 s on its worst call, longer
than a 1 s control cycle.

Starting points, measured on the 10x14 frame plant (`tests/compare_optdiag_frame.py`, 15 cycles,
FRF held fixed, 20-frame CPSD estimates), on the Mac that runs the rig software, 2026-11-02:

| law | setting | final error | lines solved | ms per call, median / worst |
| --- | --- | --- | --- | --- |
| SDP | defaults (20 lines per call) | 0.12 dB | all 180, by cycle 7 | 16 / 1508 |
| SDP | `max_bins_per_update: 400`, `max_solve_ms_per_update: 500` | 0.63 dB | 109 after 15 cycles, still converging | 490 / 512 |
| fast | defaults (20 lines per call) | 0.23 dB | all 180, by cycle 7 | 20 / 105 |
| fast | `max_bins_per_update: 400`, `max_solve_ms_per_update: 500` | 0.23 dB | all 180, during system ID | 18 / 20 |

The errors and the fast law's line counts reproduce the 2026-10-30/31 measurements.  The times,
and the budgeted SDP row (0.34 dB and 148 lines then), depend on the machine, because a
time budget buys fewer lines on a slower one.

For the fast law at 10 drives, set `max_bins_per_update: 400` and `max_solve_ms_per_update: 500`.
For the SDP law, pick a budget that fits the control cycle (about 500 ms) and expect 15 to 20
cycles to converge; starting with the fast law and finishing with the SDP law is an option.
Fully converged, the SDP law is the more accurate (0.12 against 0.23 dB at the same drive
level).

## Drive coherence

In every one of these laws, and in the test and comparison scripts, "drive coherence" is the
magnitude |X_ij| / sqrt(X_ii X_jj) of a pair of drives: 0 for independent drives, 1 for fully
dependent ones.  That is what `max_drive_coherence` caps.  The squared measure is always
called coherence squared (gamma^2); a cap of 0.95 is gamma^2 = 0.9025.

## optimal_diagonal_control (and, through it, the other two)

{base}

## optimal_diagonal_control_fast adds

{fast}

## octave_band_switching_control adds

{octave}

`*` The field number counts only the **non-empty** fields of the comma-separated
string, and these positions are also read by `optimal_diagonal_control` as
`bm_rank`, `startup_test_level_cap_db`, `error_domain`, `n_irls_passes` and
`drive_rcond`, so a comma-separated string with six or more fields sets both.  It
is kept that way so archived strings give the results they always did, and the law
prints a warning.  **Use the named form for this law.**
"""


def generate():
    laws = gopt.load_laws()
    return HEADER.format(base=laws['SDP'].parameter_reference(own_only=True),
                         fast=laws['fast'].parameter_reference(own_only=True),
                         octave=laws['octave'].parameter_reference(own_only=True))


def main():
    text = generate()
    if '--check' in sys.argv:
        current = open(OUT).read() if os.path.exists(OUT) else None
        if current != text:
            print('optimal_diagonal_parameters.md is out of date; run make_parameter_reference.py')
            return 1
        print('optimal_diagonal_parameters.md is up to date')
        return 0
    with open(OUT, 'w') as fh:
        fh.write(text)
    print('wrote', os.path.relpath(OUT, os.path.dirname(gopt.LAW_DIR)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
