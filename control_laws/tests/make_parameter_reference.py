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
