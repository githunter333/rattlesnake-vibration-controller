"""
Checks of the named parameter syntax of the optimal diagonal laws, and of the
table behind it: what is accepted, and what a mistake produces.

    python control_laws/tests/check_named_params.py [--law-dir DIR]

Run in the 'rattlesnake' conda env (cvxpy).  Constructs the laws with no FRF, so
nothing is solved and it takes a few seconds.  Exit code 1 if anything fails.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_optdiag as gopt                       # noqa: E402
from frame_plant import frame_plant                 # noqa: E402

LAW_DIR = sys.argv[sys.argv.index('--law-dir') + 1] if '--law-dir' in sys.argv else gopt.LAW_DIR
LAWS = gopt.load_laws(LAW_DIR)
PLANT = frame_plant('6x8')
RESULTS = []


def build(key, params):
    """(law, printed output); exceptions propagate."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        law = LAWS[key](PLANT['spec'], PLANT['warn'], PLANT['abort'], params)
    return law, out.getvalue()


def check(label):
    def deco(fn):
        try:
            fn()
            RESULTS.append((label, None))
        except AssertionError as e:
            RESULTS.append((label, f'assertion failed: {e}'))
        except Exception as e:                      # noqa: BLE001
            RESULTS.append((label, f'{type(e).__name__}: {e}'))
        return fn
    return deco


def raises(key, params, *fragments):
    try:
        build(key, params)
    except ValueError as e:
        for f in fragments:
            assert f in str(e), f'{f!r} not in the message: {e}'
        return str(e)
    raise AssertionError(f'{params!r} did not raise')


@check('named values are read, with the right types')
def _():
    law, _ = build('SDP', 'reg: 1e-5\nmax_bins_per_update: 10\nerror_domain: db')
    assert law.reg == 1e-5 and law.max_bins_per_update == 10 and isinstance(law.max_bins_per_update, int)
    assert law.error_domain == 'db' and law.frf_update_threshold == 0.5      # untouched = default


@check('; separates items like a newline does')
def _():
    a, _ = build('SDP', 'reg: 1e-5; max_bins_per_update: 10')
    b, _ = build('SDP', 'reg: 1e-5\nmax_bins_per_update: 10')
    assert (a.reg, a.max_bins_per_update) == (b.reg, b.max_bins_per_update) == (1e-5, 10)


@check('comments, blank lines, spaces and capitals are fine')
def _():
    law, _ = build('SDP', '# trim on\n\n  RESPONSE_Trim_Gain :  1.0   # fraction\n\nreg: 2e-6\n')
    assert law.response_trim_gain == 1.0 and law.reg == 2e-6


@check('an empty value keeps the default')
def _():
    law, _ = build('SDP', 'reg:\nmax_bins_per_update: 7')
    assert law.reg == 1e-6 and law.max_bins_per_update == 7


@check('error_domain: words and numbers')
def _():
    for text, want in [('db', 'db'), ('DB', 'db'), ('linear', 'linear'), ('1', 'db'), ('0', 'linear')]:
        law, _ = build('SDP', f'error_domain: {text}')
        assert law.error_domain == want, (text, law.error_domain)


@check('the transforms apply in the named form too')
def _():
    law, _ = build('fast', 'response_trim_gain: -2\nresponse_trim_limit_db: -1.5\nsvd_gap_min: 0.2\nn_irls_passes: -4')
    assert law.response_trim_gain == 0.0 and law.response_trim_limit_db == 1.5
    assert law.svd_gap_min == 1.0 and law.n_irls_passes == 0


@check('an unknown name raises, naming the line and the nearest valid name')
def _():
    msg = raises('SDP', 'max_bin_per_update: 5', "unknown parameter 'max_bin_per_update'",
                 "did you mean 'max_bins_per_update'")
    assert 'valid names' in msg


@check("another law's name is unknown here")
def _():
    raises('SDP', 'bm_rank: 3', "unknown parameter 'bm_rank'")
    raises('fast', 'frequency_spacing: 5', "unknown parameter 'frequency_spacing'")
    raises('octave', 'bm_rank: 3', "unknown parameter 'bm_rank'")
    build('fast', 'bm_rank: 3')
    build('octave', 'octave_fmin: 100')


@check('a value that cannot be read raises, naming the parameter')
def _():
    raises('SDP', 'reg: abc', 'parameter reg', "'abc'")
    raises('fast', 'bm_rank: three', 'parameter bm_rank')


@check('a repeated name raises')
def _():
    raises('SDP', 'reg: 1\nreg: 2', "'reg' is given twice")


@check('a line without a colon in a named string raises (no mixing the forms)')
def _():
    raises('SDP', 'reg: 1\n20', "not of the form 'name: value'")


@check('legacy: a value that cannot be read keeps its default, warns, and later fields are still read')
def _():
    law, out = build('SDP', '1e-6,abc,30')
    assert law.frf_update_threshold == 0.5 and law.max_bins_per_update == 30
    assert 'WARNING' in out and 'frf_update_threshold' in out, out


@check('legacy: empty fields keep their position and spaces are ignored')
def _():
    law, _ = build('SDP', ' ,, 200 ')
    assert law.max_bins_per_update == 200 and law.reg == 1e-6


@check('empty or missing strings give the defaults')
def _():
    for text in ('', None, '   '):
        law, _ = build('SDP', text)
        assert (law.reg, law.max_bins_per_update, law.error_domain) == (1e-6, 20, 'linear')


@check('octave law, named: its five parameters are read and the base class is not disturbed')
def _():
    law, out = build('octave', 'frequency_spacing: 5\nswitch_level_db: -3\noctave_fraction: 3\n'
                               'octave_fmin: 200\noctave_fmax: 800')
    assert (law.frequency_spacing, law.switch_level_db, law.octave_fraction) == (5.0, -3.0, 3)
    assert (law.octave_fmin, law.octave_fmax) == (200.0, 800.0)
    assert law.n_irls_passes == 2 and law.error_domain == 'linear' and law.drive_rcond == 2.0
    assert 'WARNING' not in out


@check('octave law, legacy with six or more fields: both readings still apply, and it warns')
def _():
    law, out = build('octave', '1e-6,0.5,20,1.0,0.95,5,0,6,100,1000')
    assert law.frequency_spacing == 5.0 and law.octave_fmin == 100.0            # its own reading
    assert law.error_domain == 'db' and law.n_irls_passes == 100                  # the collision
    assert 'WARNING' in out and 'named form' in out
    _, quiet = build('octave', '1e-6,0.5,20,1.0,0.95')
    assert 'WARNING' not in quiet


@check("octave law, legacy: its five positions count only the NON-EMPTY fields (as they always have)")
def _():
    law, _ = build('octave', '1e-6,,20,1.0,0.95,5')          # five non-empty fields: none is the sixth
    assert law.frequency_spacing is None
    law, _ = build('octave', '1e-6,0.5,20,1.0,0.95,,5')       # the blank sits at the sixth position
    assert law.frequency_spacing == 5.0


@check('the tables: unique names, unique legacy positions, every name is an attribute')
def _():
    for key, n in (('SDP', 15), ('fast', 25), ('octave', 20)):
        table = LAWS[key].parameter_table()
        names = [p.name for p in table]
        assert len(names) == len(set(names)) == n, (key, len(names))
        positions = [p.legacy for p in table if not p.dropblank]
        assert len(positions) == len(set(positions)), (key, 'duplicate legacy position')
        law, _ = build(key, '')
        assert all(hasattr(law, p.name) for p in table)


if __name__ == '__main__':
    width = max(len(l) for l, _ in RESULTS)
    for label, err in RESULTS:
        print(f"{'PASS' if err is None else 'FAIL'}  {label}" + ('' if err is None else f'\n        {err}'))
    bad = sum(1 for _, e in RESULTS if e)
    print(f'\n{len(RESULTS) - bad} of {len(RESULTS)} passed')
    sys.exit(1 if bad else 0)
