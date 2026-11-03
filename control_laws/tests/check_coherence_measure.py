"""
'Drive coherence' is one measure across the laws, the shared cap helper and the comparison
scripts: the magnitude |X_ij| / sqrt(X_ii X_jj) of a pair of drives.  The squared measure is
always called coherence squared (gamma^2).  Until 2026-11-02 the laws capped the magnitude
while the comparison scripts reported the square under the plain name, so a law sitting
exactly on a 0.95 cap printed 0.90.  This pins the one meaning down.

    python control_laws/tests/check_coherence_measure.py [--law-dir DIR]

Run in the 'rattlesnake' conda env.  Takes a few seconds (one small SDP run).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_optdiag as gopt                       # noqa: E402
import compare_optdiag as cmp                       # noqa: E402

LAW_DIR = sys.argv[sys.argv.index('--law-dir') + 1] if '--law-dir' in sys.argv else gopt.LAW_DIR
LAWS = gopt.load_laws(LAW_DIR)
RESULTS = []


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


def random_psd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    return A @ A.conj().T


@check('the fast law, the comparison script and the definition agree on the magnitude')
def _():
    fast = LAWS['fast']._max_drive_coherence_of
    for seed in range(5):
        X = random_psd(5, seed)
        d = np.real(np.diag(X))
        by_hand = max(abs(X[i, j])/np.sqrt(d[i]*d[j]) for i in range(5) for j in range(i + 1, 5))
        assert abs(fast(X) - by_hand) < 1e-12, (fast(X), by_hand)
        _, script_max = cmp.drive_coherence(X[None], np.array([True]))
        assert abs(script_max - by_hand) < 1e-12, (script_max, by_hand)


@check('a known pair: magnitude 0.8, gamma^2 0.64')
def _():
    X = np.array([[1.0, 0.8], [0.8, 1.0]], complex)
    assert abs(LAWS['fast']._max_drive_coherence_of(X) - 0.8) < 1e-12
    assert abs(LAWS['fast']._max_drive_coherence_of(X)**2 - 0.64) < 1e-12


@check('the shared cap helper caps the magnitude and keeps the diagonal')
def _():
    cap = LAWS['fast'].__init__.__globals__['_cap_drive_coherence']     # the helper the fast law loads
    X = np.stack([random_psd(4, s) for s in range(6)])
    out = cap(X, 0.5)
    for f in range(X.shape[0]):
        assert LAWS['fast']._max_drive_coherence_of(out[f]) <= 0.5 + 1e-9
        assert np.allclose(np.real(np.diag(out[f])), np.real(np.diag(X[f])))


@check('the SDP law holds a cap of 0.5 as a magnitude on a real plant (and the cap binds)')
def _():
    sc = gopt.scenario('c', '6x8', 'SDP', gopt.fields({2: 400, 4: 0.5}), n_cycles=2)
    last = gopt.run_scenario(sc, LAWS, {})['last'].astype(complex)      # in-band lines, all solved
    worst = max(LAWS['fast']._max_drive_coherence_of(last[f]) for f in range(last.shape[0]))
    assert worst <= 0.5 + 1e-3, f'magnitude {worst:.4f} exceeds the 0.5 cap'
    assert worst >= 0.45, f'magnitude {worst:.4f}: the cap does not bind, so this tests nothing'


@check('so a 0.95 cap is gamma^2 0.9025 or less')
def _():
    sc = gopt.scenario('c', '6x8', 'SDP', gopt.fields({2: 400}), n_cycles=2)
    last = gopt.run_scenario(sc, LAWS, {})['last'].astype(complex)
    worst = max(LAWS['fast']._max_drive_coherence_of(last[f]) for f in range(last.shape[0]))
    assert worst <= 0.95 + 1e-3 and worst**2 <= 0.9025 + 2e-3, (worst, worst**2)


if __name__ == '__main__':
    for label, err in RESULTS:
        print(f"{'PASS' if err is None else 'FAIL'}  {label}" + ('' if err is None else f'\n        {err}'))
    bad = sum(1 for _, e in RESULTS if e)
    print(f'\n{len(RESULTS) - bad} of {len(RESULTS)} passed')
    sys.exit(1 if bad else 0)
