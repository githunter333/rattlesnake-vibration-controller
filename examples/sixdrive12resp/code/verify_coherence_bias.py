"""
verify_coherence_bias.py

Where the multiple-coherence bias correction comes from, and what the correct
factor is for THIS rig's processing chain.

THE MECHANISM.  At one frequency line the multiple coherence is

    gamma^2 = Syx Sxx^-1 Sxy / Syy = || P_X y ||^2 / || y ||^2

where P_X projects onto the p-dimensional subspace spanned by the p drive
channels, and y is the stack of n_d frame values of the response.  If y is
completely unrelated to the drives it is an isotropic random vector in C^n_d,
and the expected fraction of its energy lying in ANY fixed p-dimensional
subspace is p/n_d.  Exactly: the ratio is Beta(p, n_d - p), with mean p/n_d.

So a response with NO linear relationship to the drives does not measure zero.
It measures p/n_d.  With p = 6 drives and n_d = 20 frames per CPSD that is 0.30.

For a true coherence gamma^2 the signal part projects entirely into the
subspace and the remainder contributes p/n_d of what is left:

    E[gamma_hat^2] ~= gamma^2 + (p/n_d)(1 - gamma^2)

which inverts to the correction used on the figures:

    gamma_adj^2 = (gamma_hat^2 - p/n_d) / (1 - p/n_d)
                = 1 - (1 - gamma_hat^2) * n_d/(n_d - p)

Nothing here is fitted.  The correction is a function of gamma_hat^2 and two
counts; it uses no other information from the measurement.

WHAT n_d ACTUALLY IS.  frames_in_cpsd = 20, but the frames overlap 50% with a
Hann window, so they are not 20 independent observations.  Rather than argue
the effective count from window theory, part B measures it: push white noise
that is uncorrelated with the drives through the real Welch chain and read off
the bias directly.  Result, 40 trials x 1450 lines:

    no overlap        bias 0.3008  ->  n_eff 19.94,  factor 1.4303
    50% overlap       bias 0.3107  ->  n_eff 19.31,  factor 1.4507

The no-overlap case recovers the theory to 0.3%, which validates the chain.
The rig's 50% overlap costs about 0.7 of an average, so the correct factor is
1.4507 rather than the 1.4286 that n_d = 20 would give.  The difference is
worth about 0.007 dB on the in-band rms -- negligible there, but it moves the
floor below which the correction clips from 0.300 to 0.311.

A CONJUGATION TRAP, since it cost a wrong answer once.  Sxx must be X^H X --
conjugate the FIRST factor.  Conjugating the second gives conj(X^H X), the
quadratic form then uses (Sxx^-1)^T instead of Sxx^-1, and the null case reads
0.41 instead of 0.30.  Real-valued data hides the error completely.
"""
import numpy as np

P, N_FRAMES, NPERSEG = 6, 20, 4096
MEASURED_NEFF_50PCT_OVERLAP = 19.31      # part B, 40 trials


def multiple_coherence(X, Y, n):
    """X (frames, ..., p), Y (frames, ...) -> gamma^2 over the leading axes."""
    Sxx = np.einsum('f...i,f...j->...ij', X.conj(), X) / n
    Syx = np.einsum('f...,f...i->...i', Y, X.conj()) / n
    Syy = np.einsum('f...,f...->...', Y, Y.conj()).real / n
    num = np.real(np.einsum('...i,...ij,...j->...', Syx.conj(),
                            np.linalg.inv(Sxx), Syx))
    return num / Syy


def part_a(rng, trials=20000):
    """The formula itself, on ideal circular-complex frames."""
    print('A. E[gamma_hat^2] against the prediction gamma^2 + (p/n)(1-gamma^2)')
    print('     true      measured    predicted    corrected back')
    for g2 in (0.0, 0.40, 0.70, 0.90, 0.99):
        X = (rng.standard_normal((N_FRAMES, trials, P))
             + 1j * rng.standard_normal((N_FRAMES, trials, P))) / np.sqrt(2)
        w = (rng.standard_normal((trials, P))
             + 1j * rng.standard_normal((trials, P))) / np.sqrt(2)
        sig = np.einsum('f...i,...i->f...', X, w)
        sig /= np.sqrt(np.mean(np.abs(sig) ** 2, axis=0, keepdims=True))
        nz = (rng.standard_normal((N_FRAMES, trials))
              + 1j * rng.standard_normal((N_FRAMES, trials))) / np.sqrt(2)
        Y = np.sqrt(g2) * sig + np.sqrt(1 - g2) * nz
        est = multiple_coherence(X, Y, N_FRAMES).mean()
        pred = g2 + (P / N_FRAMES) * (1 - g2)
        corr = 1 - (1 - est) * N_FRAMES / (N_FRAMES - P)
        print(f'    {g2:5.2f}      {est:7.4f}     {pred:7.4f}       {corr:7.4f}')


def part_b(rng, trials=12):
    """The effective number of averages through the rig's own Welch chain."""
    win = np.hanning(NPERSEG)
    sl = slice(50, 1500)
    print()
    print('B. bias measured on a response uncorrelated with the drives')
    print('                          bias      n_eff   correction factor')
    for overlap, name in ((0.0, 'no overlap'), (0.5, '50% overlap (the rig)')):
        step = int(NPERSEG * (1 - overlap))
        ntot = NPERSEG + step * (N_FRAMES - 1)
        out = []
        for _ in range(trials):
            x = rng.standard_normal((ntot, P))
            y = rng.standard_normal(ntot)
            Xf = np.stack([np.fft.rfft(x[k*step:k*step+NPERSEG] * win[:, None],
                                       axis=0)[sl] for k in range(N_FRAMES)])
            Yf = np.stack([np.fft.rfft(y[k*step:k*step+NPERSEG] * win)[sl]
                           for k in range(N_FRAMES)])
            out.append(multiple_coherence(Xf, Yf, N_FRAMES).mean())
        m = float(np.mean(out))
        se = float(np.std(out) / np.sqrt(trials))
        neff = P / m
        print(f'  {name:24s} {m:.4f}    {neff:5.2f}      {neff/(neff-P):.4f}'
              f'   (+/- {se:.4f})')
    print()
    print(f'  n_d = 20 would give factor {N_FRAMES/(N_FRAMES-P):.4f}')


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    part_a(rng)
    part_b(rng)
