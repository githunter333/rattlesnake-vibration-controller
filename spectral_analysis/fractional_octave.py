# -*- coding: utf-8 -*-
"""
fractional_octave.py

Standalone, Rattlesnake-independent utilities for estimating narrowband and
fractional-octave (e.g. 1/6 octave) auto spectral densities, frequency
response functions (FRFs), and coherence from time series data.

The fractional-octave estimates are NOT computed with a constant-Q filter
bank. Instead, a single narrowband Welch/CSD estimate is computed first
(the usual FFT-based, linear-frequency-spacing estimate), and the
fractional-octave bands are formed by energy-preserving averaging of that
narrowband estimate into log-spaced bands. This keeps the narrowband and
fractional-octave results directly comparable (same underlying data), and
avoids the added complexity of a decimated digital filter bank.

Three entry points map directly onto the use cases:
    time_series_to_octave_psd(x, fs, ...)          : one time series -> ASD
    time_series_to_octave_frf(x, y, fs, ...)        : input/response -> H1 FRF
    time_series_to_octave_coherence(x, y, fs, ...)  : input/response -> gamma^2

All three also return the narrowband estimate they were derived from, so
the two can be plotted/compared directly.
"""

import numpy as np
from scipy.signal import csd, get_window


# ---------------------------------------------------------------------
# Fractional-octave band definition
# ---------------------------------------------------------------------
def octave_band_frequencies(fmin=10.0, fmax=2000.0, fraction=6, base=2.0, ref=1000.0):
    """Fractional-octave center frequencies and band edges.

    Uses the standard exact-frequency definition (ANSI S1.11 / IEC 61260
    base-2 system): center frequencies are f_ref * base**(i/fraction) for
    integer i, and each band spans one bandwidth of base**(1/(2*fraction))
    above and below its center.

    Parameters
    ----------
    fmin, fmax : float
        Frequency range (Hz) to cover. Bands whose center falls in
        [fmin, fmax] are returned. Defaults 10 Hz and 2000 Hz.
    fraction : int
        Octave fraction, e.g. 6 for 1/6 octave, 3 for 1/3 octave, 1 for
        full octave. Default 6.
    base : float
        Octave base, 2.0 for the exact (base-2) system. Default 2.0.
    ref : float
        Reference center frequency (Hz) that band index 0 is anchored to.
        Default 1000.0 Hz (the standard reference).

    Returns
    -------
    centers, lower, upper : ndarray
        Center frequency, and lower/upper band edge, for each band, in
        ascending order.
    """
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Require 0 < fmin < fmax")
    i_lo = int(np.floor(fraction * np.log(fmin / ref) / np.log(base)))
    i_hi = int(np.ceil(fraction * np.log(fmax / ref) / np.log(base)))
    indices = np.arange(i_lo, i_hi + 1)
    centers = ref * base ** (indices / fraction)
    edge_factor = base ** (1.0 / (2 * fraction))
    lower = centers / edge_factor
    upper = centers * edge_factor
    keep = (centers >= fmin) & (centers <= fmax)
    return centers[keep], lower[keep], upper[keep]


# ---------------------------------------------------------------------
# Narrowband (Welch/CSD) estimate
# ---------------------------------------------------------------------
def narrowband_cross_spectra(x, y, fs, df=2.0, window='hann', overlap=0.5, detrend='constant'):
    """Narrowband auto/cross spectral density estimate via Welch's method.

    x and y are segmented identically so Sxx, Syy, and Sxy line up bin for
    bin (required for a valid H1 FRF estimate).

    Parameters
    ----------
    x, y : ndarray
        Input and response time series (same length, same sample rate).
        y may be None to skip the cross/response terms (Sxx only).
    fs : float
        Sample rate, Hz.
    df : float
        Target frequency resolution, Hz (sets nperseg = round(fs/df)).
        Default 2.0 Hz.
    window : str
        Window function name passed to scipy.signal.get_window.
    overlap : float
        Fractional segment overlap, 0-1. Default 0.5.
    detrend : str or False
        Detrend method passed to scipy.signal.csd. Default 'constant'.

    Returns
    -------
    f : ndarray
        Frequency vector, Hz (uniform spacing df).
    Sxx : ndarray
        Real auto spectral density of x.
    Syy : ndarray or None
        Real auto spectral density of y (None if y is None).
    Sxy : ndarray or None
        Complex cross spectral density x->y (None if y is None).
    """
    x = np.asarray(x)
    nperseg = int(round(fs / df))
    nperseg = min(nperseg, x.size)
    noverlap = int(round(nperseg * overlap))
    win = get_window(window, nperseg)

    f, Sxx = csd(x, x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    Sxx = np.real(Sxx)

    if y is None:
        return f, Sxx, None, None

    y = np.asarray(y)
    _, Syy = csd(y, y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    Syy = np.real(Syy)
    _, Sxy = csd(x, y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=detrend)

    return f, Sxx, Syy, Sxy


# ---------------------------------------------------------------------
# Energy-preserving band averaging of a narrowband estimate
# ---------------------------------------------------------------------
def _band_average(f, values, lower, upper):
    """Average `values` (a PSD, so units^2/Hz) into bands [lower, upper).

    Each narrowband line f_i is treated as representing the frequency
    slice [f_i - df/2, f_i + df/2]. The band level is the overlap-weighted
    mean of the narrowband values whose slices intersect the band:
        band_level = sum_i(value_i * overlap_i) / sum_i(overlap_i)
    where overlap_i is the length of the intersection of line i's slice
    with the band.

    This reduces to a plain energy-preserving average (sum(values)*df /
    bandwidth) when the band is much wider than df, since every fully-
    contained line then has overlap_i = df. But when a fractional-octave
    band is NARROWER than the narrowband resolution df -- which happens
    at the low-frequency end of a 1/6-octave analysis -- at most one line
    partially overlaps the band, and this correctly falls back to just
    that line's value instead of artificially inflating or deflating it
    by df/bandwidth (a real bug in an earlier version of this function:
    dividing a df-wide contribution by a narrower true bandwidth silently
    over-weighted the low bands whenever bandwidth < df).
    """
    df = f[1] - f[0]
    bin_lo = f - df / 2.0
    bin_hi = f + df / 2.0
    out = np.full(lower.shape, np.nan, dtype=values.dtype)
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        overlap = np.clip(np.minimum(bin_hi, hi) - np.maximum(bin_lo, lo), 0, None)
        total = overlap.sum()
        if total > 0:
            out[i] = np.sum(values * overlap) / total
    return out


def octave_band_psd(f, Sxx, lower, upper):
    """Band-average a narrowband PSD into fractional-octave bands.
    See octave_band_frequencies() for lower/upper. Returns one PSD level
    (same units^2/Hz as Sxx) per band; np.nan only for a band entirely
    outside the narrowband frequency range covered by f."""
    return _band_average(f, Sxx, lower, upper)


def octave_band_frf(f, Sxx, Sxy, lower, upper):
    """Band-average an H1 FRF into fractional-octave bands by averaging
    the numerator (Sxy) and denominator (Sxx) separately over each band
    and then dividing -- the correct way to aggregate an H1 estimate
    across multiple frequency lines (preserves phase and weights each
    line by its input energy), rather than averaging H1(f) directly."""
    num = _band_average(f, Sxy, lower, upper)
    den = _band_average(f, Sxx.astype(complex), lower, upper).real
    with np.errstate(invalid='ignore', divide='ignore'):
        H = num / den
    return H


def narrowband_coherence(Sxx, Syy, Sxy):
    """Ordinary (magnitude-squared) coherence gamma^2(f) = |Sxy|^2 / (Sxx*Syy)
    from narrowband auto/cross spectra. Values in [0, 1] (up to numerical
    noise); 1.0 means the response is a perfectly linear, noise-free
    function of the input at that frequency line."""
    with np.errstate(invalid='ignore', divide='ignore'):
        gamma2 = np.abs(Sxy) ** 2 / (Sxx * Syy)
    return gamma2


def octave_band_coherence(f, Sxx, Syy, Sxy, lower, upper):
    """Band-average coherence into fractional-octave bands.

    Averages Sxx, Syy, and Sxy separately over each band (same
    overlap-weighted averaging as octave_band_frf) and computes
    coherence from the band-averaged spectra, rather than averaging the
    narrowband coherence values directly. This matters because
    coherence is nonlinear in the spectra: naively averaging gamma^2(f)
    values from different lines double counts the "coherent" and
    "incoherent" contributions incorrectly, and can't reproduce the
    textbook property that averaging over more independent estimates at
    a genuinely coherent frequency drives gamma^2 toward 1, not away
    from it. Averaging the spectra first and computing coherence from
    the average is the standard, correct multi-line generalization of
    the multi-average coherence estimator.
    """
    Sxx_band = _band_average(f, Sxx.astype(complex), lower, upper).real
    Syy_band = _band_average(f, Syy.astype(complex), lower, upper).real
    Sxy_band = _band_average(f, Sxy, lower, upper)
    return narrowband_coherence(Sxx_band, Syy_band, Sxy_band)


# ---------------------------------------------------------------------
# Top-level convenience functions -- the two requested entry points
# ---------------------------------------------------------------------
def time_series_to_octave_psd(x, fs, fmin=10.0, fmax=2000.0, fraction=6,
                               df=2.0, window='hann', overlap=0.5, detrend='constant'):
    """Time history -> narrowband and 1/6-octave (by default) auto
    spectral density.

    Returns
    -------
    dict with keys:
        'f_narrow', 'psd_narrow'   : narrowband frequency vector and ASD
        'f_octave', 'psd_octave'   : octave-band center frequencies and ASD
        'band_edges'               : (lower, upper) octave band edges
    """
    f, Sxx, _, _ = narrowband_cross_spectra(x, None, fs, df=df, window=window,
                                             overlap=overlap, detrend=detrend)
    centers, lower, upper = octave_band_frequencies(fmin, fmax, fraction)
    psd_oct = octave_band_psd(f, Sxx, lower, upper)
    return {
        'f_narrow': f, 'psd_narrow': Sxx,
        'f_octave': centers, 'psd_octave': psd_oct,
        'band_edges': (lower, upper),
    }


def time_series_to_octave_frf(x, y, fs, fmin=10.0, fmax=2000.0, fraction=6,
                               df=2.0, window='hann', overlap=0.5, detrend='constant'):
    """Input/response time histories -> narrowband and 1/6-octave (by
    default) H1 frequency response function estimate.

    Returns
    -------
    dict with keys:
        'f_narrow', 'frf_narrow'   : narrowband frequency vector and H1(f)
        'f_octave', 'frf_octave'   : octave-band center frequencies and H1
        'band_edges'               : (lower, upper) octave band edges
    """
    f, Sxx, Syy, Sxy = narrowband_cross_spectra(x, y, fs, df=df, window=window,
                                                 overlap=overlap, detrend=detrend)
    with np.errstate(invalid='ignore', divide='ignore'):
        H_narrow = Sxy / Sxx
    centers, lower, upper = octave_band_frequencies(fmin, fmax, fraction)
    H_oct = octave_band_frf(f, Sxx, Sxy, lower, upper)
    return {
        'f_narrow': f, 'frf_narrow': H_narrow,
        'f_octave': centers, 'frf_octave': H_oct,
        'band_edges': (lower, upper),
    }


def time_series_to_octave_coherence(x, y, fs, fmin=10.0, fmax=2000.0, fraction=6,
                                     df=2.0, window='hann', overlap=0.5, detrend='constant'):
    """Input/response time histories -> narrowband and 1/6-octave (by
    default) ordinary coherence gamma^2(f).

    Returns
    -------
    dict with keys:
        'f_narrow', 'coherence_narrow' : narrowband frequency vector and gamma^2(f)
        'f_octave', 'coherence_octave' : octave-band center frequencies and gamma^2
        'band_edges'                   : (lower, upper) octave band edges
    """
    f, Sxx, Syy, Sxy = narrowband_cross_spectra(x, y, fs, df=df, window=window,
                                                 overlap=overlap, detrend=detrend)
    gamma2_narrow = narrowband_coherence(Sxx, Syy, Sxy)
    centers, lower, upper = octave_band_frequencies(fmin, fmax, fraction)
    gamma2_oct = octave_band_coherence(f, Sxx, Syy, Sxy, lower, upper)
    return {
        'f_narrow': f, 'coherence_narrow': gamma2_narrow,
        'f_octave': centers, 'coherence_octave': gamma2_oct,
        'band_edges': (lower, upper),
    }
