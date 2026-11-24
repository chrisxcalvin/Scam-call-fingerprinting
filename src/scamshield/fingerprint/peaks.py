"""Spectral peak picking (Owner A)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from ..config import (FP_BAND_HZ, FP_MAX_PEAKS_PER_S, FP_PEAK_MIN_DB_ABOVE_MEDIAN,
                      FP_PEAK_NEIGHBOURHOOD, FP_SILENCE_DB_BELOW_MAX, HOP, N_FFT, TARGET_SR)


def band_bins(sr: int = TARGET_SR, n_fft: int = N_FFT) -> tuple[int, int]:
    """FFT bin range [lo, hi) covering the telephone band 300-3400 Hz (bin k <-> k*sr/N Hz)."""
    lo = int(np.ceil(FP_BAND_HZ[0] * n_fft / sr))
    hi = int(np.floor(FP_BAND_HZ[1] * n_fft / sr)) + 1
    return lo, hi


def find_peaks(S_db: np.ndarray) -> np.ndarray:
    """Spectral peaks ("constellation map").

    A cell (k, m) is a peak if, inside the 300-3400 Hz band:
      1. S[k,m] == max over its 15x15 (freq x time) neighbourhood   (local maximum)
      2. S[k,m] > median(S_band) + 6 dB                               (above the typical level)
      3. S[k,m] > max(S_band) - 60 dB                                  (not digital silence)
    then at most FP_MAX_PEAKS_PER_S strongest peaks are kept per 1 s block (density cap, keeps
    the index small and spreads peaks over time). Band-limiting means telephone-band queries see
    the same peaks as full-band library clips.
    Returns int array (n, 2) of (freq_bin, time_frame), sorted by time then frequency.
    """
    lo, hi = band_bins()
    B = S_db[lo:hi]
    if B.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    local_max = maximum_filter(B, size=FP_PEAK_NEIGHBOURHOOD, mode="constant", cval=-np.inf) == B
    thr = max(np.median(B) + FP_PEAK_MIN_DB_ABOVE_MEDIAN, B.max() - FP_SILENCE_DB_BELOW_MAX)
    f, t = np.nonzero(local_max & (B > thr))
    mag = B[f, t]
    frames_per_s = TARGET_SR / HOP
    block = (t / frames_per_s).astype(int)
    keep = np.zeros(f.size, dtype=bool)
    for b in np.unique(block):
        idx = np.where(block == b)[0]
        keep[idx[np.argsort(-mag[idx])[:FP_MAX_PEAKS_PER_S]]] = True
    peaks = np.stack([f[keep] + lo, t[keep]], axis=1).astype(np.int64)
    order = np.lexsort((peaks[:, 0], peaks[:, 1]))
    return peaks[order]
