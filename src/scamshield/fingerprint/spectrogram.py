"""Own STFT spectrogram (Owner A). No librosa/scipy.signal.stft in src/."""
from __future__ import annotations

import numpy as np

from ..audio.preprocess import frame_signal
from ..config import HOP, N_FFT

_EPS = 1e-10


def spectrogram(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann-windowed STFT magnitude in dB.

    X[k,m] = sum_n x[n + mH] w[n] exp(-j 2 pi k n / N),  w[n] = 0.5 - 0.5 cos(2 pi n / N),
    S_db = 20 log10(|X| + eps).  N = 512 (32 ms), H = 256 (16 ms).
    Returns (S_db[freq, time], freqs_hz, times_s) where times_s is the frame start time.
    """
    n = np.arange(N_FFT)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)
    frames = frame_signal(np.asarray(x, dtype=np.float64), N_FFT, HOP) * w
    X = np.fft.rfft(frames, n=N_FFT, axis=1)            # (time, freq)
    S_db = 20 * np.log10(np.abs(X) + _EPS).T            # (freq, time)
    freqs = np.arange(N_FFT // 2 + 1) * sr / N_FFT
    times = np.arange(frames.shape[0]) * HOP / sr
    return S_db, freqs, times
