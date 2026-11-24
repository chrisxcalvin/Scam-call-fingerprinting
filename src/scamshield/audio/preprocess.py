"""Pre-processing and framing (Owner A). Pure functions: arrays in, arrays out."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def preprocess(x: np.ndarray, sr: int, *, target_sr: int = 16_000) -> np.ndarray:
    """Resample to target_sr (polyphase), remove DC (x - mean), peak-normalise to 0.99."""
    x = np.asarray(x, dtype=np.float64)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g)
    if x.size:
        x = x - x.mean()
        peak = np.max(np.abs(x))
        if peak > 1e-9:
            x = 0.99 * x / peak
    return x.astype(np.float32)


def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """First-order high-pass: y[n] = x[n] - alpha * x[n-1], y[0] = x[0]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    if x.size:
        y[0] = x[0]
        y[1:] = x[1:] - alpha * x[:-1]
    return y


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice x into overlapping frames, shape (n_frames, frame_len); zero-pad the tail.

    n_frames = 1 + ceil((N - frame_len) / hop) for N > frame_len, else 1.
    """
    x = np.asarray(x)
    n = x.shape[0]
    n_frames = 1 if n <= frame_len else 1 + int(np.ceil((n - frame_len) / hop))
    total = (n_frames - 1) * hop + frame_len
    xp = np.zeros(total, dtype=x.dtype)
    xp[:n] = x
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return xp[idx]
