"""Voice activity detection (Owner A)."""
from __future__ import annotations

import numpy as np

from ..config import VAD_FRAME_MS, VAD_HANGOVER_MS, VAD_HOP_MS
from .preprocess import frame_signal

_MIN_SPEECH_S = 0.10
_MIN_RANGE_DB = 10.0       # below this dynamic range there is nothing to separate
_ZCR_FRICATIVE = 0.25


def vad(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Return speech segments (start_s, end_s).

    Per 25 ms frame (10 ms hop): energy E = sum(x^2) in dB and zero-crossing rate.
    Noise floor = 10th percentile of E_dB; speech level = 95th percentile. Threshold
    T = floor + 0.3 * (speech - floor). Speech = E_dB > T, or (voiced-fricative rule)
    E_dB > floor + 0.15*(speech-floor) and ZCR > 0.25 (weak but noisy /s/, /f/).
    A 200 ms hangover bridges short gaps so words are not chopped; segments < 100 ms are dropped.
    """
    x = np.asarray(x, dtype=np.float64)
    fl, hop = int(sr * VAD_FRAME_MS / 1000), int(sr * VAD_HOP_MS / 1000)
    if x.size < fl:
        return []
    frames = frame_signal(x, fl, hop)
    e_db = 10 * np.log10(np.sum(frames ** 2, axis=1) + 1e-10)
    s = np.sign(frames)
    zcr = np.mean(np.abs(np.diff(s, axis=1)), axis=1) / 2
    floor, top = np.percentile(e_db, 10), np.percentile(e_db, 95)
    if top - floor < _MIN_RANGE_DB:
        return []
    thr = floor + 0.3 * (top - floor)
    act = (e_db > thr) | ((e_db > floor + 0.15 * (top - floor)) & (zcr > _ZCR_FRICATIVE))
    # hangover: dilate by N frames (bridges gaps shorter than 2*N)
    n_hang = int(VAD_HANGOVER_MS / VAD_HOP_MS)
    dil = np.convolve(act.astype(int), np.ones(2 * n_hang + 1, int), mode="same") > 0
    # only keep bridging: shrink back the outer edges by N frames
    edges = np.diff(np.r_[0, dil.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    segs = []
    for a, b in zip(starts, ends):
        # trim the dilation margin at the outer edges, but restrict to actual active frames
        idx = np.where(act[a:b])[0]
        if idx.size == 0:
            continue
        a2, b2 = a + idx[0], a + idx[-1] + 1
        t0, t1 = a2 * hop / sr, (b2 * hop + fl) / sr
        if t1 - t0 >= _MIN_SPEECH_S:
            segs.append((round(float(t0), 3), round(float(min(t1, x.size / sr)), 3)))
    return segs
