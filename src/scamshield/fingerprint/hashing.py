"""Anchor-target hashing (Owner A)."""
from __future__ import annotations

import numpy as np

from ..config import FP_DT_QUANT_S, FP_DT_RANGE_S, FP_FAN_OUT, HOP, TARGET_SR

_F_BITS, _DT_BITS = 9, 8          # f1, f2 in 0..511 (bins 0..256), dt index in 0..255


def make_hashes(peaks: np.ndarray) -> list[tuple[int, int]]:
    """Anchor-target pairing ("combinatorial hashing").

    For each anchor peak (f1, t1), pair with the next FP_FAN_OUT peaks (f2, t2) whose time gap
    dt = t2 - t1 lies in [0.1 s, 2.0 s]. dt is quantised to ~32 ms (2 frames) for tolerance.
        hash32 = (f1 << 17) | (f2 << 8) | dt_q
    The pair encodes *relative* time only, so it is the same wherever the clip starts
    (time-shift invariance); the absolute anchor time t1 is kept separately for the offset vote.
    Returns [(hash32, t1_frame)].
    """
    peaks = np.asarray(peaks)
    if peaks.size == 0:
        return []
    frame_s = HOP / TARGET_SR
    dt_min = int(round(FP_DT_RANGE_S[0] / frame_s))
    dt_max = int(round(FP_DT_RANGE_S[1] / frame_s))
    q = max(1, int(round(FP_DT_QUANT_S / frame_s)))
    order = np.lexsort((peaks[:, 0], peaks[:, 1]))
    f, t = peaks[order, 0].astype(np.int64), peaks[order, 1].astype(np.int64)
    out: list[tuple[int, int]] = []
    n = len(t)
    for i in range(n):
        j0 = np.searchsorted(t, t[i] + dt_min, side="left")
        j1 = min(np.searchsorted(t, t[i] + dt_max, side="right"), j0 + FP_FAN_OUT)
        for j in range(j0, j1):
            dt_q = int((t[j] - t[i]) // q)
            h = (int(f[i]) << (_F_BITS + _DT_BITS)) | (int(f[j]) << _DT_BITS) | dt_q
            out.append((h, int(t[i])))
    return out
