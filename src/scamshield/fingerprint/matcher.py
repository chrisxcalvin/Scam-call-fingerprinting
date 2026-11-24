"""Offset-histogram matching (Owner A)."""
from __future__ import annotations

import numpy as np

from ..config import FP_OFFSET_TOL_FRAMES, HOP, TARGET_SR
from ..schemas import FingerprintMatch
from .hashing import make_hashes
from .index import FingerprintIndex
from .peaks import find_peaks
from .spectrogram import spectrogram


def fingerprint(x: np.ndarray, sr: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """x (16 kHz, preprocessed) -> (peaks, hashes)."""
    S_db, _, _ = spectrogram(x, sr)
    peaks = find_peaks(S_db)
    return peaks, make_hashes(peaks)


def offset_histogram(offsets: list[int]) -> tuple[np.ndarray, int]:
    """Histogram of offsets with 1-frame bins. Returns (counts, min_offset) so bin i <-> offset min+i."""
    o = np.asarray(offsets, dtype=np.int64)
    lo = int(o.min())
    return np.bincount(o - lo), lo


def score_offsets(offsets: list[int]) -> tuple[int, int]:
    """Aligned-hash count = largest number of votes in any window of (2*tol+1) adjacent offset bins
    (tolerates +-1 frame jitter from codecs/resampling). Returns (aligned_count, best_offset_frames)."""
    counts, lo = offset_histogram(offsets)
    w = 2 * FP_OFFSET_TOL_FRAMES + 1
    smoothed = np.convolve(counts, np.ones(w, dtype=np.int64), mode="same")
    best = int(np.argmax(smoothed))
    return int(smoothed[best]), best + lo


def match(x: np.ndarray, sr: int, index: FingerprintIndex, meta: dict,
          hashes: list[tuple[int, int]] | None = None) -> list[FingerprintMatch]:
    """Fingerprint the query, look up the index and score each library clip by the peak of its
    offset histogram (number of hashes aligned at one constant time offset).

    meta = {"clips": {clip_id: {"script_id", "scam_type"}}, "threshold": int}. Only clips whose
    aligned count reaches meta["min_report"] (default 5) are returned; score = a / (a + threshold),
    so score >= 0.5 exactly when the calibrated threshold is reached. Sorted by aligned count.
    """
    if hashes is None:
        _, hashes = fingerprint(x, sr)
    if not hashes:
        return []
    thr = float(meta.get("threshold", 20))
    min_report = int(meta.get("min_report", 5))
    frame_s = HOP / TARGET_SR
    tq = {}
    for h, t in hashes:
        tq.setdefault(h, []).append(t)
    res: list[FingerprintMatch] = []
    for clip_id, offs in index.query(hashes).items():
        aligned, off = score_offsets(offs)
        if aligned < min_report:
            continue
        # query span: anchor times of hashes that voted for (approximately) the winning offset
        hit_t = [t for h, t in hashes for (cid, tdb) in index.table.get(h, ())
                 if cid == clip_id and abs((tdb - t) - off) <= 1] or [0]
        info = meta["clips"].get(clip_id, {})
        res.append(FingerprintMatch(
            clip_id=clip_id, script_id=info.get("script_id", clip_id), scam_type=info.get("scam_type", "unknown"),
            aligned_hashes=aligned, query_hashes=len(hashes), score=round(aligned / (aligned + thr), 4),
            offset_s=round(off * frame_s, 3),
            query_span_s=(round(min(hit_t) * frame_s, 3), round(max(hit_t) * frame_s, 3)),
        ))
    res.sort(key=lambda m: -m.aligned_hashes)
    return res
