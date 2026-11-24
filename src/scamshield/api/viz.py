"""Build the `Viz` payload for the UI (Owner C)."""
from __future__ import annotations

import base64

import numpy as np

from ..audio.preprocess import pre_emphasis
from ..schemas import Viz

MAX_WAVE_POINTS = 2000
MAX_SPEC_COLS = 1000
SPEC_DB_RANGE = 80.0
MAX_PEAKS = 3000


def downsample_wave(x: np.ndarray, n: int = MAX_WAVE_POINTS) -> list[float]:
    """Keep the signed sample of largest magnitude in each of n blocks (preserves the envelope)."""
    if x.size <= n:
        return [round(float(v), 4) for v in x]
    blocks = np.array_split(x, n)
    return [round(float(b[np.argmax(np.abs(b))]), 4) for b in blocks]


def spectrogram_u8(S_db: np.ndarray, max_cols: int = MAX_SPEC_COLS) -> tuple[np.ndarray, int]:
    """Max-pool time to <= max_cols, map [max-80 dB, max] -> 0..255. Returns (u8[freq, time], pool)."""
    pool = int(np.ceil(S_db.shape[1] / max_cols)) or 1
    T = S_db.shape[1] // pool * pool or S_db.shape[1]
    S = S_db[:, :T].reshape(S_db.shape[0], -1, pool).max(axis=2) if pool > 1 else S_db
    top = S.max()
    return (np.clip((S - (top - SPEC_DB_RANGE)) / SPEC_DB_RANGE, 0, 1) * 255).astype(np.uint8), pool


def build_viz(x: np.ndarray, sr: int, S_db: np.ndarray, vad_segments: list[tuple[float, float]],
              peaks: np.ndarray | None = None, raw: np.ndarray | None = None) -> Viz:
    """Waveform, uint8 spectrogram (base64), VAD bands, fingerprint peaks in the pooled grid as
    (time_col, freq_bin), and short stage previews (first 2 s) for the Pipeline tab."""
    u8, pool = spectrogram_u8(S_db)
    fp = []
    if peaks is not None and len(peaks):
        p = peaks[np.linspace(0, len(peaks) - 1, min(MAX_PEAKS, len(peaks))).astype(int)]
        fp = [(int(t // pool), int(f)) for f, t in p if t // pool < u8.shape[1]]
    n2 = min(len(x), 2 * sr)
    stages = {"preprocessed": downsample_wave(x[:n2], 400), "preemph": downsample_wave(pre_emphasis(x[:n2]), 400)}
    if raw is not None:
        stages["raw"] = downsample_wave(raw[: len(raw) * n2 // max(len(x), 1)], 400)
    return Viz(waveform=downsample_wave(x), spectrogram_b64=base64.b64encode(u8.tobytes()).decode(),
               spec_shape=(int(u8.shape[0]), int(u8.shape[1])), spec_freqs_hz=(0.0, sr / 2),
               spec_duration_s=round(len(x) / sr, 3), vad_segments=vad_segments, fp_peaks=fp, stages=stages)
