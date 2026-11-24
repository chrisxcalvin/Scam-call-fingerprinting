"""Audio loading (Owner A)."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    """Average channels -> mono, cast to float32."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _ffmpeg_decode(data: bytes) -> bytes:
    """Decode any container (mp3/m4a/webm...) to WAV bytes at the native sample rate via ffmpeg."""
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                       input=data, capture_output=True, check=True)
    return p.stdout


def load_audio(src: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Decode a file path or raw bytes to float32 mono in [-1, 1].

    Returns (x, original_sr). Formats libsndfile cannot read are decoded via ffmpeg.
    """
    data = src if isinstance(src, bytes) else Path(src).read_bytes()
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        x, sr = sf.read(io.BytesIO(_ffmpeg_decode(data)), dtype="float32", always_2d=False)
    return _to_mono_float32(x), int(sr)
