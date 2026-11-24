"""Voice-cue summary (Owner B)."""
from __future__ import annotations

import numpy as np

from ..dsp.features import short_time_energy, yin_pitch
from ..schemas import TranscriptSegment, VoiceCues


def _speech_mask(n_frames: int, hop_s: float, segs: list[tuple[float, float]]) -> np.ndarray:
    t = np.arange(n_frames) * hop_s
    m = np.zeros(n_frames, dtype=bool)
    for a, b in segs:
        m |= (t >= a) & (t < b)
    return m


def voice_cues(x: np.ndarray, sr: int, vad_segments: list[tuple[float, float]],
               transcript: list[TranscriptSegment] | None = None) -> VoiceCues:
    """Summary of how the caller sounds.

    pitch mean/std: YIN f0 over voiced frames (Hz); pause_ratio = 1 - speech_time / duration (VAD);
    energy_std_db: std of 10 log10(E) over speech frames (25 ms / 10 ms); speech_rate = words /
    speech_time, only if a transcript is given. pressure_prob stays None (no classifier yet).
    """
    dur = len(x) / sr
    speech_s = sum(b - a for a, b in vad_segments)
    f0, _ = yin_pitch(x, sr)
    voiced = f0[~np.isnan(f0)]
    fl, hop = int(0.025 * sr), int(0.010 * sr)
    e_db = 10 * np.log10(short_time_energy(x, fl, hop) + 1e-10)
    m = _speech_mask(len(e_db), hop / sr, vad_segments)
    rate = None
    if transcript and speech_s > 0:
        rate = round(sum(len(s.text.split()) for s in transcript) / speech_s, 2)
    return VoiceCues(
        pitch_mean_hz=round(float(voiced.mean()), 1) if voiced.size else None,
        pitch_std_hz=round(float(voiced.std()), 1) if voiced.size else None,
        speech_rate_wps=rate,
        pause_ratio=round(float(1 - speech_s / dur), 3) if dur > 0 else 0.0,
        energy_std_db=round(float(e_db[m].std()), 2) if m.any() else 0.0,
    )
