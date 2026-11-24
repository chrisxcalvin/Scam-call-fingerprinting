"""Speech-to-text via faster-whisper (Owner B)."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from ..config import TARGET_SR, get_settings
from ..schemas import TranscriptSegment

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load the Whisper model once (CPU, int8). Size from WHISPER_MODEL (default `base`)."""
    from faster_whisper import WhisperModel

    s = get_settings()
    log.info("loading whisper %s (%s)", s.whisper_model, s.whisper_compute_type)
    return WhisperModel(s.whisper_model, device="cpu", compute_type=s.whisper_compute_type)


def transcribe(x: np.ndarray, sr: int, *, offset_s: float = 0.0, language: str | None = None) -> list[TranscriptSegment]:
    """Segment-level transcription with Whisper (encoder-decoder transformer on log-mel input).

    x must be 16 kHz mono float. Language is auto-detected unless given. `offset_s` is added to
    segment times (for transcribing a VAD segment cut out of a longer call). Whisper's own VAD is
    off: we feed it our VAD segments upstream.
    """
    if sr != TARGET_SR:
        raise ValueError(f"transcribe expects {TARGET_SR} Hz, got {sr}")
    if x.size < sr * 0.2:
        return []
    if language is None:
        cfg = get_settings().stt_language
        language = None if cfg == "auto" else cfg
    segs, info = get_model().transcribe(np.asarray(x, dtype=np.float32), language=language,
                                        beam_size=1, vad_filter=False, condition_on_previous_text=False)
    out = [TranscriptSegment(start_s=round(s.start + offset_s, 2), end_s=round(s.end + offset_s, 2),
                             text=s.text.strip(), language=info.language)
           for s in segs if s.text.strip()]
    return out
