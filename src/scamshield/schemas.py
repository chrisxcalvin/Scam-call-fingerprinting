"""Shared Pydantic contracts (docs/03_API_CONTRACT.md). Change only by agreement of all three owners.

Note: `RedFlag` is defined before `TranscriptSegment` (the doc lists them in the other order) so no
forward reference is needed; fields are unchanged.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Level = Literal["safe", "suspicious", "high", "scam"]
FlagCategory = Literal[
    "urgency", "authority", "secrecy", "credential_request", "payment", "threat", "remote_access"
]


class RedFlag(BaseModel):
    category: FlagCategory
    phrase: str
    weight: float
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    negated: bool = False  # True if safe-context cancelled it


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    language: Optional[str] = None
    flags: list[RedFlag] = []


class FingerprintMatch(BaseModel):
    clip_id: str
    script_id: str
    scam_type: str
    aligned_hashes: int
    query_hashes: int
    score: float  # 0..1 calibrated confidence
    offset_s: float  # where in the library clip the query aligns
    query_span_s: tuple[float, float]


class ScriptMatch(BaseModel):
    script_id: str
    scam_type: str
    similarity: float  # cosine 0..1
    matched_text: str
    query_text: str


class VoiceCues(BaseModel):
    pitch_mean_hz: Optional[float]
    pitch_std_hz: Optional[float]
    speech_rate_wps: Optional[float]
    pause_ratio: float
    energy_std_db: float
    pressure_prob: Optional[float] = None


class RiskComponents(BaseModel):
    fingerprint: float
    script: float
    redflags: float
    voice: float  # each 0..1


class RiskState(BaseModel):
    t_s: float
    score: float
    level: Level
    components: RiskComponents
    top_reasons: list[str]


class Evidence(BaseModel):
    """Single object the agent is grounded on."""

    risk: RiskState
    fingerprint: list[FingerprintMatch]
    script: list[ScriptMatch]
    red_flags: list[RedFlag]
    voice: Optional[VoiceCues]
    transcript_tail: str
    scam_type_guess: Optional[str] = None


class Viz(BaseModel):
    waveform: list[float]  # downsampled to <= 2000 points, range -1..1
    spectrogram_b64: str
    spec_shape: tuple[int, int]  # (freq, time) uint8 dB-scaled, base64
    spec_freqs_hz: tuple[float, float]
    spec_duration_s: float
    vad_segments: list[tuple[float, float]]
    fp_peaks: list[tuple[int, int]] = []  # (time_frame, freq_bin) for overlay
    stages: dict[str, list[float]] = {}  # pipeline-view waveform previews


class AnalysisResult(BaseModel):
    session_id: str
    duration_s: float
    transcript: list[TranscriptSegment]
    fingerprint: list[FingerprintMatch]
    script: list[ScriptMatch]
    red_flags: list[RedFlag]
    voice: Optional[VoiceCues]
    risk_timeline: list[RiskState]
    final: RiskState
    viz: Viz
    timings_ms: dict[str, float]  # per-stage latency


class SpokenMessage(BaseModel):
    text: str
    audio_url: Optional[str] = None  # None => client uses speechSynthesis
