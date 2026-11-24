# 03 — API & Data Contracts

All shared types live in `src/scamshield/schemas.py` (Pydantic v2). Changing them requires agreement from all three owners.

## Schemas
```python
from pydantic import BaseModel
from typing import Literal, Optional

Level = Literal["safe", "suspicious", "high", "scam"]

class TranscriptSegment(BaseModel):
    start_s: float; end_s: float; text: str; language: Optional[str] = None
    flags: list["RedFlag"] = []

class RedFlag(BaseModel):
    category: Literal["urgency","authority","secrecy","credential_request","payment","threat","remote_access"]
    phrase: str; weight: float
    start_s: Optional[float] = None; end_s: Optional[float] = None
    negated: bool = False                      # True if safe-context cancelled it

class FingerprintMatch(BaseModel):
    clip_id: str; script_id: str; scam_type: str
    aligned_hashes: int; query_hashes: int
    score: float                               # 0..1 calibrated confidence
    offset_s: float                            # where in the library clip the query aligns
    query_span_s: tuple[float, float]

class ScriptMatch(BaseModel):
    script_id: str; scam_type: str; similarity: float      # cosine 0..1
    matched_text: str; query_text: str

class VoiceCues(BaseModel):
    pitch_mean_hz: Optional[float]; pitch_std_hz: Optional[float]
    speech_rate_wps: Optional[float]; pause_ratio: float; energy_std_db: float
    pressure_prob: Optional[float] = None

class RiskComponents(BaseModel):
    fingerprint: float; script: float; redflags: float; voice: float     # each 0..1

class RiskState(BaseModel):
    t_s: float; score: float; level: Level
    components: RiskComponents; top_reasons: list[str]

class Evidence(BaseModel):                     # single object the agent is grounded on
    risk: RiskState
    fingerprint: list[FingerprintMatch]; script: list[ScriptMatch]
    red_flags: list[RedFlag]; voice: Optional[VoiceCues]
    transcript_tail: str; scam_type_guess: Optional[str] = None

class Viz(BaseModel):
    waveform: list[float]                      # downsampled to <= 2000 points, range -1..1
    spectrogram_b64: str; spec_shape: tuple[int, int]   # (freq, time) uint8 dB-scaled, base64
    spec_freqs_hz: tuple[float, float]; spec_duration_s: float
    vad_segments: list[tuple[float, float]]
    fp_peaks: list[tuple[int, int]] = []       # (time_frame, freq_bin) for overlay (optional, downsampled)
    stages: dict[str, list[float]] = {}        # pipeline-view: "raw","preemph","denoised" short waveform previews

class AnalysisResult(BaseModel):
    session_id: str; duration_s: float
    transcript: list[TranscriptSegment]
    fingerprint: list[FingerprintMatch]; script: list[ScriptMatch]
    red_flags: list[RedFlag]; voice: Optional[VoiceCues]
    risk_timeline: list[RiskState]; final: RiskState
    viz: Viz; timings_ms: dict[str, float]     # per-stage latency (shown in UI + report)

class SpokenMessage(BaseModel):
    text: str; audio_url: Optional[str] = None   # None => client uses speechSynthesis
```

## REST
| Method & path | Request | Response |
|---|---|---|
| `GET /api/health` | – | `{status, models_loaded: {stt, embed, index}, version}` |
| `GET /api/library` | – | list of `{script_id, scam_type, lang, n_clips}` |
| `POST /api/analyze` | multipart `file` (wav/mp3/m4a/webm) | `AnalysisResult` |
| `POST /api/agent/ask` | `{session_id, question}` | `{transcript?: str, reply: SpokenMessage}` |
| `POST /api/agent/ask_audio` | multipart `file` + `session_id` | `{transcript: str, reply: SpokenMessage}` (STT then answer) |
| `GET /api/tts/{cache_key}` | – | audio bytes (cached warning phrases) |
| `GET /api/results` | – | contents of `artifacts/results/summary.json` (eval numbers for Results tab) |
| `POST /api/session` | – | `{session_id}` (live mode) |

Errors: `{error: {code, message}}` with 4xx/5xx; never leak stack traces. Max upload 25 MB / 5 min.

## WebSocket `/ws/live?session_id=...`
Client → server:
- binary frames: **PCM16 little-endian, 16 kHz, mono**, ~100 ms each (3200 bytes). (If the browser can't give 16 kHz, resample client-side in the AudioWorklet.)
- text JSON: `{"type":"start"}`, `{"type":"stop"}`, `{"type":"ping"}`.

Server → client (JSON text):
```json
{"type":"ready"}
{"type":"viz","t_s":12.0,"waveform":[...],"spec_cols_b64":"...","spec_cols":N}          // incremental spectrogram columns
{"type":"transcript","segment":{TranscriptSegment}}
{"type":"fp_match","match":{FingerprintMatch}}
{"type":"risk","state":{RiskState}}                                                      // every ~2 s
{"type":"alert","level":"high","message":"...","audio_url":"/api/tts/warn_digital_arrest_en"}
{"type":"final","result":{AnalysisResult}}                                              // after stop
{"type":"error","message":"..."}
```
Ordering guarantee: `risk` events are monotone in `t_s`. `alert` is sent only when level increases (debounced ≥ 5 s).

## Artifacts on disk (produced by `scripts/build_library.py` & `run_eval`)
```
artifacts/fingerprint_index.pkl     # FingerprintIndex
artifacts/scripts_meta.json         # script_id, type, lang, text, library_clip
artifacts/script_embeddings.npy     # chunk embeddings + chunk->script_id map in scripts_meta
artifacts/fusion_config.json        # weights, thresholds, calibration notes
artifacts/results/summary.json      # all headline metrics for UI/report
artifacts/results/*.csv|png         # per-augmentation tables, ROC, confusion matrices
```
