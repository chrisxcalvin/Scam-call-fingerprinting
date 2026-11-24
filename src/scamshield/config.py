"""Shared constants, paths and environment settings. All DSP parameters live here."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
WEB_DIR = ROOT / "web"

# --- audio / framing -------------------------------------------------------
TARGET_SR = 16_000
PRE_EMPHASIS_ALPHA = 0.97
N_FFT = 512                 # 32 ms at 16 kHz
HOP = 256                   # 16 ms
FRAME_LEN = N_FFT

# --- VAD -------------------------------------------------------------------
VAD_FRAME_MS = 25
VAD_HOP_MS = 10
VAD_HANGOVER_MS = 200

# --- fingerprint (starting values, tune with eval) ------------------------------
FP_BAND_HZ = (300.0, 3400.0)
FP_PEAK_NEIGHBOURHOOD = (15, 15)   # (freq bins, time frames)
FP_PEAK_MIN_DB_ABOVE_MEDIAN = 6.0
FP_FAN_OUT = 8
FP_DT_RANGE_S = (0.1, 2.0)
FP_DT_QUANT_S = 0.032
FP_LIVE_WINDOW_S = (8.0, 12.0)
FP_MAX_PEAKS_PER_S = 30
FP_SILENCE_DB_BELOW_MAX = 60.0
FP_OFFSET_TOL_FRAMES = 1

# --- features --------------------------------------------------------------
N_MFCC = 13
N_MELS = 26
YIN_THRESHOLD = 0.1
PITCH_RANGE_HZ = (60.0, 400.0)

# --- risk levels (docs/02): <30 safe | 30-60 suspicious | 60-85 high | >=85 scam
LEVEL_BOUNDS = (30.0, 60.0, 85.0)
FUSION_WEIGHTS = {"fingerprint": 0.95, "script": 0.80, "redflags": 0.60, "voice": 0.25}
FP_OVERRIDE_MIN_SCORE = 90.0
# PROVISIONAL (Phase 2 probe on library-voice/other-voice TTS, not yet calibrated on a dev split):
# script similarity is mapped to 0..1 linearly between these values before fusion.
SCRIPT_SIM_RANGE = (0.55, 0.85)
SAFE_CONTEXT_REDFLAG_CAP = 0.3
FUSION_CONFIG_PATH = ARTIFACTS_DIR / "fusion_config.json"
TTS_CACHE_DIR = ARTIFACTS_DIR / "tts_cache"

# --- live ------------------------------------------------------------------
LIVE_BUFFER_S = 12.0
LIVE_FRAME_MS = 100
LIVE_EVAL_EVERY_S = 2.0
ALERT_DEBOUNCE_S = 5.0

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    """Environment settings loaded from `.env` (see `.env.example`)."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    agent_offline: bool = False
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    stt_language: str = "en"  # "en" = transcribe/translate to English (lexicon is EN + romanised); "auto" = detect
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tts_voice: str = "en-IN-NeerjaNeural"
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 25


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
