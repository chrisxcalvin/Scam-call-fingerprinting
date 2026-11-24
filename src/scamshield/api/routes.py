"""REST routes (Owner C)."""
from __future__ import annotations

import csv
import json
import logging
import uuid
from collections import Counter, OrderedDict

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..config import APP_VERSION, DATA_DIR, RESULTS_DIR, get_settings
from ..schemas import AnalysisResult, Evidence
from .pipeline import analyze_audio

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
MAX_SESSIONS = 20
_evidence: "OrderedDict[str, Evidence]" = OrderedDict()   # last N analyses, for the agent
_history: dict[str, list[dict]] = {}                       # agent chat history per session
_answers: "OrderedDict[str, str]" = OrderedDict()          # ans_<id> -> reply text (for streamed TTS)
MAX_ANSWERS = 50
LIVE: dict = {}                                            # session_id -> LiveSession while a call is streaming


def _session_evidence(sid: str) -> Evidence | None:
    """Evidence of a finished analysis, or the current snapshot of a live call."""
    if sid in _evidence:
        return _evidence[sid]
    return LIVE[sid].snapshot() if sid in LIVE else None


def error(status: int, code: str, message: str) -> JSONResponse:
    """Uniform error body {error: {code, message}} (never a stack trace)."""
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@router.get("/health")
def health(request: Request) -> dict:
    """Liveness + which models are loaded."""
    models = getattr(request.app.state, "models", None)
    return {"status": "ok", "models_loaded": models.status() if models else {"stt": False, "embed": False, "index": False},
            "version": APP_VERSION}


@router.get("/library")
def library() -> list[dict]:
    """Scripts in the library: `{script_id, scam_type, lang, label, n_clips}`."""
    sp, mp = DATA_DIR / "scripts.json", DATA_DIR / "manifest.csv"
    if not sp.exists():
        return []
    counts: Counter = Counter()
    if mp.exists():
        with open(mp, encoding="utf-8", newline="") as f:
            counts = Counter(r["script_id"] for r in csv.DictReader(f))
    scripts = json.load(open(sp, encoding="utf-8"))
    return [{"script_id": s["id"], "scam_type": s["type"], "lang": s["lang"], "label": s["label"],
             "n_clips": counts.get(s["id"], 0)} for s in scripts]


@router.get("/results")
def results() -> dict:
    """Eval numbers for the Results tab: artifacts/results/summary.json, or {} if run_eval has not run."""
    p = RESULTS_DIR / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(request: Request, file: UploadFile = File(...)):
    """Analyse an uploaded call recording (wav/mp3/m4a/webm, <= 25 MB, <= 5 min)."""
    models = getattr(request.app.state, "models", None)
    if models is None or models.index is None:
        return error(503, "not_ready", "Models are not loaded. Run scripts/build_library.py and restart the server.")
    data = await file.read()
    if len(data) > get_settings().max_upload_mb * 1024 * 1024:
        return error(413, "too_large", f"File is larger than {get_settings().max_upload_mb} MB.")
    if not data:
        return error(400, "empty", "Empty upload.")
    from starlette.concurrency import run_in_threadpool
    try:
        from ..audio.io import load_audio
        x, sr = await run_in_threadpool(load_audio, data)
    except Exception:
        return error(415, "bad_audio", "Could not decode this file. Use wav, mp3, m4a or webm.")
    if len(x) / sr > 300:
        return error(413, "too_long", "Recording is longer than 5 minutes.")
    if len(x) / sr < 0.5:
        return error(400, "too_short", "Recording is shorter than 0.5 s.")
    try:
        result, evidence = await run_in_threadpool(analyze_audio, data, models)
    except Exception:
        log.exception("analysis failed")
        return error(500, "analysis_failed", "Analysis failed on the server. Check the server log.")
    _remember(_evidence, result.session_id, evidence, MAX_SESSIONS)
    return result


@router.get("/tts/{cache_key}")
async def tts_cached(cache_key: str):
    """Warning phrase audio (cached mp3), or a streamed agent answer (`ans_*` keys); 404 if unknown."""
    from ..agent.tts import cached_warning, stream_tts
    if cache_key.startswith("ans_"):
        text = _answers.get(cache_key)
        if text is None:
            return error(404, "unknown_key", "Unknown or expired answer audio.")
        return StreamingResponse(stream_tts(text), media_type="audio/mpeg")
    try:
        audio = await cached_warning(cache_key)
    except Exception:
        return error(503, "tts_unavailable", "TTS service unreachable; use browser speech instead.")
    if audio is None:
        return error(404, "unknown_key", "Unknown warning phrase.")
    return Response(content=audio, media_type="audio/mpeg")


# Curated demo clips (docs/05 "Play sample"): served only from this whitelist, never arbitrary paths.
SAMPLES = [
    ("scam_noisy_gsm", "Scam: KYC/OTP over a GSM phone line (library voice)", "augmented/scam_kyc_otp_en__enIN_Neerja__normal__gsm.wav"),
    ("scam_snr10", "Scam: digital arrest, noisy (10 dB SNR)", "augmented/scam_digital_arrest_en__enIN_Neerja__normal__snr10.wav"),
    ("scam_other_voice", "Scam: digital arrest, different voice (not in fingerprint library)", "clean/scam_digital_arrest_en__enUS_Guy__normal.wav"),
    ("scam_hinglish", "Scam: relative in trouble, Hinglish, telephone band", "augmented/scam_relative_hi__enIN_Prabhat__normal__tel.wav"),
    ("legit_bank_otp", "Legit: bank debit alert that mentions PIN/OTP", "clean/legit_bank_alert_en__enUS_Guy__normal.wav"),
    ("legit_otp_notice", "Legit: OTP notice ('do not share this code')", "clean/legit_otp_notice_en__enIN_Prabhat__normal.wav"),
    ("legit_delivery", "Legit: delivery executive at the gate", "clean/legit_delivery_en__enGB_Ryan__normal.wav"),
]


@router.get("/samples")
def samples() -> list[dict]:
    """Curated demo clips that exist on disk: `{id, title}`."""
    return [{"id": i, "title": t} for i, t, p in SAMPLES if (DATA_DIR / p).exists()]


@router.get("/samples/{sample_id}")
def sample_audio(sample_id: str):
    """WAV bytes of a curated sample (the UI plays it and posts it to /api/analyze)."""
    for i, _, p in SAMPLES:
        if i == sample_id and (DATA_DIR / p).exists():
            return Response(content=(DATA_DIR / p).read_bytes(), media_type="audio/wav")
    return error(404, "unknown_sample", "Unknown sample.")


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=500)


def _remember(store: OrderedDict, key: str, value, limit: int) -> None:
    store[key] = value
    while len(store) > limit:
        store.popitem(last=False)


async def _answer(session_id: str, question: str) -> dict:
    """Grounded answer for a session + a TTS url for it (None if TTS is disabled -> browser speech)."""
    from starlette.concurrency import run_in_threadpool

    from ..agent.agent import Agent
    ev = _session_evidence(session_id)
    hist = _history.setdefault(session_id, [])
    reply = await run_in_threadpool(Agent().answer, question, ev, hist)
    hist += [{"role": "user", "text": question}, {"role": "assistant", "text": reply.text}]
    del hist[:-12]
    key = f"ans_{uuid.uuid4().hex[:10]}"
    _remember(_answers, key, reply.text, MAX_ANSWERS)
    reply.audio_url = f"/api/tts/{key}"
    return {"reply": reply.model_dump()}


@router.post("/agent/ask")
async def agent_ask(req: AskRequest):
    """Typed question about an analysed call -> `{reply: SpokenMessage}`."""
    if _session_evidence(req.session_id) is None:
        return error(404, "unknown_session", "Analyse a call first (session not found or expired).")
    return await _answer(req.session_id, req.question)


@router.post("/agent/ask_audio")
async def agent_ask_audio(session_id: str = Form(...), file: UploadFile = File(...)):
    """Spoken question (push-to-talk recording) -> STT -> grounded answer: `{transcript, reply}`."""
    from starlette.concurrency import run_in_threadpool

    from ..audio.io import load_audio
    from ..audio.preprocess import preprocess
    from ..config import TARGET_SR
    from ..understanding.stt import transcribe
    if _session_evidence(session_id) is None:
        return error(404, "unknown_session", "Analyse a call first (session not found or expired).")
    data = await file.read()
    try:
        x, sr = await run_in_threadpool(load_audio, data)
    except Exception:
        return error(415, "bad_audio", "Could not decode the recording.")
    if len(x) / sr > 30:
        return error(413, "too_long", "Questions must be shorter than 30 seconds.")
    segs = await run_in_threadpool(transcribe, preprocess(x, sr), TARGET_SR)
    question = " ".join(s.text for s in segs).strip()
    if not question:
        return error(422, "no_speech", "I could not hear a question. Hold the button while you speak.")
    out = await _answer(session_id, question)
    return {"transcript": question, **out}


@router.get("/agent/warn/{session_id}")
def agent_warn(session_id: str):
    """The fixed spoken warning for an analysed call (text + cached audio url)."""
    from ..agent.agent import Agent
    ev = _session_evidence(session_id)
    if ev is None:
        return error(404, "unknown_session", "Session not found or expired.")
    return Agent().warn(ev).model_dump()


@router.get("/results/files/{name}")
def result_file(name: str):
    """A PNG/CSV produced by run_eval (plain file names only, no paths)."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(png|csv)", name) or not (RESULTS_DIR / name).is_file():
        return error(404, "not_found", "No such result file.")
    return Response(content=(RESULTS_DIR / name).read_bytes(),
                    media_type="image/png" if name.endswith(".png") else "text/csv")


@router.post("/session")
def new_session() -> dict:
    """A session id for live mode (pass it as /ws/live?session_id=...)."""
    return {"session_id": uuid.uuid4().hex[:12]}
