"""Live (streaming) analysis sessions (Owner C; incremental VAD/fingerprint by A, STT/fusion by B).

Audio arrives as PCM16 16 kHz mono frames. Every TICK_S seconds of *audio time*:
  1. fingerprint the last 12 s (a hit is remembered for the rest of the call)
  2. VAD on the last 12 s; speech segments followed by >= 0.5 s of silence are "finalised"
     (or forced after 10 s of unbroken speech) and only those are sent to Whisper
  3. script match + red flags over the transcript so far -> fusion
  4. LevelTracker: levels only go up (a step down needs 10 s below the level's lower bound);
     an `alert` is emitted when the level rises to high/scam, at most once every 5 s.
Times in events are audio seconds since the start of the call.
"""
from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..audio.vad import vad
from ..config import ALERT_DEBOUNCE_S, LIVE_BUFFER_S, LIVE_EVAL_EVERY_S, TARGET_SR
from ..fingerprint.hashing import make_hashes
from ..fingerprint.matcher import match
from ..fingerprint.peaks import find_peaks
from ..fingerprint.spectrogram import spectrogram
from ..fusion.risk import level_of, load_config
from ..schemas import Evidence, FingerprintMatch, RiskState, ScriptMatch, TranscriptSegment
from ..understanding import stt
from ..understanding.redflags import red_flags
from ..understanding.script_match import script_match

log = logging.getLogger(__name__)
LEVELS = ["safe", "suspicious", "high", "scam"]
SILENCE_TO_FINALISE_S = 0.5
FORCE_FINALISE_S = 10.0
STEP_DOWN_AFTER_S = 10.0
MAX_CALL_S = 300.0
SPEC_REF_RANGE_DB = 80.0
LOUD_DBFS = -35.0                 # mean power above this with no VAD segment = continuous speech


class LevelTracker:
    """Risk-level state machine for one call (docs/04): monotone up, slow step-down, debounced alerts."""

    def __init__(self, bounds: list[float]) -> None:
        self.bounds = bounds
        self.level = "safe"
        self.below_since: float | None = None
        self.alerted_level = "safe"
        self.last_alert_t = -1e9

    def update(self, score: float, t: float, fp_hit: bool) -> tuple[str, bool]:
        """Return (tracked level, alert_now)."""
        raw = "scam" if fp_hit else level_of(score, self.bounds)
        cur = LEVELS.index(self.level)
        if LEVELS.index(raw) > cur:
            self.level, self.below_since = raw, None
        elif cur > 0:
            lower = self.bounds[cur - 1]
            if score < lower:
                self.below_since = t if self.below_since is None else self.below_since
                if t - self.below_since >= STEP_DOWN_AFTER_S:
                    self.level, self.below_since = LEVELS[cur - 1], None
            else:
                self.below_since = None
        alert = (LEVELS.index(self.level) >= 2 and LEVELS.index(self.level) > LEVELS.index(self.alerted_level)
                 and t - self.last_alert_t >= ALERT_DEBOUNCE_S)
        if alert:
            self.alerted_level, self.last_alert_t = self.level, t
        return self.level, alert


@dataclass
class LiveSession:
    """State of one live call. `push()` appends PCM16 bytes; `tick()` runs one analysis step."""
    models: object                                     # api.pipeline.Models
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    transcribe_fn: object = None
    audio: list = field(default_factory=list)          # float32 chunks
    n_samples: int = 0
    next_tick_t: float = LIVE_EVAL_EVERY_S
    final_t: float = 0.0                               # audio up to here has been transcribed
    transcript: list = field(default_factory=list)
    fp_hit: FingerprintMatch | None = None
    fp_best: FingerprintMatch | None = None
    last_state: RiskState | None = None
    script: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    spec_ref_db: float = -1e9
    viz_t: float = 0.0
    tick_ms: list = field(default_factory=list)
    peak: float = 0.0                                  # loudest absolute sample received (0..1)
    zero_samples: int = 0                              # samples that were exactly zero (a muted/blocked source)

    def __post_init__(self) -> None:
        self.cfg = load_config()
        self.tracker = LevelTracker(self.cfg["level_bounds"])
        self.transcribe_fn = self.transcribe_fn or stt.transcribe
        self._sm_cache: dict[str, list[ScriptMatch]] = {}
        self._tick_adjust_ms = 0.0

    # ---- audio
    @property
    def duration_s(self) -> float:
        return self.n_samples / TARGET_SR

    def push(self, pcm16: bytes) -> None:
        """Append little-endian PCM16 mono 16 kHz audio (ignores audio beyond 5 minutes)."""
        if self.duration_s >= MAX_CALL_S or not pcm16:
            return
        x = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        self.audio.append(x)
        self.n_samples += x.size
        if x.size:
            self.peak = max(self.peak, float(np.max(np.abs(x))))
            self.zero_samples += int(np.count_nonzero(x == 0))

    def signal(self, t0: float = 0.0, t1: float | None = None) -> np.ndarray:
        """Audio between t0 and t1 seconds (concatenates chunks lazily)."""
        if len(self.audio) > 1:
            self.audio = [np.concatenate(self.audio)]
        x = self.audio[0] if self.audio else np.zeros(0, np.float32)
        return x[int(t0 * TARGET_SR): None if t1 is None else int(t1 * TARGET_SR)]

    def ready(self) -> bool:
        """True when enough new audio has arrived for the next tick."""
        return self.duration_s >= self.next_tick_t

    # ---- one analysis step
    def tick(self, final: bool = False) -> list[dict]:
        """Run one analysis step at the current audio time; returns the events to send."""
        t_start = time.perf_counter()
        self._tick_adjust_ms = 0.0
        now = self.duration_s
        self.next_tick_t = now + LIVE_EVAL_EVERY_S
        events: list[dict] = []
        w0 = max(0.0, now - LIVE_BUFFER_S)
        win = self.signal(w0, now)
        if win.size < TARGET_SR // 2:
            return events
        events.append(self._viz_event(now))
        # 1. fingerprint on the rolling window
        if self.models.index is not None and self.fp_hit is None:
            S_db, _, _ = spectrogram(win, TARGET_SR)
            fm = match(win, TARGET_SR, self.models.index, self.models.fp_meta, hashes=make_hashes(find_peaks(S_db)))
            if fm:
                m = fm[0].model_copy(update={"query_span_s": (round(fm[0].query_span_s[0] + w0, 2),
                                                               round(fm[0].query_span_s[1] + w0, 2))})
                if self.fp_best is None or m.aligned_hashes > self.fp_best.aligned_hashes:
                    self.fp_best = m
                if m.aligned_hashes >= self.models.fp_meta.get("threshold", float("inf")):
                    self.fp_hit = m
                    events.append({"type": "fp_match", "match": m.model_dump()})
        # 2. incremental STT on finalised speech
        for seg in self._finalise(win, w0, now, final):
            events.append({"type": "transcript", "segment": seg.model_dump()})
        # 3. fuse
        from .pipeline import _fuse_layers
        text = " ".join(s.text for s in self.transcript)
        if text not in self._sm_cache:
            self._sm_cache = {text: script_match(text, self.models.library) if (self.models.library and text) else []}
        self.script = self._sm_cache[text]
        self.flags = red_flags(self.transcript)
        state = _fuse_layers(self.fp_hit.scam_type if self.fp_hit else None, self.fp_hit.score if self.fp_hit else 0.0,
                             self.script, self.flags, text, self.cfg, now, self.models)
        level, alert = self.tracker.update(state.score, now, self.fp_hit is not None)
        state = state.model_copy(update={"level": level})
        self.last_state = state
        self.timeline.append(state)
        events.append({"type": "risk", "state": state.model_dump()})
        if alert:
            from ..agent.agent import Agent
            msg = Agent().warn(self.snapshot())
            events.append({"type": "alert", "level": level, "t_s": round(now, 2), "message": msg.text,
                           "audio_url": msg.audio_url})
        self.tick_ms.append((time.perf_counter() - t_start) * 1000 + self._tick_adjust_ms)
        return events

    def _finalise(self, win: np.ndarray, w0: float, now: float, final: bool) -> list[TranscriptSegment]:
        """Transcribe speech that has ended (>= 0.5 s silence after it), or force after 10 s / at call end."""
        segs = [(a + w0, b + w0) for a, b in vad(win, TARGET_SR) if b + w0 > self.final_t + 0.05]
        done = [s for s in segs if s[1] <= now - SILENCE_TO_FINALISE_S]
        end = None
        if final:
            end = now
        elif done:
            end = min(done[-1][1] + 0.2, now)
        elif now - self.final_t >= FORCE_FINALISE_S:
            end = now                                  # continuous speech (or VAD could not separate)
        if end is None:
            return []
        start = max(self.final_t, (segs[0][0] - 0.2) if segs else self.final_t)
        self.final_t = end
        chunk = self.signal(start, end)
        if end - start < 0.3 or chunk.size == 0:
            return []
        if not segs:
            # VAD found nothing: either silence (skip it: Whisper invents text on silence) or unbroken
            # speech with no pause to measure a noise floor from (loud -> transcribe anyway).
            rms_db = 10 * np.log10(np.mean(chunk.astype(np.float64) ** 2) + 1e-12)
            if rms_db < LOUD_DBFS:
                return []
        t = time.perf_counter()
        new = self.transcribe_fn(chunk, TARGET_SR, offset_s=start)
        # an eval cache reports the Whisper time of its first (real) run; count that, not the cache lookup
        recorded = getattr(self.transcribe_fn, "last_ms", None)
        if recorded is not None:
            self._tick_adjust_ms += recorded - (time.perf_counter() - t) * 1000
        # Whisper timestamps can overrun the chunk; clamp so segments stay ordered and non-overlapping
        prev_end = self.transcript[-1].end_s if self.transcript else 0.0
        for seg in new:
            seg.start_s = round(min(max(seg.start_s, start, prev_end), end), 2)
            seg.end_s = round(max(min(seg.end_s, end), seg.start_s), 2)
            prev_end = seg.end_s
        self.transcript.extend(new)
        return new

    def _viz_event(self, now: float) -> dict:
        """Waveform points + spectrogram columns for the audio since the last viz event."""
        from ..config import HOP, N_FFT
        t0 = max(0.0, self.viz_t - N_FFT / TARGET_SR)
        x = self.signal(t0, now)
        S_db, _, _ = spectrogram(x, TARGET_SR)
        skip = int(round((self.viz_t - t0) * TARGET_SR / HOP))
        S_db = S_db[:, skip:]
        self.spec_ref_db = max(self.spec_ref_db, float(S_db.max()) if S_db.size else -1e9)
        S_db = S_db[:, : S_db.shape[1] // 2 * 2].reshape(S_db.shape[0], -1, 2).max(axis=2)  # 32 ms columns
        u8 = (np.clip((S_db - (self.spec_ref_db - SPEC_REF_RANGE_DB)) / SPEC_REF_RANGE_DB, 0, 1) * 255).astype(np.uint8)
        new = self.signal(self.viz_t, now)
        n = max(1, int((now - self.viz_t) * 50))                 # 50 waveform points per second
        wf = [round(float(b[np.argmax(np.abs(b))]), 3) for b in np.array_split(new, n) if b.size]
        self.viz_t = now
        return {"type": "viz", "t_s": round(now, 2), "waveform": wf,
                "spec_cols_b64": base64.b64encode(np.ascontiguousarray(u8.T).tobytes()).decode(),
                "spec_cols": int(u8.shape[1]), "spec_rows": int(u8.shape[0])}

    def diag(self) -> dict:
        """What the server actually received: used to tell 'browser sent silence' from 'microphone too quiet'."""
        n = max(1, self.n_samples)
        rms = float(np.sqrt(np.mean(self.signal().astype(np.float64) ** 2))) if self.n_samples else 0.0
        return {"audio_s": round(self.duration_s, 1), "peak": round(self.peak, 4), "rms": round(rms, 5),
                "zero_frac": round(self.zero_samples / n, 3), "transcript_segments": len(self.transcript)}

    def snapshot(self) -> Evidence:
        """Current Evidence (what the agent may talk about)."""
        st = self.last_state or RiskState(t_s=0, score=0, level="safe",
                                          components={"fingerprint": 0, "script": 0, "redflags": 0, "voice": 0},
                                          top_reasons=[])
        top = self.script[0] if self.script else None
        guess = (self.fp_hit.scam_type if self.fp_hit else None) or \
            (top.scam_type if top and st.components.script >= 0.5 else None)
        return Evidence(risk=st, fingerprint=[self.fp_hit] if self.fp_hit else [], script=self.script[:3],
                        red_flags=self.flags, voice=None,
                        transcript_tail=" ".join(s.text for s in self.transcript)[-600:], scam_type_guess=guess)


class SessionManager:
    """In-memory live sessions (no audio is stored on disk)."""

    def __init__(self, models=None) -> None:
        self.models = models
        self.sessions: dict[str, LiveSession] = {}

    def create(self) -> str:
        """New session id with an empty buffer."""
        s = LiveSession(self.models)
        self.sessions[s.session_id] = s
        return s.session_id

    def get(self, sid: str) -> LiveSession | None:
        return self.sessions.get(sid)

    def push_audio(self, sid: str, pcm_bytes: bytes) -> None:
        """Append PCM16 16 kHz mono to the session's buffer."""
        self.sessions[sid].push(pcm_bytes)

    def snapshot(self, sid: str) -> Evidence:
        """Current Evidence for the agent."""
        return self.sessions[sid].snapshot()

    def close(self, sid: str) -> None:
        """Forget a session (its audio is dropped)."""
        self.sessions.pop(sid, None)


def simulate(models, x: np.ndarray, transcribe_fn=None, frame_ms: int = 100) -> tuple[LiveSession, list[dict]]:
    """Feed a 16 kHz float signal through a LiveSession frame by frame (simulated time, no sleeping).
    Same code path as the WebSocket; used by tests and run_eval's detection-latency measurement."""
    s = LiveSession(models, transcribe_fn=transcribe_fn)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    step = TARGET_SR * frame_ms // 1000 * 2
    events: list[dict] = []
    for i in range(0, len(pcm), step):
        s.push(pcm[i:i + step])
        if s.ready():
            events += [dict(e, wall_ms=s.tick_ms[-1] if s.tick_ms else 0) for e in s.tick()]
    events += s.tick(final=True)
    return s, events
