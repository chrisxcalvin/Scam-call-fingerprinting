"""Upload analysis pipeline: audio -> L1 fingerprint, L2 STT/script/flags, L3 voice -> fusion (Owner C)."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..audio.io import load_audio
from ..audio.preprocess import preprocess
from ..audio.vad import vad
from ..config import ARTIFACTS_DIR, FP_DT_RANGE_S, HOP, TARGET_SR
from ..fingerprint.hashing import make_hashes
from ..fingerprint.index import FingerprintIndex
from ..fingerprint.matcher import match, offset_histogram
from ..fingerprint.peaks import find_peaks
from ..fingerprint.spectrogram import spectrogram
from ..fusion.risk import fuse, load_config, script_component
from ..schemas import AnalysisResult, Evidence, RiskComponents, RiskState, TranscriptSegment
from ..understanding import stt
from ..understanding.redflags import red_flags, redflag_score, safe_context_hits
from ..understanding.script_match import ScriptLibrary, embed, script_match
from ..understanding.voice_cues import voice_cues
from .viz import build_viz

log = logging.getLogger(__name__)
MAX_STT_CHUNK_S = 28.0


@dataclass
class Models:
    """Loaded artifacts/models, shared by all requests."""
    index: FingerprintIndex | None = None
    fp_meta: dict = field(default_factory=dict)
    library: ScriptLibrary | None = None
    stt_ready: bool = False

    def load(self, warm_stt: bool = True) -> "Models":
        meta_p, idx_p = ARTIFACTS_DIR / "scripts_meta.json", ARTIFACTS_DIR / "fingerprint_index.pkl"
        if idx_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            self.index, self.fp_meta = FingerprintIndex.load(idx_p), meta["fingerprint"]
            cfg = load_config()
            if "fp_threshold" in cfg:                      # dev-calibrated by run_eval
                self.fp_meta = {**self.fp_meta, "threshold": cfg["fp_threshold"]}
            if meta.get("chunks"):
                self.library = ScriptLibrary.load()
                embed(["warm-up"])                      # load the embedder now, not on the first request
        else:
            log.warning("artifacts missing: run scripts/build_library.py")
        if warm_stt:
            stt.get_model()
            self.stt_ready = True
        return self

    def status(self) -> dict[str, bool]:
        return {"stt": self.stt_ready, "embed": self.library is not None, "index": self.index is not None}


def stt_chunks(segs: list[tuple[float, float]], max_s: float = MAX_STT_CHUNK_S) -> list[tuple[float, float]]:
    """Group consecutive VAD segments into chunks <= max_s (Whisper's window is 30 s); cut only at pauses."""
    out: list[tuple[float, float]] = []
    for a, b in segs:
        if out and b - out[-1][0] <= max_s:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


TIMELINE_STEP_S = 2.0
FP_WINDOW_S = 12.0
OFFSET_HIST_POOL = 4            # 4 frames = 64 ms bins for the UI histogram


def fp_window_hashes(hashes: list[tuple[int, int]], t_end: float, window_s: float = FP_WINDOW_S) -> list[tuple[int, int]]:
    """Hashes a live system would have at time t_end: anchor in [t_end - window, t_end - 2 s]
    (a hash also needs its target peak, up to 2 s after the anchor)."""
    frame_s = HOP / TARGET_SR
    lo, hi = (t_end - window_s) / frame_s, (t_end - FP_DT_RANGE_S[1]) / frame_s
    return [(h, t) for h, t in hashes if lo <= t <= hi]


def risk_timeline(duration_s: float, hashes, transcript: list[TranscriptSegment], models: "Models",
                  cfg: dict, step_s: float = TIMELINE_STEP_S) -> list[RiskState]:
    """Risk every `step_s` seconds using only what was available at that time (causal):
    fingerprint on the last 12 s of hashes (a hit is remembered), script/flags on transcript segments
    that had ended. This is how the score would have evolved during the call."""
    out: list[RiskState] = []
    fp_hit_type: str | None = None
    fp_best = 0.0
    sm_cache: dict[str, list] = {}
    thr = models.fp_meta.get("threshold", float("inf"))
    times = list(np.arange(step_s, duration_s, step_s)) + [duration_s]
    for t in times:
        if models.index is not None and fp_hit_type is None:
            fm = match(np.zeros(0), TARGET_SR, models.index, models.fp_meta, hashes=fp_window_hashes(hashes, t))
            if fm and fm[0].aligned_hashes >= thr:
                fp_hit_type, fp_best = fm[0].scam_type, fm[0].score
        segs = [sg for sg in transcript if sg.end_s <= t]
        text = " ".join(sg.text for sg in segs)
        if text not in sm_cache:
            sm_cache[text] = script_match(text, models.library) if (models.library and text) else []
        sm = sm_cache[text]
        flags = red_flags(segs)
        state = _fuse_layers(fp_hit_type, fp_best, sm, flags, text, cfg, t)
        out.append(state)
    return out


def _fuse_layers(fp_hit_type: str | None, fp_score: float, sm: list, flags: list, text: str, cfg: dict,
                 t_s: float) -> RiskState:
    """Shared by the timeline and the final result: layer outputs -> components -> fuse()."""
    top_sm = sm[0] if sm else None
    comps = RiskComponents(fingerprint=fp_score if fp_hit_type else 0.0,
                           script=script_component(top_sm.similarity, cfg) if top_sm else 0.0,
                           redflags=redflag_score(flags), voice=0.0)   # voice not scored until calibrated
    neg_free = [f for f in flags if not f.negated]
    cats = sorted({f.category for f in neg_free}, key=lambda c: -max(f.weight for f in neg_free if f.category == c))
    details = {"fp_scam_type": fp_hit_type, "script_scam_type": top_sm.scam_type if top_sm else None,
               "similarity": top_sm.similarity if top_sm else 0.0, "flag_categories": cats}
    return fuse(comps, fp_hit_type is not None, t_s=t_s, details=details,
                safe_context=bool(safe_context_hits(text)), cfg=cfg)


def offset_hist_stage(hashes, index: FingerprintIndex, clip_id: str) -> dict[str, list[float]]:
    """Offset histogram of the top clip for the Pipeline tab, pooled to 64 ms bins.
    stages['fp_offset_hist'] = counts; stages['fp_offset_hist_meta'] = [first_bin_offset_s, bin_s]."""
    offs = index.query(hashes).get(clip_id)
    if not offs:
        return {}
    counts, lo = offset_histogram(offs)
    pad = (-len(counts)) % OFFSET_HIST_POOL
    pooled = np.r_[counts, np.zeros(pad, dtype=counts.dtype)].reshape(-1, OFFSET_HIST_POOL).sum(axis=1)
    frame_s = HOP / TARGET_SR
    return {"fp_offset_hist": [float(v) for v in pooled],
            "fp_offset_hist_meta": [round(lo * frame_s, 4), round(OFFSET_HIST_POOL * frame_s, 4)]}


def analyze_audio(data: bytes, models: Models, session_id: str | None = None,
                  transcribe_fn=None) -> tuple[AnalysisResult, Evidence]:
    """Full upload analysis (docs/04 F2) with per-stage timings in ms.
    `transcribe_fn(x, sr, offset_s=...)` replaces Whisper (used by run_eval's transcript cache)."""
    transcribe_fn = transcribe_fn or stt.transcribe
    tm: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        tm[name] = round((time.perf_counter() - t0) * 1000, 1)
        return time.perf_counter()

    t = time.perf_counter()
    raw, sr0 = load_audio(data); t = lap("decode", t)
    x = preprocess(raw, sr0); sr = TARGET_SR; t = lap("preprocess", t)
    segs = vad(x, sr); t = lap("vad", t)
    S_db, _, _ = spectrogram(x, sr); t = lap("stft", t)
    peaks = find_peaks(S_db); hashes = make_hashes(peaks); t = lap("peaks_hashes", t)
    fp = match(x, sr, models.index, models.fp_meta, hashes=hashes) if models.index else []
    t = lap("fingerprint_match", t)

    transcript: list[TranscriptSegment] = []
    for a, b in stt_chunks(segs):
        i0, i1 = int(max(0, a - 0.2) * sr), int(min(len(x) / sr, b + 0.2) * sr)
        transcript += transcribe_fn(x[i0:i1], sr, offset_s=i0 / sr)
    t = lap("stt", t)
    text = " ".join(s.text for s in transcript)
    sm = script_match(text, models.library) if (models.library and text) else []; t = lap("script_match", t)
    flags = red_flags(transcript)
    for s in transcript:
        s.flags = [f for f in flags if f.start_s == s.start_s and f.end_s == s.end_s]
    safe = bool(safe_context_hits(text)); t = lap("red_flags", t)
    vc = voice_cues(x, sr, segs, transcript) if segs else None; t = lap("voice_cues", t)

    cfg = load_config()
    thr = models.fp_meta.get("threshold", float("inf"))
    fp_hit = bool(fp and fp[0].aligned_hashes >= thr)
    final = _fuse_layers(fp[0].scam_type if fp_hit else None, fp[0].score if fp_hit else 0.0, sm, flags, text,
                         cfg, len(x) / sr)
    timeline = risk_timeline(len(x) / sr, hashes, transcript, models, cfg)
    timeline[-1] = final; t = lap("fusion", t)
    viz = build_viz(x, sr, S_db, segs, peaks, raw if sr0 == TARGET_SR else None)
    if fp and models.index is not None:
        viz.stages.update(offset_hist_stage(hashes, models.index, fp[0].clip_id))
    lap("viz", t)
    tm["total"] = round(sum(tm.values()), 1)

    sid = session_id or uuid.uuid4().hex[:12]
    top_sm = sm[0] if sm else None
    # scam-type guess only from a fingerprint hit or a strong script match (component >= 0.5)
    guess = (fp[0].scam_type if fp_hit else None) or (top_sm.scam_type if top_sm and final.components.script >= 0.5 else None)
    result = AnalysisResult(session_id=sid, duration_s=round(len(x) / sr, 2), transcript=transcript,
                            fingerprint=fp[:5], script=sm[:5], red_flags=flags, voice=vc,
                            risk_timeline=timeline, final=final, viz=viz, timings_ms=tm)
    evidence = Evidence(risk=final, fingerprint=fp[:3], script=sm[:3], red_flags=flags, voice=vc,
                        transcript_tail=text[-600:], scam_type_guess=guess)
    log.info("analyzed %s: %.1fs audio, score %.1f (%s), %.0f ms", sid, len(x) / sr, final.score, final.level, tm["total"])
    return result, evidence
