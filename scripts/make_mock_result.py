"""Build web/fixtures/mock_result.json: a MOCK AnalysisResult (validated against the real schema) used only
to develop the UI before the real pipeline exists. Waveform/spectrogram are computed with our own DSP on a
synthetic signal; all risk/transcript/evidence values are hand-written placeholders and the UI labels them MOCK."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scamshield.audio.vad import vad  # noqa: E402
from scamshield.fingerprint.spectrogram import spectrogram  # noqa: E402
from scamshield.schemas import (AnalysisResult, FingerprintMatch, RedFlag, RiskComponents,  # noqa: E402
                                RiskState, ScriptMatch, TranscriptSegment, Viz, VoiceCues)

sr, dur = 16000, 12.0
t = np.arange(int(sr * dur)) / sr
rng = np.random.default_rng(0)
env = ((np.sin(2 * np.pi * 0.35 * t) > -0.3) * 1.0)
x = (0.35 * np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 0.7 * t)) * t) + 0.15 * np.sin(2 * np.pi * 1200 * t)) * env
x = (x + 0.01 * rng.standard_normal(x.size)).astype(np.float32)

S, freqs, _ = spectrogram(x, sr)
S8 = np.clip((S - S.max() + 80) / 80, 0, 1)                      # 80 dB range -> 0..1
u8 = (S8 * 255).astype(np.uint8)
wf = x.reshape(-1, len(x) // 1200)[:1200].mean(axis=1) if False else x[:: len(x) // 1200][:1200]

def rs(ts, score, level, fp, sc, rf, vc, why):
    return RiskState(t_s=ts, score=score, level=level, components=RiskComponents(fingerprint=fp, script=sc, redflags=rf, voice=vc), top_reasons=why)

flag = lambda c, p, w, a, b, neg=False: RedFlag(category=c, phrase=p, weight=w, start_s=a, end_s=b, negated=neg)
f1, f2, f3 = flag("authority", "Cyber Crime Department", .6, 0.5, 2.0), flag("threat", "digital arrest", .9, 3.0, 4.2), flag("secrecy", "do not tell anyone", .8, 6.0, 7.5)
f4 = flag("credential_request", "share your OTP", .7, 9.0, 10.0, neg=True)
why = ["MOCK: matches known digital-arrest script (0.87)", "MOCK: asks to keep the call secret", "MOCK: urgent, fast pace"]
final = rs(12.0, 91.0, "scam", 0.0, 0.87, 0.8, 0.3, why)
res = AnalysisResult(
    session_id="mock", duration_s=dur,
    transcript=[TranscriptSegment(start_s=0.5, end_s=4.5, text="This is the Cyber Crime Department. You are under digital arrest.", language="en", flags=[f1, f2]),
                TranscriptSegment(start_s=5.5, end_s=8.0, text="Do not tell anyone about this call.", language="en", flags=[f3]),
                TranscriptSegment(start_s=8.5, end_s=11.5, text="Never share your OTP with anyone, but transfer funds now.", language="en", flags=[f4])],
    fingerprint=[], script=[ScriptMatch(script_id="scam_digital_arrest_en", scam_type="digital_arrest", similarity=0.87, matched_text="You are under digital arrest...", query_text="You are under digital arrest.")],
    red_flags=[f1, f2, f3, f4],
    voice=VoiceCues(pitch_mean_hz=212, pitch_std_hz=41, speech_rate_wps=3.4, pause_ratio=0.12, energy_std_db=5.1),
    risk_timeline=[rs(0, 5, "safe", 0, 0.05, 0.0, 0.1, []), rs(3, 35, "suspicious", 0, 0.5, 0.5, 0.2, why[:1]), rs(6, 70, "high", 0, 0.8, 0.8, 0.3, why[:2]),
                   rs(9, 88, "scam", 0, 0.87, 0.8, 0.3, why), final],
    final=final,
    viz=Viz(waveform=[float(v) for v in wf], spectrogram_b64=base64.b64encode(u8.tobytes()).decode(), spec_shape=u8.shape,
            spec_freqs_hz=(0.0, sr / 2), spec_duration_s=dur, vad_segments=vad(x, sr)),
    timings_ms={"MOCK_total": 0.0},
)
out = Path(__file__).resolve().parents[1] / "web" / "fixtures" / "mock_result.json"
out.write_text(res.model_dump_json(), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
