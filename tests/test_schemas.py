"""Schema round-trip (docs/03)."""
from scamshield.schemas import (AnalysisResult, Evidence, FingerprintMatch, RedFlag, RiskComponents,
                                RiskState, SpokenMessage, TranscriptSegment, Viz, VoiceCues)


def _risk() -> RiskState:
    return RiskState(t_s=1.0, score=91.5, level="scam",
                     components=RiskComponents(fingerprint=1, script=0.8, redflags=0.5, voice=0.1),
                     top_reasons=["Matches known script"])


def test_analysis_result_roundtrip():
    flag = RedFlag(category="secrecy", phrase="do not tell anyone", weight=0.7, start_s=1.0, end_s=2.0)
    seg = TranscriptSegment(start_s=0, end_s=3, text="hello", language="en", flags=[flag])
    fp = FingerprintMatch(clip_id="c", script_id="s", scam_type="kyc_otp", aligned_hashes=10,
                          query_hashes=50, score=0.9, offset_s=1.5, query_span_s=(0.0, 3.0))
    voice = VoiceCues(pitch_mean_hz=200, pitch_std_hz=30, speech_rate_wps=3.1, pause_ratio=0.1, energy_std_db=4)
    viz = Viz(waveform=[0.0, 0.1], spectrogram_b64="AA==", spec_shape=(2, 2), spec_freqs_hz=(0, 8000),
              spec_duration_s=1.0, vad_segments=[(0.0, 1.0)])
    res = AnalysisResult(session_id="x", duration_s=3.0, transcript=[seg], fingerprint=[fp], script=[],
                         red_flags=[flag], voice=voice, risk_timeline=[_risk()], final=_risk(), viz=viz,
                         timings_ms={"stt": 12.0})
    assert AnalysisResult.model_validate_json(res.model_dump_json()) == res
    ev = Evidence(risk=_risk(), fingerprint=[fp], script=[], red_flags=[flag], voice=voice, transcript_tail="...")
    assert Evidence.model_validate(ev.model_dump()) == ev
    assert SpokenMessage(text="hi").audio_url is None
