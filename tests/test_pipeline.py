"""End-to-end /api/analyze on real dataset clips (skipped if data/ or artifacts/ are missing)."""
import os

import pytest
from fastapi.testclient import TestClient

from scamshield.config import ARTIFACTS_DIR, DATA_DIR

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "fingerprint_index.pkl").exists()
                                or not (DATA_DIR / "augmented").exists(), reason="needs data + build_library")


@pytest.fixture(scope="module")
def client():
    os.environ["SCAMSHIELD_WARM_TTS"] = "0"
    from scamshield.api.main import app
    with TestClient(app) as c:
        yield c


def _post(client, path):
    with open(path, "rb") as f:
        return client.post("/api/analyze", files={"file": (path.name, f, "audio/wav")})


def test_gsm_scam_clip_fingerprint_hit(client):
    r = _post(client, DATA_DIR / "augmented" / "scam_kyc_otp_en__enIN_Neerja__normal__gsm.wav")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["fingerprint"][0]["script_id"] == "scam_kyc_otp_en"
    assert j["final"]["level"] == "scam" and j["transcript"] and j["timings_ms"]["total"] > 0


def test_other_voice_scam_caught_by_script_layer(client):
    j = _post(client, DATA_DIR / "clean" / "scam_digital_arrest_en__enUS_Guy__normal.wav").json()
    assert not j["fingerprint"] or j["fingerprint"][0]["aligned_hashes"] < 50
    assert j["script"][0]["script_id"] == "scam_digital_arrest_en" and j["final"]["score"] >= 60


def test_legit_bank_alert_negated(client):
    j = _post(client, DATA_DIR / "clean" / "legit_bank_alert_en__enUS_Guy__normal.wav").json()
    assert any(f["negated"] for f in j["red_flags"])
    assert j["final"]["level"] in ("safe", "suspicious")


def test_bad_upload_returns_error_json(client):
    r = client.post("/api/analyze", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "bad_audio"


def test_timeline_is_causal_and_ordered(client):
    j = _post(client, DATA_DIR / "augmented" / "scam_digital_arrest_en__enIN_Neerja__normal__snr10.wav").json()
    tl = j["risk_timeline"]
    assert len(tl) > 3 and all(a["t_s"] < b["t_s"] for a, b in zip(tl, tl[1:]))
    assert tl[0]["score"] < tl[-1]["score"] and tl[-1] == j["final"]
    assert j["viz"]["stages"]["fp_offset_hist"] and len(j["viz"]["stages"]["fp_offset_hist_meta"]) == 2


def test_samples_listed_and_served(client):
    s = client.get("/api/samples").json()
    assert s and client.get(f"/api/samples/{s[0]['id']}").status_code == 200
    assert client.get("/api/samples/..%2F..%2Fsecret").status_code == 404


def test_agent_endpoints(client, monkeypatch):
    from scamshield.agent import llm
    monkeypatch.setattr(llm, "provider", lambda: "offline")
    sid = _post(client, DATA_DIR / "clean" / "scam_digital_arrest_en__enUS_Guy__normal.wav").json()["session_id"]
    a = client.post("/api/agent/ask", json={"session_id": sid, "question": "What should I do?"}).json()
    assert "1930" in a["reply"]["text"] and a["reply"]["audio_url"].startswith("/api/tts/ans_")
    assert client.get(f"/api/agent/warn/{sid}").json()["audio_url"] == "/api/tts/warn_digital_arrest"
    assert client.post("/api/agent/ask", json={"session_id": "nope", "question": "hi"}).status_code == 404
    with open(DATA_DIR / "clean" / "legit_friend_en__enUS_Guy__normal.wav", "rb") as f:   # spoken question path
        r = client.post("/api/agent/ask_audio", data={"session_id": sid}, files={"file": ("q.wav", f, "audio/wav")})
    assert r.status_code == 200 and r.json()["transcript"]
