"""Live mode: level state machine (unit) + simulated live and WebSocket on real clips (need data/artifacts)."""
import json
import os

import numpy as np
import pytest

from scamshield.api.session import LevelTracker
from scamshield.config import ARTIFACTS_DIR, DATA_DIR

BOUNDS = [30.0, 60.0, 85.0]


def test_levels_only_go_up_and_alert_once():
    tr = LevelTracker(BOUNDS)
    assert tr.update(40, 2, False) == ("suspicious", False)          # no alert below high
    assert tr.update(70, 4, False) == ("high", True)
    assert tr.update(20, 6, False) == ("high", False)                # drop ignored...
    assert tr.update(20, 15, False) == ("high", False)
    assert tr.update(20, 16, False) == ("suspicious", False)         # ...until 10 s below 60
    assert tr.update(90, 18, False) == ("scam", True)                # rise to scam alerts again


def test_alert_debounce_and_fp_override():
    tr = LevelTracker(BOUNDS)
    assert tr.update(70, 2, False) == ("high", True)
    assert tr.update(90, 4, False) == ("scam", False)                # within 5 s of last alert
    assert tr.update(90, 7, False) == ("scam", True)                 # delivered once debounce passed
    assert LevelTracker(BOUNDS).update(0, 2, True) == ("scam", True)  # fingerprint hit -> scam


needs_data = pytest.mark.skipif(not (ARTIFACTS_DIR / "fingerprint_index.pkl").exists()
                                or not (DATA_DIR / "clean").exists(), reason="needs data + build_library")


@pytest.fixture(scope="module")
def models():
    from scamshield.api.pipeline import Models
    return Models().load()


def _load(rel):
    from scamshield.audio.io import load_audio
    from scamshield.audio.preprocess import preprocess
    x, sr = load_audio(DATA_DIR / rel)
    return preprocess(x, sr)


@needs_data
def test_simulated_live_scam_alerts_and_legit_stays_low(models):
    from scamshield.api.session import simulate
    s, ev = simulate(models, _load("clean/scam_kyc_otp_en__enUS_Guy__normal.wav"))
    risks = [e["state"] for e in ev if e["type"] == "risk"]
    assert all(a["t_s"] < b["t_s"] for a, b in zip(risks, risks[1:]))     # monotone in time
    assert risks[0]["score"] < risks[-1]["score"] and risks[-1]["level"] in ("high", "scam")
    assert any(e["type"] == "alert" for e in ev) and any(e["type"] == "transcript" for e in ev)
    s2, ev2 = simulate(models, _load("clean/legit_bank_alert_en__enUS_Guy__normal.wav"))
    assert all(e["state"]["level"] in ("safe", "suspicious") for e in ev2 if e["type"] == "risk")
    assert not any(e["type"] == "alert" for e in ev2)


@needs_data
def test_simulated_live_replay_hits_fingerprint_fast(models):
    from scamshield.api.session import simulate
    _, ev = simulate(models, _load("augmented/scam_digital_arrest_en__enIN_Neerja__normal__gsm.wav"))
    alert = next(e for e in ev if e["type"] == "alert")
    assert alert["t_s"] <= 6.0 and any(e["type"] == "fp_match" for e in ev)


@needs_data
def test_websocket_end_to_end():
    os.environ["SCAMSHIELD_WARM_TTS"] = "0"
    from fastapi.testclient import TestClient

    from scamshield.api.main import app
    x = _load("augmented/scam_digital_arrest_en__enIN_Neerja__normal__gsm.wav")[: 16000 * 8]
    pcm = (x * 32767).astype("<i2").tobytes()
    with TestClient(app) as c, c.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "ready"
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i:i + 3200])
        ws.send_text(json.dumps({"type": "stop"}))
        types = []
        while True:
            m = ws.receive_json(); types.append(m["type"])
            if m["type"] in ("final", "error"):
                break
        assert "risk" in types and "viz" in types and types[-1] == "final"
        sid = m["result"]["session_id"]
        assert c.get(f"/api/agent/warn/{sid}").status_code == 200      # agent can use the live call
