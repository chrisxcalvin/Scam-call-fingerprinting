from fastapi.testclient import TestClient

from scamshield.api.main import app


def test_health():
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_static_index():
    assert "ScamShield" in TestClient(app).get("/").text


def test_library_lists_scripts():
    r = TestClient(app).get("/api/library")
    assert r.status_code == 200
    rows = r.json()
    for row in rows:
        assert {"script_id", "scam_type", "lang", "n_clips"} <= set(row)


def test_results_empty_until_eval_run():
    assert TestClient(app).get("/api/results").status_code == 200
