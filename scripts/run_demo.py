"""One command to start the demo:  python scripts/run_demo.py [--port 8000] [--offline] [--no-browser]

1. checks Python packages, ffmpeg and .env (creates .env from .env.example if missing)
2. validates data/ (scripts/check_data.py)
3. builds artifacts/ (scripts/build_library.py) if the index or embeddings are missing
4. warns if fusion_config.json is not dev-calibrated (run: python -m scamshield.eval.run_eval)
5. starts the server and opens http://localhost:<port>
--offline forces the agent's canned grounded replies (no LLM network calls): the venue fallback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REQUIRED = ["numpy", "scipy", "soundfile", "fastapi", "uvicorn", "pydantic_settings", "faster_whisper",
            "sentence_transformers", "edge_tts", "multipart"]


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}", flush=True)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--offline", action="store_true", help="agent uses offline templates (no LLM calls)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    step("Checking environment")
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        fail(f"missing Python packages: {', '.join(missing)}\n       run: pip install -r requirements.txt")
    if shutil.which("ffmpeg") is None:
        print("WARN: ffmpeg not on PATH: mp3/m4a/webm uploads and push-to-talk questions will fail (wav still works)")
    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        print("created .env from .env.example (add GEMINI_API_KEY or ANTHROPIC_API_KEY for the LLM agent)")
    print("ok")

    step("Validating data/")
    if subprocess.run([PY, str(ROOT / "scripts" / "check_data.py")], cwd=ROOT).returncode != 0:
        fail("data/ is incomplete. Generate it with scripts/gen_tts.py + scripts/augment.py (docs/06).")

    art = ROOT / "artifacts"
    if not (art / "fingerprint_index.pkl").exists() or not (art / "script_embeddings.npy").exists():
        step("Building library artifacts (first run only)")
        if subprocess.run([PY, str(ROOT / "scripts" / "build_library.py")], cwd=ROOT).returncode != 0:
            fail("build_library.py failed")
    cfg = art / "fusion_config.json"
    if not cfg.exists() or "fp_threshold" not in json.loads(cfg.read_text(encoding="utf-8")):
        print("\nWARN: thresholds are PROVISIONAL. Calibrate once with:  python -m scamshield.eval.run_eval --split dev")

    step(f"Starting server on http://localhost:{a.port}  (Ctrl+C to stop)")
    envv = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if a.offline:
        envv["AGENT_OFFLINE"] = "1"
        print("agent: OFFLINE templates")
    srv = subprocess.Popen([PY, "-m", "uvicorn", "scamshield.api.main:app", "--host", "127.0.0.1",
                            "--port", str(a.port)], cwd=ROOT, env=envv)

    def open_when_ready() -> None:
        for _ in range(180):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/health", timeout=2) as r:
                    h = json.loads(r.read())
                print(f"\nREADY  models: {h['models_loaded']}  ->  http://localhost:{a.port}", flush=True)
                if not a.no_browser:
                    webbrowser.open(f"http://localhost:{a.port}")
                return
            except Exception:
                time.sleep(1)
        print("server did not become ready in 3 minutes; check the log above")

    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        srv.wait()
    except KeyboardInterrupt:
        srv.terminate()
        srv.wait()


if __name__ == "__main__":
    main()
