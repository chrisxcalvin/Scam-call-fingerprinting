# ScamShield Voice

A speech-processing system that listens to a phone call (uploaded file or live mic), detects likely scams, shows the evidence, and **talks to the user** — spoken warnings plus voice Q&A.

> Course project — MSc AIML, *Speech Processing and Recognition* (MAI522-5), CIA III. Staged and synthetic data only.

## How it works
1. **Acoustic fingerprint (from scratch)** — spectrogram peaks → hashes → offset-histogram matching against known scam audio; robust to noise, telephone band and codecs.
2. **Script matching** — Whisper transcript → multilingual embeddings + weighted red-flag lexicon (negation aware) vs known scam scripts.
3. **Voice cues** — pitch (YIN), energy, pace, pauses.
4. **Fusion** → risk 0–100 with plain-language reasons.
5. **Voice agent** — speaks warnings and answers spoken questions, grounded only in the evidence.

See `docs/` for the full design: problem & scope, architecture, API contract, flows, UI spec, data & evaluation, phases.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # also install ffmpeg and put it on PATH
cp .env.example .env                                     # add your API keys
# put the generated dataset into data/ (see docs/06_DATA_AND_EVAL.md)
python scripts/check_data.py
python scripts/build_library.py
PYTHONPATH=src uvicorn scamshield.api.main:app --reload --port 8000
# open http://localhost:8000
```

## Evaluate
```bash
PYTHONPATH=src python -m scamshield.eval.run_eval        # writes artifacts/results/
pytest -q
```

## Repo map
`src/scamshield/` audio · fingerprint · dsp · understanding · fusion · agent · api · eval — `web/` static UI — `docs/` design docs — `scripts/` data + build tools — `artifacts/` generated indices/results (gitignored).

## Limitations
Detects known and similar scams, not all scams. Live phone-call interception is not possible on iOS/stock Android; the demo uses speakerphone/mic or uploaded audio. Data is mostly synthetic TTS plus a small consented human set. Not a substitute for calling your bank.

## Team
Owner A: audio + fingerprint · Owner B: features + understanding + fusion + eval · Owner C: agent + API + UI.
