# CLAUDE.md — ScamShield Voice

Read this file first, every session. Then read `docs/01` → `docs/07` in order before writing code.

## What this project is
A speech-processing system that listens to a call (uploaded file or live mic), decides whether it is a scam, shows the evidence, and **talks to the user** (spoken warnings + voice Q&A). It is a university project (MSc AIML, *Speech Processing and Recognition*, CIA III). It is graded on: working live demo, correctness of the speech-processing pipeline, concept clarity in an **individual viva**, presentation, and innovation. Team of 3; each member is questioned on their own module.

**Consequence for you (Claude Code):** every core signal-processing step must be implemented by us (NumPy/SciPy only) so a student can explain it line by line. Never swap a from-scratch module for a one-line library call.

## Detection design (do not change without asking)
1. **Layer 1 – acoustic fingerprint** (from scratch): spectrogram peaks → hashes → offset-histogram matching against a library of known scam audio. Catches replayed / robocall audio through noise, telephone band, codec.
2. **Layer 2 – script matching**: STT transcript → embedding similarity vs scam-script library + weighted red-flag lexicon (with negation/safe-context handling). Catches live callers / different voices.
3. **Layer 3 – voice cues**: pitch, energy, pace, pauses (from scratch) + optional small pressure classifier. Low weight; supporting evidence only.
4. **Fusion** → risk 0–100 + level + human-readable reasons.
5. **Voice agent**: speaks warnings; answers spoken questions grounded ONLY in the evidence object.

## Stack (decided)
Python 3.11, FastAPI + WebSocket, NumPy/SciPy/soundfile, faster-whisper (STT), sentence-transformers (embeddings), scikit-learn, edge-tts (TTS, with pre-cached warnings), Anthropic API for the agent (provider abstracted), static frontend (HTML/JS/CSS + Canvas/Chart.js, **no build step**). No Docker, no external DB (SQLite/pickle/JSON only).

## Commands
```
pip install -r requirements.txt
cp .env.example .env            # fill keys
python scripts/check_data.py    # validates data/ layout + manifests
python scripts/build_library.py # builds fingerprint index + script embeddings -> artifacts/
uvicorn scamshield.api.main:app --reload --port 8000   # run from repo root with PYTHONPATH=src
python -m scamshield.eval.run_eval   # writes artifacts/results/*.json + figures
pytest -q
```

## Repo layout & ownership
```
src/scamshield/
  config.py schemas.py                   # shared contracts (change only via PR to all 3)
  audio/        io.py preprocess.py vad.py            # Owner A
  fingerprint/  spectrogram.py peaks.py hashing.py index.py matcher.py   # Owner A
  dsp/          features.py (MFCC, YIN pitch, energy, ZCR)               # Owner B
  understanding/ stt.py script_match.py redflags.py voice_cues.py        # Owner B
  fusion/       risk.py                                                  # Owner B
  eval/         metrics.py run_eval.py                                   # Owner B (A adds fingerprint evals)
  agent/        llm.py prompts.py tts.py agent.py                        # Owner C
  api/          main.py routes.py ws.py session.py                       # Owner C
web/            index.html app.js styles.css                             # Owner C
```
Stay inside your module. If you need a change in another module or in `schemas.py`, stop and say so.

## Hard rules
- **From scratch (NumPy/SciPy only):** resampling call may use `scipy.signal.resample_poly`; pre-emphasis, framing, windowing, STFT magnitude, peak picking, hashing, matching, VAD, MFCC (mel filterbank + DCT), YIN pitch, energy/ZCR. `librosa` is allowed **only in tests** as a reference to check our output.
- **Never fake results.** No hardcoded metrics, thresholds copied from nowhere, or mock "accuracy" in the UI. All numbers shown come from `artifacts/results/*.json` produced by `run_eval`. If something doesn't work, say so.
- **No data in git.** `data/` and `artifacts/` are gitignored except `.gitkeep`. No real phone numbers, names, bank details, or real victims' audio anywhere. All scripts use fictional details.
- **Secrets** only via `.env`. Never log API keys or full transcripts at INFO.
- **Agent grounding:** the agent may only state what is in the `Evidence` object. It must say "likely/possibly", never accuse a named person, and always recommend verifying via the official bank/police number. No fabricated scam statistics.
- **Latency budgets:** fingerprint query on 10 s audio < 300 ms; STT on a 5 s segment < 2.5 s (CPU, `base` model); alert shown < 5 s after the scam phrase ends; agent reply starts < 3 s after the user stops speaking.
- **Keep it simple.** No microservices, queues, auth, or ORMs. Prefer readable 30-line functions with docstrings that state the math.

## Coding conventions
Type hints everywhere; Pydantic models for every cross-module payload (`schemas.py`); pure functions in DSP code (arrays in, arrays out, no globals); constants (frame length, hop, n_fft, thresholds) live in `config.py`; every DSP function has a docstring with the formula and a unit test; log with `logging` (no `print` outside scripts).

## Definition of done (per module)
1. Public interface matches `docs/02_ARCHITECTURE.md` / `docs/03_API_CONTRACT.md`.
2. Unit tests pass (`pytest -q`) incl. a reference/sanity test.
3. A short `docs/modules/<name>.md` note: what it does, parameters, the key formula, known limits (this becomes viva prep).
4. Evaluated on the dataset where applicable, numbers saved by `run_eval`.

## Workflow
One branch per module (`feat/fingerprint`, `feat/understanding`, `feat/agent-api`). Small commits. Implement interfaces and stubs first (Phase 0), then fill in. When a phase's acceptance criteria in `docs/07_PHASES.md` are met, stop and summarize what changed and what's left.

## Honest limitations (keep visible in UI + report)
Detects known and similar scams, not all scams. Live phone-call interception is not possible on iOS/stock Android; demo uses speakerphone/mic or uploaded audio. Training/eval data is mostly synthetic TTS plus a small set of consented human recordings.
