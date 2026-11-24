# ScamShield Voice

Listens to a phone call (uploaded recording or live microphone), decides whether it is likely a scam, **shows the evidence**, and **talks to the user**: a spoken warning plus push-to-talk questions ("Why do you think it's a scam?", "What should I do?") answered only from the evidence.

> Course project: MSc AIML, *Speech Processing and Recognition* (MAI522-5), CIA III. Staged and synthetic data only; all names and numbers in the scripts are fictional.

![Architecture](docs/architecture.svg)

## How it works
| Layer | What it does | Catches |
|---|---|---|
| **L1 Acoustic fingerprint** (from scratch) | own STFT → spectral peaks in 300–3400 Hz → anchor/target hashes → inverted index → **offset-histogram spike** | replayed / robocall audio, even through noise, telephone band, GSM codec, ±5% speed |
| **L2 Script matching** | Whisper transcript → multilingual sentence embeddings vs known scam-script chunks + weighted red-flag lexicon (EN + romanised Hindi) with negation / safe-context handling | new voices and live callers using known scam scripts |
| **L3 Voice cues** (from scratch) | YIN pitch, energy, ZCR, MFCC, pace, pauses | shown as context; not scored yet (no calibration data) |
| **Fusion** | noisy-OR `100·(1 − Π(1 − wᵢsᵢ))`, fingerprint override, safe-context cap → `safe / suspicious / high / scam` + plain-language reasons | |
| **Voice agent** | Gemini or Anthropic, grounded only on the `Evidence` object, number/statistic guard, offline template fallback; cached spoken warnings | |

Every DSP step (pre-emphasis, framing, VAD, STFT, peak picking, hashing, matching, MFCC, YIN) is implemented with NumPy/SciPy and unit-tested against `scipy`/`librosa` references. See `docs/modules/` for per-module notes with formulas (viva prep).

## Results (test split, run once; thresholds calibrated on dev)
Full tables, dev vs test, and caveats: **[docs/RESULTS.md](docs/RESULTS.md)** (generated from `run_eval` output, never typed by hand).

| | Test split |
|---|---|
| Fingerprint recall@1, every augmentation (clean, tel, GSM, SNR 20/10/5 dB, speed 0.95/1.05) | 1.0 (n = 5 each) |
| Fingerprint false matches (legit audio or wrong script) | 0 / 173 |
| Fusion precision / recall / FPR on legit | 0.959 / 0.875 / 0.062 (256 clips) |
| English clips: script-similarity AUC | 1.0 |
| Hinglish clips: script-similarity AUC | **0.745** (weak spot; see below) |
| Live: scam calls alerted / legit calls alerted | 0.867 / 0.0 |
| Live: fingerprint replay alert | ~2 s into the call |
| Live: words-only latency after first red-flag phrase (p50) | 8.2 s |
| Upload analysis per clip (p50 / p95) | 1.27 s / 2.28 s |

**Honest limitations.** Detects *known and similar* scams, not all scams. Layer 2 numbers are a sanity check (library scripts read by other voices) until holdout paraphrases and human recordings are added. Hinglish read with a British accent defeats Whisper: on the test split 19 of 20 missed scams and all 6 false alarms are such clips. Live words-only detection needs several sentences before alerting. Live interception of phone calls is not possible on iOS/stock Android; the demo uses speakerphone/mic or uploaded audio. Not a substitute for calling your bank.

## Quick start
Requires Python 3.11 and **ffmpeg** on PATH.
```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # optional: GEMINI_API_KEY (or ANTHROPIC_API_KEY) for the LLM agent
```
Generate the dataset into `data/` (synthetic TTS + phone-line augmentation; needs internet once):
```bash
cd data && python ../scripts/gen_tts.py && python ../scripts/augment.py && cd ..
python scripts/check_data.py           # expects 136 clean + 952 augmented clips, 16 kHz mono
```
Build the library, calibrate on dev, start:
```bash
python scripts/build_library.py
PYTHONPATH=src python -m scamshield.eval.run_eval --split dev     # Windows PowerShell: $env:PYTHONPATH="src"
python scripts/run_demo.py             # checks setup, starts the server, opens http://localhost:8000
python scripts/run_demo.py --offline   # agent uses offline grounded templates (no LLM calls)
```

## Using it
- **Upload** a recording (wav/mp3/m4a/webm) or drop it anywhere on the page.
- **Play sample**: curated clips; with *as live call* ticked, the sample streams through the live WebSocket path at real-time speed (reliable demo fallback).
- **● Live mic**: real-time analysis from the microphone (use headphones; the mic is muted while the agent speaks).
- **Agent**: hold 🎤 and ask, or type. Toggle 🔊 to mute.
- **Pipeline** tab: offset histogram, stage timings, fusion components. **Results** tab: `summary.json` rendered as tables and plots.

## Evaluate and test
```bash
PYTHONPATH=src python -m scamshield.eval.run_eval --split dev [--quick] [--no-live]   # calibrates + writes artifacts/results/
PYTHONPATH=src python -m scamshield.eval.run_eval --split test                       # ONCE, at the end
python scripts/results_tables.py --split test                                       # Markdown tables for the report
pytest -q                                                                           # data-dependent tests skip without data/
```

## Repo map
```
src/scamshield/
  config.py schemas.py                         shared constants + Pydantic contracts (docs/03)
  audio/        io, preprocess, vad            A
  fingerprint/  spectrogram, peaks, hashing, index, matcher, plots   A
  dsp/          features (MFCC, YIN, energy, ZCR)                    B
  understanding/ stt, script_match, redflags, voice_cues             B
  fusion/       risk                                                 B
  eval/         metrics, run_eval                                    B (+A for Layer 1)
  agent/        llm, prompts, tts, agent                             C
  api/          main, routes, pipeline, viz, session (live), ws      C
web/            index.html, app.js, styles.css, pcm-worklet.js       C
scripts/        gen_tts, augment, check_data, build_library, run_demo, results_tables
docs/           design (00-09), modules/ (per-module notes), RESULTS.md, DEMO.md, architecture.svg
data/, artifacts/   generated locally, gitignored
```

## Team
Owner A: audio + fingerprint · Owner B: features + understanding + fusion + eval · Owner C: agent + API + UI.

## Ethics and privacy
Staged, consented, fictional data only. Live audio is kept in memory for the session and never written to disk. The agent never names or accuses anyone, hedges ("likely"), and always points to the official number and 1930 / cybercrime.gov.in. False-positive risk is measured and reported.
