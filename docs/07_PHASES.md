# 07 — Phases, Tasks, Acceptance Criteria

Owners: **A** = audio + fingerprint · **B** = dsp features + understanding + fusion + eval · **C** = agent + API + web UI.
Time budget is relative (day 0 = start). **First confirm the real CIA III deadline with faculty** and compress using the cut list at the bottom.

## Phase 0 — Foundations (≈ 0.5 day, all three, sequential first)
- Scaffold repo per `docs/02_ARCHITECTURE.md`; `config.py`, `schemas.py` (from `docs/03`), stubs with docstrings for every public function, `tests/` skeleton.
- `scripts/check_data.py`: validates `data/` layout, manifests, sample rates (16 kHz mono), label counts; prints summary.
- Put dataset in `data/`; verify 136 clean + ~950 augmented clips load.
- **Accept:** `pytest -q` runs (stubs xfail), `check_data.py` passes, `uvicorn` boots and `/api/health` returns.

## Phase 1 — DSP core (≈ 1 day, A ∥ B ∥ C)
- **A:** `io`, `preprocess` (resample, DC removal, normalise, pre-emphasis), `frame_signal`, `vad` (short-time energy + ZCR, adaptive threshold), `fingerprint/spectrogram` (own STFT). Tests: STFT vs `scipy.signal.stft`/librosa shape & peak-frequency of a synthetic sine; VAD on a clip with known silence.
- **B:** `dsp/features`: MFCC (framing → Hamming → FFT → mel filterbank → log → DCT), `yin_pitch`, energy, ZCR. Tests vs librosa (correlation threshold, not exact equality); pitch on synthetic 150/220 Hz tones within 2%.
- **C:** FastAPI skeleton, `/api/health`, `/api/library`, static serving of `web/`; UI shell with mocked `AnalysisResult` JSON rendering all panels (waveform, spectrogram canvas, gauge, transcript, evidence).
- **Accept:** all unit tests green; mocked UI renders from a fixture JSON; each module has `docs/modules/*.md`.

## Phase 2 — Detection layers (≈ 1 day, parallel)
- **A:** peaks → hashes → `FingerprintIndex` → `match`; `scripts/build_library.py` (fingerprint part). Parameter sweep on dev; offset-histogram plot function.
- **B:** `stt.py` (faster-whisper, segment output), `script_match.py` (chunk embeddings, cosine), `redflags.py` (lexicon EN + romanised Hindi, negation/safe-context), `voice_cues.py`.
- **C:** `/api/analyze` pipeline orchestration with stage timings; `Viz` builder (downsampled waveform, spectrogram uint8 base64, VAD); `/api/tts` + warning phrase cache (edge-tts).
- **Accept:** clean + tel + gsm versions of library clips are matched to the correct script (A); a scam transcript from a different voice gets script similarity > legit transcripts (B); `/api/analyze` returns a real (non-mock) `AnalysisResult` (C).

## Phase 3 — Fusion + end-to-end upload (≈ 0.5–1 day) — **MVP GATE**
- **B:** `fusion/risk.py` (noisy-OR + overrides + levels + `top_reasons`), windowed risk timeline, `fusion_config.json`.
- **C:** wire fusion into `/api/analyze`; UI renders real results; alert banner; Play-sample dropdown.
- **A:** peaks overlay + match region data for the UI; offset-histogram data in `Viz`.
- **Accept (MVP):** In the browser, upload a noisy scam clip → risk ≥ 85, scam type correct, evidence cards populated, spectrogram with match overlay; upload a legit bank-alert clip → risk < 30–60 with negation shown. Commit + tag `mvp`.

## Phase 4 — Voice agent (≈ 1 day)
- **C:** `agent/` (prompts, LLM provider abstraction, grounding on `Evidence`), warnings (cached TTS per scam type), `/api/agent/ask` and `/ask_audio`, push-to-talk in UI (MediaRecorder), speaking state, `speechSynthesis` fallback.
- **B:** `run_eval` skeleton + Layer 2/fusion metrics on dev; produce first `summary.json`. **A:** Layer 1 eval by augmentation.
- **Accept:** after analysing a scam clip, say "Why do you think it's a scam?" → spoken reply cites only evidence, ≤ 3 sentences, reply starts < 3 s; ask an out-of-scope question → agent stays in scope.

## Phase 5 — Live mode (≈ 1 day)
- **C:** `session.py` rolling buffer, `/ws/live`, AudioWorklet capture (16 kHz PCM16), event streaming, **Simulated live** (file → WS at 1× speed).
- **A:** incremental VAD + windowed fingerprint query. **B:** incremental STT on finalised segments, windowed script/flags/voice, fusion state machine (levels up only, debounce).
- **Accept:** simulated-live scam sample: risk rises, alert fires < 5 s after key phrase, agent speaks warning; mic mode works on headphones in a quiet room; legit sample never exceeds "suspicious".

## Phase 6 — Evaluation, polish, report, rehearsal (≈ 1 day, all)
- Run full `run_eval` on **test** once; fill report tables; Results + Pipeline tabs complete; README final; architecture diagram image.
- Add human + holdout data results; document failures honestly.
- Demo rehearsal ×3 using `docs/09_VIVA_AND_DEMO.md`; record a fallback video; viva Q&A drills per module.
- **Accept:** `make demo`-style single command starts everything; fresh-clone setup works from README.

## Stretch (only if Phases 0–6 are done)
1. Pressure classifier on RAVDESS/CREMA-D with held-out-actor accuracy. 2. Synthetic-voice (spoof) detector on ASVspoof subset. 3. "Call-screen" mode where the agent answers and asks the caller to identify themselves. 4. Barge-in. 5. Hindi (Devanagari) scripts.

## Cut list if time is short (in this order)
1. Drop stretch items. 2. Live **mic** → keep Simulated-live only. 3. Pressure classifier/voice-cue ML (keep simple features). 4. Hinglish human recordings → English-only human set. 5. Results tab charts → tables. **Never cut:** fingerprint from scratch, script layer with eval numbers, agent warning + Q&A, the demo fallback.

## Daily checkpoints (suggested)
End of each day: all three merge to `main`, run `pytest -q` + `run_eval --quick`, 10-minute sync on interface changes. Don't let one module block: use fixtures/mocks until the real one lands.
