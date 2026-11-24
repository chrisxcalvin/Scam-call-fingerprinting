# 08 — Per-person prompts for Claude Code (paste after Phase 0 is merged)

Common preamble for all: *"Read CLAUDE.md and docs/ first. Work only in your modules on your branch. Follow the Definition of Done in CLAUDE.md. Do not modify schemas.py or other owners' modules — if you need a change, stop and tell me. Write docs/modules/<name>.md for each module (what it does, parameters, key formula, limits). Stop when the phase acceptance criteria are met and summarise."*

---
## Person A — audio + fingerprint (Phases 1–3, 5)
```
You are Owner A. Modules: src/scamshield/audio/*, src/scamshield/fingerprint/*, scripts/build_library.py (fingerprint part).
Phase 1: implement io.py, preprocess.py (resample via scipy resample_poly, DC removal, peak-normalise, pre_emphasis alpha=0.97, frame_signal), vad.py (short-time energy + zero-crossing rate with adaptive threshold from noise-floor estimate, hangover smoothing), fingerprint/spectrogram.py (own STFT: Hann window 512, hop 256, log-magnitude dB). Tests: sine peak frequency, VAD on constructed silence/speech, STFT shape + compare with scipy.signal.stft.
Phase 2: peaks.py (local maxima via scipy.ndimage.maximum_filter, 15x15 neighbourhood, restrict to 300–3400 Hz, adaptive amplitude threshold, cap peaks/sec), hashing.py (anchor–target pairing, fan_out=8, dt 0.1–2.0 s, pack f1,f2,dt-quantised into 32-bit int), index.py (FingerprintIndex add/save/load/query), matcher.py (offset-histogram scoring -> FingerprintMatch with calibrated score). build_library.py: index one clip per scam script (voice enIN_Neerja, rate normal) from data/clean.
Evaluate on data/augmented: recall@1 per augmentation + false match rate on legit and other-voice clips; sweep parameters on the dev split only; write a function that returns the offset histogram for plotting. Report honestly where it fails (noise 5 dB, speed change).
Phase 3: provide peaks overlay + match region + offset histogram for Viz. Phase 5: incremental VAD and windowed matching for the live path (last 8–12 s).
Never call librosa outside tests.
```

---
## Person B — features + understanding + fusion + eval (Phases 1–5)
```
You are Owner B. Modules: src/scamshield/dsp/features.py, understanding/*, fusion/risk.py, eval/*.
Phase 1: from-scratch mfcc (frame→Hamming→|FFT|²→mel filterbank (26 filters)→log→DCT, 13 coeffs), yin_pitch (difference function, cumulative mean normalised difference, absolute threshold 0.1, parabolic interpolation; 70–400 Hz), short_time_energy, zcr. Tests vs librosa (correlation) and synthetic tones.
Phase 2: stt.py (faster-whisper base int8 CPU, segment timestamps, lazy singleton), script_match.py (paraphrase-multilingual-MiniLM-L12-v2; chunk scam scripts into 2–3 sentence windows; cosine; max per scam type; build embeddings in build step), redflags.py (7-category weighted lexicon EN + romanised Hindi + negation/safe-context rules so 'do not share your OTP' and 'if this was not you call the number on your card' do not trigger credential/payment flags), voice_cues.py (pitch mean/std, speech rate from STT timing, pause ratio from VAD, energy std).
Phase 3: fusion/risk.py: noisy-OR over {fingerprint, script, redflags, voice} with weights from docs/02, fingerprint override >= 90, levels (<30/60/85), top_reasons strings, windowed risk timeline; fusion_config.json.
Phase 4–6: eval/metrics.py + run_eval.py per docs/06 (dev/test split by script_id+voice recorded in artifacts/split.json; Layer 2 on holdout+human; ablation; latency; summary.json). Tune ONLY on dev. Never hardcode a metric. If holdout data does not exist yet, tell me and generate the TODO list instead of faking it.
Phase 5: incremental STT on finalised VAD segments and the fusion state machine (level up-only, step-down after 10 s low, debounce).
```

---
## Person C — agent + API + web UI (Phases 0–6)
```
You are Owner C. Modules: src/scamshield/agent/*, api/*, web/*.
Phase 1: FastAPI skeleton, /api/health, /api/library, static serving; build web/ (index.html, app.js, styles.css) per docs/05_UI_SPEC.md rendering a mocked AnalysisResult fixture (tests/fixtures/analysis_example.json — create it from the schema). Canvas waveform + spectrogram (decode uint8 base64), risk gauge, timeline (Chart.js vendored in web/vendor), transcript with highlighted flags, evidence cards, tabs (Console/Pipeline/Results). No framework, no build step.
Phase 2: /api/analyze orchestration (call A and B modules via their interfaces, collect timings_ms), Viz builder, /api/tts cache + pre-generated warning phrases per scam type using edge-tts (voice en-IN-NeerjaNeural), speechSynthesis fallback flag.
Phase 3: wire fusion output to UI; alert banner; Play-sample dropdown.
Phase 4: agent/: LLM provider abstraction (Anthropic default via env), system prompt that grounds ONLY on Evidence, hedged language, ≤ 3 sentences, recommended actions (hang up, call official number, never share OTP/PIN, report at 1930 / cybercrime.gov.in), refuse out-of-scope. Endpoints /api/agent/ask and /ask_audio (reuse B's stt). Push-to-talk with MediaRecorder; states listening/thinking/speaking.
Phase 5: session.py rolling buffer, /ws/live per docs/03 (PCM16 16k frames, events), AudioWorklet capture, 'Simulated live' mode streaming a sample file at 1x through the same WS.
Handle all error states in docs/05. Keep the honesty strip visible. Never show metrics that are not read from /api/results.
```
