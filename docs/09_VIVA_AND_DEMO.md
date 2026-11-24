# 09 — Viva Prep & Demo Plan

## Concepts every member must be able to explain (with the formula)
**Common (all three):** why speech is processed in short frames (quasi-stationary ~20–30 ms); sampling & Nyquist (16 kHz → 8 kHz band; telephone band 300–3400 Hz); time vs frequency domain; the full pipeline order and why.

**A — preprocessing & fingerprint**
- Pre-emphasis `y[n] = x[n] − 0.97·x[n−1]` — boosts high frequencies, flattens spectral tilt.
- Framing/windowing: frame 32 ms, hop 16 ms, Hann window to reduce spectral leakage; STFT `X[k,m] = Σ x[n+mH]·w[n]·e^{−j2πkn/N}`; spectrogram = 20·log10|X|.
- VAD: short-time energy `E = Σ x²` and zero-crossing rate; speech = high energy, moderate ZCR; unvoiced fricatives = low energy, high ZCR; adaptive threshold from noise floor; hangover to avoid chopping.
- Fingerprint: why spectral **peaks** are robust to noise/codecs; anchor–target pairs → hash(f1,f2,Δt) makes matching time-shift invariant; **offset histogram** — true match produces a spike at constant Δ = t_db − t_query; why band-limiting to 300–3400 Hz; why speed change hurts (Δt and frequency shift); why different voices don't match (different peaks) → need Layer 2.
- Trade-offs: fan-out vs index size vs false matches; threshold from negatives.

**B — features, understanding, fusion**
- MFCC: frame → window → |FFT|² → mel filterbank (mel = 2595·log10(1+f/700)) → log → DCT → first 13 coeffs; why mel (perception), why log (loudness), why DCT (decorrelate).
- YIN: difference function `d(τ)=Σ(x[j]−x[j+τ])²`, cumulative mean normalised difference, threshold 0.1, parabolic interpolation; vs plain autocorrelation (fewer octave errors).
- STT (Whisper) at a high level: encoder-decoder transformer on log-mel input; WER definition `(S+D+I)/N`; why romanised Hinglish has high WER.
- Embeddings & cosine similarity `cos = a·b/(‖a‖‖b‖)`; why multilingual model; why lexicon + negation (legit bank alerts say "OTP").
- Fusion noisy-OR `1 − Π(1 − w_i s_i)`; why not a learned classifier (little data, need explainability); threshold calibration on dev; ablation meaning.
- Metrics: precision, recall, F1, ROC/AUC, FPR; why FPR on legit matters more than headline accuracy.

**C — system, agent, UI**
- Streaming architecture: PCM16, 16 kHz, 100 ms frames, rolling buffer, why windowed (latency vs context).
- Why push-to-talk (acoustic echo) and what barge-in would need (AEC + VAD on mic while TTS plays).
- Agent grounding: why constrain to Evidence (hallucination risk in safety-critical advice); latency budget; TTS caching.
- Privacy: no audio retention, consent, on-device future path.

## Likely examiner questions (and the honest short answers)
1. *"Isn't this just API calls?"* — Core DSP (preprocess, VAD, STFT, peaks, hashing, MFCC, YIN) is ours; APIs only for STT/TTS/LLM, and we evaluate the pipeline separately from them.
2. *"Does it work on real calls?"* — Prototype on staged/synthetic + human recordings with phone-line augmentation; real call interception isn't possible on iOS/stock Android; deployment path is OS/carrier level.
3. *"Why would a different voice not match your fingerprint?"* — Fingerprint matches the exact acoustic content; different speaker = different spectral peaks. That's why Layer 2 exists.
4. *"What is your false positive rate?"* — Read from `summary.json`; mention legit calls that mention OTP and how negation handling helps; state the n.
5. *"How did you pick thresholds/weights?"* — On dev split only; test touched once.
6. *"What are the limitations?"* — Known/similar scams only; synthetic-heavy data; speed change & 5 dB SNR degrade fingerprint; Hinglish WER high; pressure cue weak on TTS.
7. *"What about privacy/ethics?"* — Consent, fictional data, no retention, agent doesn't accuse individuals.

## Demo script (5 minutes)
**Setup (before):** headphones on the "victim" laptop; server running; venue-safe fallback video ready; `Play sample` list verified.
1. (30 s) Problem + one-line solution. Show architecture slide.
2. (60 s) **Upload mode:** drop a noisy telephone-band scam clip → gauge jumps to SCAM, spectrogram overlay shows matched region, **offset-histogram spike** in Pipeline tab ("this spike is the fingerprint match"). Evidence cards explain why.
3. (45 s) **Legit clip:** genuine bank alert that says "OTP" → stays Safe/Suspicious; point out the negated flag ("do not share" recognised as safe context).
4. (90 s) **Live (or simulated live):** teammate reads a *new* scam in their own words (not in the library) → fingerprint silent, script layer + red flags raise risk → agent speaks the warning.
5. (45 s) **Ask the agent:** hold-to-talk "Why do you think it's a scam?" then "What should I do?" → grounded, short spoken answers.
6. (30 s) Results tab: real numbers incl. a failure case; ethics + limitations strip; deployment path.

**Fallbacks:** mic fails → Simulated live; network fails → cached warnings + `speechSynthesis`, local Whisper, agent LLM replaced by a canned grounded template (`AGENT_OFFLINE=1`); everything fails → recorded demo video.

## Rehearsal checklist
- [ ] Fresh-clone setup works from README on a clean machine.
- [ ] Demo clips chosen (1 scam clean, 1 scam noisy/GSM, 1 legit-with-OTP, 1 new-wording live script).
- [ ] Each member can explain their module's formulas without notes.
- [ ] `run_eval` numbers match what's in the slides/report.
- [ ] Headphones tested in the demo room; volume & mic level set.
