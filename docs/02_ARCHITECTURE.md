# 02 — Architecture

## System diagram
```
 Browser (web/)                                   Server (FastAPI, src/scamshield)
 ┌────────────────────────┐   WS /ws/live        ┌───────────────────────────────────────────────────────────┐
 │ Mic → AudioWorklet     │ ── PCM16 16k mono ─► │ session.py  (rolling buffer, per-session state)           │
 │ Upload file            │ ── POST /api/analyze►│   │                                                       │
 │ Push-to-talk (agent)   │ ── POST /agent/ask ─►│   ▼                                                       │
 │                        │                      │ audio/: io → preprocess → vad                             │
 │ Dashboard:             │ ◄─ JSON events ───── │   │                                                       │
 │  waveform/spectrogram  │                      │   ├─► fingerprint/: spectrogram→peaks→hashes→index→matcher│ L1
 │  risk gauge/timeline   │                      │   ├─► understanding/stt → script_match + redflags         │ L2
 │  transcript + flags    │                      │   └─► dsp/features (MFCC,pitch,energy) → voice_cues       │ L3
 │  evidence cards        │                      │            │                                              │
 │  agent chat + speaker  │ ◄─ audio (mp3/wav) ─ │            ▼                                              │
 └────────────────────────┘                      │        fusion/risk → RiskState + reasons                  │
                                                 │            │                                              │
                                                 │            ▼                                              │
                                                 │ agent/: prompts + llm (grounded on Evidence) → tts        │
                                                 └───────────────────────────────────────────────────────────┘
 Offline artifacts (artifacts/): fingerprint_index.pkl, script_embeddings.npy, scripts_meta.json, fusion_config.json, results/*.json
```

## Module interfaces (stable contracts)
All payload types are Pydantic models in `schemas.py` (see `docs/03_API_CONTRACT.md`).

### audio (Owner A)
```python
load_audio(path|bytes) -> tuple[np.ndarray, int]            # float32 mono, original sr
preprocess(x, sr, *, target_sr=16000) -> np.ndarray          # resample, DC remove, peak-normalise
pre_emphasis(x, alpha=0.97) -> np.ndarray                    # y[n] = x[n] - a x[n-1]
frame_signal(x, frame_len, hop) -> np.ndarray                # (n_frames, frame_len)
vad(x, sr) -> list[tuple[float,float]]                       # speech segments in seconds (energy + ZCR, adaptive threshold)
```
### fingerprint (Owner A)
```python
spectrogram(x, sr) -> (S_db: np.ndarray[freq,time], freqs, times)       # own STFT, Hann window
find_peaks(S_db) -> np.ndarray[(n,2)]  # (freq_bin, time_frame), local maxima in 300–3400 Hz band
make_hashes(peaks) -> list[tuple[int, int]]                             # (hash32, anchor_time_frame)
class FingerprintIndex: add(clip_id, hashes); save(path); load(path); query(hashes) -> dict[clip_id, list[offset]]
match(x, sr, index, meta) -> list[FingerprintMatch]                      # offset-histogram scoring, sorted
```
### dsp + understanding (Owner B)
```python
mfcc(x, sr, n_mfcc=13) -> np.ndarray ; yin_pitch(x, sr) -> (f0, voiced_prob) ; short_time_energy(...) ; zcr(...)
transcribe(x, sr) -> list[TranscriptSegment]                  # faster-whisper, segment-level, language auto
script_match(transcript_text) -> list[ScriptMatch]            # cosine sim vs scripts (sentence-window chunks)
red_flags(transcript_segments) -> list[RedFlag]               # lexicon + negation/safe-context handling
voice_cues(x, sr, vad_segments) -> VoiceCues
fuse(components: RiskComponents, fp_hit: bool) -> RiskState   # fusion/risk.py
```
### agent + api (Owner C)
```python
Agent.warn(evidence: Evidence) -> SpokenMessage                # short warning text + audio url (cached phrases)
Agent.answer(question: str, evidence: Evidence, history) -> SpokenMessage
tts(text, voice) -> bytes                                      # edge-tts; fall back to browser speechSynthesis flag
SessionManager: create(); push_audio(sid, pcm_bytes); snapshot(sid) -> Evidence
```

## Fingerprint design (starting parameters — tune with eval)
- sr 16 kHz; STFT window 512 (32 ms) Hann, hop 256 (16 ms); log-magnitude dB.
- **Band-limit peaks to 300–3400 Hz** so telephone-band queries match full-band library clips.
- Peak = local maximum in a 15×15 (freq×time) neighbourhood and > (local median + 6 dB) or top-percentile threshold; cap peaks/sec.
- Pairing: each anchor pairs with up to `fan_out=8` peaks in target zone Δt ∈ [0.1 s, 2.0 s].
- Hash = pack(f1_bin, f2_bin, Δt quantised to ~32 ms) into 32-bit int. Coarse Δt quantisation gives tolerance to ±5% speed change (expect degradation; report it).
- Index: dict[hash] → list[(clip_id, t_anchor)]. Query: for each query hash collect (clip_id, t_db − t_query); the **peak of the offset histogram** per clip is the score (count of aligned hashes). Decision threshold calibrated on negatives (legit + unrelated clips) — pick a value giving ~0 false matches on dev set.
- Partial/live queries: query the last 8–12 s window; offset histogram still works on partial audio.

## Script-matching design
- STT: faster-whisper `base` (CPU, int8). Run on VAD-finalised segments, not raw chunks.
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (handles mixed language better than English-only). Library = sentence-window chunks (2–3 sentences) of each scam script; query = sliding windows of transcript. Score = max cosine similarity per scam type.
- Red flags: weighted lexicon in 7 categories (urgency, authority, secrecy, credential_request, payment, threat, remote_access) in English + romanised Hindi. **Negation / safe-context**: phrases like "do not share", "never ask for", "if this was not you, call the number on your card" cancel or reduce credential_request/payment flags (legit bank alerts contain these words).

## Voice cues
Pitch (YIN) mean/std, speech rate (words/s from STT timing), pause ratio (VAD), energy variability, optional `pressure_prob` from a small scikit-learn classifier (MFCC stats) trained on RAVDESS/CREMA-D (stretch). Weight is deliberately low; TTS data has little real emotion — report this honestly.

## Fusion
Noisy-OR with fixed, interpretable weights, then rule overrides:
```
score = 100 * (1 - Π_i (1 - w_i * s_i))    i ∈ {fingerprint, script, redflags, voice}
initial w = {fp: 0.95, script: 0.80, redflags: 0.60, voice: 0.25}
override: fingerprint match above calibrated threshold -> score >= 90
safe-context: strong negation cues cap redflags contribution
levels: <30 safe | 30–60 suspicious | 60–85 high | >=85 scam
```
Weights/thresholds are written to `artifacts/fusion_config.json` after tuning on the **dev** split; test split is touched once for final numbers. `top_reasons` = ordered human-readable strings from the strongest components (used verbatim by UI and agent).

## Live (streaming) design
Client sends 100 ms PCM16 frames. Server keeps a rolling 12 s buffer per session. Every ~2 s (and on each VAD-finalised segment): (1) fingerprint query on the window, (2) STT on new finalised speech, (3) update script/red-flag scores over the accumulated transcript, (4) recompute voice cues, (5) fuse → push `risk` event; on level increase push `alert` event and trigger agent warning. "Simulated live" mode streams a file at real-time speed through the same WS path (demo fallback + integration tests).

## Agent design
- Inputs: `Evidence` (risk, level, top_reasons, fingerprint match, script match, red-flag phrases, transcript tail, voice cues) + user question + short history.
- System prompt rules: ground every claim in Evidence; hedge ("likely"); never invent stats; never name/accuse individuals; give concrete steps (hang up, call the official number from the card/website, don't share OTP/PIN, report to 1930 / cybercrime.gov.in); keep replies ≤ 3 sentences for speech.
- Warnings: pre-generate and cache audio for fixed messages per scam type at startup → zero-latency alert. Free-form answers use live TTS; browser `speechSynthesis` is the offline fallback.
- Interaction: **push-to-talk** (not always-on barge-in) — avoids speaker echo; document barge-in as future work.

## Key decisions (ADR summary)
| Decision | Why | Alternative rejected |
|---|---|---|
| Static JS frontend, no build | zero tooling risk before a demo | React/Next.js (more moving parts) |
| Push-to-talk | echo-safe, simple, reliable | full-duplex barge-in |
| Two detection layers | fingerprint only matches identical audio; script layer handles other voices | single ML classifier (not explainable, no data) |
| Own DSP for fingerprint/VAD/MFCC/pitch | viva + rubric | librosa/pyannote one-liners |
| Noisy-OR fusion | explainable, tunable, no training data needed | learned classifier on ~100 samples (overfit) |
| Pickle/JSON artifacts | simple | DB |

## Dependencies
See `requirements.txt`. `ffmpeg` must be on PATH (used by scripts/augmentation and for decoding browser uploads).
