# 05 — UI Spec (static frontend: `web/index.html`, `app.js`, `styles.css`)

Goal: a **call console** that a judge understands in 5 seconds, and that also exposes the DSP for the viva. No framework, no build step. Canvas for waveform/spectrogram, Chart.js (CDN is fine; also vendor a copy in `web/vendor/` for offline demo) for risk timeline. Dark theme, high contrast, large type; works at 1366×768 projector size.

## Layout (single page, three tabs: **Console · Pipeline · Results**)
```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ ScamShield Voice        [● Live mic] [⬆ Upload]  [▶ Play sample ▾]       status: ready ●  │
├──────────────────────────────┬───────────────────────────────┬───────────────────────────┤
│ WAVEFORM  (canvas, VAD bands)│          RISK GAUGE            │  TRANSCRIPT               │
│ SPECTROGRAM (canvas)         │            78 / 100            │  [t=3.2] "your Aadhaar…"  │
│  + fingerprint peaks overlay │         ▲ HIGH  (colour)       │   ⟨authority⟩ ⟨threat⟩    │
│  + matched-region highlight  │  top reasons:                  │  [t=9.1] "do not tell…"   │
│                              │   • Matches known digital-     │   ⟨secrecy⟩               │
│                              │     arrest script (0.87)       │   (flags highlighted;     │
│                              │   • Asks to keep call secret   │    negated ones struck)  │
│                              │   • Urgent, fast pace          │                           │
├──────────────────────────────┴───────────────┬───────────────┴───────────────────────────┤
│ RISK TIMELINE (line, coloured bands, alert pins)│ EVIDENCE CARDS                          │
│                                              │ [Fingerprint] match: digital_arrest_en     │
│                                              │   aligned 142 / 610 hashes, offset 3.4 s    │
│                                              │ [Script] sim 0.87 → "digital_arrest"        │
│                                              │ [Voice] pitch 212±41 Hz · 3.4 w/s · pause 12%│
├──────────────────────────────────────────────┴───────────────────────────────────────────┤
│ AGENT  🔊 [Speak: on]  chat bubbles…            [🎤 Hold to ask]  or type: [__________][↵] │
│ "Warning: this call matches a known digital-arrest scam. Don't share OTP. Hang up and…"   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Tab: Pipeline (for viva/demo)
Vertical stage list for the current clip; each stage shows a small before/after plot and its timing: **Raw → Resample/normalise → Pre-emphasis → VAD segments → STFT spectrogram → Peaks → Hashes (count) → Index lookup (offset histogram plot) → STT → Script similarity bars → Voice features (pitch contour, energy) → Fusion (component bars → score)**. The offset-histogram plot (spike at the true offset vs flat background) is the single most convincing visual for the fingerprint — make it prominent.

### Tab: Results
Renders `GET /api/results`: fingerprint recall by augmentation (bar), script-match ROC (line), confusion matrix (table), ablation (bar), latency table. Show "n = …" counts and the data split used. If a metric is missing, show "not run yet" — never a placeholder number.

## Components & behaviour
- **Source controls:** Live mic (permission prompt; if denied → inline error + suggest Upload), Upload (drag-drop; accept wav/mp3/m4a/webm; show progress), **Play sample** dropdown listing curated clips from `/api/library` samples (one scam, one noisy scam, one legit) for reliable demos.
- **Risk gauge:** semicircle, colour by level: safe `#2ecc71`, suspicious `#f1c40f`, high `#e67e22`, scam `#e74c3c`. Animate score changes (≤ 300 ms). Level text + icon (not colour alone — accessibility).
- **Transcript:** appended live; flagged phrases are `<mark>` with category tooltip; negated flags rendered struck-through grey with tooltip "safe context".
- **Evidence cards:** only show cards with data; each has an info icon explaining the method in one sentence.
- **Alert banner:** on `alert` event, full-width banner slides in + plays the audio (`audio_url`) or uses `speechSynthesis` if null. Banner text = `message`. Dismissible.
- **Agent panel:** chat bubbles (user/agent), **Hold-to-ask** push-to-talk (MediaRecorder → `/api/agent/ask_audio`), text fallback input, `Speak` toggle (mute replies). Show "listening…", "thinking…", "speaking…" states.
- **Honesty strip (always visible, small):** "Detects known & similar scams. Demo data is synthetic/staged. Not a substitute for calling your bank."

## States & errors
idle · requesting-mic · recording · analyzing (skeleton loaders) · result · alert · error (mic denied, upload too large, server down → "Backend unreachable" with retry) · offline-TTS fallback.
All errors show human text + a next step; never raw stack traces.

## Data → UI mapping
| UI element | Source |
|---|---|
| waveform/VAD | `Viz.waveform`, `Viz.vad_segments` |
| spectrogram | decode `Viz.spectrogram_b64` (uint8 matrix `spec_shape`) → colour-map on canvas (inferno-like) |
| peak overlay / match highlight | `Viz.fp_peaks`, `FingerprintMatch.query_span_s` |
| gauge/reasons | `RiskState` |
| timeline | `AnalysisResult.risk_timeline` or streamed `risk` events |
| transcript/flags | `TranscriptSegment.flags` |
| evidence | `fingerprint`, `script`, `voice` |
| pipeline timings | `timings_ms`, `Viz.stages` |

## Accessibility & polish
Keyboard focus order; ARIA live region for alerts; min 16 px text, 4.5:1 contrast; works with captions on (transcript is always visible); responsive down to 1024 px. Keep file sizes small; no external fonts required.

## Demo-critical requirement
A **"Simulated live"** option streams a chosen sample file through the real WebSocket at real-time speed — identical UI path as mic — so the demo works if the venue mic/noise is bad.
