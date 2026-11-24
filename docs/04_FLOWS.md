# 04 — Flows

## F1. Offline build (once, and after any data/library change)
```mermaid
flowchart TD
  A[data/scripts.json + data/clean/*.wav + manifest.csv] --> B[check_data.py: validate files/labels]
  B --> C[build_library.py]
  C --> D1[For each scam script: library clip = enIN_Neerja, normal rate, clean]
  D1 --> E1[preprocess → spectrogram → peaks → hashes → FingerprintIndex.pkl]
  C --> D2[Chunk scam script texts → embed → script_embeddings.npy + scripts_meta.json]
  C --> D3[Optional: train pressure classifier on RAVDESS → artifacts/pressure_model.pkl]
  E1 --> F[run_eval on dev split → tune thresholds + fusion weights → fusion_config.json]
  D2 --> F
  F --> G[run_eval on test split once → results/summary.json]
```

## F2. Upload analysis (MVP path)
```mermaid
sequenceDiagram
  participant U as User/Browser
  participant API as FastAPI /api/analyze
  participant A as audio+fingerprint (A)
  participant B as understanding+dsp+fusion (B)
  U->>API: POST file
  API->>A: decode → preprocess → VAD
  par Layer 1
    A->>A: spectrogram → peaks → hashes → match(index)
  and Layer 2
    API->>B: STT on VAD segments
    B->>B: script_match + red_flags (negation aware)
  and Layer 3
    B->>B: MFCC/pitch/energy → VoiceCues
  end
  B->>B: fuse over time windows → risk_timeline, final
  API-->>U: AnalysisResult (+ Viz, timings)
  U->>U: render waveform/spectrogram+overlay, gauge, transcript, evidence
  API-->>U: (client calls /api/agent/ask for warning text+audio)
```

## F3. Live call (streaming)
```mermaid
sequenceDiagram
  participant M as Mic/AudioWorklet
  participant WS as /ws/live
  participant S as Session (rolling 12 s buffer)
  participant P as Pipeline (A+B)
  participant AG as Agent (C)
  M->>WS: PCM16 16k frames (100 ms)
  WS->>S: append
  loop every ~2 s or on VAD segment end
    S->>P: window audio
    P->>P: fingerprint(window) ; STT(new segments) ; update script/flags ; voice cues ; fuse
    P-->>WS: viz / transcript / fp_match / risk
    alt level increased
      P->>AG: warn(Evidence)
      AG-->>WS: alert(message, audio_url)
    end
  end
  M->>WS: {"type":"stop"}
  WS->>P: finalize full-call analysis
  WS-->>M: final AnalysisResult
```

## F4. Voice Q&A (push-to-talk)
```mermaid
sequenceDiagram
  participant U as User
  participant UI as Browser
  participant API as /api/agent/ask_audio
  participant ST as STT
  participant AG as Agent
  U->>UI: hold button, speak ("Why do you think it's a scam?")
  UI->>API: audio + session_id
  API->>ST: transcribe question
  API->>AG: answer(question, Evidence(session), history)
  AG->>AG: LLM grounded on Evidence → ≤3 sentences
  AG->>AG: TTS (or fallback flag)
  API-->>UI: {transcript, reply{text, audio_url}}
  UI->>U: plays reply, shows chat bubble
```

## F5. Evaluation
```mermaid
flowchart LR
  M[manifest_augmented.csv + human manifest + holdout] --> S[split: dev / test by script & voice]
  S --> L1[Layer 1: query degraded clips vs index → recall@1 by augmentation; FMR on legit + other voices]
  S --> L2[Layer 2: STT → script sim + flags → ROC/AUC, P/R on holdout paraphrases + human clips]
  S --> FU[Fusion: confusion matrix, FPR on legit, detection latency, ablations]
  L1 --> R[artifacts/results/*]
  L2 --> R
  FU --> R
  R --> UI[Results tab + report tables]
```

## Risk level state machine (live)
`safe → suspicious → high → scam`. Level can only go up within a call unless score stays below the lower threshold for ≥ 10 s (then step down one level). Alerts fire on upward transitions, debounced ≥ 5 s. Fingerprint hit above threshold jumps straight to `scam`.
