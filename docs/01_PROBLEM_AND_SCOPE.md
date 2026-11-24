# 01 — Problem, Users, Scope

## Problem (plain language)
Scam calls (fake bank officers, "digital arrest", OTP/KYC fraud, lottery, cloned-voice "relative in trouble") succeed because the victim is scared, rushed, and alone. Nobody is available in that moment to say "this is a known scam, hang up".

## What ScamShield Voice does
Listens to a call (uploaded recording, or live audio via mic/speakerphone) and:
1. Matches the audio against a library of known scam recordings (**acoustic fingerprint**).
2. Transcribes it and compares the words to known scam scripts and red-flag patterns (**script matching**).
3. Measures how the caller sounds — pace, pitch, energy, pauses (**voice cues**).
4. Combines these into a live **risk score 0–100** with reasons.
5. **Speaks** to the user ("This matches a known digital-arrest script. Don't share your OTP.") and answers spoken questions ("Why?", "What should I do?").

## Target users
- Primary: a person receiving a suspicious call (demo persona: elderly relative / first-time bank customer).
- Secondary: banks/telecom fraud teams analysing reported call recordings (upload mode).

## In scope
- Upload mode (full analysis report) and live mode (mic stream, rolling analysis).
- Scam library: 10 scam scripts (EN + Hinglish) × several voices; legit set; augmented phone-line variants.
- From-scratch DSP: preprocessing, VAD, spectrogram, peak-picking, hashing, MFCC, YIN pitch.
- Voice agent: warnings (TTS) + push-to-talk Q&A, grounded in evidence.
- Evaluation harness producing the numbers shown in the report and UI.
- Dashboard with waveform, spectrogram with match overlay, risk timeline, transcript with highlighted flags, evidence cards, pipeline view.

## Out of scope (say so in viva)
Real telephony integration; training large models; detecting every scam type; speaker identification; multilingual ASR beyond Whisper defaults; mobile app.

## Rubric mapping (CIA III, 25 marks)
| Rubric item | Marks | Where it is satisfied |
|---|---|---|
| Execution of speech tasks | 10 | acquisition (mic/upload/dataset) → preprocess → features → fingerprint/STT → risk → live demo; Pipeline View tab shows each stage |
| Correctness of code/algorithm | 4 | unit tests vs librosa references; `run_eval` metrics (fingerprint recall by augmentation, script-match AUC, FPR on legit calls) |
| Concept clarity (viva) | 5 | every DSP step is ours; `docs/09_VIVA_AND_DEMO.md` has formulas + likely questions |
| Presentation & neatness | 3 | spectrogram/waveform/feature plots, architecture diagram, clean repo + README |
| Innovation & real-world use | 3 | scam-call protection with interactive voice agent; deployment path + ethics section |

## Success criteria
- Upload any provided scam clip (incl. noisy/telephone/GSM versions) → correct scam type flagged with evidence.
- Legit clips (including genuine bank alerts mentioning OTP/account) are **not** flagged above "suspicious".
- Live demo: staged scam call from a teammate raises risk within ~5 s of the key phrase; agent speaks the warning; user asks "why?" and gets a grounded answer.
- Report contains honest numbers from `run_eval`, including failures (e.g. fingerprint degrades under speed change).

## Real-world deployment path (for report/viva)
Realistic routes: built into OS dialer (on-device), carrier-network robocall fingerprinting, speakerphone mic capture, desktop/system-audio capture, or upload/forward-a-recording. iOS and stock Android block third-party access to call audio; recording/analysing calls requires consent under privacy law. Fingerprint layer is strongest at carrier scale (robocalls reuse identical audio); script layer is the demo-friendly live layer.

## Ethics & privacy
Staged, consented audio only; fictional names/numbers; no storage of user audio by default (session audio deleted on close); agent never accuses individuals; false-positive risk acknowledged and measured; accessibility (voice interface) is a design goal.
