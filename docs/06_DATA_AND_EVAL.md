# 06 — Data & Evaluation

## Data layout (`data/` is gitignored)
```
data/
  scripts.json                  # 17 scripts: id,label(scam|legit),type,lang,text        (from generator zip)
  clean/*.wav                   # TTS clips 16 kHz mono; name: {script_id}__{voiceTag}__{rate}.wav   e.g. scam_kyc_otp_en__enIN_Neerja__normal.wav
  augmented/*.wav               # name: {clean_stem}__{aug}.wav ; aug ∈ tel,snr20,snr10,snr5,gsm,spd95,spd105
  manifest.csv                  # file,label,type,lang,script_id,voice,rate,source,augmentation
  manifest_augmented.csv        # same columns, augmentation filled
  human/*.wav + manifest_human.csv   # consented human recordings (source=human) — TARGET: ≥ 3 clips per scam type? (min 40 total)
  holdout/ scripts_holdout.json + clips   # NEW paraphrased scam scripts + new legit scripts (see below)
  noise/*.wav                   # optional MUSAN/DEMAND noise
  ravdess|cremad/               # optional, for pressure classifier
```

## Critical honesty rule: avoid leakage
- The scam-script **text library** (Layer 2) is built from `scripts.json` scam scripts. Testing Layer 2 on TTS of those same scripts is trivially easy — report it only as a *sanity* result.
- The honest Layer-2 test = **holdout set**: ≥ 6 *paraphrased* scam scripts (same scam types, different wording, different details) + ≥ 6 *new legit* scripts (including legit calls that mention OTP/account/KYC) + all **human** recordings. Hand-edit these so they are not near-copies. Generate audio with voices **not used** for the library, and record some by humans.
- Layer 1 (fingerprint) library = **one clip per scam script** (`enIN_Neerja`, `normal`, clean). Queries:
  - *Positive:* the 7 augmentations of each library clip.
  - *Negative-for-L1 (expected no fingerprint match):* other voices/rates of the same script (different audio!), all legit clips and their augmentations. This shows exactly what L1 can and cannot do; L2 must catch the rest.

## Splits
- **dev:** used to calibrate fingerprint threshold, red-flag weights, fusion weights.
- **test:** untouched until final run. Split **by script_id and voice**, not by file, so no near-duplicate leakage. Suggested: dev = 6 scam types + 4 legit scripts; test = remaining + holdout + human. Record the split in `artifacts/split.json`.

## Metrics (all produced by `python -m scamshield.eval.run_eval`)
**Layer 1 — fingerprint**
- Recall@1 (correct `script_id`) per augmentation: clean, tel, snr20, snr10, snr5, gsm, spd95, spd105.
- False match rate on negatives at the chosen threshold; ROC over the aligned-hash threshold.
- Query time vs audio length; index size.
- Expect (do not assume): strong on clean/tel/gsm/noise, degraded at snr5 and speed change. Report whatever it is.

**Layer 2 — script/red-flag**
- ROC-AUC, precision/recall/F1 at the operating threshold on holdout + human; per-language (EN vs Hinglish) breakdown; confusion by scam type.
- STT quality: WER vs ground-truth script text on clean vs tel/noisy; note Hinglish romanised text will have high WER.
- False positives on legit clips that contain "OTP/account/bank" — list them.

**Layer 3 — voice cues**
- Report feature distributions scam vs legit (box plots). If a pressure classifier exists: accuracy on held-out RAVDESS actors. State clearly if it adds no value on TTS data.

**Fusion & system**
- Confusion matrix and FPR on legit; level distribution; **detection latency** = seconds from first scam-phrase end to first `high`/`scam` alert (live-simulated runs).
- **Ablation:** fp-only, script-only, flags-only, voice-only, full — same test set.
- End-to-end timings per stage (p50/p95).

## Output files
`artifacts/results/summary.json` (headline numbers + n + split name + git commit + timestamp), `layer1_by_aug.csv`, `layer2_holdout.csv`, `fusion_confusion.csv`, `ablation.csv`, `timings.csv`, PNGs (`roc_layer2.png`, `recall_by_aug.png`, `offset_hist_example.png`). `GET /api/results` serves `summary.json`.

## Report table templates (fill from files only)
| Augmentation | n | Recall@1 | Avg aligned hashes |
|---|---|---|---|
| clean | | | |
| tel | | | |
| gsm | | | |
| snr20/10/5 | | | |
| spd95/105 | | | |

| Config | Precision | Recall | FPR (legit) | Latency |
|---|---|---|---|---|
| fp only / script only / flags only / voice only / full | | | | |

## Data collection tasks for humans (team)
1. Record ≥ 40 human clips total (each member ≥ 14): mix of scam scripts (own wording encouraged) and legit scripts, on a phone, some on speakerphone, some noisy. Fill `manifest_human.csv`: file,label,type,lang,script_id,voice(speaker initials),rate=natural,source=human,augmentation=none.
2. Write the holdout scripts (≥ 6 scam paraphrases, ≥ 6 legit incl. tricky ones).
3. Everything staged and consented; fictional details only.

## Ethics
No real calls, numbers, names, bank details. Human speakers consent to use in the project. Delete session audio after analysis in live mode unless "save sample" is explicitly toggled for demos.
