# Holdout set (docs/06: the honest Layer 2 test)

`scripts/holdout/scripts_holdout.json` (tracked) → `python scripts/gen_holdout.py` → `data/holdout/` (gitignored) → `python -m scamshield.eval.run_eval --split holdout`.

## What is in it
| | Count | Notes |
|---|---|---|
| Scam scripts | 10 | all 6 scam types: digital arrest (EN+HI), KYC/OTP (EN+HI), lottery (EN+HI), relative in trouble (EN+HI), fake courier (EN), tech support (EN) |
| Legit scripts | 8 | new; deliberately tricky (below) |
| Voices | 4 | `en-US-AriaNeural`, `en-GB-SoniaNeural`, `en-IN-NeerjaExpressiveNeural`, `en-AU-WilliamMultilingualNeural`: none appears in the library, dev or test sets |
| Versions per clip | 4 | clean, telephone band, white noise at 10 dB SNR, GSM codec |
| Clips | 288 | 72 clean + 216 augmented |

**Scams are paraphrases, not copies.** Same scam type, different scenario and details (Narcotics Control + airport seizure instead of a parcel with illegal items; a card-deactivation/PAN story instead of KYC expiry; a car + GST charge instead of a cash prize + processing fee; a fight near college instead of an accident; customs duty via SMS link instead of an incomplete address; a "hacker" story instead of virus alerts). Hinglish scripts were rewritten a second time with new scenarios after the first draft was too close to the library wording.

**Tricky legit scripts** (they contain words a naive detector flags): a genuine KYC reminder ("the bank will never ask for your OTP"), a delivery partner asking for a delivery code, an automated bank fraud-check call, a hospital nurse reporting a child's minor injury, a traffic-police e-challan notice ("do not pay anyone who calls you"), an electricity bill reminder ("ignore any message that threatens to cut your power"), a Hinglish branch call about a new debit card, and a Hinglish family call about a late train.

## Why this is a fairer test than dev/test
- Different wording and details, so the script-embedding library cannot match it by memory.
- Voices unseen anywhere else.
- Legit scripts include words from every red-flag category.
- Thresholds, similarity range and weights are the dev-calibrated ones, **unchanged**; `run_eval --split holdout` refuses to run without them and never rewrites `fusion_config.json`.
- Every clip is a Layer 1 negative (none is the indexed audio), so every fingerprint match counts as a false match.

## Known weaknesses of this holdout (state these in the viva)
1. **Author bias (expected optimistic, observed the opposite).** The scripts were written by the same assistant that wrote the red-flag lexicon. We expected red-flag numbers to be flattering; in practice the reworded scams *avoid* the lexicon's words ("tell me that code", "download a small support tool") and score no flags, while tricky legit scripts (traffic police, KYC reminder) do trigger them (red-flag AUC 0.55). Independent human-written scripts and recordings (still to do) would remove any remaining bias.
2. **Hinglish distance cannot be measured with the embedder.** The multilingual MiniLM places romanised Hinglish in one tight cluster: even a message about a train running late scores 0.84 cosine against a scam script, so "similarity to the library" does not show how different the Hinglish wording is. For English the holdout scams sit at 0.57–0.66 cosine to the nearest library script (a real paraphrase distance).
3. **Still synthetic TTS.** No human speech, no spontaneous wording, no overlapping speech.
4. **Small.** 10 scam + 8 legit scripts; clips of one script are strongly correlated, so confidence intervals are wide.

## Result (first and only run; nothing was tuned afterwards)
`python -m scamshield.eval.run_eval --split holdout`, dev-calibrated thresholds, 288 clips (160 scam, 128 legit):
- Scams caught **18 / 160 (11%)**; false alarms 0 / 128; precision 1.00 (of the few flagged).
- Fingerprint: 0 false matches in 288 (none of this audio is in the library, as intended).
- Script-similarity AUC 0.77 overall: English 0.86, Hinglish 0.63. Red-flag AUC 0.55. Scam type identified 80% of the time.
- Mean similarity to the library: reworded scams 0.62 (55% below the 0.633 match floor), legit 0.54.
- Per script (4 voices each): the reworded digital-arrest script reaches *high* for all 4; fake-courier, lottery (EN) and digital-arrest (HI) reach *suspicious*; KYC/OTP, relative-in-trouble, tech-support (EN) and lottery (HI) stay *safe*.
- **Diagnosis:** a small, near-verbatim script library and a keyword lexicon do not generalise to reworded scams. Raising recall by lowering thresholds would trade against false alarms (legit mean similarity is 0.54, max 0.64). Real fixes need more diverse scam data (many paraphrases per type), a stronger semantic model or classifier, and then a *new* untouched test set, because this holdout must not be tuned on.
