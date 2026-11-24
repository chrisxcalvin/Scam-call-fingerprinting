# Module B (so far): frame features

**File:** `dsp/features.py`

- **MFCC:** frame 25 ms / hop 10 ms -> Hamming -> `|FFT|^2` (512) -> 26 triangular mel filters (`mel = 2595 log10(1+f/700)`) -> `10 log10` -> DCT-II (ortho) -> first 13. Mel = perceptual pitch scale, log = loudness perception, DCT = decorrelation.
- **YIN pitch:** `d(τ)=Σ_{j<W}(x[j]-x[j+τ])²` via FFT cross-correlation; cumulative-mean-normalised `d'(τ)=d(τ)·τ/Σ_{k≤τ}d(k)`; first τ with `d' < 0.1` then descend to the local minimum; parabolic interpolation; `f0 = sr/τ`; `voiced_prob = 1 - d'(τ)`; unvoiced frames -> `nan`. Range 60-400 Hz, W = 512, hop 10 ms.
- **Energy / ZCR:** `E[m]=Σx²`; ZCR = fraction of sign changes.

**Tests:** MFCC coefficients 1-12 correlate > 0.99 with a librosa reference (same HTK mel, Hamming, no norm, no dB clamp). Pitch within 2 % on 150 Hz and 220 Hz harmonic tones; silence -> unvoiced.

**Known limits:** YIN on a 10 s clip takes ~0.3 s (Python loop per frame); unvoiced/noisy speech returns `nan`, so downstream code must use `nanmean`. TTS speech has little real prosody, so pitch cues are weak evidence.

## Phase 2: understanding + fusion
- **STT** (`stt.py`): faster-whisper `base`, CPU int8, beam 1, Whisper's own VAD off. ~1 s per 20 s clip. `STT_LANGUAGE=en` by default: with auto-detect, Hinglish was recognised as `hi` and written in Urdu/Devanagari script that the romanised lexicon cannot read; forcing `en` gives English text. Set `STT_LANGUAGE=auto` to compare.
- **Script match** (`script_match.py`): scam scripts -> 2-sentence sliding chunks (29 chunks) -> `paraphrase-multilingual-MiniLM-L12-v2` normalised embeddings; transcript -> 2-sentence windows; `S = Q L^T`; best cosine per script.
- **Red flags** (`redflags.py`): generic regex lexicon, 7 categories, EN + romanised Hindi. Negation word within 7 words before a credential/payment/remote phrase -> `negated`. Safe-context phrases ("if this was not you", "we will never ask...") halve credential/payment weights. `redflag_score = 1 - prod_c(1 - max_w_c)`.
- **Voice cues** (`voice_cues.py`): YIN pitch mean/std, pause ratio from VAD, energy std (dB) over speech, words/s from the transcript. **Not used in fusion yet (voice = 0)**: no calibration data.
- **Fusion** (`fusion/risk.py`): noisy-OR with doc weights; fingerprint hit -> >= 90; safe context caps red flags at 0.3; script similarity mapped linearly from [0.55, 0.85] to [0, 1] (**provisional**, from the Phase 2 probe).

**Phase 2 probe (enUS_Guy voice, i.e. not the fingerprint library voice; same script texts = sanity check only, leakage):** top script similarity scam 0.71-1.00 vs legit 0.34-0.61. Hinglish is weaker and overlaps: scam_hi (enGB_Ryan) down to 0.62, legit_friend_hi (enGB_Ryan) up to 0.71.
**Known failure:** `scam_lottery_hi__enGB_Ryan__snr5` scores 0. STT output is nonsense ("lucky-throwed mine park"), so no layer has evidence. The honest Layer 2 test needs the holdout paraphrases and human clips.

## Phase 3: fusion config
`artifacts/fusion_config.json` is written by `build_library.py` with the provisional defaults and `"source": "PROVISIONAL ..."`; `run_eval` must overwrite it with dev-calibrated values (Phase 4). `fusion/risk.load_config()` reads it. Reasons scale their wording to strength ("closely match" vs "partly resemble").

## Phase 4: evaluation (`eval/metrics.py`, `eval/run_eval.py`)
- **Metrics (NumPy, tested against sklearn):** P/R/F1, FPR, ROC-AUC = Mann-Whitney U / (n_pos n_neg) with ties = 1/2, ROC curve, WER = Levenshtein (S+D+I)/N.
- **Split (`run_eval.SPLIT`), by script AND voice:** dev = 5 scam scripts (digital_arrest_en, kyc_otp_hi, lottery_en, relative_hi, courier_en) + 4 legit, voices Prabhat + Guy; test = the other 5 scam + 3 legit scripts, voice Ryan. Library voice (Neerja) clips appear in a split only for that split's scripts. Test is untouched until Phase 6.
- **Runs the real pipeline** (`analyze_audio`) per clip; Whisper output is cached in `artifacts/cache/stt/` (first dev run 613 s, reruns ~200 s).
- **Calibration (dev only, written to `fusion_config.json`):**
  - fingerprint threshold = 2 x (max aligned hashes over *harmful* negatives: legit audio or a match to the wrong script) + 1 -> **25** (harmful max = 12).
  - script range: lo = max legit similarity (0.628 -> 0.633), hi = max(lo + 0.10, scam p25 0.786) -> **[0.633, 0.786]**.
  - fusion weights and level bounds: documented values, not fitted (too little data).
- **Design decision (flagged):** docs/06 counts "other rates of the same script" as Layer-1 negatives. The library voice at the fast rate slowed by 5% is ~6% from the original tempo, i.e. nearly the indexed audio, and always matched the *correct* script (up to 70 aligned hashes). Calibrating on those pushed the threshold to 71 and lost speed-changed positives, so they are reported separately as "same-script near-copies" (21.7% matched, n = 46) and the threshold uses harmful negatives only.
- **Eval bug found and fixed:** first run counted legit library-voice clips as Layer-1 positives (recall looked like 5/9 everywhere).

**Dev results (432 clips: 240 scam, 192 legit; all 8 versions). Optimistic: calibrated on the same split, and Layer 2 scripts are the library scripts.**
- Layer 1: recall@1 = 1.00 for all 8 augmentations (n = 5 each; spd95 min 62 aligned); harmful false-match rate 0 / 346; AUC 1.0; fingerprint stages p50 ~19 ms.
- Layer 2: AUC script similarity 0.996, red flags 0.996; scam-type top-1 1.00; WER (English) 0.064-0.079 except snr5 0.119.
- Fusion (flagged = high or scam): TP 233, FP 0, TN 192, FN 7 -> precision 1.00, recall 0.971, FPR(legit) 0.00. All 7 misses are noisy/fast **Hinglish** (kyc_otp_hi, relative_hi at snr5/snr10/tel). All 6 clean legit bank alerts stay safe with 1-2 negated flags.
- Ablation (recall / FPR): fp only 0.208 / 0; script only 0.863 / 0; flags only 0.0 / 0 (weight 0.6 caps a flags-only score at 60, so it can never reach "high" alone, even though its AUC is 0.996); voice only 0 (unscored); full 0.971 / 0.
- Latency p50 / p95: total 1227 / 1953 ms per clip (STT 822 / 1333 ms).

## Phase 5: detection latency (`run_eval.live_eval`)
Dev scam/legit clips (clean, tel, snr10; normal rate) are streamed through `session.simulate()` (same code as the WebSocket, simulated time). Latency = audio time of the first high/scam alert + compute time of that tick - end of the first transcript segment containing a non-negated red flag. Cached Whisper calls report the Whisper time of their first real run, so reruns don't understate latency (a bug found and fixed: the first rerun reported p95 8.3 s instead of 9.8 s).

**Dev results (45 scam, 36 legit streams; optimistic, see above):**
- Legit: **0 / 36 alerts**, all stayed `safe` for the whole call.
- Scam alerted: 93.3% (42/45); the 3 misses are Hinglish (relative_hi, kyc_otp_hi) that peak at `suspicious`.
- Fingerprint alerts (library-voice replays, n=15): all alerted, first alert at ~2 s into the call (before the first scam phrase ends).
- Words-only alerts (other voices, n=30): alerted 90%, latency p50 **1.6 s**, p95 10.3 s, within 5 s: 59%. Slow cases: Hinglish and noisy clips, where the script similarity only crosses the threshold after more sentences arrive.
- Tick compute p50 22 ms; max 3.0 s (Whisper on a long finalised chunk).
- **Acceptance "alert < 5 s after key phrase": met for fingerprint replays and the median words-only case, not for the slowest ~40% of words-only streams.**
