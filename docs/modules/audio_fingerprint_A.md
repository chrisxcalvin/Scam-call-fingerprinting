# Module A (so far): audio I/O, preprocessing, VAD, STFT

**Files:** `audio/io.py`, `audio/preprocess.py`, `audio/vad.py`, `fingerprint/spectrogram.py`

| Function | What / formula | Key parameters |
|---|---|---|
| `load_audio` | libsndfile, ffmpeg fallback for mp3/m4a/webm; stereo -> mean | float32 mono, native sr |
| `preprocess` | `resample_poly` (allowed), DC removal `x - mean(x)`, peak-normalise to 0.99 | target 16 kHz |
| `pre_emphasis` | `y[n] = x[n] - 0.97 x[n-1]` (high-pass, flattens spectral tilt) | alpha 0.97 |
| `frame_signal` | `n_frames = 1 + ceil((N - L)/H)`, zero-padded tail | via index matrix |
| `vad` | frame energy `E = sum x^2` (dB) + ZCR; floor = 10th pct, speech level = 95th pct; threshold `floor + 0.3 (top - floor)`; low-energy/high-ZCR rule for fricatives; 200 ms hangover bridges gaps; drop < 100 ms | 25 ms frame, 10 ms hop |
| `spectrogram` | `X[k,m] = sum_n x[n+mH] w[n] e^{-j2πkn/N}`, Hann `w`, `S_db = 20 log10(|X|+eps)` | N=512 (32 ms), H=256 (16 ms) |

**Tests:** STFT equals `scipy.signal.stft` to 1e-6 dB (scaling 1/sum(w) accounted for); sine peaks at the right bin; VAD finds a known 1 s segment within 0.1 s and returns nothing on silence.

**Known limits:** VAD is energy-based, so loud steady noise can be treated as speech and very quiet speech is missed; it needs >= 10 dB range between noise floor and speech. Real-clip check: 26.7 s TTS clip -> 4 segments in 19 ms.

## Phase 2: fingerprint (`peaks.py`, `hashing.py`, `index.py`, `matcher.py`, `plots.py`)
- **Peaks:** inside 300-3400 Hz only, local max in a 15x15 (freq x time) neighbourhood (`scipy.ndimage.maximum_filter`), > band median + 6 dB and > max - 60 dB; at most 30 strongest peaks per second.
- **Hashes:** each anchor pairs with the next 8 peaks with dt in [0.1, 2.0] s; `hash = f1<<17 | f2<<8 | dt_q`, dt quantised to 2 frames (32 ms). Relative times only -> time-shift invariant.
- **Index:** `hash -> [(clip_id, t_anchor)]`; query returns offset votes `t_db - t_query` per clip.
- **Score:** largest number of votes in any 3 adjacent offset bins (+-1 frame jitter) = aligned hashes. `score = a / (a + threshold)`.
- **Library:** one clip per scam script (enIN_Neerja, normal, clean): 10 clips, 28,628 hashes; build 0.3 s.

**Phase 2 probe (all 7 augmentations of each library clip; not yet the formal dev/test eval):** recall@1 = 1.00 for clean/tel/snr20/snr10/snr5/gsm/spd95/spd105. Median aligned hashes: clean 2546, tel 2216, snr20 1749, snr10 1014, snr5 692, gsm 644, spd95 83, spd105 104. Negatives: other voices of the same script up to 40, legit clips <= 7. Query ~30 ms for ~20 s audio.
**Threshold:** provisional 50 (between worst negative 40 and weakest positive 62); recalibrate on dev in Phase 4. A 10 s GSM excerpt gave 88 aligned hashes, so live windows will be close to the threshold: calibrate per window length.
**Why speed change hurts:** `atempo` keeps pitch but stretches time, so dt values and offsets drift; only hashes from short spans still align.
**Plot:** `plot_offset_histograms(...)` -> `artifacts/results/offset_hist_example.png` (spike at +5.0 s for an excerpt starting at 5 s).
