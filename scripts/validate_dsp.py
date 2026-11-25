"""Validate our from-scratch DSP against reference libraries on real clips -> artifacts/results/dsp_validation.json.

  STFT         vs scipy.signal.stft          (max abs difference in dB)
  mel filters  vs librosa.filters.mel        (max abs difference; same HTK mel scale, no normalisation)
  MFCC         vs librosa mel power -> dB -> DCT (min per-coefficient correlation c1..c12, max abs difference)
  pitch (YIN)  vs librosa.yin                (share of our voiced frames within 5 %, median abs error in Hz)

librosa and scipy are used ONLY here and in tests, never inside src/ (project rule).
Usage: python scripts/validate_dsp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as ss
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scamshield.audio.io import load_audio  # noqa: E402
from scamshield.audio.preprocess import preprocess  # noqa: E402
from scamshield.dsp.features import mel_filterbank, mfcc, yin_pitch  # noqa: E402
from scamshield.fingerprint.spectrogram import spectrogram  # noqa: E402

CLIPS = ["scam_digital_arrest_en__enIN_Neerja__normal", "scam_kyc_otp_hi__enUS_Guy__normal", "scam_lottery_en__enGB_Ryan__normal",
         "legit_bank_alert_en__enIN_Prabhat__normal", "legit_friend_en__enGB_Ryan__normal", "legit_appointment_en__enIN_Neerja__normal"]
SR = 16000


def main() -> None:
    fb_ours = mel_filterbank(SR, 512, 26)
    fb_ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=26, htk=True, norm=None)
    fb_err = float(np.max(np.abs(fb_ours - fb_ref)))
    rows = []
    for name in CLIPS:
        x, sr = load_audio(ROOT / "data" / "clean" / f"{name}.wav")
        x = preprocess(x, sr)
        # STFT
        S, _, _ = spectrogram(x, SR)
        _, _, Z = ss.stft(x.astype(np.float64), SR, window="hann", nperseg=512, noverlap=256, boundary=None, padded=False)
        ref = 20 * np.log10(np.abs(Z) * 256 + 1e-10)
        n = ref.shape[1]
        stft_err = float(np.max(np.abs(S[:, :n] - ref)))
        # MFCC
        ours = mfcc(x, SR)
        # librosa centres the 400-sample window inside its 512-sample FFT frame (a 56-sample shift); prepend 56 zeros so both
        # use the same time alignment, and give it our symmetric Hamming window (np.hamming) as well.
        def ref_mfcc(window):
            Sm = librosa.feature.melspectrogram(y=np.r_[np.zeros(56), x.astype(np.float64)], sr=SR, n_fft=512, hop_length=160, win_length=400,
                                                window=window, center=False, power=2.0, n_mels=26, htk=True, norm=None)
            return dct(10 * np.log10(np.maximum(Sm, 1e-10)), type=2, axis=0, norm="ortho")[:13].T
        refm = ref_mfcc(np.hamming(400))
        m = min(len(ours), len(refm)) - 2
        d = np.abs(ours[2:m, 1:13] - refm[2:m, 1:13])
        corr = [float(np.corrcoef(ours[2:m, k], refm[2:m, k])[0, 1]) for k in range(1, 13)]
        mfcc_err, mfcc_med = float(d.max()), float(np.median(d))
        per = ref_mfcc("hamming")
        mfcc_med_periodic = float(np.median(np.abs(ours[2:m, 1:13] - per[2:m, 1:13])))
        # pitch: align frame m (starts at m*hop) with librosa.yin(center=False) on the same frame length
        f0, vp = yin_pitch(x, SR)
        seg_len = 512 + int(SR / 60) + 1
        # same structure as ours: difference function over W = 512 samples, lags up to sr/60 -> frame_length = W + tau_max + 1
        ref_f0 = librosa.yin(x.astype(np.float64), fmin=60, fmax=400, sr=SR, frame_length=seg_len, win_length=512, hop_length=160,
                             center=False, trough_threshold=0.1)
        k = min(len(f0), len(ref_f0))
        voiced = ~np.isnan(f0[:k])
        rel = np.abs(f0[:k][voiced] - ref_f0[:k][voiced]) / ref_f0[:k][voiced]
        rows.append({"clip": name, "stft_max_err_db": stft_err, "mfcc_min_corr": min(corr), "mfcc_max_abs_err": mfcc_err, "mfcc_median_abs_err": mfcc_med,
                     "mfcc_median_abs_err_periodic_window": mfcc_med_periodic,
                     "voiced_frames": int(voiced.sum()), "f0_within_5pct": float(np.mean(rel < 0.05)),
                     "f0_median_abs_err_hz": float(np.median(np.abs(f0[:k][voiced] - ref_f0[:k][voiced])))})
        print(rows[-1], flush=True)
    out = {"filterbank_max_abs_err": fb_err, "clips": rows, "librosa": librosa.__version__, "scipy": __import__("scipy").__version__}
    (ROOT / "artifacts" / "results" / "dsp_validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("filterbank max abs err", fb_err)


if __name__ == "__main__":
    main()
