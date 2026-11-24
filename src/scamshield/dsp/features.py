"""Frame-level features (Owner B). NumPy/SciPy only (scipy.fft.dct allowed for the DCT step)."""
from __future__ import annotations

import numpy as np
from scipy.fft import dct

from ..audio.preprocess import frame_signal
from ..config import N_MELS, PITCH_RANGE_HZ, YIN_THRESHOLD

_EPS = 1e-10


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """mel = 2595 log10(1 + f / 700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    """Triangular filters, centres equally spaced on the mel scale 0..sr/2. Shape (n_mels, n_fft//2+1)."""
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    fft_freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, fft_freqs.size))
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (fft_freqs - lo) / (ce - lo)
        down = (hi - fft_freqs) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """MFCC: frame (25 ms / hop 10 ms) -> Hamming -> |FFT|^2 (512) -> mel filterbank -> 10*log10 -> DCT-II (ortho).

    Mel: m = 2595 log10(1 + f/700) (perceptual pitch scale); log: loudness is perceived
    logarithmically; DCT decorrelates the filterbank energies. Returns (n_frames, n_mfcc).
    """
    fl, hop, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    frames = frame_signal(np.asarray(x, dtype=np.float64), fl, hop) * np.hamming(fl)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel_e = power @ mel_filterbank(sr, n_fft).T
    log_mel = 10.0 * np.log10(mel_e + _EPS)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def yin_pitch(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """YIN pitch tracker. Returns (f0_hz, voiced_prob) per frame (hop 10 ms); f0 = nan when unvoiced.

    1. d(tau) = sum_{j<W} (x[j] - x[j+tau])^2         (computed via FFT cross-correlation)
    2. d'(tau) = d(tau) * tau / sum_{k<=tau} d(k),  d'(0) = 1   (cumulative-mean normalisation)
    3. first tau with d'(tau) < 0.1, then descend to the local minimum (absolute threshold)
    4. parabolic interpolation around tau;  f0 = sr / tau.   voiced_prob = 1 - d'(tau).
    """
    x = np.asarray(x, dtype=np.float64)
    tau_min, tau_max = int(sr / PITCH_RANGE_HZ[1]), int(sr / PITCH_RANGE_HZ[0])
    W, hop = 512, int(0.010 * sr)
    seg_len = W + tau_max + 1
    frames = frame_signal(x, seg_len, hop)
    n_fft = 1 << int(np.ceil(np.log2(seg_len + W)))
    f0 = np.full(frames.shape[0], np.nan)
    vp = np.zeros(frames.shape[0])
    for m, seg in enumerate(frames):
        a = seg[:W]
        e0 = float(np.dot(a, a))
        if e0 < 1e-8:
            continue
        cs = np.r_[0.0, np.cumsum(seg ** 2)]
        # r[tau] = sum_j a[j] * seg[j+tau]
        r = np.fft.irfft(np.conj(np.fft.rfft(a, n_fft)) * np.fft.rfft(seg, n_fft), n_fft)[: tau_max + 2]
        taus = np.arange(tau_max + 2)
        d = e0 + (cs[W + taus] - cs[taus]) - 2.0 * r
        d[0] = 0.0
        cm = np.cumsum(d[1:])
        dn = np.ones_like(d)
        dn[1:] = d[1:] * taus[1:] / np.maximum(cm, _EPS)
        tau = None
        t = tau_min
        while t <= tau_max:
            if dn[t] < YIN_THRESHOLD:
                while t + 1 <= tau_max and dn[t + 1] < dn[t]:
                    t += 1
                tau = t
                break
            t += 1
        if tau is None:
            continue
        if 1 <= tau < tau_max:                       # parabolic interpolation
            s0, s1, s2 = dn[tau - 1], dn[tau], dn[tau + 1]
            denom = s0 - 2 * s1 + s2
            shift = 0.5 * (s0 - s2) / denom if abs(denom) > _EPS else 0.0
            tau_est = tau + shift
        else:
            tau_est = float(tau)
        f0[m] = sr / tau_est
        vp[m] = float(np.clip(1.0 - dn[tau], 0.0, 1.0))
    return f0, vp


def short_time_energy(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """E[m] = sum_n x[n + mH]^2 per frame."""
    return np.sum(frame_signal(np.asarray(x, dtype=np.float64), frame_len, hop) ** 2, axis=1)


def zcr(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Zero-crossing rate per frame: mean(|sign(x[n]) - sign(x[n-1])|) / 2 (fraction of sign changes)."""
    s = np.sign(frame_signal(np.asarray(x, dtype=np.float64), frame_len, hop))
    return np.mean(np.abs(np.diff(s, axis=1)), axis=1) / 2
