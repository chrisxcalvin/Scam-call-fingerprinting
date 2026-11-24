import io

import numpy as np
import scipy.signal as ss
import soundfile as sf

from scamshield.audio.io import load_audio
from scamshield.audio.preprocess import frame_signal, pre_emphasis, preprocess
from scamshield.audio.vad import vad
from scamshield.fingerprint.spectrogram import spectrogram


def _sine(f, sr=16000, dur=1.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_load_audio_bytes_and_stereo():
    st = np.stack([_sine(300, 8000), _sine(300, 8000)], axis=1)
    buf = io.BytesIO(); sf.write(buf, st, 8000, format="WAV")
    x, sr = load_audio(buf.getvalue())
    assert sr == 8000 and x.ndim == 1 and x.dtype == np.float32 and x.size == 8000


def test_preprocess_resample_dc_norm():
    x = _sine(440, 8000) * 0.2 + 0.3                      # DC offset, quiet
    y = preprocess(x, 8000)
    assert y.size == 16000 and abs(y.mean()) < 1e-3 and abs(np.max(np.abs(y)) - 0.99) < 1e-3


def test_pre_emphasis_formula():
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(pre_emphasis(x, 0.97), [1.0, 2 - 0.97, 4 - 0.97 * 2])


def test_frame_signal_shape_and_content():
    x = np.arange(10, dtype=float)
    f = frame_signal(x, 4, 2)
    assert f.shape == (4, 4) and list(f[1]) == [2, 3, 4, 5] and list(f[3]) == [6, 7, 8, 9]
    assert frame_signal(np.arange(10.0), 4, 4).shape == (3, 4)   # padded tail
    assert frame_signal(np.ones(3), 4, 2).shape == (1, 4)


def test_vad_finds_known_segment():
    sr = 16000
    rng = np.random.default_rng(0)
    sil = lambda d: 1e-4 * rng.standard_normal(int(sr * d))
    t = np.arange(sr) / sr
    speech = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) + 0.02 * rng.standard_normal(sr)
    x = np.r_[sil(1.0), speech, sil(1.0)]
    segs = vad(x, sr)
    assert len(segs) == 1
    assert abs(segs[0][0] - 1.0) < 0.1 and abs(segs[0][1] - 2.0) < 0.1


def test_vad_silence_returns_empty():
    assert vad(np.zeros(16000), 16000) == []


def test_spectrogram_peak_frequency_and_shape():
    sr, f0 = 16000, 1000
    S, freqs, times = spectrogram(_sine(f0), sr)
    assert S.shape[0] == 257 and S.shape[1] == times.size
    assert abs(freqs[np.argmax(S.mean(axis=1))] - f0) <= sr / 512


def test_spectrogram_matches_scipy_stft():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    S, _, _ = spectrogram(x, 16000)
    _, _, Z = ss.stft(x, 16000, window="hann", nperseg=512, noverlap=256, boundary=None, padded=False)
    ref = 20 * np.log10(np.abs(Z) * 256 + 1e-10)         # scipy scales by 1/sum(w) = 1/256
    n = ref.shape[1]
    np.testing.assert_allclose(S[:, :n], ref, atol=1e-6)
