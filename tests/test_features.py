import librosa
import numpy as np
from scipy.fft import dct

from scamshield.dsp.features import mel_filterbank, mfcc, short_time_energy, yin_pitch, zcr


def _harmonic(f0, sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    return sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in [(1, 1.0), (2, 0.5), (3, 0.3)]) * 0.3


def test_mfcc_matches_librosa_reference():
    rng = np.random.default_rng(0)
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.1 * rng.standard_normal(t.size)
    ours = mfcc(x, sr)
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, win_length=400,
                                       window="hamming", center=False, power=2.0, n_mels=26,
                                       htk=True, norm=None)
    ref = dct(librosa.power_to_db(S, top_db=None), type=2, axis=0, norm="ortho")[:13].T
    n = min(len(ours), len(ref))
    corr = np.corrcoef(ours[:n, 1:].ravel(), ref[:n, 1:].ravel())[0, 1]
    assert corr > 0.99
    assert ours.shape[1] == 13


def test_mel_filterbank_shape_and_partition():
    fb = mel_filterbank(16000, 512, 26)
    assert fb.shape == (26, 257) and (fb >= 0).all() and fb.max() <= 1.0 + 1e-9


def test_yin_pitch_synthetic_tones_within_2_percent():
    for f0 in (150.0, 220.0):
        f, vp = yin_pitch(_harmonic(f0), 16000)
        est = np.nanmedian(f)
        assert abs(est - f0) / f0 < 0.02, (f0, est)
        assert np.nanmean(vp[~np.isnan(f)]) > 0.9


def test_yin_silence_unvoiced():
    f, vp = yin_pitch(np.zeros(8000), 16000)
    assert np.isnan(f).all() and (vp == 0).all()


def test_energy_and_zcr():
    x = np.ones(800)
    np.testing.assert_allclose(short_time_energy(x, 400, 400), [400, 400])
    alt = np.tile([1.0, -1.0], 400)
    assert zcr(alt, 400, 400)[0] > 0.99
    assert zcr(x, 400, 400)[0] == 0
