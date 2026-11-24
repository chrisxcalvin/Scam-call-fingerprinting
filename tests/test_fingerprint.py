import numpy as np

from scamshield.fingerprint.hashing import make_hashes
from scamshield.fingerprint.index import FingerprintIndex
from scamshield.fingerprint.matcher import fingerprint, match, score_offsets
from scamshield.fingerprint.peaks import band_bins, find_peaks
from scamshield.fingerprint.spectrogram import spectrogram

SR = 16000


def _tone_sequence(seed: int, dur: float = 10.0) -> np.ndarray:
    """Random sequence of 120 ms tone pairs in 300-3400 Hz: a stand-in for speech-like spectral structure."""
    rng = np.random.default_rng(seed)
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    parts = [0.3 * np.sin(2 * np.pi * rng.uniform(350, 3300) * t) + 0.2 * np.sin(2 * np.pi * rng.uniform(350, 3300) * t)
             for _ in range(int(dur / 0.12))]
    return np.concatenate(parts) * np.hanning(n * len(parts)) ** 0.1


def _meta(clip_ids):
    return {"clips": {c: {"script_id": c, "scam_type": "t"} for c in clip_ids}, "threshold": 20, "min_report": 1}


def test_peaks_are_local_maxima_in_band():
    S, _, _ = spectrogram(_tone_sequence(0), SR)
    p = find_peaks(S)
    lo, hi = band_bins()
    assert len(p) > 50 and p[:, 0].min() >= lo and p[:, 0].max() < hi
    assert np.all(np.diff(p[:, 1]) >= 0)          # sorted by time


def test_hashes_are_time_shift_invariant():
    peaks = np.array([[20, 10], [40, 20], [60, 30], [30, 60]])
    shifted = peaks + np.array([0, 100])
    h1, h2 = make_hashes(peaks), make_hashes(shifted)
    assert [h for h, _ in h1] == [h for h, _ in h2]
    assert [t + 100 for _, t in h1] == [t for _, t in h2]


def test_index_roundtrip(tmp_path):
    idx = FingerprintIndex(); idx.add("a", [(1, 5), (2, 9)])
    idx.save(tmp_path / "i.pkl")
    back = FingerprintIndex.load(tmp_path / "i.pkl")
    assert back.query([(1, 2), (3, 0)]) == {"a": [3]} and len(back) == 2


def test_offset_histogram_spike():
    aligned, off = score_offsets([7] * 30 + [8] * 5 + list(range(-50, 50, 7)))
    assert aligned >= 35 and off in (7, 8)


def test_match_noisy_excerpt_and_reject_unrelated():
    a, b = _tone_sequence(1), _tone_sequence(2)
    idx = FingerprintIndex()
    for cid, x in (("A", a), ("B", b)):
        idx.add(cid, fingerprint(x, SR)[1])
    rng = np.random.default_rng(3)
    q = a[3 * SR: 8 * SR] + 0.05 * rng.standard_normal(5 * SR)       # 5 s excerpt starting at 3 s, noisy
    m = match(q, SR, idx, _meta(["A", "B"]))
    assert m[0].clip_id == "A" and abs(m[0].offset_s - 3.0) < 0.05
    assert len(m) == 1 or m[1].aligned_hashes < m[0].aligned_hashes / 5
    unrelated = match(_tone_sequence(9), SR, idx, _meta(["A", "B"]))
    assert not unrelated or unrelated[0].aligned_hashes < m[0].aligned_hashes / 5


def test_offset_histogram_plot(tmp_path):
    from scamshield.fingerprint.plots import plot_offset_histograms
    idx = FingerprintIndex(); x = _tone_sequence(1); idx.add("A", fingerprint(x, SR)[1])
    p = plot_offset_histograms(fingerprint(x[2 * SR:], SR)[1], idx, tmp_path / "h.png")
    assert p.exists() and p.stat().st_size > 1000
