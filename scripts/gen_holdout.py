"""Generate the HOLDOUT audio set from scripts/holdout/scripts_holdout.json (docs/06).

Voices used here appear nowhere in the library, dev or test sets. Output (gitignored, under data/holdout/):
  scripts_holdout.json, clean/*.wav, augmented/*.wav (tel, snr10, gsm), manifest_holdout.csv
Needs internet once (edge-tts) and ffmpeg.  Usage: python scripts/gen_holdout.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "holdout" / "scripts_holdout.json"
OUT = ROOT / "data" / "holdout"
VOICES = ["en-US-AriaNeural", "en-GB-SoniaNeural", "en-IN-NeerjaExpressiveNeural", "en-AU-WilliamMultilingualNeural"]
random.seed(1)
np.random.seed(1)


def telephone(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """300-3400 Hz band, 8 kHz sampling, back to 16 kHz."""
    y = sosfilt(butter(4, [300, 3400], btype="band", fs=sr, output="sos"), x)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)


def add_white_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    n = np.random.randn(len(x))
    n *= np.sqrt((np.mean(x ** 2) + 1e-12) / ((np.mean(n ** 2) + 1e-12) * 10 ** (snr_db / 10)))
    return x + n


def gsm(src: Path, dst: Path) -> None:
    """GSM 6.10 round trip (raw .gsm container: the WAV container does not support the codec)."""
    tmp = dst.with_suffix(".tmp.gsm")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "gsm", str(tmp)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(dst)], check=True)
    tmp.unlink()


def _tts_ssl() -> None:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


async def synth(text: str, voice: str, mp3: Path, tries: int = 4) -> None:
    """edge-tts occasionally returns no audio (transient); retry with a growing pause."""
    import edge_tts
    for k in range(tries):
        try:
            await edge_tts.Communicate(text, voice).save(str(mp3))
            return
        except edge_tts.exceptions.NoAudioReceived:
            if k == tries - 1:
                raise
            await asyncio.sleep(2 * (k + 1))


async def main() -> None:
    _tts_ssl()
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "augmented").mkdir(exist_ok=True)
    shutil.copy(SRC, OUT / "scripts_holdout.json")
    scripts = json.load(open(SRC, encoding="utf-8"))
    rows = []
    for s in scripts:
        for v in VOICES:
            tag = "_".join(v.split("-")[:2]) + "_" + v.split("-")[2].replace("Neural", "").replace("Multilingual", "M").replace("Expressive", "X")
            stem = f"{s['id']}__{tag}__normal"
            wav = OUT / "clean" / f"{stem}.wav"
            if not wav.exists():
                mp3 = wav.with_suffix(".mp3")
                await synth(s["text"], v, mp3)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)], check=True)
                mp3.unlink()
            base = dict(label=s["label"], type=s["type"], lang=s["lang"], script_id=s["id"], voice=v, rate="normal", source="tts")
            rows.append(dict(file=wav.name, augmentation="none", **base))
            x, sr = sf.read(wav)
            for aug in ("tel", "snr10", "gsm"):
                name = f"{stem}__{aug}.wav"
                dst = OUT / "augmented" / name
                if not dst.exists():
                    if aug == "tel":
                        sf.write(dst, telephone(x), sr)
                    elif aug == "snr10":
                        sf.write(dst, add_white_noise(x, 10), sr)
                    else:
                        gsm(wav, dst)
                rows.append(dict(file=name, augmentation=aug, **base))
            print("ok", stem, flush=True)
    cols = ["file", "label", "type", "lang", "script_id", "voice", "rate", "source", "augmentation"]
    with open(OUT / "manifest_holdout.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    print(f"{len(rows)} clips -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
