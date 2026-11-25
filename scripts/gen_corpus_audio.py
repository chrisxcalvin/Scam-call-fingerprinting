"""TTS audio for the generated corpus splits dev (-> eval split `dev2`) and test2, so Layer 2 is evaluated through the
real pipeline (TTS -> Whisper -> text model), not on clean text.  Output (gitignored):
  data/corpus_audio/<split>/{scripts.json, clean/*.wav, augmented/*.wav (tel), manifest.csv}
Voices rotate per script; none was used for training (training is text only).  Needs internet once and ffmpeg.
Usage: python scripts/gen_corpus_audio.py [dev|test2]"""
from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gen_holdout import _tts_ssl, synth, telephone  # noqa: E402

VOICES = {"dev": ["en-US-JennyNeural", "en-GB-ThomasNeural", "en-US-ChristopherNeural", "en-US-EmmaNeural"],
          "test2": ["en-US-AriaNeural", "en-GB-SoniaNeural", "en-IN-NeerjaExpressiveNeural", "en-AU-WilliamMultilingualNeural"]}
OUT = ROOT / "data" / "corpus_audio"


async def main(split: str) -> None:
    _tts_ssl()
    src = ROOT / "scripts" / "corpus" / f"{split}.json"
    scripts = json.load(open(src, encoding="utf-8"))
    base = OUT / ("dev2" if split == "dev" else split)
    (base / "clean").mkdir(parents=True, exist_ok=True)
    (base / "augmented").mkdir(exist_ok=True)
    (base / "scripts.json").write_text(json.dumps(scripts, ensure_ascii=False), encoding="utf-8")
    sem = asyncio.Semaphore(6)                       # a few parallel TTS requests (each clip takes seconds)
    done = 0

    async def one(i: int, s: dict) -> list[dict]:
        nonlocal done
        v = VOICES[split][i % 4]
        tag = "_".join(v.split("-")[:2]) + "_" + v.split("-")[2].replace("Neural", "").replace("Multilingual", "M").replace("Expressive", "X")
        wav = base / "clean" / f"{s['id']}__{tag}__normal.wav"
        if not wav.exists():
            async with sem:
                mp3 = wav.with_suffix(".mp3")
                await synth(s["text"], v, mp3)
                proc = await asyncio.create_subprocess_exec("ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav))
                await proc.wait()
                mp3.unlink(missing_ok=True)
        tel = base / "augmented" / f"{wav.stem}__tel.wav"
        if not tel.exists():
            x, sr = sf.read(wav)
            sf.write(tel, telephone(x), sr)
        done += 1
        if done % 20 == 0:
            print(f"{split}: {done}/{len(scripts)}", flush=True)
        meta = dict(label=s["label"], type=s["type"], lang=s["lang"], script_id=s["id"], voice=v, rate="normal", source="tts")
        return [dict(file=wav.name, augmentation="none", **meta), dict(file=tel.name, augmentation="tel", **meta)]

    rows = [r for rs in await asyncio.gather(*(one(i, s) for i, s in enumerate(scripts))) for r in rs]
    cols = ["file", "label", "type", "lang", "script_id", "voice", "rate", "source", "augmentation"]
    with open(base / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    print(f"{split}: {len(rows)} clips -> {base}")


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["dev", "test2"]):
        asyncio.run(main(sp))
