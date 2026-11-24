"""Generate clean TTS clips for every script x voice. Needs internet: pip install edge-tts ; ffmpeg on PATH."""
import asyncio, csv, json, subprocess, sys
from pathlib import Path
import edge_tts

VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-US-GuyNeural", "en-GB-RyanNeural"]
RATES = {"normal": "+0%", "fast": "+12%"}          # speaking-rate variants
OUT = Path("clean"); OUT.mkdir(exist_ok=True)

async def one(text, voice, rate, mp3):
    await edge_tts.Communicate(text, voice, rate=rate).save(str(mp3))

async def main():
    scripts = json.load(open("scripts.json", encoding="utf-8"))
    rows = []
    for s in scripts:
        for v in VOICES:
            for rname, rate in RATES.items():
                stem = f"{s['id']}__{v.split('-')[0]}{v.split('-')[1]}_{v.split('-')[2][:-6]}__{rname}"
                mp3, wav = OUT / f"{stem}.mp3", OUT / f"{stem}.wav"
                if not wav.exists():
                    await one(s["text"], v, rate, mp3)
                    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(mp3),"-ar","16000","-ac","1",str(wav)], check=True)
                    mp3.unlink()
                rows.append(dict(file=wav.name, label=s["label"], type=s["type"], lang=s["lang"],
                                 script_id=s["id"], voice=v, rate=rname, source="tts", augmentation="none"))
                print("ok", wav.name)
    with open("manifest.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

asyncio.run(main())
