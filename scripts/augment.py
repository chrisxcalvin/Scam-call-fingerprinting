"""Make phone-line / noisy / speed-changed query clips from clean/*.wav.
pip install numpy scipy soundfile ; ffmpeg on PATH. Optional: put real noise wavs (MUSAN/DEMAND) in noise/."""
import csv, subprocess, random
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

CLEAN, OUT, NOISE = Path("clean"), Path("augmented"), Path("noise")
OUT.mkdir(exist_ok=True); random.seed(0); np.random.seed(0)

def telephone(x, sr=16000):                       # 300-3400 Hz band, 8 kHz sampling, back to 16 kHz
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, x)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)

def add_noise(x, snr_db):
    files = list(NOISE.glob("*.wav"))
    if files:
        n, _ = sf.read(random.choice(files)); n = n if n.ndim == 1 else n[:,0]
        n = np.resize(n, len(x))
    else:
        n = np.random.randn(len(x))               # fallback: white noise
    px, pn = np.mean(x**2) + 1e-12, np.mean(n**2) + 1e-12
    n = n * np.sqrt(px / (pn * 10**(snr_db/10)))
    return x + n

def ffmpeg_codec(src, dst, codec_args):           # GSM 6.10 round trip = real telephony codec artifacts
    tmp = dst.with_suffix(".tmp.gsm")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),"-ar","8000","-ac","1",*codec_args,str(tmp)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(tmp),"-ar","16000","-ac","1",str(dst)], check=True)
    tmp.unlink()

def speed(src, dst, f):
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),"-filter:a",f"atempo={f}","-ar","16000",str(dst)], check=True)

base = {r["file"]: r for r in csv.DictReader(open("manifest.csv", encoding="utf-8"))}
rows = []
def log(name, parent, tag):
    r = dict(base[parent]); r.update(file=name, augmentation=tag); rows.append(r)

for wav in sorted(CLEAN.glob("*.wav")):
    x, sr = sf.read(wav)
    jobs = {"tel": lambda d: sf.write(d, telephone(x), sr),
            "snr20": lambda d: sf.write(d, add_noise(x, 20), sr),
            "snr10": lambda d: sf.write(d, add_noise(x, 10), sr),
            "snr5":  lambda d: sf.write(d, add_noise(telephone(x), 5), sr),
            "gsm":   lambda d: ffmpeg_codec(wav, d, ["-c:a","gsm"]),
            "spd95": lambda d: speed(wav, d, 0.95),
            "spd105":lambda d: speed(wav, d, 1.05)}
    for tag, fn in jobs.items():
        name = f"{wav.stem}__{tag}.wav"; fn(OUT / name); log(name, wav.name, tag)
    print("done", wav.name)

with open("manifest_augmented.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
