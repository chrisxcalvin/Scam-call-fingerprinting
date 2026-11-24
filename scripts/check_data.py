"""Validate the data/ layout, manifests, audio format and label counts (docs/06_DATA_AND_EVAL.md).

Usage: python scripts/check_data.py [--data DIR]
Exit code 0 = required data OK; 1 = problems. Missing optional sets (human, holdout) are warnings.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

COLUMNS = ["file", "label", "type", "lang", "script_id", "voice", "rate", "source", "augmentation"]
AUGS = {"tel", "snr20", "snr10", "snr5", "gsm", "spd95", "spd105"}


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_set(name: str, manifest: Path, wav_dir: Path, scripts: dict, errors: list, warns: list) -> list[dict]:
    """Check one manifest + its wav directory; append problems to errors/warns."""
    if not manifest.exists():
        errors.append(f"{name}: missing {manifest.name}")
        return []
    rows = read_manifest(manifest)
    if rows and list(rows[0]) != COLUMNS:
        errors.append(f"{name}: manifest columns {list(rows[0])} != {COLUMNS}")
    listed = {r["file"] for r in rows}
    on_disk = {p.name for p in wav_dir.glob("*.wav")} if wav_dir.exists() else set()
    for f in sorted(listed - on_disk):
        errors.append(f"{name}: in manifest but missing on disk: {f}")
    for f in sorted(on_disk - listed):
        warns.append(f"{name}: on disk but not in manifest (ignored): {f}")
    bad_fmt = 0
    for r in rows:
        p = wav_dir / r["file"]
        if not p.exists():
            continue
        info = sf.info(p)
        if info.samplerate != 16000 or info.channels != 1 or info.frames == 0:
            bad_fmt += 1
            errors.append(f"{name}: not 16 kHz mono / empty: {r['file']} ({info.samplerate} Hz, {info.channels} ch)")
        if r["script_id"] not in scripts:
            errors.append(f"{name}: unknown script_id {r['script_id']} in {r['file']}")
        elif scripts[r["script_id"]]["label"] != r["label"]:
            errors.append(f"{name}: label mismatch for {r['file']}")
    print(f"[{name}] {len(rows)} manifest rows, {len(on_disk)} wavs on disk, {bad_fmt} bad format")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    data = ap.parse_args().data
    errors: list[str] = []
    warns: list[str] = []

    sp = data / "scripts.json"
    if not sp.exists():
        print(f"ERROR: {sp} not found"); return 1
    scripts = {s["id"]: s for s in json.load(open(sp, encoding="utf-8"))}
    lc = Counter(s["label"] for s in scripts.values())
    print(f"scripts.json: {len(scripts)} scripts  {dict(lc)}")

    clean = check_set("clean", data / "manifest.csv", data / "clean", scripts, errors, warns)
    aug = check_set("augmented", data / "manifest_augmented.csv", data / "augmented", scripts, errors, warns)

    if clean:
        print("  clean labels:", dict(Counter(r["label"] for r in clean)),
              "| voices:", sorted({r["voice"] for r in clean}), "| rates:", sorted({r["rate"] for r in clean}))
        per_script = Counter(r["script_id"] for r in clean)
        if set(per_script) != set(scripts):
            errors.append(f"clean: scripts without clips: {sorted(set(scripts) - set(per_script))}")
    if aug:
        print("  augmented labels:", dict(Counter(r["label"] for r in aug)))
        print("  augmentations:", dict(sorted(Counter(r["augmentation"] for r in aug).items())))
        if {r["augmentation"] for r in aug} != AUGS:
            errors.append(f"augmented: augmentation set != {sorted(AUGS)}")
        clean_files = {r["file"] for r in clean}
        orphans = [r["file"] for r in aug if f"{r['file'].rsplit('__', 1)[0]}.wav" not in clean_files]
        if orphans:
            errors.append(f"augmented: {len(orphans)} clips whose clean parent is not in manifest.csv")

    for name, m in [("human", "manifest_human.csv"), ("holdout", "scripts_holdout.json")]:
        if not (data / name).exists() and not (data / m).exists():
            warns.append(f"{name}: not present yet (optional until Phase 6 / data collection)")

    for w in warns[:15]:
        print("WARN :", w)
    if len(warns) > 15:
        print(f"WARN : ... {len(warns) - 15} more warnings")
    for e in errors[:30]:
        print("ERROR:", e)
    print("RESULT:", "FAIL" if errors else "OK", f"({len(errors)} errors, {len(warns)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
