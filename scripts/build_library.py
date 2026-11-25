"""Build offline artifacts (docs/04 F1):
  artifacts/fingerprint_index.pkl   Layer 1 index over ONE clip per scam script (enIN_Neerja, normal, clean)
  artifacts/scripts_meta.json       scripts, library clips, script chunks, fingerprint meta
  artifacts/script_embeddings.npy   Layer 2 chunk embeddings (row i <-> chunks[i])
Usage: python scripts/build_library.py [--skip-embeddings]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scamshield.audio.io import load_audio  # noqa: E402
from scamshield.audio.preprocess import preprocess  # noqa: E402
from scamshield.config import ARTIFACTS_DIR, DATA_DIR, TARGET_SR  # noqa: E402
from scamshield.fingerprint.index import FingerprintIndex  # noqa: E402
from scamshield.fingerprint.matcher import fingerprint  # noqa: E402

LIBRARY_VOICE, LIBRARY_RATE = "en-IN-NeerjaNeural", "normal"
# Provisional fingerprint decision threshold (aligned hashes). Phase 2 probe: worst negative 40,
# weakest positive (spd95) 62. Must be recalibrated on the dev split by run_eval (Phase 4).
PROVISIONAL_FP_THRESHOLD = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    scripts = json.load(open(DATA_DIR / "scripts.json", encoding="utf-8"))
    rows = list(csv.DictReader(open(DATA_DIR / "manifest.csv", encoding="utf-8")))
    lib = [r for r in rows if r["label"] == "scam" and r["voice"] == LIBRARY_VOICE and r["rate"] == LIBRARY_RATE]

    t0 = time.time()
    index = FingerprintIndex()
    clips = {}
    for r in lib:
        x, sr = load_audio(DATA_DIR / "clean" / r["file"])
        x = preprocess(x, sr)
        peaks, hashes = fingerprint(x, TARGET_SR)
        cid = Path(r["file"]).stem
        index.add(cid, hashes)
        clips[cid] = {"script_id": r["script_id"], "scam_type": r["type"], "lang": r["lang"],
                      "file": r["file"], "duration_s": round(len(x) / TARGET_SR, 2), "n_peaks": int(len(peaks)),
                      "n_hashes": len(hashes)}
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    index.save(ARTIFACTS_DIR / "fingerprint_index.pkl")
    print(f"fingerprint: {len(clips)} library clips, {len(index)} hashes, {len(index.table)} distinct, {time.time() - t0:.1f}s")

    meta = {"scripts": scripts, "library_clips": clips,
            "fingerprint": {"clips": {k: {"script_id": v["script_id"], "scam_type": v["scam_type"]} for k, v in clips.items()},
                            "threshold": PROVISIONAL_FP_THRESHOLD, "threshold_source": "provisional (Phase 2 probe)",
                            "min_report": 5},
            "chunks": []}
    old_meta = ARTIFACTS_DIR / "scripts_meta.json"
    if args.skip_embeddings and old_meta.exists():   # keep the existing Layer 2 chunks
        meta["chunks"] = json.loads(old_meta.read_text(encoding="utf-8")).get("chunks", [])
    if not args.skip_embeddings:
        from scamshield.understanding.script_match import ScriptLibrary
        t0 = time.time()
        corpus = [x for n in ("train.json", "train_direct.json") for x in json.load(open(ROOT / "scripts" / "corpus" / n, encoding="utf-8"))
                  if x["label"] == "scam"]
        lib_scripts = scripts + [{"id": x["id"], "label": "scam", "type": x["type"], "lang": x["lang"], "text": x["text"]} for x in corpus]
        sl = ScriptLibrary.build(lib_scripts)
        print(f"script library: {len(scripts)} original + {len(corpus)} generated scam scripts")
        np.save(ARTIFACTS_DIR / "script_embeddings.npy", sl.emb)
        meta["chunks"] = sl.chunks
        print(f"embeddings: {len(sl.chunks)} chunks x {sl.emb.shape[1]}-d, {time.time() - t0:.1f}s")
    (ARTIFACTS_DIR / "scripts_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    fc = ARTIFACTS_DIR / "fusion_config.json"
    if not fc.exists():   # run_eval overwrites this with dev-calibrated values (Phase 4)
        from scamshield.fusion.risk import load_config
        cfg = load_config()
        cfg["source"] = "PROVISIONAL defaults written by build_library.py; not yet calibrated on the dev split"
        fc.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("wrote", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
