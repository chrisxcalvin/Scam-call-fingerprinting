"""Train the Layer-2 text model on the generated TRAIN corpus only -> artifacts/text_model.pkl.
dev (calibration) and test2 (locked) are never read here.   Usage: python scripts/train_text_model.py"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scamshield.understanding.text_model import TextModel  # noqa: E402

files = [ROOT / "scripts" / "corpus" / n for n in ("train.json", "train_direct.json")]
train = [x for p in files for x in json.load(open(p, encoding="utf-8"))]
m = TextModel.fit([x["text"] for x in train], [x["label"] == "scam" for x in train],
                  meta={"corpus": [p.name for p in files], "corpus_sha256": [hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files]})
m.save()
sc, lg = m.top_features(12)
print(f"trained on {len(train)} scripts ({sum(x['label'] == 'scam' for x in train)} scam) -> artifacts/text_model.pkl")
print("top SCAM n-grams :", ", ".join(f"{a} ({b:+.1f})" for a, b in sc))
print("top LEGIT n-grams:", ", ".join(f"{a} ({b:+.1f})" for a, b in lg))
