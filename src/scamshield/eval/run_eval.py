"""`python -m scamshield.eval.run_eval [--split dev|test] [--quick]` (Owner B; Layer 1 parts by A).

Runs the REAL pipeline (`api.pipeline.analyze_audio`) on every clip of a split, then:
  * dev  : calibrates the fingerprint threshold and the script-similarity range, writes fusion_config.json
  * both : writes metrics to artifacts/results/ (summary.json + CSV + PNG). Nothing is hand-entered.
Whisper transcripts are cached in artifacts/cache/stt/ so reruns are fast (timings come from the first run).
The test split must be run ONCE, at the end (docs/06).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..api.pipeline import Models, analyze_audio
from ..config import ARTIFACTS_DIR, DATA_DIR, FUSION_CONFIG_PATH, RESULTS_DIR, get_settings
from ..fusion.risk import fuse, load_config, script_component
from ..schemas import RiskComponents, TranscriptSegment
from ..understanding import stt
from .metrics import false_positive_rate, precision_recall_f1, roc_auc, roc_curve, wer

log = logging.getLogger(__name__)
LIBRARY_VOICE, LIBRARY_RATE = "en-IN-NeerjaNeural", "normal"
AUGS = ["none", "tel", "gsm", "snr20", "snr10", "snr5", "spd95", "spd105"]
QUICK_AUGS = ["none", "tel", "gsm", "snr10"]
# Split by script_id AND voice (never by file) so near-duplicates cannot leak (docs/06).
# Each scam type has a script on the dev side except tech_support (only one script exists).
SPLIT = {
    "dev": {"scripts": ["scam_digital_arrest_en", "scam_kyc_otp_hi", "scam_lottery_en", "scam_relative_hi",
                        "scam_courier_en", "legit_bank_alert_en", "legit_delivery_en", "legit_appointment_en",
                        "legit_friend_hi"],
            "voices": ["en-IN-PrabhatNeural", "en-US-GuyNeural"]},
    "test": {"scripts": ["scam_digital_arrest_hi", "scam_kyc_otp_en", "scam_lottery_hi", "scam_relative_en",
                         "scam_techsupport_en", "legit_otp_notice_en", "legit_bank_alert_hi", "legit_friend_en"],
             "voices": ["en-GB-RyanNeural"]},
}


# ---------------------------------------------------------------- data
def load_clips(split: str, quick: bool) -> list[dict]:
    """Manifest rows of the split: its scripts, read by its voices or by the library voice.
    `holdout` = data/holdout (paraphrased + new scripts, voices unseen anywhere else)."""
    if split == "holdout":
        base = DATA_DIR / "holdout"
        out = []
        for r in csv.DictReader(open(base / "manifest_holdout.csv", encoding="utf-8")):
            if quick and r["augmentation"] not in QUICK_AUGS:
                continue
            r = dict(r)
            r["path"] = str(base / ("clean" if r["augmentation"] == "none" else "augmented") / r["file"])
            r["is_library_audio"] = False      # nothing here is the indexed audio: every fingerprint match is wrong
            out.append(r)
        return out
    sp = SPLIT[split]
    rows = list(csv.DictReader(open(DATA_DIR / "manifest.csv", encoding="utf-8")))
    rows += list(csv.DictReader(open(DATA_DIR / "manifest_augmented.csv", encoding="utf-8")))
    augs = set(QUICK_AUGS if quick else AUGS)
    out = []
    for r in rows:
        if r["script_id"] in sp["scripts"] and (r["voice"] in sp["voices"] or r["voice"] == LIBRARY_VOICE) \
                and r["augmentation"] in augs:
            r = dict(r)
            r["path"] = str(DATA_DIR / ("clean" if r["augmentation"] == "none" else "augmented") / r["file"])
            # Layer-1 positive = the exact audio indexed in the library (scam scripts only, library voice/rate)
            r["is_library_audio"] = r["label"] == "scam" and r["voice"] == LIBRARY_VOICE and r["rate"] == LIBRARY_RATE
            out.append(r)
    return out


class CachedSTT:
    """transcribe() with a JSON cache per (file, chunk offset); remembers the Whisper time of the first run."""

    def __init__(self, prefix: str = "") -> None:
        s = get_settings()
        self.prefix = prefix
        self.dir = ARTIFACTS_DIR / "cache" / "stt" / f"{s.whisper_model}_{s.stt_language}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = ""
        self.ms = 0.0
        self.last_ms: float | None = None

    def __call__(self, x, sr, offset_s: float = 0.0):
        p = self.dir / f"{self.prefix}{Path(self.file).stem}@{offset_s:.2f}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
        else:
            t = time.perf_counter()
            segs = stt.transcribe(x, sr, offset_s=offset_s)
            d = {"ms": (time.perf_counter() - t) * 1000, "segments": [s.model_dump() for s in segs]}
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        self.ms += d["ms"]
        self.last_ms = d["ms"]                        # read by LiveSession to report real Whisper time
        return [TranscriptSegment(**s) for s in d["segments"]]


def run_pipeline(clips: list[dict], models: Models, scripts: dict) -> list[dict]:
    """Analyse every clip with the real pipeline and keep the raw layer outputs."""
    cache = CachedSTT()
    recs = []
    for i, c in enumerate(clips):
        cache.file, cache.ms = c["file"], 0.0
        res, ev = analyze_audio(Path(c["path"]).read_bytes(), models, transcribe_fn=cache)
        fp0, sm0 = (res.fingerprint[0] if res.fingerprint else None), (res.script[0] if res.script else None)
        text = " ".join(s.text for s in res.transcript)
        tm = dict(res.timings_ms); tm["stt"] = cache.ms           # Whisper time from the uncached run
        recs.append({
            "file": c["file"], "label": c["label"], "type": c["type"], "lang": c["lang"], "script_id": c["script_id"],
            "voice": c["voice"], "rate": c["rate"], "aug": c["augmentation"], "is_library_audio": c["is_library_audio"],
            "duration_s": res.duration_s,
            "fp_aligned": fp0.aligned_hashes if fp0 else 0, "fp_script": fp0.script_id if fp0 else "",
            "fp_type": fp0.scam_type if fp0 else "", "fp_query_hashes": fp0.query_hashes if fp0 else 0,
            "sim": sm0.similarity if sm0 else 0.0, "sim_script": sm0.script_id if sm0 else "",
            "sim_type": sm0.scam_type if sm0 else "",
            "redflags": res.final.components.redflags,
            "flag_cats": sorted({f.category for f in res.red_flags if not f.negated}),
            "n_negated": sum(f.negated for f in res.red_flags),
            "safe_context": any("safe-context" in r for r in res.final.top_reasons),
            "wer": wer(scripts[c["script_id"]]["text"], text) if c["lang"] == "en" else None,
            "transcript": text, "timings": tm,
        })
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            log.warning("analysed %d/%d clips", i + 1, len(clips))
    return recs


# ---------------------------------------------------------------- calibration (dev only)
def harmful(r: dict) -> bool:
    """A Layer-1 negative whose top match would be harmful: legit audio, or a match to the WRONG script.
    (Same-script audio in the library voice at another rate is a near-copy of the indexed recording;
    matching it to the right script is not a false alarm, so it is reported separately.)"""
    return not r["is_library_audio"] and (r["label"] == "legit" or r["fp_script"] != r["script_id"])


def calibrate(recs: list[dict]) -> dict:
    """Fingerprint threshold = 2 x (max aligned hashes over HARMFUL dev negatives) + 1, i.e. a 2x safety
    margin over the worst legit/wrong-script collision (0 harmful matches on dev).
    Script range: lo = highest legit similarity (legit maps to 0), hi = max(lo + 0.10, 25th pct of scam sim).
    Weights and level bounds stay at the documented values (too little data to fit them)."""
    neg = [r["fp_aligned"] for r in recs if harmful(r)]
    legit = [r["sim"] for r in recs if r["label"] == "legit"]
    scam = [r["sim"] for r in recs if r["label"] == "scam"]
    lo = round(float(max(legit)) + 0.005, 3)
    hi = round(max(lo + 0.10, float(np.percentile(scam, 25))), 3)
    cfg = load_config()
    cfg.update({"fp_threshold": 2 * int(max(neg)) + 1, "script_sim_range": [lo, hi],
                "source": "run_eval dev calibration", "calibration": {
                    "fp_threshold_rule": "2 x max aligned hashes over harmful dev Layer-1 negatives (legit or wrong script) + 1",
                    "fp_harmful_negatives_n": len(neg), "fp_harmful_negatives_max": int(max(neg)),
                    "script_range_rule": "lo = max legit sim; hi = max(lo+0.10, p25 scam sim)",
                    "legit_sim_max": float(max(legit)), "scam_sim_p25": float(np.percentile(scam, 25)),
                    "n_legit": len(legit), "n_scam": len(scam),
                    "weights_and_levels": "documented defaults (docs/02), not fitted"}})
    return cfg


def fused(r: dict, cfg: dict, use: set[str] | None = None) -> tuple[float, str]:
    """Recompute fusion for a record with config `cfg`, optionally keeping only some layers (ablation)."""
    use = use or {"fingerprint", "script", "redflags", "voice"}
    thr = cfg["fp_threshold"]
    hit = r["fp_aligned"] >= thr and "fingerprint" in use
    comps = RiskComponents(fingerprint=r["fp_aligned"] / (r["fp_aligned"] + thr) if hit else 0.0,
                           script=script_component(r["sim"], cfg) if "script" in use else 0.0,
                           redflags=r["redflags"] if "redflags" in use else 0.0, voice=0.0)
    st = fuse(comps, hit, safe_context=r["safe_context"] and "redflags" in use, cfg=cfg)
    return st.score, st.level


# ---------------------------------------------------------------- metrics
def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else None


def evaluate(recs: list[dict], cfg: dict, split: str, out: Path) -> dict:
    thr = cfg["fp_threshold"]
    y = np.array([r["label"] == "scam" for r in recs])
    # ---- Layer 1
    pos = [r for r in recs if r["is_library_audio"]]
    neg = [r for r in recs if not r["is_library_audio"]]
    l1_rows = []
    for a in AUGS:
        g = [r for r in pos if r["aug"] == a]
        if g:
            l1_rows.append({"augmentation": "clean" if a == "none" else a, "n": len(g),
                            "recall_at_1": round(np.mean([r["fp_script"] == r["script_id"] and r["fp_aligned"] >= thr for r in g]), 3),
                            "top1_correct_any_threshold": round(np.mean([r["fp_script"] == r["script_id"] for r in g]), 3),
                            "median_aligned": int(np.median([r["fp_aligned"] for r in g])),
                            "min_aligned": int(min(r["fp_aligned"] for r in g))})
    harm = [r for r in neg if harmful(r)]
    near = [r for r in neg if not harmful(r)]
    fmr = float(np.mean([r["fp_aligned"] >= thr for r in harm])) if harm else None
    near_rate = float(np.mean([r["fp_aligned"] >= thr for r in near])) if near else None
    l1_auc = roc_auc(np.r_[np.ones(len(pos)), np.zeros(len(harm))], [r["fp_aligned"] for r in pos + harm]) if pos else None
    fp_ms = [r["timings"]["fingerprint_match"] + r["timings"]["peaks_hashes"] + r["timings"]["stft"] for r in recs]
    # ---- Layer 2
    sims, flags = np.array([r["sim"] for r in recs]), np.array([r["redflags"] for r in recs])
    by_lang = {}
    for lang in sorted({r["lang"] for r in recs}):
        m = np.array([r["lang"] == lang for r in recs])
        by_lang[lang] = {"n": int(m.sum()), "auc_script_sim": round(roc_auc(y[m], sims[m]), 3),
                         "auc_redflags": round(roc_auc(y[m], flags[m]), 3)}
    wers = defaultdict(list)
    for r in recs:
        if r["wer"] is not None:
            wers["clean" if r["aug"] == "none" else r["aug"]].append(r["wer"])
    type_ok = [r["sim_type"] == r["type"] for r in recs if r["label"] == "scam"]
    # ---- Fusion + ablation
    configs = {"fp only": {"fingerprint"}, "script only": {"script"}, "flags only": {"redflags"},
               "voice only": {"voice"}, "full": None}
    abl = []
    full_levels = []
    for name, use in configs.items():
        res = [fused(r, cfg, use) for r in recs]
        pred = np.array([lv in ("high", "scam") for _, lv in res])
        p, rc, f1 = precision_recall_f1(y, pred)
        abl.append({"config": name, "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3),
                    "fpr_legit": round(false_positive_rate(y, pred), 3),
                    "auc": round(roc_auc(y, [s for s, _ in res]), 3)})
        if use is None:
            full_levels, full_scores, full_pred = [lv for _, lv in res], np.array([s for s, _ in res]), pred
    tp = int(np.sum(y & full_pred)); fpn = int(np.sum(~y & full_pred)); tn = int(np.sum(~y & ~full_pred)); fn = int(np.sum(y & ~full_pred))
    lvl = {lab: dict(Counter(lv for lv, r in zip(full_levels, recs) if r["label"] == lab)) for lab in ("scam", "legit")}
    misses = [{"file": r["file"], "score": round(float(s), 1), "level": lv}
              for r, s, lv, p in zip(recs, full_scores, full_levels, full_pred) if r["label"] == "scam" and not p]
    false_alarms = [{"file": r["file"], "score": round(float(s), 1), "level": lv}
                    for r, s, lv, p in zip(recs, full_scores, full_levels, full_pred) if r["label"] == "legit" and p]
    legit_otp = [{"file": r["file"], "level": lv, "negated_flags": r["n_negated"]} for r, lv in zip(recs, full_levels)
                 if r["label"] == "legit" and r["script_id"].startswith(("legit_bank", "legit_otp")) and r["aug"] == "none"]
    # ---- timings
    stages = sorted({k for r in recs for k in r["timings"]})
    timings = [{"stage": k, "p50_ms": pct([r["timings"][k] for r in recs if k in r["timings"]], 50),
                "p95_ms": pct([r["timings"][k] for r in recs if k in r["timings"]], 95)} for k in stages]

    # ---- write files
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / f"layer1_by_aug_{split}.csv", l1_rows)
    _csv(out / f"ablation_{split}.csv", abl)
    _csv(out / f"timings_{split}.csv", timings)
    _csv(out / f"fusion_confusion_{split}.csv", [{"": "actual scam", "flagged": tp, "not flagged": fn},
                                                 {"": "actual legit", "flagged": fpn, "not flagged": tn}])
    _csv(out / f"clips_{split}.csv", [{k: v for k, v in r.items() if k not in ("timings", "transcript", "flag_cats")} |
                                      {"fused_score": round(float(s), 1), "level": lv}
                                      for r, s, lv in zip(recs, full_scores, full_levels)])
    _plots(out, split, l1_rows, y, sims, flags, full_scores)
    return {
        "split": split, "n_clips": len(recs), "n_scam": int(y.sum()), "n_legit": int((~y).sum()),
        "scripts": SPLIT[split]["scripts"] if split in SPLIT else sorted({r["script_id"] for r in recs}),
        "voices": (SPLIT[split]["voices"] + [f"{LIBRARY_VOICE} (library voice)"]) if split in SPLIT
        else sorted({r["voice"] for r in recs}),
        "augmentations": sorted({r["aug"] for r in recs}),
        "layer1": {"threshold_aligned_hashes": thr, "by_augmentation": l1_rows, "n_positives": len(pos),
                   "n_negatives": len(harm), "false_match_rate": round(fmr, 4) if fmr is not None else None,
                   "false_match_definition": "top match >= threshold on legit audio or to the wrong script",
                   "n_same_script_near_copies": len(near),
                   "same_script_near_copy_match_rate": round(near_rate, 4) if near_rate is not None else None,
                   "near_copy_definition": "same scam script, library voice at the other speaking rate (+augmentations)",
                   "auc_aligned_hashes": round(l1_auc, 3) if l1_auc is not None else None, "query_ms_p50": pct(fp_ms, 50), "query_ms_p95": pct(fp_ms, 95),
                   "index_hashes": None},
        "layer2": {"note": ("HOLDOUT: scam scripts are newly written paraphrases (different wording and details) of the six "
                            "scam types, legit scripts are new and include genuine calls that mention OTP/KYC/police/accident, "
                            "read by voices unseen anywhere else. Thresholds are the dev-calibrated ones, unchanged. "
                            "Author note: the scripts were written by the same assistant that wrote the red-flag lexicon, so "
                            "red-flag numbers may be optimistic." if split == "holdout" else
                            "Scripts here are the library scripts read by other voices: a SANITY result, not the honest "
                            "holdout/human test (docs/06)."),
                   "auc_script_sim": round(roc_auc(y, sims), 3), "auc_redflags": round(roc_auc(y, flags), 3),
                   "by_language": by_lang, "scam_type_top1_accuracy": round(float(np.mean(type_ok)), 3),
                   "wer_en_by_aug": {k: round(float(np.mean(v)), 3) for k, v in sorted(wers.items())}},
        "fusion": {"flag_rule": "level in {high, scam}", "tp": tp, "fp": fpn, "tn": tn, "fn": fn,
                   "precision": abl[-1]["precision"], "recall": abl[-1]["recall"], "f1": abl[-1]["f1"],
                   "fpr_legit": abl[-1]["fpr_legit"], "auc": abl[-1]["auc"], "levels": lvl,
                   "missed_scams": misses[:40], "n_missed": len(misses), "false_alarms": false_alarms,
                   "legit_otp_bank_clips": legit_otp},
        "ablation": abl, "timings": timings,
        "detection_latency": "not run yet (needs simulated-live mode, Phase 5)",
        "not_yet_available": ["holdout paraphrase scripts", "human recordings", "voice-cue scoring"],
    }


LIVE_AUGS = {"none", "tel", "snr10"}


def live_eval(clips: list[dict], models: Models, out: Path, split: str) -> dict:
    """Detection latency in simulated live mode (same code path as the WebSocket).

    latency = (audio time of the first high/scam alert + compute time of that tick) - end time of the
    transcript segment holding the first non-negated red-flag phrase. Negative = alert before any scam
    phrase was spoken (fingerprint hit). Legit clips: any alert is a live false alarm."""
    from ..api.session import simulate
    from ..audio.io import load_audio
    from ..audio.preprocess import preprocess
    from ..understanding.redflags import red_flags
    cache = CachedSTT(prefix="live_")
    rows = []
    sel = [c for c in clips if c["augmentation"] in LIVE_AUGS and c["rate"] == "normal"]
    for c in sel:
        cache.file = c["file"]
        x, sr = load_audio(c["path"])
        sess, ev = simulate(models, preprocess(x, sr), transcribe_fn=cache)
        alerts = [e for e in ev if e["type"] == "alert"]
        first = alerts[0] if alerts else None
        key_end = None
        for seg in sess.transcript:
            if any(not f.negated for f in red_flags([seg])):
                key_end = seg.end_s
                break
        lat = None
        if first is not None and key_end is not None:
            lat = round(first["t_s"] + first.get("wall_ms", 0) / 1000 - key_end, 2)
        rows.append({"file": c["file"], "label": c["label"], "aug": c["augmentation"], "voice": c["voice"],
                     "duration_s": round(sess.duration_s, 1), "first_alert_t_s": first["t_s"] if first else None,
                     "alert_level": first["level"] if first else None, "first_flag_segment_end_s": key_end,
                     "latency_s": lat, "by_fingerprint": sess.fp_hit is not None and first is not None
                     and sess.fp_hit.query_span_s[1] <= (first["t_s"] if first else 0) + 0.01,
                     "max_level": max((e["state"]["level"] for e in ev if e["type"] == "risk"),
                                      key=["safe", "suspicious", "high", "scam"].index, default="safe"),
                     "tick_ms_p50": round(float(np.median(sess.tick_ms)), 1) if sess.tick_ms else None,
                     "tick_ms_max": round(float(max(sess.tick_ms)), 1) if sess.tick_ms else None})
    _csv(out / f"live_latency_{split}.csv", rows)
    scam = [r for r in rows if r["label"] == "scam"]
    legit = [r for r in rows if r["label"] == "legit"]
    lats = [r["latency_s"] for r in scam if r["latency_s"] is not None]
    return {"n_scam": len(scam), "n_legit": len(legit), "augmentations": sorted(LIVE_AUGS),
            "scam_alerted": round(float(np.mean([r["first_alert_t_s"] is not None for r in scam])), 3) if scam else None,
            "latency_s_p50": pct(lats, 50), "latency_s_p95": pct(lats, 95),
            "latency_s_max": round(float(max(lats)), 2) if lats else None,
            "share_within_5s": round(float(np.mean([v <= 5 for v in lats])), 3) if lats else None,
            "alerts_before_first_phrase": int(sum(v < 0 for v in lats)),
            "words_only": _lat_stats([r for r in scam if not r["by_fingerprint"]]),
            "fingerprint_alerts": _lat_stats([r for r in scam if r["by_fingerprint"]]),
            "legit_alert_rate": round(float(np.mean([r["first_alert_t_s"] is not None for r in legit])), 3) if legit else None,
            "legit_max_levels": dict(Counter(r["max_level"] for r in legit)),
            "tick_ms_p50": pct([r["tick_ms_p50"] for r in rows if r["tick_ms_p50"] is not None], 50),
            "tick_ms_max": round(float(max(r["tick_ms_max"] for r in rows if r["tick_ms_max"] is not None)), 1),
            "definition": "alert time + tick compute - end of the first transcript segment with a red flag"}


def _lat_stats(rows: list[dict]) -> dict:
    """Latency summary for a subset of scam streams (alerts caused by words vs by the fingerprint)."""
    lats = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    return {"n": len(rows), "alerted": round(float(np.mean([r["first_alert_t_s"] is not None for r in rows])), 3) if rows else None,
            "latency_s_p50": pct(lats, 50), "latency_s_p95": pct(lats, 95),
            "share_within_5s": round(float(np.mean([v <= 5 for v in lats])), 3) if lats else None,
            "first_alert_t_s_p50": pct([r["first_alert_t_s"] for r in rows if r["first_alert_t_s"] is not None], 50)}


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _plots(out: Path, split: str, l1_rows, y, sims, flags, fused_scores) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([r["augmentation"] for r in l1_rows], [r["recall_at_1"] for r in l1_rows], color="#4aa3ff")
    for i, r in enumerate(l1_rows):
        ax.text(i, r["recall_at_1"] + 0.02, f"n={r['n']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("recall@1"); ax.set_title(f"Layer 1 fingerprint recall by augmentation ({split})")
    fig.tight_layout(); fig.savefig(out / f"recall_by_aug_{split}.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, s in [("script similarity", sims), ("red flags", flags), ("fused score", fused_scores)]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc(y, s):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"Layer 2 / fusion ROC ({split}, n={len(y)})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"roc_layer2_{split}.png", dpi=120); plt.close(fig)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ARTIFACTS_DIR.parent, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "holdout"], default="dev")
    ap.add_argument("--quick", action="store_true", help="only clean/tel/gsm/snr10 versions")
    ap.add_argument("--no-live", action="store_true", help="skip the simulated-live latency evaluation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    t0 = time.time()
    scripts = {s["id"]: s for s in json.load(open(DATA_DIR / "scripts.json", encoding="utf-8"))}
    if args.split == "holdout":
        scripts = {s["id"]: s for s in json.load(open(DATA_DIR / "holdout" / "scripts_holdout.json", encoding="utf-8"))}
    models = Models().load()
    clips = load_clips(args.split, args.quick)
    print(f"{args.split}: {len(clips)} clips")
    recs = run_pipeline(clips, models, scripts)
    if args.split == "dev":
        cfg = calibrate(recs)
        FUSION_CONFIG_PATH.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print("wrote", FUSION_CONFIG_PATH, "| fp_threshold", cfg["fp_threshold"], "| script range", cfg["script_sim_range"])
    else:
        cfg = load_config()
        if "fp_threshold" not in cfg:
            raise SystemExit("Run the dev split first: test/holdout must use dev-calibrated settings.")
    res = evaluate(recs, cfg, args.split, RESULTS_DIR)
    res["layer1"]["index_hashes"] = len(models.index) if models.index else None
    if not args.no_live:
        models = Models().load(warm_stt=False)        # reload so the calibrated threshold is used live
        res["live"] = live_eval(clips, models, RESULTS_DIR, args.split)
        res["detection_latency"] = f"p50 {res['live']['latency_s_p50']} s, p95 {res['live']['latency_s_p95']} s (see live)"
    res["fusion_config"] = {k: cfg[k] for k in ("weights", "level_bounds", "fp_threshold", "script_sim_range", "source")}
    res["quick"] = args.quick
    res["runtime_s"] = round(time.time() - t0, 1)
    summ_p = RESULTS_DIR / "summary.json"
    summary = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {"splits": {}}
    summary["splits"][args.split] = res
    summary.update({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _git_commit(),
                    "stt": {"model": get_settings().whisper_model, "language": get_settings().stt_language}})
    summ_p.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    f = res["fusion"]
    if "live" in res:
        lv = res["live"]
        print(f"live: scam alerted {lv['scam_alerted']} | latency p50 {lv['latency_s_p50']} s p95 {lv['latency_s_p95']} s "
              f"| within 5 s {lv['share_within_5s']} | legit alert rate {lv['legit_alert_rate']}")
    print(f"fusion ({args.split}): precision {f['precision']} recall {f['recall']} FPR(legit) {f['fpr_legit']} "
          f"| L1 false-match rate {res['layer1']['false_match_rate']} | {res['runtime_s']} s")


if __name__ == "__main__":
    main()
