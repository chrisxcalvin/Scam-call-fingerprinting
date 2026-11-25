"""Print the report/README tables as Markdown from artifacts/results/summary.json (never typed by hand).
Usage: python scripts/results_tables.py [--split test]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    summ = json.loads((ROOT / "artifacts" / "results" / "summary.json").read_text(encoding="utf-8"))
    s = summ["splits"][a.split]
    l1, l2, f, lv = s["layer1"], s["layer2"], s["fusion"], s.get("live")
    print(f"**Split `{a.split}`**: {s['n_clips']} clips ({s['n_scam']} scam, {s['n_legit']} legit); "
          f"generated {summ['generated_at']}, commit {summ.get('git_commit')}.\n")
    print(f"**Layer 1: fingerprint** (threshold {l1['threshold_aligned_hashes']} aligned hashes; "
          f"false-match rate {l1['false_match_rate']} on {l1['n_negatives']} legit/wrong-script negatives; "
          f"same-script near-copies matched {l1['same_script_near_copy_match_rate']} (n={l1['n_same_script_near_copies']}); "
          f"query p50 {l1['query_ms_p50']} ms)\n")
    print(md(l1["by_augmentation"], ["augmentation", "n", "recall_at_1", "median_aligned", "min_aligned"]))
    print(f"\n**Layer 2: script + red flags** (sanity: library scripts read by other voices). AUC script similarity "
          f"{l2['auc_script_sim']} (old similarity-to-library), **learned wording model {l2.get('auc_text_model')}**, red flags {l2['auc_redflags']}; scam-type top-1 {l2['scam_type_top1_accuracy']}; "
          f"by language (wording model): " + ", ".join(f"{k}: AUC {v.get('auc_text_model', v['auc_script_sim'])} (n={v['n']})" for k, v in l2["by_language"].items())
          + "; STT WER (English): " + ", ".join(f"{k} {v}" for k, v in l2["wer_en_by_aug"].items()) + "\n")
    print("**Fusion + ablation** (flagged = level high or scam)\n")
    print(md(s["ablation"], ["config", "precision", "recall", "f1", "fpr_legit", "auc"]))
    print(f"\nConfusion (full): TP {f['tp']}, FN {f['fn']}, FP {f['fp']}, TN {f['tn']}. Missed scams: {f['n_missed']}"
          + (" (" + ", ".join(m["file"] for m in f["missed_scams"][:6]) + ("…" if f["n_missed"] > 6 else "") + ")" if f["n_missed"] else "")
          + f". False alarms: {len(f['false_alarms'])}" + (" (" + ", ".join(m["file"] for m in f["false_alarms"][:6]) + ")" if f["false_alarms"] else "") + ".\n")
    if lv:
        w, fp = lv["words_only"], lv["fingerprint_alerts"]
        print(f"**Live (simulated real-time)**: scam alerted {lv['scam_alerted']} (n={lv['n_scam']}); legit alerted "
              f"{lv['legit_alert_rate']} (n={lv['n_legit']}); fingerprint alerts at p50 {fp['first_alert_t_s_p50']} s into the call; "
              f"words-only latency p50 {w['latency_s_p50']} s, p95 {w['latency_s_p95']} s, within 5 s {w['share_within_5s']}.\n")
    t = {r["stage"]: r for r in s["timings"]}
    print(f"**Latency (upload, per clip)**: total p50 {t['total']['p50_ms']} ms / p95 {t['total']['p95_ms']} ms; "
          f"STT p50 {t['stt']['p50_ms']} ms; fingerprint match p50 {t['fingerprint_match']['p50_ms']} ms.")


if __name__ == "__main__":
    main()
