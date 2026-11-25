"""Generate every figure and LaTeX table of the report from the real result files (no number is typed by hand).

Inputs : artifacts/results/{summary.json, dsp_validation.json, clips_<split>.csv}, data/ (audio), artifacts/ (models, STT cache)
Outputs: report/Figures/*.png and report/generated_tables.tex
Usage  : PYTHONPATH=src python scripts/make_report_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scamshield.audio.io import load_audio  # noqa: E402
from scamshield.audio.preprocess import preprocess  # noqa: E402
from scamshield.audio.vad import vad  # noqa: E402
from scamshield.config import ARTIFACTS_DIR, DATA_DIR, HOP, TARGET_SR  # noqa: E402
from scamshield.dsp.features import short_time_energy, yin_pitch, zcr  # noqa: E402
from scamshield.eval.metrics import roc_auc, roc_curve  # noqa: E402
from scamshield.fingerprint.hashing import make_hashes  # noqa: E402
from scamshield.fingerprint.index import FingerprintIndex  # noqa: E402
from scamshield.fingerprint.matcher import offset_histogram  # noqa: E402
from scamshield.fingerprint.peaks import find_peaks  # noqa: E402
from scamshield.fingerprint.spectrogram import spectrogram  # noqa: E402

OUT = ROOT / "report" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)
R = ARTIFACTS_DIR / "results"
S = json.load(open(R / "summary.json", encoding="utf-8"))["splits"]
V = json.load(open(R / "dsp_validation.json", encoding="utf-8"))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"blue": "#2557d6", "red": "#d3243a", "green": "#0b8a57", "orange": "#c2570c", "grey": "#667088", "purple": "#7c3aed"}


def clips(split: str) -> list[dict]:
    return list(csv.DictReader(open(R / f"clips_{split}.csv", encoding="utf-8")))


def load(rel: str) -> np.ndarray:
    x, sr = load_audio(DATA_DIR / rel)
    return preprocess(x, sr)


# ------------------------------------------------------------------ Fig: signal analysis
def fig_signal() -> None:
    x = load("clean/scam_digital_arrest_en__enIN_Neerja__normal.wav")
    sr, dur = TARGET_SR, len(x) / TARGET_SR
    segs = vad(x, sr)
    fl, hop = int(0.025 * sr), int(0.010 * sr)
    e_db = 10 * np.log10(short_time_energy(x, fl, hop) + 1e-10)
    z = zcr(x, fl, hop)
    f0, _ = yin_pitch(x, sr)
    t = np.arange(len(e_db)) * hop / sr
    floor, top = np.percentile(e_db, 10), np.percentile(e_db, 95)
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 6.2), sharex=True)
    ax[0].plot(np.arange(len(x)) / sr, x, color=C["grey"], lw=0.4)
    for a, b in segs:
        ax[0].axvspan(a, b, color=C["green"], alpha=0.18, lw=0)
    ax[0].set_ylabel("amplitude"); ax[0].set_title("Waveform and detected speech (green)", fontsize=9, loc="left")
    ax[1].plot(t, e_db, color=C["blue"], lw=0.8, label="short-time energy")
    ax[1].axhline(floor + 0.3 * (top - floor), color=C["red"], ls="--", lw=1, label="VAD threshold")
    ax[1].axhline(floor, color=C["grey"], ls=":", lw=1, label="noise floor (10th percentile)")
    ax[1].set_ylabel("dB"); ax[1].legend(fontsize=7, ncol=3, loc="lower right"); ax[1].set_title("Short-time energy", fontsize=9, loc="left")
    ax[2].plot(t, z, color=C["orange"], lw=0.7); ax[2].axhline(0.25, color=C["red"], ls="--", lw=1)
    ax[2].set_ylabel("ZCR"); ax[2].set_title("Zero-crossing rate (dashed: fricative rule 0.25)", fontsize=9, loc="left")
    ax[3].plot(np.arange(len(f0)) * hop / sr, f0, ".", ms=1.5, color=C["purple"])
    ax[3].set_ylabel("F0 (Hz)"); ax[3].set_xlabel("time (s)"); ax[3].set_title("Pitch contour (YIN, own implementation)", fontsize=9, loc="left")
    ax[3].set_xlim(0, dur)
    fig.tight_layout(); fig.savefig(OUT / "fig_signal.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig: fingerprint (constellation + offset histogram)
def fig_fingerprint() -> None:
    idx = FingerprintIndex.load(ARTIFACTS_DIR / "fingerprint_index.pkl")
    clean = load("clean/scam_digital_arrest_en__enIN_Neerja__normal.wav")
    gsm = load("augmented/scam_digital_arrest_en__enIN_Neerja__normal__gsm.wav")
    t0, t1 = 5.0, 9.0
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.4))
    peaks_by = {}
    for k, (name, sig) in enumerate([("clean TTS recording (library)", clean), ("same recording after GSM 6.10 codec (query)", gsm)]):
        Sdb, freqs, times = spectrogram(sig, TARGET_SR)
        p = find_peaks(Sdb); peaks_by[name] = p
        a, b = int(t0 * TARGET_SR / HOP), int(t1 * TARGET_SR / HOP)
        ax[k].imshow(Sdb[:130, a:b], origin="lower", aspect="auto", cmap="magma", extent=[t0, t1, 0, freqs[129]], vmin=Sdb.max() - 80)
        sel = (p[:, 1] >= a) & (p[:, 1] < b)
        ax[k].plot(p[sel, 1] * HOP / TARGET_SR, freqs[p[sel, 0]], "o", ms=3.2, mfc="none", mec="cyan", mew=0.8)
        ax[k].set_ylabel("frequency (Hz)"); ax[k].set_title(f"Spectrogram and spectral peaks: {name}", fontsize=9, loc="left"); ax[k].grid(False)
        for fr in (300, 3400):
            ax[k].axhline(fr, color="white", ls="--", lw=0.6)
    ax[1].set_xlabel("time (s)")
    q = sig = gsm[int(5 * TARGET_SR): int(15 * TARGET_SR)]
    hashes = make_hashes(find_peaks(spectrogram(q, TARGET_SR)[0]))
    votes = idx.query(hashes)
    cid = "scam_digital_arrest_en__enIN_Neerja__normal"
    counts, lo = offset_histogram(votes[cid])
    other = max((offset_histogram(v)[0].max() for c, v in votes.items() if c != cid), default=0)
    xs = (np.arange(len(counts)) + lo) * HOP / TARGET_SR
    ax[2].bar(xs, counts, width=HOP / TARGET_SR, color=C["red"])
    ax[2].set_xlabel("offset t_library - t_query (s)"); ax[2].set_ylabel("aligned hashes")
    ax[2].set_title(f"Offset histogram of a 10 s GSM excerpt cut at 5 s: {int(counts.max())} hashes agree at {xs[np.argmax(counts)]:.2f} s "
                    f"(best other clip: {int(other)})", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_fingerprint.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig: aligned hashes by condition
def fig_aligned() -> None:
    rows = clips("dev") + clips("test")
    order = ["none", "tel", "gsm", "snr20", "snr10", "snr5", "spd95", "spd105"]
    label = {"none": "clean", "tel": "tel", "gsm": "GSM", "snr20": "20 dB", "snr10": "10 dB", "snr5": "5 dB", "spd95": "0.95x", "spd105": "1.05x"}
    pos = {a: [int(r["fp_aligned"]) for r in rows if r["is_library_audio"] == "True" and r["aug"] == a] for a in order}
    harm = [int(r["fp_aligned"]) for r in rows if r["is_library_audio"] == "False" and (r["label"] == "legit" or r["fp_script"] != r["script_id"])]
    thr = S["dev"]["layer1"]["threshold_aligned_hashes"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.boxplot([pos[a] for a in order], labels=[label[a] for a in order], showfliers=True, patch_artist=True,
               boxprops=dict(facecolor="#cfe0ff"), medianprops=dict(color=C["blue"]), flierprops=dict(markersize=2.5))
    ax.axhline(max(harm), color=C["orange"], ls=":", lw=1.2, label=f"worst legit / wrong-script negative ({max(harm)}, n={len(harm)})")
    ax.axhline(thr, color=C["red"], ls="--", lw=1.2, label=f"decision threshold ({thr}, = 2 x worst negative + 1)")
    ax.set_yscale("log"); ax.set_ylabel("aligned hashes (log)"); ax.set_xlabel("condition of the query clip")
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("Fingerprint evidence for exact library recordings under degradation (dev + test, n=10 per box)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_aligned.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig: ROC before / after
def fig_roc() -> None:
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for a, split, title in zip(ax, ["test", "holdout", "test2"], ["Test (library scripts)", "Holdout (hand-written)", "Test 2 (locked)"]):
        rows = clips(split)
        y = np.array([r["label"] == "scam" for r in rows])
        series = [("old: similarity to library", np.array([float(r["sim"]) for r in rows]), C["grey"]),
                  ("red-flag lexicon", np.array([float(r["redflags"]) for r in rows]), C["orange"]),
                  ("learned wording model", np.array([float(r["text_dec"]) for r in rows]), C["blue"]),
                  ("fused score", np.array([float(r["fused_score"]) for r in rows]), C["green"])]
        for name, s, col in series:
            fpr, tpr, _ = roc_curve(y, s)
            a.plot(fpr, tpr, color=col, lw=1.3, label=f"{name} ({roc_auc(y, s):.2f})")
        a.plot([0, 1], [0, 1], ":", color="k", lw=0.7); a.set_title(title, fontsize=9); a.set_xlabel("false positive rate")
        a.legend(fontsize=5.6, loc="lower right")
    ax[0].set_ylabel("true positive rate")
    fig.tight_layout(); fig.savefig(OUT / "fig_roc.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig: test 2 by scam type
def fig_by_type() -> None:
    rows = [r for r in clips("test2") if r["label"] == "scam"]
    types = ["digital_arrest", "fake_courier", "kyc_otp", "lottery_prize", "relative_emergency", "tech_support"]
    names = ["digital\narrest", "fake\ncourier", "KYC /\nOTP", "lottery\nprize", "relative in\ntrouble", "tech\nsupport"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.38
    for k, (lang, col, lab) in enumerate([("en", C["blue"], "English"), ("hi", C["orange"], "Hinglish")]):
        vals, ns = [], []
        for t in types:
            g = [r for r in rows if r["type"] == t and r["lang"] == lang]
            ns.append(len(g)); vals.append(np.mean([r["level"] in ("high", "scam") for r in g]) if g else np.nan)
        xs = np.arange(len(types)) + (k - 0.5) * w
        ax.bar(xs, [0 if np.isnan(v) else v for v in vals], width=w, color=col, label=lab)
        for x_, v, n in zip(xs, vals, ns):
            ax.text(x_, (0 if np.isnan(v) else v) + 0.02, "n/a" if np.isnan(v) else f"{v*100:.0f}%", ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(len(types))); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0, 1.12); ax.set_ylabel("share flagged (high or scam)")
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("Test 2: share of scam clips flagged, by scam type and language", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_by_type.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig: live risk timelines
def fig_live() -> None:
    from scamshield.api.pipeline import Models
    from scamshield.api.session import simulate
    from scamshield.eval.run_eval import CachedSTT
    models = Models().load(warm_stt=False)
    cache = CachedSTT(prefix="live_")
    cases = [("replayed scam recording (GSM)", "augmented/scam_digital_arrest_en__enIN_Neerja__normal__gsm.wav", C["red"]),
             ("reworded scam, new voice", "holdout/clean/holdout_scam_kyc_otp_en__en_GB_Sonia__normal.wav", C["orange"]),
             ("genuine hospital call", "holdout/clean/holdout_legit_hospital_en__en_US_Aria__normal.wav", C["blue"]),
             ("genuine bank alert", "clean/legit_bank_alert_en__enUS_Guy__normal.wav", C["green"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for lo_, hi_, c in [(0, 30, "#2ecc71"), (30, 60, "#f1c40f"), (60, 85, "#e67e22"), (85, 100, "#e74c3c")]:
        ax.axhspan(lo_, hi_, color=c, alpha=0.08, lw=0)
    for name, rel, col in cases:
        cache.file = Path(rel).name
        x, sr = load_audio(DATA_DIR / rel)
        sess, ev = simulate(models, preprocess(x, sr), transcribe_fn=cache)
        rs = [e["state"] for e in ev if e["type"] == "risk"]
        ax.plot([r["t_s"] for r in rs], [r["score"] for r in rs], "-o", ms=2.5, lw=1.4, color=col, label=name)
        al = next((e for e in ev if e["type"] == "alert"), None)
        if al:
            ax.axvline(al["t_s"], color=col, ls=":", lw=0.9)
            ax.annotate("alert", (al["t_s"], 97), fontsize=6.5, color=col, ha="center")
    ax.set_xlabel("time since the call began (s)"); ax.set_ylabel("risk score"); ax.set_ylim(0, 102)
    ax.legend(fontsize=7, loc="center right"); ax.set_title("Live risk over time (simulated real-time streaming; dotted: first alert)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_live.png"); plt.close(fig)


# ------------------------------------------------------------------ LaTeX tables
def pc(v: float, d: int = 1) -> str:
    return (f"{v * 100:.{d}f}").rstrip("0").rstrip(".") + r"\%" if d else f"{v * 100:.0f}\\%"


def tables() -> None:
    L: list[str] = []
    # -- splits
    desc = {"dev": ("Dev", "library scripts, 2 voices, 8 versions", "calibrate fingerprint threshold"),
            "test": ("Test", "library scripts, 1 other voice", "regression check, run once"),
            "holdout": ("Holdout", "18 hand-written scripts, 4 unseen voices", "second look after diagnosis"),
            "dev2": ("Dev 2", "generated corpus (dev), TTS $\\rightarrow$ Whisper", "calibrate wording-model range"),
            "test2": ("Test 2", "generated by another model, unseen voices", "locked final test, run once")}
    L.append("\\newcommand{\\tabsplits}{\\begin{tabular}{@{}llrrrl@{}}\\toprule Split & Content & Clips & Scam & Legit & Purpose \\\\ \\midrule")
    for k in ["dev", "test", "holdout", "dev2", "test2"]:
        s = S[k]; n, c, p = desc[k]
        L.append(f"{n} & {c} & {s['n_clips']} & {s['n_scam']} & {s['n_legit']} & {p} \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- DSP validation
    rows = V["clips"]
    L.append("\\newcommand{\\tabdsp}{\\begin{tabular}{@{}lrrrrr@{}}\\toprule Clip & STFT max err (dB) & MFCC median err & MFCC min corr. & Voiced frames & F0 within 5\\% \\\\ \\midrule")
    for i, r in enumerate(rows, 1):
        L.append(f"C{i} & {r['stft_max_err_db']:.1e} & {r['mfcc_median_abs_err']:.1e} & {r['mfcc_min_corr']:.6f} & {r['voiced_frames']} & {r['f0_within_5pct']*100:.1f}\\% \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- fingerprint by condition (dev + test pooled)
    rws = clips("dev") + clips("test")
    order = [("none", "clean"), ("tel", "telephone band"), ("gsm", "GSM codec"), ("snr20", "SNR 20 dB"), ("snr10", "SNR 10 dB"), ("snr5", "SNR 5 dB"),
             ("spd95", "speed 0.95"), ("spd105", "speed 1.05")]
    thr = S["dev"]["layer1"]["threshold_aligned_hashes"]
    L.append("\\newcommand{\\tabfp}{\\begin{tabular}{@{}lrrrrr@{}}\\toprule Condition & $n$ & Recall@1 & Median aligned & Min aligned & Share $\\geq$ threshold \\\\ \\midrule")
    for a, name in order:
        g = [r for r in rws if r["is_library_audio"] == "True" and r["aug"] == a]
        v = np.array([int(r["fp_aligned"]) for r in g]); ok = np.mean([r["fp_script"] == r["script_id"] and int(r["fp_aligned"]) >= thr for r in g])
        L.append(f"{name} & {len(g)} & {ok*100:.0f}\\% & {int(np.median(v))} & {int(v.min())} & {np.mean(v >= thr)*100:.0f}\\% \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- main results
    L.append("\\newcommand{\\tabmain}{\\begin{tabular}{@{}lrrrrrr@{}}\\toprule Split & Clips & Scams caught & False alarms & Precision & AUC (en / hi) & Fusion AUC \\\\ \\midrule")
    for k, name in [("test", "Test"), ("holdout", "Holdout"), ("test2", "Test 2")]:
        s = S[k]; f = s["fusion"]; bl = s["layer2"]["by_language"]
        L.append(f"{name} & {s['n_clips']} & {f['recall']*100:.1f}\\% ({f['tp']}/{f['tp']+f['fn']}) & {f['fpr_legit']*100:.1f}\\% ({f['fp']}/{f['fp']+f['tn']}) & {f['precision']:.3f} & "
                 f"{bl['en']['auc_text_model']:.2f} / {bl['hi']['auc_text_model']:.2f} & {f['auc']:.3f} \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- ablation on test2
    L.append("\\newcommand{\\tabablation}{\\begin{tabular}{@{}lrrrrr@{}}\\toprule Configuration & Precision & Recall & F1 & False alarms & AUC \\\\ \\midrule")
    names = {"fp only": "fingerprint only", "script only": "wording model only", "flags only": "red-flag lexicon only", "voice only": "voice cues only", "full": "full fusion"}
    for a in S["test2"]["ablation"]:
        L.append(f"{names[a['config']]} & {a['precision']:.3f} & {a['recall']:.3f} & {a['f1']:.3f} & {a['fpr_legit']*100:.1f}\\% & {a['auc']:.3f} \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- live
    L.append("\\newcommand{\\tablive}{\\begin{tabular}{@{}lrrrrr@{}}\\toprule Split & Scams alerted & Legit alerted & Replay alert & Latency p50 / p95 & Within 5 s \\\\ \\midrule")
    for k, name in [("test", "Test"), ("holdout", "Holdout"), ("test2", "Test 2")]:
        l = S[k]["live"]; fa = l["fingerprint_alerts"]; wo = l["words_only"]
        rep = f"{fa['first_alert_t_s_p50']:.0f} s" if fa["n"] else "n/a"
        L.append(f"{name} & {l['scam_alerted']*100:.0f}\\% & {l['legit_alert_rate']*100:.0f}\\% & {rep} & {wo['latency_s_p50']:.1f} / {wo['latency_s_p95']:.1f} s & {wo['share_within_5s']*100:.0f}\\% \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    # -- latency
    t = {x["stage"]: x for x in S["test"]["timings"]}
    stages = [("decode", "decode"), ("preprocess", "resample, normalise"), ("vad", "voice activity detection"), ("stft", "STFT"), ("peaks_hashes", "peaks and hashes"),
              ("fingerprint_match", "index lookup and scoring"), ("stt", "speech-to-text (Whisper base)"), ("script_match", "closest known scripts"),
              ("red_flags", "red-flag lexicon"), ("voice_cues", "pitch, energy, pauses"), ("fusion", "wording model, risk timeline, fusion"), ("viz", "visualisation payload"), ("total", "total")]
    L.append("\\newcommand{\\tablatency}{\\begin{tabular}{@{}lrr@{}}\\toprule Stage & Median (ms) & 95th percentile (ms) \\\\ \\midrule")
    for k, name in stages[:-1]:
        L.append(f"{name} & {t[k]['p50_ms']:.0f} & {t[k]['p95_ms']:.0f} \\\\")
    # the stored 'total' excludes Whisper (transcripts were cached), so report the sum of the stage medians instead
    total = sum(t[k]["p50_ms"] for k, _ in stages[:-1])
    L.append(f"\\midrule sum of the stage medians & {total:.0f} & -- \\\\")
    L.append("\\bottomrule\\end{tabular}}")
    (ROOT / "report" / "generated_tables.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for fn in (fig_signal, fig_fingerprint, fig_aligned, fig_roc, fig_by_type, tables, fig_live):
        print("running", fn.__name__, flush=True)
        fn()
    print("done ->", OUT)
