"""Layer-2 model selection on TEXT, train -> dev only (test2 is never read here).
Compares: baseline (original 10-script library) vs expanded library vs +legit negatives vs TF-IDF classifier vs stacked.
Usage: python scripts/experiment_l2.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scamshield.eval.metrics import roc_auc  # noqa: E402
from scamshield.schemas import TranscriptSegment  # noqa: E402
from scamshield.understanding.redflags import red_flags, redflag_score  # noqa: E402
from scamshield.understanding.script_match import embed  # noqa: E402

load = lambda s: json.load(open(ROOT / "scripts" / "corpus" / f"{s}.json", encoding="utf-8"))
train, dev = load("train") + load("train_direct"), load("dev")
orig = json.load(open(ROOT / "data" / "scripts.json", encoding="utf-8"))
y = lambda d: np.array([x["label"] == "scam" for x in d])


def tpr_at_fpr(yt, s, fpr=0.05):
    thr = np.percentile(s[~yt], 100 * (1 - fpr))
    return float(np.mean(s[yt] > thr))


def report(name, yt, s, langs):
    row = f"{name:34s} AUC {roc_auc(yt, s):.3f}  TPR@5%FPR {tpr_at_fpr(yt, s, .05):.2f}  TPR@1%FPR {tpr_at_fpr(yt, s, .01):.2f}"
    for lg in ("en", "hi"):
        m = langs == lg
        row += f" | {lg}: AUC {roc_auc(yt[m], s[m]):.3f}"
    print(row)


ytr, ydv = y(train), y(dev)
ldv = np.array([x["lang"] for x in dev])
Etr, Edv = embed([x["text"] for x in train]), embed([x["text"] for x in dev])
Eor = embed([x["text"] for x in orig if x["label"] == "scam"])

print(f"train {len(train)} ({ytr.sum()} scam)  dev {len(dev)} ({ydv.sum()} scam)\n")
report("baseline: orig 10 scripts, max sim", ydv, (Edv @ Eor.T).max(1), ldv)
S = Edv @ Etr[ytr].T
report("expanded scam library, max sim", ydv, S.max(1), ldv)
report("expanded scam library, top-3 mean", ydv, np.sort(S, 1)[:, -3:].mean(1), ldv)
N = Edv @ Etr[~ytr].T
report("margin: scam max - legit max", ydv, S.max(1) - N.max(1), ldv)
k = 5
report("margin: top-5 mean scam - legit", ydv, np.sort(S, 1)[:, -k:].mean(1) - np.sort(N, 1)[:, -k:].mean(1), ldv)

# embedding + logistic regression directly on the 384-d vectors
for C in (0.3, 1, 3, 10):
    lr = LogisticRegression(C=C, max_iter=2000).fit(Etr, ytr)
    report(f"LR on embeddings (C={C})", ydv, lr.decision_function(Edv), ldv)

# TF-IDF word + char n-grams
def tfidf_model(C):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, lowercase=True)
    Xtr = hstack([w.fit_transform([x["text"] for x in train]), c.fit_transform([x["text"] for x in train])]).tocsr()
    Xdv = hstack([w.transform([x["text"] for x in dev]), c.transform([x["text"] for x in dev])]).tocsr()
    return LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr), Xdv
for C in (1, 3, 10, 30):
    m, Xdv = tfidf_model(C)
    report(f"TF-IDF word+char LR (C={C})", ydv, m.decision_function(Xdv), ldv)

# lexicon score alone (existing red flags)
lex = np.array([redflag_score(red_flags([TranscriptSegment(start_s=0, end_s=1, text=x["text"])])) for x in dev])
report("existing red-flag lexicon", ydv, lex, ldv)


# ---- shift probe: the 17 hand-written scripts of the original dataset (different author; never used for training)
print("\n== shift probe: original 17 hand-written scripts (10 scam, 7 legit), models trained on train only")
yo = np.array([x["label"] == "scam" for x in orig]); lo_ = np.array([x["lang"] for x in orig])
Eo = embed([x["text"] for x in orig])
So, No = Eo @ Etr[ytr].T, Eo @ Etr[~ytr].T
report("margin: top-5 mean scam - legit", yo, np.sort(So, 1)[:, -5:].mean(1) - np.sort(No, 1)[:, -5:].mean(1), lo_)
lr = LogisticRegression(C=10, max_iter=2000).fit(Etr, ytr); report("LR on embeddings (C=10)", yo, lr.decision_function(Eo), lo_)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
Xtr = hstack([w.fit_transform([x["text"] for x in train]), c.fit_transform([x["text"] for x in train])]).tocsr()
Xo = hstack([w.transform([x["text"] for x in orig]), c.transform([x["text"] for x in orig])]).tocsr()
tf = LogisticRegression(C=10, max_iter=3000).fit(Xtr, ytr); report("TF-IDF word+char LR (C=10)", yo, tf.decision_function(Xo), lo_)
Xdv = hstack([w.transform([x["text"] for x in dev]), c.transform([x["text"] for x in dev])]).tocsr()
z = lambda a, ref: (a - ref.mean()) / (ref.std() + 1e-9)
mg_dev = np.sort(S, 1)[:, -5:].mean(1) - np.sort(N, 1)[:, -5:].mean(1); mg_o = np.sort(So, 1)[:, -5:].mean(1) - np.sort(No, 1)[:, -5:].mean(1)
tf_dev, tf_o = tf.decision_function(Xdv), tf.decision_function(Xo)
report("avg z(margin)+z(tfidf) [dev]", ydv, z(mg_dev, mg_dev) + z(tf_dev, tf_dev), ldv)
report("avg z(margin)+z(tfidf) [orig17]", yo, z(mg_o, mg_dev) + z(tf_o, tf_dev), lo_)
print("orig17 per-script scores (margin | tfidf):")
for x, a, b in zip(orig, mg_o, tf_o): print(f"  {x['label']:5s} {x['id']:26s} {a:+.3f} | {b:+.2f}")
