"""Evaluation metrics (Owner B). Plain NumPy so every number can be explained."""
from __future__ import annotations

import numpy as np


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Binary P/R/F1 with scam = 1.  P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R)."""
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred)); fn = int(np.sum(y_true & ~y_pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN): share of legit calls wrongly flagged."""
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    neg = int(np.sum(~y_true))
    return float(np.sum(~y_true & y_pred) / neg) if neg else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC = P(score of a random positive > score of a random negative), ties count 1/2
    (Mann-Whitney U / (n_pos * n_neg)). Equals the area under the ROC curve."""
    y = np.asarray(y_true).astype(bool); s = np.asarray(scores, dtype=float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(FPR, TPR, thresholds) for every distinct score used as threshold (score >= thr -> positive)."""
    y = np.asarray(y_true).astype(bool); s = np.asarray(scores, dtype=float)
    thr = np.r_[np.inf, np.unique(s)[::-1]]
    tpr = np.array([np.mean(s[y] >= t) if y.any() else 0 for t in thr])
    fpr = np.array([np.mean(s[~y] >= t) if (~y).any() else 0 for t in thr])
    return fpr, tpr, thr


def wer(ref: str, hyp: str) -> float:
    """Word error rate (S + D + I) / N via Levenshtein distance on lower-cased words (punctuation removed)."""
    import re
    r = re.findall(r"[a-z0-9']+", ref.lower()); h = re.findall(r"[a-z0-9']+", hyp.lower())
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d = d, np.empty(len(h) + 1, dtype=int)
        d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1] / max(1, len(r)))
