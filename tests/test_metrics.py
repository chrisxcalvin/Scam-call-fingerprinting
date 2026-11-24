import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scamshield.eval.metrics import false_positive_rate, precision_recall_f1, roc_auc, roc_curve, wer


def test_prf1_and_fpr():
    y, p = np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1])
    assert precision_recall_f1(y, p) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert false_positive_rate(y, p) == 0.5


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); s = rng.normal(size=200) + y
    s[:20] = np.round(s[:20])                       # include ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    assert fpr[0] == 0 and tpr[-1] == 1 and np.all(np.diff(fpr) >= 0)


def test_wer():
    assert wer("the cat sat", "the cat sat") == 0
    assert wer("the cat sat", "the bat sat down") == pytest.approx(2 / 3)
