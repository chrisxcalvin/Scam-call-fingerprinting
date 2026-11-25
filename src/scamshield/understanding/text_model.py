"""Learned scam-likeness of a transcript (Layer 2). Owner B.

A logistic regression on TF-IDF features of the transcript:
    x = [ word 1-2-gram TF-IDF  |  character 3-5-gram (word-bounded) TF-IDF ]
    d(text) = w . x + b          (decision value; larger = more scam-like)
Word n-grams capture phrases ("do not tell anyone", "read out the code"); character n-grams make it robust to the
spelling variation of romanised Hinglish and to small speech-to-text errors. Trained on the generated corpus
(scripts/corpus/train.json) of scam and legitimate scripts, so honest calls that use risky-sounding words
(bank, OTP, police) are learned as legitimate. The decision value is mapped to a 0..1 fusion component with a
range calibrated on dev (fusion_config.json: `text_range`).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "text_model.pkl"


@dataclass
class TextModel:
    word: TfidfVectorizer
    char: TfidfVectorizer
    clf: LogisticRegression
    meta: dict

    @classmethod
    def fit(cls, texts: list[str], is_scam: list[bool], C: float = 10.0, meta: dict | None = None) -> "TextModel":
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, lowercase=True)
        X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
        clf = LogisticRegression(C=C, max_iter=5000).fit(X, np.asarray(is_scam, dtype=int))
        return cls(word, char, clf, {**(meta or {}), "C": C, "n_train": len(texts)})

    def decision(self, texts: list[str]) -> np.ndarray:
        """Decision values d(text); larger = more scam-like. Empty text -> a very negative value."""
        texts = [t if t.strip() else "." for t in texts]
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.decision_function(X)

    def top_features(self, n: int = 15) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Highest-weight word n-grams for scam and for legit (for the viva: what did it learn?)."""
        names = np.array(self.word.get_feature_names_out())
        w = self.clf.coef_[0][: len(names)]
        order = np.argsort(w)
        return ([(names[i], float(w[i])) for i in order[::-1][:n]], [(names[i], float(w[i])) for i in order[:n]])

    def save(self, path: str | Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "TextModel":
        """Load a model you trained yourself (pickle executes code: never load files from others)."""
        with open(path, "rb") as f:
            return pickle.load(f)


def to_component(d: float, text_range: tuple[float, float]) -> float:
    """Map a decision value to 0..1: clip((d - lo) / (hi - lo), 0, 1). lo/hi come from dev calibration."""
    lo, hi = text_range
    return round(float(min(1.0, max(0.0, (d - lo) / (hi - lo)))), 4)
