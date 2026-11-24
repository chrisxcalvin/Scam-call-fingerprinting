"""Embedding similarity vs the scam-script library (Owner B)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import ARTIFACTS_DIR, get_settings
from ..schemas import ScriptMatch

log = logging.getLogger(__name__)
CHUNK_SENTENCES = 2


@lru_cache(maxsize=1)
def get_embedder():
    """Multilingual sentence embedder (paraphrase-multilingual-MiniLM-L12-v2, 384-d)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (n, d); cosine similarity is then a plain dot product."""
    return np.asarray(get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False))


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? (keeps it simple and language-agnostic for romanised Hinglish)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def windows(sentences: list[str], k: int = CHUNK_SENTENCES) -> list[str]:
    """Sliding windows of k consecutive sentences (stride 1); a short text is one window."""
    if len(sentences) <= k:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i:i + k]) for i in range(len(sentences) - k + 1)]


@dataclass
class ScriptLibrary:
    chunks: list[dict]          # {script_id, scam_type, text}
    emb: np.ndarray             # (n_chunks, d), normalised

    @classmethod
    def build(cls, scripts: list[dict]) -> "ScriptLibrary":
        """Chunk every *scam* script into 2-sentence windows and embed them."""
        chunks = [{"script_id": s["id"], "scam_type": s["type"], "text": w}
                  for s in scripts if s["label"] == "scam" for w in windows(split_sentences(s["text"]))]
        return cls(chunks, embed([c["text"] for c in chunks]))

    @classmethod
    def load(cls) -> "ScriptLibrary":
        meta = json.loads((ARTIFACTS_DIR / "scripts_meta.json").read_text(encoding="utf-8"))
        return cls(meta["chunks"], np.load(ARTIFACTS_DIR / "script_embeddings.npy"))


@lru_cache(maxsize=1)
def default_library() -> ScriptLibrary:
    return ScriptLibrary.load()


def script_match(transcript_text: str, library: ScriptLibrary | None = None) -> list[ScriptMatch]:
    """Best cosine similarity per scam script.

    Query = 2-sentence sliding windows of the transcript; S = Q @ L^T (cos = a.b/(|a||b|), both
    normalised). For each script_id: max over its chunks and all query windows. Negative cosines
    are clipped to 0. Sorted by similarity, one entry per script_id.
    """
    lib = library or default_library()
    qw = windows(split_sentences(transcript_text))
    if not qw:
        return []
    sims = embed(qw) @ lib.emb.T                         # (n_query, n_chunks)
    best: dict[str, ScriptMatch] = {}
    for j, c in enumerate(lib.chunks):
        i = int(np.argmax(sims[:, j]))
        s = float(max(0.0, sims[i, j]))
        if c["script_id"] not in best or s > best[c["script_id"]].similarity:
            best[c["script_id"]] = ScriptMatch(script_id=c["script_id"], scam_type=c["scam_type"],
                                               similarity=round(s, 4), matched_text=c["text"], query_text=qw[i])
    return sorted(best.values(), key=lambda m: -m.similarity)
