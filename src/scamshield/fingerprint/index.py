"""Inverted fingerprint index (Owner A)."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path


class FingerprintIndex:
    """Inverted index: hash -> list of (clip_id, anchor_time_frame) in the library."""

    def __init__(self) -> None:
        self.table: dict[int, list[tuple[str, int]]] = defaultdict(list)
        self.clip_hash_counts: dict[str, int] = {}

    def add(self, clip_id: str, hashes: list[tuple[int, int]]) -> None:
        """Insert all hashes of a library clip."""
        for h, t in hashes:
            self.table[h].append((clip_id, t))
        self.clip_hash_counts[clip_id] = self.clip_hash_counts.get(clip_id, 0) + len(hashes)

    def save(self, path: str | Path) -> None:
        """Pickle the index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"table": dict(self.table), "counts": self.clip_hash_counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FingerprintIndex":
        """Load a pickled index (only load artifacts you built yourself: pickle runs code)."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls()
        idx.table.update(d["table"])
        idx.clip_hash_counts = d["counts"]
        return idx

    def query(self, hashes: list[tuple[int, int]]) -> dict[str, list[int]]:
        """Offset votes: for each query hash found in the table, offset = t_db - t_query (frames).

        A true match makes many hashes agree on the same offset (the clip is just shifted in time);
        random collisions spread their offsets out.
        """
        out: dict[str, list[int]] = defaultdict(list)
        for h, tq in hashes:
            for clip_id, tdb in self.table.get(h, ()):
                out[clip_id].append(tdb - tq)
        return dict(out)

    def __len__(self) -> int:
        return sum(self.clip_hash_counts.values())
