"""Offset-histogram plot for the report / viva (Owner A): the single most convincing fingerprint visual."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import HOP, TARGET_SR
from .index import FingerprintIndex
from .matcher import offset_histogram


def plot_offset_histograms(hashes: list[tuple[int, int]], index: FingerprintIndex, path: str | Path,
                           title: str = "", max_clips: int = 3) -> Path:
    """Bar plot of offset votes (t_db - t_query, in seconds) for the clips with most votes.

    True match: one tall spike (many hashes agree on one offset). Other clips: low, flat background.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    votes = index.query(hashes)
    clips = sorted(votes, key=lambda c: -np.max(offset_histogram(votes[c])[0]))[:max_clips]
    fig, axes = plt.subplots(len(clips) or 1, 1, figsize=(8, 2.2 * max(1, len(clips))), squeeze=False)
    frame_s = HOP / TARGET_SR
    for ax, c in zip(axes[:, 0], clips):
        counts, lo = offset_histogram(votes[c])
        ax.bar((np.arange(len(counts)) + lo) * frame_s, counts, width=frame_s, color="#e74c3c" if c == clips[0] else "#7f8c8d")
        ax.set_title(f"{c}  (peak = {counts.max()} aligned hashes)", fontsize=9)
        ax.set_ylabel("votes")
    axes[-1, 0].set_xlabel("offset t_db - t_query (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
