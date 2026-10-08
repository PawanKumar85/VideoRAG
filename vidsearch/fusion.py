"""Reciprocal Rank Fusion.

    score(d) = sum over rankings r of  w_r / (k + rank_r(d))      rank is 1-based

Why RRF and not a weighted sum of raw scores: BM25 scores are unbounded and
corpus-dependent, cosine similarity lives in [-1, 1]; the two are not on a common
scale and any fixed weight is fragile. Ranks are scale-free. The constant k
(60 in the original paper) damps the influence of the very top ranks so a single
list cannot dominate on one lucky first place.
"""

from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked lists of doc ids (best first). Returns (doc_id, score), best first."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    fused: dict[int, float] = {}
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        seen: set[int] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # defensive: a duplicate must not be counted twice
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
