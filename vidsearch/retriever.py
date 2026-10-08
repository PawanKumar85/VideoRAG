"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from vidsearch.bm25 import BM25Index
from vidsearch.embeddings import Embedder
from vidsearch.fusion import reciprocal_rank_fusion
from vidsearch.index import VectorIndex
from vidsearch.models import Chunk, Hit
from vidsearch.reranker import Reranker

Mode = Literal["hybrid", "dense", "bm25"]
MODES: tuple[str, ...] = ("hybrid", "dense", "bm25")


def suppress_overlaps(hits: list[Hit], max_overlap: float = 0.5) -> list[Hit]:
    """Drop a lower-ranked hit if it largely overlaps a better hit from the same video.

    Chunk overlap and near-duplicate segments make neighbouring chunks both rank
    highly; showing both wastes result slots and (in RAG) context tokens.
    ``overlap`` is intersection / duration of the shorter chunk.
    """
    kept: list[Hit] = []
    for hit in hits:
        c = hit.chunk
        duplicate = False
        for other in kept:
            o = other.chunk
            if o.video_id != c.video_id:
                continue
            inter = min(c.end, o.end) - max(c.start, o.start)
            shorter = max(1e-9, min(c.end - c.start, o.end - o.start))
            if inter > 0 and inter / shorter > max_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(hit)
    return kept


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        rerank_top_n: int = 20,
        rrf_k: int = 60,
        candidate_k: int = 50,
        quantize: bool = False,
        max_overlap: float = 0.5,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = list(chunks)
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.max_overlap = max_overlap
        self.index = VectorIndex(vectors, quantize=quantize)
        self.bm25 = BM25Index()
        self.bm25.build([c.text for c in self.chunks])

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        rerank: bool = True,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.chunks or not query.strip():
            return []

        qv = self.embedder.encode([query])[0]
        has_dense = bool(np.any(qv))  # an all-stopword query embeds to the zero vector

        dense = self.index.search(qv, self.candidate_k) if mode != "bm25" and has_dense else []
        lexical = self.bm25.search(query, self.candidate_k) if mode != "dense" else []

        dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
        lex_rank = {i: r for r, (i, _) in enumerate(lexical, start=1)}

        if mode == "dense":
            ordered = [(i, s) for i, s in dense]
        elif mode == "bm25":
            ordered = [(i, s) for i, s in lexical]
        else:
            ordered = reciprocal_rank_fusion(
                [[i for i, _ in dense], [i for i, _ in lexical]], k=self.rrf_k
            )

        # Pull candidate pool for reranking and overlap suppression
        pool_size = max(self.rerank_top_n, k * 3) if (self.reranker and rerank) else max(k * 3, k)
        pool = ordered[:pool_size]

        rerank_scores: dict[int, float] = {}
        if self.reranker is not None and rerank and pool:
            top_n = min(len(pool), self.rerank_top_n)
            to_rerank = pool[:top_n]
            rest = pool[top_n:]
            texts = [self.chunks[idx].text for idx, _ in to_rerank]
            scores = self.reranker.score(query, texts)
            for (idx, _), s in zip(to_rerank, scores):
                rerank_scores[idx] = float(s)
            # Sort reranked candidates by reranker score descending, tie-breaking by initial position
            reranked_with_pos = [
                (idx, rerank_scores[idx], pos)
                for pos, (idx, _) in enumerate(to_rerank)
            ]
            reranked_with_pos.sort(key=lambda t: (-t[1], t[2]))
            ordered = [(idx, s) for idx, s, _ in reranked_with_pos] + rest
        else:
            ordered = pool

        hits: list[Hit] = []
        for idx, score in ordered:
            # Exact float32 cosine from the master vectors, regardless of which
            # retriever surfaced the chunk (also corrects quantisation error).
            cos = float(self.vectors[idx] @ qv) if has_dense else None
            rr_score = rerank_scores.get(idx)
            hits.append(
                Hit(
                    chunk=self.chunks[idx],
                    score=float(rr_score if rr_score is not None else score),
                    dense_score=cos,
                    dense_rank=dense_rank.get(idx),
                    lexical_rank=lex_rank.get(idx),
                    rerank_score=rr_score,
                )
            )
        return suppress_overlaps(hits, self.max_overlap)[:k]
