"""Exact vector index over L2-normalised vectors, with optional int8 quantisation.

Memory math (float32): bytes = N * d * 4.
    1M chunks * 1536 dims * 4 B  ~= 6.1 GB
    int8 scalar quantisation      ~= 1.5 GB  (4x smaller, plus 4 B/vector for the scale)

Search is brute force: one matrix-vector product, O(N*d) multiply-adds. That is
the *correct* choice at this project's scale (tens of thousands of chunks: a few
milliseconds) and gives 100% recall, which makes it the ground truth against which
an approximate index (HNSW) would be measured. Past roughly a million vectors,
switch to an ANN index (HNSW in Qdrant/pgvector/FAISS) and measure recall@k vs
this exact baseline.

Quantisation here is symmetric, per vector:  q = round(v / s),  s = max|v| / 127.
The dot product is recovered as (q . query) * s. Rows are dequantised in blocks so
query-time memory stays small instead of materialising a full float32 copy.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 8192


class VectorIndex:
    def __init__(self, vectors: np.ndarray, quantize: bool = False) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError("vectors must be a 2-D array (n, dim)")
        self.n, self.dim = v.shape
        self.quantize = quantize
        self._m: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        if quantize:
            scale = np.abs(v).max(axis=1) / 127.0 if self.n else np.zeros(0, np.float32)
            scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
            self._q = np.round(v / scale[:, None]).astype(np.int8)
            self._scale = scale
        else:
            self._m = v

    @property
    def nbytes(self) -> int:
        if self.quantize:
            return int(self._q.nbytes + self._scale.nbytes)
        return int(self._m.nbytes)

    def scores(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query must have shape ({self.dim},), got {q.shape}")
        if not self.quantize:
            return self._m @ q
        out = np.empty(self.n, dtype=np.float32)
        for s in range(0, self.n, _BLOCK):
            blk = self._q[s : s + _BLOCK].astype(np.float32)
            out[s : s + _BLOCK] = (blk @ q) * self._scale[s : s + _BLOCK]
        return out

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (row index, cosine) pairs, best first, ties broken by lower index."""
        k = min(k, self.n)
        if k <= 0:
            return []
        s = self.scores(query)
        # argpartition is O(N) to find the top-k set; only those k are then sorted.
        top = np.argpartition(-s, k - 1)[:k] if k < self.n else np.arange(self.n)
        order = top[np.lexsort((top, -s[top]))]
        return [(int(i), float(s[i])) for i in order]
