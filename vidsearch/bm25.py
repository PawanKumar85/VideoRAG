"""Okapi BM25 over an inverted index, implemented from scratch.

Score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (the +1 keeps it non-negative)

Why it earns a place next to dense vectors: embeddings blur rare exact tokens
(product names, error codes, IDs); BM25 weights them heavily via idf.

Complexity: a query touches only the posting lists of its own terms, so it is
O(sum of posting lengths), not O(corpus size).
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from vidsearch.text import content_tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._n = 0

    def build(self, docs: list[str]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len: list[int] = []
        for doc_id, text in enumerate(docs):
            tokens = content_tokens(text)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_id, tf))
        self._postings = dict(postings)
        self._doc_len = doc_len
        self._n = len(docs)
        self._avgdl = (sum(doc_len) / self._n) if self._n else 0.0

    def _idf(self, df: int) -> float:
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_id, score) pairs, best first; empty if nothing matches."""
        if not self._n or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(content_tokens(query)):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = self._idf(len(plist))
            for doc_id, tf in plist:
                norm = 1.0 - self.b + self.b * (self._doc_len[doc_id] / self._avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        # Deterministic: highest score first, ties by lower doc id.
        return heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
