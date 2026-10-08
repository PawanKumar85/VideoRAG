"""Embedders. Every embedder returns L2-normalised float32 vectors.

Normalising once at write time means cosine similarity == dot product, so a query
is a single matrix-vector product (BLAS/SIMD) with no per-row norm computation.

Two implementations:

* HashingEmbedder: offline, deterministic, no model download. It is a feature-hashing
  bag of words / bigrams / character trigrams, so it captures lexical overlap and
  morphology ("refund"/"refunds"), NOT meaning. It exists so tests, CI and a first
  run work without a model, and so the evaluation can show what a real model adds.
* SentenceTransformerEmbedder: real semantic embeddings (optional dependency).

The embedder's ``name`` is stored with the index. Vectors from different models live
in different spaces and are not comparable, so loading an index with a different
embedder is refused (see store.EmbedderMismatchError).
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from vidsearch.text import content_tokens


class Embedder(Protocol):
    name: str
    dim: int
    default_min_score: float

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # all-zero rows (empty text) stay zero instead of NaN
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing into a fixed-size vector.

    A feature (word, bigram, char trigram) is mapped to a bucket by CRC32 and added
    with a +/-1 sign taken from another bit. The random sign makes collisions cancel
    in expectation instead of piling up. CRC32 is used rather than Python's built-in
    hash(), which is salted per process and would make saved vectors unreadable
    after a restart.
    """

    default_min_score = 0.12

    def __init__(self, dim: int = 512, char_weight: float = 0.35) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_weight = char_weight
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        words = content_tokens(text)
        for w in words:
            feats[("w", w)] += 1
            if len(w) >= 4:
                padded = f"#{w}#"
                for i in range(len(padded) - 2):
                    feats[("c", padded[i : i + 3])] += 1
        for a, b in zip(words, words[1:]):
            feats[("b", f"{a}_{b}")] += 1
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for (kind, feat), count in self._features(text).items():
                h = zlib.crc32(f"{kind}:{feat}".encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                weight = self.char_weight if kind == "c" else 1.0
                out[row, h % self.dim] += sign * weight * (1.0 + math.log(count))
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (pip install sentence-transformers)."""

    default_min_score = 0.30

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs model
        vecs = self._model.encode(
            list(texts), # Texts ki list ko AI model mein daal kar dense vectors nikalta hai.
            batch_size=32, # Ek sath 32 texts process honge taaki RAM/GPU par zyada load na pade.
            normalize_embeddings=True, # Model khud vectors ko normalize kar deta hai.
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))


def get_embedder(spec: str) -> Embedder:
    """'hashing' | 'hashing:<dim>' | 'st:<model name>' | 'st' (default MiniLM)."""
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("hashing:"):
        return HashingEmbedder(dim=int(spec.split(":", 1)[1]))
    if spec == "st":
        return SentenceTransformerEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"Unknown embedder '{spec}'. Use 'hashing' or 'st:<model>'.")
