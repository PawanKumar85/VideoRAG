"""Rerankers: refine first-stage retrieval candidates with cross-scoring.

First-stage retrieval (dense vectors, BM25, RRF) quickly narrows thousands of
chunks down to the top ~20 candidates using independent representations. A reranker
performs deep cross-scoring on (query, chunk_text) pairs, observing terms in
context to sharpen ordering among near-duplicates.

Implementations:
* ProximityReranker: offline, zero-dependency reranker. Combines exact phrase
  matches, query term coverage, character n-gram morphology, and term span
  proximity (rewarding chunks where query words appear close together).
* CrossEncoderReranker: transformer cross-encoder via sentence-transformers
  (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2). Full cross-attention between
  query and passage.
"""

from __future__ import annotations

import math
import re
from typing import Protocol, Sequence

from vidsearch.text import content_tokens


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return a relevance score for each text given the query (higher is better)."""
        ...


_WS = re.compile(r"\s+")


class ProximityReranker:
    """Zero-dependency lexical-semantic reranker.

    Scores candidate chunks using four signals:
      1. Term coverage: fraction of unique query content tokens found in the chunk.
      2. Phrase bonus: exact substring presence of the normalized query.
      3. Term proximity: minimum span in tokens that contains all matched query words.
         Words appearing within the same sentence/phrase score higher than words
         scattered across hundreds of tokens.
      4. Char trigram overlap: Jaccard similarity of character trigrams for morphology.
    """

    def __init__(self, name: str = "proximity") -> None:
        self.name = name

    def _trigrams(self, text: str) -> set[str]:
        padded = f"#{text.lower()}#"
        return {padded[i : i + 3] for i in range(len(padded) - 2)} if len(padded) >= 3 else set()

    def _score_single(self, query: str, text: str) -> float:
        q_norm = _WS.sub(" ", query.lower()).strip()
        t_norm = _WS.sub(" ", text.lower()).strip()
        if not q_norm or not t_norm:
            return 0.0

        # Exact phrase match gets immediate high baseline
        phrase_bonus = 0.35 if q_norm in t_norm else 0.0

        q_tokens = content_tokens(query)
        if not q_tokens:
            return phrase_bonus

        t_tokens = content_tokens(text)
        if not t_tokens:
            return phrase_bonus

        q_set = set(q_tokens)
        positions: dict[str, list[int]] = {term: [] for term in q_set}
        for pos, tok in enumerate(t_tokens):
            if tok in positions:
                positions[tok].append(pos)

        matched = [term for term, pos_list in positions.items() if pos_list]
        coverage = len(matched) / len(q_set)

        # Proximity score: if >= 2 terms match, find shortest span containing at least
        # one occurrence of each matched term.
        proximity_bonus = 0.0
        if len(matched) >= 2:
            # Flatten to (pos, term) pairs sorted by pos
            events: list[tuple[int, str]] = []
            for term in matched:
                for p in positions[term]:
                    events.append((p, term))
            events.sort(key=lambda x: x[0])

            # Two-pointer minimum window covering all matched terms
            req = len(matched)
            counts: dict[str, int] = {}
            have = 0
            left = 0
            min_span = float("inf")
            for right in range(len(events)):
                r_pos, r_term = events[right]
                counts[r_term] = counts.get(r_term, 0) + 1
                if counts[r_term] == 1:
                    have += 1
                while have == req:
                    l_pos, l_term = events[left]
                    span = r_pos - l_pos + 1
                    if span < min_span:
                        min_span = span
                    counts[l_term] -= 1
                    if counts[l_term] == 0:
                        have -= 1
                    left += 1

            if min_span < float("inf"):
                # Perfect span equals number of matched tokens (min_span == req => bonus = 0.3)
                # Decays smoothly as distance grows: 1 / (1 + log(span / req))
                ratio = max(1.0, min_span / req)
                proximity_bonus = 0.30 / (1.0 + math.log(ratio))

        # Character trigram Jaccard for morphology (e.g. refund vs refunds)
        q_tri = self._trigrams(query)
        t_tri = self._trigrams(text)
        jaccard = len(q_tri & t_tri) / len(q_tri | t_tri) if (q_tri and t_tri) else 0.0

        # Weighted combination: bounded roughly in [0, 1]
        score = (coverage * 0.40) + proximity_bonus + phrase_bonus + (jaccard * 0.15)
        return min(1.0, round(score, 6))

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [self._score_single(query, t) for t in texts]


class CrossEncoderReranker:
    """Neural cross-encoder reranker via sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:  # pragma: no cover
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def get_reranker(spec: str | None) -> Reranker | None:
    """'proximity' | 'lexical' | 'ce' | 'cross-encoder' | 'ce:<model>' | 'none' / None."""
    if not spec or spec.strip().lower() in {"none", "off", "false", ""}:
        return None
    s = spec.strip()
    if s in {"proximity", "lexical"}:
        return ProximityReranker()
    if s in {"ce", "cross-encoder"}:
        return CrossEncoderReranker()
    raise ValueError(
        f"Unknown reranker '{spec}'. Use 'proximity', 'ce', 'ce:<model>' or 'none'."
    )
