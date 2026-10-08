"""Sentence/segment-aware chunking with time metadata.

Why chunk on segment boundaries instead of fixed character windows:
  * a cut in the middle of a sentence produces an embedding of half a thought;
  * start/end timestamps stay exact, so a hit can deep-link to the right second.

Why overlap: a topic that straddles a boundary would otherwise be split across two
vectors, and neither may match the query well. Overlap duplicates a small tail
(~10-15% of max_tokens) into the next chunk. The cost is a slightly larger index
and near-duplicate neighbours in results (handled in retriever.suppress_overlaps).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from vidsearch.models import Chunk, Segment

_WS = re.compile(r"\s+")


def approx_tokens(text: str) -> int:
    """Cheap token estimate: ~1.33 tokens per English word.

    An exact count needs the target model's tokenizer; for chunk sizing a rough
    estimate is enough and keeps the pipeline free of heavy dependencies.
    """
    return max(1, math.ceil(len(text.split()) * 1.33))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_segments(
    segments: Iterable[Segment],
    video_id: str,
    video_title: str,
    max_tokens: int = 300,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Group consecutive segments into chunks of about ``max_tokens`` with overlap.

    Invariants (covered by tests):
      * chunk.start == first segment start, chunk.end == last segment end;
      * every segment appears in at least one chunk;
      * the final chunk is never a pure duplicate of the previous overlap tail;
      * a single segment longer than ``max_tokens`` becomes its own chunk.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens")

    segs = [
        Segment(s.start, s.end, _WS.sub(" ", s.text).strip())
        for s in segments
        if s.text and s.text.strip()
    ]

    chunks: list[Chunk] = []
    window: list[Segment] = []
    window_tokens = 0
    new_since_emit = 0  # segments added since the last emit (excludes overlap carry-over)

    def emit() -> None:
        text = " ".join(s.text for s in window)
        chunks.append(
            Chunk(
                chunk_id=f"{video_id}:{len(chunks)}",
                video_id=video_id,
                video_title=video_title,
                start=window[0].start,
                end=window[-1].end,
                text=text,
                text_hash=sha1_text(text),
            )
        )

    for seg in segs:
        window.append(seg)
        window_tokens += approx_tokens(seg.text)
        new_since_emit += 1
        if window_tokens >= max_tokens:
            emit()
            # Carry the trailing segments that fit in the overlap budget.
            carry: list[Segment] = []
            carried = 0
            for s in reversed(window):
                t = approx_tokens(s.text)
                if carried + t > overlap_tokens:
                    break
                carry.insert(0, s)
                carried += t
            window = carry
            window_tokens = carried
            new_since_emit = 0

    if window and new_since_emit > 0:
        emit()
    return chunks
