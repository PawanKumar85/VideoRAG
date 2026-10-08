"""Plain data types shared across the pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass


def format_timestamp(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass(frozen=True)
class Segment:
    """One timed piece of transcript, as produced by ASR or parsed from SRT/VTT."""

    start: float
    end: float
    text: str


@dataclass
class Chunk:
    """A retrievable unit: consecutive segments with start/end metadata for deep-linking."""

    chunk_id: str
    video_id: str
    video_title: str
    start: float
    end: float
    text: str
    text_hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.video_title} @ {format_timestamp(self.start)}"


@dataclass
class Hit:
    """A ranked retrieval result.

    score        : the score used for ordering (RRF, cosine or BM25 depending on mode)
    dense_score  : cosine similarity to the query (always filled when vectors exist);
                   the answerer uses it as an absolute relevance signal, because RRF
                   scores are rank-based and cannot say "nothing relevant was found".
    """

    chunk: Chunk
    score: float
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class VideoMeta:
    video_id: str
    title: str
    content_hash: str
    n_segments: int
    n_chunks: int
    duration: float
    media_file: str | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
