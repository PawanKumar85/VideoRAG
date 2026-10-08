"""Persistent library: videos, chunks and vectors, with incremental indexing.

On disk (data_dir):
    library.json   embedder name, video metadata, chunk metadata + text
    vectors.npy    float32 matrix, row i belongs to chunk i
    media/         optional copies of uploaded media for playback

Incremental indexing: each chunk carries sha1(text). When a video is added or
re-added, chunks whose text hash already exists (in any video) reuse the stored
vector instead of being embedded again, and re-adding byte-identical content is a
no-op. Embedding is the expensive step, so this is what keeps updates cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vidsearch.chunking import chunk_segments
from vidsearch.config import Config
from vidsearch.embeddings import Embedder, get_embedder
from vidsearch.models import Chunk, Hit, Segment, VideoMeta
from vidsearch.reranker import Reranker, get_reranker
from vidsearch.retriever import HybridRetriever, Mode


class EmbedderMismatchError(RuntimeError):
    """The persisted index was built with a different embedding model."""


@dataclass
class AddResult:
    video_id: str
    status: str  # "added" | "updated" | "unchanged"
    n_chunks: int
    n_embedded: int  # chunks that actually had to be embedded (the rest were reused)


def make_video_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "video"
    return f"{slug}-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:6]}"


def hash_segments(segments: Sequence[Segment]) -> str:
    h = hashlib.sha1()
    for s in segments:
        h.update(f"{s.start:.3f}|{s.end:.3f}|{s.text}\n".encode("utf-8"))
    return h.hexdigest()


class Library:
    def __init__(
        self,
        config: Config | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.embedder = embedder or get_embedder(self.config.embedder)
        self.reranker = reranker or get_reranker(self.config.reranker)
        self.data_dir = Path(self.config.data_dir)
        self._lock = threading.RLock()
        self.videos: dict[str, VideoMeta] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._retriever: HybridRetriever | None = None
        self._load()

    # ------------------------------------------------------------------ queries
    @property
    def min_dense_score(self) -> float:
        if self.config.min_dense_score is not None:
            return self.config.min_dense_score
        return self.embedder.default_min_score

    def retriever(self) -> HybridRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = HybridRetriever(
                    self.chunks,
                    self.vectors,
                    self.embedder,
                    reranker=self.reranker,
                    rerank_top_n=self.config.rerank_top_n,
                    rrf_k=self.config.rrf_k,
                    candidate_k=self.config.candidate_k,
                    quantize=self.config.quantize,
                )
            return self._retriever

    def search(
        self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = True
    ) -> list[Hit]:
        return self.retriever().search(query, k=k, mode=mode, rerank=rerank)

    # ---------------------------------------------------------------- mutations
    def add_transcript(
        self,
        title: str,
        segments: Sequence[Segment],
        *,
        source: str | None = None,
        media_file: str | None = None,
    ) -> AddResult:
        usable = sorted((s for s in segments if s.text and s.text.strip()), key=lambda s: s.start)
        if not usable:
            raise ValueError("transcript has no usable segments")
        title = title.strip() or "Untitled"

        with self._lock:
            video_id = make_video_id(title)
            content_hash = hash_segments(usable)
            existing = self.videos.get(video_id)

            if existing and existing.content_hash == content_hash:
                if media_file and existing.media_file != media_file:
                    existing.media_file = media_file
                    self._save()
                return AddResult(video_id, "unchanged", existing.n_chunks, 0)

            new_chunks = chunk_segments(
                usable,
                video_id,
                title,
                max_tokens=self.config.chunk_max_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )

            # Build the reuse cache BEFORE dropping the old version of this video,
            # so unchanged chunks of an edited transcript are not re-embedded.
            cache = {c.text_hash: self.vectors[i] for i, c in enumerate(self.chunks)}
            missing = list({c.text_hash: c.text for c in new_chunks if c.text_hash not in cache}.items())
            if missing:
                encoded = self.embedder.encode([text for _, text in missing])
                cache.update({h: encoded[i] for i, (h, _) in enumerate(missing)})

            new_vectors = np.stack([cache[c.text_hash] for c in new_chunks]).astype(np.float32)

            keep = [i for i, c in enumerate(self.chunks) if c.video_id != video_id]
            self.chunks = [self.chunks[i] for i in keep] + new_chunks
            self.vectors = np.vstack([self.vectors[keep], new_vectors]) if keep else new_vectors

            self.videos[video_id] = VideoMeta(
                video_id=video_id,
                title=title,
                content_hash=content_hash,
                n_segments=len(usable),
                n_chunks=len(new_chunks),
                duration=max(s.end for s in usable),
                media_file=media_file or (existing.media_file if existing else None),
                source=source,
            )
            self._retriever = None
            self._save()
            return AddResult(
                video_id, "updated" if existing else "added", len(new_chunks), len(missing)
            )

    def remove(self, video_id: str) -> bool:
        with self._lock:
            if video_id not in self.videos:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c.video_id != video_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = (
                self.vectors[keep]
                if keep
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
            meta = self.videos.pop(video_id)
            if meta.media_file:
                (self.data_dir / meta.media_file).unlink(missing_ok=True)
            self._retriever = None
            self._save()
            return True

    # -------------------------------------------------------------- persistence
    @property
    def _json_path(self) -> Path:
        return self.data_dir / "library.json"

    @property
    def _vec_path(self) -> Path:
        return self.data_dir / "vectors.npy"

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Write to temp files then rename: os.replace is atomic on one filesystem,
        # so a crash never leaves a half-written file under the real name.
        vec_tmp = self._vec_path.with_suffix(".npy.tmp")
        with open(vec_tmp, "wb") as f:
            np.save(f, self.vectors)
        os.replace(vec_tmp, self._vec_path)

        payload = {
            "version": 1,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "videos": {vid: m.to_dict() for vid, m in self.videos.items()},
            "chunks": [c.to_dict() for c in self.chunks],
        }
        json_tmp = self._json_path.with_suffix(".json.tmp")
        json_tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(json_tmp, self._json_path)

    def _load(self) -> None:
        if not self._json_path.exists():
            return
        payload = json.loads(self._json_path.read_text(encoding="utf-8"))
        stored = payload.get("embedder")
        if stored != self.embedder.name:
            raise EmbedderMismatchError(
                f"Index in '{self.data_dir}' was built with embedder '{stored}', but the "
                f"configured embedder is '{self.embedder.name}'. Vectors from different "
                f"models are not comparable. Either set VIDSEARCH_EMBEDDER to match the "
                f"index, or delete the data directory and re-index."
            )
        vectors = np.load(self._vec_path)
        chunks = [Chunk(**c) for c in payload["chunks"]]
        if len(chunks) != len(vectors):
            raise RuntimeError("Corrupt index: chunk count does not match vector count.")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32, copy=False)
        self.videos = {vid: VideoMeta(**m) for vid, m in payload["videos"].items()}
