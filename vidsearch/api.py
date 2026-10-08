"""FastAPI service: ingest, search and ask over the video library."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from vidsearch.answer import Answerer, build_answerer
from vidsearch.config import Config
from vidsearch.ingest import ALL_SUPPORTED_SUFFIXES, ingest_file, ingest_url
from vidsearch.models import Hit, format_timestamp
from vidsearch.retriever import MODES
from vidsearch.store import Library
from vidsearch.transcribe import MEDIA_SUFFIXES, TRANSCRIPT_SUFFIXES

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|bm25)$")


class ScrapeRequest(BaseModel):
    url: str = Field(min_length=5, max_length=2000)
    title: str | None = None



def create_app(
    config: Config | None = None,
    library: Library | None = None,
    answerer: Answerer | None = None,
) -> FastAPI:
    config = config or Config.from_env()
    library = library or Library(config)
    answerer = answerer or build_answerer(
        llm_model=config.resolve_llm_model(),
        min_dense_score=library.min_dense_score,
        ollama_host=config.llm_host or config.ollama_host,
        api_key=config.llm_api_key,
        max_tokens=config.llm_max_tokens,
        provider=config.llm_provider,
    )
    app = FastAPI(title="VideoRAG · Timestamped Video Search (RAG)", version="0.1.0")

    def media_url(video_id: str) -> str | None:
        meta = library.videos.get(video_id)
        return f"/media/{video_id}" if meta and meta.media_file else None

    def hit_json(hit: Hit) -> dict:
        c = hit.chunk
        url = media_url(c.video_id)
        return {
            "video_id": c.video_id,
            "video_title": c.video_title,
            "start": c.start,
            "end": c.end,
            "start_label": format_timestamp(c.start),
            "end_label": format_timestamp(c.end),
            "text": c.text,
            "score": round(hit.score, 6),
            "dense_score": None if hit.dense_score is None else round(hit.dense_score, 4),
            "dense_rank": hit.dense_rank,
            "lexical_rank": hit.lexical_rank,
            "rerank_score": None if hit.rerank_score is None else round(hit.rerank_score, 4),
            "media_url": url,
            "deeplink": f"{url}#t={int(c.start)}" if url else None,
        }

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "embedder": library.embedder.name,
            "reranker": library.reranker.name if library.reranker else None,
            "videos": len(library.videos),
            "chunks": len(library.chunks),
        }

    @app.get("/videos")
    def list_videos() -> list[dict]:
        return [m.to_dict() for m in library.videos.values()]

    PROGRESS_REGISTRY: dict[str, dict] = {}

    @app.post("/videos", status_code=201)
    def add_video(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        whisper_model: str | None = Form(default=None),
        task_id: str | None = Form(default=None),
    ) -> dict:
        """Upload a transcript, document (.pdf/.txt/.md), or media file to index."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALL_SUPPORTED_SUFFIXES:
            raise HTTPException(415, f"Unsupported file type '{suffix}'")

        progress_cb = None
        if task_id:
            PROGRESS_REGISTRY[task_id] = {
                "percent": 0.0,
                "current_label": "00:00",
                "total_label": "00:00",
                "status": "transcribing",
            }

            def progress_cb(cur: float, total: float) -> None:
                pct = round((cur / total) * 100, 1) if total > 0 else 0.0
                PROGRESS_REGISTRY[task_id] = {
                    "percent": pct,
                    "current_seconds": cur,
                    "total_seconds": total,
                    "current_label": format_timestamp(cur),
                    "total_label": format_timestamp(total),
                    "status": "transcribing",
                }

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / f"upload{suffix}"
            written = 0
            with open(dest, "wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "File too large")
                    out.write(chunk)
            try:
                # Use the original filename stem as the default title.
                default_title = Path(file.filename or "upload").stem
                result = ingest_file(
                    library,
                    dest,
                    title=title or default_title,
                    whisper_model=whisper_model,
                    progress_callback=progress_cb,
                )
                if task_id:
                    PROGRESS_REGISTRY[task_id] = {
                        "percent": 100.0,
                        "status": "completed",
                        "chunks": result.n_chunks,
                    }
            except (ValueError, KeyError) as exc:
                if task_id:
                    PROGRESS_REGISTRY[task_id] = {"percent": 0.0, "status": "error", "error": str(exc)}
                raise HTTPException(422, f"Could not ingest file: {exc}") from exc
            except RuntimeError as exc:  # missing optional dependency
                if task_id:
                    PROGRESS_REGISTRY[task_id] = {"percent": 0.0, "status": "error", "error": str(exc)}
                raise HTTPException(501, str(exc)) from exc
        return asdict(result)

    @app.post("/websites", status_code=201)
    def add_website(payload: ScrapeRequest) -> dict:
        """Scrape and index an online webpage or article using Scrapy."""
        url = payload.url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(422, "URL must start with http:// or https://")
        try:
            result = ingest_url(library, url, title=payload.title)
            return asdict(result)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"Could not scrape webpage: {exc}") from exc
        except Exception as exc:
            raise HTTPException(502, f"Failed to fetch webpage: {exc}") from exc

    @app.get("/videos/progress/{task_id}")
    def get_progress(task_id: str) -> dict:
        return PROGRESS_REGISTRY.get(task_id, {"percent": 0.0, "status": "waiting"})

    @app.delete("/videos/{video_id}", status_code=204)
    def delete_video(video_id: str) -> None:
        if not library.remove(video_id):
            raise HTTPException(404, "Unknown video")

    @app.get("/media/{video_id}")
    def media(video_id: str) -> FileResponse:
        meta = library.videos.get(video_id)
        if not meta or not meta.media_file:
            raise HTTPException(404, "No media for this video")
        path = (library.data_dir / meta.media_file).resolve()
        if not path.is_relative_to((library.data_dir / "media").resolve()) or not path.exists():
            raise HTTPException(404, "No media for this video")
        return FileResponse(path)

    @app.get("/search")
    def search(
        q: str = Query(min_length=1, max_length=500),
        k: int = Query(default=5, ge=1, le=20),
        mode: str = Query(default="hybrid"),
    ) -> dict:
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        hits = library.search(q, k=k, mode=mode)  # type: ignore[arg-type]
        return {"query": q, "mode": mode, "results": [hit_json(h) for h in hits]}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        hits = library.search(req.question, k=req.k, mode=req.mode)  # type: ignore[arg-type]
        ans = answerer.answer(req.question, hits)
        by_id = {(h.chunk.video_id, h.chunk.start): h for h in hits}
        citations = []
        for c in ans.citations:
            item = hit_json(by_id[(c.video_id, c.start)])
            item["n"] = c.n
            citations.append(item)
        return {
            "question": req.question,
            "answer": ans.text,
            "grounded": ans.grounded,
            "mode": ans.mode,
            "citations": citations,
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "UI not found")
        return FileResponse(page)

    return app


def app_factory() -> FastAPI:  # for: uvicorn vidsearch.api:app_factory --factory
    return create_app()


app = create_app()

