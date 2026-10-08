"""Ingest a transcript or media file into a Library."""

from __future__ import annotations

import shutil
from pathlib import Path

from vidsearch.store import AddResult, Library, make_video_id
from vidsearch.transcribe import (
    MEDIA_SUFFIXES,
    TRANSCRIPT_SUFFIXES,
    load_transcript,
    transcribe_media,
)


from collections.abc import Callable
import shutil
from pathlib import Path

from vidsearch.scraper import (
    DOCUMENT_SUFFIXES,
    parse_pdf_document,
    parse_text_document,
    scrape_website,
)
from vidsearch.store import AddResult, Library, make_video_id
from vidsearch.transcribe import (
    MEDIA_SUFFIXES,
    TRANSCRIPT_SUFFIXES,
    load_transcript,
    transcribe_media,
)

ALL_SUPPORTED_SUFFIXES = TRANSCRIPT_SUFFIXES | MEDIA_SUFFIXES | DOCUMENT_SUFFIXES


def ingest_file(
    library: Library,
    path: str | Path,
    title: str | None = None,
    whisper_model: str | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
) -> AddResult:
    """Add a transcript, document (.pdf/.txt/.md), or transcribe audio/video file.

    Media and document files are preserved in the library for direct reference.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in TRANSCRIPT_SUFFIXES:
        file_title, segments = load_transcript(p)
        return library.add_transcript(title or file_title or p.stem, segments, source=p.name)

    if suffix in DOCUMENT_SUFFIXES:
        if suffix == ".pdf":
            doc_title, segments = parse_pdf_document(p)
        else:
            doc_title, segments = parse_text_document(p)
        final_title = title or doc_title or p.stem
        video_id = make_video_id(final_title)
        rel = f"media/{video_id}{suffix}"
        dest = library.data_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        return library.add_transcript(final_title, segments, source=p.name, media_file=rel)

    if suffix in MEDIA_SUFFIXES:
        model_to_use = whisper_model or library.config.whisper_model
        segments = transcribe_media(p, model_to_use, progress_callback=progress_callback)
        final_title = title or p.stem
        video_id = make_video_id(final_title)
        rel = f"media/{video_id}{suffix}"
        dest = library.data_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        return library.add_transcript(final_title, segments, source=p.name, media_file=rel)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Use one of "
        f"{sorted(ALL_SUPPORTED_SUFFIXES)}."
    )


def ingest_url(
    library: Library,
    url: str,
    title: str | None = None,
) -> AddResult:
    """Scrape and index an online website or article using Scrapy's extraction engine."""
    scraped_title, segments = scrape_website(url)
    final_title = title or scraped_title or url
    return library.add_transcript(final_title, segments, source=url)

