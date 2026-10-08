"""Scraping and document extraction engine for VideoRAG.

Supports:
- Web scraping via Scrapy Selector from HTTP/HTTPS URLs
- PDF document extraction via pypdf
- Plain text, Markdown, and documentation extraction (.txt, .md, .markdown)
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from vidsearch.models import Segment
from vidsearch.text import clean_repeated_phrases

DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def _build_segments_from_paragraphs(paragraphs: list[str], time_step: float = 10.0) -> list[Segment]:
    """Convert a sequence of textual paragraphs into timestamped segments.

    Assigns synthetic time offsets so downstream vector retrieval, chunking,
    and reference citing operate uniformly across video, web, and documents.
    """
    segments: list[Segment] = []
    current_time = 0.0

    for raw in paragraphs:
        cleaned = clean_repeated_phrases(raw.strip())
        if not cleaned or len(cleaned) < 5:
            continue
        # Split very long paragraphs (> 800 chars) into manageable sentence blocks
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if not sentences:
            sentences = [cleaned]

        buf: list[str] = []
        buf_len = 0
        for sent in sentences:
            buf.append(sent)
            buf_len += len(sent)
            if buf_len >= 300:
                block_text = " ".join(buf)
                segments.append(Segment(current_time, current_time + time_step, block_text))
                current_time += time_step
                buf = []
                buf_len = 0
        if buf:
            block_text = " ".join(buf)
            segments.append(Segment(current_time, current_time + time_step, block_text))
            current_time += time_step

    return segments


def parse_text_document(path: str | Path) -> tuple[str, list[Segment]]:
    """Parse a plain text or Markdown file into segments."""
    p = Path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = p.read_text(encoding="latin-1", errors="replace")

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    title = p.stem.replace("_", " ").replace("-", " ").title()
    segments = _build_segments_from_paragraphs(paragraphs)
    if not segments:
        raise ValueError(f"Document '{p.name}' contains no readable text content.")
    return title, segments


def parse_pdf_document(path: str | Path) -> tuple[str, list[Segment]]:
    """Parse a PDF document page by page."""
    p = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf") from exc

    reader = PdfReader(str(p))
    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()
    if not title:
        title = p.stem.replace("_", " ").replace("-", " ").title()

    segments: list[Segment] = []
    current_time = 0.0

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = clean_repeated_phrases(text.strip())
        if not cleaned:
            continue
        page_header = f"[Page {page_idx}]"
        # Each page is mapped to a 60-second segment step (e.g., Page 1 = 0:00, Page 2 = 1:00)
        page_segments = _build_segments_from_paragraphs([f"{page_header} {cleaned}"], time_step=30.0)
        for ps in page_segments:
            segments.append(Segment(current_time, current_time + ps.end - ps.start, ps.text))
            current_time += (ps.end - ps.start)

    if not segments:
        raise ValueError(f"PDF document '{p.name}' contains no readable text (it may be scanned/image-only).")
    return title, segments


def scrape_website(url: str, timeout: float = 15.0) -> tuple[str, list[Segment]]:
    """Scrape and extract article content from a web page using Scrapy's Selector."""
    from scrapy import Selector

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "xml" not in content_type:
            raise ValueError(f"URL returned non-HTML content type: {content_type}")
        charset = response.headers.get_content_charset() or "utf-8"
        html_bytes = response.read()
        html_text = html_bytes.decode(charset, errors="replace")

    sel = Selector(text=html_text)

    # Extract title
    title = (
        sel.xpath("//meta[@property='og:title']/@content").get()
        or sel.xpath("//title/text()").get()
        or sel.xpath("//h1/text()").get()
        or url
    )
    title = re.sub(r"\s+", " ", title).strip()

    # Extract meaningful content elements (prefer article/main if present)
    content_sel = sel.xpath("//article | //main")
    if not content_sel:
        content_sel = sel.xpath("//body")

    # Extract text from headings, paragraphs, and list items
    raw_blocks: list[str] = []
    # Drop noisy elements: script, style, nav, footer, header, form, noscript, svg
    for el in content_sel.xpath(".//p | .//h1 | .//h2 | .//h3 | .//h4 | .//li"):
        # Ensure element isn't inside a nav/footer/script
        parent_tags = [t.lower() for t in el.xpath("ancestor::*[self::nav or self::footer or self::header or self::form or self::aside]").getall()]
        if parent_tags:
            continue
        text = "".join(el.xpath(".//text()").getall()).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 20:
            raw_blocks.append(text)

    if not raw_blocks:
        # Fallback to general paragraph extraction
        for text in sel.xpath("//p//text()").getall():
            cleaned = re.sub(r"\s+", " ", text).strip()
            if len(cleaned) > 25:
                raw_blocks.append(cleaned)

    if not raw_blocks:
        raise ValueError(f"Could not extract readable article text from {url}.")

    segments = _build_segments_from_paragraphs(raw_blocks, time_step=15.0)
    if not segments:
        raise ValueError(f"No textual content could be parsed from {url}.")

    return title, segments
