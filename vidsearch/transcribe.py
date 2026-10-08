"""Getting timed text: parse existing transcripts (JSON/SRT/VTT) or run Whisper on media."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from vidsearch.models import Segment
from vidsearch.text import clean_repeated_phrases

TRANSCRIPT_SUFFIXES = {".json", ".srt", ".vtt"}
MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAGS = re.compile(r"<[^>]+>")


def _to_seconds(match: re.Match) -> float:
    h, m, s, ms = match.groups()
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_vtt(text: str) -> list[Segment]:
    """Parse SRT or WebVTT. Cue identifiers, headers and styling tags are ignored."""
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            if "-->" not in line:
                continue
            left, _, right = line.partition("-->")
            a, b = _TIME.search(left), _TIME.search(right)
            if not (a and b):
                break
            body = _TAGS.sub("", " ".join(lines[i + 1 :])).strip()
            if body:
                segments.append(Segment(_to_seconds(a), _to_seconds(b), body))
            break
    return segments


def load_transcript(path: str | Path) -> tuple[str | None, list[Segment]]:
    """Return (title or None, segments) from a .json, .srt or .vtt file.

    JSON accepts either ``{"title": ..., "segments": [{"start","end","text"}]}``
    or a bare list of such segment objects.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    raw = p.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw)
        title = data.get("title") if isinstance(data, dict) else None
        
        items = data["segments"] if isinstance(data, dict) else data
        segments = [Segment(float(s["start"]), float(s["end"]), str(s["text"])) for s in items]
        return title, segments
    if suffix in {".srt", ".vtt"}:
        return None, parse_srt_vtt(raw)
    raise ValueError(f"Unsupported transcript format: {suffix}")


def _patch_pyav_compatibility() -> None:
    """PyAV 14+ removed metadata_errors parameter, but faster-whisper still passes it."""
    try:
        import av  # type: ignore[import-not-found]
        if not getattr(av, "_patched_metadata_errors", False):
            orig_open = av.open

            def _safe_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return orig_open(*args, **kwargs)

            av.open = _safe_av_open
            av._patched_metadata_errors = True
    except Exception:
        pass


def transcribe_media(
    path: str | Path,
    model_size: str = "base",
    progress_callback: Callable[[float, float], None] | None = None,
) -> list[Segment]:
    """Transcribe audio/video with faster-whisper (pip install faster-whisper).

    Optimized for high-speed CPU inference:
    - multi-core thread pool (cpu_threads)
    - beam_size=1 (greedy decoding is 3x-5x faster than beam search)
    - vad_filter=True to skip silence and avoid hallucination
    - int8 quantization for low memory & fast math
    """
    _patch_pyav_compatibility()
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper "
            "(or provide a .json/.srt/.vtt transcript instead)."
        ) from exc
    import os

    num_threads = min(8, max(4, os.cpu_count() or 4))
    model = WhisperModel(model_size, device="auto", compute_type="int8", cpu_threads=num_threads)
    raw_segments, _info = model.transcribe(
        str(path),
        vad_filter=True,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        hallucination_silence_threshold=2.0,
    )
    total_duration = float(getattr(_info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []
    for s in raw_segments:
        clean_text = clean_repeated_phrases(s.text.strip())
        if clean_text:
            segments.append(Segment(float(s.start), float(s.end), clean_text))
        if progress_callback and total_duration > 0:
            progress_callback(min(float(s.end), total_duration), total_duration)

    if progress_callback and total_duration > 0:
        progress_callback(total_duration, total_duration)

    return segments
