"""Data types for citations and answers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    n: int
    video_id: str
    video_title: str
    start: float
    end: float
    text: str


@dataclass
class Answer:
    text: str
    grounded: bool
    mode: str  # "llm" | "extractive" | "abstain"
    citations: list[Citation] = field(default_factory=list)
