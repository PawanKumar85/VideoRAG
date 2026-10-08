"""Retrieval evaluation: recall@k and MRR against a labelled query set.

A query is labelled with the time span(s) that answer it. A returned chunk counts as
relevant when it comes from the same video and its time range overlaps a labelled
span. Time-overlap labelling (rather than chunk ids) keeps the labels valid when
the chunking parameters change.

  recall@k : fraction of queries with at least one relevant chunk in the top k.
  MRR      : mean of 1/rank of the first relevant chunk (0 if none in the top k).

These are only meaningful on a labelled set you wrote yourself; report the size of
that set next to every number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vidsearch.models import Chunk
from vidsearch.retriever import MODES
from vidsearch.store import Library


@dataclass(frozen=True)
class LabelledQuery:
    query: str
    video_title: str
    start: float
    end: float
    kind: str = "general"  # e.g. "paraphrase" or "exact" for per-kind breakdowns


def load_queries(path: str | Path) -> list[LabelledQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data:
        rel = item["relevant"]
        out.append(
            LabelledQuery(
                query=item["query"],
                video_title=rel["video"],
                start=float(rel["start"]),
                end=float(rel["end"]),
                kind=item.get("kind", "general"),
            )
        )
    return out


def is_relevant(chunk: Chunk, q: LabelledQuery) -> bool:
    return chunk.video_title == q.video_title and min(chunk.end, q.end) > max(chunk.start, q.start)


def first_relevant_rank(chunks: Sequence[Chunk], q: LabelledQuery) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if is_relevant(c, q):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def evaluate(library: Library, queries: Sequence[LabelledQuery], k: int = 5) -> dict:
    """Return per-mode metrics, overall and per query kind."""
    report: dict = {"n_queries": len(queries), "k": k, "modes": {}}
    kinds = sorted({q.kind for q in queries})
    for mode in MODES:
        ranks = []
        for q in queries:
            hits = library.search(q.query, k=k, mode=mode)  # type: ignore[arg-type]
            ranks.append(first_relevant_rank([h.chunk for h in hits], q))
        entry = {f"recall@{k}": recall_at_k(ranks, k), "mrr": mrr(ranks), "by_kind": {}}
        for kind in kinds:
            kr = [r for r, q in zip(ranks, queries) if q.kind == kind]
            entry["by_kind"][kind] = {
                "n": len(kr),
                f"recall@{k}": recall_at_k(kr, k),
                "mrr": mrr(kr),
            }
        report["modes"][mode] = entry
    return report


def format_report(report: dict) -> str:
    k = report["k"]
    lines = [f"Queries: {report['n_queries']}   k={k}", ""]
    header = f"{'mode':<8} {'recall@' + str(k):>10} {'MRR':>8}"
    lines += [header, "-" * len(header)]
    for mode, e in report["modes"].items():
        lines.append(f"{mode:<8} {e[f'recall@{k}']:>10.3f} {e['mrr']:>8.3f}")
    kinds = list(next(iter(report["modes"].values()))["by_kind"].keys())
    if len(kinds) > 1:
        lines += ["", "By query kind (recall@k / MRR):"]
        for kind in kinds:
            row = [f"  {kind:<12}"]
            for mode, e in report["modes"].items():
                b = e["by_kind"][kind]
                row.append(f"{mode}={b[f'recall@{k}']:.2f}/{b['mrr']:.2f}")
            n = next(iter(report["modes"].values()))["by_kind"][kind]["n"]
            lines.append(" ".join(row) + f"   (n={n})")
    return "\n".join(lines)
