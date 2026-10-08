"""CLI entry point for vidsearch (e.g. python -m vidsearch <command>)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidsearch.answer import build_answerer
from vidsearch.config import Config
from vidsearch.evaluation import evaluate, load_queries
from vidsearch.ingest import ingest_file
from vidsearch.models import format_timestamp
from vidsearch.store import Library


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m vidsearch",
        description="VideoRAG: Timestamped video search with hybrid retrieval and cited answers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI web server & UI.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable live reload for development")

    # add command
    add_parser = subparsers.add_parser("add", help="Add transcripts or video/audio files to library.")
    add_parser.add_argument("files", nargs="+", help="Files to index (.json, .srt, .vtt, .mp4, .mp3, etc.)")
    add_parser.add_argument("--title", help="Optional title override (for single file)")

    # search command
    search_parser = subparsers.add_parser("search", help="Search the library with hybrid/dense/bm25.")
    search_parser.add_argument("query", help="Search query string")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"], help="Search mode")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question and generate a grounded, cited answer.")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("-k", type=int, default=5, help="Candidate chunks to retrieve")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval recall and MRR.")
    eval_parser.add_argument("--queries", default="eval/queries.json", help="Path to labelled queries JSON")
    eval_parser.add_argument("--reranker", help="Optional reranker: proximity or cross-encoder")
    eval_parser.add_argument("-k", type=int, default=5, help="Recall cut-off (default: 5)")

    args = parser.parse_args()

    config = Config.from_env()

    if args.command == "serve":
        import uvicorn

        print(f"Starting VideoRAG server on http://{args.host}:{args.port}")
        uvicorn.run("vidsearch.api:app", host=args.host, port=args.port, reload=args.reload)
        return

    library = Library(config)

    if args.command == "add":
        for file_path in args.files:
            p = Path(file_path)
            if not p.exists():
                print(f"File not found: {p}", file=sys.stderr)
                continue
            res = ingest_file(library, p, title=args.title)
            print(f"[{res.status.upper()}] {p.name} -> {res.n_chunks} chunks ({res.n_embedded} embedded)")

    elif args.command == "search":
        hits = library.search(args.query, k=args.k, mode=args.mode)
        if not hits:
            print("No matching videos found.")
            return
        print(f"\nTop {len(hits)} results for '{args.query}':\n")
        for rank, h in enumerate(hits, start=1):
            c = h.chunk
            print(f"[{rank}] {c.video_title} @ {format_timestamp(c.start)} - {format_timestamp(c.end)}")
            print(f"    Score: {h.score:.4f} (dense={h.dense_score if h.dense_score is not None else 0.0:.4f})")
            print(f"    Text:  {c.text}\n")

    elif args.command == "ask":
        hits = library.search(args.question, k=args.k)
        answerer = build_answerer(
            llm_model=config.resolve_llm_model(),
            min_dense_score=library.min_dense_score,
            ollama_host=config.llm_host,
            api_key=config.llm_api_key,
            max_tokens=config.llm_max_tokens,
            provider=config.llm_provider,
        )
        ans = answerer.answer(args.question, hits)
        print(f"\nQuestion: {args.question}")
        print(f"Answer:   {ans.text}")
        print(f"Mode:     {ans.mode} (grounded={ans.grounded})\n")
        if ans.citations:
            print("Citations:")
            for c in ans.citations:
                print(f"  [{c.n}] {c.video_title} ({format_timestamp(c.start)} - {format_timestamp(c.end)})")

    elif args.command == "eval":
        q_path = Path(args.queries)
        if not q_path.exists():
            print(f"Queries file not found: {q_path}", file=sys.stderr)
            sys.exit(1)
        if args.reranker:
            from vidsearch.reranker import get_reranker

            library.reranker = get_reranker(args.reranker)
        queries = load_queries(q_path)
        report = evaluate(library, queries, k=args.k)
        print(f"\nEvaluation on {report['n_queries']} queries (recall@{args.k}):\n")
        print(f"{'Mode':<10} {'Recall@k':<12} {'MRR':<10}")
        print("-" * 34)
        for mode, metrics in report["modes"].items():
            print(f"{mode:<10} {metrics[f'recall@{args.k}']:<12.4f} {metrics['mrr']:<10.4f}")


if __name__ == "__main__":
    main()
