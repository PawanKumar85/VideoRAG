# VideoRAG — Grounded Multimodal Search & RAG

Ask questions across your entire library of **videos, audio recordings, documents, and web articles**, and get back verifiable answers with **click-to-seek timestamp deep-links** and exact excerpt citations. If the indexed knowledge does not contain the answer, the system abstains instead of hallucinating.

```
"How do I request a refund?"  ──►  [1] Billing & Refund Policy @ 00:00
                                   "To request a refund, open the billing tab, choose the invoice,
                                    and press request refund. [1]"
```

---

## Key Features

- **Multimodal Ingestion**:
  - **Video & Audio**: `.mp4`, `.mov`, `.mkv`, `.webm`, `.mp3`, `.wav`, `.m4a` transcribed with fast local Whisper.
  - **Transcripts**: `.json`, `.srt`, `.vtt` timed subtitle files.
  - **Documents**: `.pdf`, `.md`, `.txt` documentation, papers, and lecture notes.
  - **Web Scraping**: Ingest any web URL / article using an integrated **Scrapy** engine.
- **Anti-Hallucination & Repetition Suppression**:
  - Whisper decoding with `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, and `hallucination_silence_threshold=2.0`.
  - Multi-scale text sanitization to collapse runaway transcription loops and background noise artifacts.
- **Hybrid Retrieval with RRF**:
  - Dense semantic vectors (offline hashing or `sentence-transformers`) + BM25 inverted index.
  - Fused via **Reciprocal Rank Fusion (RRF)** with optional cross-encoder / proximity reranking.
- **Grounded Verification & Abstain Path**:
  - LLM prompt fences untrusted transcripts as data.
  - Absolute cosine threshold triggers an early abstain path (`"I could not find this in the indexed videos."`) to avoid ungrounded answers.
  - Offline fallback to extractive quoting when no LLM API key is present.
- **Canvas UI (`canvasui.dev`)**:
  - Ambient glassmorphic dark theme with live background canvas particles.
  - **Left Pane**: Knowledge source manager, real-time progress bars, citation inspector, and HTML5 video player with deep-link seeking.
  - **Right Pane**: Conversational chat interface with quick suggestion chips, strategy picker, and grounded badge metadata.

---

## Architecture

```
  Video / Audio ──► Whisper (greedy int8) ─┐
  Documents (.pdf, .md, .txt) ─────────────┼──► Timed Segments
  Web URLs (Scrapy extractor) ─────────────┤
  Subtitles (.json, .srt, .vtt) ───────────┘
                                           │
                                           ▼
                     Chunk on segment boundaries (~300 tokens, ~13% overlap)
                     Retain start/end timestamps on every chunk
                                           │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
             Dense Vector Embeddings                 BM25 Inverted Index
             (Unit norm / int8 option)               (Exact lexical matching)
                       └───────────────────┬───────────────────┘
                                           │
  Query ──► Dense Top-50 ──────────────────┤
  Query ──► BM25  Top-50 ──────────────────┴──► Reciprocal Rank Fusion (RRF)
                                                       │
                                                       ▼
                                            (Optional) Reranker (Proximity / CE)
                                                       │
                                                       ▼
                                            Overlap Suppression ──► Top-k Hits
                                                       │
             Cosine < Min Dense Threshold ─────────────┤
                     ├──► ABSTAIN ("I could not find this in the indexed videos.")
                     │
                     └──► Fenced LLM Generation ──► Citation [n] Check ──► UI / API
```

| Component | Responsibility |
|---|---|
| [`vidsearch/scraper.py`](vidsearch/scraper.py) | Web scraping (Scrapy Selector), PDF extraction (`pypdf`), Markdown/Text parsing |
| [`vidsearch/transcribe.py`](vidsearch/transcribe.py) | Faster-Whisper CPU inference, PyAV compatibility patch, anti-repetition decoding |
| [`vidsearch/chunking.py`](vidsearch/chunking.py) | Segment-aware windowing with timestamp metadata and token overlap |
| [`vidsearch/text.py`](vidsearch/text.py) | Tokenization, stopword filtering, and runaway repetition suppression |
| [`vidsearch/embeddings.py`](vidsearch/embeddings.py) | Unit-norm embeddings (offline hashing or `sentence-transformers`) |
| [`vidsearch/bm25.py`](vidsearch/bm25.py) | Okapi BM25 inverted index built from scratch |
| [`vidsearch/fusion.py`](vidsearch/fusion.py) | Reciprocal Rank Fusion (RRF) combining dense and lexical ranks |
| [`vidsearch/reranker.py`](vidsearch/reranker.py) | Cross-encoder or zero-dependency term proximity reranking |
| [`vidsearch/retriever.py`](vidsearch/retriever.py) | Multi-stage retrieval coordinator and near-duplicate suppression |
| [`vidsearch/store.py`](vidsearch/store.py) | Persistent library store, incremental indexing (`sha1` caching), vector store |
| [`vidsearch/answer.py`](vidsearch/answer.py) | Fenced prompt builder, citation validator, and multi-provider LLM client |
| [`vidsearch/api.py`](vidsearch/api.py) | FastAPI service with upload progress registry, search, and Q&A routes |
| [`web/index.html`](web/index.html) | Canvas UI with video seeking, citation cards, and chat |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository and enter directory
cd Timestamped_video_search_with_RAG

# Create virtual environment and install dependencies
uv sync
source .venv/bin/activate
```

### 2. Run the Test Suite

```bash
pytest
# 78 passed in ~0.4s (100% offline, covers chunking, retrieval, RRF, API, and scrapers)
```

### 3. Launch the Web Application

```bash
uvicorn vidsearch.api:app --reload --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## Ingesting Content

### A. From the Canvas Web UI
1. **Videos & Audio**: Drag & drop `.mp4`, `.mov`, `.mp3` into the upload zone. Choose Whisper speed (`base` recommended for fast CPU transcription). Real-time progress displays transferred MB and transcribed minutes/seconds.
2. **Documents**: Drag & drop `.pdf`, `.md`, `.txt` files directly into the same drop zone.
3. **Websites**: Enter any URL in the **🌐 Scrape Website / Article** field and click **Ingest**.

### B. From the CLI
```bash
# Index sample transcripts
python -m vidsearch add samples/*.json

# Ingest your own video or audio file
python -m vidsearch add ~/Videos/lecture.mp4 --title "Linear Regression Lecture"

# Ingest an SRT or VTT transcript
python -m vidsearch add talk.srt --title "Conference Talk"

# Search and Ask from terminal
python -m vidsearch search "error 4031"
python -m vidsearch ask "What causes error 4031?"
```

### C. Using Python Code
```python
from vidsearch.config import Config
from vidsearch.store import Library
from vidsearch.ingest import ingest_file, ingest_url

config = Config.from_env()
library = Library(config)

# 1. Ingest a video
ingest_file(library, "presentation.mp4")

# 2. Ingest a document
ingest_file(library, "notes.pdf")

# 3. Scrape and index a website
ingest_url(library, "https://en.wikipedia.org/wiki/Linear_regression")
```

---

## Evaluation Benchmark

Evaluate retrieval precision using the built-in benchmark harness across 24 labelled queries:

```bash
# Baseline evaluation (Hybrid vs Dense vs BM25)
python -m vidsearch eval

# With Proximity Reranker enabled
python -m vidsearch eval --reranker proximity
```

Benchmark output on sample index:
| Mode | recall@5 | MRR |
|---|---|---|
| **Hybrid (RRF)** | **0.792** | **0.701** |
| Dense | 0.750 | 0.564 |
| BM25 | 0.792 | 0.729 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/videos` | Upload media (`.mp4`, `.mp3`), documents (`.pdf`, `.txt`, `.md`), or transcripts (`.json`, `.srt`) |
| `POST` | `/websites` | Scrape and index a web URL using Scrapy (`{"url": "https://..."}`) |
| `GET` | `/videos/progress/{task_id}` | Poll real-time upload and transcription progress |
| `GET` | `/videos` | List all indexed knowledge sources |
| `DELETE` | `/videos/{id}` | Delete a source and its embedded chunks |
| `GET` | `/search?q=&k=&mode=` | Search indexed chunks (`hybrid`, `dense`, or `bm25`) |
| `POST` | `/ask` | Ask a grounded question (`{"question": "...", "mode": "hybrid"}`) |
| `GET` | `/media/{id}` | Stream video/audio files with byte-range support for `#t=` deep-links |
| `GET` | `/health` | Service health, active LLM provider, chunk counts |

Interactive OpenAPI documentation is available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

---

## Configuration (`.env`)

| Environment Variable | Default | Description |
|---|---|---|
| `VIDSEARCH_DATA_DIR` | `data` | Directory for vector store and indexed media |
| `VIDSEARCH_EMBEDDER` | `hashing` | `hashing` or `st:sentence-transformers/all-MiniLM-L6-v2` |
| `VIDSEARCH_WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `VIDSEARCH_RERANKER` | `off` | `proximity`, `ce`, or `ce:<model_name>` |
| `VIDSEARCH_CHUNK_TOKENS` | `300` | Target tokens per chunk (~100s audio span) |
| `VIDSEARCH_CHUNK_OVERLAP` | `40` | Sliding window overlap tokens |
| `VIDSEARCH_MIN_DENSE_SCORE` | `0.2` | Cosine similarity threshold below which system abstains |
| `VIDSEARCH_LLM_PROVIDER` | `ollama` | Provider: `ollama`, `groq`, `openrouter`, `huggingface`, `anthropic` |
| `VIDSEARCH_LLM_MODEL` | `llama3:8b` | Model identifier |
| `GROQ_API_KEY` | - | API key for Groq ultra-fast cloud inference |
| `ANTHROPIC_API_KEY` | - | API key for Claude models |
| `OPENROUTER_API_KEY` | - | API key for OpenRouter gateway |

---

## Engineering Highlights

1. **Exact-Time Deep-Linking**: Retaining start and end timestamps on segment boundaries allows the web UI to jump straight to the exact second in the video where the answer was spoken.
2. **Zero-Overhead Vector Math**: Unit-normalized vectors turn cosine similarity into a single matrix-vector dot product without computing square roots at query time.
3. **Incremental Hashing**: Every chunk is keyed by `sha1(text)`. Re-uploading identical or slightly edited documents only computes embeddings for new or changed chunks.
4. **Resilient Offline Architecture**: If no external API key is provided, VideoRAG functions completely offline using extractive citation matching and local hashing embeddings.
