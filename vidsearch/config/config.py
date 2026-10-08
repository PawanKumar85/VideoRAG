"""Runtime configuration, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(keys: tuple[str, ...], default: str) -> str:
    for k in keys:
        val = os.environ.get(k)
        if val is not None and val.strip():
            return val.strip()
    return default


def _env_int(keys: tuple[str, ...], default: int) -> int:
    for k in keys:
        val = os.environ.get(k)
        if val is not None and val.strip():
            try:
                return int(val.strip())
            except ValueError:
                pass
    return default


def _env_optional_str(keys: tuple[str, ...]) -> str | None:
    for k in keys:
        val = os.environ.get(k)
        if val is not None and val.strip():
            return val.strip()
    return None


def _load_dotenv(env_path: Path | None = None, override: bool = False) -> None:
    """Load variables from .env into os.environ."""
    candidates = (
        [env_path]
        if env_path
        else [
            Path(".env"),
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent.parent / ".env",
        ]
    )
    for path in candidates:
        if path and path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if key and (override or key not in os.environ):
                        os.environ[key] = val
            except Exception:
                pass
            break


# Auto-load .env into os.environ at module import time
_load_dotenv()


@dataclass(frozen=True)
class Config:
    # Where the library (chunks, vectors, media) is persisted.
    data_dir: Path = Path("data")

    # "hashing" (offline, lexical-ish, for dev/tests) or "st:<sentence-transformers model>".
    embedder: str = "hashing"

    # Chunking. Tokens are approximated (see chunking.approx_tokens).
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 40

    # Retrieval.
    rrf_k: int = 60
    candidate_k: int = 50
    quantize: bool = False

    # Dense cosine below this => the answerer abstains. None => embedder default.
    min_dense_score: float | None = None

    # Reranking: None | "proximity" | "ce" | "cross-encoder" | "ce:<model>".
    reranker: str | None = None
    rerank_top_n: int = 20

    # Loaded directly from .env / environment variables:
    whisper_model: str = field(
        default_factory=lambda: _env_str(("VIDSEARCH_WHISPER_MODEL", "WHISPER_MODEL"), "base")
    )

    # LLM generation configuration - loaded automatically from .env
    llm_provider: str = field(
        default_factory=lambda: _env_str(
            ("VIDSEARCH_LLM_PROVIDER", "LLM_PROVIDER", "PROVIDER", "provider"), "ollama"
        )
    )
    llm_model: str = field(
        default_factory=lambda: _env_str(
            ("VIDSEARCH_LLM_MODEL", "LLM_MODEL", "MODEL", "model"), "qwen2.5:3b-instruct"
        )
    )
    llm_host: str = field(
        default_factory=lambda: _env_str(
            ("LLM_HOST", "HOST", "host", "OLLAMA_HOST"), "http://localhost:11434"
        )
    )
    llm_api_key: str | None = field(
        default_factory=lambda: _env_optional_str(
            (
                "LLM_API_KEY",
                "API_KEY",
                "api_key",
                "GROQ_API_KEY",
                "OPENROUTER_API_KEY",
                "HF_TOKEN",
                "ANTHROPIC_API_KEY",
            )
        )
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int(
            ("VIDSEARCH_LLM_MAX_TOKENS", "LLM_MAX_TOKENS", "MAX_TOKENS"), 400
        )
    )

    # Provider-specific default models & endpoints:
    ollama_host: str = field(
        default_factory=lambda: _env_str(("OLLAMA_HOST", "LLM_HOST"), "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env_str(("OLLAMA_MODEL",), "qwen2.5:3b-instruct")
    )
    openrouter_model: str = field(
        default_factory=lambda: _env_str(("OPENROUTER_MODEL",), "anthropic/claude-3.5-sonnet")
    )
    huggingface_model: str = field(
        default_factory=lambda: _env_str(("HF_MODEL",), "meta-llama/Meta-Llama-3-8B-Instruct")
    )
    groq_model: str = field(
        default_factory=lambda: _env_str(("GROQ_MODEL",), "llama-3.3-70b-versatile")
    )
    anthropic_model: str = field(
        default_factory=lambda: _env_str(("ANTHROPIC_MODEL",), "claude-3-5-haiku-latest")
    )

    def resolve_llm_model(self) -> str:
        """Resolve effective model string with appropriate provider prefix."""
        provider = (self.llm_provider or "").lower().strip()
        model = self.llm_model or ""

        if provider == "ollama":
            chosen = self.ollama_model if model in {"", "default"} else model
            return chosen if chosen.startswith("ollama:") else f"ollama:{chosen}"
        if provider == "openrouter":
            chosen = self.openrouter_model if model in {"", "qwen2.5:3b-instruct", "llama3:8b", "default"} else model
            return chosen if chosen.startswith("openrouter:") else f"openrouter:{chosen}"
        if provider in {"huggingface", "hf"}:
            chosen = self.huggingface_model if model in {"", "qwen2.5:3b-instruct", "llama3:8b", "default"} else model
            return chosen if (chosen.startswith("hf:") or chosen.startswith("huggingface:")) else f"hf:{chosen}"
        if provider == "groq":
            chosen = self.groq_model if model in {"", "qwen2.5:3b-instruct", "llama3:8b", "default"} else model
            return chosen if chosen.startswith("groq:") else f"groq:{chosen}"
        if provider == "anthropic":
            chosen = self.anthropic_model if model in {"", "qwen2.5:3b-instruct", "llama3:8b", "default"} else model
            return chosen if chosen.startswith("anthropic:") else f"anthropic:{chosen}"
        return self.llm_model



    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> "Config":
        _load_dotenv(Path(env_file) if env_file else None)
        d = cls()
        min_score = os.environ.get("VIDSEARCH_MIN_DENSE_SCORE")

        # Flexible .env keys (accepts LLM_MODEL, MODEL, host, API_KEY, etc.)
        provider = (
            os.environ.get("VIDSEARCH_LLM_PROVIDER")
            or os.environ.get("LLM_PROVIDER")
            or os.environ.get("PROVIDER")
            or d.llm_provider
        )
        model = (
            os.environ.get("VIDSEARCH_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or os.environ.get("MODEL")
            or os.environ.get("model")
            or d.llm_model
        )
        host = (
            os.environ.get("LLM_HOST")
            or os.environ.get("HOST")
            or os.environ.get("host")
            or os.environ.get("OLLAMA_HOST")
            or d.llm_host
        )
        api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("api_key")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
            or d.llm_api_key
        )
        # Provider-specific model overrides
        ollama_model = os.environ.get("OLLAMA_MODEL", d.ollama_model)
        openrouter_model = os.environ.get("OPENROUTER_MODEL", d.openrouter_model)
        huggingface_model = os.environ.get("HF_MODEL", d.huggingface_model)
        groq_model = os.environ.get("GROQ_MODEL", d.groq_model)
        anthropic_model = os.environ.get("ANTHROPIC_MODEL", d.anthropic_model)

        if provider == "groq" and (os.environ.get("GROQ_MODEL") or not os.environ.get("VIDSEARCH_LLM_MODEL")):
            model = groq_model if (os.environ.get("GROQ_MODEL") or model == d.llm_model) else model
        elif provider == "openrouter" and (os.environ.get("OPENROUTER_MODEL") or not os.environ.get("VIDSEARCH_LLM_MODEL")):
            model = openrouter_model if (os.environ.get("OPENROUTER_MODEL") or model == d.llm_model) else model
        elif provider in {"huggingface", "hf"} and (os.environ.get("HF_MODEL") or not os.environ.get("VIDSEARCH_LLM_MODEL")):
            model = huggingface_model if (os.environ.get("HF_MODEL") or model == d.llm_model) else model
        elif provider == "anthropic" and (os.environ.get("ANTHROPIC_MODEL") or not os.environ.get("VIDSEARCH_LLM_MODEL")):
            model = anthropic_model if (os.environ.get("ANTHROPIC_MODEL") or model == d.llm_model) else model
        elif provider == "ollama" and os.environ.get("OLLAMA_MODEL"):
            model = ollama_model

        max_tokens_raw = (
            os.environ.get("VIDSEARCH_LLM_MAX_TOKENS")
            or os.environ.get("LLM_MAX_TOKENS")
            or os.environ.get("MAX_TOKENS")
        )
        max_tokens = int(max_tokens_raw) if max_tokens_raw else d.llm_max_tokens

        return cls(
            data_dir=Path(os.environ.get("VIDSEARCH_DATA_DIR", str(d.data_dir))),
            embedder=os.environ.get("VIDSEARCH_EMBEDDER", d.embedder),
            chunk_max_tokens=int(os.environ.get("VIDSEARCH_CHUNK_TOKENS", d.chunk_max_tokens)),
            chunk_overlap_tokens=int(
                os.environ.get("VIDSEARCH_CHUNK_OVERLAP", d.chunk_overlap_tokens)
            ),
            rrf_k=int(os.environ.get("VIDSEARCH_RRF_K", d.rrf_k)),
            candidate_k=int(os.environ.get("VIDSEARCH_CANDIDATE_K", d.candidate_k)),
            quantize=_env_bool("VIDSEARCH_QUANTIZE", d.quantize),
            min_dense_score=float(min_score) if min_score else None,
            reranker=os.environ.get("VIDSEARCH_RERANKER", d.reranker),
            rerank_top_n=int(os.environ.get("VIDSEARCH_RERANK_TOP_N", d.rerank_top_n)),
            whisper_model=os.environ.get("VIDSEARCH_WHISPER_MODEL", d.whisper_model),
            llm_provider=provider,
            llm_model=model,
            llm_host=host,
            llm_api_key=api_key,
            llm_max_tokens=max_tokens,
            ollama_host=os.environ.get("OLLAMA_HOST", host),
            ollama_model=os.environ.get("OLLAMA_MODEL", d.ollama_model),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", d.openrouter_model),
            huggingface_model=os.environ.get("HF_MODEL", d.huggingface_model),
            groq_model=os.environ.get("GROQ_MODEL", d.groq_model),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", d.anthropic_model),
        )

