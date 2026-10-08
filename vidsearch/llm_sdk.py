"""LLM client implementations and protocol."""

from __future__ import annotations

import os
from typing import Protocol


__all__ = [
    "LLMClient",
    "AnthropicClient",
    "OllamaClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "GroqClient",
]


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...



class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API (pip install anthropic)."""

    def __init__(self, model: str, max_tokens: int = 400) -> None:  # pragma: no cover
        import anthropic # type: ignore[import-not-found]

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network call
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OllamaClient:
    """Local Ollama client using standard HTTP requests (zero-dependency fallback)."""

    def __init__(self, model: str = "llama3:8b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            import requests  # type: ignore[import-not-found]

            res = requests.post(url, json=payload, timeout=60).json()
            return res["message"]["content"]
        except ImportError:
            import json
            import urllib.request

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["message"]["content"]


class HuggingFaceClient:
    """Hugging Face Serverless Inference API."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        api_key: str | None = None,
        max_tokens: int = 400,
    ) -> None:
        from huggingface_hub import InferenceClient  # type: ignore[import-not-found]

        token = api_key or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(api_key=token)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""


class OpenRouterClient:
    """OpenRouter API client (access Claude, GPT-4, Llama, Gemini via single key)."""

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: str | None = None,
        max_tokens: int = 400,
    ) -> None:
        from openai import OpenAI  # type: ignore[import-not-found]

        token = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=token,
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""


class GroqClient:
    """Groq API client for ultra-fast inference (pip install groq or openai)."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        max_tokens: int = 400,
    ) -> None:
        token = api_key or os.environ.get("GROQ_API_KEY")
        try:
            from groq import Groq  # type: ignore[import-not-found]

            self.client = Groq(api_key=token)
        except ImportError:
            from openai import OpenAI  # type: ignore[import-not-found]

            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=token,
            )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""

