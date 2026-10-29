"""Minimal Ollama API client, dependency-free and intentionally local-first."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """An Ollama API request failed."""


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:11434"
    fast: str = "gemma4:12b"
    coder: str = "qwen3-coder:30b"
    reasoner: str = "qwen3.5:35b-mlx"
    context: int = 16384
    keep_alive: int = 0  # unload between calls: important for a 48 GB Mac
    timeout: int = 600

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            fast=os.getenv("STACK_FAST_MODEL", "gemma4:12b"),
            coder=os.getenv("STACK_CODER_MODEL", "qwen3-coder:30b"),
            reasoner=os.getenv("STACK_REASONER_MODEL", "qwen3.5:35b-mlx"),
            context=int(os.getenv("STACK_NUM_CTX", "16384")),
            keep_alive=int(os.getenv("STACK_KEEP_ALIVE", "0")),
            timeout=int(os.getenv("STACK_TIMEOUT", "600")),
        )

    def model(self, role: str) -> str:
        if role not in ("fast", "coder", "reasoner"):
            raise ValueError(f"Unknown model role: {role}")
        return getattr(self, role)


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_lock = Lock()  # serialize large model calls on 48 GB unified memory

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(
            self.settings.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.settings.timeout) as response:
                result: Any = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise OllamaError(f"Unexpected Ollama response for {path}")
                return result
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.settings.base_url}: {exc}. "
                "Check `ollama serve` and `ollama list`."
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError(f"Invalid Ollama JSON response: {exc}") from exc

    def models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", []) if "name" in m]

    def loaded(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/ps").get("models", []) if "name" in m]

    def chat(self, role: str, prompt: str, system: str = "") -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.settings.model(role),
            "messages": messages,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {"num_ctx": self.settings.context},
        }
        with self._chat_lock:
            result = self._request("POST", "/api/chat", payload)
        content = result.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError(f"Model {payload['model']} returned no response text")
        return content.strip()
