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


def is_installed(model: str, installed: list[str]) -> bool:
    """Ollama lists untagged models as 'name:latest', so match either spelling."""
    return model in installed or f"{model}:latest" in installed or (
        model.endswith(":latest") and model[:-7] in installed)


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:11434"
    fast: str = "gemma4:12b"
    coder: str = "qwen3-coder:30b"
    reasoner: str = "qwen3.5:35b-mlx"
    context: int = 16384
    keep_alive: int = 0  # unload between calls: important for a 48 GB Mac
    timeout: int = 600
    opencode_model: str = "qwen3-coder-32k"  # coder copy with a 32k num_ctx; see `stack setup`
    opencode_ctx: int = 32768

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            fast=os.getenv("STACK_FAST_MODEL", "gemma4:12b"),
            coder=os.getenv("STACK_CODER_MODEL", "qwen3-coder:30b"),
            reasoner=os.getenv("STACK_REASONER_MODEL", "qwen3.5:35b-mlx"),
            context=_int_env("STACK_NUM_CTX", 16384, 1),
            keep_alive=_int_env("STACK_KEEP_ALIVE", 0),
            timeout=_int_env("STACK_TIMEOUT", 600, 1),
            opencode_model=os.getenv("STACK_OPENCODE_MODEL", "qwen3-coder-32k"),
            opencode_ctx=_int_env("STACK_OPENCODE_CTX", 32768, 1),
        )

    def model(self, role: str) -> str:
        if role not in ("fast", "coder", "reasoner"):
            raise ValueError(f"Unknown model role: {role}")
        return getattr(self, role)


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_lock = Lock()  # serialize large model calls on 48 GB unified memory

    def _raw(self, method: str, path: str, body: dict | None = None) -> str:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(
            self.settings.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.settings.timeout) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.settings.base_url}: {exc}. "
                "Check `ollama serve` and `ollama list`."
            ) from exc
        except UnicodeError as exc:
            raise OllamaError(f"Invalid Ollama JSON response: {exc}") from exc

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        try:
            result: Any = json.loads(self._raw(method, path, body))
        except ValueError as exc:
            raise OllamaError(f"Invalid Ollama JSON response: {exc}") from exc
        if not isinstance(result, dict):
            raise OllamaError(f"Unexpected Ollama response for {path}")
        return result

    def models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", []) if "name" in m]

    def loaded(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/ps").get("models", []) if "name" in m]

    def create_variant(self) -> None:
        """Create settings.opencode_model from the coder model with a bigger num_ctx (no CLI needed)."""
        s = self.settings
        text = self._raw("POST", "/api/create", {
            "model": s.opencode_model, "from": s.coder,
            "parameters": {"num_ctx": s.opencode_ctx}, "stream": False,
        })
        # Some Ollama versions stream NDJSON status lines even with stream=false: read every line.
        for line in filter(None, (ln.strip() for ln in text.splitlines())):
            try:
                status = json.loads(line)
            except ValueError as exc:
                raise OllamaError(f"Invalid Ollama JSON response: {exc}") from exc
            if isinstance(status, dict) and status.get("error"):
                raise OllamaError(f"Ollama could not create {s.opencode_model}: {status['error']}")

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
