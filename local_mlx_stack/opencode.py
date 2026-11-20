"""Build (never run) the OpenCode command; shared by the CLI and the web UI."""
from __future__ import annotations

from pathlib import Path

from .ollama import OllamaError, Settings, is_installed
from .routing import select

CONFIG = (Path(__file__).resolve().parent.parent / "opencode.json").resolve()
NO_VARIANT_WARNING = "no 32k variant installed; OpenCode may loop. Run: stack setup"


def installed_models(client) -> list[str] | None:
    """Installed model names, or None when Ollama can't be reached."""
    try:
        return client.models()
    except OllamaError:
        return None


def choose_model(role: str, settings: Settings, installed: list[str] | None) -> tuple[str, str | None]:
    """Pick the model OpenCode should use; the coder role prefers the 32k variant. Returns (model, warning)."""
    if role != "coder":
        return settings.model(role), None
    if installed is not None and is_installed(settings.opencode_model, installed):
        return settings.opencode_model, None
    return settings.coder, NO_VARIANT_WARNING


def plan(prompt: str, project: Path, role: str, settings: Settings,
         installed: list[str] | None = None) -> dict:
    """`installed` is the Ollama model list (None = unknown, which falls back with a warning)."""
    root = project.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    route = select(prompt, role)
    model, warning = choose_model(route.role, settings, installed)
    return {
        "project": str(root),
        "role": route.role,
        "model": model,
        "warning": warning,
        "why": route.why,
        "command": ["opencode", "run", "--model", f"ollama/{model}", prompt],
        "config": str(CONFIG) if CONFIG.exists() else None,
    }
