"""Build (never run) the OpenCode command; shared by the CLI and the web UI."""
from __future__ import annotations

from pathlib import Path

from .ollama import Settings
from .routing import select

CONFIG = (Path(__file__).resolve().parent.parent / "opencode.json").resolve()


def plan(prompt: str, project: Path, role: str, settings: Settings) -> dict:
    root = project.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    route = select(prompt, role)
    model = settings.model(route.role)
    return {
        "project": str(root),
        "role": route.role,
        "model": model,
        "why": route.why,
        "command": ["opencode", "run", "--model", f"ollama/{model}", prompt],
        "config": str(CONFIG) if CONFIG.exists() else None,
    }
