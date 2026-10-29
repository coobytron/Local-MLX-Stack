"""Read-only, bounded project context for a text-only planning pipeline.

Only explicitly selected project files are sent to the model. No commands are
run against the project, and likely secret files and generated folders are
excluded. Retrieved file contents are evidence, not model instructions.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    '.git', '.venv', 'venv', 'node_modules', '.next', '.cache', '__pycache__',
    'dist', 'build', 'coverage', '.turbo', 'target', '.pytest_cache',
    '.mypy_cache', '.tox', '.idea', '.vscode', 'vendor',
}
SUFFIXES = {'.md', '.txt', '.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.toml', '.yaml', '.yml', '.sh'}
SECRET_PARTS = ('secret', 'credential', 'private', 'token', 'password', 'apikey', 'api_key', 'keychain')
EXCLUDED_NAMES = {'AGENTS.md', 'CLAUDE.md', 'MISTRAL.md', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'}


@dataclass(frozen=True)
class ProjectSnapshot:
    root: str
    files: tuple[str, ...]
    text: str


def _safe(relative: Path) -> bool:
    if relative.name in EXCLUDED_NAMES or relative.name.startswith('.'):
        return False
    if relative.suffix.lower() not in SUFFIXES:
        return False
    lowered = str(relative).lower()
    return not any(part in lowered for part in SECRET_PARTS)


def collect_project(project: Path, task: str, *, max_chars: int = 18000) -> ProjectSnapshot:
    """Inspect an existing project with a bounded text budget.

    Symlinks are never followed and sensitive/common config files are omitted.
    A file manifest is provided, but only selected files have contents.
    """
    root = project.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Project directory not found: {project}')
    if max_chars < 2000:
        raise ValueError('Project context budget must be at least 2000 characters')
    paths: list[Path] = []
    for directory, folders, files in os.walk(root, topdown=True, followlinks=False):
        here = Path(directory)
        depth = len(here.relative_to(root).parts)
        folders[:] = sorted(f for f in folders if f not in SKIP_DIRS and not f.startswith('.') and depth < 5)
        for name in sorted(files):
            item = here / name
            rel = item.relative_to(root)
            if _safe(rel) and item.is_file() and not item.is_symlink():
                paths.append(rel)
        if len(paths) > 4000:
            raise ValueError('Project has too many source files to snapshot safely')
    names = {str(p) for p in paths}
    # Select documented architecture first. Actual file existence is checked.
    priorities = [
        'README.md', 'docs/ARCHITECTURE.md', 'docs/LOCAL-LANGUAGE-MODELS.md',
        'backend/app/local_model_routes.py', 'backend/app/local_language_model.py',
        'docs/DRAW-THINGS-API.md', 'backend/app/main.py',
        'docs/GRPC-PROTOCOL.md', 'package.json', 'docs/REFERENCE-LAB.md',
    ]
    words = {w for w in re.findall(r'[a-z0-9][a-z0-9_-]{2,}', task.lower())
             if w not in {'design', 'local', 'this', 'that', 'with', 'from', 'into', 'using', 'things', 'controller'}}
    def rank(p: Path) -> int:
        value = str(p).lower()
        score = sum(3 for w in words if w in value)
        if any(s in value for s in ('backend/app', 'api', 'provider', 'transport', 'language-model')):
            score += 2
        if p.suffix.lower() in ('.py', '.ts', '.tsx'):
            score += 1
        return score
    chosen = [p for name in priorities if name in names for p in paths if str(p) == name]
    chosen_names = {str(p) for p in chosen}
    remaining = sorted((p for p in paths if str(p) not in chosen_names), key=lambda p: (-rank(p), str(p)))
    chosen += remaining[:12]

    manifest = '\n'.join(str(p) for p in paths[:100])[:3200]
    parts = [f'PROJECT ROOT: {root.name}\n',
             'OBSERVED SOURCE FILE PATHS (partial manifest; filenames are not proof of behavior):\n' + manifest]
    used = sum(len(s) for s in parts)
    included: list[str] = []
    for p in chosen:
        item = root / p
        # File size cap avoids reading large assets and logs into model context.
        try:
            if item.stat().st_size > 100_000:
                continue
            content = item.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            continue
        remaining_budget = max_chars - used - 250
        if remaining_budget < 450:
            break
        section_limit = min(2600 if p.suffix == '.md' else 1600, remaining_budget)
        if section_limit < 450:
            break
        excerpt = content[:section_limit]
        marker = '\n[TRUNCATED: inspect source file for full context]' if len(content) > len(excerpt) else ''
        section = f'\n\nFILE: {p}\n```\n{excerpt}\n```{marker}'
        parts.append(section)
        used += len(section)
        included.append(str(p))
    parts.append('\n\nCONTEXT LIMIT: Only the above excerpts were inspected. Missing files, APIs, and behavior are UNVERIFIED.')
    return ProjectSnapshot(root=str(root), files=tuple(included), text=''.join(parts))
