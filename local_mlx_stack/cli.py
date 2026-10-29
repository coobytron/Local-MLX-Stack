"""CLI entry point for the local model stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .acb import ACB_URL, TRIGGERS, locate
from .core import Stack
from .ollama import OllamaClient, OllamaError, Settings
from .routing import select
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stack", description="Local Ollama / MLX multi-model stack")
    parser.add_argument("--acb-root", help="Local checkout of coobytron/Agent-Cody-Banks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Ollama and model installation")
    route = sub.add_parser("route", help="Show routing decision without calling a model")
    route.add_argument("prompt")
    route.add_argument("--role", choices=("auto", "fast", "coder", "reasoner"), default="auto")
    ask = sub.add_parser("ask", help="Run one primary-model task (default)")
    ask.add_argument("prompt")
    ask.add_argument("--role", choices=("auto", "fast", "coder", "reasoner"), default="auto")
    ask.add_argument("--json", action="store_true", help="Print structured result")
    pipe = sub.add_parser("pipeline", help="Opt in to sequential brief/plan/draft/review text-only stages")
    pipe.add_argument("prompt")
    pipe.add_argument("--trigger", choices=TRIGGERS, required=True, help="Exactly one Agent Cody Banks trigger")
    pipe.add_argument("--output", type=Path, help="Save result as JSON")
    api = sub.add_parser("serve", help="Start localhost-only JSON API")
    api.add_argument("--port", type=int, default=8788)
    oc = sub.add_parser("opencode", help="Launch local OpenCode against a project, or print dry run")
    oc.add_argument("prompt")
    oc.add_argument("--project", type=Path, default=Path.cwd())
    oc.add_argument("--role", choices=("auto", "fast", "coder", "reasoner"), default="coder")
    oc.add_argument("--execute", action="store_true", help="Actually start OpenCode; may modify project files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    client = OllamaClient(settings)
    acb_root = locate(args.acb_root)
    stack = Stack(client, acb_root)
    try:
        if args.command == "doctor":
            models = client.models()
            loaded = client.loaded()
            print(f"Ollama: {settings.base_url} (reachable)")
            for role in ("fast", "coder", "reasoner"):
                model = settings.model(role)
                present = model in models or (model.endswith(":latest") and model[:-7] in models)
                print(f"  {role:8} {model:25} {'INSTALLED' if present else 'MISSING'}")
            print(f"Loaded: {', '.join(loaded) if loaded else '(none)'}")
            print(f"48GB profile: num_ctx={settings.context}, keep_alive={settings.keep_alive}")
            print(f"OpenCode: {shutil.which('opencode') or 'not in PATH'}")
            print(f"Agent Cody Banks: {acb_root or 'not found (optional)'} — {ACB_URL}")
            return 0 if all(settings.model(r) in models for r in ("fast", "coder", "reasoner")) else 1
        if args.command == "route":
            r = select(args.prompt, args.role)
            print(json.dumps({"role": r.role, "model": settings.model(r.role), "why": r.why}, indent=2))
        elif args.command == "ask":
            a = stack.ask(args.prompt, args.role)
            print(json.dumps(a.as_dict(), indent=2) if args.json else f"[{a.role} · {a.model}]\n{a.response}")
        elif args.command == "pipeline":
            result = stack.pipeline(args.prompt, args.trigger)
            if args.output:
                dest = args.output.expanduser().resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"Saved text-only proposal: {dest}")
            else:
                for k, v in result.items():
                    if isinstance(v, str):
                        print(f"\n--- {k.upper()} ---\n{v}")
        elif args.command == "serve":
            if not (1 <= args.port <= 65535):
                raise ValueError("port must be between 1 and 65535")
            serve(stack, args.port)
        elif args.command == "opencode":
            project = args.project.expanduser().resolve()
            if not project.is_dir():
                raise ValueError(f"Not a directory: {project}")
            r = select(args.prompt, args.role)
            cmd = ["opencode", "run", "--model", f"ollama/{settings.model(r.role)}", args.prompt]
            config = (Path(__file__).resolve().parent.parent / "opencode.json").resolve()
            env = os.environ.copy()
            if config.exists():
                env["OPENCODE_CONFIG"] = str(config)
            print(f"Project: {project}\nCommand: {cmd!r}\nConfig: {env.get('OPENCODE_CONFIG', '(autodiscovery)')}", flush=True)
            if args.execute:
                if not shutil.which("opencode"):
                    raise ValueError("OpenCode not found. Install it first: https://opencode.ai")
                return subprocess.run(cmd, cwd=project, env=env, check=False).returncode
            print("Dry run only. Pass --execute to allow OpenCode to run with your configured permissions.")
    except (OllamaError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
