"""CLI entry point for the local model stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import agentic, bench, opencode
from .acb import ACB_URL, TRIGGERS, locate
from .core import Stack
from .ollama import OllamaClient, OllamaError, Settings, is_installed
from .routing import select
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stack", description="Local Ollama / MLX multi-model stack")
    parser.add_argument("--acb-root", help="Local checkout of coobytron/Agent-Cody-Banks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Create the 32k-context OpenCode model if it is missing")
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
    pipe.add_argument("--trigger", choices=TRIGGERS, required=True, help="Pipeline stage selector (review adds critique)")
    pipe.add_argument("--project", type=Path, help="Read bounded source context from this existing project")
    pipe.add_argument("--no-acb", action="store_true", help="Disable optional external agent guidance for this run")
    pipe.add_argument("--output", type=Path, help="Save result as JSON")
    api = sub.add_parser("serve", help="Start localhost-only JSON API")
    api.add_argument("--port", type=int, default=8788)
    bench = sub.add_parser("bench", help="Run fixed tasks on one or more models and compare plain-code scores")
    bench.add_argument("--models", help="Comma-separated Ollama tags (default: the coder model)")
    bench.add_argument("--tasks", help="Comma-separated task names (default: all except heavy ones)")
    bench.add_argument("--heavy", action="store_true", help="Also run heavy tasks (several model calls each)")
    bench.add_argument("--hard", action="store_true", help="Also run the hard tier (tasks that tell strong models apart)")
    bench.add_argument("--output", type=Path, help="Append results as JSON lines")
    bench.add_argument("--list", action="store_true", help="List tasks and exit")
    bench.add_argument("--agentic", action="store_true",
                       help="Run OpenCode edit jobs in temporary project copies instead of chat tasks (slow)")
    bench.add_argument("--repeat", type=int, default=1, help="Run the whole selection N times (1-10) to expose noise")
    bench.add_argument("--timeout", type=int, help="Seconds per agentic task (default: per task, 300-420)")
    oc = sub.add_parser("opencode", help="Launch local OpenCode against a project, or print dry run")
    oc.add_argument("prompt")
    oc.add_argument("--project", type=Path, default=Path.cwd())
    oc.add_argument("--role", choices=("auto", "fast", "coder", "reasoner"), default="coder")
    oc.add_argument("--execute", action="store_true", help="Actually start OpenCode; may modify project files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
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
                present = is_installed(model, models)
                print(f"  {role:8} {model:25} {'INSTALLED' if present else 'MISSING'}")
            print(f"Loaded: {', '.join(loaded) if loaded else '(none)'}")
            oc_ok = is_installed(settings.opencode_model, models)
            print(f"  {'opencode':8} {settings.opencode_model:25} "
                  f"{'INSTALLED' if oc_ok else 'MISSING (run: stack setup)'}")
            print(f"48GB profile: num_ctx={settings.context}, keep_alive={settings.keep_alive}")
            print(f"OpenCode: {shutil.which('opencode') or 'not in PATH'}")
            print(f"Agent Cody Banks: {acb_root or 'not found (optional)'} — {ACB_URL}")
            return 0 if all(is_installed(settings.model(r), models) for r in ("fast", "coder", "reasoner")) else 1
        if args.command == "setup":
            if is_installed(settings.opencode_model, client.models()):
                print(f"{settings.opencode_model} is already installed.")
            else:
                client.create_variant()
                print(f"Created {settings.opencode_model} from {settings.coder} "
                      f"with context {settings.opencode_ctx}.")
            return 0
        if args.command == "route":
            r = select(args.prompt, args.role)
            print(json.dumps({"role": r.role, "model": settings.model(r.role), "why": r.why}, indent=2))
        elif args.command == "ask":
            a = stack.ask(args.prompt, args.role)
            print(json.dumps(a.as_dict(), indent=2) if args.json else f"[{a.role} · {a.model}]\n{a.response}")
        elif args.command == "pipeline":
            result = stack.pipeline(args.prompt, args.trigger, project=args.project, use_acb=not args.no_acb)
            if args.output:
                dest = args.output.expanduser().resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"Saved text-only proposal: {dest}")
            else:
                if args.project:
                    print(f"Project grounding: {result['project']} ({len(result['context_files'])} files excerpted)")
                else:
                    print("Warning: no --project provided; proposal is conceptual and unverified.", file=sys.stderr)
                for k, v in result.items():
                    if isinstance(v, str):
                        print(f"\n--- {k.upper()} ---\n{v}")
        elif args.command == "serve":
            if not (1 <= args.port <= 65535):
                raise ValueError("port must be between 1 and 65535")
            serve(stack, args.port)
        elif args.command == "bench":
            if not 1 <= args.repeat <= 10:
                raise ValueError("--repeat must be between 1 and 10")
            pool = agentic.TASKS if args.agentic else bench.TASKS
            by_name = {t.name: t for t in pool}
            if args.list:
                for t in pool:
                    print(f"{t.name:<20} {t.category:<12} {getattr(t, 'tier', 'easy'):<5}"
                          f"{' (heavy)' if getattr(t, 'heavy', False) else ''}")
                return 0
            wanted = ([n.strip() for n in args.tasks.split(",")] if args.tasks
                      else [t.name for t in pool if args.agentic
                            or ((args.heavy or not t.heavy) and (args.hard or t.tier == "easy"))])
            unknown = [n for n in wanted if n not in by_name]
            if unknown:
                raise ValueError(f"Unknown task(s): {', '.join(unknown)}. Try --list.")
            models = [m.strip() for m in args.models.split(",")] if args.models else [settings.coder]
            progress = lambda s: print(s, file=sys.stderr, flush=True)
            chosen = [by_name[n] for n in wanted]
            if args.agentic:
                print(agentic.WARNING, flush=True)
                run_once = lambda: agentic.run_agentic(settings, models, chosen, timeout=args.timeout, progress=progress)
            else:
                right, total, misses = bench.router_accuracy()
                print(f"Router (no model): {right}/{total} typical prompts sent to the right role"
                      + "".join(f"\n  miss: {m}" for m in misses))
                run_once = lambda: bench.run(settings, models, chosen, progress=progress)
            results = []
            for i in range(1, args.repeat + 1):
                if args.repeat > 1:
                    progress(f"run {i}/{args.repeat}")
                results += [{**r, "run": i} for r in run_once()]
            print("\n" + bench.format_table(results))
            if args.output:
                bench.save(results, args.output)
                print(f"\nSaved {len(results)} result(s) to {args.output.expanduser().resolve()}")
        elif args.command == "opencode":
            p = opencode.plan(args.prompt, args.project, args.role, settings,
                              opencode.installed_models(client))
            if p["warning"]:
                print(f"WARNING: {p['warning']}", file=sys.stderr)
            cmd, project = p["command"], p["project"]
            env = os.environ.copy()
            if p["config"]:
                env["OPENCODE_CONFIG"] = p["config"]
            print(f"Project: {project}\nModel: {p['model']}\nCommand: {cmd!r}\nConfig: {env.get('OPENCODE_CONFIG', '(autodiscovery)')}", flush=True)
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
