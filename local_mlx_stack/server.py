"""Loopback-only JSON API + single-page UI; no shell/file actions."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import opencode
from .acb import TRIGGERS
from .core import Stack
from .crawl import CrawlError, check_public
from .ollama import OllamaError
from .routing import select

MAX_BODY = 100_000
MAX_JOBS = 20
UI = Path(__file__).with_name("ui.html")
POST_PATHS = ("/route", "/chat", "/pipeline", "/opencode", "/crawl")


def status(stack: Stack) -> dict:
    s = stack.client.settings
    try:
        installed, loaded, error = stack.client.models(), stack.client.loaded(), None
    except OllamaError as exc:
        installed, loaded, error = [], [], str(exc)
    roles = [{"role": r, "model": s.model(r), "installed": s.model(r) in installed}
             for r in ("fast", "coder", "reasoner")]
    return {
        "ollama": s.base_url, "error": error, "roles": roles, "loaded": loaded,
        "num_ctx": s.context, "keep_alive": s.keep_alive,
        "opencode": shutil.which("opencode"),
        "acb": str(stack.acb_root) if stack.acb_root else None,
        "triggers": list(TRIGGERS),
    }


def serve(stack: Stack, port: int = 8788, *, allow_private: bool = False, crawl_delay: float = 1.0) -> None:
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def start_job(work) -> str:
        """Run work(on_stage) in a thread; the page polls /job/<id> for stages as they land."""
        job = {"status": "running", "stages": [], "result": None, "error": None}

        def run() -> None:
            try:
                job["result"] = work(lambda name, text: job["stages"].append({"name": name, "text": text}))
                job["status"] = "done"
            except (OllamaError, ValueError, OSError) as exc:
                job["error"], job["status"] = str(exc), "error"

        with jobs_lock:
            while len(jobs) >= MAX_JOBS:
                jobs.pop(next(iter(jobs)))
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = job
        threading.Thread(target=run, daemon=True).start()
        return job_id

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code: int, value: dict):
            payload = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def host_ok(self) -> bool:
            # Rejects DNS-rebinding: only literal loopback names on our port.
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def origin_ok(self) -> bool:
            # Same-origin page only; other websites are still refused.
            origin = self.headers.get("Origin")
            return origin is None or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

        def do_GET(self):  # noqa: N802
            if not self.host_ok():
                return self.send_json(403, {"error": "unexpected Host header"})
            if self.path in ("/", "/index.html"):
                body = UI.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            if self.path == "/health":
                try:
                    return self.send_json(200, {"status": "ok", "models": stack.client.models()})
                except OllamaError as exc:
                    return self.send_json(503, {"status": "unavailable", "error": str(exc)})
            if self.path == "/status":
                return self.send_json(200, status(stack))
            if self.path.startswith("/job/"):
                job = jobs.get(self.path[5:])
                if job is None:
                    return self.send_json(404, {"error": "unknown job"})
                return self.send_json(200, job)
            self.send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path not in POST_PATHS:
                return self.send_json(404, {"error": "not found"})
            if not self.host_ok():
                return self.send_json(403, {"error": "unexpected Host header"})
            if not self.origin_ok():
                return self.send_json(403, {"error": "browser origins are not permitted"})
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self.send_json(415, {"error": "expected application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    return self.send_json(413, {"error": "invalid request size"})
                data = json.loads(self.rfile.read(length))
                if self.path == "/crawl":
                    ok = isinstance(data, dict)
                    url, focus = (data.get("url"), data.get("focus", "")) if ok else (None, "")
                    if not isinstance(url, str) or not url.strip() or not isinstance(focus, str):
                        return self.send_json(400, {"error": "a start URL is required"})
                    try:
                        pages = int(data.get("max_pages", 5))
                        check_public(url.strip(), allow_private)
                    except CrawlError as exc:
                        return self.send_json(400, {"error": str(exc)})
                    job_id = start_job(lambda on_stage: stack.crawl(
                        url.strip(), focus=focus, max_pages=pages, on_stage=on_stage,
                        allow_private=allow_private, delay=crawl_delay))
                    return self.send_json(202, {"id": job_id})
                if not isinstance(data, dict) or not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
                    return self.send_json(400, {"error": "non-empty prompt is required"})
                prompt, role = data["prompt"], data.get("role", "auto")
                if not isinstance(role, str):
                    return self.send_json(400, {"error": "invalid role"})
                settings = stack.client.settings
                if self.path == "/route":
                    r = select(prompt, role)
                    return self.send_json(200, {"role": r.role, "model": settings.model(r.role), "why": r.why})
                if self.path == "/opencode":  # plan only: the UI never launches OpenCode
                    project = data.get("project")
                    if not isinstance(project, str) or not project.strip():
                        return self.send_json(400, {"error": "project folder is required"})
                    return self.send_json(200, opencode.plan(prompt, Path(project), role, settings))
                if self.path == "/pipeline":
                    trigger, project = data.get("trigger", "review"), data.get("project") or None
                    if trigger not in TRIGGERS or (project is not None and not isinstance(project, str)):
                        return self.send_json(400, {"error": "invalid trigger or project"})
                    job_id = start_job(lambda on_stage: stack.pipeline(
                        prompt, trigger, project=Path(project) if project else None,
                        use_acb=bool(data.get("use_acb", True)), on_stage=on_stage))
                    return self.send_json(202, {"id": job_id})
                return self.send_json(200, stack.ask(prompt, role).as_dict())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OllamaError as exc:
                self.send_json(503, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local MLX Stack at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
