"""Loopback-only JSON API for local creative applications; no shell/file actions."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import Stack
from .ollama import OllamaError
from .routing import select

MAX_BODY = 100_000


def serve(stack: Stack, port: int = 8788) -> None:
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, value: dict):
            payload = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                try:
                    available = stack.client.models()
                    self.send_json(200, {"status": "ok", "models": available})
                except OllamaError as exc:
                    self.send_json(503, {"status": "unavailable", "error": str(exc)})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path not in ("/route", "/chat"):
                return self.send_json(404, {"error": "not found"})
            # Prevent cross-site webpages from issuing local model requests.
            if self.headers.get("Origin"):
                return self.send_json(403, {"error": "browser origins are not permitted"})
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self.send_json(415, {"error": "expected application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    return self.send_json(413, {"error": "invalid request size"})
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
                    return self.send_json(400, {"error": "non-empty prompt is required"})
                prompt, role = data["prompt"], data.get("role", "auto")
                if not isinstance(role, str):
                    return self.send_json(400, {"error": "invalid role"})
                if self.path == "/route":
                    r = select(prompt, role)
                    return self.send_json(200, {"role": r.role, "model": stack.client.settings.model(r.role), "why": r.why})
                return self.send_json(200, stack.ask(prompt, role).as_dict())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OllamaError as exc:
                self.send_json(503, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local MLX Stack API at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
