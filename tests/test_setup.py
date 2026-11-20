import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from local_mlx_stack import opencode
from local_mlx_stack.cli import main
from local_mlx_stack.core import Stack
from local_mlx_stack.ollama import OllamaClient, OllamaError, Settings
from local_mlx_stack.server import serve


class FakeOllama:
    """Loopback fake: /api/tags lists `models`; /api/create records its body and adds the model."""

    def __init__(self, models, create_status=200):
        self.models, self.creates, outer = list(models), [], self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, value):
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                self.reply(200, {"models": [{"name": m} for m in outer.models]})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if create_status != 200:
                    return self.reply(create_status, {"error": "boom"})
                outer.creates.append(body)
                outer.models.append(body["model"])
                self.reply(200, {"status": "success"})

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_cli(argv, host):
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ, {"OLLAMA_HOST": host}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def dead_host() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class SetupTests(unittest.TestCase):
    def test_creates_when_missing(self):
        fake = FakeOllama(["qwen3-coder:30b"])
        self.addCleanup(fake.close)
        code, out, _ = run_cli(["setup"], fake.url)
        self.assertEqual(code, 0)
        self.assertEqual(fake.creates, [{"model": "qwen3-coder-32k", "from": "qwen3-coder:30b",
                                         "parameters": {"num_ctx": 32768}, "stream": False}])
        self.assertIn("Created qwen3-coder-32k from qwen3-coder:30b with context 32768", out)

    def test_noop_when_present(self):
        fake = FakeOllama(["qwen3-coder:30b", "qwen3-coder-32k:latest"])
        self.addCleanup(fake.close)
        code, out, _ = run_cli(["setup"], fake.url)
        self.assertEqual(code, 0)
        self.assertEqual(fake.creates, [])
        self.assertIn("already installed", out)

    def test_env_overrides(self):
        fake = FakeOllama([])
        self.addCleanup(fake.close)
        with patch.dict(os.environ, {"STACK_OPENCODE_MODEL": "mine", "STACK_OPENCODE_CTX": "8192"}):
            run_cli(["setup"], fake.url)
        self.assertEqual((fake.creates[0]["model"], fake.creates[0]["parameters"]), ("mine", {"num_ctx": 8192}))

    def test_ollama_down_exits_2(self):
        code, _, err = run_cli(["setup"], dead_host())
        self.assertEqual(code, 2)
        self.assertIn("Error:", err)

    def test_create_error_exits_2(self):
        fake = FakeOllama([], create_status=500)
        self.addCleanup(fake.close)
        code, _, err = run_cli(["setup"], fake.url)
        self.assertEqual(code, 2)
        self.assertIn("HTTP 500", err)

    def test_create_variant_raises_ollama_error(self):
        with self.assertRaises(OllamaError):
            OllamaClient(Settings(base_url=dead_host(), timeout=2)).create_variant()


class OpenCodeChoiceTests(unittest.TestCase):
    def dry_run(self, host):
        with tempfile.TemporaryDirectory() as d:
            return run_cli(["opencode", "Fix a Python bug", "--project", d], host)

    def test_picks_32k_when_installed(self):
        fake = FakeOllama(["qwen3-coder:30b", "qwen3-coder-32k"])
        self.addCleanup(fake.close)
        code, out, err = self.dry_run(fake.url)
        self.assertEqual(code, 0)
        self.assertIn("Model: qwen3-coder-32k", out)
        self.assertIn("ollama/qwen3-coder-32k", out)
        self.assertNotIn("WARNING", err)

    def test_falls_back_with_warning_when_missing(self):
        fake = FakeOllama(["qwen3-coder:30b"])
        self.addCleanup(fake.close)
        code, out, err = self.dry_run(fake.url)
        self.assertEqual(code, 0)
        self.assertIn("Model: qwen3-coder:30b", out)
        self.assertIn("WARNING: no 32k variant installed; OpenCode may loop. Run: stack setup", err)

    def test_falls_back_with_warning_when_ollama_down(self):
        code, out, err = self.dry_run(dead_host())
        self.assertEqual(code, 0)
        self.assertIn("Model: qwen3-coder:30b", out)
        self.assertIn("Run: stack setup", err)

    def test_non_coder_role_unchanged(self):
        p = opencode.plan("x", Path("."), "fast", Settings(), ["qwen3-coder-32k"])
        self.assertEqual((p["model"], p["warning"]), (Settings().fast, None))

    def test_plan_is_pure(self):
        s = Settings()
        self.assertEqual(opencode.plan("x", Path("."), "coder", s, ["qwen3-coder-32k"])["model"], s.opencode_model)
        missing = opencode.plan("x", Path("."), "coder", s, None)
        self.assertEqual((missing["model"], missing["warning"]), (s.coder, opencode.NO_VARIANT_WARNING))


class ServerAgreesTests(unittest.TestCase):
    def check(self, models):
        fake = FakeOllama(models)
        self.addCleanup(fake.close)
        settings = Settings(base_url=fake.url)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        threading.Thread(target=serve, args=(Stack(OllamaClient(settings)), port), daemon=True).start()
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)

        def call(method, path, body=None):
            h = {"Host": f"127.0.0.1:{port}"}
            if body is not None:
                h["Content-Type"] = "application/json"
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request(method, path, json.dumps(body) if body is not None else None, h)
            return json.loads(c.getresponse().read())

        with tempfile.TemporaryDirectory() as d:
            plan = call("POST", "/opencode", {"prompt": "Fix a Python bug", "project": d})
            cli_out = run_cli(["opencode", "Fix a Python bug", "--project", d], fake.url)
        return plan, call("GET", "/status")["opencode_model"], cli_out

    def test_installed(self):
        plan, status, (_, out, _) = self.check(["qwen3-coder:30b", "qwen3-coder-32k"])
        self.assertEqual(plan["model"], "qwen3-coder-32k")
        self.assertIn(f"Model: {plan['model']}", out)
        self.assertTrue(status["installed"])
        self.assertIsNone(plan["warning"])

    def test_missing(self):
        plan, status, (_, out, err) = self.check(["qwen3-coder:30b"])
        self.assertEqual(plan["model"], "qwen3-coder:30b")
        self.assertIn(f"Model: {plan['model']}", out)
        self.assertIn(plan["warning"], err)
        self.assertFalse(status["installed"])
        self.assertEqual(status["hint"], "run: stack setup")


class DoctorTests(unittest.TestCase):
    def test_missing_opencode_model_does_not_change_exit_code(self):
        settings = Settings()
        fake = FakeOllama([settings.fast, settings.coder, settings.reasoner])
        self.addCleanup(fake.close)
        code, out, _ = run_cli(["doctor"], fake.url)
        self.assertEqual(code, 0)
        self.assertIn("MISSING (run: stack setup)", out)


if __name__ == "__main__":
    unittest.main()
