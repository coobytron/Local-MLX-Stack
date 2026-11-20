import io
import os
import threading
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from local_mlx_stack import bench
from local_mlx_stack.cli import main
from local_mlx_stack.ollama import OllamaClient, OllamaError, Settings


def serve_body(body: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class CreateVariantTests(unittest.TestCase):
    def client(self, body: bytes):
        server = serve_body(body)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return OllamaClient(Settings(base_url=f"http://127.0.0.1:{server.server_port}"))

    def test_streamed_ndjson_status_lines_are_accepted(self):
        self.client(b'{"status":"creating model layer"}\n{"status":"writing manifest"}\n{"status":"success"}\n').create_variant()

    def test_single_json_reply_is_accepted(self):
        self.client(b'{"status":"success"}').create_variant()

    def test_error_line_inside_a_stream_is_raised(self):
        with self.assertRaisesRegex(OllamaError, "could not create"):
            self.client(b'{"status":"creating"}\n{"error":"model not found"}\n').create_variant()

    def test_garbage_reply_is_an_ollama_error(self):
        with self.assertRaises(OllamaError):
            self.client(b"not json at all").create_variant()


class EnvParsingTests(unittest.TestCase):
    def run_main(self, **env):
        err = io.StringIO()
        with mock.patch.dict(os.environ, env), redirect_stderr(err):
            code = main(["route", "hello"])
        return code, err.getvalue()

    def test_bad_numbers_exit_cleanly_instead_of_a_traceback(self):
        for name, value in (("STACK_OPENCODE_CTX", "abc"), ("STACK_OPENCODE_CTX", "0"),
                            ("STACK_NUM_CTX", "-5"), ("STACK_TIMEOUT", "0")):
            code, err = self.run_main(**{name: value})
            self.assertEqual(code, 2, (name, value))
            self.assertIn(name, err)

    def test_good_values_still_work(self):
        self.assertEqual(self.run_main(STACK_OPENCODE_CTX="65536", STACK_KEEP_ALIVE="-1")[0], 0)


def run_result(model, task, run, passed, failed_checks):
    checks = [{"name": n, "ok": False, "note": note} for n, note in failed_checks]
    checks += [{"name": f"ok{i}", "ok": True, "note": ""} for i in range(passed)]
    return {"model": model, "task": task, "run": run, "passed": passed, "failed": len(failed_checks),
            "skipped": 0, "seconds": 1.0, "checks": checks}


class TableEdgeCaseTests(unittest.TestCase):
    def test_same_score_but_different_failures_is_flagged_flaky(self):
        rows = [run_result("m", "t", 1, 3, [("c1", "")]), run_result("m", "t", 2, 3, [("c2", "")])]
        table = bench.format_table(rows)
        self.assertIn("*", table)
        self.assertIn("results differed between runs", table)

    def test_identical_runs_are_not_flagged(self):
        rows = [run_result("m", "t", i, 3, [("c1", "same")]) for i in (1, 2)]
        self.assertNotIn("results differed", bench.format_table(rows))

    def test_differing_notes_are_called_out(self):
        rows = [run_result("m", "t", 1, 3, [("c1", "note A")]), run_result("m", "t", 2, 3, [("c1", "note B")])]
        self.assertIn("note A; notes vary", bench.format_table(rows))

    def test_matching_notes_are_not_called_out(self):
        rows = [run_result("m", "t", i, 3, [("c1", "note A")]) for i in (1, 2)]
        self.assertNotIn("notes vary", bench.format_table(rows))


if __name__ == "__main__":
    unittest.main()
