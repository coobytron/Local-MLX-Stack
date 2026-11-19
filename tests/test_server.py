import http.client
import json
import socket
import tempfile
import threading
import time
import unittest

from local_mlx_stack.core import Stack
from local_mlx_stack.ollama import Settings
from local_mlx_stack.server import serve


class FakeClient:
    def __init__(self):
        self.settings = Settings()

    def chat(self, role, prompt, system=""):
        return f"response for {role}"

    def models(self):
        # Ollama reports untagged models with a ':latest' suffix; status must still see them.
        return [self.settings.model("fast"), self.settings.model("coder") + ":latest"]

    def loaded(self):
        return []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        threading.Thread(target=serve, args=(Stack(FakeClient()), cls.port), daemon=True).start()
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", cls.port), 0.1).close()
                return
            except OSError:
                time.sleep(0.05)

    def call(self, method, path, body=None, headers=None):
        h = {"Host": f"127.0.0.1:{self.port}", **(headers or {})}
        if body is not None:
            h["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, json.dumps(body) if body is not None else None, h)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        return r.status, raw

    def test_ui_and_status(self):
        code, raw = self.call("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(b"Local MLX Stack", raw)
        code, raw = self.call("GET", "/status")
        roles = {r["role"]: r["installed"] for r in json.loads(raw)["roles"]}
        self.assertEqual(roles, {"fast": True, "coder": True, "reasoner": False})

    def test_host_and_origin_rules(self):
        self.assertEqual(self.call("GET", "/status", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.call("POST", "/route", {"prompt": "x"}, {"Origin": "http://evil.example"})[0], 403)
        own = {"Origin": f"http://127.0.0.1:{self.port}"}
        self.assertEqual(self.call("POST", "/route", {"prompt": "x"}, own)[0], 200)

    def test_opencode_plan_does_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            code, raw = self.call("POST", "/opencode", {"prompt": "fix a bug", "project": d})
        self.assertEqual(code, 200)
        plan = json.loads(raw)
        self.assertEqual(plan["role"], "coder")
        self.assertEqual(plan["command"][:2], ["opencode", "run"])
        self.assertEqual(self.call("POST", "/opencode", {"prompt": "x", "project": "/nonexistent/zz"})[0], 400)

    def test_pipeline_job_reports_stages(self):
        code, raw = self.call("POST", "/pipeline", {"prompt": "Plan a thing", "trigger": "review"})
        self.assertEqual(code, 202)
        job_id = json.loads(raw)["id"]
        for _ in range(100):
            job = json.loads(self.call("GET", f"/job/{job_id}")[1])
            if job["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "done")
        self.assertEqual([s["name"] for s in job["stages"]], ["brief", "plan", "draft", "review"])
        self.assertEqual(self.call("POST", "/pipeline", {"prompt": "x", "trigger": "bogus"})[0], 400)


if __name__ == "__main__":
    unittest.main()
