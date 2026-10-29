import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from local_mlx_stack.ollama import OllamaClient, OllamaError, Settings

class MockOllama(BaseHTTPRequestHandler):
    last_request = None
    def log_message(self, *_):
        pass
    def do_GET(self):
        if self.path == "/api/tags":
            data = {"models": [{"name": "gemma4:12b"}, {"name": "qwen3-coder:30b"}, {"name": "qwen3.5:35b-mlx"}]}
        elif self.path == "/api/ps":
            data = {"models": []}
        else:
            self.send_error(404)
            return
        b = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        MockOllama.last_request = json.loads(body)
        b = json.dumps({"message": {"role": "assistant", "content": "hello from model"}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

class OllamaClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_models_loaded_and_chat(self):
        client = OllamaClient(Settings(base_url=f"http://127.0.0.1:{self.server.server_port}"))
        self.assertEqual(len(client.models()), 3)
        self.assertEqual(client.loaded(), [])
        self.assertEqual(client.chat("reasoner", "Explain", "You are helpful"), "hello from model")
        self.assertEqual(MockOllama.last_request["model"], "qwen3.5:35b-mlx")
        self.assertEqual(MockOllama.last_request["keep_alive"], 0)
        self.assertEqual(MockOllama.last_request["stream"], False)
        self.assertEqual(MockOllama.last_request["options"]["num_ctx"], 16384)

    def test_unreachable(self):
        client = OllamaClient(Settings(base_url="http://127.0.0.1:1", timeout=1))
        with self.assertRaises(OllamaError):
            client.models()
