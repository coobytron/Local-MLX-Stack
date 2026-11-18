import http.client
import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_mlx_stack.core import Stack
from local_mlx_stack.crawl import CrawlError, check_public, crawl, parse_page, same_site_links
from local_mlx_stack.ollama import Settings
from local_mlx_stack.server import serve

SITE = {
    "/": '<html><head><title>Home</title><style>x{}</style></head><body><script>bad()</script>'
         '<p>Welcome home</p><a href="/a">A</a><a href="/b#top">B</a><a href="/a#again">A2</a>'
         '<a href="https://other.example/x">out</a><a href="/file.pdf">pdf</a></body></html>',
    "/a": '<html><title>Page A</title><body><p>About A</p><a href="/c">C</a></body></html>',
    "/b": "<html><title>Page B</title><body>secret B</body></html>",
    "/c": "<html><title>Page C</title><body>About C</body></html>",
    "/robots.txt": "User-agent: *\nDisallow: /b\n",
}


class SiteHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = SITE.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        kind = "text/plain" if self.path == "/robots.txt" else "text/html; charset=utf-8"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeClient:
    def __init__(self):
        self.settings = Settings()
        self.prompts = []

    def chat(self, role, prompt, system=""):
        self.prompts.append((role, prompt))
        return "- a summary"


class CrawlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.site = ThreadingHTTPServer(("127.0.0.1", cls.port), SiteHandler)
        threading.Thread(target=cls.site.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.site.shutdown()
        cls.site.server_close()

    def test_parse_page_drops_scripts_and_finds_links(self):
        title, text, links = parse_page(SITE["/"])
        self.assertEqual(title, "Home")
        self.assertIn("Welcome home", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn("x{}", text)
        self.assertIn("/a", links)

    def test_links_stay_on_site_and_skip_files(self):
        links = same_site_links("http://x.test/", ["/a", "/a#frag", "https://other.test/z", "/f.pdf", "mailto:a@b.c", "/b"])
        self.assertEqual(links, ["http://x.test/a", "http://x.test/b"])

    def test_private_hosts_are_blocked_by_default(self):
        with self.assertRaises(CrawlError):
            check_public("http://127.0.0.1:1/")
        with self.assertRaises(CrawlError):
            check_public("file:///etc/passwd", allow_private=True)

    def test_crawl_is_bounded_same_site_and_honors_robots(self):
        pages = list(crawl(self.base + "/", max_pages=10, delay=0, allow_private=True))
        urls = [p["url"].replace(self.base, "") for p in pages]
        self.assertEqual(urls, ["/", "/a", "/b", "/c"])
        blocked = {p["url"].replace(self.base, ""): p["error"] for p in pages}
        self.assertEqual(blocked["/b"], "blocked by robots.txt")
        self.assertEqual(len(list(crawl(self.base + "/", max_pages=2, delay=0, allow_private=True))), 2)

    def test_stack_summarizes_each_fetched_page_with_the_fast_model(self):
        client, stages = FakeClient(), []
        out = Stack(client).crawl(self.base + "/", focus="about", max_pages=3, delay=0, allow_private=True,
                                  on_stage=lambda n, t: stages.append(n))
        self.assertEqual([r for r, _ in client.prompts], ["fast", "fast"])  # /b was skipped, not summarized
        self.assertIn("Focus on: about.", client.prompts[0][1])
        self.assertEqual(len(out["pages"]), 3)
        self.assertEqual(len(stages), 3)

    def test_server_blocks_private_urls_and_runs_allowed_crawls(self):
        locked, open_ = free_port(), free_port()
        threading.Thread(target=serve, args=(Stack(FakeClient()), locked), daemon=True).start()
        threading.Thread(target=serve, args=(Stack(FakeClient()), open_),
                         kwargs={"allow_private": True, "crawl_delay": 0}, daemon=True).start()
        for p in (locked, open_):
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", p), 0.1).close()
                    break
                except OSError:
                    time.sleep(0.05)

        def call(port, method, path, body=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            headers = {"Host": f"127.0.0.1:{port}"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn.request(method, path, json.dumps(body) if body is not None else None, headers)
            r = conn.getresponse()
            raw = r.read()
            conn.close()
            return r.status, json.loads(raw)

        self.assertEqual(call(locked, "POST", "/crawl", {"url": self.base + "/"})[0], 400)
        self.assertEqual(call(open_, "POST", "/crawl", {"url": ""})[0], 400)
        code, body = call(open_, "POST", "/crawl", {"url": self.base + "/", "max_pages": 3})
        self.assertEqual(code, 202)
        for _ in range(100):
            job = call(open_, "GET", "/job/" + body["id"])[1]
            if job["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "done")
        self.assertEqual(len(job["stages"]), 3)
        self.assertTrue(job["stages"][0]["name"].startswith(self.base))


if __name__ == "__main__":
    unittest.main()
