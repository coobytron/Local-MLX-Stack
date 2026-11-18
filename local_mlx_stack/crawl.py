"""Bounded, polite, same-site page crawler. Plain code decides what to fetch; no model does."""
from __future__ import annotations

import ipaddress
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from html.parser import HTMLParser
from typing import Iterator
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

UA = "LocalMLXStack-crawler/0.1 (personal, local use)"
MAX_BYTES = 1_000_000
MAX_PAGES_LIMIT = 15
PAGE_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
SKIP_EXT = (".pdf", ".zip", ".gz", ".dmg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
            ".mp3", ".mp4", ".mov", ".css", ".js", ".json", ".xml", ".ico", ".woff", ".woff2")
SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
BREAK_TAGS = {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}


class CrawlError(ValueError):
    pass


def check_public(url: str, allow_private: bool = False) -> None:
    """Only http(s) to public addresses, so a page can't point us at local services."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CrawlError(f"Only http/https URLs are allowed: {url}")
    if allow_private:
        return
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(parts.hostname, parts.port or 80)}
    except OSError as exc:
        raise CrawlError(f"Cannot resolve {parts.hostname}: {exc}") from exc
    if not addrs or not all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs):
        raise CrawlError(f"{parts.hostname} is not a public address (private/internal hosts are blocked)")


class _Guard(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public(newurl, self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, *, allow_private: bool = False, timeout: float = 10) -> tuple[str, str]:
    """Return (final_url, body_text). Raises CrawlError with a short reason."""
    check_public(url, allow_private)
    opener = urllib.request.build_opener(_Guard(allow_private))
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,text/plain;q=0.9"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            kind = resp.headers.get_content_type()
            if kind not in PAGE_TYPES:
                raise CrawlError(f"Not a text page ({kind})")
            raw = resp.read(MAX_BYTES + 1)
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.geturl(), raw[:MAX_BYTES].decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise CrawlError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, LookupError) as exc:
        raise CrawlError(f"Fetch failed: {getattr(exc, 'reason', exc)}") from exc


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.links, self._text = "", [], []
        self._skip, self._in_title = 0, False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in BREAK_TAGS:
            self._text.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in BREAK_TAGS:
            self._text.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._text.append(data)

    @property
    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._text).splitlines())
        return "\n".join(line for line in lines if line)


def parse_page(html: str) -> tuple[str, str, list[str]]:
    p = _Extract()
    p.feed(html)
    p.close()
    return " ".join(p.title.split()), p.text, p.links


def same_site_links(base: str, hrefs: list[str]) -> list[str]:
    host, out, seen = urlparse(base).netloc.lower(), [], set()
    for href in hrefs:
        url = urldefrag(urljoin(base, href.strip()))[0]
        parts = urlparse(url)
        if (parts.scheme in ("http", "https") and parts.netloc.lower() == host
                and not parts.path.lower().endswith(SKIP_EXT) and url not in seen):
            seen.add(url)
            out.append(url)
    return out


def _robots(start: str, allow_private: bool) -> RobotFileParser:
    rp = RobotFileParser()
    parts = urlparse(start)
    try:
        _, body = fetch(f"{parts.scheme}://{parts.netloc}/robots.txt", allow_private=allow_private)
        rp.parse(body.splitlines())
    except CrawlError:
        rp.parse([])  # no readable robots.txt: nothing is disallowed
    return rp


def crawl(start: str, *, max_pages: int = 5, delay: float = 1.0, allow_private: bool = False,
          sleep=time.sleep) -> Iterator[dict]:
    """Breadth-first, same host only. Every URL tried counts toward max_pages."""
    check_public(start, allow_private)
    max_pages = max(1, min(int(max_pages), MAX_PAGES_LIMIT))
    start = urldefrag(start)[0]
    robots = _robots(start, allow_private)
    queue, seen, tried = deque([start]), {start}, 0
    while queue and tried < max_pages:
        url = queue.popleft()
        if tried:
            sleep(delay)
        tried += 1
        if not robots.can_fetch(UA, url):
            yield {"url": url, "title": "", "text": "", "error": "blocked by robots.txt"}
            continue
        try:
            final, html = fetch(url, allow_private=allow_private)
        except CrawlError as exc:
            yield {"url": url, "title": "", "text": "", "error": str(exc)}
            continue
        title, text, hrefs = parse_page(html)
        for link in same_site_links(final, hrefs):
            if link not in seen:
                seen.add(link)
                queue.append(link)
        yield {"url": final, "title": title, "text": text, "error": None}
