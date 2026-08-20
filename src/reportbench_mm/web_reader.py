from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import ipaddress
import re
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .cache import JsonCache
from .schemas import Paper


class _AcademicPageParser(HTMLParser):
    """Extract title and readable evidence without an external HTML dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.paragraphs: list[str] = []
        self._capture = ""
        self._parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {str(key).lower(): str(value) for key, value in attrs if value is not None}
        if tag in {"script", "style", "nav", "footer", "header", "form", "svg"}:
            self._ignored += 1
            return
        if tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            content = attributes.get("content", "").strip()
            if key and content:
                self.meta.setdefault(key, content)
        if self._ignored == 0 and tag in {"title", "p", "article", "h1", "h2"}:
            self._capture = tag
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "form", "svg"}:
            self._ignored = max(0, self._ignored - 1)
            return
        if self._ignored or tag != self._capture:
            return
        text = " ".join("".join(self._parts).split())
        if text:
            if tag == "title":
                self.title_parts.append(text)
            elif len(text) >= 40:
                self.paragraphs.append(text)
        self._capture = ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._ignored == 0 and self._capture:
            self._parts.append(data)


def parse_academic_html(html: str, max_chars: int = 12000) -> dict[str, str]:
    parser = _AcademicPageParser()
    parser.feed(html)
    title = next(
        (
            parser.meta.get(key, "").strip()
            for key in ("citation_title", "dc.title", "dcterms.title", "og:title", "twitter:title")
            if parser.meta.get(key, "").strip()
        ),
        parser.title_parts[0].strip() if parser.title_parts else "",
    )
    descriptions = [
        parser.meta.get(key, "").strip()
        for key in ("citation_abstract", "dc.description", "dcterms.abstract", "description", "og:description")
        if parser.meta.get(key, "").strip()
    ]
    chunks: list[str] = []
    seen: set[str] = set()
    for value in descriptions + parser.paragraphs:
        value = " ".join(value.split())
        key = value.lower()
        if len(value) >= 40 and key not in seen:
            seen.add(key)
            chunks.append(value)
    return {"title": title, "text": "\n\n".join(chunks)[:max_chars]}


def _safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return parsed.hostname.lower() not in {"localhost", "localhost.localdomain"}
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)


class WebPageReader:
    """Cached, bounded page reader used as the free Firecrawl replacement."""

    def __init__(self, cache: JsonCache, timeout: int = 12, max_bytes: int = 2_000_000):
        self.cache = cache
        self.timeout = timeout
        self.max_bytes = max_bytes

    def read(self, url: str) -> dict[str, str]:
        if not _safe_public_url(url):
            return {"title": "", "text": ""}
        payload = {"url": url, "max_bytes": self.max_bytes}

        def fetch() -> dict[str, str]:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ReportBenchResearch/1.0)",
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                },
            )
            with urlopen(request, timeout=self.timeout) as response:
                final_url = response.geturl()
                if not _safe_public_url(final_url):
                    return {"title": "", "text": ""}
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    return {"title": "", "text": ""}
                raw = response.read(self.max_bytes + 1)[: self.max_bytes]
                charset = response.headers.get_content_charset() or "utf-8"
                decoded = raw.decode(charset, errors="replace")
                if content_type == "text/plain":
                    return {"title": "", "text": " ".join(decoded.split())[:12000]}
                return parse_academic_html(decoded)

        try:
            return self.cache.get_or_create("web-page-reader-v1", payload, fetch)
        except (OSError, ValueError, socket.timeout):
            return {"title": "", "text": ""}

    def enrich(self, paper: Paper) -> Paper:
        page = self.read(paper.url)
        page_title = re.sub(r"\s+", " ", page.get("title", "")).strip()
        current = paper.title.strip()
        # Search engines often truncate titles. Prefer explicit scholarly page
        # metadata, but avoid replacing a good paper title with a site heading.
        if page_title and ("..." in current or "…" in current):
            paper.title = page_title[:500]
        paper.full_text = page.get("text", "")
        return paper

    def enrich_many(self, papers: list[Paper], workers: int = 4) -> list[Paper]:
        if not papers:
            return papers
        with ThreadPoolExecutor(max_workers=min(max(1, workers), len(papers))) as executor:
            return list(executor.map(self.enrich, papers))
