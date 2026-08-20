from __future__ import annotations

from datetime import date
import re
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..cache import JsonCache
from ..schemas import Paper


class ArxivProvider:
    base_url = "https://export.arxiv.org/api/query"
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self, cache: JsonCache, timeout: int = 60):
        self.cache, self.timeout = cache, timeout
        self._request_lock = threading.Lock()
        self._last_request = 0.0

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        payload = {"query": query, "limit": min(limit, 30)}

        def request() -> str:
            with self._request_lock:
                delay = 3.1 - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
                terms = [term for term in re.findall(r"[A-Za-z0-9-]{3,}", query)][:5]
                search_query = " AND ".join(f"all:{term}" for term in terms)
                params = {
                    "search_query": search_query,
                    "start": 0, "max_results": min(limit, 30), "sortBy": "relevance",
                    "sortOrder": "descending",
                }
                req = Request(
                    self.base_url + "?" + urlencode(params),
                    headers={"User-Agent": "ReportBench-MiniMax/0.2"},
                )
                try:
                    with urlopen(req, timeout=self.timeout) as response:
                        return response.read().decode("utf-8")
                finally:
                    self._last_request = time.monotonic()

        xml = self.cache.get_or_create("arxiv-v1", payload, request)
        root = ElementTree.fromstring(xml)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", self.namespace):
            identifier = (entry.findtext("atom:id", default="", namespaces=self.namespace)).rsplit("/", 1)[-1]
            identifier = re.sub(r"v\d+$", "", identifier)
            published = entry.findtext("atom:published", default="", namespaces=self.namespace)
            year = int(published[:4]) if re.match(r"\d{4}", published) else None
            if cutoff and year and year > cutoff.year:
                continue
            title = " ".join(entry.findtext("atom:title", default="", namespaces=self.namespace).split())
            abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=self.namespace).split())
            papers.append(Paper(
                paper_id=f"ARXIV:{identifier}", title=title, year=year,
                url=f"https://arxiv.org/abs/{identifier}", abstract=abstract,
                source="arxiv",
            ))
        return papers

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        return None
