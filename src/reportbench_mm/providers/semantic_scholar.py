from __future__ import annotations

from datetime import date
import json
import time
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..schemas import Paper


class SemanticScholarProvider:
    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = "paperId,title,year,url,abstract,citationCount,externalIds,references.paperId"

    def __init__(self, cache: JsonCache, timeout: int = 60):
        self.cache = cache
        self.timeout = timeout
        self._request_lock = threading.Lock()
        self._last_request = 0.0

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
        payload = {"path": path, "params": params}

        def request() -> Any:
            url = f"{self.base_url}{path}"
            if params:
                url += "?" + urlencode(params)
            req = Request(url, headers={"User-Agent": "ReportBench-MiniMax/0.1"})
            with self._request_lock:
                delay = 1.1 - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
                try:
                    for attempt in range(5):
                        try:
                            with urlopen(req, timeout=self.timeout) as response:
                                return json.loads(response.read().decode("utf-8"))
                        except HTTPError as exc:
                            if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                                raise
                        except (URLError, TimeoutError):
                            if attempt == 4:
                                raise
                        time.sleep(min(16, 2 ** attempt))
                finally:
                    self._last_request = time.monotonic()
            raise RuntimeError("Semantic Scholar request failed")

        return self.cache.get_or_create("semantic-scholar-v1", payload, request)

    @staticmethod
    def _paper(item: dict[str, Any], depth: int = 0) -> Paper:
        external = item.get("externalIds") or {}
        doi = external.get("DOI") or ""
        arxiv = external.get("ArXiv") or ""
        url = item.get("url") or (f"https://doi.org/{doi}" if doi else "")
        if arxiv:
            url = f"https://arxiv.org/abs/{arxiv}"
        references = [
            f"S2:{ref['paperId']}" for ref in item.get("references") or [] if ref and ref.get("paperId")
        ]
        return Paper(
            paper_id=f"S2:{item.get('paperId', '')}", title=item.get("title") or "", year=item.get("year"),
            url=url, abstract=item.get("abstract") or "", doi=doi,
            cited_by_count=int(item.get("citationCount") or 0), referenced_work_ids=references,
            depth=depth, source="semantic-scholar",
        )

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        data = self._get("/paper/search", {"query": query, "limit": min(limit, 50), "fields": self.fields})
        papers = [self._paper(item) for item in data.get("data", [])]
        return [paper for paper in papers if not cutoff or not paper.year or paper.year <= cutoff.year]

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        identifier = paper_id.removeprefix("S2:")
        try:
            return self._paper(self._get(f"/paper/{quote(identifier, safe='')}", {"fields": self.fields}), depth)
        except Exception:
            return None
