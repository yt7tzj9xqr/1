from __future__ import annotations

from datetime import date
import html
import json
import re
import threading
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..schemas import Paper


class CrossrefProvider:
    base_url = "https://api.crossref.org"

    def __init__(self, cache: JsonCache, mailto: str = "", timeout: int = 60):
        self.cache, self.mailto, self.timeout = cache, mailto, timeout
        self._request_lock = threading.Lock()
        self._last_request = 0.0

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
        if self.mailto:
            params["mailto"] = self.mailto
        payload = {"path": path, "params": params}

        def request() -> Any:
            url = f"{self.base_url}{path}" + (("?" + urlencode(params)) if params else "")
            req = Request(url, headers={"User-Agent": "ReportBench-MiniMax/0.1 (mailto: anonymous)"})
            with self._request_lock:
                delay = 0.25 - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
                try:
                    with urlopen(req, timeout=self.timeout) as response:
                        return json.loads(response.read().decode("utf-8"))
                finally:
                    self._last_request = time.monotonic()

        return self.cache.get_or_create("crossref-v1", payload, request)

    @staticmethod
    def _year(item: dict[str, Any]) -> int | None:
        parts = ((item.get("published-print") or item.get("published-online") or {}).get("date-parts") or [[]])
        return int(parts[0][0]) if parts and parts[0] else None

    @classmethod
    def _paper(cls, item: dict[str, Any], depth: int = 0) -> Paper:
        doi = item.get("DOI") or ""
        abstract = html.unescape(re.sub(r"<[^>]+>", " ", item.get("abstract") or ""))
        refs = [f"CR:{ref['DOI']}" for ref in item.get("reference") or [] if ref.get("DOI")]
        return Paper(
            paper_id=f"CR:{doi}", title=" ".join(item.get("title") or []), year=cls._year(item),
            url=(f"https://doi.org/{doi}" if doi else item.get("URL") or ""), abstract=" ".join(abstract.split()),
            doi=doi, cited_by_count=int(item.get("is-referenced-by-count") or 0),
            referenced_work_ids=refs, depth=depth, source="crossref",
        )

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(limit, 50)}
        if cutoff:
            params["filter"] = f"until-pub-date:{cutoff.isoformat()}"
        data = self._get("/works", params)
        return [self._paper(item) for item in data.get("message", {}).get("items", [])]

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        doi = paper_id.removeprefix("CR:")
        try:
            return self._paper(self._get(f"/works/{quote(doi, safe='')}" )["message"], depth)
        except Exception:
            return None
