from __future__ import annotations

from datetime import date
import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..schemas import Paper


class MiniMaxSearchProvider:
    """Official Coding Plan web_search endpoint; no SerpAPI key required."""

    def __init__(self, cache: JsonCache, api_key: str, base_url: str, timeout: int = 120):
        self.cache, self.api_key, self.base_url, self.timeout = cache, api_key, base_url.rstrip("/"), timeout

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        search_query = f"{query} before:{cutoff.isoformat()}" if cutoff else query
        payload = {"q": search_query}

        def request() -> dict[str, Any]:
            req = Request(
                f"{self.base_url}/coding_plan/search",
                data=json.dumps(payload).encode("utf-8"), method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            for attempt in range(6):
                try:
                    with urlopen(req, timeout=self.timeout) as response:
                        return json.loads(response.read().decode("utf-8"))
                except HTTPError as exc:
                    if exc.code not in {429, 500, 502, 503, 504, 529} or attempt == 5:
                        detail = exc.read().decode("utf-8", errors="replace")[:300]
                        raise RuntimeError(f"MiniMax search HTTP {exc.code}: {detail}") from exc
                except (URLError, TimeoutError, ConnectionResetError) as exc:
                    if attempt == 5:
                        raise RuntimeError(f"MiniMax search connection failed: {exc}") from exc
                time.sleep(min(16, 2 ** attempt))
            raise RuntimeError("MiniMax search failed")

        data = self.cache.get_or_create("minimax-web-search-v1", payload, request)
        papers: list[Paper] = []
        for index, item in enumerate(data.get("organic") or []):
            title = re.sub(r"\s+", " ", item.get("title") or "").strip()
            url = item.get("link") or ""
            snippet = re.sub(r"\s+", " ", item.get("snippet") or "").strip()
            raw_date = str(item.get("date") or "")
            match = re.search(r"(?:19|20)\d{2}", f"{raw_date} {title} {snippet}")
            year = int(match.group()) if match else None
            if cutoff and year and year > cutoff.year:
                continue
            if title and url:
                papers.append(Paper(
                    paper_id=f"MMSEARCH:{abs(hash(url))}:{index}", title=title, year=year,
                    url=url, abstract=snippet, source="minimax-search",
                ))
            if len(papers) >= limit:
                break
        return papers

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        return None
