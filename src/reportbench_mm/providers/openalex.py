from __future__ import annotations

from datetime import date
import json
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..schemas import Paper


MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    )
    if name
}


def extract_cutoff(prompt: str) -> date | None:
    matches = re.findall(
        r"(?:before|prior to|on or before)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})",
        prompt,
        flags=re.I,
    )
    if not matches:
        matches = re.findall(
            r"(?:before|prior to|on or before)\s+(20\d{2})",
            prompt,
            flags=re.I,
        )
        return date(int(matches[-1]), 12, 31) if matches else None
    month, year = matches[-1]
    return date(int(year), MONTHS[month.lower()], 1)


def normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = sorted((position, word) for word, offsets in index.items() for position in offsets)
    return " ".join(word for _, word in positions)


class OpenAlexProvider:
    base_url = "https://api.openalex.org"

    def __init__(self, cache: JsonCache, mailto: str = "", timeout: int = 60):
        self.cache = cache
        self.mailto = mailto
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
        if self.mailto:
            params["mailto"] = self.mailto
        payload = {"path": path, "params": params}

        def request() -> Any:
            url = f"{self.base_url}{path}"
            if params:
                url += "?" + urlencode(params)
            req = Request(url, headers={"User-Agent": "ReportBench-MiniMax/0.1"})
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        return self.cache.get_or_create("openalex-v1", payload, request)

    @staticmethod
    def _paper(work: dict[str, Any], depth: int = 0) -> Paper:
        primary = work.get("primary_location") or {}
        best = work.get("best_oa_location") or {}
        url = best.get("landing_page_url") or primary.get("landing_page_url") or work.get("id", "")
        doi = (work.get("doi") or "").removeprefix("https://doi.org/")
        return Paper(
            paper_id=(work.get("id") or "").rsplit("/", 1)[-1],
            title=work.get("display_name") or "",
            year=work.get("publication_year"),
            url=url,
            abstract=_abstract(work.get("abstract_inverted_index")),
            doi=doi,
            cited_by_count=int(work.get("cited_by_count") or 0),
            referenced_work_ids=[item.rsplit("/", 1)[-1] for item in work.get("referenced_works", [])],
            depth=depth,
        )

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        filters = ["type:article|review"]
        if cutoff:
            filters.append(f"from_publication_date:1900-01-01")
            filters.append(f"to_publication_date:{cutoff.isoformat()}")
        data = self._get(
            "/works",
            {"search": query, "filter": ",".join(filters), "per-page": min(limit, 50), "sort": "relevance_score:desc"},
        )
        return [self._paper(work) for work in data.get("results", [])]

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        try:
            return self._paper(self._get(f"/works/{paper_id}"), depth=depth)
        except Exception:
            return None


def filter_papers(papers: list[Paper], *, forbidden_title: str, cutoff: date | None) -> list[Paper]:
    forbidden = normalize_title(forbidden_title)
    seen: set[str] = set()
    accepted: list[Paper] = []
    for paper in papers:
        title = normalize_title(paper.title)
        if not title or title == forbidden or title in seen:
            continue
        if cutoff and paper.year and paper.year > cutoff.year:
            continue
        seen.add(title)
        accepted.append(paper)
    return accepted

