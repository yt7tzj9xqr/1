from __future__ import annotations

from datetime import date
import json
import re
import time
import threading
from typing import Any
from urllib.error import HTTPError, URLError
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


QUERY_STOP_WORDS = {
    "please", "help", "research", "researching", "application", "field", "within", "aim", "understand",
    "focus", "summarizing", "analyzing", "methods", "based", "different", "types", "examining", "developments",
    "characteristics", "ensure", "referenced", "papers", "published", "before", "such", "including", "domain",
    "that", "this", "with", "from", "into", "were", "all", "and", "the", "for", "conducting",
    "literature", "review", "explore", "exploring", "utilized", "provide", "write", "detailed",
    "comprehensive", "academic", "specific", "requirements", "follows", "current", "state",
}


def compact_query(prompt: str, max_terms: int = 18) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", prompt):
        normalized = word.lower()
        if normalized in QUERY_STOP_WORDS or normalized in seen or re.fullmatch(r"20\d{2}", normalized):
            continue
        seen.add(normalized)
        terms.append(word)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def search_queries(prompt: str, limit: int = 5) -> list[str]:
    """Create short, high-recall scholarly queries without using the forbidden survey title."""
    candidates: list[str] = []
    # Preserve the technical subject together with its application domain.
    # A bare ``field of X`` match (for example, "autonomous driving") is often
    # much broader than the requested method family ("radar data representations").
    subject_domain_patterns = [
        r"(?:advancements?|developments?|progress)\s+in\s+(?:different\s+)?(.{3,80}?)\s+in\s+(?:the\s+)?field\s+of\s+([^.,;:\n]{3,70})",
        r"(.{3,70}?(?:methods?|approaches|techniques|models|representations?))\s+in\s+(?:the\s+)?field\s+of\s+([^.,;:\n]{3,70})",
    ]
    for pattern in subject_domain_patterns:
        for subject, domain in re.findall(pattern, prompt, flags=re.I):
            candidates.append(f"{subject} {domain}")
            candidates.append(subject)
    # Explicitly named and grammatical domain phrases are more reliable than
    # arbitrary capitalized fragments such as "Please Focus Research".
    candidates.extend(re.findall(r'["“]([^"”]{4,100})["”]', prompt))
    # Capitalized technical phrases often name the actual method family.
    candidates.extend(
        match.strip()
        for match in re.findall(r"(?:[A-Z][A-Za-z0-9-]+(?:\s+|$)){2,5}", prompt)
    )
    domain_patterns = [
        r"application of\s+(.{3,60}?)\s+(?:in|to)\s+(?:the\s+)?(?:field of\s+)?([^.,;:\n]{3,70})",
        r"(?:field|domain|area) of\s+([^.,;:\n]{4,100})",
        r"research(?:ing)?\s+(?:studies related to\s+)?([^.,;:\n]{4,90})",
    ]
    for pattern in domain_patterns:
        for match in re.findall(pattern, prompt, flags=re.I):
            candidates.append(" ".join(match) if isinstance(match, tuple) else match)
    # Parenthetical examples provide valuable method/subtopic queries.
    for group in re.findall(r"\(([^)]{4,160})\)", prompt):
        cleaned = re.sub(r"\b(?:e\.g\.|such as|and|or)\b", " ", group, flags=re.I)
        parts = re.split(r"[,;/]", cleaned)
        candidates.extend(part.strip() for part in parts)
    terms = compact_query(prompt, max_terms=20).split()
    candidates.extend(" ".join(terms[index:index + 4]) for index in range(0, min(len(terms), 16), 4))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = re.split(
            r"\b(?:before|on or before|published before|published on or before|ensure|note that|and ensure)\b",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0]
        candidate = " ".join(candidate.split()).strip(" .,:;-")
        candidate = candidate.strip('"“”')
        key = candidate.lower()
        if len(candidate) < 4 or len(candidate.split()) < 2 or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique or [compact_query(prompt, max_terms=8)]


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
        self._request_lock = threading.Lock()
        self._last_request = 0.0

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
            with self._request_lock:
                throttle = 1.1 - (time.monotonic() - self._last_request)
                if throttle > 0:
                    time.sleep(throttle)
                last_error: Exception | None = None
                try:
                    for attempt in range(8):
                        try:
                            with urlopen(req, timeout=self.timeout) as response:
                                return json.loads(response.read().decode("utf-8"))
                        except HTTPError as exc:
                            last_error = exc
                            if exc.code not in {429, 500, 502, 503, 504} or attempt == 7:
                                raise
                        except (URLError, TimeoutError) as exc:
                            last_error = exc
                            if attempt == 7:
                                raise
                        retry_header = last_error.headers.get("Retry-After", "") if isinstance(last_error, HTTPError) else ""
                        retry_after = int(retry_header) if retry_header.isdigit() else 0
                        if retry_after > 300:
                            raise last_error
                        delay = min(60, max(retry_after, 2 ** attempt))
                        print(f"OpenAlex transient error; retry {attempt + 2}/8 in {delay}s", flush=True)
                        time.sleep(delay)
                finally:
                    self._last_request = time.monotonic()
            raise RuntimeError(f"OpenAlex request failed: {last_error}")

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
        # ReportBench gold sets contain conference papers, preprints, books,
        # datasets and standards in addition to journal articles/reviews.
        filters: list[str] = []
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
        title_without_identifier = re.sub(r"^\d{4}\s+\d+\s+", "", title)
        forbidden_overlap = (
            len(title_without_identifier) >= 18
            and (
                forbidden.startswith(title_without_identifier)
                or title_without_identifier.startswith(forbidden)
            )
        )
        if not title or title == forbidden or forbidden_overlap or title in seen:
            continue
        if cutoff and paper.year and paper.year > cutoff.year:
            continue
        seen.add(title)
        accepted.append(paper)
    return accepted
