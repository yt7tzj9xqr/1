from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import html
import re

from ..schemas import Paper


def _title_key(value: str) -> str:
    value = html.unescape(value).strip()
    value = re.sub(r"^\[(?:19|20)?\d{2,4}\.\d+(?:v\d+)?\]\s*", "", value)
    value = re.sub(r"^(?:GitHub\s+-\s+[^:]+:\s*|[^ /]+/[^ ]+\s+-\s+)", "", value, flags=re.I)
    value = re.sub(r"\s+(?:\||»|-)\s+(?:Request PDF|PMC|PubMed|Springer Nature|GitHub)\s*$", "", value, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _paper_key(paper: Paper) -> str:
    return (paper.doi or _title_key(paper.title)).lower()


def _url_quality(url: str) -> int:
    value = url.lower()
    if "arxiv.org/abs/" in value:
        return 4
    if value.startswith("https://doi.org/"):
        return 3
    if "semanticscholar.org/paper/" in value:
        return 2
    return 1 if value else 0


def merge_paper_metadata(target: Paper, candidate: Paper) -> Paper:
    """Merge a search hit with structured scholarly metadata.

    MiniMax web search is useful for discovery, while OpenAlex/S2/Crossref are
    useful for stable identifiers and citation edges. Keeping only whichever
    provider ranked a title first silently discarded the graph metadata needed
    by the RAG pipeline.
    """
    target_was_web_only = target.paper_id.startswith("MMSEARCH:")
    if len(candidate.abstract) > len(target.abstract):
        target.abstract = candidate.abstract
    if len(candidate.full_text) > len(target.full_text):
        target.full_text = candidate.full_text
    if not target.doi and candidate.doi:
        target.doi = candidate.doi
    if target.year is None and candidate.year is not None:
        target.year = candidate.year
    target.cited_by_count = max(target.cited_by_count, candidate.cited_by_count)
    target.referenced_work_ids = list(dict.fromkeys(
        target.referenced_work_ids + candidate.referenced_work_ids
    ))
    if (
        target_was_web_only
        and candidate.paper_id
        and not candidate.paper_id.startswith("MMSEARCH:")
    ):
        target.paper_id = candidate.paper_id
    if _url_quality(candidate.url) > _url_quality(target.url):
        target.url = candidate.url
    sources = [item for item in target.source.split("+") if item]
    if candidate.source not in sources:
        sources.append(candidate.source)
    target.source = "+".join(sources)
    return target


class CompositeScholarProvider:
    def __init__(self, providers):
        self.providers = list(providers)

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        errors: list[str] = []
        pages: list[list[Paper]] = []
        seen: set[str] = set()
        for provider_index, provider in enumerate(self.providers):
            try:
                papers = provider.search(query, cutoff=cutoff, limit=limit)
                if papers:
                    pages.append(papers)
                # MiniMax Coding Plan search is the free replacement for the
                # paper's SerpAPI. Avoid exhausting anonymous scholarly APIs
                # when it already returned a complete search page.
                if provider_index == 0 and type(provider).__name__ == "MiniMaxSearchProvider" and papers:
                    return papers[:limit]
            except Exception as exc:
                errors.append(f"{type(provider).__name__}: {exc}")
                print(f"scholar fallback: {errors[-1]}", flush=True)
        # Interleave provider ranks so OpenAlex cannot monopolize the result page.
        collected: list[Paper] = []
        for rank in range(max((len(page) for page in pages), default=0)):
            for page in pages:
                if rank >= len(page):
                    continue
                paper = page[rank]
                key = (paper.doi or " ".join(paper.title.lower().split())).lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append(paper)
                if len(collected) >= limit:
                    return collected
        if collected:
            return collected[:limit]
        raise RuntimeError("All free scholarly providers failed: " + " | ".join(errors))

    def search_many(
        self, queries: list[str], *, cutoff: date | None, limit: int = 20, workers: int = 5,
    ) -> list[Paper]:
        """Run the full MiniMax search budget plus one free structured search.

        Calling every anonymous scholarly API for every query is slow and is
        likely to trigger rate limits. Conversely, returning the MiniMax page
        immediately loses citation edges. The central query is therefore sent
        once to each structured provider, matching the useful behavior found in
        the reference run, while MiniMax handles all planned subtopic queries.
        """
        if not queries:
            return []
        primary = self.providers[0]
        jobs = [(index, primary, query) for index, query in enumerate(queries)]
        jobs.extend((0, provider, queries[0]) for provider in self.providers[1:])
        rows: list[tuple[int, int, Paper]] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(workers + 2, len(jobs)))) as executor:
            futures = {
                executor.submit(provider.search, query, cutoff=cutoff, limit=limit):
                (query_index, provider)
                for query_index, provider, query in jobs
            }
            for future in as_completed(futures):
                query_index, provider = futures[future]
                try:
                    rows.extend(
                        (query_index, rank, paper)
                        for rank, paper in enumerate(future.result())
                    )
                except Exception as exc:
                    message = f"{type(provider).__name__}: {exc}"
                    errors.append(message)
                    print(f"scholar supplement failed: {message}", flush=True)

        merged: dict[str, tuple[Paper, set[int]]] = {}
        aliases: dict[str, str] = {}
        for query_index, rank, paper in sorted(rows, key=lambda item: (item[0], item[1])):
            title_alias = _title_key(paper.title)
            doi_alias = paper.doi.lower() if paper.doi else ""
            key = aliases.get(doi_alias) or aliases.get(title_alias)
            if not key:
                key = doi_alias or title_alias
            if not key:
                continue
            if key not in merged:
                paper.search_query_index = query_index
                paper.search_rank = rank
                merged[key] = (paper, {query_index})
                if title_alias:
                    aliases[title_alias] = key
                if doi_alias:
                    aliases[doi_alias] = key
                continue
            target, query_hits = merged[key]
            query_hits.add(query_index)
            merge_paper_metadata(target, paper)
            if title_alias:
                aliases[title_alias] = key
            if doi_alias:
                aliases[doi_alias] = key
            if rank < target.search_rank:
                target.search_rank = rank
                target.search_query_index = query_index
        papers = []
        for paper, query_hits in merged.values():
            paper.query_hits = len(query_hits)
            papers.append(paper)
        if not papers:
            raise RuntimeError("All free scholarly providers failed: " + " | ".join(errors))
        return sorted(papers, key=lambda paper: (paper.search_query_index, paper.search_rank))

    def get_work(self, paper_id: str, depth: int = 0) -> Paper | None:
        if paper_id.startswith("S2:"):
            candidates = [provider for provider in self.providers if type(provider).__name__ == "SemanticScholarProvider"]
        elif paper_id.startswith("CR:"):
            candidates = [provider for provider in self.providers if type(provider).__name__ == "CrossrefProvider"]
        else:
            candidates = self.providers
        for provider in candidates:
            paper = provider.get_work(paper_id, depth=depth)
            if paper:
                return paper
        return None

    def get_works(self, paper_ids: list[str], depth: int = 0) -> list[Paper]:
        """Batch OpenAlex graph nodes and bound slower singleton fallbacks."""
        wanted = list(dict.fromkeys(paper_ids))
        collected: dict[str, Paper] = {}
        openalex_provider = next(
            (provider for provider in self.providers if type(provider).__name__ == "OpenAlexProvider"),
            None,
        )
        openalex_ids = [paper_id for paper_id in wanted if re.fullmatch(r"W\d+", paper_id)]
        if openalex_provider and openalex_ids:
            for paper in openalex_provider.get_works(openalex_ids, depth=depth):
                collected[paper.paper_id] = paper
        unresolved = [paper_id for paper_id in wanted if paper_id not in collected and not re.fullmatch(r"W\d+", paper_id)]
        # Citation lists can contain hundreds of nodes. The graph policy has
        # already ranked and bounded each layer, so singleton fallbacks are
        # deliberately capped to avoid an API-call explosion.
        for paper_id in unresolved[:12]:
            paper = self.get_work(paper_id, depth=depth)
            if paper:
                collected[paper.paper_id] = paper
        return list(collected.values())
