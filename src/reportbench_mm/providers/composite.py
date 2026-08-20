from __future__ import annotations

from datetime import date

from ..schemas import Paper


class CompositeScholarProvider:
    def __init__(self, providers):
        self.providers = list(providers)

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        errors: list[str] = []
        pages: list[list[Paper]] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                papers = provider.search(query, cutoff=cutoff, limit=limit)
                if papers:
                    pages.append(papers)
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
