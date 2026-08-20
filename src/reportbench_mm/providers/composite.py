from __future__ import annotations

from datetime import date

from ..schemas import Paper


class CompositeScholarProvider:
    def __init__(self, providers):
        self.providers = list(providers)

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        errors: list[str] = []
        collected: list[Paper] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                papers = provider.search(query, cutoff=cutoff, limit=limit)
                for paper in papers:
                    key = " ".join(paper.title.lower().split())
                    if key and key not in seen:
                        seen.add(key)
                        collected.append(paper)
                usable = sum(bool(paper.abstract and paper.url) for paper in collected)
                # Avoid extra public-API traffic when the first provider already
                # supplied a useful page; otherwise let the next free provider
                # repair sparse or abstract-less search results.
                if usable >= min(limit, max(3, limit // 2)):
                    return collected[:limit]
            except Exception as exc:
                errors.append(f"{type(provider).__name__}: {exc}")
                print(f"scholar fallback: {errors[-1]}", flush=True)
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
