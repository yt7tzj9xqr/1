from __future__ import annotations

from datetime import date

from ..schemas import Paper


class CompositeScholarProvider:
    def __init__(self, providers):
        self.providers = list(providers)

    def search(self, query: str, *, cutoff: date | None, limit: int = 20) -> list[Paper]:
        errors: list[str] = []
        for provider in self.providers:
            try:
                papers = provider.search(query, cutoff=cutoff, limit=limit)
                if papers:
                    return papers
            except Exception as exc:
                errors.append(f"{type(provider).__name__}: {exc}")
                print(f"scholar fallback: {errors[-1]}", flush=True)
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
