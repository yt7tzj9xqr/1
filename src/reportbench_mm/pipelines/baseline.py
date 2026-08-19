from __future__ import annotations

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import BASELINE_SYSTEM, evidence_block
from ..providers.openalex import extract_cutoff, filter_papers, search_queries
from .rag import anchor_coverage, keywords, matches_anchor_phrase, score_paper
from ..schemas import Paper, Task


class BaselinePipeline:
    def __init__(self, settings: Settings, model: MiniMaxClient, scholar):
        self.settings = settings
        self.model = model
        self.scholar = scholar

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        queries = search_queries(task.prompt, limit=self.settings.baseline_search_budget)
        found: list[Paper] = []
        for query in queries:
            found.extend(self.scholar.search(query, cutoff=cutoff, limit=10))
        found = filter_papers(found, forbidden_title=task.title, cutoff=cutoff)
        query_terms = keywords(task.prompt)
        anchor_query = queries[0] if queries else ""
        for paper in found:
            paper.relevance = score_paper(paper, query_terms)
        ranked = sorted(
            (
                paper for paper in found
                if paper.abstract and paper.url and paper.relevance >= 0.08
            ),
            key=lambda paper: paper.relevance,
            reverse=True,
        )
        usable = [paper for paper in ranked if matches_anchor_phrase(paper, anchor_query)]
        if not usable:
            anchor_terms = keywords(anchor_query)
            usable = [
                paper for paper in ranked
                if anchor_coverage(paper, anchor_terms) >= 0.5 or paper.relevance >= 0.16
            ]
        return usable[: self.settings.baseline_papers]

    def run(self, task: Task) -> tuple[str, list[Paper]]:
        papers = self.retrieve(task)
        if not papers:
            raise RuntimeError(f"No usable scholarly sources found for {task.arxiv_id}")
        user = (
            f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
            f"DOMAIN:\n{task.application_domain}\n\nSOURCES:\n"
            + evidence_block(papers, self.settings.evidence_char_limit)
        )
        report = self.model.generate(
            [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}],
            cache_namespace=f"baseline-report-v1:{self.settings.model}",
        )
        return report, papers
