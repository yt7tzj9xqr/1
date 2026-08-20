from __future__ import annotations

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import BASELINE_SYSTEM, evidence_block
from ..providers.openalex import extract_cutoff, filter_papers
from ..retrieval import diverse_top_papers, is_scholarly_candidate, parallel_search, plan_search_queries
from .rag import anchor_coverage, keywords, matches_anchor_phrase, score_paper
from ..schemas import Paper, Task
from ..web_reader import WebPageReader


class BaselinePipeline:
    def __init__(self, settings: Settings, model: MiniMaxClient, scholar, reader: WebPageReader | None = None):
        self.settings = settings
        self.model = model
        self.scholar = scholar
        self.reader = reader

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        queries = plan_search_queries(task, self.model, self.settings.baseline_search_budget)
        print(f"{task.arxiv_id} search queries: {queries}", flush=True)
        found = parallel_search(
            self.scholar, queries, cutoff=cutoff,
            per_query=self.settings.search_results_per_query, workers=self.settings.search_workers,
        )
        found = filter_papers(found, forbidden_title=task.title, cutoff=cutoff)
        query_terms = keywords(task.prompt)
        for paper in found:
            semantic = score_paper(paper, query_terms)
            rank_bonus = 1.0 / (1.0 + max(0, paper.search_rank))
            paper.relevance = 0.72 * semantic + 0.23 * rank_bonus + 0.05 * min(2, paper.query_hits) / 2
        ranked = sorted(
            (
                paper for paper in found
                if paper.abstract and paper.url and paper.relevance >= 0.06 and is_scholarly_candidate(paper)
            ),
            key=lambda paper: paper.relevance,
            reverse=True,
        )
        # Query planning already constrains every search to the central topic;
        # an exact first-query anchor discarded valid subtopic results.
        selected = diverse_top_papers(ranked, len(queries), self.settings.baseline_papers)
        if self.reader:
            selected = self.reader.enrich_many(selected, self.settings.reader_workers)
        return [
            paper for paper in filter_papers(selected, forbidden_title=task.title, cutoff=cutoff)
            if is_scholarly_candidate(paper)
        ]

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
            cache_namespace=f"baseline-report-v3:{self.settings.model}",
        )
        return report, papers
