from __future__ import annotations

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import BASELINE_SYSTEM, evidence_block, repair_grounded_report
from ..providers.openalex import extract_cutoff, filter_papers
from ..retrieval import (
    diverse_top_papers, is_scholarly_candidate, model_rerank_papers,
    parallel_search, plan_search_queries,
)
from .rag import anchor_coverage, keywords, matches_anchor_phrase, sanitize_report, score_paper
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
        found = parallel_search(
            self.scholar, queries, cutoff=cutoff,
            per_query=self.settings.search_results_per_query, workers=self.settings.search_workers,
        )
        print(f"{task.arxiv_id} search queries: {queries}", flush=True)
        found = filter_papers(found, forbidden_title=task.title, cutoff=cutoff)
        query_terms = keywords(task.prompt)
        anchor_terms = keywords(queries[0]) if queries else query_terms
        for paper in found:
            semantic = score_paper(paper, query_terms)
            anchor = anchor_coverage(paper, anchor_terms)
            rank_bonus = 1.0 / (1.0 + max(0, paper.search_rank))
            paper.relevance = (
                0.50 * semantic + 0.30 * anchor + 0.15 * rank_bonus
                + 0.05 * min(2, paper.query_hits) / 2
            )
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
        candidate_limit = max(self.settings.baseline_papers, self.settings.retrieval_candidate_pool)
        candidates = diverse_top_papers(ranked, len(queries), candidate_limit)
        selected = model_rerank_papers(
            task, candidates, self.model, self.settings.baseline_papers,
        )
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
            + "\n\nLENGTH: Write 700-850 English words. This is a hard range. Delete any factual sentence without one directly supporting URL."
        )
        report = self.model.generate(
            [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}],
            cache_namespace=f"baseline-report-v4:{self.settings.model}",
        )
        report = repair_grounded_report(
            report, papers, self.model,
            f"baseline-evidence-repair-v1:{self.settings.model}", "650-800",
        )
        report = sanitize_report(report)
        return report, papers
