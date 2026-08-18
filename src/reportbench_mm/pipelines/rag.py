from __future__ import annotations

import math
import re

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import RAG_SYSTEM, evidence_block
from ..providers.openalex import OpenAlexProvider, compact_query, extract_cutoff, filter_papers
from ..schemas import Paper, Task


STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by", "from", "before",
    "please", "help", "research", "paper", "papers", "field", "academic", "review", "ensure", "only",
}


def keywords(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if word not in STOP_WORDS}


def score_paper(paper: Paper, query_terms: set[str]) -> float:
    paper_terms = keywords(f"{paper.title} {paper.abstract}")
    overlap = len(query_terms & paper_terms) / max(1, len(query_terms))
    impact = math.log1p(max(0, paper.cited_by_count)) / 15.0
    abstract_bonus = 0.1 if len(paper.abstract) >= 300 else 0.0
    depth_penalty = paper.depth * 0.04
    return 0.72 * overlap + 0.18 * min(1.0, impact) + abstract_bonus - depth_penalty


class CitationRagPipeline:
    """Builds an ephemeral, per-task citation graph; only raw API responses are cached."""

    def __init__(self, settings: Settings, model: MiniMaxClient, scholar: OpenAlexProvider):
        self.settings = settings
        self.model = model
        self.scholar = scholar

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        terms = keywords(f"{task.application_domain} {task.prompt}")
        seeds = self.scholar.search(compact_query(task.prompt), cutoff=cutoff, limit=max(12, self.settings.rag_seed_count * 4))
        seeds = filter_papers(seeds, forbidden_title=task.title, cutoff=cutoff)
        for paper in seeds:
            paper.relevance = score_paper(paper, terms)
        seeds.sort(key=lambda paper: paper.relevance, reverse=True)
        frontier = seeds[: self.settings.rag_seed_count]
        graph: dict[str, Paper] = {paper.paper_id: paper for paper in frontier}

        # Fixed total budget prevents exponential growth and makes every task resumable.
        per_parent = {1: 5, 2: 3, 3: 2}
        for depth in range(1, self.settings.rag_depth + 1):
            next_frontier: list[Paper] = []
            for parent in frontier:
                for reference_id in parent.referenced_work_ids[: per_parent.get(depth, 2)]:
                    if len(graph) >= self.settings.rag_max_papers:
                        break
                    if reference_id in graph:
                        continue
                    paper = self.scholar.get_work(reference_id, depth=depth)
                    if not paper:
                        continue
                    kept = filter_papers([paper], forbidden_title=task.title, cutoff=cutoff)
                    if not kept:
                        continue
                    paper = kept[0]
                    paper.relevance = score_paper(paper, terms)
                    # Weakly related references are graph context, not writing evidence.
                    if paper.relevance < 0.08:
                        continue
                    graph[paper.paper_id] = paper
                    next_frontier.append(paper)
                if len(graph) >= self.settings.rag_max_papers:
                    break
            next_frontier.sort(key=lambda paper: paper.relevance, reverse=True)
            frontier = next_frontier[: max(4, self.settings.rag_seed_count * 2)]
            if not frontier or len(graph) >= self.settings.rag_max_papers:
                break

        papers = [paper for paper in graph.values() if paper.abstract and paper.url]
        papers.sort(key=lambda paper: (paper.relevance, -paper.depth), reverse=True)
        return papers[: self.settings.rag_max_papers]

    def run(self, task: Task) -> tuple[str, list[Paper]]:
        papers = self.retrieve(task)
        if not papers:
            raise RuntimeError(f"No usable RAG evidence found for {task.arxiv_id}")
        cards = evidence_block(papers, self.settings.evidence_char_limit * 2)
        user = (
            f"RESEARCH TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
            f"APPLICATION DOMAIN:\n{task.application_domain}\n\n"
            "EVIDENCE CARDS (each card is an allowed source; ignore citation-count metadata as factual evidence):\n"
            f"{cards}"
        )
        report = self.model.generate(
            [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}],
            cache_namespace=f"citation-rag-report-v1:{self.settings.model}",
        )
        return report, papers
