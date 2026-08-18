from __future__ import annotations

import math
import re

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import RAG_SYSTEM, evidence_block
from ..providers.openalex import OpenAlexProvider, extract_cutoff, filter_papers, search_queries
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


def anchor_coverage(paper: Paper, anchor_terms: set[str]) -> float:
    if not anchor_terms:
        return 1.0
    content = keywords(f"{paper.title} {paper.abstract}")
    return len(anchor_terms & content) / len(anchor_terms)


def matches_anchor_phrase(paper: Paper, anchor_query: str) -> bool:
    anchor = normalize_text(anchor_query)
    content = normalize_text(f"{paper.title} {paper.abstract}")
    if not anchor:
        return True
    if anchor in content:
        return True
    terms = keywords(anchor_query)
    # Longer named topics tolerate one missing modifier; two-term topics require the exact phrase.
    return len(terms) >= 4 and anchor_coverage(paper, terms) >= 0.75


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


class CitationRagPipeline:
    """Builds an ephemeral, per-task citation graph; only raw API responses are cached."""

    def __init__(self, settings: Settings, model: MiniMaxClient, scholar: OpenAlexProvider):
        self.settings = settings
        self.model = model
        self.scholar = scholar

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        terms = keywords(f"{task.application_domain} {task.prompt}")
        queries = search_queries(task.prompt, limit=5)
        anchor_query = queries[0] if queries else ""
        anchor_terms = keywords(anchor_query) if anchor_query else terms
        seeds: list[Paper] = []
        for query in queries:
            seeds.extend(self.scholar.search(query, cutoff=cutoff, limit=8))
        seeds = filter_papers(seeds, forbidden_title=task.title, cutoff=cutoff)
        seeds = [paper for paper in seeds if matches_anchor_phrase(paper, anchor_query)]
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
        # Keep the complete graph in the result for auditability, but expose only
        # its strongest nodes to the writer. Bridge nodes are useful for traversal
        # and frequently too weakly related to be safe writing evidence.
        writing_papers = papers[: self.settings.rag_evidence_papers]
        cards = evidence_block(writing_papers, self.settings.evidence_char_limit * 2)
        user = (
            f"RESEARCH TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
            f"APPLICATION DOMAIN:\n{task.application_domain}\n\n"
            "EVIDENCE CARDS (each card is an allowed source; ignore citation-count metadata as factual evidence):\n"
            f"{cards}\n\n"
            "LENGTH: Write a focused survey of 900-1,200 English words. This is a hard maximum. Prioritize the task's central taxonomy and "
            "strongest evidence; omit tangential material.\n\n"
            "MANDATORY FINAL CHECK: Before returning the report, inspect every prose sentence. Delete any factual sentence "
            "that lacks an adjacent URL or whose exact claim is not explicit in the cited evidence card. This applies to "
            "the introduction, transitions, synthesis, and conclusion as strictly as it applies to method descriptions."
        )
        report = self.model.generate(
            [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}],
            cache_namespace=f"citation-rag-report-v5:{self.settings.model}",
        )
        return report, papers
