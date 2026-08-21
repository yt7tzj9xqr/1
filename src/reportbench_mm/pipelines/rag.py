from __future__ import annotations

import math
import re

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import (
    RAG_SYSTEM, evidence_block, generated_report_is_usable,
    recover_sanitized_report, repair_grounded_report,
)
from ..providers.openalex import extract_cutoff, filter_papers, search_queries
from ..retrieval import (
    diverse_top_papers, is_scholarly_candidate, model_rerank_papers,
    parallel_search, plan_search_queries,
)
from ..schemas import Paper, Task
from ..web_reader import WebPageReader


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


SECONDARY_LITERATURE_RE = re.compile(
    r"\b(?:survey|review|overview|meta-analysis|bibliometric|systematic mapping)\b", re.I
)


def writing_score(paper: Paper, query_terms: set[str], anchor_terms: set[str]) -> float:
    """Rank evidence for citation accuracy, separately from graph traversal.

    ReportBench's gold references predominantly reward central primary/canonical
    work. Deep graph nodes were useful as traversal bridges in the pilot, but no
    cited depth-2/3 node matched gold, so depth is deliberately expensive here.
    """
    coverage = anchor_coverage(paper, anchor_terms)
    title_overlap = len(keywords(paper.title) & query_terms) / max(1, len(query_terms))
    impact = min(1.0, math.log1p(max(0, paper.cited_by_count)) / 10.0)
    secondary_penalty = 0.09 if paper.depth == 0 and SECONDARY_LITERATURE_RE.search(paper.title) else 0.0
    # A highly lexical direct-search match is not necessarily a canonical paper.
    # Keep low-impact work available, but rank it below established citation-path
    # evidence when both are otherwise relevant.
    low_impact_direct_penalty = 0.13 if paper.depth == 0 and paper.cited_by_count < 100 else 0.0
    return (
        0.42 * paper.relevance
        + 0.25 * coverage
        + 0.16 * title_overlap
        + 0.17 * impact
        - 0.18 * max(0, paper.depth - 1)
        - secondary_penalty
        - low_impact_direct_penalty
    )


def select_writing_papers(papers: list[Paper], task: Task, limit: int) -> list[Paper]:
    """Select auditable evidence while keeping deep nodes traversal-only."""
    queries = search_queries(task.prompt, limit=5)
    anchor_terms = keywords(queries[0]) if queries else keywords(task.application_domain)
    query_terms = keywords(f"{task.application_domain} {task.prompt}")
    eligible = [
        paper for paper in papers
        if paper.depth <= 1
        and paper.abstract
        and paper.url
        and (
            paper.depth == 0
            or anchor_coverage(paper, anchor_terms) >= 0.25
            or paper.relevance >= 0.20
        )
    ]
    ranked_direct = sorted(
        (paper for paper in eligible if paper.depth == 0),
        key=lambda paper: writing_score(paper, query_terms, anchor_terms), reverse=True,
    )
    ranked_graph = sorted(
        (paper for paper in eligible if paper.depth == 1),
        key=lambda paper: writing_score(paper, query_terms, anchor_terms), reverse=True,
    )
    # The broad direct pool has already been selected by M3 from the full search
    # page. Citation edges supplement that evidence; they must not displace it
    # wholesale merely because citation-count metadata raises their score.
    direct_budget = min(len(ranked_direct), max(8, math.ceil(limit * 0.65)))
    selected = ranked_direct[:direct_budget] + ranked_graph[: max(0, limit - direct_budget)]
    if len(selected) < limit:
        selected_ids = {paper.paper_id for paper in selected}
        remaining = sorted(
            (paper for paper in eligible if paper.paper_id not in selected_ids),
            key=lambda paper: writing_score(paper, query_terms, anchor_terms), reverse=True,
        )
        selected.extend(remaining[: limit - len(selected)])
    if selected:
        return selected
    # Sparse metadata must not make the whole task unrunnable. This fallback is
    # intentionally small and prefers the shallowest available evidence.
    usable = [paper for paper in papers if paper.abstract and paper.url]
    return sorted(
        usable,
        key=lambda paper: (paper.depth, -writing_score(paper, query_terms, anchor_terms)),
    )[: min(limit, 4)]


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


def normalize_source_citations(report: str, papers: list[Paper]) -> str:
    """Convert evidence-card labels into the canonical URLs expected by ReportBench."""
    urls = {index: paper.url for index, paper in enumerate(papers, 1) if paper.url}

    def replace(match: re.Match[str]) -> str:
        return urls.get(int(match.group(1)), match.group(0))

    normalized = re.sub(r"\bSource\s+(\d+)\b", replace, report, flags=re.I)
    # Keep the generated reference list readable after label expansion.
    normalized = re.sub(r"(?m)^-\s+(https?://\S+):\s+\1\s*$", r"- \1", normalized)
    return normalized


def sanitize_report(report: str) -> str:
    """Enforce the atomic-citation contract before ReportBench extraction."""
    # A trailing bibliography repeats bare URLs, which the statement extractor
    # reasonably interprets as additional (unsupported) cited statements.
    report = re.split(
        r"(?im)^\s*(?:#{1,6}\s*|\*\*)?(?:references|bibliography)\s*:?[ \t]*(?:\*\*)?\s*$",
        report,
        maxsplit=1,
    )[0]
    kept: list[str] = []
    for line in report.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z]|\*\*|#{1,6}\s))", line)
        atomic: list[str] = []
        for sentence in sentences:
            urls = re.findall(r"https?://[^\s)\]>]+", sentence, flags=re.I)
            # Multi-source synthesis cannot be attributed atomically. Omitting it
            # is safer than pretending every source supports the combined claim.
            if len({url.rstrip(".,;:'\"") for url in urls}) <= 1:
                atomic.append(sentence)
        kept.append(" ".join(atomic))
    return "\n".join(kept).strip()


class CitationRagPipeline:
    """Builds an ephemeral, per-task citation graph; only raw API responses are cached."""

    def __init__(self, settings: Settings, model: MiniMaxClient, scholar, reader: WebPageReader | None = None):
        self.settings = settings
        self.model = model
        self.scholar = scholar
        self.reader = reader

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        terms = keywords(f"{task.application_domain} {task.prompt}")
        queries = plan_search_queries(task, self.model, self.settings.baseline_search_budget)
        seeds = parallel_search(
            self.scholar, queries, cutoff=cutoff,
            per_query=self.settings.search_results_per_query, workers=self.settings.search_workers,
        )
        anchor_query = queries[0] if queries else ""
        anchor_terms = keywords(anchor_query) if anchor_query else terms
        print(f"{task.arxiv_id} search queries: {queries}", flush=True)
        seeds = filter_papers(seeds, forbidden_title=task.title, cutoff=cutoff)
        seeds = [paper for paper in seeds if paper.abstract and paper.url and is_scholarly_candidate(paper)]
        for paper in seeds:
            semantic = score_paper(paper, terms)
            anchor = anchor_coverage(paper, anchor_terms)
            rank_bonus = 1.0 / (1.0 + max(0, paper.search_rank))
            paper.relevance = (
                0.50 * semantic + 0.30 * anchor + 0.15 * rank_bonus
                + 0.05 * min(2, paper.query_hits) / 2
            )
        seeds.sort(key=lambda paper: paper.relevance, reverse=True)
        candidate_seeds = diverse_top_papers(
            seeds, len(queries), max(self.settings.rag_evidence_papers, self.settings.retrieval_candidate_pool),
        )
        direct_seeds = model_rerank_papers(
            task, candidate_seeds, self.model, self.settings.rag_evidence_papers,
        )
        frontier = direct_seeds[: self.settings.rag_seed_count]
        # Web-search-only seeds have no outbound edges. Guarantee that the
        # frontier contains up to three structured seeds when they are relevant,
        # otherwise a configured three-layer graph degenerates into reranking.
        graph_ready = [paper for paper in direct_seeds if paper.referenced_work_ids]
        frontier_ids = {paper.paper_id for paper in frontier}
        for paper in graph_ready[:3]:
            if paper.paper_id in frontier_ids:
                continue
            replace_index = next(
                (index for index in range(len(frontier) - 1, -1, -1)
                 if not frontier[index].referenced_work_ids),
                None,
            )
            if replace_index is None:
                break
            frontier_ids.discard(frontier[replace_index].paper_id)
            frontier[replace_index] = paper
            frontier_ids.add(paper.paper_id)
        graph: dict[str, Paper] = {paper.paper_id: paper for paper in direct_seeds}

        # Inspect a broader set of references, then admit only the best global
        # candidates at each depth. Reference-list order is not a relevance rank.
        inspect_per_parent = {1: 8, 2: 4, 3: 3}
        depth_budget = {1: 18, 2: 10, 3: 6}
        preserve_per_parent = {1: 5, 2: 2, 3: 1}
        for depth in range(1, self.settings.rag_depth + 1):
            candidates: dict[str, Paper] = {}
            parent_coverage_ids: list[str] = []
            reference_requests: list[tuple[int, str]] = []
            for parent in frontier:
                for reference_rank, reference_id in enumerate(
                    parent.referenced_work_ids[: inspect_per_parent.get(depth, 3)]
                ):
                    if reference_id in graph:
                        continue
                    reference_requests.append((reference_rank, reference_id))
            requested_ids = list(dict.fromkeys(reference_id for _, reference_id in reference_requests))
            get_works = getattr(self.scholar, "get_works", None)
            if callable(get_works):
                resolved = {paper.paper_id: paper for paper in get_works(requested_ids, depth=depth)}
            else:
                resolved = {}
                for reference_id in requested_ids:
                    paper = self.scholar.get_work(reference_id, depth=depth)
                    if not paper:
                        continue
                    resolved[paper.paper_id] = paper
            for reference_rank, reference_id in reference_requests:
                paper = resolved.get(reference_id)
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
                previous = candidates.get(paper.paper_id)
                if previous is None or paper.relevance > previous.relevance:
                    candidates[paper.paper_id] = paper
                if (
                    reference_rank < preserve_per_parent.get(depth, 1)
                    and paper.paper_id not in parent_coverage_ids
                ):
                    parent_coverage_ids.append(paper.paper_id)
            ranked_candidates = sorted(
                candidates.values(),
                key=lambda paper: paper.relevance,
                reverse=True,
            )
            remaining = self.settings.rag_max_papers - len(graph)
            budget = min(depth_budget.get(depth, 6), remaining)
            covered = [candidates[paper_id] for paper_id in parent_coverage_ids if paper_id in candidates]
            covered = covered[:budget]
            covered_ids = {paper.paper_id for paper in covered}
            admitted = covered + [
                paper for paper in ranked_candidates if paper.paper_id not in covered_ids
            ][: max(0, budget - len(covered))]
            for paper in admitted:
                graph[paper.paper_id] = paper
            next_frontier = admitted
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
        writing_papers = select_writing_papers(papers, task, self.settings.rag_evidence_papers)
        if not writing_papers:
            raise RuntimeError(f"No high-confidence writing evidence found for {task.arxiv_id}")
        if self.reader:
            self.reader.enrich_many(writing_papers, self.settings.reader_workers)
        cards = evidence_block(writing_papers, self.settings.evidence_char_limit * 2)
        user = (
            f"RESEARCH TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
            f"APPLICATION DOMAIN:\n{task.application_domain}\n\n"
            "EVIDENCE CARDS (each card is an allowed source; ignore citation-count metadata as factual evidence):\n"
            f"{cards}\n\n"
            "REFERENCE BUDGET: Cite 10-12 distinct sources when they have explicit usable evidence. Select the most central primary or canonical works; do not cite a source merely "
            "because it is highly cited, and avoid secondary-survey claims when a supplied primary source supports the same point.\n\n"
            "LENGTH: Write a focused survey of 650-850 English words. This is a hard maximum. Prioritize the task's central taxonomy and "
            "strongest evidence; omit tangential material.\n\n"
            "MANDATORY FINAL CHECK: Before returning the report, inspect every prose sentence. Delete any factual sentence "
            "that lacks an adjacent URL or whose exact claim is not explicit in the cited evidence card. This applies to "
            "the introduction, transitions, synthesis, and conclusion as strictly as it applies to method descriptions."
        )
        messages = [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}]
        try:
            report = self.model.generate(
                messages, max_tokens=self.settings.rag_output_tokens,
                cache_namespace=f"citation-rag-report-v8:{self.settings.model}",
            )
            if not generated_report_is_usable(report):
                raise RuntimeError("RAG writer failed the report quality gate")
        except RuntimeError as exc:
            if "finish_reason=length" not in str(exc) and "quality gate" not in str(exc):
                raise
            print("RAG writer exhausted its reasoning budget; retrying once with the full completion budget", flush=True)
            try:
                report = self.model.generate(
                    messages, max_tokens=self.settings.max_output_tokens,
                    cache_namespace=f"citation-rag-report-v8-length-retry:{self.settings.model}",
                )
                if not generated_report_is_usable(report):
                    raise RuntimeError("RAG writer failed the report quality gate")
            except RuntimeError as retry_exc:
                if "finish_reason=length" not in str(retry_exc) and "quality gate" not in str(retry_exc):
                    raise
                # A small minority of M3 calls spend the entire 32k allowance
                # in hidden reasoning. Preserve the same source ordering but
                # shorten the evidence context for one final deterministic
                # recovery attempt instead of failing the experiment task.
                compact_cards = evidence_block(
                    writing_papers[:12], self.settings.evidence_char_limit,
                )
                compact_user = user.replace(cards, compact_cards).replace(
                    "800-1,050 English words", "650-900 English words",
                )
                print("RAG full-budget retry also exhausted; using compact evidence recovery", flush=True)
                report = self.model.generate(
                    [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": compact_user}],
                    temperature=0, max_tokens=self.settings.max_output_tokens,
                    cache_namespace=f"citation-rag-report-v8-compact-recovery:{self.settings.model}",
                )
                if not generated_report_is_usable(report, minimum_words=300):
                    raise RuntimeError("RAG writer returned an unusable report after recovery")
        report = repair_grounded_report(
            report, writing_papers, self.model,
            f"citation-rag-evidence-repair-v2:{self.settings.model}", "600-800", "10-12",
        )
        report = sanitize_report(report)
        report = recover_sanitized_report(
            report, writing_papers, self.model,
            f"citation-rag-post-sanitize-recovery-v1:{self.settings.model}", "500-650",
        )
        report = sanitize_report(report)
        if not generated_report_is_usable(report, minimum_words=300):
            raise RuntimeError("RAG report collapsed after atomic-citation cleanup")
        report = normalize_source_citations(report, writing_papers)
        return report, papers
