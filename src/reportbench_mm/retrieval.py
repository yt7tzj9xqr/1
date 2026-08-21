from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import re
from urllib.parse import urlparse

from .models import MiniMaxClient
from .providers.openalex import search_queries
from .schemas import Paper, Task


NON_SCHOLARLY_TITLE_RE = re.compile(
    r"^(?:client challenge|verifying your browser|what is |the illustrated |awesome[- :]|pubmed\.ncbi\.nlm\.nih\.gov$)"
    r"|\b(?:paper explained|paper review|paper list|top research papers|info for participants|smart speech therapy llc|"
    r"youtube|wikipedia|request pdf|homepage)\b|^\[discussion\]",
    re.I,
)


def canonical_search_title(value: str) -> str:
    value = re.sub(r"^\[(?:19|20)?\d{2,4}\.\d+(?:v\d+)?\]\s*", "", value.strip())
    value = re.sub(r"^(?:GitHub\s+-\s+[^:]+:\s*|[^ /]+/[^ ]+\s+-\s+)", "", value, flags=re.I)
    value = re.sub(r"\s+(?:\||»|-)\s+(?:Request PDF|PMC|PubMed|Springer Nature|GitHub)\s*$", "", value, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def is_scholarly_candidate(paper: Paper) -> bool:
    if NON_SCHOLARLY_TITLE_RE.search(paper.title.strip()):
        return False
    host = (urlparse(paper.url).hostname or "").lower().removeprefix("www.")
    if host in {
        "aws.amazon.com", "ibm.com", "medium.com", "m.blog.naver.com", "github.com",
        "youtube.com", "youtu.be", "wikipedia.org", "en.wikipedia.org", "scribd.com",
        "researchgate.net",
    }:
        return False
    return len(canonical_search_title(paper.title).split()) >= 3


def _clean_query(value: str) -> str:
    value = re.split(
        r"\b(?:before|on or before|published before|published on or before|ensure|reference materials)\b",
        value, maxsplit=1, flags=re.I,
    )[0]
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*", value)
    return " ".join(words[:10])[:160].strip()


def plan_search_queries(task: Task, model: MiniMaxClient, limit: int = 5) -> list[str]:
    """Use the base model as the query planner, matching the paper's agentic baseline."""
    deterministic = search_queries(task.prompt, limit=max(8, limit))
    prompt = (
        "Plan scholarly literature searches for the research task below. Return JSON only as "
        f"{{\"queries\":[strings]}} with exactly {limit} concise search-engine queries. "
        "Use this fixed allocation: query 1 is the central topic; queries 2 and 3 target likely landmark "
        "or representative PRIMARY papers (use an exact paper title when you know it with high confidence); "
        "queries 4 and 5 target distinct named method families, benchmarks, datasets, or application subfields. "
        "Every query must remain unambiguous in the central research topic. Do not emit "
        "generic fragments, instructions, dates, venue names alone, or the forbidden survey title. "
        "Never invent a paper title: when uncertain, use technical keywords instead. Prefer terminology likely "
        "to occur in titles and abstracts.\n\n"
        f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
        f"DETERMINISTIC CANDIDATES:\n{deterministic}"
    )
    try:
        value = model.generate_json(
            [{"role": "user", "content": prompt}], temperature=0,
            # M3 may spend most of a 4k budget in hidden reasoning and return no
            # final JSON. Successful plans stay cached; failed plans get enough
            # room on retry instead of dropping to weak heuristic fragments.
            max_tokens=8192, cache_namespace=f"search-planner-v3:{model.settings.model}",
        )
        planned = value.get("queries", []) if isinstance(value, dict) else value
    except Exception as exc:
        print(f"search planner fallback: {exc}", flush=True)
        planned = []
    combined = list(planned or []) + deterministic
    queries: list[str] = []
    seen: set[str] = set()
    for item in combined:
        query = _clean_query(str(item))
        key = query.lower()
        if len(query.split()) < 2 or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries or deterministic[:limit]


def parallel_search(
    scholar, queries: list[str], *, cutoff: date | None, per_query: int = 20, workers: int = 5,
    index_offset: int = 0,
) -> list[Paper]:
    """Execute the paper's five-search budget concurrently and merge by DOI/title."""
    search_many = getattr(scholar, "search_many", None)
    if callable(search_many):
        papers = search_many(
            queries, cutoff=cutoff, limit=per_query, workers=workers,
        )
        for paper in papers:
            paper.search_query_index += index_offset
        return papers
    rows: list[tuple[int, int, Paper]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(queries)))) as executor:
        futures = {
            executor.submit(scholar.search, query, cutoff=cutoff, limit=per_query): index + index_offset
            for index, query in enumerate(queries)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows.extend((index, rank, paper) for rank, paper in enumerate(future.result()))
            except Exception as exc:
                print(f"search query {index + 1} failed: {exc}", flush=True)
    best: dict[str, tuple[int, int, Paper, set[int]]] = {}
    for query_index, rank, paper in rows:
        title_key = canonical_search_title(paper.title)
        key = paper.doi.lower() if paper.doi else title_key
        if not key:
            continue
        previous = best.get(key)
        if previous is None:
            best[key] = (query_index, rank, paper, {query_index})
        else:
            previous[3].add(query_index)
            if rank < previous[1]:
                best[key] = (query_index, rank, paper, previous[3])
    papers: list[Paper] = []
    for query_index, rank, paper, query_hits in sorted(best.values(), key=lambda item: (item[0], item[1])):
        paper.search_query_index = query_index
        paper.search_rank = rank
        paper.query_hits = len(query_hits)
        papers.append(paper)
    return papers


def merge_search_results(papers: list[Paper]) -> list[Paper]:
    """Merge multiple search rounds without losing query-coverage metadata."""
    merged: dict[str, Paper] = {}
    query_sets: dict[str, set[int]] = {}
    for paper in papers:
        key = paper.doi.lower() if paper.doi else canonical_search_title(paper.title)
        if not key:
            continue
        query_sets.setdefault(key, set()).add(paper.search_query_index)
        current = merged.get(key)
        if current is None or paper.search_rank < current.search_rank:
            merged[key] = paper
    for key, paper in merged.items():
        paper.query_hits = len(query_sets[key])
    return sorted(merged.values(), key=lambda paper: (paper.search_query_index, paper.search_rank))


def adaptive_search(
    task: Task, model: MiniMaxClient, scholar, settings, cutoff: date | None,
) -> tuple[list[str], list[Paper]]:
    """Spend the five-search budget as three initial and two result-aware calls."""
    planned = plan_search_queries(task, model, settings.baseline_search_budget)
    initial_queries = planned[:3]
    initial = parallel_search(
        scholar, initial_queries, cutoff=cutoff,
        per_query=settings.search_results_per_query, workers=min(settings.search_workers, 3),
    )
    rows = [
        {
            "query": initial_queries[min(paper.search_query_index, len(initial_queries) - 1)],
            "title": paper.title,
            "evidence": " ".join(paper.abstract.split())[:160],
        }
        for paper in initial[:12]
    ]
    prompt = (
        "You control the final two searches of a five-call academic research agent. Read the first-round results and "
        "find important branches of the TASK that are missing or weakly covered. Use titles and terminology "
        "visible in RESULTS to resolve canonical names instead of guessing from memory. Return JSON only as "
        "{\"queries\":[string,string]}. Each query must be either the exact title of a real primary/landmark paper "
        "you know with high confidence, or a precise combination of method, dataset, application, and author terms. "
        "Do not repeat an existing query, search for the forbidden survey, emit a broad topic, or include dates. "
        "Favor one missing landmark/primary-paper query and one missing benchmark, dataset, or named-method query. "
        "Sources found by both rounds are especially valuable because repeated discovery is a relevance signal.\n\n"
        f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
        f"FIRST-ROUND QUERIES:\n{initial_queries}\n\nRESULTS:\n{rows}"
    )
    try:
        value = model.generate_json(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=2048,
            cache_namespace=f"search-feedback-v3:{model.settings.model}",
        )
        proposed = value.get("queries", []) if isinstance(value, dict) else value
    except Exception as exc:
        print(f"search feedback fallback: {exc}", flush=True)
        proposed = []
    followups: list[str] = []
    seen = {query.lower() for query in initial_queries}
    for item in list(proposed or []) + planned[3:]:
        query = _clean_query(str(item))
        if len(query.split()) < 2 or query.lower() in seen:
            continue
        seen.add(query.lower())
        followups.append(query)
        if len(followups) == 2:
            break
    second = parallel_search(
        scholar, followups, cutoff=cutoff,
        per_query=settings.search_results_per_query, workers=min(settings.search_workers, 2),
        index_offset=len(initial_queries),
    ) if followups else []
    return initial_queries + followups, merge_search_results(initial + second)


def diverse_top_papers(papers: list[Paper], query_count: int, limit: int) -> list[Paper]:
    """Reserve top slots per query before globally filling the evidence budget."""
    selected: list[Paper] = []
    selected_ids: set[str] = set()
    # Three slots per query are enough to preserve topic coverage. Let global
    # relevance fill the rest instead of allowing one weak result page to
    # consume six or more evidence slots in larger candidate pools.
    per_query = min(3, max(1, limit // max(1, query_count)))
    for query_index in range(query_count):
        candidates = sorted(
            (paper for paper in papers if paper.search_query_index == query_index),
            key=lambda paper: (paper.search_rank, -paper.relevance),
        )
        for paper in candidates[:per_query]:
            key = paper.doi or paper.paper_id or paper.title.lower()
            if key not in selected_ids:
                selected_ids.add(key)
                selected.append(paper)
    for paper in sorted(papers, key=lambda item: item.relevance, reverse=True):
        key = paper.doi or paper.paper_id or paper.title.lower()
        if key not in selected_ids:
            selected_ids.add(key)
            selected.append(paper)
        if len(selected) >= limit:
            break
    return selected[:limit]


def model_rerank_papers(
    task: Task, papers: list[Paper], model: MiniMaxClient, limit: int,
) -> list[Paper]:
    """Let the same base model select central papers from a broad, cached pool."""
    if len(papers) <= limit:
        return papers
    cards = []
    for index, paper in enumerate(papers, 1):
        abstract = " ".join(paper.abstract.split())[:360]
        cards.append(
            f"{index}. {paper.title} ({paper.year or 'unknown'}; source={paper.source}; "
            f"citations={paper.cited_by_count})\n{abstract}"
        )
    prompt = (
        "Select scholarly evidence for the research task. Return JSON only as "
        f"{{\"indices\":[integers]}} with exactly {limit} unique 1-based indices. "
        "Maximize relevance to the exact central topic and coverage of its named subtopics. "
        "Prefer primary, landmark, benchmark, dataset, or canonical method papers over broad surveys, "
        "tutorials, repositories, homepages, and merely high-citation but generic work. Do not select "
        "the forbidden survey or post-cutoff work. Citation count is only a tie-breaker.\n\n"
        f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\nCANDIDATES:\n"
        + "\n".join(cards)
    )
    try:
        value = model.generate_json(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=8192,
            cache_namespace=f"retrieval-reranker-v1:{model.settings.model}",
        )
        indices = value.get("indices", []) if isinstance(value, dict) else []
    except Exception as exc:
        print(f"retrieval reranker fallback: {exc}", flush=True)
        indices = []
    selected: list[Paper] = []
    seen: set[int] = set()
    for raw_index in indices:
        try:
            index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(papers) or index in seen:
            continue
        seen.add(index)
        selected.append(papers[index])
        if len(selected) >= limit:
            return selected
    for index, paper in enumerate(papers):
        if index not in seen:
            selected.append(paper)
        if len(selected) >= limit:
            break
    return selected
