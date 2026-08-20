from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import re

from .models import MiniMaxClient
from .providers.openalex import search_queries
from .schemas import Paper, Task


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
            max_tokens=4096, cache_namespace=f"search-planner-v3:{model.settings.model}",
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
) -> list[Paper]:
    """Execute the paper's five-search budget concurrently and merge by DOI/title."""
    rows: list[tuple[int, int, Paper]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(queries)))) as executor:
        futures = {
            executor.submit(scholar.search, query, cutoff=cutoff, limit=per_query): index
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
        title_key = " ".join(re.findall(r"[a-z0-9]+", paper.title.lower()))
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


def diverse_top_papers(papers: list[Paper], query_count: int, limit: int) -> list[Paper]:
    """Reserve top slots per query before globally filling the evidence budget."""
    selected: list[Paper] = []
    selected_ids: set[str] = set()
    per_query = max(1, limit // max(1, query_count))
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
