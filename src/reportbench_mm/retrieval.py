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
    return " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*", value))[:160].strip()


def plan_search_queries(task: Task, model: MiniMaxClient, limit: int = 5) -> list[str]:
    """Use the base model as the query planner, matching the paper's agentic baseline."""
    deterministic = search_queries(task.prompt, limit=max(8, limit))
    prompt = (
        "Plan scholarly literature searches for the research task below. Return JSON only as "
        f"{{\"queries\":[strings]}} with exactly {limit} concise search-engine queries. "
        "Every query must retain the central research topic. Cover distinct named method families, "
        "benchmarks, datasets, or application subfields explicitly requested by the task. Do not emit "
        "generic fragments, instructions, dates, venue names alone, or the forbidden survey title. "
        "Prefer terminology likely to occur in titles and abstracts.\n\n"
        f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
        f"DETERMINISTIC CANDIDATES:\n{deterministic}"
    )
    try:
        value = model.generate_json(
            [{"role": "user", "content": prompt}], temperature=0,
            max_tokens=4096, cache_namespace=f"search-planner-v2:{model.settings.model}",
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
    rows: list[tuple[int, Paper]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(queries)))) as executor:
        futures = {
            executor.submit(scholar.search, query, cutoff=cutoff, limit=per_query): index
            for index, query in enumerate(queries)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows.extend((index, paper) for paper in future.result())
            except Exception as exc:
                print(f"search query {index + 1} failed: {exc}", flush=True)
    best: dict[str, tuple[int, Paper]] = {}
    for query_index, paper in rows:
        title_key = " ".join(re.findall(r"[a-z0-9]+", paper.title.lower()))
        key = paper.doi.lower() if paper.doi else title_key
        if not key:
            continue
        previous = best.get(key)
        if previous is None or query_index < previous[0]:
            best[key] = (query_index, paper)
    return [paper for _, paper in sorted(best.values(), key=lambda item: item[0])]
