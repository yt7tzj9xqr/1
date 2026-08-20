from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from reportbench_mm.cache import JsonCache
from reportbench_mm.cli import scholar_provider
from reportbench_mm.config import Settings
from reportbench_mm.dataset import load_tasks
from reportbench_mm.evaluation.reference import load_gt_titles, maximum_matches, title_match
from reportbench_mm.models import MiniMaxClient
from reportbench_mm.pipelines import BaselinePipeline
from reportbench_mm.web_reader import WebPageReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/subsets/reportbench_30.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="artifacts/retrieval_audit_10.json")
    args = parser.parse_args()

    settings = Settings.load()
    cache = JsonCache(settings.root / "cache" / "runtime.sqlite3")
    pipeline = BaselinePipeline(
        settings, MiniMaxClient(settings, cache), scholar_provider(cache, settings), WebPageReader(cache)
    )
    tasks = load_tasks(Path(args.tasks))[: args.limit]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(pipeline.retrieve, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            papers = future.result()
            gold = load_gt_titles(Path("ReportBench_v1.1_GT") / f"{task.arxiv_id}.jsonl")
            matches = maximum_matches([paper.title for paper in papers], gold)
            matched_titles = [
                paper.title for paper in papers if any(title_match(paper.title, gold_title) for gold_title in gold)
            ]
            row = {
                "arxiv_id": task.arxiv_id, "pool": len(papers), "pool_gold": matches,
                "gold": len(gold), "matched_titles": matched_titles,
                "pool_titles": [paper.title for paper in papers],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {
        "tasks": len(rows), "pool": sum(row["pool"] for row in rows),
        "pool_gold": sum(row["pool_gold"] for row in rows), "gold": sum(row["gold"] for row in rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "tasks": sorted(rows, key=lambda row: row["arxiv_id"])}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
