from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportbench_mm.evaluation.reference import load_gt_titles, maximum_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/subsets/reportbench_30.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--systems", default="baseline,citation-rag-v12")
    args = parser.parse_args()

    task_ids = [
        json.loads(line)["arxiv_id"]
        for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    for system in args.systems.split(","):
        totals = {"pool": 0, "pool_matches": 0, "cited": 0, "cited_matches": 0, "gold": 0}
        print(f"SYSTEM {system}")
        for arxiv_id in task_ids:
            result_path = Path("runs/MiniMax-M3") / system / arxiv_id / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            gold = load_gt_titles(Path("ReportBench_v1.1_GT") / f"{arxiv_id}.jsonl")
            pool = [paper["title"] for paper in result.get("papers", []) if paper.get("title")]
            pool_matches = maximum_matches(pool, gold)
            report_urls = set()
            for paper in result.get("papers", []):
                if paper.get("url") and paper["url"] in result.get("response", ""):
                    report_urls.add(paper["url"])
            cited = [paper["title"] for paper in result.get("papers", []) if paper.get("url") in report_urls]
            cited_matches = maximum_matches(cited, gold)
            print(
                f"{arxiv_id}: pool={len(pool)} pool_gold={pool_matches} "
                f"cited={len(cited)} cited_gold={cited_matches} gold={len(gold)}"
            )
            for key, value in {
                "pool": len(pool), "pool_matches": pool_matches, "cited": len(cited),
                "cited_matches": cited_matches, "gold": len(gold),
            }.items():
                totals[key] += value
        print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
