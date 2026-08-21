from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from reportbench_mm.cache import JsonCache
from reportbench_mm.config import Settings
from reportbench_mm.models import MiniMaxClient
from reportbench_mm.pipelines.rag import select_writing_papers
from reportbench_mm.prompts import VERIFIED_BACKGROUND_HEADING, add_verified_noncited_facts
from reportbench_mm.schemas import Paper, Task


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add evidence-verified URL-free technical background facts to an existing RAG run. "
            "The cited report and retrieval pool are otherwise preserved."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--target-facts", type=int, default=4)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ids", default="", help="Optional comma-separated task IDs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    target_root = Path(args.target).resolve()
    if source_root == target_root:
        raise ValueError("--target must differ from --source")
    result_paths = sorted(source_root.glob("*/result.json"))
    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    if wanted:
        result_paths = [path for path in result_paths if path.parent.name in wanted]
    if not result_paths:
        raise FileNotFoundError(f"No result files found below {source_root}")

    settings = Settings.load()
    settings.require_api_key()
    model = MiniMaxClient(settings, JsonCache(settings.root / "cache" / "runtime.sqlite3"))

    def process(result_path: Path) -> str:
        task_id = result_path.parent.name
        destination = target_root / task_id
        output_path = destination / "result.json"
        if output_path.exists() and not args.overwrite:
            return "skipped"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task = Task.from_dict(result["task"])
        paper_fields = set(Paper.__dataclass_fields__)
        papers = [
            Paper(**{key: value for key, value in row.items() if key in paper_fields})
            for row in result.get("papers", [])
        ]
        writing_papers = select_writing_papers(papers, task, settings.rag_evidence_papers)
        augmented = add_verified_noncited_facts(
            result.get("response", ""), writing_papers, model,
            f"citation-rag-verified-background-v2:{settings.model}:{task_id}",
            target=max(1, args.target_facts), votes=max(1, args.votes),
        )
        if VERIFIED_BACKGROUND_HEADING not in augmented:
            raise RuntimeError(f"No verified background facts survived for {task_id}")
        result["system"] = args.system
        result["response"] = augmented
        result["postprocessing"] = {
            "kind": "evidence-verified-noncited-background",
            "source_result": str(result_path),
            "target_facts": args.target_facts,
            "verification_votes": args.votes,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        destination.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        (destination / "report.md").write_text(augmented, encoding="utf-8")
        return "completed"

    summary = {"completed": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process, path): path for path in result_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                summary["failed"] += 1
                print(f"{path.parent.name}: failed: {exc}", flush=True)
                continue
            summary[status] += 1
            print(f"{path.parent.name}: {status}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
