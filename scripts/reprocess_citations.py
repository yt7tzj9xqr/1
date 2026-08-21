from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from reportbench_mm.pipelines.rag import sanitize_report
from reportbench_mm.prompts import generated_report_is_usable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run deterministic citation cleanup without regenerating reports or retrieval pools."
    )
    parser.add_argument("--source", required=True, help="Existing run root containing <arxiv_id>/result.json")
    parser.add_argument("--target", required=True, help="New run root; source files are never modified")
    parser.add_argument("--system", required=True, help="System label written into copied result files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    target_root = Path(args.target).resolve()
    if source_root == target_root:
        raise ValueError("--target must differ from --source")
    result_paths = sorted(source_root.glob("*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No result files found below {source_root}")

    completed = 0
    for result_path in result_paths:
        task_id = result_path.parent.name
        destination = target_root / task_id
        output_path = destination / "result.json"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite explicitly")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        original = result.get("response", "")
        cleaned = sanitize_report(original)
        if not generated_report_is_usable(cleaned, minimum_words=300):
            raise RuntimeError(f"Citation cleanup collapsed {task_id}; target was not written")
        result["system"] = args.system
        result["response"] = cleaned
        result["postprocessing"] = {
            "kind": "deterministic-citation-reprocess",
            "source_result": str(result_path),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        destination.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "report.md").write_text(cleaned, encoding="utf-8")
        completed += 1
        print(f"{task_id}: completed", flush=True)
    print(json.dumps({"completed": completed, "failed": 0}, indent=2))


if __name__ == "__main__":
    main()
