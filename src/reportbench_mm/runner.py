from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from .schemas import Task


class ExperimentRunner:
    def __init__(self, output_root: Path, model_name: str, system_name: str):
        self.output_root = output_root / model_name / system_name

    def run_task(self, task: Task, pipeline, *, overwrite: bool = False) -> str:
        task_dir = self.output_root / task.arxiv_id
        result_path = task_dir / "result.json"
        if result_path.exists() and not overwrite:
            return "skipped"
        task_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        try:
            report, papers = pipeline.run(task)
            result = {
                "status": "completed",
                "model": self.output_root.parent.name,
                "system": self.output_root.name,
                "task": task.to_dict(),
                "response": report,
                "papers": [paper.to_dict() for paper in papers],
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            self._atomic_json(result_path, result)
            (task_dir / "report.md").write_text(report, encoding="utf-8")
            return "completed"
        except Exception as exc:
            self._atomic_json(
                task_dir / "error.json",
                {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "task": asdict(task)},
            )
            return "failed"

    @staticmethod
    def _atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

