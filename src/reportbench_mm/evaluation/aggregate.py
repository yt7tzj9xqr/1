from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


def aggregate(metric_files: list[Path], output: Path) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
    if not rows:
        raise ValueError("No metric files found")
    numeric = [
        "reference_precision", "reference_recall", "reference_count",
        "cited_match_rate", "cited_count", "noncited_factual_accuracy", "noncited_count",
    ]
    summary = {"task_count": len(rows)}
    for key in numeric:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if values:
            summary[key] = mean(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    keys = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    return summary

