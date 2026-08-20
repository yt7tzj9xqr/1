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

    def total(key: str) -> int:
        return sum(int(row.get(key) or 0) for row in rows)

    reference_matches = total("reference_matches")
    reference_count = total("reference_count")
    ground_truth_count = total("ground_truth_count")
    cited_supported = total("cited_supported")
    cited_count = total("cited_count")
    noncited_correct = total("noncited_correct")
    noncited_count = total("noncited_count")
    summary.update({
        "reference_matches_total": reference_matches,
        "ground_truth_count_total": ground_truth_count,
        "reference_micro_precision": reference_matches / reference_count if reference_count else None,
        "reference_micro_recall": reference_matches / ground_truth_count if ground_truth_count else None,
        "cited_micro_accuracy": cited_supported / cited_count if cited_count else None,
        "noncited_micro_accuracy": noncited_correct / noncited_count if noncited_count else None,
        "noncited_evaluated_total": noncited_count,
    })
    nonempty_noncited = [
        float(row["noncited_factual_accuracy"])
        for row in rows
        if int(row.get("noncited_count") or 0) > 0
    ]
    summary["noncited_nonempty_task_accuracy"] = (
        mean(nonempty_noncited) if nonempty_noncited else None
    )
    summary["noncited_nonempty_task_count"] = len(nonempty_noncited)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    keys = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    return summary
