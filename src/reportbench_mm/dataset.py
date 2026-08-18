from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

from .schemas import Task


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    tasks.append(Task.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid task at line {line_number}: {exc}") from exc
    return tasks


def stratified_subset(tasks: list[Task], count: int, seed: int = 818) -> list[Task]:
    if not 0 < count <= len(tasks):
        raise ValueError(f"count must be in [1, {len(tasks)}]")
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.application_domain].append(task)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda item: item.arxiv_id)
        rng.shuffle(group)
    ordered_domains = sorted(groups)
    selected: list[Task] = []
    while len(selected) < count:
        progressed = False
        for domain in ordered_domains:
            if groups[domain] and len(selected) < count:
                selected.append(groups[domain].pop())
                progressed = True
        if not progressed:
            break
    return selected


def write_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            stream.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")

