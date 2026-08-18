from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import JsonCache
from .config import Settings
from .dataset import load_tasks, stratified_subset, write_tasks
from .models import MiniMaxClient


def command_prepare(args: argparse.Namespace) -> None:
    tasks = load_tasks(Path(args.input))
    subset = stratified_subset(tasks, args.count, args.seed)
    write_tasks(Path(args.output), subset)
    counts: dict[str, int] = {}
    for task in subset:
        counts[task.application_domain] = counts.get(task.application_domain, 0) + 1
    print(json.dumps({"output": args.output, "count": len(subset), "domains": counts}, ensure_ascii=False, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    settings = Settings.load()
    if not args.execute:
        print(json.dumps({"status": "dry-run", "model": settings.model, "base_url": settings.base_url}, indent=2))
        return
    cache = JsonCache(settings.root / "cache" / "llm.sqlite3")
    answer = MiniMaxClient(settings, cache).generate(
        [{"role": "user", "content": "Reply with exactly: MINIMAX_OK"}],
        temperature=0,
        max_tokens=32,
        cache_namespace="smoke-v1",
    )
    if "MINIMAX_OK" not in answer:
        raise RuntimeError(f"Unexpected smoke-test response: {answer!r}")
    print("MINIMAX_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reportbench-mm")
    sub = parser.add_subparsers(required=True)
    prepare = sub.add_parser("prepare-subset")
    prepare.add_argument("--input", default="ReportBench_v1.1.jsonl")
    prepare.add_argument("--output", default="data/subsets/reportbench_30.jsonl")
    prepare.add_argument("--count", type=int, default=30)
    prepare.add_argument("--seed", type=int, default=818)
    prepare.set_defaults(func=command_prepare)
    smoke = sub.add_parser("smoke-api")
    smoke.add_argument("--execute", action="store_true")
    smoke.set_defaults(func=command_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

