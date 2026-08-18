from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import JsonCache
from .config import Settings
from .dataset import load_tasks, stratified_subset, write_tasks
from .models import MiniMaxClient
from .pipelines import BaselinePipeline, CitationRagPipeline
from .providers import OpenAlexProvider
from .runner import ExperimentRunner
from .evaluation.reference import evaluate_reference
from .evaluation.aggregate import aggregate
from .evaluation.statements import cited_statements, evaluate_cited, evaluate_noncited, extract_noncited
from .providers.openalex import extract_cutoff


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


def command_run_baseline(args: argparse.Namespace) -> None:
    settings = Settings.load()
    tasks = load_tasks(Path(args.tasks))[: args.limit]
    if not args.execute:
        print(json.dumps({"status": "dry-run", "model": settings.model, "system": "baseline", "tasks": [t.arxiv_id for t in tasks]}, indent=2))
        return
    cache = JsonCache(settings.root / "cache" / "runtime.sqlite3")
    model = MiniMaxClient(settings, cache)
    pipeline = BaselinePipeline(settings, model, OpenAlexProvider(cache, settings.openalex_mailto))
    runner = ExperimentRunner(settings.root / "runs", settings.model, "baseline")
    summary = {"completed": 0, "skipped": 0, "failed": 0}
    for task in tasks:
        status = runner.run_task(task, pipeline, overwrite=args.overwrite)
        summary[status] += 1
        print(f"{task.arxiv_id}: {status}")
    print(json.dumps(summary, indent=2))


def command_run_rag(args: argparse.Namespace) -> None:
    settings = Settings.load()
    tasks = load_tasks(Path(args.tasks))[: args.limit]
    if not args.execute:
        print(json.dumps({"status": "dry-run", "model": settings.model, "system": "citation-rag", "tasks": [t.arxiv_id for t in tasks]}, indent=2))
        return
    cache = JsonCache(settings.root / "cache" / "runtime.sqlite3")
    model = MiniMaxClient(settings, cache)
    pipeline = CitationRagPipeline(settings, model, OpenAlexProvider(cache, settings.openalex_mailto))
    runner = ExperimentRunner(settings.root / "runs", settings.model, "citation-rag")
    summary = {"completed": 0, "skipped": 0, "failed": 0}
    for task in tasks:
        status = runner.run_task(task, pipeline, overwrite=args.overwrite)
        summary[status] += 1
        print(f"{task.arxiv_id}: {status}")
    print(json.dumps(summary, indent=2))


def command_evaluate_reference(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    output_root = Path(args.output)
    metric_files: list[Path] = []
    for result_path in sorted(run_root.glob("*/result.json")):
        arxiv_id = result_path.parent.name
        gt_path = Path(args.gt_root) / f"{arxiv_id}.jsonl"
        if not gt_path.exists():
            print(f"{arxiv_id}: missing ground truth")
            continue
        metrics = evaluate_reference(result_path, gt_path)
        metric_path = output_root / f"{arxiv_id}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_files.append(metric_path)
        print(f"{arxiv_id}: P={metrics['reference_precision']:.3f} R={metrics['reference_recall']:.3f}")
    if metric_files:
        print(json.dumps(aggregate(metric_files, output_root / "summary.json"), indent=2))


def command_evaluate_statements(args: argparse.Namespace) -> None:
    settings = Settings.load()
    settings.require_api_key()
    cache = JsonCache(settings.root / "cache" / "evaluation.sqlite3")
    model = MiniMaxClient(settings, cache)
    scholar = OpenAlexProvider(cache, settings.openalex_mailto)
    output_root = Path(args.output)
    metric_files: list[Path] = []
    for result_path in sorted(Path(args.run_root).glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task = result["task"]
        cited = cited_statements(result["response"])
        cited_metrics = evaluate_cited(result_path, model, votes=args.votes)
        noncited = extract_noncited(result["response"], cited, model, limit=args.max_noncited)
        noncited_metrics = evaluate_noncited(noncited, model, scholar, extract_cutoff(task["prompt"]), votes=args.votes)
        ref_path = Path(args.reference_root) / f"{task['arxiv_id']}.json"
        base = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {
            "arxiv_id": task["arxiv_id"], "model": result["model"], "system": result["system"]
        }
        metrics = {**base, **cited_metrics, **noncited_metrics}
        metric_path = output_root / f"{task['arxiv_id']}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_files.append(metric_path)
        print(f"{task['arxiv_id']}: cited={metrics['cited_match_rate']:.3f}, noncited={metrics['noncited_factual_accuracy']:.3f}")
    if metric_files:
        print(json.dumps(aggregate(metric_files, output_root / "summary.json"), indent=2))


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
    baseline = sub.add_parser("run-baseline")
    baseline.add_argument("--tasks", default="data/subsets/reportbench_30.jsonl")
    baseline.add_argument("--limit", type=int, default=1)
    baseline.add_argument("--execute", action="store_true")
    baseline.add_argument("--overwrite", action="store_true")
    baseline.set_defaults(func=command_run_baseline)
    rag = sub.add_parser("run-rag")
    rag.add_argument("--tasks", default="data/subsets/reportbench_30.jsonl")
    rag.add_argument("--limit", type=int, default=1)
    rag.add_argument("--execute", action="store_true")
    rag.add_argument("--overwrite", action="store_true")
    rag.set_defaults(func=command_run_rag)
    reference = sub.add_parser("evaluate-reference")
    reference.add_argument("--run-root", required=True)
    reference.add_argument("--gt-root", default="ReportBench_v1.1_GT")
    reference.add_argument("--output", required=True)
    reference.set_defaults(func=command_evaluate_reference)
    statements = sub.add_parser("evaluate-statements")
    statements.add_argument("--run-root", required=True)
    statements.add_argument("--reference-root", required=True)
    statements.add_argument("--output", required=True)
    statements.add_argument("--votes", type=int, default=3)
    statements.add_argument("--max-noncited", type=int, default=20)
    statements.set_defaults(func=command_evaluate_statements)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
