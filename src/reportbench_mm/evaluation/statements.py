from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

from ..models import MiniMaxClient
from .reference import URL_RE, normalize_url


def cited_statements(report: str) -> list[dict]:
    statements: list[dict] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", report):
        urls = URL_RE.findall(part)
        clean = part.strip()
        if not clean or not urls:
            continue
        for url in urls:
            statements.append({"statement": clean, "url": normalize_url(url)})
    return statements


def _parse_decisions(value, expected: int) -> list[bool]:
    if isinstance(value, dict):
        value = value.get("decisions", [])
    decisions = [bool(item.get("match", False)) if isinstance(item, dict) else bool(item) for item in value]
    if len(decisions) != expected:
        raise RuntimeError(f"Judge returned {len(decisions)} decisions, expected {expected}")
    return decisions


def _judge_batches(
    records: list[dict], model: MiniMaxClient, *, prompt_prefix: str, namespace: str,
    votes: int, batch_size: int = 8,
) -> list[list[bool]]:
    all_votes = [[] for _ in range(votes)]

    def judge_one(batch: list[dict], label: str, vote: int) -> list[bool]:
        prompt = (
            prompt_prefix
            + " Return JSON only as {\"decisions\":[{\"match\":true|false}, ...]} in exactly the input order. "
            + "No explanation is needed.\n\n"
            + json.dumps(batch, ensure_ascii=False)
        )
        try:
            response = model.generate_json(
                [{"role": "user", "content": prompt}],
                model=model.settings.judge_model,
                temperature=0.2,
                max_tokens=8192,
                cache_namespace=f"{namespace}:{label}:vote-{vote}",
            )
            print(f"judge {namespace} {label} vote={vote + 1}/{votes} size={len(batch)} ok", flush=True)
            return _parse_decisions(response, len(batch))
        except RuntimeError as exc:
            if "API HTTP" in str(exc) or "connection failed" in str(exc):
                raise
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            print(f"judge {namespace} {label} size={len(batch)} split: {exc}", flush=True)
            return judge_one(batch[:middle], label + "a", vote) + judge_one(batch[middle:], label + "b", vote)

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for vote in range(votes):
            all_votes[vote].extend(judge_one(batch, f"batch-{start // batch_size}", vote))
    return all_votes


def evaluate_cited(result_path: Path, model: MiniMaxClient, votes: int = 3) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    items = cited_statements(result.get("response", ""))
    source_by_url = {normalize_url(paper.get("url", "")): paper for paper in result.get("papers", [])}
    evidence_items = []
    valid_indexes = []
    for index, item in enumerate(items):
        paper = source_by_url.get(item["url"])
        if paper and paper.get("abstract"):
            valid_indexes.append(index)
            evidence_items.append({
                "id": index,
                "claim": item["statement"],
                "source_title": paper.get("title"),
                "source_text": paper.get("abstract"),
            })
    vote_rows: list[list[bool]] = []
    if evidence_items:
        vote_rows = _judge_batches(
            evidence_items,
            model,
            prompt_prefix="Determine whether each claim is completely and faithfully supported by its source text. Do not use outside knowledge.",
            namespace=f"cited-judge-v2:{model.settings.judge_model}",
            votes=votes,
        )
    supported = 0
    details = []
    positions = {original: local for local, original in enumerate(valid_indexes)}
    for index, item in enumerate(items):
        local = positions.get(index)
        yes = sum(row[local] for row in vote_rows) if local is not None else 0
        match = yes > votes / 2
        supported += int(match)
        details.append({**item, "support_votes": yes, "total_votes": votes if local is not None else 0, "match": match})
    return {
        "cited_match_rate": supported / len(items) if items else 0.0,
        "cited_count": len(items),
        "cited_supported": supported,
        "cited_details": details,
    }


def extract_noncited(report: str, cited: list[dict], model: MiniMaxClient, limit: int = 20) -> list[str]:
    candidates: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", report):
        text = re.sub(r"^[#>*\-\d.\s]+", "", part).strip()
        if URL_RE.search(text) or len(text) < 30 or len(text) > 1000:
            continue
        if text.lower().startswith(("references", "source ", "table ")):
            continue
        candidates.append(text)
    # Preserve order and remove repeated prose before asking the model to classify factuality.
    candidates = list(dict.fromkeys(candidates))[:60]
    prompt = (
        "From CANDIDATES, select externally verifiable factual claims that lack a URL citation. "
        "Exclude opinions, headings, common knowledge, vague statements, and methodological instructions. "
        f"Return JSON {{\"statements\":[strings]}} with at most {limit} atomic claims.\n\n"
        f"CANDIDATES:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    value = model.generate_json(
        [{"role": "user", "content": prompt}],
        model=model.settings.judge_model,
        temperature=0,
        max_tokens=16384,
        cache_namespace=f"noncited-extract-v3:{model.settings.judge_model}",
    )
    if isinstance(value, dict):
        value = value.get("statements", [])
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def evaluate_noncited(
    statements: list[str], model: MiniMaxClient, scholar, cutoff, votes: int = 3,
    local_papers: list[dict] | None = None,
) -> dict:
    records = []
    for statement in statements:
        claim_terms = set(re.findall(r"[a-z][a-z0-9-]{2,}", statement.lower()))
        ranked_local = sorted(
            (paper for paper in (local_papers or []) if paper.get("abstract")),
            key=lambda paper: len(
                claim_terms & set(re.findall(r"[a-z][a-z0-9-]{2,}", f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()))
            ),
            reverse=True,
        )
        evidence_rows = [
            {"title": p.get("title"), "abstract": p.get("abstract"), "url": p.get("url")}
            for p in ranked_local[:3]
        ]
        if not evidence_rows:
            evidence = scholar.search(statement[:300], cutoff=cutoff, limit=3)
            evidence_rows = [
                {"title": p.title, "abstract": p.abstract, "url": p.url} for p in evidence if p.abstract
            ][:3]
        records.append({
            "claim": statement,
            "evidence": evidence_rows,
        })
    vote_rows: list[list[bool]] = []
    if records:
        vote_rows = _judge_batches(
            records,
            model,
            prompt_prefix=(
                "Verify each claim using only its supplied web-retrieved scholarly evidence. "
                "A claim is true only when the evidence directly supports it; insufficient evidence is false."
            ),
            namespace=f"noncited-judge-v2:{model.settings.judge_model}",
            votes=votes,
        )
    correct = 0
    details = []
    for index, record in enumerate(records):
        yes = sum(row[index] for row in vote_rows)
        decision = yes > votes / 2
        correct += int(decision)
        details.append({**record, "true_votes": yes, "total_votes": votes, "decision": decision})
    return {
        "noncited_factual_accuracy": correct / len(records) if records else 0.0,
        "noncited_count": len(records),
        "noncited_correct": correct,
        "noncited_details": details,
    }
