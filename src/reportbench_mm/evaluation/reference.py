from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from ..providers.openalex import normalize_title


URL_RE = re.compile(r"https?://[^\s)\]>]+", re.I)


def normalize_url(url: str) -> str:
    url = url.rstrip(".,;:'\"")
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))


def load_gt_titles(path: Path) -> list[str]:
    titles: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                title = json.loads(line).get("title")
                if title:
                    titles.append(title)
    return titles


def title_match(predicted: str, gold: str, threshold: float = 0.88) -> bool:
    left, right = normalize_title(predicted), normalize_title(gold)
    if not left or not right:
        return False
    return left == right or SequenceMatcher(None, left, right).ratio() >= threshold


def maximum_matches(predicted: list[str], gold: list[str]) -> int:
    # Greedy best-first one-to-one matching prevents duplicate predictions inflating recall.
    candidates: list[tuple[float, int, int]] = []
    for i, left in enumerate(predicted):
        for j, right in enumerate(gold):
            lnorm, rnorm = normalize_title(left), normalize_title(right)
            ratio = SequenceMatcher(None, lnorm, rnorm).ratio()
            if lnorm == rnorm or ratio >= 0.88:
                candidates.append((ratio, i, j))
    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_predicted and j not in used_gold:
            used_predicted.add(i)
            used_gold.add(j)
    return len(used_predicted)


def evaluate_reference(result_path: Path, gt_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report_urls = {normalize_url(url) for url in URL_RE.findall(result.get("response", ""))}
    papers = result.get("papers", [])
    predicted: list[str] = []
    seen_titles: set[str] = set()
    for paper in papers:
        url = normalize_url(paper.get("url", ""))
        title = paper.get("title", "")
        normalized = normalize_title(title)
        if url in report_urls and normalized and normalized not in seen_titles:
            predicted.append(title)
            seen_titles.add(normalized)
    gold = load_gt_titles(gt_path)
    matches = maximum_matches(predicted, gold)
    return {
        "arxiv_id": result["task"]["arxiv_id"],
        "model": result["model"],
        "system": result["system"],
        "reference_precision": matches / len(predicted) if predicted else 0.0,
        "reference_recall": matches / len(gold) if gold else 0.0,
        "reference_count": len(predicted),
        "reference_matches": matches,
        "ground_truth_count": len(gold),
        "predicted_titles": predicted,
    }

