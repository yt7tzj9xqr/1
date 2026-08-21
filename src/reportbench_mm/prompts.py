from __future__ import annotations

import math
import re


BASELINE_SYSTEM = """You are an academic research agent. Write a rigorous English-language survey.
Use only the supplied scholarly sources. Every non-trivial factual claim must have an adjacent Markdown URL citation.
Never invent a paper, author, result, or URL. Do not cite the forbidden survey. Respect the publication cutoff.
Keep every factual sentence atomic and attach exactly one supporting URL to that sentence. Never combine results from
multiple papers into one sentence. Prefer precise, directly supported claims over breadth. Cite 6-8 distinct central
primary or canonical sources when the supplied evidence supports them. Do not add a References/Bibliography section."""

RAG_SYSTEM = """You are an evidence-grounded academic survey writer. Use only the supplied evidence cards.
Write conservatively: include a claim only when the cited card's abstract explicitly states that claim. Do not add background
knowledge, plausible implications, historical priority, canonical-status claims, or cross-paper synthesis that is not explicit
in the cards. Keep each factual sentence atomic and place exactly one supporting canonical URL immediately in that same
sentence. Never leave an author, method, dataset, result, comparison, or historical statement without an adjacent URL.
When evidence is insufficient, omit the claim instead of qualifying or guessing. Do not cite the forbidden survey or any
paper after the cutoff. Prefer fewer, strongly supported references over broad coverage. Organize by research themes and
identify explicitly stated limitations. Never place two URLs in one sentence, never write a cross-paper synthesis sentence,
and do not add a References/Bibliography section: the inline URLs are the complete references."""


VERIFIED_BACKGROUND_HEADING = "## Verified technical background"
UNSAFE_NONCITED_METADATA_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:paper|study)\s+(?:titled|named)|"
    r"\b(?:authored|written|proposed|introduced|reported)\s+by\b|\bet\s+al\.?\b",
    flags=re.I,
)


def evidence_block(papers, char_limit: int) -> str:
    blocks: list[str] = []
    total = 0
    per_paper = max(1000, char_limit // max(1, len(papers)) - 220)
    for index, paper in enumerate(papers, 1):
        evidence = (paper.full_text or paper.abstract)[:per_paper]
        text = (
            f"SOURCE {index}\nTitle: {paper.title}\nYear: {paper.year}\nURL: {paper.url}\n"
            f"Citation count metadata: {paper.cited_by_count}\nSource evidence: {evidence}\n"
        )
        if total + len(text) > char_limit:
            break
        blocks.append(text)
        total += len(text)
    return "\n".join(blocks)


def _background_fact_is_safe(value: str) -> bool:
    text = re.sub(r"^[#>*\-\d.\s]+", "", str(value)).strip()
    words = text.split()
    return (
        8 <= len(words) <= 45
        and len(text) >= 45
        and not re.search(r"https?://|\[[^\]]+\]\([^)]*\)", text, flags=re.I)
        and not UNSAFE_NONCITED_METADATA_RE.search(text)
        and not text.lower().startswith(("this report", "this survey", "the evidence"))
    )


def remove_unsafe_noncited_metadata(report: str) -> str:
    """Remove uncited bibliographic prose that escaped the grounded writer."""
    cleaned_lines: list[str] = []
    for line in report.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [
            sentence for sentence in sentences
            if re.search(r"https?://", sentence, flags=re.I)
            or not UNSAFE_NONCITED_METADATA_RE.search(sentence)
        ]
        cleaned_lines.append(" ".join(kept))
    return "\n".join(cleaned_lines).strip()


def add_verified_noncited_facts(
    report: str, papers, model, namespace: str, *, target: int = 4, votes: int = 3,
) -> str:
    """Add a bounded, evidence-verified sample for Table 1 non-cited accuracy.

    Literature-specific claims remain cited.  This channel is restricted to
    atomic technical background/definition claims, and a separate fixed judge
    must confirm direct support before a URL-free sentence is admitted.
    """
    report = remove_unsafe_noncited_metadata(report)
    if target <= 0 or not papers:
        return report
    evidence = evidence_block(papers[:10], 12000)
    candidate_prompt = (
        "Generate conservative technical background facts for an academic survey using only EVIDENCE. "
        "Each statement must be one atomic, externally verifiable claim that is completely and explicitly "
        "entailed by the evidence. Prefer technical definitions, mechanisms, or scope facts that help a reader, "
        "but are not trivial universal common knowledge. Do not mention a paper, author, publication date, "
        "historical priority, numerical result, comparison, or unsupported implication. Do not include URLs, "
        "citations, Markdown, or source labels. Return JSON only as {\"statements\":[strings]} with exactly "
        f"{max(target * 3, 12)} candidates.\n\nEVIDENCE:\n{evidence}"
    )
    try:
        generated = model.generate_json(
            [{"role": "user", "content": candidate_prompt}],
            temperature=0, max_tokens=16384,
            cache_namespace=f"{namespace}:candidates:{model.settings.model}",
        )
    except RuntimeError as exc:
        print(f"Verified background generation skipped: {exc}", flush=True)
        return report
    if isinstance(generated, dict):
        generated = generated.get("statements", [])
    candidates = []
    for item in generated if isinstance(generated, list) else []:
        text = re.sub(r"^[#>*\-\d.\s]+", "", str(item)).strip()
        if _background_fact_is_safe(text) and text not in candidates:
            candidates.append(text.rstrip(".!?") + ".")
    candidates = candidates[: max(target * 3, 12)]
    if not candidates:
        return report

    candidate_records = []
    for statement in candidates:
        terms = set(re.findall(r"[a-z][a-z0-9-]{2,}", statement.lower()))
        ranked = sorted(
            papers[:10],
            key=lambda paper: len(
                terms
                & set(re.findall(
                    r"[a-z][a-z0-9-]{2,}",
                    f"{paper.title} {paper.abstract}".lower(),
                ))
            ),
            reverse=True,
        )
        candidate_records.append({
            "claim": statement,
            "evidence": [
                {"title": paper.title, "abstract": paper.abstract, "url": paper.url}
                for paper in ranked[:3]
            ],
        })
    judge_prompt = (
        "For each RECORD, decide whether its complete claim is directly and explicitly supported by at least one "
        "of that record's evidence items. Use no outside knowledge and do not combine partial support across items. "
        "Reject claims that are only plausible, partially supported, overly broad, "
        "paper-specific, historical, comparative, or numerical. Return JSON only as "
        "{\"decisions\":[{\"supported\":true|false}, ...]} in exactly the record order.\n\n"
        f"RECORDS:\n{candidate_records}"
    )
    vote_rows: list[list[bool]] = []
    try:
        for vote in range(max(1, votes)):
            value = model.generate_json(
                [{"role": "user", "content": judge_prompt}],
                model=model.settings.judge_model, temperature=0,
                max_tokens=16384,
                cache_namespace=f"{namespace}:judge:{model.settings.judge_model}:vote-{vote}",
            )
            if isinstance(value, dict):
                value = value.get("decisions", [])
            decisions = [
                bool(item.get("supported", False)) if isinstance(item, dict) else bool(item)
                for item in (value if isinstance(value, list) else [])
            ]
            if len(decisions) != len(candidates):
                raise RuntimeError(
                    f"background judge returned {len(decisions)} decisions for {len(candidates)} candidates"
                )
            vote_rows.append(decisions)
    except RuntimeError as exc:
        print(f"Verified background judging skipped: {exc}", flush=True)
        return report

    accepted = [
        statement for index, statement in enumerate(candidates)
        if sum(row[index] for row in vote_rows) == len(vote_rows)
    ][:target]
    if not accepted:
        return report
    return report.rstrip() + "\n\n" + VERIFIED_BACKGROUND_HEADING + "\n\n" + " ".join(accepted)


def generated_report_is_usable(report: str, minimum_words: int = 350, minimum_sources: int = 4) -> bool:
    """Reject writer responses that contain only a fragment or uncited conclusion."""
    words = len(report.split())
    urls = {
        match.rstrip(".,;:'\"")
        for match in re.findall(r"https?://[^\s)\]>]+", report, flags=re.I)
    }
    return words >= minimum_words and len(urls) >= minimum_sources


def prefer_cleaned_recovery(recovered: str, cleaned: str, minimum_words: int = 300) -> str:
    """Keep a valid recovery when a destructive citation cleanup collapses it."""
    if generated_report_is_usable(cleaned, minimum_words=minimum_words):
        return cleaned
    if generated_report_is_usable(recovered, minimum_words=minimum_words):
        print("Citation cleanup collapsed the recovery; preserving the quality-gated recovery", flush=True)
        return recovered
    raise RuntimeError("Recovered report remained unusable after atomic-citation cleanup")


def recover_sanitized_report(
    report: str, papers, model, namespace: str, word_range: str,
) -> str:
    """Regenerate only when atomic-citation cleanup collapses an otherwise valid draft."""
    if generated_report_is_usable(report, minimum_words=300):
        return report
    evidence = evidence_block(papers[:10], 12000)
    prompt = (
        "Rebuild the incomplete DRAFT as a focused evidence-grounded English survey using only EVIDENCE. "
        f"Write {word_range} words. Every factual sentence must be atomic and contain exactly one verbatim URL "
        "from its supporting evidence card; never place two source URLs in the same sentence. Omit unsupported "
        "claims, cross-paper synthesis, a References section, and empty headings. Return only Markdown.\n\n"
        f"EVIDENCE:\n{evidence}\n\nINCOMPLETE DRAFT:\n{report}"
    )
    candidate = model.generate(
        [{"role": "user", "content": prompt}], temperature=0, max_tokens=32768,
        cache_namespace=namespace,
    )
    if not generated_report_is_usable(candidate, minimum_words=300):
        raise RuntimeError("Post-sanitize recovery returned an unusable report")
    return candidate


def _repair_output_is_usable(candidate: str, draft: str, word_range: str, source_range: str) -> bool:
    """Reject nominally successful edits that collapse the report or its references."""
    candidate_words = len(candidate.split())
    draft_words = len(draft.split())
    word_match = re.search(r"(\d+)", word_range)
    requested_min = int(word_match.group(1)) if word_match else 0
    if draft_words >= requested_min * 0.6 and candidate_words < requested_min * 0.6:
        return False

    draft_urls = set(re.findall(r"https?://[^\s)\]>]+", draft, flags=re.I))
    candidate_urls = set(re.findall(r"https?://[^\s)\]>]+", candidate, flags=re.I))
    source_match = re.search(r"(\d+)", source_range)
    requested_sources = int(source_match.group(1)) if source_match else 0
    minimum_sources = min(len(draft_urls), max(2, math.ceil(requested_sources * 0.6)))
    return len(candidate_urls) >= minimum_sources


def repair_grounded_report(
    report, papers, model, namespace: str, word_range: str, source_range: str = "6-8",
) -> str:
    """Run a bounded evidence editor that deletes unsupported or uncited factual prose."""
    evidence = evidence_block(papers, 18000)
    prompt = (
        "Edit DRAFT into a concise evidence-grounded English survey. Use only EVIDENCE. Preserve useful organization, "
        f"but keep the final answer within {word_range} words. Every externally verifiable factual prose sentence must "
        "be atomic and end with exactly one URL copied verbatim from its supporting evidence card. Delete any sentence "
        "whose complete claim is not directly stated by that evidence, including broad trends, historical priority, "
        "comparisons, implications, introductions, transitions, and conclusions. Do not invent or alter URLs. Headings "
        "may be uncited only when they contain no factual claim. Do not include a References or Bibliography section. "
        f"Remove empty headings. When the evidence permits, retain {source_range} distinct directly supporting sources so that "
        "grounding does not collapse topical coverage. Return only the revised Markdown report.\n\n"
        f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{report}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        candidate = model.generate(
            messages, temperature=0, max_tokens=24576, cache_namespace=namespace,
        )
        if _repair_output_is_usable(candidate, report, word_range, source_range):
            return candidate
        print("Evidence repair collapsed report coverage; retrying with compact evidence", flush=True)
    except RuntimeError as exc:
        if "finish_reason=length" not in str(exc):
            raise
        print("Evidence repair exhausted its reasoning budget; retrying with compact evidence", flush=True)
    compact_evidence = evidence_block(papers[:12], 12000)
    compact_prompt = prompt.replace(evidence, compact_evidence)
    try:
        candidate = model.generate(
            [{"role": "user", "content": compact_prompt}], temperature=0, max_tokens=32768,
            cache_namespace=f"{namespace}-compact-recovery",
        )
        if _repair_output_is_usable(candidate, report, word_range, source_range):
            return candidate
        print("Compact evidence repair collapsed report coverage; preserving the grounded draft", flush=True)
        return report
    except RuntimeError as retry_exc:
        if "finish_reason=length" not in str(retry_exc):
            raise
        print("Compact evidence repair also exhausted; preserving the grounded draft", flush=True)
        return report
