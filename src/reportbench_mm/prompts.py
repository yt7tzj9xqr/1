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


def repair_grounded_report(report, papers, model, namespace: str, word_range: str) -> str:
    """Run a bounded evidence editor that deletes unsupported or uncited factual prose."""
    evidence = evidence_block(papers, 18000)
    prompt = (
        "Edit DRAFT into a concise evidence-grounded English survey. Use only EVIDENCE. Preserve useful organization, "
        f"but keep the final answer within {word_range} words. Every externally verifiable factual prose sentence must "
        "be atomic and end with exactly one URL copied verbatim from its supporting evidence card. Delete any sentence "
        "whose complete claim is not directly stated by that evidence, including broad trends, historical priority, "
        "comparisons, implications, introductions, transitions, and conclusions. Do not invent or alter URLs. Headings "
        "may be uncited only when they contain no factual claim. Do not include a References or Bibliography section. "
        "Remove empty headings. When the evidence permits, retain 6-8 distinct directly supporting sources so that "
        "grounding does not collapse topical coverage. Return only the revised Markdown report.\n\n"
        f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{report}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return model.generate(
            messages, temperature=0, max_tokens=24576, cache_namespace=namespace,
        )
    except RuntimeError as exc:
        if "finish_reason=length" not in str(exc):
            raise
        compact_evidence = evidence_block(papers[:12], 12000)
        compact_prompt = prompt.replace(evidence, compact_evidence)
        print("Evidence repair exhausted its reasoning budget; retrying with compact evidence", flush=True)
        try:
            return model.generate(
                [{"role": "user", "content": compact_prompt}], temperature=0, max_tokens=32768,
                cache_namespace=f"{namespace}-compact-recovery",
            )
        except RuntimeError as retry_exc:
            if "finish_reason=length" not in str(retry_exc):
                raise
            print("Compact evidence repair also exhausted; preserving the grounded draft", flush=True)
            return report
