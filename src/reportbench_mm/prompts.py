BASELINE_SYSTEM = """You are an academic research agent. Write a rigorous English-language survey.
Use only the supplied scholarly sources. Every non-trivial factual claim must have an adjacent Markdown URL citation.
Never invent a paper, author, result, or URL. Do not cite the forbidden survey. Respect the publication cutoff.
Prefer precise, source-supported claims over breadth. End with a References section listing every cited paper."""

RAG_SYSTEM = """You are an evidence-grounded academic survey writer. Use only the supplied evidence cards.
Every factual claim must be faithfully supported by its cited source and followed immediately by that source's canonical URL.
Do not cite the forbidden survey or any paper after the cutoff. Do not infer experimental results absent from evidence.
Organize the report by research themes, compare methods where evidence permits, identify limitations, and include References."""


def evidence_block(papers, char_limit: int) -> str:
    blocks: list[str] = []
    total = 0
    for index, paper in enumerate(papers, 1):
        text = (
            f"SOURCE {index}\nTitle: {paper.title}\nYear: {paper.year}\nURL: {paper.url}\n"
            f"Citation count metadata: {paper.cited_by_count}\nAbstract/evidence: {paper.abstract}\n"
        )
        if total + len(text) > char_limit:
            break
        blocks.append(text)
        total += len(text)
    return "\n".join(blocks)

