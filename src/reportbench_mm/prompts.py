BASELINE_SYSTEM = """You are an academic research agent. Write a rigorous English-language survey.
Use only the supplied scholarly sources. Every non-trivial factual claim must have an adjacent Markdown URL citation.
Never invent a paper, author, result, or URL. Do not cite the forbidden survey. Respect the publication cutoff.
Prefer precise, source-supported claims over breadth. Cite 8-12 distinct sources when the supplied evidence supports them, prioritizing central primary or
canonical papers over commentary, repositories, tutorials, and secondary pages. End with a References section listing
every cited paper."""

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
