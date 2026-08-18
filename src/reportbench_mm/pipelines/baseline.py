from __future__ import annotations

from ..config import Settings
from ..models import MiniMaxClient
from ..prompts import BASELINE_SYSTEM, evidence_block
from ..providers.openalex import OpenAlexProvider, compact_query, extract_cutoff, filter_papers
from ..schemas import Paper, Task


class BaselinePipeline:
    def __init__(self, settings: Settings, model: MiniMaxClient, scholar: OpenAlexProvider):
        self.settings = settings
        self.model = model
        self.scholar = scholar

    def retrieve(self, task: Task) -> list[Paper]:
        cutoff = extract_cutoff(task.prompt)
        query = compact_query(task.prompt)
        found = self.scholar.search(query, cutoff=cutoff, limit=self.settings.baseline_papers * 2)
        found = filter_papers(found, forbidden_title=task.title, cutoff=cutoff)
        usable = [paper for paper in found if paper.abstract and paper.url]
        return usable[: self.settings.baseline_papers]

    def run(self, task: Task) -> tuple[str, list[Paper]]:
        papers = self.retrieve(task)
        if not papers:
            raise RuntimeError(f"No usable scholarly sources found for {task.arxiv_id}")
        user = (
            f"TASK:\n{task.prompt}\n\nFORBIDDEN SURVEY:\n{task.title}\n\n"
            f"DOMAIN:\n{task.application_domain}\n\nSOURCES:\n"
            + evidence_block(papers, self.settings.evidence_char_limit)
        )
        report = self.model.generate(
            [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}],
            cache_namespace=f"baseline-report-v1:{self.settings.model}",
        )
        return report, papers
