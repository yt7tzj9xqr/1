from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    arxiv_id: str
    title: str
    prompt: str
    application_domain: str
    update_date: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Task":
        required = ("arxiv_id", "title", "prompt", "application_domain")
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"Task is missing fields: {missing}")
        return cls(**{key: row.get(key, "") for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Paper:
    paper_id: str
    title: str
    year: int | None
    url: str
    abstract: str = ""
    doi: str = ""
    cited_by_count: int = 0
    referenced_work_ids: list[str] = field(default_factory=list)
    depth: int = 0
    relevance: float = 0.0
    source: str = "openalex"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
