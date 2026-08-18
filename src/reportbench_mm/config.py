from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path



def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    root: Path
    model: str
    judge_model: str
    api_key: str
    base_url: str
    openalex_mailto: str
    request_timeout: int = 300
    max_output_tokens: int = 8192
    baseline_search_budget: int = 5
    rag_seed_count: int = 4
    rag_depth: int = 3
    rag_max_papers: int = 40

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        root = (root or Path.cwd()).resolve()
        _load_env_file(root / ".env")
        return cls(
            root=root,
            model=os.getenv("MINIMAX_MODEL", "MiniMax-M3"),
            judge_model=os.getenv("MINIMAX_JUDGE_MODEL", "MiniMax-M3"),
            api_key=os.getenv("MINIMAX_API_KEY", ""),
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1").rstrip("/"),
            openalex_mailto=os.getenv("OPENALEX_MAILTO", ""),
        )

    def require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is missing. Put it in an untracked .env file; "
                "never commit credentials."
            )
