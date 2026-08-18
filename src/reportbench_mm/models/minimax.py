from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..config import Settings


class MiniMaxClient:
    def __init__(self, settings: Settings, cache: JsonCache | None = None):
        self.settings = settings
        self.cache = cache

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        cache_namespace: str = "llm",
    ) -> str:
        self.settings.require_api_key()
        payload = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.settings.max_output_tokens,
        }

        def request() -> dict[str, Any]:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = Request(
                f"{self.settings.base_url}/text/chatcompletion_v2",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(http_request, timeout=self.settings.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"MiniMax API HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"MiniMax API connection failed: {exc.reason}") from exc

        data = self.cache.get_or_create(cache_namespace, payload, request) if self.cache else request()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected MiniMax response: {str(data)[:1000]}") from exc

    def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        text = self.generate(messages, **kwargs)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
            if match:
                return json.loads(match.group(1))
            raise RuntimeError(f"Model did not return valid JSON: {text[:500]}") from exc
