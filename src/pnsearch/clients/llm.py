from __future__ import annotations

import json
import re
from typing import Any

from .http import post_json


class LLMResponseError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"invalid chat response: {data}") from exc
        return parse_json_object(content)


def parse_json_object(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise LLMResponseError("model response contains no JSON object")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("model response contains invalid JSON") from exc
