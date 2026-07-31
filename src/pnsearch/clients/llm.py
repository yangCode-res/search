from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .http import HTTPError, post_json


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
        data = None
        last_error: Exception | None = None
        for attempt in range(3):
            request_payload = dict(payload)
            if attempt == 2:
                # Some OpenAI-compatible controllers enforce JSON through the prompt but do not
                # implement response_format. The final retry keeps interoperability with them.
                request_payload.pop("response_format", None)
            try:
                data = await post_json(
                    f"{self.base_url}/chat/completions",
                    request_payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                break
            except HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        if data is None:
            raise LLMResponseError(f"chat completion failed after retries: {last_error}")
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
