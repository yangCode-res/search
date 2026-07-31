import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pnsearch.clients.llm import OpenAICompatibleClient


class LLMClientUsageTest(unittest.TestCase):
    def test_tracks_controller_token_usage(self):
        response = {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }
        client = OpenAICompatibleClient("https://example.test/v1", "secret")
        with patch("pnsearch.clients.llm.post_json", new=AsyncMock(return_value=response)):
            result = asyncio.run(
                client.chat_json(model="mimo", system="system", user="user")
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            client.usage_snapshot(),
            {
                "request_attempts": 1,
                "successful_requests": 1,
                "failed_requests": 0,
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        )
