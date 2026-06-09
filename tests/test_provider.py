"""Tests for the OpenAI-compatible provider adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agent_vision_mcp.providers.openai_compatible import OpenAICompatibleVisionProvider


class OpenAICompatibleProviderTest(unittest.TestCase):
    def test_max_tokens_is_passed_as_model_argument(self) -> None:
        provider = object.__new__(OpenAICompatibleVisionProvider)
        provider.settings = SimpleNamespace(
            vision_supports_image_detail=False,
            vision_max_retries=1,
            vision_timeout=60,
        )
        provider.vlm = Mock()
        provider.vlm.invoke.return_value = SimpleNamespace(content="done")

        result = provider.analyze(
            images=[{"url": "https://example.com/image.png"}],
            prompt="describe",
            max_tokens=123,
        )

        self.assertEqual(result, "done")
        _, kwargs = provider.vlm.invoke.call_args
        self.assertEqual(kwargs, {"max_tokens": 123})
