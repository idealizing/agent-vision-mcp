"""Tests for the OpenAI-compatible provider adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agent_vision_mcp.providers.openai_compatible import OpenAICompatibleVisionProvider
from agent_vision_mcp.provider_result import ProviderResult


class OpenAICompatibleProviderTest(unittest.TestCase):
    def test_max_tokens_is_passed_as_model_argument(self) -> None:
        provider = object.__new__(OpenAICompatibleVisionProvider)
        provider.settings = SimpleNamespace(
            vision_supports_image_detail=False,
            vision_max_retries=1,
            vision_timeout=60,
        )
        provider.model_id = "glm-4v-flash"
        provider.vlm = Mock()
        provider.vlm.invoke.return_value = SimpleNamespace(
            content="done",
            response_metadata={},
            usage_metadata=None,
        )

        result = provider.analyze(
            images=[{"url": "https://example.com/image.png"}],
            prompt="describe",
            max_tokens=123,
        )

        self.assertIsInstance(result, ProviderResult)
        self.assertEqual(result.text, "done")
        _, kwargs = provider.vlm.invoke.call_args
        self.assertEqual(kwargs, {"max_tokens": 123})

    def test_response_metadata_is_sanitized(self) -> None:
        provider = object.__new__(OpenAICompatibleVisionProvider)
        provider.settings = SimpleNamespace(
            vision_supports_image_detail=False,
            vision_max_retries=1,
            vision_timeout=60,
        )
        provider.model_id = "glm-4v-flash"
        provider.vlm = Mock()
        provider.vlm.invoke.return_value = SimpleNamespace(
            content="ok",
            response_metadata={
                "model_name": "glm-4v-flash",
                "finish_reason": "stop",
                "headers": {"Authorization": "Bearer SECRET"},
                "request_id": "abc-123",
            },
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "request_id": "leaked",
            },
        )

        result = provider.analyze(
            images=[{"url": "https://example.com/image.png"}],
            prompt="describe",
        )

        # Whitelisted keys survive; headers/request_id are dropped.
        self.assertEqual(result.response_metadata["model_name"], "glm-4v-flash")
        self.assertEqual(result.response_metadata["finish_reason"], "stop")
        self.assertNotIn("headers", result.response_metadata)
        self.assertNotIn("request_id", result.response_metadata)
        # Whitelisted numeric usage survives; non-whitelisted is dropped.
        self.assertEqual(result.usage_metadata["total_tokens"], 30)
        self.assertNotIn("request_id", result.usage_metadata)

    def test_empty_text_becomes_empty_string(self) -> None:
        provider = object.__new__(OpenAICompatibleVisionProvider)
        provider.settings = SimpleNamespace(
            vision_supports_image_detail=False,
            vision_max_retries=1,
            vision_timeout=60,
        )
        provider.model_id = "glm-4v-flash"
        provider.vlm = Mock()
        provider.vlm.invoke.return_value = SimpleNamespace(
            content=None,
            response_metadata=None,
            usage_metadata=None,
        )

        result = provider.analyze(
            images=[{"url": "https://example.com/image.png"}],
            prompt="describe",
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.response_metadata, {})
        self.assertEqual(result.usage_metadata, {})
