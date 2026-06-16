"""Tests for sanitized provider result handling.

Pins the contract that nothing sensitive leaks from a provider's raw
response into the envelope's `raw_model_output`:
  - headers / Authorization / request_id are dropped
  - signed URLs are dropped
  - non-JSON values become None
  - only whitelisted keys survive
"""

from __future__ import annotations

import unittest
from datetime import datetime

from agent_vision_mcp.provider_result import (
    ALLOWED_RESPONSE_METADATA_KEYS,
    ALLOWED_USAGE_KEYS,
    ProviderResult,
    _json_safe,
    build_provider_result,
    sanitize_response_metadata,
    sanitize_usage_metadata,
)


class SanitizeResponseMetadataTest(unittest.TestCase):
    def test_empty_input_returns_empty_dict(self) -> None:
        self.assertEqual(sanitize_response_metadata(None), {})
        self.assertEqual(sanitize_response_metadata({}), {})

    def test_keeps_whitelisted_keys(self) -> None:
        raw = {"model_name": "glm-4v-flash", "finish_reason": "stop"}
        out = sanitize_response_metadata(raw)
        self.assertEqual(out, {"model_name": "glm-4v-flash", "finish_reason": "stop"})

    def test_drops_unwhitelisted_keys(self) -> None:
        raw = {
            "model_name": "glm-4v-flash",
            "headers": {"Authorization": "Bearer SECRET"},
            "request_id": "abc-123",
            "x_request_signature": "deadbeef",
        }
        out = sanitize_response_metadata(raw)
        self.assertIn("model_name", out)
        self.assertNotIn("headers", out)
        self.assertNotIn("request_id", out)
        self.assertNotIn("x_request_signature", out)

    def test_drops_authorization_header(self) -> None:
        raw = {
            "headers": {
                "Authorization": "Bearer SECRET_TOKEN",
                "X-Request-Id": "abc-123",
            },
        }
        out = sanitize_response_metadata(raw)
        self.assertEqual(out, {})

    def test_drops_signed_url_substring(self) -> None:
        raw = {
            "url": "https://example.com/image.png?signature=DEADBEEF",
            "model_name": "glm-4v-flash",
        }
        out = sanitize_response_metadata(raw)
        self.assertIn("model_name", out)
        self.assertNotIn("url", out)

    def test_nested_dict_in_whitelisted_key_is_safe(self) -> None:
        # If a whitelisted key happens to contain a non-JSON value, the
        # nested value is coerced to None rather than stringified (which
        # could leak bytes-like repr).
        raw = {"model_name": "ok", "system_fingerprint": {"nested": datetime.now()}}
        out = sanitize_response_metadata(raw)
        self.assertEqual(out["model_name"], "ok")
        # datetime -> None under _json_safe.
        self.assertIsNone(out["system_fingerprint"]["nested"])

    def test_list_in_whitelisted_key_is_recursively_safe(self) -> None:
        raw = {"finish_reason": ["stop", {"inner": object()}]}
        out = sanitize_response_metadata(raw)
        self.assertEqual(out["finish_reason"][0], "stop")
        self.assertIsNone(out["finish_reason"][1]["inner"])


class SanitizeUsageMetadataTest(unittest.TestCase):
    def test_empty_input_returns_empty_dict(self) -> None:
        self.assertEqual(sanitize_usage_metadata(None), {})
        self.assertEqual(sanitize_usage_metadata({}), {})

    def test_keeps_numeric_token_counts(self) -> None:
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        self.assertEqual(sanitize_usage_metadata(raw), raw)

    def test_drops_non_numeric_token_counts(self) -> None:
        raw = {
            "input_tokens": "100",       # string — drop
            "output_tokens": None,       # None — drop
            "total_tokens": True,        # bool — drop (even though isinstance(int))
        }
        out = sanitize_usage_metadata(raw)
        self.assertEqual(out, {})

    def test_drops_unwhitelisted_keys(self) -> None:
        raw = {
            "input_tokens": 1,
            "request_id": "abc-123",
            "headers": {"Authorization": "x"},
        }
        out = sanitize_usage_metadata(raw)
        self.assertIn("input_tokens", out)
        self.assertNotIn("request_id", out)
        self.assertNotIn("headers", out)


class JsonSafeTest(unittest.TestCase):
    def test_primitives_pass_through(self) -> None:
        self.assertEqual(_json_safe("a"), "a")
        self.assertEqual(_json_safe(1), 1)
        self.assertEqual(_json_safe(1.5), 1.5)
        self.assertEqual(_json_safe(True), True)
        self.assertIsNone(_json_safe(None))

    def test_dict_recurses(self) -> None:
        out = _json_safe({"a": 1, "b": [1, 2, None]})
        self.assertEqual(out, {"a": 1, "b": [1, 2, None]})

    def test_datetime_becomes_none(self) -> None:
        self.assertIsNone(_json_safe(datetime.now()))

    def test_custom_object_becomes_none(self) -> None:
        class Custom:
            pass

        self.assertIsNone(_json_safe(Custom()))

    def test_tuple_becomes_list(self) -> None:
        self.assertEqual(_json_safe((1, 2, 3)), [1, 2, 3])


class BuildProviderResultTest(unittest.TestCase):
    def test_sanitizes_metadata(self) -> None:
        r = build_provider_result(
            text="hello",
            model="glm-4v-flash",
            response_metadata={
                "model_name": "glm-4v-flash",
                "headers": {"Authorization": "Bearer SECRET"},
                "request_id": "abc-123",
            },
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "request_id": "leaked",  # dropped
            },
        )
        self.assertEqual(r.text, "hello")
        self.assertEqual(r.model, "glm-4v-flash")
        self.assertIn("model_name", r.response_metadata)
        self.assertNotIn("headers", r.response_metadata)
        self.assertNotIn("request_id", r.response_metadata)
        self.assertEqual(r.usage_metadata["total_tokens"], 30)
        self.assertNotIn("request_id", r.usage_metadata)

    def test_empty_inputs(self) -> None:
        r = build_provider_result(text="hi")
        self.assertEqual(r.text, "hi")
        self.assertEqual(r.response_metadata, {})
        self.assertEqual(r.usage_metadata, {})

    def test_provider_result_is_pydantic_model(self) -> None:
        # Sanity: ProviderResult is a Pydantic BaseModel.
        r = ProviderResult(text="hi")
        self.assertEqual(r.text, "hi")
        self.assertIsNone(r.model)
        self.assertEqual(r.response_metadata, {})

    def test_whitelist_constants_are_frozen(self) -> None:
        # Whitelists are immutable — accidental mutation would be a security bug.
        self.assertIsInstance(ALLOWED_RESPONSE_METADATA_KEYS, frozenset)
        self.assertIsInstance(ALLOWED_USAGE_KEYS, frozenset)

    def test_whitelists_contain_expected_keys(self) -> None:
        self.assertEqual(
            ALLOWED_RESPONSE_METADATA_KEYS,
            {"model_name", "finish_reason", "system_fingerprint"},
        )
        self.assertEqual(
            ALLOWED_USAGE_KEYS,
            {"input_tokens", "output_tokens", "total_tokens"},
        )
