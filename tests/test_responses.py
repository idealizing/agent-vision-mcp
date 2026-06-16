"""Tests for the unified response envelope.

These tests pin down the contract that server.py will rely on:
- Round-trip via ResponseEnvelopeAdapter
- Discriminator routing on `ok`
- Literal enforcement on Confidence / DifferenceType / SourceType
- BBoxNormalized tuple shape
- ValueError on unknown tool
- Stable key set: warnings: [] and sources: [] always present
- Per-tool result validation against the right model
"""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from agent_vision_mcp.responses import (
    BBoxNormalized,
    CompareDifference,
    Confidence,
    CropRegion,
    ErrorPayload,
    FailureEnvelope,
    ResponseEnvelope,
    ResponseEnvelopeAdapter,
    RESULT_MODEL_BY_TOOL,
    SourceMeta,
    SourceType,
    SuccessEnvelope,
    TextBlock,
    VisionAnalyzeResult,
    VisionCapabilitiesResult,
    VisionCompareResult,
    VisionCropAnalyzeResult,
    VisionExtractTextResult,
    VisionInspectResult,
    make_envelope,
    make_error_envelope,
)


def _parse(env_json: str):
    """Helper: parse a wire string into a typed envelope."""
    return ResponseEnvelopeAdapter.validate_json(env_json)


# ---------------------------------------------------------------------------
# Result model schema
# ---------------------------------------------------------------------------


class ResultModelTest(unittest.TestCase):
    def test_vision_analyze_result_required_summary(self) -> None:
        with self.assertRaises(ValidationError):
            VisionAnalyzeResult.model_validate({})

    def test_vision_analyze_result_default_empty_collections(self) -> None:
        r = VisionAnalyzeResult(summary="hi")
        self.assertEqual(r.observations, [])
        self.assertEqual(r.inferences, [])
        self.assertEqual(r.uncertainties, [])
        self.assertEqual(r.suggested_followups, [])

    def test_vision_extract_text_result_blocks_required(self) -> None:
        with self.assertRaises(ValidationError):
            VisionExtractTextResult.model_validate({"text": "abc"})

    def test_vision_compare_result_differences_default_empty(self) -> None:
        r = VisionCompareResult(summary="x")
        self.assertEqual(r.differences, [])
        self.assertEqual(r.same_elements, [])

    def test_crop_region_required_fields(self) -> None:
        with self.assertRaises(ValidationError):
            CropRegion.model_validate({"x": 0.1})
        r = CropRegion(x=0.1, y=0.2, width=0.3, height=0.4)
        self.assertEqual((r.x, r.y, r.width, r.height), (0.1, 0.2, 0.3, 0.4))


# ---------------------------------------------------------------------------
# Literal enforcement
# ---------------------------------------------------------------------------


class LiteralTest(unittest.TestCase):
    def test_compare_difference_type_literal_rejects_unknown(self) -> None:
        with self.assertRaises(ValidationError):
            CompareDifference(type="maybe", description="x")  # type: ignore[arg-type]
        # Valid types pass through.
        CompareDifference(type="added", description="x")
        CompareDifference(type="removed", description="x")
        CompareDifference(type="modified", description="x")
        CompareDifference(type="moved", description="x")

    def test_source_type_literal_rejects_unknown(self) -> None:
        with self.assertRaises(ValidationError):
            SourceMeta(type="ftp")  # type: ignore[arg-type]

    def test_text_block_confidence_default_medium(self) -> None:
        b = TextBlock(order=1, type="text", text="hi")
        self.assertEqual(b.confidence, "medium")
        self.assertIsNone(b.bbox_normalized)

    def test_text_block_invalid_confidence_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TextBlock(order=1, type="text", text="hi", confidence="maybe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BBoxNormalized shape
# ---------------------------------------------------------------------------


class BBoxTest(unittest.TestCase):
    def test_bbox_four_floats_accepted(self) -> None:
        b = TextBlock(order=1, type="text", text="x", bbox_normalized=(0.1, 0.2, 0.3, 0.4))
        # Pydantic stores tuples as tuples when input is a tuple.
        self.assertEqual(b.bbox_normalized, (0.1, 0.2, 0.3, 0.4))

    def test_bbox_wrong_length_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TextBlock(order=1, type="text", text="x", bbox_normalized=(0.1, 0.2, 0.3))  # type: ignore[arg-type]

    def test_bbox_non_numeric_rejected(self) -> None:
        # A dict is not coercible to a 4-tuple of floats.
        with self.assertRaises(ValidationError):
            TextBlock(
                order=1,
                type="text",
                text="x",
                bbox_normalized={"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},  # type: ignore[arg-type]
            )

    def test_bbox_list_coerced_to_tuple(self) -> None:
        # Pydantic v2's lax tuple validation accepts a list and coerces. We
        # only care that the resulting value is 4 floats in a tuple shape.
        b = TextBlock(
            order=1,
            type="text",
            text="x",
            bbox_normalized=[0.1, 0.2, 0.3, 0.4],  # type: ignore[arg-type]
        )
        self.assertEqual(tuple(b.bbox_normalized), (0.1, 0.2, 0.3, 0.4))


# ---------------------------------------------------------------------------
# Builders: success envelope
# ---------------------------------------------------------------------------


class MakeEnvelopeTest(unittest.TestCase):
    def test_unknown_tool_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            make_envelope(
                tool="vision_nonexistent",
                result=VisionAnalyzeResult(summary="x"),
            )
        self.assertIn("vision_nonexistent", str(cm.exception))

    def test_result_dict_validated_against_per_tool_model(self) -> None:
        # Missing required field triggers ValidationError, not silent coercion.
        with self.assertRaises(ValidationError):
            make_envelope(tool="vision_analyze", result={})

    def test_result_dict_passes_through(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "ok"},
        )
        parsed = _parse(env_json)
        self.assertIsInstance(parsed, SuccessEnvelope)
        self.assertEqual(parsed.result["summary"], "ok")
        self.assertEqual(parsed.result["observations"], [])

    def test_result_basemodel_accepted(self) -> None:
        r = VisionAnalyzeResult(summary="ok", suggested_followups=[{"tool": "x"}])
        env_json = make_envelope(tool="vision_analyze", result=r)
        parsed = _parse(env_json)
        self.assertIsInstance(parsed, SuccessEnvelope)
        self.assertEqual(parsed.result["summary"], "ok")
        self.assertEqual(parsed.result["suggested_followups"], [{"tool": "x"}])

    def test_wrong_basemodel_type_is_revalidated(self) -> None:
        # Passing a VisionExtractTextResult where vision_analyze is expected
        # should fail validation (no `summary` field).
        wrong = VisionExtractTextResult(
            text="t", blocks=[TextBlock(order=1, type="text", text="b")]
        )
        with self.assertRaises(ValidationError):
            make_envelope(tool="vision_analyze", result=wrong)

    def test_success_envelope_always_has_warnings_list(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
        )
        parsed = _parse(env_json)
        self.assertEqual(parsed.warnings, [])
        # Round-trip: warnings is present in the raw JSON, not omitted.
        raw = json.loads(env_json)
        self.assertIn("warnings", raw)
        self.assertEqual(raw["warnings"], [])

    def test_success_envelope_always_has_sources_list(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
        )
        raw = json.loads(env_json)
        self.assertIn("sources", raw)
        self.assertEqual(raw["sources"], [])

    def test_single_image_tool_passes_source(self) -> None:
        src = SourceMeta(type="file", mime_type="image/png", width=10, height=20)
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
            source=src,
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["source"]["type"], "file")
        self.assertEqual(raw["source"]["width"], 10)
        self.assertEqual(raw["sources"], [])

    def test_vision_compare_passes_sources(self) -> None:
        env_json = make_envelope(
            tool="vision_compare",
            result={"summary": "x"},
            sources=[
                SourceMeta(type="file", mime_type="image/png"),
                SourceMeta(type="file", mime_type="image/png"),
            ],
        )
        raw = json.loads(env_json)
        self.assertIsNone(raw["source"])
        self.assertEqual(len(raw["sources"]), 2)

    def test_raw_model_output_optional(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
        )
        raw = json.loads(env_json)
        self.assertIn("raw_model_output", raw)
        self.assertIsNone(raw["raw_model_output"])

    def test_raw_model_output_passes_through(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
            raw_model_output={
                "model": "glm-4v-flash",
                "response_metadata": {"model_name": "glm-4v-flash", "finish_reason": "stop"},
                "usage_metadata": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["raw_model_output"]["model"], "glm-4v-flash")
        self.assertEqual(raw["raw_model_output"]["response_metadata"]["finish_reason"], "stop")

    def test_task_and_model_optional(self) -> None:
        env_json = make_envelope(
            tool="vision_capabilities",
            result={
                "server": "x",
                "version": "0",
                "vlm_provider": {},
                "ocr_provider": None,
                "ocr_enabled": False,
                "tools": {},
                "supports": {},
                "limits": {},
                "task_types": [],
            },
        )
        raw = json.loads(env_json)
        self.assertIn("task", raw)
        self.assertIn("model", raw)
        self.assertIsNone(raw["task"])
        self.assertIsNone(raw["model"])

    def test_schema_version_default(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["schema_version"], "1.0")
        self.assertTrue(raw["ok"])


# ---------------------------------------------------------------------------
# Builders: error envelope
# ---------------------------------------------------------------------------


class MakeErrorEnvelopeTest(unittest.TestCase):
    def test_basic_error_shape(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="INVALID_INPUT",
            message="bad input",
            retryable=False,
        )
        parsed = _parse(env_json)
        self.assertIsInstance(parsed, FailureEnvelope)
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.error.code, "INVALID_INPUT")
        self.assertEqual(parsed.error.message, "bad input")
        self.assertFalse(parsed.error.retryable)
        self.assertEqual(parsed.error.details, {})

    def test_error_envelope_always_has_warnings_list(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="X",
            message="x",
        )
        raw = json.loads(env_json)
        self.assertIn("warnings", raw)
        self.assertEqual(raw["warnings"], [])

    def test_error_envelope_always_has_sources_list(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="X",
            message="x",
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["sources"], [])
        self.assertIsNone(raw["source"])
        self.assertIsNone(raw["result"])
        self.assertIsNone(raw["raw_model_output"])

    def test_error_envelope_task_and_model_propagate(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="X",
            message="x",
            task="ui",
            model="glm-4v-flash",
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["task"], "ui")
        self.assertEqual(raw["model"], "glm-4v-flash")

    def test_error_details_default_empty_dict(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="X",
            message="x",
        )
        raw = json.loads(env_json)
        self.assertEqual(raw["error"]["details"], {})


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


class DiscriminatorTest(unittest.TestCase):
    def test_ok_true_picks_success(self) -> None:
        env_json = make_envelope(
            tool="vision_analyze",
            result={"summary": "x"},
        )
        parsed = _parse(env_json)
        self.assertIsInstance(parsed, SuccessEnvelope)
        self.assertTrue(parsed.ok)

    def test_ok_false_picks_failure(self) -> None:
        env_json = make_error_envelope(
            tool="vision_analyze",
            code="X",
            message="x",
        )
        parsed = _parse(env_json)
        self.assertIsInstance(parsed, FailureEnvelope)
        self.assertFalse(parsed.ok)

    def test_failure_result_must_be_none(self) -> None:
        bad = {
            "schema_version": "1.0",
            "ok": False,
            "tool": "vision_analyze",
            "task": None,
            "model": None,
            "source": None,
            "sources": [],
            "result": {"summary": "x"},  # failure must have result=None
            "warnings": [],
            "raw_model_output": None,
            "error": {"code": "X", "message": "x", "retryable": False, "details": {}},
        }
        with self.assertRaises(ValidationError):
            ResponseEnvelopeAdapter.validate_python(bad)

    def test_success_must_have_error_none(self) -> None:
        bad = {
            "schema_version": "1.0",
            "ok": True,
            "tool": "vision_analyze",
            "task": None,
            "model": None,
            "source": None,
            "sources": [],
            "result": {"summary": "x"},
            "warnings": [],
            "raw_model_output": None,
            "error": {"code": "X", "message": "x", "retryable": False, "details": {}},  # must be None
        }
        with self.assertRaises(ValidationError):
            ResponseEnvelopeAdapter.validate_python(bad)

    def test_response_envelope_alias_exported(self) -> None:
        # Smoke test: the Annotated alias exists and can be referenced.
        self.assertIsNotNone(ResponseEnvelope)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistryTest(unittest.TestCase):
    def test_all_six_tools_registered(self) -> None:
        self.assertEqual(
            set(RESULT_MODEL_BY_TOOL.keys()),
            {
                "vision_analyze",
                "vision_extract_text",
                "vision_compare",
                "vision_crop_analyze",
                "vision_inspect",
                "vision_capabilities",
            },
        )

    def test_registry_models_match_exports(self) -> None:
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_analyze"], VisionAnalyzeResult)
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_extract_text"], VisionExtractTextResult)
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_compare"], VisionCompareResult)
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_crop_analyze"], VisionCropAnalyzeResult)
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_inspect"], VisionInspectResult)
        self.assertIs(RESULT_MODEL_BY_TOOL["vision_capabilities"], VisionCapabilitiesResult)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class ReExportsTest(unittest.TestCase):
    def test_error_payload_independent_of_error_envelope(self) -> None:
        # ErrorPayload is a standalone model used by FailureEnvelope.error.
        e = ErrorPayload(code="X", message="x", retryable=False, details={"k": 1})
        self.assertEqual(e.code, "X")
        self.assertEqual(e.details, {"k": 1})
