"""Tests for server-side envelope construction.

These tests pin down the wire format for each tool by mocking the providers
and invoking the tool functions directly (no real VLM, no real OCR).
"""

from __future__ import annotations

import base64
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import agent_vision_mcp.server as server
from agent_vision_mcp.errors import InvalidInputError, VisionMCPError
from agent_vision_mcp.provider_result import build_provider_result
from agent_vision_mcp.responses import (
    ResponseEnvelopeAdapter,
    FailureEnvelope,
    SuccessEnvelope,
)


def _png_bytes(width: int = 4, height: int = 3) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _png_path() -> tuple[TemporaryDirectory, str]:
    directory = TemporaryDirectory()
    path = Path(directory.name) / "tiny.png"
    path.write_bytes(_png_bytes())
    return directory, str(path)


def _stub_vlm(text: str = "model says hi", model: str = "glm-4v-flash"):
    """Replace the module-level vlm_provider with a stub returning a ProviderResult."""
    stub = SimpleNamespace(
        model_id=model,
        analyze=lambda *args, **kwargs: build_provider_result(text=text, model=model),
    )
    return stub


def _stub_ocr(text: str = "ocr text", model: str = "ocr-model"):
    return SimpleNamespace(
        model_id=model,
        analyze=lambda *args, **kwargs: build_provider_result(text=text, model=model),
    )


class VisionAnalyzeEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._vlm_backup = server.vlm_provider

    def tearDown(self) -> None:
        server.vlm_provider = self._vlm_backup

    def _patch_vlm(self, text: str = "model says hi") -> None:
        server.vlm_provider = _stub_vlm(text=text)

    def test_success_envelope_shape(self) -> None:
        directory, path = _png_path()
        try:
            self._patch_vlm()
            out = server.vision_analyze(image_source=path, task="ui")
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, SuccessEnvelope)
            self.assertEqual(parsed.tool, "vision_analyze")
            self.assertEqual(parsed.task, "ui")
            self.assertEqual(parsed.model, "glm-4v-flash")
            self.assertEqual(parsed.result["summary"], "model says hi")
            self.assertEqual(parsed.result["observations"], [])
            self.assertEqual(parsed.sources, [])
            self.assertIsNotNone(parsed.source)
            self.assertEqual(parsed.source.type, "file")
            self.assertEqual(parsed.warnings, [])
            self.assertIsNone(parsed.raw_model_output)
            self.assertEqual(
                parsed.result["suggested_followups"],
                [{"tool": "vision_crop_analyze",
                  "hint": "Zoom into specific regions for more detail"}],
            )
        finally:
            directory.cleanup()

    def test_invalid_input_returns_failure_envelope(self) -> None:
        self._patch_vlm()
        out = server.vision_analyze(
            image_source="data:image/png;base64,SGVsbG8=",
            task="ui",
        )
        parsed = ResponseEnvelopeAdapter.validate_json(out)
        self.assertIsInstance(parsed, FailureEnvelope)
        self.assertEqual(parsed.tool, "vision_analyze")
        self.assertEqual(parsed.error.code, "INVALID_INPUT")
        self.assertEqual(parsed.task, "ui")
        self.assertEqual(parsed.model, "glm-4v-flash")
        self.assertIsNone(parsed.source)
        self.assertEqual(parsed.sources, [])
        self.assertEqual(parsed.warnings, [])

    def test_invalid_task_returns_failure_envelope(self) -> None:
        out = server.vision_analyze(image_source="/tmp/x.png", task="nope")
        parsed = ResponseEnvelopeAdapter.validate_json(out)
        self.assertIsInstance(parsed, FailureEnvelope)
        self.assertEqual(parsed.error.code, "INVALID_INPUT")

    def test_include_raw_passes_through(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm()
            out = server.vision_analyze(
                image_source=path,
                task="ui",
                include_raw=True,
            )
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsNotNone(parsed.raw_model_output)
            self.assertEqual(parsed.raw_model_output["text"], "model says hi")
            self.assertEqual(parsed.raw_model_output["model"], "glm-4v-flash")
        finally:
            directory.cleanup()

    def test_source_ref_redacted_by_default(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm()
            out = server.vision_analyze(image_source=path, task="ui")
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsNone(parsed.source.source_ref)
        finally:
            directory.cleanup()

    def test_source_ref_basename_when_enabled(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm()
            out = server.vision_analyze(
                image_source=path, task="ui", include_source_ref=True
            )
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertEqual(parsed.source.source_ref, "tiny.png")
        finally:
            directory.cleanup()


class VisionInspectEnvelopeTest(unittest.TestCase):
    def test_returns_envelope_with_metadata(self) -> None:
        directory, path = _png_path()
        try:
            out = server.vision_inspect(image_source=path)
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, SuccessEnvelope)
            self.assertEqual(parsed.tool, "vision_inspect")
            self.assertEqual(parsed.result["width"], 4)
            self.assertEqual(parsed.result["height"], 3)
            self.assertEqual(parsed.result["source_type"], "file")
            self.assertIsNone(parsed.source.source_ref)
        finally:
            directory.cleanup()


class VisionCropAnalyzeEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._vlm_backup = server.vlm_provider

    def tearDown(self) -> None:
        server.vlm_provider = self._vlm_backup

    def test_crop_envelope_has_region(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm(text="cropped text")
            out = server.vision_crop_analyze(
                image_source=path,
                x=0.1, y=0.2, width=0.5, height=0.5,
                task="document",
            )
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, SuccessEnvelope)
            self.assertEqual(parsed.tool, "vision_crop_analyze")
            crop = parsed.result["crop"]
            self.assertEqual(crop, {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5})
            self.assertEqual(parsed.result["summary"], "cropped text")
            self.assertEqual(parsed.result["observations"], [])
        finally:
            directory.cleanup()

    def test_invalid_crop_returns_failure_envelope(self) -> None:
        out = server.vision_crop_analyze(
            image_source="/tmp/x.png",
            x=0.8, y=0.8, width=0.3, height=0.3,
        )
        parsed = ResponseEnvelopeAdapter.validate_json(out)
        self.assertIsInstance(parsed, FailureEnvelope)
        self.assertEqual(parsed.error.code, "INVALID_INPUT")


class VisionExtractTextEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._vlm_backup = server.vlm_provider
        self._ocr_backup = server.ocr_provider

    def tearDown(self) -> None:
        server.vlm_provider = self._vlm_backup
        server.ocr_provider = self._ocr_backup

    def test_no_ocr_provider_uses_vlm(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm(text="extracted text")
            server.ocr_provider = None
            out = server.vision_extract_text(image_source=path)
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, SuccessEnvelope)
            self.assertEqual(parsed.tool, "vision_extract_text")
            self.assertEqual(parsed.result["text"], "extracted text")
            self.assertEqual(len(parsed.result["blocks"]), 1)
            self.assertEqual(parsed.warnings, [])
        finally:
            directory.cleanup()

    def test_ocr_success_no_warning(self) -> None:
        directory, path = _png_path()
        try:
            server.ocr_provider = _stub_ocr(text="ocr says hi")
            server.vlm_provider = _stub_vlm(text="fallback text")
            out = server.vision_extract_text(image_source=path)
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertEqual(parsed.result["text"], "ocr says hi")
            self.assertEqual(parsed.warnings, [])
        finally:
            directory.cleanup()

    def test_ocr_failure_falls_back_to_vlm_with_warning(self) -> None:
        directory, path = _png_path()
        try:
            def ocr_stub_analyze(*args, **kwargs):
                raise VisionMCPError("boom", code="PROVIDER_ERROR", retryable=False)

            server.ocr_provider = SimpleNamespace(
                model_id="ocr-model",
                analyze=ocr_stub_analyze,
            )
            server.vlm_provider = _stub_vlm(text="vlm fallback text")
            out = server.vision_extract_text(image_source=path)
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertEqual(parsed.result["text"], "vlm fallback text")
            self.assertEqual(
                parsed.warnings,
                ["Dedicated OCR provider failed; used VLM fallback."],
            )
        finally:
            directory.cleanup()


class VisionCompareEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._vlm_backup = server.vlm_provider

    def tearDown(self) -> None:
        server.vlm_provider = self._vlm_backup

    def test_compare_passes_multiple_sources(self) -> None:
        directory_a, path_a = _png_path()
        directory_b, path_b = _png_path()
        try:
            server.vlm_provider = _stub_vlm(text="differences here")
            out = server.vision_compare(image_sources=[path_a, path_b], focus="layout")
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, SuccessEnvelope)
            self.assertEqual(parsed.tool, "vision_compare")
            self.assertIsNone(parsed.source)
            self.assertEqual(len(parsed.sources), 2)
            self.assertEqual(parsed.result["summary"], "differences here")
            self.assertEqual(parsed.result["differences"], [])
            self.assertEqual(parsed.result["same_elements"], [])
        finally:
            directory_a.cleanup()
            directory_b.cleanup()

    def test_too_few_images_returns_failure_envelope(self) -> None:
        directory, path = _png_path()
        try:
            server.vlm_provider = _stub_vlm()
            out = server.vision_compare(image_sources=[path])
            parsed = ResponseEnvelopeAdapter.validate_json(out)
            self.assertIsInstance(parsed, FailureEnvelope)
            self.assertEqual(parsed.error.code, "INVALID_INPUT")
        finally:
            directory.cleanup()


class VisionCapabilitiesEnvelopeTest(unittest.TestCase):
    def test_capabilities_envelope(self) -> None:
        out = server.vision_capabilities()
        parsed = ResponseEnvelopeAdapter.validate_json(out)
        self.assertIsInstance(parsed, SuccessEnvelope)
        self.assertEqual(parsed.tool, "vision_capabilities")
        self.assertIsNone(parsed.task)
        self.assertIsNone(parsed.model)
        self.assertIsNone(parsed.source)
        self.assertEqual(parsed.sources, [])
        self.assertEqual(parsed.warnings, [])
        self.assertIsNone(parsed.raw_model_output)
        self.assertEqual(parsed.result["server"], "agent-vision-mcp")


class SourceRefRedactionTest(unittest.TestCase):
    def test_url_netloc_and_path(self) -> None:
        normalized = SimpleNamespace(
            source_type="url",
            original_source="https://cdn.example.com/img/foo.png?token=DEADBEEF",
        )
        ref = server._build_source_ref(normalized)  # type: ignore[attr-defined]
        self.assertEqual(ref, "cdn.example.com/img/foo.png")
        # Query string is dropped — no signed-token leak.
        self.assertNotIn("token", ref)
        self.assertNotIn("DEADBEEF", ref)

    def test_file_basename_only(self) -> None:
        normalized = SimpleNamespace(
            source_type="file",
            original_source="/var/secret/credit_card.png",
        )
        ref = server._build_source_ref(normalized)  # type: ignore[attr-defined]
        self.assertEqual(ref, "credit_card.png")

    def test_data_url_returns_none(self) -> None:
        normalized = SimpleNamespace(source_type="data_url", original_source="...")
        self.assertIsNone(server._build_source_ref(normalized))  # type: ignore[attr-defined]

    def test_base64_returns_none(self) -> None:
        normalized = SimpleNamespace(source_type="base64", original_source="...")
        self.assertIsNone(server._build_source_ref(normalized))  # type: ignore[attr-defined]
