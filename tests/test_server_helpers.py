"""Tests for server-side routing and argument validation."""

import unittest
from unittest.mock import patch

from agent_vision_mcp.errors import VisionMCPError
from agent_vision_mcp.server import load_and_validate_image, validate_crop


class ServerHelpersTest(unittest.TestCase):
    def test_auto_mode_passes_analysis_url_through(self) -> None:
        with patch("agent_vision_mcp.server.validate_image_source"), patch(
            "agent_vision_mcp.server.settings.vision_url_mode", "auto"
        ):
            normalized, source_type = load_and_validate_image("https://example.com/image.png")

        self.assertEqual(source_type, "url")
        self.assertEqual(normalized.data_url, "https://example.com/image.png")

    def test_invalid_crop_is_rejected(self) -> None:
        with self.assertRaises(VisionMCPError):
            validate_crop(0.8, 0.8, 0.3, 0.3)
