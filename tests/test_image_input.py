"""Tests for image input normalization and validation."""

import base64
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from PIL import Image

from agent_vision_mcp.errors import ImageTooLargeError, InvalidInputError, SecurityError
from agent_vision_mcp.image.input import normalize_image_source, normalize_url
from agent_vision_mcp.image.security import is_private_ip


def png_bytes(width: int = 4, height: int = 3) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class ImageInputTest(unittest.TestCase):
    def test_url_passthrough_does_not_download(self) -> None:
        normalized = normalize_image_source(
            "https://example.com/image.png",
            url_mode="passthrough",
        )
        self.assertEqual(normalized.data_url, "https://example.com/image.png")
        self.assertEqual(normalized.source_type, "url")

    def test_rejects_fake_data_url(self) -> None:
        source = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
        with self.assertRaises(InvalidInputError):
            normalize_image_source(source)

    def test_rejects_oversized_base64_before_decoding(self) -> None:
        source = "data:image/png;base64," + ("A" * 1_500_000)
        with patch("agent_vision_mcp.image.input.base64.b64decode") as decode:
            with self.assertRaises(ImageTooLargeError):
                normalize_image_source(source, max_size_mb=1)
        decode.assert_not_called()

    def test_detects_mime_from_image_content(self) -> None:
        source = "data:image/jpeg;base64," + base64.b64encode(png_bytes()).decode()
        normalized = normalize_image_source(source)
        self.assertEqual(normalized.mime_type, "image/png")
        self.assertTrue(normalized.data_url.startswith("data:image/png;base64,"))

    def test_rejects_excessive_pixel_count(self) -> None:
        source = "data:image/png;base64," + base64.b64encode(png_bytes(10, 10)).decode()
        with self.assertRaises(InvalidInputError):
            normalize_image_source(source, max_pixels=99)

    def test_file_content_is_verified(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fake.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(InvalidInputError):
                normalize_image_source(str(path))

    def test_non_global_addresses_are_blocked(self) -> None:
        self.assertTrue(is_private_ip("127.0.0.1"))
        self.assertTrue(is_private_ip("224.0.0.1"))
        self.assertTrue(is_private_ip("::"))
        self.assertFalse(is_private_ip("1.1.1.1"))

    def test_download_rechecks_redirect_target(self) -> None:
        real_client = httpx.Client
        checked_urls = []

        def check_url(url: str, **_: object) -> None:
            checked_urls.append(url)
            if "127.0.0.1" in url:
                raise SecurityError("blocked redirect")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://127.0.0.1/image.png"})

        def make_client(**kwargs: object) -> httpx.Client:
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(**kwargs)

        with patch("agent_vision_mcp.image.input.check_url_security", side_effect=check_url), patch(
            "agent_vision_mcp.image.input.httpx.Client", side_effect=make_client
        ):
            with self.assertRaises(SecurityError):
                normalize_url("https://example.com/image.png")

        self.assertEqual(
            checked_urls,
            ["https://example.com/image.png", "http://127.0.0.1/image.png"],
        )
