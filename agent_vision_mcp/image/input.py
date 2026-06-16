"""Image input normalization and processing"""

import base64
import binascii
import io
import re
import mimetypes
import httpx
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

from agent_vision_mcp.errors import InvalidInputError, UnsupportedFormatError, ImageTooLargeError
from agent_vision_mcp.image.security import check_url_security

# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
}
FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}

# Data URL regex
DATA_URL_REGEX = re.compile(r"^data:([^;]+);base64,(.+)$", re.IGNORECASE)


class NormalizedImage:
    """Normalized image data for VLM consumption"""

    def __init__(
        self,
        source_type: str,  # url, file, data_url, base64
        mime_type: str,
        data_url: str,  # Data URL format for VLM
        width: Optional[int] = None,
        height: Optional[int] = None,
        size_bytes: Optional[int] = None,
        original_source: str = "",
    ):
        self.source_type = source_type
        self.mime_type = mime_type
        self.data_url = data_url
        self.width = width
        self.height = height
        self.size_bytes = size_bytes
        self.original_source = original_source


def detect_source_type(source: str) -> str:
    """Detect the type of image source"""
    source = source.strip()

    # Check for data URL
    if source.lower().startswith("data:"):
        return "data_url"

    # Check for file:// URL
    if source.lower().startswith("file://"):
        return "file"

    # Check for URL (http/https)
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return "url"

    # Check if it's a local path (starts with / or contains path separators)
    if source.startswith("/") or (("\\" in source) and not source.startswith("http")):
        return "file"

    # Check for bare base64 (no special characters that would indicate otherwise)
    # Base64 strings only contain A-Za-z0-9+/
    if re.match(r"^[A-Za-z0-9+/]+=*$", source) and len(source) >= 100:
        return "base64"

    # Default to file path
    return "file"


def get_mime_type_from_extension(path: str) -> Optional[str]:
    """Get MIME type from file extension"""
    ext = Path(path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(f"x{ext}")
    return mime_type


def normalize_data_url(
    source: str, max_pixels: int = 40_000_000
) -> Tuple[str, str, int, int]:
    """Normalize data URL to standard format. Returns (mime, data_url, width, height)."""
    match = DATA_URL_REGEX.match(source.strip())
    if not match:
        raise InvalidInputError("Invalid data URL format")

    mime_type = match.group(1).lower()
    base64_data = "".join(match.group(2).split())

    # Validate MIME type
    if mime_type not in SUPPORTED_MIME_TYPES:
        # Try to guess from extension in the data URL
        if "image/png" in mime_type or "png" in mime_type:
            mime_type = "image/png"
        elif "image/jpeg" in mime_type or "jpeg" in mime_type or "jpg" in mime_type:
            mime_type = "image/jpeg"
        elif "image/webp" in mime_type or "webp" in mime_type:
            mime_type = "image/webp"
        elif "image/gif" in mime_type or "gif" in mime_type:
            mime_type = "image/gif"
        else:
            raise UnsupportedFormatError(mime_type, list(SUPPORTED_MIME_TYPES))

    try:
        data = base64.b64decode(base64_data, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidInputError("Invalid base64 data in data URL")

    actual_mime, width, height = validate_image_bytes(data, max_pixels=max_pixels)
    data_url = f"data:{actual_mime};base64,{base64.b64encode(data).decode('ascii')}"
    return actual_mime, data_url, width, height


def normalize_base64(
    source: str, max_pixels: int = 40_000_000
) -> Tuple[str, str, int, int]:
    """Normalize bare base64 to data URL. Returns (mime, data_url, width, height)."""
    source = source.strip()

    # Remove any whitespace
    source = "".join(source.split())

    try:
        data = base64.b64decode(source, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidInputError("Invalid base64 image data")

    mime_type, width, height = validate_image_bytes(data, max_pixels=max_pixels)
    data_url = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"

    return mime_type, data_url, width, height


def validate_image_bytes(
    data: bytes, max_pixels: int = 40_000_000
) -> Tuple[str, int, int]:
    """Verify image bytes and return (mime_type, width, height)."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidInputError(
                    f"Image pixel count exceeds limit ({width}x{height}, max {max_pixels})"
                )
            image_format = image.format
            image.verify()
    except InvalidInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise InvalidInputError("Input is not a valid supported image")

    mime_type = FORMAT_TO_MIME.get(image_format or "")
    if not mime_type:
        raise UnsupportedFormatError(image_format or "unknown", list(FORMAT_TO_MIME))
    return mime_type, width, height


def normalize_url(
    source: str,
    max_size_mb: int = 10,
    max_pixels: int = 40_000_000,
    block_private_ips: bool = True,
) -> Tuple[str, str, int, int, int]:
    """Securely download a URL and convert the verified image to base64.

    Returns (mime, data_url, width, height, size_bytes).
    """
    source = source.strip()
    parsed = urlparse(source)

    if parsed.scheme not in ("http", "https"):
        raise InvalidInputError(f"Unsupported URL scheme: {parsed.scheme}")

    max_bytes = max_size_mb * 1024 * 1024
    current_url = source

    try:
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            for _ in range(6):
                check_url_security(current_url, block_private_ips=block_private_ips)
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise InvalidInputError("Image URL redirect has no location")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    try:
                        declared_size = int(content_length) if content_length else None
                    except ValueError:
                        declared_size = None
                    if declared_size and declared_size > max_bytes:
                        raise ImageTooLargeError(declared_size / (1024 * 1024), max_size_mb)

                    chunks = []
                    size_bytes = 0
                    for chunk in response.iter_bytes():
                        size_bytes += len(chunk)
                        if size_bytes > max_bytes:
                            raise ImageTooLargeError(size_bytes / (1024 * 1024), max_size_mb)
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    break
            else:
                raise InvalidInputError("Too many redirects while downloading image")
    except httpx.HTTPError as e:
        raise InvalidInputError(f"Failed to download image: {str(e)}")

    content_type, width, height = validate_image_bytes(content, max_pixels=max_pixels)
    b64_data = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64_data}"

    return content_type, data_url, width, height, size_bytes


def normalize_file(
    source: str,
    max_size_mb: int = 10,
    max_pixels: int = 40_000_000,
) -> Tuple[str, str, int, int, int]:
    """Normalize file path to data URL. Returns (mime, data_url, width, height, size_bytes)."""
    # Remove file:// prefix if present
    if source.lower().startswith("file://"):
        source = source[7:]  # Remove "file://"

    path = Path(source.strip())

    # Check file size before reading
    try:
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ImageTooLargeError(size_mb, max_size_mb)
    except FileNotFoundError:
        raise InvalidInputError(f"File not found: {path}")

    # Read file and encode to base64
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise InvalidInputError(f"File not found: {path}")
    except PermissionError:
        raise InvalidInputError(f"Permission denied: {path}")

    mime_type, width, height = validate_image_bytes(data, max_pixels=max_pixels)
    base64_data = base64.b64encode(data).decode("utf-8")
    data_url = f"data:{mime_type};base64,{base64_data}"

    return mime_type, data_url, width, height, size_bytes


def normalize_image_source(
    source: str,
    max_size_mb: int = 10,
    max_pixels: int = 40_000_000,
    url_mode: str = "download",
    block_private_ips: bool = True,
) -> NormalizedImage:
    """
    Normalize image source to data URL format.

    Args:
        source: Image source (URL, file path, data URL, or base64)
        max_size_mb: Maximum allowed image size in MB

    Returns:
        NormalizedImage object

    Raises:
        InvalidInputError: If input is invalid
        UnsupportedFormatError: If image format is not supported
        ImageTooLargeError: If image exceeds size limit
    """
    source = source.strip()
    source_type = detect_source_type(source)
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str = ""

    # Reject oversized encoded inputs before allocating decoded image bytes.
    if source_type in ("data_url", "base64"):
        encoded_data = source
        if source_type == "data_url":
            match = DATA_URL_REGEX.match(source)
            if not match:
                raise InvalidInputError("Invalid data URL format")
            encoded_data = match.group(2)
        encoded_length = len("".join(encoded_data.split()))
        size_bytes = encoded_length * 3 // 4
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ImageTooLargeError(size_mb, max_size_mb)

    if source_type == "data_url":
        mime_type, data_url, width, height = normalize_data_url(
            source, max_pixels=max_pixels
        )

    elif source_type == "base64":
        mime_type, data_url, width, height = normalize_base64(
            source, max_pixels=max_pixels
        )

    elif source_type == "url":
        if url_mode == "passthrough":
            return NormalizedImage(
                source_type=source_type,
                mime_type="",
                data_url=source,
                original_source=source,
            )
        mime_type, data_url, width, height, size_bytes = normalize_url(
            source,
            max_size_mb=max_size_mb,
            max_pixels=max_pixels,
            block_private_ips=block_private_ips,
        )

    elif source_type == "file":
        mime_type, data_url, width, height, size_bytes = normalize_file(
            source, max_size_mb, max_pixels
        )

    else:
        raise InvalidInputError(f"Unknown source type: {source_type}")

    return NormalizedImage(
        source_type=source_type,
        mime_type=mime_type,
        data_url=data_url,
        width=width,
        height=height,
        size_bytes=size_bytes,
        original_source=source,
    )


def is_supported_format(path_or_url: str) -> bool:
    """Check if the file extension is a supported image format"""
    ext = Path(path_or_url).suffix.lower()
    return ext in SUPPORTED_FORMATS
