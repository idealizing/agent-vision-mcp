"""Error handling for agent-vision-mcp"""

from typing import Optional, Any
import json


class VisionMCPError(Exception):
    """Base exception for agent-vision-mcp"""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        retryable: bool = False,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class InvalidInputError(VisionMCPError):
    """Invalid input error"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="INVALID_INPUT", retryable=False, details=details)


class ImageTooLargeError(VisionMCPError):
    """Image exceeds size limit"""

    def __init__(self, size_mb: float, max_size_mb: int):
        super().__init__(
            f"Image exceeds max size {max_size_mb}MB (actual: {size_mb:.1f}MB)",
            code="IMAGE_TOO_LARGE",
            retryable=False,
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
        )


class UnsupportedFormatError(VisionMCPError):
    """Unsupported image format"""

    def __init__(self, format: str, supported: list):
        super().__init__(
            f"Unsupported image format: {format}. Supported: {', '.join(supported)}",
            code="UNSUPPORTED_FORMAT",
            retryable=False,
            details={"format": format, "supported": supported},
        )


class SecurityError(VisionMCPError):
    """Security violation"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="SECURITY_ERROR", retryable=False, details=details)


class ProviderError(VisionMCPError):
    """VLM provider error"""

    def __init__(self, message: str, retryable: bool = True, details: Optional[dict] = None):
        super().__init__(message, code="PROVIDER_ERROR", retryable=retryable, details=details)


class TimeoutError(VisionMCPError):
    """Request timeout"""

    def __init__(self, timeout: int):
        super().__init__(
            f"Request timeout after {timeout}s",
            code="TIMEOUT",
            retryable=True,
            details={"timeout": timeout},
        )


def handle_exception(e: Exception) -> str:
    """Convert exception to JSON error response"""
    if isinstance(e, VisionMCPError):
        return e.to_json()

    # Unknown error
    error = VisionMCPError(
        message="Internal error occurred",
        code="INTERNAL_ERROR",
        retryable=False,
        details={"type": type(e).__name__},
    )
    return error.to_json()