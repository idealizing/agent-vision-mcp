"""Sanitized provider output container.

Providers (VLM, OCR) return a `ProviderResult` rather than a raw SDK response.
This module owns the sanitization contract: only whitelisted metadata fields
are allowed through, and only when they are JSON-serializable.

The contract prevents accidental leaks of:
  - HTTP headers (Authorization, request IDs)
  - signed URLs
  - raw exception text
  - provider-internal objects (langchain AIMessage, OpenAI objects, etc.)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Whitelists
# ---------------------------------------------------------------------------

# Anything else in response_metadata is dropped before reaching the envelope.
ALLOWED_RESPONSE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "finish_reason",
        "system_fingerprint",
    }
)

# Only numeric token stats; never leak request IDs, headers, or signed URLs.
ALLOWED_USAGE_KEYS: frozenset[str] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
)


# ---------------------------------------------------------------------------
# JSON-safety coercion
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value to JSON-serializable form.

    `headers`/etc. are expected to be dropped (not str()-coerced) by the
    caller before this helper sees them. Anything that is not a primitive,
    list, tuple, or dict falls back to `None` to avoid leaking provider
    objects verbatim.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def sanitize_response_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only whitelisted response_metadata keys; coerce values safely."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key in ALLOWED_RESPONSE_METADATA_KEYS:
        if key in raw:
            out[key] = _json_safe(raw[key])
    return out


def sanitize_usage_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only whitelisted numeric usage keys."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key in ALLOWED_USAGE_KEYS:
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class ProviderResult(BaseModel):
    """Sanitized raw provider response.

    This is the only type that crosses the provider -> server boundary.
    Construct it via `build_provider_result(...)` to ensure sanitization
    runs, or instantiate directly only when the caller has already
    sanitized the inputs.
    """

    text: str
    model: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


def build_provider_result(
    *,
    text: str,
    model: str | None = None,
    response_metadata: dict[str, Any] | None = None,
    usage_metadata: dict[str, Any] | None = None,
) -> ProviderResult:
    """Build a sanitized `ProviderResult`.

    Always run inputs through `sanitize_response_metadata` and
    `sanitize_usage_metadata` so accidental leaks (headers, request IDs,
    signed URLs) cannot reach the envelope's `raw_model_output` field.
    """
    return ProviderResult(
        text=text,
        model=model,
        response_metadata=sanitize_response_metadata(response_metadata),
        usage_metadata=sanitize_usage_metadata(usage_metadata),
    )
