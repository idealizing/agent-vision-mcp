"""Response envelope for agent-vision-mcp tools.

Every MCP tool returns a JSON string produced by `make_envelope` (success) or
`make_error_envelope` (failure). The wire format is:

    {
      "schema_version": "1.0",
      "ok": true,
      "tool": "<tool_name>",
      "task": "<task_type>" | null,
      "model": "<configured_model_id>" | null,
      "source": SourceMeta | null,         # single-image tools
      "sources": [SourceMeta, ...],         # vision_compare only
      "result": <per-tool result model>,
      "warnings": [],                        # always a list
      "raw_model_output": { ... } | null,   # opt-in via include_raw
      "error": null | ErrorPayload
    }

Phase 1 (this file): tools return JSON strings. Phase 2 may switch to returning
Pydantic models directly, after validating with MCP clients that the wire
format is preserved.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Confidence = Literal["high", "medium", "low"]
DifferenceType = Literal["added", "removed", "modified", "moved"]
SourceType = Literal["url", "file", "data_url", "base64"]

# Normalized bounding box: (x, y, width, height), all in [0.0, 1.0]. Range
# validation is intentionally out of scope for this commit (a follow-up).
BBoxNormalized = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Envelope payload models
# ---------------------------------------------------------------------------


class SourceMeta(BaseModel):
    """Image source metadata. `source_ref` is opt-in and redacted by default.

    When VISION_URL_MODE=passthrough, `mime_type`/`width`/`height`/`size_bytes`
    may be `None` because the URL is not fetched locally.
    """

    type: SourceType
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    source_ref: str | None = None


class ErrorPayload(BaseModel):
    code: str
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class SuccessEnvelope(BaseModel):
    """Wire shape for a successful tool call."""

    schema_version: str = "1.0"
    ok: Literal[True] = True
    tool: str
    task: str | None = None
    # Configured model identifier. The actual model name returned by the API
    # is not captured (out of scope for this refactor).
    model: str | None = None
    # single-image tools set `source`; vision_compare uses `sources`; both keys
    # are always present so consumers can rely on a stable key set.
    source: SourceMeta | None = None
    sources: list[SourceMeta] = Field(default_factory=list)
    result: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    raw_model_output: dict[str, Any] | None = None
    error: None = None


class FailureEnvelope(BaseModel):
    """Wire shape for a failed tool call."""

    schema_version: str = "1.0"
    ok: Literal[False] = False
    tool: str
    task: str | None = None
    model: str | None = None
    source: None = None
    sources: list[SourceMeta] = Field(default_factory=list)
    result: None = None
    warnings: list[str] = Field(default_factory=list)
    raw_model_output: None = None
    error: ErrorPayload


# ---------------------------------------------------------------------------
# Discriminated union + runtime validator
# ---------------------------------------------------------------------------

ResponseEnvelope = Annotated[
    Union[SuccessEnvelope, FailureEnvelope],
    Field(discriminator="ok"),
]
ResponseEnvelopeAdapter: TypeAdapter = TypeAdapter(ResponseEnvelope)


# ---------------------------------------------------------------------------
# Per-tool result models
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    order: int
    # OCR block kind: kept open ("text", "heading", "table_row", ...). Not
    # pinned to a Literal because block kinds are open-ended across providers.
    type: str
    text: str
    confidence: Confidence = "medium"
    bbox_normalized: BBoxNormalized | None = None


class VisionAnalyzeResult(BaseModel):
    summary: str
    observations: list[dict[str, Any]] = Field(default_factory=list)
    inferences: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    suggested_followups: list[dict[str, Any]] = Field(default_factory=list)


class VisionExtractTextResult(BaseModel):
    text: str
    blocks: list[TextBlock]
    layout_preserved: bool = True
    unclear_segments: list[str] = Field(default_factory=list)


class CompareDifference(BaseModel):
    type: DifferenceType
    area: str | None = None
    description: str
    confidence: Confidence = "medium"


class VisionCompareResult(BaseModel):
    summary: str
    # Empty until a real parser exists; we never fabricate partial structure.
    differences: list[CompareDifference] = Field(default_factory=list)
    same_elements: list[str] = Field(default_factory=list)


class CropRegion(BaseModel):
    x: float
    y: float
    width: float
    height: float


class VisionCropAnalyzeResult(BaseModel):
    crop: CropRegion
    summary: str
    observations: list[dict[str, Any]] = Field(default_factory=list)


class VisionInspectResult(BaseModel):
    width: int
    height: int
    format: str
    mime_type: str
    mode: str
    size_bytes: int
    has_transparency: bool
    source_type: SourceType


class VisionCapabilitiesResult(BaseModel):
    server: str
    version: str
    vlm_provider: dict[str, Any]
    ocr_provider: dict[str, Any] | None = None
    ocr_enabled: bool
    tools: dict[str, Any]
    supports: dict[str, Any]
    limits: dict[str, Any]
    task_types: list[str]


# Tool name -> result model. Used by make_envelope to validate per-tool results.
RESULT_MODEL_BY_TOOL: dict[str, type[BaseModel]] = {
    "vision_analyze": VisionAnalyzeResult,
    "vision_extract_text": VisionExtractTextResult,
    "vision_compare": VisionCompareResult,
    "vision_crop_analyze": VisionCropAnalyzeResult,
    "vision_inspect": VisionInspectResult,
    "vision_capabilities": VisionCapabilitiesResult,
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_envelope(
    *,
    tool: str,
    result: BaseModel | dict[str, Any],
    task: str | None = None,
    model: str | None = None,
    source: SourceMeta | None = None,
    sources: list[SourceMeta] | None = None,
    warnings: list[str] | None = None,
    raw_model_output: dict[str, Any] | None = None,
) -> str:
    """Build a success-envelope JSON string.

    `tool` must be a known tool name (member of RESULT_MODEL_BY_TOOL); unknown
    names raise ValueError because the tool set is closed and a typo should
    fail loudly rather than silently bypass per-tool schema validation.

    `result` is validated against the per-tool result model when given as a
    dict; when given as a BaseModel, the model type is checked (a wrong-type
    BaseModel is dumped and re-validated against the right model).
    """
    if tool not in RESULT_MODEL_BY_TOOL:
        raise ValueError(f"Unknown tool for response envelope: {tool!r}")
    result_cls = RESULT_MODEL_BY_TOOL[tool]
    if not isinstance(result, result_cls):
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        result = result_cls.model_validate(result)  # raises if shape is wrong
    result_payload = (
        result.model_dump(mode="json")
        if isinstance(result, BaseModel)
        else result
    )
    return SuccessEnvelope(
        tool=tool,
        task=task,
        model=model,
        source=source,
        sources=list(sources or []),
        result=result_payload,
        warnings=list(warnings or []),
        raw_model_output=raw_model_output,
    ).model_dump_json(indent=2, exclude_none=False)


def make_error_envelope(
    *,
    tool: str,
    code: str,
    message: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    task: str | None = None,
    model: str | None = None,
) -> str:
    """Build a failure-envelope JSON string."""
    return FailureEnvelope(
        tool=tool,
        task=task,
        model=model,
        error=ErrorPayload(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        ),
    ).model_dump_json(indent=2, exclude_none=False)
