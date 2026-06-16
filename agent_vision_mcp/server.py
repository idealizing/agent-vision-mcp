"""agent-vision-mcp - MCP Server with vision capabilities"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from PIL import Image

from agent_vision_mcp.config import Settings
from agent_vision_mcp.errors import handle_exception, InvalidInputError, VisionMCPError, SecurityError
from agent_vision_mcp.image.input import NormalizedImage, normalize_image_source, detect_source_type
from agent_vision_mcp.image.security import validate_image_source
from agent_vision_mcp.providers.openai_compatible import OpenAICompatibleVisionProvider
from agent_vision_mcp.providers.ocr_provider import OCRProvider
from agent_vision_mcp.provider_result import ProviderResult
from agent_vision_mcp.responses import (
    CropRegion,
    SourceMeta,
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

# Initialize settings and providers
settings = Settings.from_env()
vlm_provider = OpenAICompatibleVisionProvider(settings)
ocr_provider = OCRProvider(settings) if settings.ocr_enabled else None

# Create MCP server
mcp = FastMCP("agent-vision-mcp")

# ==================== Task Prompt Templates ====================

TASK_PROMPTS = {
    "general": "请分析这张图片，区分：可直接观察到的事实 / 合理推断 / 无法确认的内容。",
    "qa": "请回答关于这张图片的问题。仅基于图片中可直接观察到的内容回答，不确定的部分请标注。",
    "ui": "请分析这个UI界面，重点关注：1.布局与组件 2.交互状态 3.错误提示 4.可访问性问题 5.文案问题。区分事实与推断。",
    "chart": "请分析这个图表，重点关注：1.坐标轴与单位 2.数据趋势 3.异常点 4.近似数值。区分可直接读出的数据和推断。",
    "document": "请分析这份文档，重点关注：1.标题层级 2.关键文字内容 3.表格数据 4.印章/签名 5.关键字段。按阅读顺序提取。",
    "object": "请识别图片中的主要对象，为每个对象提供：名称、位置描述、置信度（高/中/低）。",
    "screenshot": "请分析这个截图，重点关注：1.应用程序类型 2.当前状态 3.可见文字 4.错误信息 5.关键UI元素。区分事实与推断。",
    "code_screenshot": "请分析这个代码截图，重点关注：1.文件名 2.行号 3.代码内容 4.错误信息 5.高亮/标记。精确读取所有可见文字。",
    "compare": "请比较这些图片的异同，重点关注：1.新增内容 2.删除内容 3.位置变化 4.颜色/数值变化。每项标注置信度。",
}
TASK_TYPES = set(TASK_PROMPTS) - {"compare"}
DETAIL_LEVELS = {"auto", "low", "high"}
COMPARE_FOCUS_TYPES = {"general", "layout", "text", "colors", "changes"}
OCR_LANGUAGES = {"auto", "chinese", "english", "japanese", "korean"}


def build_prompt(user_prompt: str, task: str) -> str:
    """Build effective prompt by combining task template and user prompt."""
    task_instruction = TASK_PROMPTS.get(task, TASK_PROMPTS["general"])
    if user_prompt and user_prompt != TASK_PROMPTS.get("general", ""):
        return f"{task_instruction}\n\n用户具体问题：{user_prompt}"
    return task_instruction


def load_and_validate_image(image_source: str, require_bytes: bool = False) -> tuple:
    """Common security validation and normalization. Returns (normalized, source_type)."""
    source_type = detect_source_type(image_source)

    if source_type == "url":
        validate_image_source(
            image_source, source_type,
            block_private_ips=settings.vision_block_private_ips,
        )
    elif source_type == "file":
        if not settings.vision_allow_local_files:
            raise SecurityError("Local file access is disabled")
        validate_image_source(
            image_source, source_type,
            allowed_paths=settings.vision_allowed_paths,
        )

    url_mode = settings.vision_url_mode
    if source_type == "url" and url_mode == "auto":
        url_mode = "download" if require_bytes else "passthrough"
    if require_bytes and url_mode == "passthrough":
        url_mode = "download"

    normalized = normalize_image_source(
        image_source,
        max_size_mb=settings.vision_max_image_size_mb,
        max_pixels=settings.vision_max_image_pixels,
        url_mode=url_mode,
        block_private_ips=settings.vision_block_private_ips,
    )
    return normalized, source_type


def validate_choice(value: str, allowed: set[str], name: str) -> None:
    """Validate a string option exposed by an MCP tool."""
    if value not in allowed:
        raise VisionMCPError(
            f"Invalid {name}: {value}. Allowed: {', '.join(sorted(allowed))}",
            code="INVALID_INPUT",
        )


def validate_crop(x: float, y: float, width: float, height: float) -> None:
    """Validate normalized crop coordinates."""
    values = (x, y, width, height)
    if not all(0 <= value <= 1 for value in values):
        raise VisionMCPError("Crop coordinates must be between 0.0 and 1.0", code="INVALID_INPUT")
    if width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise VisionMCPError("Crop region must have positive size and fit inside the image", code="INVALID_INPUT")


def crop_image(
    data_url: str,
    x: float, y: float, width: float, height: float,
) -> str:
    """
    Crop image by normalized coordinates (0.0-1.0).
    Returns new data URL of cropped region.
    """
    # Extract base64 data
    header, b64_data = data_url.split(",", 1)
    image_data = base64.b64decode(b64_data)

    img = Image.open(io.BytesIO(image_data))
    img_width, img_height = img.size

    # Convert normalized coords to pixels
    left = int(x * img_width)
    top = int(y * img_height)
    right = int((x + width) * img_width)
    bottom = int((y + height) * img_height)

    # Clamp to image bounds
    left = max(0, min(left, img_width))
    top = max(0, min(top, img_height))
    right = max(left + 1, min(right, img_width))
    bottom = max(top + 1, min(bottom, img_height))

    cropped = img.crop((left, top, right, bottom))

    # Convert back to data URL
    buffer = io.BytesIO()
    fmt = img.format or "PNG"
    if fmt == "JPEG":
        cropped.save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        cropped.save(buffer, format="PNG")
        mime = "image/png"

    cropped_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{cropped_b64}"


def get_image_metadata(image_source: str) -> dict:
    """Get image metadata without calling VLM."""
    normalized, source_type = load_and_validate_image(image_source, require_bytes=True)

    # Decode image for metadata
    header, b64_data = normalized.data_url.split(",", 1)
    image_data = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(image_data))

    mime_match = header.split(":")[1].split(";")[0] if ":" in header else "image/png"

    return {
        "width": img.width,
        "height": img.height,
        "format": img.format or "UNKNOWN",
        "mime_type": mime_match,
        "mode": img.mode,
        "size_bytes": len(image_data),
        "has_transparency": img.mode in ("RGBA", "PA", "LA"),
        "source_type": source_type,
    }


# ==================== Source metadata redaction ====================


def _build_source_ref(normalized: NormalizedImage) -> str | None:
    """Return a redacted `source_ref` string for the envelope, or None.

    - url: netloc + path, with query string stripped (no signed tokens leak).
    - file: basename only (no full path).
    - data_url / base64: None (no safe representation).
    """
    if normalized.source_type == "url":
        parsed = urlparse(normalized.original_source)
        if not parsed.netloc:
            return None
        return f"{parsed.netloc}{parsed.path}"
    if normalized.source_type == "file":
        return Path(normalized.original_source).name or None
    return None


def _build_source_meta(
    normalized: NormalizedImage, *, include_source_ref: bool = False
) -> SourceMeta:
    """Build a SourceMeta from a NormalizedImage.

    The passthrough URL mode produces mime_type="" with no width/height/size_bytes;
    those are surfaced as None so the envelope is honest about the gap.
    """
    mime_type = normalized.mime_type or None
    return SourceMeta(
        type=normalized.source_type,  # type: ignore[arg-type]
        mime_type=mime_type,
        width=normalized.width,
        height=normalized.height,
        size_bytes=normalized.size_bytes,
        source_ref=_build_source_ref(normalized) if include_source_ref else None,
    )


def _provider_result_to_envelope_payload(
    pr: ProviderResult, *, include_raw: bool
) -> dict | None:
    """Build the envelope's `raw_model_output` payload, or None when disabled.

    `pr.model_dump(mode="json")` produces a sanitized dict (already passed
    through the whitelist when the provider used `build_provider_result`).
    """
    if not include_raw:
        return None
    return pr.model_dump(mode="json")


# ==================== MCP Tools ====================


@mcp.tool(
    description=(
        "Analyze an image using a vision-language model. Returns a unified JSON "
        "envelope wrapping summary, observations, uncertainties, and suggested "
        "follow-ups (see README 'Response format' for the full schema).\n\n"
        "Supports URL, local file path, data URL, and Base64 input.\n\n"
        "Task types guide the model:\n"
        "- general: General analysis (default)\n"
        "- qa: Answer a specific question about the image\n"
        "- ui: Analyze UI/layout/interactions/accessibility\n"
        "- chart: Analyze charts/graphs/data\n"
        "- document: OCR and document structure\n"
        "- object: Identify and locate objects\n"
        "- screenshot: Analyze application screenshots\n"
        "- code_screenshot: Read code from screenshots"
    )
)
def vision_analyze(
    image_source: Annotated[str, "Image: URL, file path, data URL, or base64"],
    prompt: Annotated[str, "Your question or instruction about the image"] = "请描述这张图片的内容。",
    task: Annotated[str, "Task type: general/qa/ui/chart/document/object/screenshot/code_screenshot"] = "general",
    detail: Annotated[str, "Detail level: auto/low/high"] = "auto",
    include_raw: Annotated[bool, "Include the sanitized provider response in `raw_model_output`"] = False,
    include_source_ref: Annotated[bool, "Include a redacted `source_ref` (URL netloc+path or file basename) in the source block"] = False,
) -> str:
    """Analyze an image with structured output."""
    tool = "vision_analyze"
    model = vlm_provider.model_id
    try:
        validate_choice(task, TASK_TYPES, "task")
        validate_choice(detail, DETAIL_LEVELS, "detail")
        normalized, _ = load_and_validate_image(image_source)
        effective_prompt = build_prompt(prompt, task)

        provider_result = vlm_provider.analyze(
            images=[{"data_url": normalized.data_url}],
            prompt=effective_prompt,
            detail=detail or settings.vision_default_detail,
        )

        analyze_result = VisionAnalyzeResult(
            summary=provider_result.text,
            suggested_followups=[
                {
                    "tool": "vision_crop_analyze",
                    "hint": "Zoom into specific regions for more detail",
                }
            ],
        )
        return make_envelope(
            tool=tool,
            result=analyze_result,
            task=task,
            model=model,
            source=_build_source_meta(normalized, include_source_ref=include_source_ref),
            raw_model_output=_provider_result_to_envelope_payload(
                provider_result, include_raw=include_raw
            ),
        )
    except VisionMCPError as e:
        return e.to_envelope(tool=tool, task=task, model=model)
    except Exception as e:
        return handle_exception(e, tool=tool, task=task, model=model)


@mcp.tool(
    description=(
        "Inspect image metadata (dimensions, format, size, mode) without calling VLM. "
        "Use this before detailed analysis to understand the image dimensions and plan crop coordinates."
    )
)
def vision_inspect(
    image_source: Annotated[str, "Image: URL, file path, data URL, or base64"],
    include_source_ref: Annotated[bool, "Include a redacted `source_ref` (URL netloc+path or file basename) in the source block"] = False,
) -> str:
    """Return image metadata without calling VLM."""
    tool = "vision_inspect"
    try:
        normalized, _ = load_and_validate_image(image_source, require_bytes=True)
        metadata = get_image_metadata(image_source)
        inspect_result = VisionInspectResult(**metadata)
        return make_envelope(
            tool=tool,
            result=inspect_result,
            source=_build_source_meta(normalized, include_source_ref=include_source_ref),
        )
    except VisionMCPError as e:
        return e.to_envelope(tool=tool)
    except Exception as e:
        return handle_exception(e, tool=tool)


@mcp.tool(
    description=(
        "Crop a region of an image and analyze it with VLM. This is the most powerful tool "
        "for inspecting small text, UI elements, chart data, or error messages.\n\n"
        "Coordinates are NORMALIZED (0.0 to 1.0), where (0,0) is top-left and (1,1) is bottom-right.\n\n"
        "Workflow: Use vision_inspect first to get dimensions, then vision_analyze for overview, "
        "then vision_crop_analyze to zoom into specific regions of interest."
    )
)
def vision_crop_analyze(
    image_source: Annotated[str, "Image: URL, file path, data URL, or base64"],
    x: Annotated[float, "Left edge of crop region (0.0-1.0, normalized)"],
    y: Annotated[float, "Top edge of crop region (0.0-1.0, normalized)"],
    width: Annotated[float, "Width of crop region (0.0-1.0, normalized)"],
    height: Annotated[float, "Height of crop region (0.0-1.0, normalized)"],
    prompt: Annotated[str, "What to look for in the cropped region"] = "请详细描述这个区域的内容",
    task: Annotated[str, "Task type: general/qa/ui/chart/document/object/screenshot/code_screenshot"] = "general",
    include_raw: Annotated[bool, "Include the sanitized provider response in `raw_model_output`"] = False,
    include_source_ref: Annotated[bool, "Include a redacted `source_ref` (URL netloc+path or file basename) in the source block"] = False,
) -> str:
    """Crop a region and analyze it."""
    tool = "vision_crop_analyze"
    model = vlm_provider.model_id
    try:
        validate_crop(x, y, width, height)
        validate_choice(task, TASK_TYPES, "task")
        normalized, _ = load_and_validate_image(image_source, require_bytes=True)

        # Crop the image
        cropped_data_url = crop_image(normalized.data_url, x, y, width, height)

        effective_prompt = build_prompt(prompt, task)
        effective_prompt += f"\n\n[这是原图区域 x={x:.2f}, y={y:.2f}, w={width:.2f}, h={height:.2f} 的裁剪放大图]"

        provider_result = vlm_provider.analyze(
            images=[{"data_url": cropped_data_url}],
            prompt=effective_prompt,
            detail="high",
        )

        crop_result = VisionCropAnalyzeResult(
            crop=CropRegion(x=x, y=y, width=width, height=height),
            summary=provider_result.text,
        )
        return make_envelope(
            tool=tool,
            result=crop_result,
            task=task,
            model=model,
            source=_build_source_meta(normalized, include_source_ref=include_source_ref),
            raw_model_output=_provider_result_to_envelope_payload(
                provider_result, include_raw=include_raw
            ),
        )
    except VisionMCPError as e:
        return e.to_envelope(tool=tool, task=task, model=model)
    except Exception as e:
        return handle_exception(e, tool=tool, task=task, model=model)


@mcp.tool(
    description=(
        "Extract visible text from an image using OCR. Returns structured text organized by reading order.\n\n"
        "Use this for: screenshots with text, scanned documents, receipts, tables, forms, "
        "Chinese/English OCR, and any text-heavy images.\n\n"
        "Uses a configured dedicated OCR model when enabled. "
        "If the dedicated OCR model is unavailable, automatically falls back to the VLM provider."
    )
)
def vision_extract_text(
    image_source: Annotated[str, "Image: URL, file path, data URL, or base64"],
    language: Annotated[str, "Expected language hint: chinese, english, japanese, korean, or auto"] = "auto",
    preserve_layout: Annotated[bool, "Try to preserve the original layout in the output"] = True,
    include_raw: Annotated[bool, "Include the sanitized provider response in `raw_model_output`"] = False,
    include_source_ref: Annotated[bool, "Include a redacted `source_ref` (URL netloc+path or file basename) in the source block"] = False,
) -> str:
    """Extract text from image using OCR model (with VLM fallback)."""
    tool = "vision_extract_text"
    model = vlm_provider.model_id
    try:
        validate_choice(language, OCR_LANGUAGES, "language")
        normalized, _ = load_and_validate_image(image_source, require_bytes=True)

        layout_hint = "请保持原文的阅读顺序和布局。" if preserve_layout else ""
        language_hint = "" if language == "auto" else f" Expected language: {language}."

        # Use short English prompt for OCR models to avoid hallucination loops
        ocr_english_prompt = (
            f"<image>\nExtract all visible text from this image in reading order. "
            f"{layout_hint}{language_hint} For tables, use Markdown. Mark unclear as [unclear]. "
            f"Do not invent text."
        )

        def _make_extract_envelope(
            text: str, *, warnings: list[str] | None = None
        ) -> str:
            extract_result = VisionExtractTextResult(
                text=text,
                blocks=[TextBlock(order=1, type="text", text=text)],
            )
            return make_envelope(
                tool=tool,
                result=extract_result,
                model=model,
                source=_build_source_meta(normalized, include_source_ref=include_source_ref),
                warnings=warnings,
            )

        # Try dedicated OCR provider first
        if ocr_provider:
            try:
                ocr_result = ocr_provider.analyze(
                    images=[{"data_url": normalized.data_url}],
                    prompt=ocr_english_prompt,
                    detail="high",
                )
                if ocr_result.text and ocr_result.text.strip():
                    return _make_extract_envelope(ocr_result.text)
                # Empty result - fall through to VLM
            except Exception:
                # Fall back to VLM with Chinese prompt
                vlm_result = vlm_provider.analyze(
                    images=[{"data_url": normalized.data_url}],
                    prompt=(
                        f"请提取图片中的所有可见文字。{layout_hint}\n"
                        f"预期语言：{language}。\n"
                        "如果是表格，请输出 Markdown 表格。\n"
                        "如果有无法识别的文字，用 [unclear] 标注。\n"
                        "不要编造图片中不存在的文字。"
                    ),
                    detail="high",
                )
                return _make_extract_envelope(
                    vlm_result.text,
                    warnings=["Dedicated OCR provider failed; used VLM fallback."],
                )

        # No OCR provider or empty result - use VLM
        vlm_result = vlm_provider.analyze(
            images=[{"data_url": normalized.data_url}],
            prompt=(
                f"请提取图片中的所有可见文字。{layout_hint}\n"
                f"预期语言：{language}。\n"
                "如果是表格，请输出 Markdown 表格。\n"
                "如果有无法识别的文字，用 [unclear] 标注。\n"
                "不要编造图片中不存在的文字。"
            ),
            detail="high",
        )
        return _make_extract_envelope(vlm_result.text)
    except VisionMCPError as e:
        return e.to_envelope(tool=tool, model=model)
    except Exception as e:
        return handle_exception(e, tool=tool, model=model)


@mcp.tool(
    description=(
        "Compare two or more images and identify differences. Use for:\n"
        "- UI regression testing (before/after screenshots)\n"
        "- Design vs implementation comparison\n"
        "- Bug screenshot comparison\n"
        "- Version diff of documents\n\n"
        "Returns structured differences with confidence levels."
    )
)
def vision_compare(
    image_sources: Annotated[list[str], "2-4 image sources to compare (URLs, file paths, etc.)"],
    prompt: Annotated[str, "What to compare between the images"] = "请比较这些图片的异同",
    focus: Annotated[str, "Focus area: general/layout/text/colors/changes"] = "general",
    include_raw: Annotated[bool, "Include the sanitized provider response in `raw_model_output`"] = False,
    include_source_ref: Annotated[bool, "Include a redacted `source_ref` (URL netloc+path or file basename) in the source block for each input"] = False,
) -> str:
    """Compare multiple images and find differences."""
    tool = "vision_compare"
    model = vlm_provider.model_id
    try:
        validate_choice(focus, COMPARE_FOCUS_TYPES, "focus")
        if len(image_sources) < 2:
            raise InvalidInputError("At least 2 images are required for comparison")
        max_compare_images = min(4, settings.vision_max_batch_images)
        if len(image_sources) > max_compare_images:
            raise InvalidInputError(f"Maximum {max_compare_images} images are allowed for comparison")

        # Normalize all images
        normalized_images = []
        sources: list[SourceMeta] = []
        for src in image_sources:
            normalized, _ = load_and_validate_image(src)
            normalized_images.append({"data_url": normalized.data_url})
            sources.append(
                _build_source_meta(normalized, include_source_ref=include_source_ref)
            )

        focus_hints = {
            "general": "比较所有方面的异同",
            "layout": "重点关注布局和位置变化",
            "text": "重点关注文字内容变化",
            "colors": "重点关注颜色和样式变化",
            "changes": "重点关注新增、删除和修改的内容",
        }
        focus_text = focus_hints.get(focus, focus_hints["general"])

        compare_prompt = (
            f"请逐项比较这些图片的异同。{focus_text}。\n\n"
            "对每个差异，请标注：\n"
            "1. 变化类型：新增/删除/修改/位置变化\n"
            "2. 置信度：高/中/低\n"
            "3. 具体描述\n\n"
            f"用户关注：{prompt}"
        )

        provider_result = vlm_provider.analyze(
            images=normalized_images,
            prompt=compare_prompt,
            detail="high",
        )

        compare_result = VisionCompareResult(summary=provider_result.text)
        return make_envelope(
            tool=tool,
            result=compare_result,
            model=model,
            sources=sources,
            raw_model_output=_provider_result_to_envelope_payload(
                provider_result, include_raw=include_raw
            ),
        )
    except VisionMCPError as e:
        return e.to_envelope(tool=tool, model=model)
    except Exception as e:
        return handle_exception(e, tool=tool, model=model)


@mcp.tool(
    description=(
        "Return current agent-vision-mcp server capabilities, supported models, and limits. "
        "Call this to discover what the server can do before using other tools."
    )
)
def vision_capabilities() -> str:
    """Return server capabilities."""
    tool = "vision_capabilities"
    try:
        capabilities = {
            "server": "agent-vision-mcp",
            "version": "0.0.2",
            "vlm_provider": vlm_provider.get_capabilities(),
            "ocr_provider": ocr_provider.get_capabilities() if ocr_provider else None,
            "ocr_enabled": settings.ocr_enabled,
            "tools": {
                "vision_analyze": "Analyze image with structured output and task-specific prompts",
                "vision_inspect": "Get image metadata without VLM",
                "vision_crop_analyze": "Crop and analyze specific region (normalized coordinates 0-1)",
                "vision_extract_text": "OCR text extraction (dedicated OCR model if configured)",
                "vision_compare": "Compare 2-4 images for differences",
                "vision_capabilities": "This tool - show server capabilities",
            },
            "supports": {
                "url": True,
                "local_file": settings.vision_allow_local_files,
                "base64": True,
                "data_url": True,
                "crop": True,
                "ocr": settings.ocr_enabled,
                "multi_image_compare": True,
            },
            "limits": {
                "max_image_size_mb": settings.vision_max_image_size_mb,
                "max_image_pixels": settings.vision_max_image_pixels,
                "max_batch_images": settings.vision_max_batch_images,
                "max_compare_images": 4,
                "timeout": settings.vision_timeout,
                "url_mode": settings.vision_url_mode,
            },
            "task_types": list(TASK_PROMPTS.keys()),
        }
        return make_envelope(
            tool=tool,
            result=VisionCapabilitiesResult(**capabilities),
        )
    except VisionMCPError as e:
        return e.to_envelope(tool=tool)
    except Exception as e:
        return handle_exception(e, tool=tool)


def run_server():
    """Run the MCP server"""
    transport = os.getenv("VISION_TRANSPORT", "stdio")
    print(f"Starting agent-vision-mcp server (transport: {transport})", file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    run_server()
