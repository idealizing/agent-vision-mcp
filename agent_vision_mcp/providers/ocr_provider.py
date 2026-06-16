"""OCR provider - supports DeepSeek-OCR and other OCR models via OpenAI-compatible API"""

import time
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agent_vision_mcp.config import Settings
from agent_vision_mcp.errors import ProviderError, TimeoutError
from agent_vision_mcp.providers.base import BaseVisionProvider
from agent_vision_mcp.provider_result import ProviderResult, build_provider_result


class OCRProvider(BaseVisionProvider):
    """
    OCR provider using OpenAI-compatible multimodal API.

    Supports DeepSeek-OCR and similar models that use the standard
    OpenAI chat completions format with image_url content parts.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.ocr_model_id
        self.base_url = settings.ocr_base_url
        self.api_key = settings.ocr_api_key

        self.vlm = ChatOpenAI(
            model=self.model_id,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=settings.vision_timeout,
            max_retries=0,
        )

    def analyze(
        self,
        images: list[Dict[str, Any]],
        prompt: str,
        detail: str = "auto",
        max_tokens: Optional[int] = None,
    ) -> ProviderResult:
        """
        Extract text from images using OCR.

        Uses standard OpenAI multimodal format. Returns a sanitized
        ProviderResult — only whitelisted metadata keys cross this boundary.
        """
        if not images:
            raise ProviderError("No image provided", retryable=False)

        content = [{"type": "text", "text": prompt}]
        for image in images:
            image_url = {"url": image.get("url") or image["data_url"]}
            if self.settings.vision_supports_image_detail and detail != "auto":
                image_url["detail"] = detail
            content.append({"type": "image_url", "image_url": image_url})

        message = HumanMessage(content=content)

        last_error = None
        for attempt in range(self.settings.vision_max_retries):
            try:
                response = self.vlm.invoke(
                    [message],
                    max_tokens=max_tokens or 4096,
                )
                return build_provider_result(
                    text=response.content or "",
                    model=self.model_id,
                    response_metadata=getattr(response, "response_metadata", None),
                    usage_metadata=getattr(response, "usage_metadata", None),
                )

            except Exception as e:
                error_str = str(e).lower()

                if "timeout" in error_str or "timed out" in error_str:
                    last_error = e
                    if attempt < self.settings.vision_max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise TimeoutError(self.settings.vision_timeout)

                if "rate limit" in error_str or "429" in error_str:
                    last_error = e
                    if attempt < self.settings.vision_max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise ProviderError(
                        message="Rate limit exceeded",
                        retryable=True,
                        details={"error": str(e)},
                    )

                if any(code in error_str for code in ["500", "502", "503", "504"]):
                    last_error = e
                    if attempt < self.settings.vision_max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue

                raise ProviderError(
                    message=f"OCR API error: {str(e)[:200]}",
                    retryable=False,
                    details={"error": str(e)},
                )

        raise ProviderError(
            message=f"OCR request failed after retries: {str(last_error)}",
            retryable=True,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider": "ocr",
            "type": "ocr",
            "model": self.model_id,
            "base_url": self.base_url,
            "supports": {
                "url": True,
                "local_file": True,
                "base64": True,
                "data_url": True,
                "multi_image": False,
                "ocr": True,
            },
            "limits": {
                "timeout": self.settings.vision_timeout,
                "max_retries": self.settings.vision_max_retries,
            },
        }
