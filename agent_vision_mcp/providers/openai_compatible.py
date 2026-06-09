"""OpenAI-compatible VLM provider using langchain"""

import time
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agent_vision_mcp.config import Settings
from agent_vision_mcp.errors import ProviderError, TimeoutError
from agent_vision_mcp.providers.base import BaseVisionProvider


class OpenAICompatibleVisionProvider(BaseVisionProvider):
    """VLM provider that works with OpenAI-compatible APIs using langchain"""

    def __init__(self, settings: Settings, provider_type: str = "vlm"):
        self.settings = settings
        self.provider_type = provider_type

        # Select config based on provider type
        if provider_type == "ocr":
            model_id = settings.ocr_model_id
            base_url = settings.ocr_base_url
            api_key = settings.ocr_api_key
        else:
            model_id = settings.vision_model_id
            base_url = settings.vision_base_url
            api_key = settings.vision_api_key

        self.model_id = model_id
        self.base_url = base_url

        self.vlm = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            timeout=settings.vision_timeout,
            max_retries=0,
        )

    def analyze(
        self,
        images: list[Dict[str, Any]],
        prompt: str,
        detail: str = "auto",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Analyze images with a text prompt using langchain + OpenAI-compatible API.
        """
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
                    max_tokens=max_tokens or 2048,
                )
                return response.content or ""

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
                    continue

                raise ProviderError(
                    message=f"API error: {str(e)}",
                    retryable=False,
                    details={"error": str(e)},
                )

        raise ProviderError(
            message=f"Request failed after {self.settings.vision_max_retries} attempts: {str(last_error)}",
            retryable=True,
            details={"last_error": str(last_error)},
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities"""
        return {
            "provider": "openai-compatible",
            "type": self.provider_type,
            "model": self.model_id,
            "base_url": self.base_url,
            "supports": {
                "url": True,
                "local_file": True,
                "base64": True,
                "data_url": True,
                "multi_image": True,
                "image_detail": self.settings.vision_supports_image_detail,
            },
            "limits": {
                "timeout": self.settings.vision_timeout,
                "max_retries": self.settings.vision_max_retries,
            },
        }
