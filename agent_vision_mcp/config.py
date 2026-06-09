"""Configuration management for agent-vision-mcp"""

import os
from pathlib import Path
from typing import List, Optional


class Settings:
    """Vision-mcp settings from environment variables"""

    # VLM Provider
    vision_api_key: str = ""
    vision_base_url: str = "https://api.example.com/v1"
    vision_model_id: str = "glm-4v-flash"

    # OCR Provider
    ocr_api_key: str = ""
    ocr_base_url: str = ""
    ocr_model_id: str = "DeepSeek-OCR"

    # Runtime
    vision_timeout: int = 60
    vision_max_retries: int = 3
    vision_default_detail: str = "auto"
    vision_supports_image_detail: bool = False
    vision_url_mode: str = "auto"

    # Input limits
    vision_max_image_size_mb: int = 10
    vision_max_image_pixels: int = 40_000_000
    vision_max_batch_images: int = 10

    # Security
    vision_allow_local_files: bool = True
    vision_allowed_paths: List[str] = ["/data", "/tmp"]
    vision_block_private_ips: bool = True

    # Transport
    vision_transport: str = "stdio"

    # Optional dedicated OCR provider
    dedicated_ocr_enabled: bool = False

    @property
    def ocr_enabled(self) -> bool:
        return bool(self.dedicated_ocr_enabled and self.ocr_api_key and self.ocr_base_url)

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Settings":
        """Load settings from environment variables"""
        from dotenv import load_dotenv

        if env_file is None:
            env_file = Path(__file__).parent.parent / ".env"
        load_dotenv(env_file)

        settings = cls()

        # VLM Provider
        settings.vision_api_key = os.getenv("VISION_API_KEY", "")
        settings.vision_base_url = os.getenv("VISION_BASE_URL", "https://api.example.com/v1")
        settings.vision_model_id = os.getenv("VISION_MODEL_ID", "glm-4v-flash")

        # OCR Provider - defaults to VLM credentials if not specified
        settings.ocr_api_key = os.getenv("OCR_API_KEY", settings.vision_api_key)
        settings.ocr_base_url = os.getenv("OCR_BASE_URL", settings.vision_base_url)
        settings.ocr_model_id = os.getenv("OCR_MODEL_ID", "DeepSeek-OCR")

        # Runtime
        settings.vision_timeout = int(os.getenv("VISION_TIMEOUT", "60"))
        settings.vision_max_retries = int(os.getenv("VISION_MAX_RETRIES", "3"))
        settings.vision_default_detail = os.getenv("VISION_DEFAULT_DETAIL", "auto")
        settings.vision_supports_image_detail = os.getenv("VISION_SUPPORTS_IMAGE_DETAIL", "false").lower() == "true"
        settings.vision_url_mode = os.getenv("VISION_URL_MODE", "auto").lower()
        if settings.vision_url_mode not in {"auto", "passthrough", "download"}:
            raise ValueError("VISION_URL_MODE must be one of: auto, passthrough, download")

        # Input limits
        settings.vision_max_image_size_mb = int(os.getenv("VISION_MAX_IMAGE_SIZE_MB", "10"))
        settings.vision_max_image_pixels = int(os.getenv("VISION_MAX_IMAGE_PIXELS", "40000000"))
        settings.vision_max_batch_images = int(os.getenv("VISION_MAX_BATCH_IMAGES", "10"))
        if settings.vision_timeout <= 0:
            raise ValueError("VISION_TIMEOUT must be greater than 0")
        if settings.vision_max_retries <= 0:
            raise ValueError("VISION_MAX_RETRIES must be greater than 0")
        if settings.vision_max_image_size_mb <= 0:
            raise ValueError("VISION_MAX_IMAGE_SIZE_MB must be greater than 0")
        if settings.vision_max_image_pixels <= 0:
            raise ValueError("VISION_MAX_IMAGE_PIXELS must be greater than 0")
        if settings.vision_max_batch_images < 2:
            raise ValueError("VISION_MAX_BATCH_IMAGES must be at least 2")

        # Security
        settings.vision_allow_local_files = os.getenv("VISION_ALLOW_LOCAL_FILES", "true").lower() == "true"
        allowed_paths_str = os.getenv("VISION_ALLOWED_PATHS", "/data,/tmp")
        settings.vision_allowed_paths = [p.strip() for p in allowed_paths_str.split(",") if p.strip()]
        settings.vision_block_private_ips = os.getenv("VISION_BLOCK_PRIVATE_IPS", "true").lower() == "true"
        settings.dedicated_ocr_enabled = os.getenv("OCR_ENABLED", "false").lower() == "true"

        # Transport
        settings.vision_transport = os.getenv("VISION_TRANSPORT", "stdio")

        return settings


# Global settings instance
settings = Settings.from_env()
