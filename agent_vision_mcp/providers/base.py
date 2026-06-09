"""Base provider interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseVisionProvider(ABC):
    """Abstract base class for vision providers"""

    @abstractmethod
    def analyze(
        self,
        images: list[Dict[str, Any]],
        prompt: str,
        detail: str = "auto",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Analyze images with a text prompt.

        Args:
            images: List of image dictionaries with 'data_url' key
            prompt: Text prompt to send to the model
            detail: Detail level (auto, low, high)
            max_tokens: Maximum tokens in response

        Returns:
            Model's text response
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities"""
        pass