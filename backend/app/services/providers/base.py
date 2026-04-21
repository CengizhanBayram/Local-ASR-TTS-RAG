"""
Base LLM Provider — tüm provider'lar bu arayüzü uygular
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider'ın kısa adı (gemini, ollama, openai_compatible)"""

    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str) -> str:
        """Tam cevabı bekle ve döndür"""

    @abstractmethod
    async def generate_stream(
        self, system_prompt: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """Token token async generator olarak döndür"""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Provider erişilebilir mi?"""
