"""
Ollama Provider — lokal GPU LLM (llama3, mistral, phi3 …)
API: http://localhost:11434
"""

import json
import logging
from typing import AsyncGenerator

import httpx

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):

    def __init__(self, settings):
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        logger.info(f"Ollama provider: {self._base_url}  model={self._model}")

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, system_prompt: str, user_message: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "system": system_prompt,
                    "prompt": user_message,
                    "stream": False,
                },
            )
            r.raise_for_status()
            return r.json()["response"]

    async def generate_stream(
        self, system_prompt: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "system": system_prompt,
                    "prompt": user_message,
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("response"):
                        yield data["response"]
                    if data.get("done"):
                        break

    def is_healthy(self) -> bool:
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False
