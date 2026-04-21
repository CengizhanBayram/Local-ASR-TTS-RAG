"""
LLM Service — provider-agnostic katman
Gemini / Ollama / OpenAI-compat üzerinde çalışır
"""

import logging
from typing import AsyncGenerator, List, Optional

from ..config import get_settings
from ..models.exceptions import LLMError, ConfigurationError
from .providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMService:

    # ── System Prompts ───────────────────────────────────────────────────────
    DEFAULT_SYSTEM_PROMPT = """You are a helpful Turkish-language assistant.
You answer questions based solely on the source documents provided to you.

Rules:
1. Answer only using information found in the provided sources
2. If the answer is not in the sources, say "Bu konuda kaynaklarda bilgi bulamadım"
3. Keep your answers concise and to the point
4. Respond in natural, fluent Turkish
5. Do not guess or infer information you are not certain about"""

    CITATION_SYSTEM_PROMPT = """You are a helpful Turkish-language assistant.
You are given numbered source documents. When answering, append the relevant source number like [1] or [2] after each claim.

Rules:
1. Answer only using information found in the provided sources
2. Mark each piece of information with [N] indicating which source it came from
3. If the answer is not in the sources, say "Bu konuda kaynaklarda bilgi bulamadım"
4. Respond in concise, fluent Turkish"""

    FREE_SYSTEM_PROMPT = """You are a helpful Turkish-language assistant.
Answer the user's questions naturally and in a friendly tone.
Keep answers concise. Use clear, correct Turkish."""

    # ── Prompt Templates ────────────────────────────────────────────────────
    CONTEXT_TEMPLATE = """{history_section}Sources:
{context}

---
User Question: {query}

Please answer based on the sources above:"""

    QUERY_REWRITE_PROMPT = """Rewrite the user's question to be more specific and suitable for document retrieval.
Preserve the original meaning. Return only the rewritten query, nothing else.

Original: {query}
Rewritten:"""

    MULTI_QUERY_PROMPT = """Rewrite the following question in {n} different ways to improve document retrieval coverage.
Output one query per line. Do not write anything else.

Question: {query}"""

    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.settings = get_settings()
        self._provider: BaseLLMProvider = self._init_provider()

    def _init_provider(self) -> BaseLLMProvider:
        p = self.settings.llm_provider.lower()
        try:
            if p == "gemini":
                from .providers.gemini import GeminiProvider
                return GeminiProvider(self.settings)
            elif p == "ollama":
                from .providers.ollama import OllamaProvider
                return OllamaProvider(self.settings)
            elif p in ("openai_compatible", "vllm", "lmstudio", "localai"):
                from .providers.openai_compat import OpenAICompatProvider
                return OpenAICompatProvider(self.settings)
            else:
                raise ConfigurationError(f"Bilinmeyen LLM_PROVIDER: '{p}'")
        except (ValueError, ImportError) as e:
            raise ConfigurationError(str(e))

    # ── Public helpers ───────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def _build_user_message(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
    ) -> str:
        history_section = (
            f"Önceki Konuşma:\n{conversation_history}\n\n"
            if conversation_history
            else ""
        )
        if context:
            return self.CONTEXT_TEMPLATE.format(
                history_section=history_section,
                context=context,
                query=query,
            )
        return (
            f"{history_section}Soru: {query}\n\n"
            "Not: Şu anda yüklü belge bulunmuyor."
        )

    def _system_prompt(self, override: Optional[str] = None) -> str:
        if override:
            return override
        if self.settings.enable_citations:
            return self.CITATION_SYSTEM_PROMPT
        return self.DEFAULT_SYSTEM_PROMPT

    # ── Core generation ──────────────────────────────────────────────────────

    async def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: str = "",
    ) -> str:
        try:
            return await self._provider.generate(
                self._system_prompt(system_prompt),
                self._build_user_message(query, context, conversation_history),
            )
        except Exception as e:
            logger.error(f"generate_response error: {e}")
            raise LLMError(f"Cevap üretilemedi: {e}")

    async def generate_response_stream(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: str = "",
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self._provider.generate_stream(
                self._system_prompt(system_prompt),
                self._build_user_message(query, context, conversation_history),
            ):
                yield chunk
        except Exception as e:
            logger.error(f"generate_stream error: {e}")
            raise LLMError(f"Streaming başarısız: {e}")

    async def generate_free_response(
        self,
        query: str,
        conversation_history: str = "",
    ) -> str:
        history = f"Önceki Konuşma:\n{conversation_history}\n\n" if conversation_history else ""
        user_msg = f"{history}Kullanıcı: {query}"
        try:
            return await self._provider.generate(self.FREE_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            raise LLMError(f"Cevap üretilemedi: {e}")

    async def generate_free_stream(
        self,
        query: str,
        conversation_history: str = "",
    ) -> AsyncGenerator[str, None]:
        history = f"Önceki Konuşma:\n{conversation_history}\n\n" if conversation_history else ""
        user_msg = f"{history}Kullanıcı: {query}"
        async for chunk in self._provider.generate_stream(self.FREE_SYSTEM_PROMPT, user_msg):
            yield chunk

    # ── Query helpers ────────────────────────────────────────────────────────

    async def rewrite_query(self, query: str) -> str:
        try:
            result = await self._provider.generate(
                "Sen bir arama sorgusu optimize uzmanısın.",
                self.QUERY_REWRITE_PROMPT.format(query=query),
            )
            rewritten = result.strip()
            return rewritten if len(rewritten) > 5 else query
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    async def generate_query_variations(
        self, query: str, n: int = 3
    ) -> List[str]:
        try:
            result = await self._provider.generate(
                "Sen bir arama sorgusu uzmanısın.",
                self.MULTI_QUERY_PROMPT.format(query=query, n=n),
            )
            lines = [
                ln.strip().lstrip("0123456789.-) ")
                for ln in result.strip().split("\n")
                if ln.strip()
            ]
            variations = lines[:n]
            if query not in variations:
                variations.insert(0, query)
            return variations[: n + 1]
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return [query]

    # ── Health ───────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        return self._provider.is_healthy()
