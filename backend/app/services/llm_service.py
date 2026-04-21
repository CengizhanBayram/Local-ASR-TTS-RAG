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
    DEFAULT_SYSTEM_PROMPT = """Sen yardımcı bir Türkçe asistansın.
Sana verilen kaynak belgelerine dayanarak soruları yanıtlıyorsun.

Kurallar:
1. Sadece verilen kaynaklardaki bilgilere dayanarak cevap ver
2. Eğer cevap kaynaklarda yoksa, "Bu konuda kaynaklarda bilgi bulamadım" de
3. Cevaplarını kısa ve öz tut
4. Doğal ve akıcı bir Türkçe kullan
5. Emin olmadığın bilgileri tahmin etme"""

    CITATION_SYSTEM_PROMPT = """Sen yardımcı bir Türkçe asistansın.
Sana numaralı kaynak belgeler verilir. Cevap verirken ilgili cümlelerin sonuna [1], [2] gibi kaynak numarası ekle.

Kurallar:
1. Sadece verilen kaynaklardaki bilgilere dayanarak cevap ver
2. Her bilginin yanına hangi kaynaktan geldiğini [N] ile belirt
3. Eğer cevap kaynaklarda yoksa, "Bu konuda kaynaklarda bilgi bulamadım" de
4. Kısa, akıcı Türkçe kullan"""

    FREE_SYSTEM_PROMPT = """Sen yardımsever bir Türkçe asistansın.
Kullanıcının sorularına doğal ve samimi bir şekilde cevap ver.
Cevapların kısa ve öz olsun. Türkçe'yi akıcı ve doğru kullan."""

    # ── Prompt Templates ────────────────────────────────────────────────────
    CONTEXT_TEMPLATE = """{history_section}Kaynaklar:
{context}

---
Kullanıcı Sorusu: {query}

Lütfen yukarıdaki kaynaklara dayanarak yanıtla:"""

    QUERY_REWRITE_PROMPT = """Kullanıcının sorusunu belge araması için daha spesifik hale getir.
Özgün anlamı koru. Yalnızca yeniden yazılmış sorguyu döndür.

Orijinal: {query}
Yeniden yazılmış:"""

    MULTI_QUERY_PROMPT = """Aşağıdaki soruyu {n} farklı şekilde yeniden yaz.
Her satırda bir sorgu olsun. Başka hiçbir şey yazma.

Soru: {query}"""

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
