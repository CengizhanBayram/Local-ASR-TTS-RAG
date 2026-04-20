"""
LLM Service - Google Gemini API
"""

import logging
from typing import Optional

import google.generativeai as genai

from ..config import get_settings
from ..models.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)


class LLMService:
    DEFAULT_SYSTEM_PROMPT = """Sen yardımcı bir Türkçe asistansın.
Sana verilen kaynak belgelerine dayanarak soruları yanıtlıyorsun.

Kurallar:
1. Sadece verilen kaynaklardaki bilgilere dayanarak cevap ver
2. Eğer cevap kaynaklarda yoksa, "Bu konuda kaynaklarda bilgi bulamadım" de
3. Cevaplarını kısa ve öz tut
4. Doğal ve akıcı bir Türkçe kullan
5. Gerektiğinde kaynak belgeyi belirt
6. Emin olmadığın bilgileri tahmin etme"""

    CONTEXT_TEMPLATE = """Aşağıdaki kaynak belgeler verilmiştir:

{context}

---

Kullanıcı Sorusu: {query}

Lütfen yukarıdaki kaynaklara dayanarak soruyu yanıtla:"""

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            if not self.settings.gemini_api_key:
                raise ConfigurationError("GEMINI_API_KEY eksik.")
            genai.configure(api_key=self.settings.gemini_api_key)
            self._model = genai.GenerativeModel(self.settings.gemini_model)
            logger.info(f"LLM Service initialized: {self.settings.gemini_model}")
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM Service: {e}")
            raise LLMError(f"LLM başlatılamadı: {str(e)}")

    async def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        try:
            system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

            if context:
                user_message = self.CONTEXT_TEMPLATE.format(context=context, query=query)
            else:
                user_message = f"Soru: {query}\n\nNot: Şu anda yüklü belge bulunmuyor."

            full_prompt = f"{system}\n\n{user_message}"

            response = self._model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )

            answer = response.text
            logger.info(f"Generated response: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise LLMError(f"Cevap üretilemedi: {str(e)}")

    async def generate_with_custom_prompt(
        self,
        query: str,
        context: str,
        custom_instructions: str
    ) -> str:
        enhanced_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nEk Talimatlar:\n{custom_instructions}"
        return await self.generate_response(query, context, enhanced_prompt)

    def is_healthy(self) -> bool:
        return self._model is not None
