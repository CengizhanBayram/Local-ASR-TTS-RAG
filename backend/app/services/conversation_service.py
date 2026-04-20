"""
Conversation Service - Session bazlı konuşma hafızası
Her session için kullanıcı/asistan mesajlarını saklar
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ConversationService:
    """
    Session bazlı konuşma hafızası.
    Her kullanıcı oturumu için mesaj geçmişini tutar.
    """

    def __init__(self):
        self.settings = get_settings()
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Varolan session'ı döndür veya yeni oluştur"""
        if session_id and session_id in self._sessions:
            self._sessions[session_id].last_active = time.time()
            return session_id
        new_id = session_id or str(uuid4())
        self._sessions[new_id] = Session(session_id=new_id)
        logger.info(f"New session created: {new_id}")
        return new_id

    def add_user_message(self, session_id: str, content: str) -> None:
        self._add_message(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self._add_message(session_id, "assistant", content)

    def _add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)

        session = self._sessions[session_id]
        session.messages.append(Message(role=role, content=content))
        session.last_active = time.time()

        # En fazla max_history * 2 mesaj tut (user+assistant çiftleri)
        max_msgs = self.settings.max_conversation_history * 2
        if len(session.messages) > max_msgs:
            session.messages = session.messages[-max_msgs:]

    def get_history(self, session_id: str) -> List[Message]:
        if session_id not in self._sessions:
            return []
        return list(self._sessions[session_id].messages)

    def get_history_as_text(self, session_id: str, max_turns: int = 5) -> str:
        """Son N tur konuşmayı metin olarak döndür (LLM prompt'u için)"""
        messages = self.get_history(session_id)
        if not messages:
            return ""

        # Son max_turns * 2 mesajı al
        recent = messages[-(max_turns * 2):]
        lines = []
        for msg in recent:
            label = "Kullanıcı" if msg.role == "user" else "Asistan"
            lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)

    def get_turn_count(self, session_id: str) -> int:
        """Kullanıcı mesajı sayısını döndür"""
        if session_id not in self._sessions:
            return 0
        return sum(1 for m in self._sessions[session_id].messages if m.role == "user")

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False

    def cleanup_expired(self) -> int:
        """Süresi dolmuş session'ları temizle"""
        expiry = self.settings.session_expiry_minutes * 60
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > expiry
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def is_healthy(self) -> bool:
        return True
