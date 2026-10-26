"""Local persistent conversation storage service for BookMind.

Stores conversation threads isolated per book ID in:
data/uploads/books/{book_id}/conversations/{conversation_id}.json
"""

import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.schemas.chat import ChatMessage, ConversationHistory
from app.services import storage_service

logger = logging.getLogger(__name__)


def sanitize_id(identifier: str) -> str:
    """Sanitize conversation or book ID to prevent directory traversal."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
    return cleaned[:80]


class ConversationService:
    """Manages creation, message appending, and persistence of conversation history."""

    def get_conversations_dir(self, book_id: str) -> Path:
        """Return path to conversations directory inside the book folder."""
        book_dir = storage_service.get_book_dir(book_id)
        conv_dir = book_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        return conv_dir

    def get_conversation_path(self, book_id: str, conversation_id: str) -> Path:
        """Return path to a specific conversation JSON file."""
        safe_conv_id = sanitize_id(conversation_id)
        return self.get_conversations_dir(book_id) / f"{safe_conv_id}.json"

    def get_conversation(self, book_id: str, conversation_id: str) -> Optional[ConversationHistory]:
        """Load conversation history from disk."""
        path = self.get_conversation_path(book_id, conversation_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ConversationHistory.model_validate(data)
        except Exception as e:
            logger.warning(f"Error loading conversation '{conversation_id}' for book '{book_id}': {e}")
            return None

    def create_or_get_conversation(
        self,
        book_id: str,
        conversation_id: Optional[str] = None,
    ) -> ConversationHistory:
        """Retrieve existing conversation or create a new conversation thread."""
        if conversation_id:
            existing = self.get_conversation(book_id, conversation_id)
            if existing:
                return existing

        # Generate a new unique conversation ID
        new_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        history = ConversationHistory(
            conversation_id=new_id,
            book_id=book_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        self.save_conversation(history)
        return history

    def save_conversation(self, history: ConversationHistory) -> None:
        """Persist conversation history to disk."""
        path = self.get_conversation_path(history.book_id, history.conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.updated_at = datetime.now(timezone.utc)
        with open(path, "w", encoding="utf-8") as f:
            f.write(history.model_dump_json(indent=2))

    def append_message(
        self,
        book_id: str,
        conversation_id: str,
        message: ChatMessage,
    ) -> ConversationHistory:
        """Append a user or assistant message to the thread and persist."""
        history = self.create_or_get_conversation(book_id, conversation_id)
        history.messages.append(message)
        self.save_conversation(history)
        return history

    def list_conversations(self, book_id: str) -> List[ConversationHistory]:
        """List all active conversation threads for a book."""
        conv_dir = self.get_conversations_dir(book_id)
        conversations: List[ConversationHistory] = []
        for file in conv_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conversations.append(ConversationHistory.model_validate(data))
            except Exception:
                continue
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations

    def delete_conversation(self, book_id: str, conversation_id: str) -> bool:
        """Delete a single conversation thread."""
        path = self.get_conversation_path(book_id, conversation_id)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as e:
                logger.warning(f"Failed to delete conversation '{conversation_id}': {e}")
                return False
        return False

    def delete_all_book_conversations(self, book_id: str) -> None:
        """Delete all conversation threads when a book is deleted."""
        conv_dir = storage_service.get_book_dir(book_id) / "conversations"
        if conv_dir.exists():
            try:
                shutil.rmtree(conv_dir, ignore_errors=True)
            except Exception:
                pass


_conversation_service_instance: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """Return singleton ConversationService instance."""
    global _conversation_service_instance
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService()
    return _conversation_service_instance
