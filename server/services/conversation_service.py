"""
Conversation service for managing chat context and history.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""

    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }


@dataclass
class Conversation:
    """Represents a conversation session."""

    id: str
    subject_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[Message]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata or {},
        }


class ConversationService:
    """Manages conversation sessions and context."""

    def __init__(
        self, max_conversations_per_subject: int = 10, conversation_ttl_hours: int = 24
    ):
        """
        Initialize the conversation service.

        Args:
            max_conversations_per_subject: Maximum number of conversations to keep per subject
            conversation_ttl_hours: Hours after which inactive conversations are cleaned up
        """
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations_per_subject = max_conversations_per_subject
        self.conversation_ttl_hours = conversation_ttl_hours

    def get_or_create_conversation(
        self, subject_id: str, conversation_id: Optional[str] = None
    ) -> Conversation:
        """
        Get an existing conversation or create a new one.

        Args:
            subject_id: The subject ID
            conversation_id: Optional specific conversation ID to retrieve

        Returns:
            Conversation object
        """
        if conversation_id and conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            if conv.subject_id == subject_id:
                conv.last_activity = datetime.now()
                return conv
            else:
                logger.warning(
                    f"Conversation {conversation_id} belongs to different subject"
                )

        # Create new conversation
        new_conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=new_conversation_id,
            subject_id=subject_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            messages=[],
            metadata={},
        )

        self.conversations[new_conversation_id] = conversation

        # Clean up old conversations for this subject
        self._cleanup_old_conversations(subject_id)

        logger.info(
            f"Created new conversation {new_conversation_id} for subject {subject_id}"
        )
        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            conversation_id: The conversation ID
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata

        Returns:
            The created Message object

        Raises:
            KeyError: If conversation doesn't exist
        """
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]
        conversation.last_activity = datetime.now()

        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        conversation.messages.append(message)
        logger.debug(f"Added {role} message to conversation {conversation_id}")

        return message

    def get_conversation_history(
        self, conversation_id: str, max_messages: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation history, optionally limited to recent messages.

        Args:
            conversation_id: The conversation ID
            max_messages: Maximum number of messages to return (None for all)

        Returns:
            List of messages

        Raises:
            KeyError: If conversation doesn't exist
        """
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")

        messages = self.conversations[conversation_id].messages

        if max_messages:
            messages = messages[-max_messages:]

        return messages

    def get_conversation_context(
        self, conversation_id: str, max_messages: int = 10
    ) -> str:
        """
        Get conversation context as a formatted string for the agent.

        Args:
            conversation_id: The conversation ID
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation context string
        """
        try:
            messages = self.get_conversation_history(conversation_id, max_messages)

            if not messages:
                return ""

            context_lines = ["Previous conversation:"]
            for msg in messages:
                role_display = "User" if msg.role == "user" else "Assistant"
                context_lines.append(f"{role_display}: {msg.content}")

            return "\n".join(context_lines)

        except KeyError:
            return ""

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            Conversation object or None if not found
        """
        return self.conversations.get(conversation_id)

    def list_conversations(self, subject_id: str) -> List[Conversation]:
        """
        List all conversations for a subject.

        Args:
            subject_id: The subject ID

        Returns:
            List of conversations sorted by last activity
        """
        subject_conversations = [
            conv
            for conv in self.conversations.values()
            if conv.subject_id == subject_id
        ]

        return sorted(
            subject_conversations, key=lambda x: x.last_activity, reverse=True
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if deleted, False if not found
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        return False

    def cleanup_expired_conversations(self) -> int:
        """
        Clean up conversations that have exceeded the TTL.

        Returns:
            Number of conversations cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=self.conversation_ttl_hours)
        expired_ids = [
            conv_id
            for conv_id, conv in self.conversations.items()
            if conv.last_activity < cutoff_time
        ]

        for conv_id in expired_ids:
            del self.conversations[conv_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired conversations")

        return len(expired_ids)

    def _cleanup_old_conversations(self, subject_id: str) -> None:
        """
        Clean up old conversations for a subject, keeping only the most recent ones.

        Args:
            subject_id: The subject ID
        """
        subject_conversations = self.list_conversations(subject_id)

        if len(subject_conversations) > self.max_conversations_per_subject:
            # Keep only the most recent conversations
            conversations_to_delete = subject_conversations[
                self.max_conversations_per_subject :
            ]

            for conv in conversations_to_delete:
                del self.conversations[conv.id]

            logger.info(
                f"Cleaned up {len(conversations_to_delete)} old conversations for subject {subject_id}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with service stats
        """
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())

        # Group by subject
        subjects = {}
        for conv in self.conversations.values():
            if conv.subject_id not in subjects:
                subjects[conv.subject_id] = {"conversations": 0, "messages": 0}
            subjects[conv.subject_id]["conversations"] += 1
            subjects[conv.subject_id]["messages"] += len(conv.messages)

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "subjects": subjects,
            "max_conversations_per_subject": self.max_conversations_per_subject,
            "conversation_ttl_hours": self.conversation_ttl_hours,
        }


# Global instance
conversation_service = ConversationService()
