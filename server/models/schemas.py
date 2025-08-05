# src/models/schemas.py

from pydantic import BaseModel
from typing import Any, Optional


class ChatRequest(BaseModel):
    """
    The expected JSON body for POST /chat.
    """
    subject_id: str  # e.g. "Subject 9"
    text: str        # the user’s message


class ChatResponse(BaseModel):
    """
    The JSON response from POST /chat.
    """
    reply: str                 # the assistant’s reply
    tool_used: Optional[str]   # name of the tool called, if any
    tool_result: Optional[Any] # the raw result from that tool
