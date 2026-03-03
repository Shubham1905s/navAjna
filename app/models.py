"""
DATA MODELS MODULE
=================

This file defines the pydantic models used for API request, response, and
internal chat storage. FastAPI uses these o validate incoming JSON and to 
serialize responses; the chat service uses them when saving/loading sessions.

MODELS:
   ChatRequest     - Body of POST /chat and POST /chat/realtime (message + optional session_id).
   ChatResponse    - returned by both chat endpoints (response text + session_id).
   ChatMessage     - One message in a conversation (role + content). Used inside ChatHistory.
   ChatHistory     - Full conversation: session_id + list of  ChatMessage. Used when saving to disk
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=22_000)
    session_id: Optional[str] = None
    tts: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
