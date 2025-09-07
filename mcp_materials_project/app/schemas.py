from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]