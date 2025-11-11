from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from typing_extensions import Literal

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str
    # detail can be "low" | "high" | "auto" in OpenAI spec; accept anything if present
    # Using Any to avoid strict validation issues
    # Optional to keep compatibility with various clients
    detail: Optional[Any] = None


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    # OpenAI allows either a string URL or an object { url, detail }
    image_url: Union[str, ImageURL]


MessageContent = Union[str, List[Union[ContentPartText, ContentPartImage, Dict[str, Any]]]]


class ChatMessage(BaseModel):
    role: str
    content: MessageContent

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enabled_functions: Optional[List[str]] = None  # List of function names to enable

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]

class CohenKappaRequest(BaseModel):
    y1: List[int]
    y2: List[int]

class CohenKappaResponse(BaseModel):
    kappa: float
    agreement: str

class AIFunctionInfo(BaseModel):
    name: str
    description: str
    category: str

class AIFunctionsListResponse(BaseModel):
    functions: List[AIFunctionInfo]