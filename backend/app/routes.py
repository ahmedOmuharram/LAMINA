from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
import logging
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest, ChatMessage
from .core import do_stream_response, do_openai_stream_response
from dotenv import load_dotenv
from pathlib import Path

APP_TITLE = "LAMINA: LLM-Assisted Material INformatics and Analysis"
DEFAULT_MODEL = "gpt-4o-mini"
APP_VERSION = "0.0.1"

load_dotenv()

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
_log = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for interactive plots
interactive_plots_dir = Path("/Users/ahmedmuharram/thesis/interactive_plots")
interactive_plots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/plots", StaticFiles(directory=str(interactive_plots_dir)), name="plots")

@app.get("/")
async def root():
    return {
        "message": "LAMINA: LLM-Assisted Material INformatics and Analysis",
        "version": APP_VERSION,
        "status": "operational"
    }

@app.get("/v1/models")
async def list_models():
    """List available models for LAMINA."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
                "permission": [],
                "root": "gpt-4o-mini",
                "parent": None,
                "name": "GPT-4o Mini",
                "description": "Fast and intelligent model optimized for materials science analysis"
            },
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
                "permission": [],
                "root": "gpt-4o",
                "parent": None,
                "name": "GPT-4o",
                "description": "Most capable model for complex materials science queries"
            },
            {
                "id": "o1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
                "permission": [],
                "root": "o1",
                "parent": None,
                "name": "OpenAI o1",
                "description": "Advanced reasoning model for complex analysis tasks"
            },
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw: Request):
    """
    LAMINA chat completions endpoint.
    Always streams responses. Supports text and images (via base64 or URLs).
    """
    _log.info("/v1/chat/completions: model=%s msgs=%s", 
              request.model, len(request.messages or []))
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Check if last message is from user (only when no images)
    has_input = bool(request.input)
    has_image = False
    
    if not has_input and request.messages:
        last_message = request.messages[-1]
        
        # Check for images in last message
        if isinstance(last_message.content, list):
            for part in last_message.content:
                part_dict = part if isinstance(part, dict) else None
                if part_dict is None:
                    try:
                        from pydantic import BaseModel as _BM
                        if isinstance(part, _BM):
                            part_dict = part.model_dump()
                    except Exception:
                        pass
                
                if isinstance(part_dict, dict):
                    t = part_dict.get("type")
                    if t in ("image_url", "input_image", "image"):
                        has_image = True
                        break
                    if any(k in part_dict for k in ("image_url", "b64", "data", "url")):
                        has_image = True
                        break
        
        # Enforce user message only when no images
        if not has_image and last_message.role.lower() != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
    
    model = request.model or DEFAULT_MODEL
    
    # Route based on whether images are present
    if has_input or has_image:
        # Images present → use OpenAI streaming
        _log.info("/v1/chat/completions: OpenAI streaming (images present)")
        return await do_openai_stream_response(request, model)
    else:
        # Text-only → use Kani streaming  
        _log.info("/v1/chat/completions: Kani streaming (text-only)")
        return await do_stream_response(request.messages, model, raw)
