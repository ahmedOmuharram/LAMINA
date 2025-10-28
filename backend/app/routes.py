from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
import logging
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest
from .core import do_stream_response
from dotenv import load_dotenv
from pathlib import Path

APP_TITLE = "LAMINA: LLM-Assisted Material INformatics and Analysis"
DEFAULT_MODEL = "gpt-4o"
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
    Always streams responses. Supports both text-only and multimodal (text + images) messages.
    """
    _log.info("/v1/chat/completions: model=%s msgs=%s", 
              request.model, len(request.messages or []))
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Validate last message is from user
    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    
    model = request.model or DEFAULT_MODEL
    
    # Use unified streaming path (handles both text and images)
    _log.info("/v1/chat/completions: Unified streaming (Kani + OpenAI engine)")
    return await do_stream_response(request.messages, model, raw)
