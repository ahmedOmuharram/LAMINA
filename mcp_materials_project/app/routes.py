from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest
from .core import do_stream_response, do_json_response
from dotenv import load_dotenv

APP_TITLE = "MPKani OpenWebUI Integration"
APP_VERSION = "1.1.6"
DEFAULT_MODEL = "gpt-4.1"

load_dotenv()

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS (tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MPKani OpenWebUI Integration Server"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenWebUI compatibility)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4.1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "mpkani",
                "permission": [],
                "root": "gpt-4.1",
                "parent": None,
                "name": "OpenAI GPT-4.1",
                "description": "Most capable model for complex materials science queries"
            },
            {
                "id": "o1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "mpkani",
                "permission": [],
                "root": "o1",
                "parent": None,
                "name": "OpenAI o1",
                "description": "For complex reasoning tasks"
            },
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1700000000,
                "owned_by": "mpkani",
                "permission": [],
                "root": "gpt-4o-mini",
                "parent": None,
                "name": "OpenAI GPT-4o Mini",
                "description": "Fast and cost-effective for quick materials queries"
            },
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    model = request.model or DEFAULT_MODEL

    # If OpenWebUI asked for streaming, keep your current SSE path.
    if request.stream:
        return await do_stream_response(request.messages, model)

    # Otherwise return a standard OpenAI Chat Completions JSON object.
    return await do_json_response(request.messages, model)
