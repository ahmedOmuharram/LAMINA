#!/usr/bin/env python3
"""OpenWebUI server for MPKani chatbot (fixed mp-NUMBER link rendering)."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
from dotenv import load_dotenv

from kani import ChatRole            # role types from kani
from kani import ChatMessage as KChatMessage  # alias to avoid clashing with pydantic ChatMessage
from .kani_client import MPKani      # your kani wrapper/client

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv()
OPENWEBUI_TOOL_PILL = os.getenv("OPENWEBUI_TOOL_PILL", "false").lower() in ("1", "true", "yes")

app = FastAPI(title="MPKani OpenWebUI Integration", version="1.1.3")

# CORS (tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
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
    """OpenWebUI-compatible endpoint."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    model = request.model or "gpt-4.1"  # Default to gpt-4.1 if no model specified
    
    return await stream_response(request.messages, model)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
async def stream_response(messages: List[ChatMessage], model: str = "gpt-4.1"):
    import time
    model_name = model

    # Tool tracking
    tool_fifo: List[str] = []
    tool_name_by_id: Dict[str, str] = {}
    tool_started_at: Dict[str, float] = {}
    tools_open = 0  # count of tools currently running

    # Answer buffering + de-dup (only while tools are running)
    buffer_chunks: List[str] = []
    seen_chunks: set[str] = set()

    def _norm(s: str) -> str:
        return " ".join((s or "").split())

    def _details(summary: str, body_md: str = "", open_: bool = False) -> str:
        open_attr = " open" if open_ else ""
        body_md = (body_md or "").strip()
        inner = f"{body_md}\n" if body_md else ""
        # blank lines around details so it doesn't glue to text
        return f"\n<details{open_attr}>\n  <summary>{summary}</summary>\n\n{inner}</details>\n\n"

    def _tool_panel_done(tool_name: str, duration: float, logs_md: str = "") -> str:
        return _details(f"✅ {tool_name} — done ({duration:.2f}s)", logs_md, open_=False)

    def tools_in_progress() -> bool:
        return tools_open > 0 or len(tool_fifo) > 0

    def _to_kani_history(api_messages: List[ChatMessage]) -> list[KChatMessage]:
        out: list[KChatMessage] = []
        for m in api_messages:
            role = (m.role or "").lower()
            content = m.content or ""
            if role == "system":
                out.append(KChatMessage.system(content))
            elif role == "user":
                out.append(KChatMessage.user(content))
            elif role == "assistant":
                out.append(KChatMessage.assistant(content))
            elif role == "function":
                # If your UI sends tool results, keep them; name unknown is OK.
                out.append(KChatMessage.function(name=None, content=content))
            # else: ignore unknown roles
        return out

    async def sse():
        nonlocal tools_open

        # Validate & extract last user message
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        if messages[-1].role.lower() != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")

        user_prompt = messages[-1].content

        # Build prior history (everything except the last user turn)
        prior_api_msgs = messages[:-1]
        kani_chat_history = _to_kani_history(prior_api_msgs)

        # Create MPKani instance with specified model
        kani_instance = MPKani(model=model, chat_history=kani_chat_history)

        # Debug: Log the start of streaming

        async for stream in kani_instance.full_round_stream(user_prompt):
            role = getattr(stream, "role", None)

            if role == ChatRole.ASSISTANT:
                # Emit role header once per assistant turn
                yield _role_header_chunk(model_name)

                # STREAM TOKENS IMMEDIATELY
                async for token in stream:
                    if token:
                        yield _delta_chunk(token, model_name)

                # AFTER tokens, fetch the finalized message (for tool_calls, etc.)
                msg = await stream.message()

                # Emit tool pills (if any) AFTER content tokens, then register tools as in-progress
                if getattr(msg, "tool_calls", None):
                    if OPENWEBUI_TOOL_PILL:
                        tool_delta = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": i,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": tc.function.name},
                                    } for i, tc in enumerate(msg.tool_calls)],
                                },
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(tool_delta)}\n\n"

                    # Register tools as "in progress"
                    for tc in msg.tool_calls:
                        tc_id = getattr(tc, "id", f"tc-{int(time.time()*1e6)}")
                        tool_fifo.append(tc_id)
                        tool_name_by_id[tc_id] = getattr(tc.function, "name", "tool")
                        tool_started_at[tc_id] = time.time()
                        tools_open += 1

            elif role == ChatRole.FUNCTION:
                # A tool finished: emit its final panel; if this was the last tool, flush buffer once
                tool_msg = await stream.message()

                if tool_fifo:
                    tc_id = tool_fifo.pop(0)
                    tool_name = tool_name_by_id.pop(tc_id, tool_msg.name or "tool")
                    started = tool_started_at.pop(tc_id, None)
                    duration = time.time() - started if started else 0.0
                    tools_open = max(0, tools_open - 1)

                    # Optional logs (from your wrapper); keep empty if you don't have any
                    logs_md = ""
                    if getattr(kani_instance, "recent_tool_outputs", None):
                        try:
                            latest = kani_instance.recent_tool_outputs[-1]
                            try:
                                display_obj = dict(latest)
                            except Exception:
                                display_obj = {k: latest[k] for k in latest} if isinstance(latest, dict) else {"record": str(latest)}
                            res_val = display_obj.get("result")
                            if isinstance(res_val, str):
                                try:
                                    display_obj["result"] = json.loads(res_val)
                                except Exception:
                                    pass
                            pretty = json.dumps(display_obj, indent=2, ensure_ascii=False)
                            logs_md = f"**Tool**: `{tool_name}`\n\n**Output**:\n\n```json\n{pretty}\n```"
                        except Exception:
                            try:
                                pretty = json.dumps(kani_instance.recent_tool_outputs[-1], indent=2, ensure_ascii=False)
                                logs_md = f"**Tool**: `{tool_name}`\n\n**Output**:\n\n```json\n{pretty}\n```"
                            except Exception:
                                logs_md = f"**Tool**: `{tool_name}`\n\n**Output**:\n\n```\n{str(kani_instance.recent_tool_outputs[-1])}\n```"

                    yield _delta_chunk(_tool_panel_done(tool_name, duration, logs_md), model_name)

                    # If this was the LAST tool, flush the buffered answer exactly once
                    if not tools_in_progress() and buffer_chunks:
                        yield _delta_chunk("".join(buffer_chunks), model_name)
                        buffer_chunks.clear()
                        seen_chunks.clear()

                else:
                    # No queued tool but got a FUNCTION end — show panel anyway
                    tool_name = getattr(tool_msg, "name", None) or "tool"
                    yield _delta_chunk(_tool_panel_done(tool_name, 0.0, ""), model_name)
                    if not tools_in_progress() and buffer_chunks:
                        yield _delta_chunk("".join(buffer_chunks), model_name)
                        buffer_chunks.clear()
                        seen_chunks.clear()

            else:
                # Other roles: stream tokens immediately as well
                yield _role_header_chunk(model_name)
                async for token in stream:
                    if token:
                        yield _delta_chunk(token, model_name)

        # End-of-stream safety: flush anything still buffered
        if buffer_chunks:
            yield _delta_chunk("".join(buffer_chunks), model_name)
            buffer_chunks.clear()
            seen_chunks.clear()

        # Close any orphan tools with zero-duration panels
        while tool_fifo:
            tc_id = tool_fifo.pop(0)
            tool_name = tool_name_by_id.pop(tc_id, "tool")
            tool_started_at.pop(tc_id, None)
            yield _delta_chunk(_tool_panel_done(tool_name, 0.0, ""), model_name)

        stop = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(stop)}\n\n"
        yield "data: [DONE]\n\n"

        kani_instance.recent_tool_outputs = []


    return StreamingResponse(sse(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# SSE helpers
# -----------------------------------------------------------------------------
def _role_header_chunk(model_name: str) -> str:
    payload = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

def _delta_chunk(text: str, model_name: str) -> str:
    import re

    def replace_in_plain(s: str) -> str:
        # Match mp-<digits> not embedded in a larger word (allows following punctuation)
        pattern = re.compile(r'(?<!\w)(mp-(\d+))(?!\w)')

        def repl(m: re.Match) -> str:
            full = m.group(1)  # e.g. mp-12345
            return f"[{full}](https://next-gen.materialsproject.org/materials/{full})"

        return pattern.sub(repl, s)

    # Preserve inline code spans; don't auto-link inside them
    parts = re.split(r'(`[^`]*`)', text)  # keep delimiters
    for i, part in enumerate(parts):
        if i % 2 == 0:  # non-code
            parts[i] = replace_in_plain(part)
    text = "".join(parts)

    payload = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time() * 10_000)}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
