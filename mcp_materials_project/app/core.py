from __future__ import annotations
from typing import List, Iterable
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from .schemas import ChatMessage
from .utils import role_header_chunk, final_stop_chunk, linkify_mp_numbers
from .stream_state import StreamState
from ..kani_client import MPKani
from kani import ChatRole
from .utils import to_kani_history
import time

DEFAULT_MODEL = "gpt-4.1"

async def sse_generator(messages: List[ChatMessage], model: str) -> Iterable[str]:
    """Produce SSE chunks for a single chat round."""
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    if messages[-1].role.lower() != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    model_name = model or DEFAULT_MODEL
    user_prompt = messages[-1].content
    prior_api_msgs = messages[:-1]
    kani_chat_history = to_kani_history(prior_api_msgs)

    kani_instance = MPKani(model=model_name, chat_history=kani_chat_history)
    state = StreamState(model_name=model_name)

    async for stream in kani_instance.full_round_stream(user_prompt):
        role = getattr(stream, "role", None)

        if role == ChatRole.ASSISTANT:
            # Emit role header once per assistant turn
            yield role_header_chunk(model_name)

            # Stream tokens immediately (with buffering linkifier)
            async for token in stream:
                if token is None:
                    continue
                chunk = state.emit_stream_text(token)
                if chunk:
                    yield chunk

            # AFTER tokens, fetch the finalized message (for tool_calls, etc.)
            msg = await stream.message()
            state.register_tool_calls(msg)

        elif role == ChatRole.FUNCTION:
            # A tool finished: emit its final panel; if this was the last tool, flush buffer once
            tool_msg = await stream.message()
            for chunk in state.complete_next_tool(tool_msg, kani_instance):
                yield chunk

        else:
            # Other roles: stream tokens immediately as well
            yield role_header_chunk(model_name)
            async for token in stream:
                if token is None:
                    continue
                chunk = state.emit_stream_text(token)
                if chunk:
                    yield chunk

    # End-of-stream safety: flush any remaining linkifier tail
    tail_chunk = state.flush_linkbuf()
    if tail_chunk:
        yield tail_chunk

    # End-of-stream safety: flush anything still buffered
    for chunk in state.flush_buffer_if_any():
        yield chunk

    # Close any orphan tools with zero-duration panels
    for chunk in state.close_orphan_tools():
        yield chunk

    # Final stop
    yield final_stop_chunk(model_name)
    yield "data: [DONE]\n\n"

    # Clear any saved tool outputs on the kani wrapper (if present)
    try:
        kani_instance.recent_tool_outputs = []
    except Exception:
        pass

async def do_stream_response(messages: List[ChatMessage], model: str = DEFAULT_MODEL) -> StreamingResponse:
    """Return a StreamingResponse wrapping the SSE generator."""
    return StreamingResponse(sse_generator(messages, model), media_type="text/event-stream")

async def do_json_response(messages: List[ChatMessage], model: str) -> JSONResponse:
    """Run a non-streaming round and return OpenAI-style JSON."""
    user_prompt = messages[-1].content
    kani_instance = MPKani(model=model, chat_history=to_kani_history(messages[:-1]))

    content_parts: list[str] = []

    async for stream in kani_instance.full_round_stream(user_prompt):
        role = getattr(stream, "role", None)
        if role == ChatRole.ASSISTANT:
            async for token in stream:
                if token:
                    content_parts.append(token)
        elif role == ChatRole.FUNCTION:
            # For title/follow-up prompts, tools shouldn't be called; ignore.
            _ = await stream.message()

    full_text = "".join(content_parts) or ""

    payload = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    return JSONResponse(payload, status_code=200)