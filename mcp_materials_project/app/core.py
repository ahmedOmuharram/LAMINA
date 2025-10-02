from __future__ import annotations
from typing import List, Iterable, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import logging
from .schemas import ChatMessage, ChatRequest
from .utils import role_header_chunk, final_stop_chunk, linkify_mp_numbers, delta_chunk_raw
from .stream_state import StreamState
from ..kani_client import MPKani
from kani import ChatRole
from .utils import to_kani_history, extract_text_from_message_content, _owui_input_to_image_parts
import time
import os
import json
import openai
from pydantic import BaseModel

DEFAULT_MODEL = "gpt-4.1"
_log = logging.getLogger(__name__)

async def sse_generator(messages: List[ChatMessage], model: str) -> Iterable[str]:
    """Produce SSE chunks for a single chat round."""
    if not messages:
        _log.error("sse_generator: no messages provided")
        print("sse_generator: no messages provided", flush=True)
        raise HTTPException(status_code=400, detail="No messages provided")
    if messages[-1].role.lower() != "user":
        _log.error("sse_generator: last message must be from user (got role=%s)", messages[-1].role)
        print(f"sse_generator: last message must be from user (got role={messages[-1].role})", flush=True)
        raise HTTPException(status_code=400, detail="Last message must be from user")

    model_name = model or DEFAULT_MODEL
    user_prompt = extract_text_from_message_content(messages[-1].content)
    prior_api_msgs = messages[:-1]
    kani_chat_history = to_kani_history(prior_api_msgs)

    _log.info("sse_generator: model=%s messages=%s prior=%s", model_name, 1, len(prior_api_msgs))
    print(f"sse_generator: model={model_name} messages=1 prior={len(prior_api_msgs)}", flush=True)
    kani_instance = MPKani(model=model_name, chat_history=kani_chat_history)
    state = StreamState(model_name=model_name)
    state._kani_instance = kani_instance  # Pass kani instance for image access

    async for stream in kani_instance.full_round_stream(user_prompt):
        role = getattr(stream, "role", None)

        if role == ChatRole.FUNCTION:
            tool_msg = await stream.message()
            for chunk in state.complete_next_tool(tool_msg, kani_instance):
                yield chunk
            continue

        # For ANY other role, buffer first, then decide
        yield role_header_chunk(model_name)

        tmp_tokens: list[str] = []
        async for token in stream:
            if token is not None:
                tmp_tokens.append(token)

        msg = await stream.message()
        state.register_tool_calls(msg)

        has_tools = bool(
            getattr(msg, "tool_calls", None) or
            getattr(msg, "function_call", None)  # older field, just in case
        )

        # Heuristic fallback: if the buffered text contains a function tag, treat as tool call
        raw = "".join(tmp_tokens)
        looks_like_func_tag = ("<|functions." in raw) or ("<|function." in raw)

        if not (has_tools or looks_like_func_tag):
            for tok in tmp_tokens:
                chunk = state.emit_stream_text(tok)
                if chunk:
                    yield chunk
        # else: drop the prelude & markup entirely

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

    # Emit any pending interactive plot link right at the end
    plot_link_chunk = state.emit_pending_plot_link()
    if plot_link_chunk:
        yield plot_link_chunk

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

async def do_json_response(messages: List[ChatMessage] | ChatRequest, model: str) -> JSONResponse:
    """Run a non-streaming round and return OpenAI-style JSON.

    If images are present (either in message content or in request.input), forward the
    request directly to OpenAI's Chat Completions API so binary/base64 images are preserved.
    Otherwise, use the local Kani pipeline.
    """
    # Back-compat: support being called as do_json_response(request, model)
    request_like: ChatRequest | None = None
    if isinstance(messages, ChatRequest):  # type: ignore[unreachable]
        request_like = messages
        messages = request_like.messages

    def _has_image_content(msgs: List[ChatMessage], input_field: Any | None) -> bool:
        if input_field:
            return True
        for m in msgs:
            content = m.content
            if isinstance(content, list):
                for part in content:
                    part_dict = None
                    if isinstance(part, dict):
                        part_dict = part
                    else:
                        # Handle Pydantic models like ContentPartImage/ContentPartText
                        try:
                            from pydantic import BaseModel as _BM
                            if isinstance(part, _BM):
                                part_dict = part.model_dump()
                        except Exception:
                            part_dict = None

                    if isinstance(part_dict, dict):
                        ptype = part_dict.get("type")
                        if ptype in ("image_url", "input_image", "image"):
                            return True
                        if any(k in part_dict for k in ("image_url", "b64", "data", "url")):
                            return True
        return False

    input_field = getattr(request_like, "input", None) if request_like else None
    temperature = getattr(request_like, "temperature", None) if request_like else None
    max_tokens = getattr(request_like, "max_tokens", None) if request_like else None

    # ---------- Path A: Images present → forward to OpenAI and RETURN here ----------
    if _has_image_content(messages, input_field):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)

        DEFAULT_IMAGE_PROMPT = "Describe the attached image in detail."

        # Build messages payload (preserve multimodal parts)
        messages_payload: list[dict[str, Any]] = []
        for m in messages:
            content = m.content
            if isinstance(content, list):
                normalized: list[dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("input_image", "image"):
                        img = part.get("image") or part.get("data") or part.get("b64")
                        mime = None
                        if isinstance(img, dict):
                            b64 = img.get("data") or img.get("b64") or img.get("base64")
                            mime = img.get("mime_type") or img.get("media_type") or img.get("type") or "image/png"
                            if isinstance(img.get("url"), str):
                                normalized.append({"type": "image_url", "image_url": {"url": img["url"], "detail": "high"}})
                            elif isinstance(b64, str):
                                url = f"data:{mime};base64,{b64}"
                                normalized.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                        elif isinstance(img, str):
                            if img.startswith("data:"):
                                url = img
                            else:
                                url = f"data:{mime or 'image/png'};base64,{img}"
                            normalized.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                    else:
                        # pass through any existing parts (text, image_url, etc.)
                        normalized.append(part)
                content = normalized

            messages_payload.append({"role": m.role, "content": content})

        # If OpenWebUI sent request.input, append those images to the last user message
        if input_field:
            img_parts = _owui_input_to_image_parts(input_field)

            if messages_payload and messages_payload[-1]["role"] == "user":
                last = messages_payload[-1]
                if isinstance(last["content"], list):
                    last["content"].extend(img_parts)
                else:
                    last["content"] = [{"type": "text", "text": str(last["content"])}] + img_parts
            else:
                messages_payload.append({"role": "user", "content": img_parts})

        # Ensure final user message has a text instruction if it's image-only
        if not messages_payload:
            # No messages at all; create a user msg with default prompt
            messages_payload.append({"role": "user", "content": [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]})
        else:
            last = messages_payload[-1]
            if last.get("role") != "user":
                # Make sure the last message to the model is a user instruction
                messages_payload.append({"role": "user", "content": [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]})
            else:
                content = last.get("content")
                if isinstance(content, list):
                    has_text = any(
                        isinstance(p, dict) and p.get("type") == "text"
                        and isinstance(p.get("text"), str) and p["text"].strip()
                        for p in content
                    )
                    has_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
                    if has_image and not has_text:
                        content.insert(0, {"type": "text", "text": DEFAULT_IMAGE_PROMPT})
                elif isinstance(content, str):
                    # if it's a non-empty string, fine; if empty, replace with default text
                    if not content.strip():
                        last["content"] = [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]
                else:
                    # unknown shape → set a default instruction
                    last["content"] = [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]

        # Convert any lingering Pydantic models to plain Python types
        def to_plain(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return {k: to_plain(v) for k, v in obj.model_dump().items()}
            if isinstance(obj, dict):
                return {k: to_plain(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_plain(v) for v in obj]
            return obj

        messages_payload = to_plain(messages_payload)

        model_to_use = model or "gpt-4o"
        try:
            _log.info("do_json_response: calling OpenAI with model=%s msgs=%s", model_to_use, len(messages_payload))
            print(f"do_json_response: calling OpenAI model={model_to_use} msgs={len(messages_payload)}", flush=True)
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=messages_payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            _log.exception("do_json_response: OpenAI error")
            print(f"do_json_response: OpenAI error: {e}", flush=True)
            raise HTTPException(status_code=502, detail=f"Upstream OpenAI error: {e}")

        # Build OpenAI-style JSON response and RETURN
        choice0 = resp.choices[0]
        assistant_message = getattr(choice0, "message", None)
        content = assistant_message.content if assistant_message else ""
        usage_obj = getattr(resp, "usage", None)
        try:
            usage_plain = to_plain(usage_obj) if usage_obj is not None else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception:
            usage_plain = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        payload = {
            "id": getattr(resp, "id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_to_use,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": getattr(choice0, "finish_reason", "stop"),
            }],
            "usage": usage_plain,
        }
        return JSONResponse(payload, status_code=200)

    # ---------- Path B: No images → run locally via Kani and RETURN ----------
    # Convert prior messages to Kani history; last one must be user
    if not messages:
        _log.error("do_json_response[Kani]: no messages provided")
        print("do_json_response[Kani]: no messages provided", flush=True)
        raise HTTPException(status_code=400, detail="No messages provided")
    if (messages[-1].role or "").lower() != "user":
        _log.error("do_json_response[Kani]: last message must be from user (got role=%s)", messages[-1].role)
        print(f"do_json_response[Kani]: last message must be from user (got role={messages[-1].role})", flush=True)
        raise HTTPException(status_code=400, detail="Last message must be from user")

    model_name = model or DEFAULT_MODEL
    user_prompt = extract_text_from_message_content(messages[-1].content)
    kani_history = to_kani_history(messages[:-1])

    _log.info("do_json_response[Kani]: model=%s prior=%s", model_name, len(kani_history))
    print(f"do_json_response[Kani]: model={model_name} prior={len(kani_history)}", flush=True)
    kani_instance = MPKani(model=model_name, chat_history=kani_history)

    # Collect assistant text by consuming the streaming iterator
    assistant_text_parts: list[str] = []
    async for stream in kani_instance.full_round_stream(user_prompt):
        role = getattr(stream, "role", None)
        
        if role == ChatRole.FUNCTION:
            # For JSON path, we don't need to render tool panels; ignore.
            try:
                await stream.message()
            except Exception:
                pass
            continue

        # For ANY other role, buffer first, then decide
        tmp_tokens = []
        async for token in stream:
            if token:
                tmp_tokens.append(token)
        msg = await stream.message()
        
        has_tools = bool(
            getattr(msg, "tool_calls", None) or
            getattr(msg, "function_call", None)  # older field, just in case
        )

        # Heuristic fallback: if the buffered text contains a function tag, treat as tool call
        raw = "".join(tmp_tokens)
        looks_like_func_tag = ("<|functions." in raw) or ("<|function." in raw)

        if not (has_tools or looks_like_func_tag):
            assistant_text_parts.extend(tmp_tokens)
        # else: drop tmp_tokens (tool panels will be represented by subsequent FUNCTION turns)

    content = "".join(assistant_text_parts)
    payload = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(payload, status_code=200)


async def openai_stream_generator(request_like: ChatRequest, model: str) -> Iterable[str]:
    """Stream assistant tokens from OpenAI Chat Completions when images are present.

    We convert any input/image parts to OpenAI-compatible content, call with stream=True,
    and yield SSE chunks in OpenAI Chat Completions format.
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("openai_stream_generator: ERROR - OPENAI_API_KEY is not configured", flush=True)
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)

    temperature = request_like.temperature
    max_tokens = request_like.max_tokens

    # Reuse the same normalization used in do_json_response
    msgs: List[ChatMessage] = request_like.messages
    messages_payload: list[dict[str, Any]] = []
    for m in msgs:
        content = m.content
        if isinstance(content, list):
            normalized: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("input_image", "image"):
                    img = part.get("image") or part.get("data") or part.get("b64")
                    mime = None
                    if isinstance(img, dict):
                        b64 = img.get("data") or img.get("b64") or img.get("base64")
                        mime = img.get("mime_type") or img.get("media_type") or img.get("type") or "image/png"
                        if isinstance(img.get("url"), str):
                            normalized.append({"type": "image_url", "image_url": {"url": img["url"], "detail": "high"}})
                        elif isinstance(b64, str):
                            url = f"data:{mime};base64,{b64}"
                            normalized.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                    elif isinstance(img, str):
                        if img.startswith("data:"):
                            url = img
                        else:
                            url = f"data:{mime or 'image/png'};base64,{img}"
                        normalized.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                else:
                    normalized.append(part if isinstance(part, dict) else part)  # pass-through
            content = normalized
        messages_payload.append({"role": m.role, "content": content})

    # Append request.input parts if present
    if request_like.input:
        from .utils import _owui_input_to_image_parts
        img_parts = _owui_input_to_image_parts(request_like.input)  # type: ignore[arg-type]
        if messages_payload and messages_payload[-1]["role"] == "user":
            last = messages_payload[-1]
            if isinstance(last["content"], list):
                last["content"].extend(img_parts)
            else:
                last["content"] = [{"type": "text", "text": str(last["content"])}] + img_parts
        else:
            messages_payload.append({"role": "user", "content": img_parts})

    # Ensure there is a final user instruction text if message is image-only
    DEFAULT_IMAGE_PROMPT = "Describe the attached image in detail."
    if not messages_payload:
        messages_payload.append({"role": "user", "content": [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]})
    else:
        last = messages_payload[-1]
        if last.get("role") != "user":
            messages_payload.append({"role": "user", "content": [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]})
        else:
            content = last.get("content")
            if isinstance(content, list):
                has_text = any(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) and p["text"].strip() for p in content)
                has_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
                if has_image and not has_text:
                    content.insert(0, {"type": "text", "text": DEFAULT_IMAGE_PROMPT})
            elif isinstance(content, str):
                if not content.strip():
                    last["content"] = [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]
            else:
                last["content"] = [{"type": "text", "text": DEFAULT_IMAGE_PROMPT}]

    # Convert any lingering Pydantic models to plain Python types
    from pydantic import BaseModel as _BM
    def to_plain(obj: Any) -> Any:
        if isinstance(obj, _BM):
            return {k: to_plain(v) for k, v in obj.model_dump().items()}
        if isinstance(obj, dict):
            return {k: to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_plain(v) for v in obj]
        return obj
    messages_payload = to_plain(messages_payload)

    # Ensure we always send a role header and stream deltas
    model_name = model or DEFAULT_MODEL
    print(f"openai_stream_generator: starting stream with model={model_name} msgs={len(messages_payload)}", flush=True)
    
    # Send role header first
    role_chunk = role_header_chunk(model_name)
    print(f"openai_stream_generator: sending role header: {role_chunk[:100]}...", flush=True)
    yield role_chunk

    try:
        print(f"openai_stream_generator: calling OpenAI streaming API", flush=True)
        with client.chat.completions.stream(
            model=model_name,
            messages=messages_payload,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            chunk_count = 0
            text_chunk_count = 0
            for event in stream:
                chunk_count += 1
                try:
                    # Handle OpenAI's new streaming event format
                    if hasattr(event, 'type') and event.type == 'content.delta':
                        # This is a ContentDeltaEvent
                        if hasattr(event, 'delta') and hasattr(event.delta, 'content'):
                            text = event.delta.content
                            if text:
                                text_chunk_count += 1
                                chunk = delta_chunk_raw(text, model_name)
                                if text_chunk_count <= 3:  # Log first few text chunks
                                    print(f"openai_stream_generator: text chunk {text_chunk_count}: '{text}' -> {chunk[:100]}...", flush=True)
                                yield chunk
                        elif chunk_count <= 5:
                            print(f"openai_stream_generator: chunk {chunk_count}: ContentDeltaEvent but no delta.content", flush=True)
                    elif hasattr(event, 'type') and event.type == 'chunk':
                        # This is a ChunkEvent - try to extract from the raw chunk
                        if hasattr(event, 'chunk'):
                            chunk_obj = event.chunk
                            if hasattr(chunk_obj, 'choices') and chunk_obj.choices:
                                choice = chunk_obj.choices[0]
                                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                                    text = choice.delta.content
                                    if text:
                                        text_chunk_count += 1
                                        chunk = delta_chunk_raw(text, model_name)
                                        if text_chunk_count <= 3:
                                            print(f"openai_stream_generator: text chunk {text_chunk_count}: '{text}' -> {chunk[:100]}...", flush=True)
                                        yield chunk
                        elif chunk_count <= 5:
                            print(f"openai_stream_generator: chunk {chunk_count}: ChunkEvent but no extractable content", flush=True)
                    elif chunk_count <= 5:
                        event_type = getattr(event, 'type', 'unknown')
                        print(f"openai_stream_generator: chunk {chunk_count}: unhandled event type {event_type}", flush=True)
                except Exception as e:
                    if chunk_count <= 5:
                        print(f"openai_stream_generator: chunk {chunk_count}: error {e}", flush=True)
                    continue
            print(f"openai_stream_generator: stream completed, total chunks: {chunk_count}", flush=True)
    except Exception as e:
        _log.exception("openai_stream_generator: OpenAI streaming error")
        print(f"openai_stream_generator: error: {e}", flush=True)
        raise HTTPException(status_code=502, detail=f"Upstream OpenAI stream error: {e}")

    print("openai_stream_generator: sending final stop chunk", flush=True)
    yield final_stop_chunk(model_name)
    yield "data: [DONE]\n\n"


async def do_openai_stream_response(request_like: ChatRequest, model: str) -> StreamingResponse:
    """Return a StreamingResponse that relays OpenAI streaming chunks for image requests."""
    return StreamingResponse(openai_stream_generator(request_like, model), media_type="text/event-stream")
