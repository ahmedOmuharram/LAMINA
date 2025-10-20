from __future__ import annotations
from typing import List, Iterable, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import logging
from .schemas import ChatMessage, ChatRequest
from .utils import role_header_chunk, final_stop_chunk, delta_chunk_raw, usage_event
from .stream_state import StreamState
from ..kani_client import MPKani
from kani import ChatRole
from .utils import to_kani_history, extract_text_from_message_content, _owui_input_to_image_parts
import time
import os
import json
import openai
from pydantic import BaseModel
try:
    import tiktoken
except ImportError:
    tiktoken = None

DEFAULT_MODEL = "gpt-4.1"
_log = logging.getLogger(__name__)

def count_tokens(text: str, model: str) -> int:
    """Count tokens using tiktoken for the given model."""
    if not tiktoken or not text:
        # Fallback to rough estimate if tiktoken not available
        return len(text.split()) if text else 0
    
    try:
        # Map model names to tiktoken encodings
        if "gpt-4" in model.lower() or "gpt-4o" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Default to cl100k_base (used by gpt-4, gpt-3.5-turbo, gpt-4o)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}", flush=True)
        # Fallback to word count estimate (roughly 1.3 tokens per word for English)
        return int(len(text.split()) * 1.3) if text else 0

def count_message_tokens(messages: List[ChatMessage], model: str) -> int:
    """Count tokens for a list of messages including formatting overhead."""
    if not tiktoken:
        return sum(len(extract_text_from_message_content(msg.content).split()) for msg in messages)
    
    try:
        # Map model names to tiktoken encodings
        if "gpt-4" in model.lower() or "gpt-4o" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name, the role is omitted
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.role))
            
            # Get text content
            content = extract_text_from_message_content(message.content)
            num_tokens += len(encoding.encode(content))
        
        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        
        return num_tokens
    except Exception as e:
        print(f"Error counting message tokens: {e}", flush=True)
        # Fallback
        return sum(len(extract_text_from_message_content(msg.content).split()) * 1.3 for msg in messages)

async def sse_generator(messages: List[ChatMessage], model: str, request: Any = None) -> Iterable[str]:
    """Produce SSE chunks for a single chat round."""
    if not messages:
        _log.error("sse_generator: no messages provided")
        print("sse_generator: no messages provided", flush=True)
        raise HTTPException(status_code=400, detail="No messages provided")
    if messages[-1].role.lower() != "user":
        _log.error("sse_generator: last message must be from user (got role=%s)", messages[-1].role)
        print(f"sse_generator: last message must be from user (got role={messages[-1].role})", flush=True)
        raise HTTPException(status_code=400, detail="Last message must be from user")
    
    # Shared cancellation flag
    cancelled = {"value": False}
    
    # Helper to check if client disconnected
    async def is_disconnected() -> bool:
        if request is None:
            return False
        try:
            disconnected = await request.is_disconnected()
            if disconnected:
                cancelled["value"] = True
            return disconnected
        except Exception:
            return False
    
    # Background task to periodically check for disconnection
    import asyncio
    async def monitor_connection():
        """Continuously check if client is still connected."""
        while not cancelled["value"]:
            try:
                if await is_disconnected():
                    print("monitor_connection: Client disconnected!", flush=True)
                    cancelled["value"] = True
                    break
                await asyncio.sleep(0.5)  # Check every 500ms
            except asyncio.CancelledError:
                print("monitor_connection: Monitor cancelled", flush=True)
                cancelled["value"] = True
                break
            except Exception as e:
                print(f"monitor_connection: Error - {e}", flush=True)
                break
    
    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_connection())
    
    # Wrapper to check cancellation flag
    def check_cancelled():
        if cancelled["value"]:
            raise asyncio.CancelledError("Client disconnected")

    model_name = model or DEFAULT_MODEL
    user_prompt = extract_text_from_message_content(messages[-1].content)
    prior_api_msgs = messages[:-1]
    kani_chat_history = to_kani_history(prior_api_msgs)

    _log.info("sse_generator: model=%s messages=%s prior=%s", model_name, 1, len(prior_api_msgs))
    print(f"sse_generator: model={model_name} messages=1 prior={len(prior_api_msgs)}", flush=True)
    kani_instance = MPKani(model=model_name, chat_history=kani_chat_history)
    state = StreamState(model_name=model_name)
    state._kani_instance = kani_instance  # Pass kani instance for image access
    
    # Track tokens
    prompt_tokens = 0
    completion_tokens = 0
    accumulated_text = ""
    
    # Count prompt tokens using proper message formatting
    prompt_tokens = count_message_tokens(messages, model_name)

    try:
        stream_iterator = kani_instance.full_round_stream(user_prompt)
        async for stream in stream_iterator:
            # Check cancellation flag (set by background monitor)
            check_cancelled()
            
            role = getattr(stream, "role", None)

            if role == ChatRole.FUNCTION:
                check_cancelled()
                tool_msg = await stream.message()
                for chunk in state.complete_next_tool(tool_msg, kani_instance):
                    check_cancelled()
                    yield chunk
                continue

            # For ANY other role, buffer first, then decide
            yield role_header_chunk(model_name)

            tmp_tokens: list[str] = []
            async for token in stream:
                check_cancelled()
                if token is not None:
                    tmp_tokens.append(token)

            msg = await stream.message()
            
            has_tools = bool(
                getattr(msg, "tool_calls", None) or
                getattr(msg, "function_call", None)  # older field, just in case
            )
            
            # Register and emit tool start events
            if has_tools:
                state.register_tool_calls(msg)
                # Emit tool start events for each registered tool
                for tc_id in list(state.tool_fifo):  # Copy to avoid modification during iteration
                    if tc_id not in state.completed_tools:
                        tool_name = state.tool_name_by_id.get(tc_id, "tool")
                        tool_input = state.tool_input_by_id.get(tc_id, None)
                        yield state.emit_tool_start(tool_name, tc_id, tool_input)

            # Heuristic fallback: if the buffered text contains a function tag, treat as tool call
            raw = "".join(tmp_tokens)
            looks_like_func_tag = ("<|functions." in raw) or ("<|function." in raw)

            if not (has_tools or looks_like_func_tag):
                for tok in tmp_tokens:
                    chunk = state.emit_stream_text(tok)
                    if chunk:
                        accumulated_text += tok
                        yield chunk
            # else: drop the prelude & markup entirely

        # End-of-stream safety: flush anything still buffered
        for chunk in state.flush_buffer_if_any():
            yield chunk

        # Close any orphan tools with zero-duration events
        for chunk in state.close_orphan_tools():
            yield chunk

        # Use the tracked text from StreamState for accurate token counting
        completion_text = state.all_emitted_text
        completion_tokens = count_tokens(completion_text, model_name)
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"DEBUG: Token counts - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}", flush=True)
        print(f"DEBUG: Completion text length: {len(completion_text)} chars", flush=True)
        
        # Send usage information
        yield usage_event(prompt_tokens, completion_tokens, total_tokens, model_name)

        # Final stop
        yield final_stop_chunk(model_name)
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        print("sse_generator: Cancelled by client disconnection", flush=True)
        _log.info("sse_generator: Cancelled")
        # Don't re-raise, just stop gracefully
    except Exception as e:
        _log.exception("sse_generator: Error during streaming")
        print(f"sse_generator: ERROR - {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        print(f"sse_generator: Traceback:\n{traceback.format_exc()}", flush=True)
        # Try to send an error message to the user
        error_msg = f"\n\n**Error**: An error occurred during generation: {str(e)}\n\n"
        try:
            yield delta_chunk_raw(error_msg, model_name)
            yield final_stop_chunk(model_name)
            yield "data: [DONE]\n\n"
        except Exception as inner_e:
            _log.exception("sse_generator: Failed to send error message")
            print(f"sse_generator: Failed to send error message: {inner_e}", flush=True)
        raise
    finally:
        # Cancel the background monitor task
        cancelled["value"] = True
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        print("sse_generator: Cleanup complete", flush=True)

    # Clear any saved tool outputs on the kani wrapper (if present)
    try:
        kani_instance.recent_tool_outputs = []
    except Exception:
        pass

async def do_stream_response(messages: List[ChatMessage], model: str = DEFAULT_MODEL, request: Any = None) -> StreamingResponse:
    """Return a StreamingResponse wrapping the SSE generator."""
    import asyncio
    
    async def cancellable_generator():
        """Wrapper to make the generator properly cancellable."""
        generator = sse_generator(messages, model, request)
        try:
            async for chunk in generator:
                yield chunk
        except asyncio.CancelledError:
            print("do_stream_response: Stream cancelled by client", flush=True)
            _log.info("do_stream_response: Stream cancelled")
            # Try to close the generator
            if hasattr(generator, 'aclose'):
                try:
                    await generator.aclose()
                except Exception as e:
                    print(f"do_stream_response: Error closing generator: {e}", flush=True)
            raise
        except GeneratorExit:
            print("do_stream_response: Generator exit", flush=True)
            raise
    
    return StreamingResponse(
        cancellable_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

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

    try:
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
    
    except Exception as e:
        _log.exception("do_json_response[Kani]: Error during generation")
        print(f"do_json_response[Kani]: ERROR - {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        print(f"do_json_response[Kani]: Traceback:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")


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
