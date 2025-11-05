from __future__ import annotations
from typing import List, Iterable, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import logging
from .schemas import ChatMessage
from .utils import role_header_chunk, final_stop_chunk, delta_chunk_raw, usage_event
from .stream_state import StreamState
from ..kani_client import MPKani
from kani import ChatRole
from .utils import to_kani_history, extract_text_from_message_content
import os
try:
    import tiktoken
except ImportError:
    tiktoken = None

DEFAULT_MODEL = "gpt-4o-mini"
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
    # Preserve multimodal content (text + images) for the last message
    user_content = messages[-1].content
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
        # Pass raw content (supports both str and multimodal list)
        stream_iterator = kani_instance.full_round_stream(user_content)
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

            # For ANY other role, stream tokens immediately
            yield role_header_chunk(model_name)

            # Stream tokens as they arrive for real-time display
            async for token in stream:
                check_cancelled()
                if token is not None:
                    # Check if this looks like a function call tag
                    if "<|functions." in token or "<|function." in token:
                        # Buffer function tags - don't emit them
                        continue
                    
                    # Emit token immediately for real-time streaming
                    chunk = state.emit_stream_text(token)
                    if chunk:
                        accumulated_text += token
                        yield chunk

            # Get the final message to check for tool calls
            msg = await stream.message()
            
            has_tools = bool(
                getattr(msg, "tool_calls", None) or
                getattr(msg, "function_call", None)  # older field, just in case
            )
            
            # Register and emit tool start events if tools were called
            if has_tools:
                state.register_tool_calls(msg)
                # Emit tool start events for each registered tool
                for tc_id in list(state.tool_fifo):  # Copy to avoid modification during iteration
                    if tc_id not in state.completed_tools:
                        tool_name = state.tool_name_by_id.get(tc_id, "tool")
                        tool_input = state.tool_input_by_id.get(tc_id, None)
                        yield state.emit_tool_start(tool_name, tc_id, tool_input)

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
    """Return a StreamingResponse wrapping the SSE generator.
    
    Handles both text-only and multimodal (text + images) messages.
    """
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
