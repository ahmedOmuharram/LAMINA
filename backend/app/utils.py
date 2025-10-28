from __future__ import annotations
import asyncio
import json
import time
import re
from typing import Any, List
from .schemas import ChatMessage
from kani import ChatMessage as KChatMessage

def tool_start_event(tool_name: str, tool_id: str, model_name: str, tool_input: Any = None) -> str:
    """Send a structured tool start event."""
    tool_call_data = {
        "id": tool_id,
        "name": tool_name,
        "status": "started"
    }
    if tool_input is not None:
        tool_call_data["input"] = tool_input
    
    payload = {
        "id": f"chatcmpl-{int(time.time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_call": tool_call_data
            },
            "finish_reason": None
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

def tool_end_event(tool_name: str, tool_id: str, duration: float, output: Any, model_name: str, tool_input: Any = None) -> str:
    """Send a structured tool end event."""
    tool_call_data = {
        "id": tool_id,
        "name": tool_name,
        "status": "completed",
        "duration": duration,
        "output": output
    }
    if tool_input is not None:
        tool_call_data["input"] = tool_input
    
    payload = {
        "id": f"chatcmpl-{int(time.time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_call": tool_call_data
            },
            "finish_reason": None
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

def image_event(image_url: str, metadata: dict, model_name: str) -> str:
    """Send a structured image event."""
    payload = {
        "id": f"chatcmpl-{int(time.time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "image": {
                    "url": image_url,
                    "metadata": metadata
                }
            },
            "finish_reason": None
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

def analysis_event(analysis: str, model_name: str) -> str:
    """Send a structured analysis event."""
    payload = {
        "id": f"chatcmpl-{int(time.time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "analysis": {
                    "content": analysis
                }
            },
            "finish_reason": None
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

def usage_event(prompt_tokens: int, completion_tokens: int, total_tokens: int, model_name: str) -> str:
    """Send a usage/token count event."""
    payload = {
        "id": f"chatcmpl-{int(time.time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            },
            "finish_reason": None
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

def linkify_mp_numbers(text: str) -> str:
    """Auto-link mp-<digits> to Materials Project, excluding code spans."""
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
    return "".join(parts)

def role_header_chunk(model_name: str) -> str:
    """SSE chunk that announces the assistant role for a turn."""
    payload = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time()*10_000)}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

def delta_chunk_raw(text: str, model_name: str) -> str:
    """SSE delta without additional processing (text already prepared)."""
    payload = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time() * 10_000)}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

def final_stop_chunk(model_name: str) -> str:
    """Final SSE 'stop' event."""
    stop = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(stop)}\n\n"

def to_kani_history(api_messages: List[ChatMessage]) -> list[KChatMessage]:
    """Convert API messages to Kani history, excluding the last user message.
    
    Preserves multimodal content (text + images) for vision-capable models.
    """
    out: list[KChatMessage] = []
    for m in api_messages:
        role = (m.role or "").lower()
        # Preserve raw content for multimodal support (Kani/OpenAI handles it)
        content = m.content
        if role == "system":
            # System messages are always text
            text_content = extract_text_from_message_content(content) if not isinstance(content, str) else content
            out.append(KChatMessage.system(text_content))
        elif role == "user":
            # User messages can be multimodal (text + images)
            out.append(KChatMessage.user(content))
        elif role == "assistant":
            # Assistant messages are always text
            text_content = extract_text_from_message_content(content) if not isinstance(content, str) else content
            out.append(KChatMessage.assistant(text_content))
        elif role == "function":
            # Function results are always text
            text_content = extract_text_from_message_content(content) if not isinstance(content, str) else content
            out.append(KChatMessage.function(name=None, content=text_content))
        # else: ignore unknown roles
    return out

def extract_text_from_message_content(content: Any) -> str:
    """Convert multimodal message content into a plain-text prompt.

    Supports OpenAI-style parts: [{type: 'text'|'image_url', ...}]. Image URLs
    are summarized as a bracketed note appended to the text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        image_descriptors: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            # Text part
            if ptype == "text":
                text_val = part.get("text")
                if isinstance(text_val, str):
                    texts.append(text_val)
                continue

            # URL image part
            if ptype == "image_url":
                image_field = part.get("image_url")
                if isinstance(image_field, str):
                    desc = _summarize_image_url(image_field)
                    image_descriptors.append(desc)
                elif isinstance(image_field, dict):
                    url_val = image_field.get("url")
                    if isinstance(url_val, str):
                        desc = _summarize_image_url(url_val)
                        image_descriptors.append(desc)
                continue

            # Base64 image part variants used by OpenWebUI / OpenAI
            if ptype in ("input_image", "image"):
                image_obj = part.get("image") or part.get("data") or part.get("b64")
                mime = None
                data_str = None
                if isinstance(image_obj, dict):
                    # Common keys: data|b64|base64, mime_type|media_type
                    data_str = image_obj.get("data") or image_obj.get("b64") or image_obj.get("base64")
                    mime = image_obj.get("mime_type") or image_obj.get("media_type") or image_obj.get("type")
                    if isinstance(image_obj.get("url"), str):
                        # Sometimes it's provided as data URL in url field
                        image_descriptors.append(_summarize_image_url(image_obj.get("url")))
                        continue
                elif isinstance(image_obj, str):
                    data_str = image_obj

                if isinstance(data_str, str):
                    image_descriptors.append(_summarize_base64_image(data_str, mime))
                continue

            # Generic fallback: detect top-level base64 or url keys
            if "image_url" in part and isinstance(part["image_url"], str):
                image_descriptors.append(_summarize_image_url(part["image_url"]))
            elif "url" in part and isinstance(part["url"], str):
                image_descriptors.append(_summarize_image_url(part["url"]))
            elif "b64" in part and isinstance(part["b64"], str):
                image_descriptors.append(_summarize_base64_image(part["b64"], part.get("mime_type")))
            elif "data" in part and isinstance(part["data"], str):
                image_descriptors.append(_summarize_base64_image(part["data"], part.get("mime_type")))

        images_note = "" if not image_descriptors else "\n\n[Images: " + ", ".join(image_descriptors) + "]"
        combined = "\n\n".join(texts) + images_note
        return combined.strip()

    # Fallback: JSON dump other structures
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _summarize_image_url(url: str) -> str:
    """Summarize an image URL for text extraction."""
    if url.startswith("data:"):
        # data URL; try to extract mime
        try:
            header = url.split(",", 1)[0]
            # data:image/png;base64
            if ";" in header:
                mime = header.split(":", 1)[1].split(";", 1)[0]
            else:
                mime = header.split(":", 1)[1]
            return f"base64 ({mime})"
        except Exception:
            return "base64 (data URL)"
    return url


def _summarize_base64_image(data_str: str, mime: Any = None) -> str:
    """Summarize a base64 image for text extraction."""
    # Avoid including raw data; just a short descriptor
    mime_str = f"{mime}" if isinstance(mime, str) else None
    if data_str.startswith("data:"):
        return _summarize_image_url(data_str)
    # Heuristic: show length and mime
    length = len(data_str)
    if mime_str:
        return f"base64 ({mime_str}, {length} chars)"
    return f"base64 ({length} chars)"
