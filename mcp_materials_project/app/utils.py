from __future__ import annotations
import asyncio
import json
import time
import re
from typing import Any, List
from .schemas import ChatMessage
from kani import ChatMessage as KChatMessage

def details_block(summary: str, body_md: str = "", open_: bool = False) -> str:
    """Produce a collapsible <details> block."""
    open_attr = " open" if open_ else ""
    body_md = (body_md or "").strip()
    inner = f"{body_md}\n" if body_md else ""
    # blank lines around details so it doesn't glue to text
    return f"\n<details{open_attr}>\n  <summary>{summary}</summary>\n\n{inner}</details>\n\n"

def tool_panel_done(tool_name: str, duration: float, logs_md: str = "") -> str:
    """Standardized 'tool finished' panel."""
    return details_block(f"✅ {tool_name} — done ({duration:.2f}s)", logs_md, open_=False)

def tool_panel_general(tool_name: str, logs_md: str = "") -> str:
    """Standardized 'tool finished' panel."""
    return details_block(f"{tool_name}", logs_md, open_=False)

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

def delta_chunk(text: str, model_name: str) -> str:
    """SSE chunk that streams assistant content, with mp-linkification (stateless)."""
    return delta_chunk_raw(linkify_mp_numbers(text or ""), model_name)

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
    """Convert API messages to Kani history, excluding the last user message."""
    out: list[KChatMessage] = []
    for m in api_messages:
        role = (m.role or "").lower()
        content = extract_text_from_message_content(m.content)
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

def pretty_print_tool_output(obj: Any) -> str:
    """Best-effort JSON pretty-printer for tool output, for logs panel."""
    # Handle different types of tool outputs
    if isinstance(obj, dict):
        display_obj = obj.copy()
    elif hasattr(obj, '__dict__'):
        # Convert object with attributes to dict
        display_obj = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        # Fallback for other types
        display_obj = {"output": str(obj)}

    # Special handling for image data - avoid streaming massive base64 strings
    if isinstance(display_obj, dict):
        display_obj = display_obj.copy()
        
        # Handle _image_data specially (internal field for image storage)
        if "_image_data" in display_obj:
            image_data = display_obj.pop("_image_data")
            if isinstance(image_data, str) and len(image_data) > 1000:
                display_obj["_image_data_summary"] = f"<base64 image data: {len(image_data)} characters>"
        
        # Handle nested image objects
        if "image" in display_obj and isinstance(display_obj["image"], dict):
            image_info = display_obj["image"].copy()
            if "data" in image_info and isinstance(image_info["data"], str):
                data_len = len(image_info["data"])
                if data_len > 1000:  # Large base64 string
                    image_info["data"] = f"<base64 image data: {data_len} characters>"
            display_obj["image"] = image_info
        
        # Also handle direct base64 fields
        for key in ["base64_image", "image_data", "data"]:
            if key in display_obj and isinstance(display_obj[key], str) and len(display_obj[key]) > 1000:
                display_obj[key] = f"<base64 data: {len(display_obj[key])} characters>"

    # Pretty print the dictionary
    try:
        pretty = json.dumps(display_obj, indent=2, ensure_ascii=False, default=str)
        return f"```json\n{pretty}\n```"
    except Exception:
        return f"```\n{str(obj)}\n```"


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
    # Avoid including raw data; just a short descriptor
    mime_str = f"{mime}" if isinstance(mime, str) else None
    if data_str.startswith("data:"):
        return _summarize_image_url(data_str)
    # Heuristic: show length and mime
    length = len(data_str)
    if mime_str:
        return f"base64 ({mime_str}, {length} chars)"
    return f"base64 ({length} chars)"

def _owui_input_to_image_parts(input_field: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenWebUI `request.input` array into Chat Completions image_url parts."""
    parts: list[dict[str, Any]] = []
    for item in input_field or []:
        # Recognized shapes:
        # 1) {type: 'input_image', image: {data|b64|base64, mime_type|media_type|type, url?}}
        # 2) {type: 'image_url', image_url: string|{url, detail?}}
        # 3) {image|data|b64: string|{...}} (missing type)
        mime: str | None = None

        # Case 2: direct image_url
        if isinstance(item.get("image_url"), str):
            parts.append({"type": "image_url", "image_url": {"url": item["image_url"], "detail": "high"}})
            continue
        if isinstance(item.get("image_url"), dict):
            iu = item["image_url"]
            url = iu.get("url") if isinstance(iu.get("url"), str) else None
            if url:
                detail = iu.get("detail") if isinstance(iu.get("detail"), str) else "high"
                parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                continue

        # Case 1/3: embedded image data
        img = item.get("image") or item.get("data") or item.get("b64")
        data: str | None = None
        if isinstance(img, dict):
            data = img.get("data") or img.get("b64") or img.get("base64")
            mime = img.get("mime_type") or img.get("media_type") or img.get("type") or "image/png"
            if isinstance(img.get("url"), str):
                parts.append({"type": "image_url", "image_url": {"url": img["url"], "detail": "high"}})
                continue
        elif isinstance(img, str):
            data = img

        if isinstance(data, str):
            url = data if data.startswith("data:") else f"data:{mime or 'image/png'};base64,{data}"
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})

    return parts
