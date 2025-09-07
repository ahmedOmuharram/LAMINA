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

def pretty_print_tool_output(obj: Any) -> str:
    """Best-effort JSON pretty-printer for tool output, for logs panel."""
    try:
        display_obj = dict(obj)  # may raise
    except Exception:
        try:
            display_obj = {k: obj[k] for k in obj} if isinstance(obj, dict) else {"record": str(obj)}
        except Exception:
            return f"```\n{str(obj)}\n```"

    res_val = display_obj.get("result")
    if isinstance(res_val, str):
        try:
            display_obj["result"] = json.loads(res_val)
        except Exception:
            pass

    pretty = json.dumps(display_obj, indent=2, ensure_ascii=False)
    return f"```json\n{pretty}\n```"