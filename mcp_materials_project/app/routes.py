from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request, UploadFile
import logging
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest, ChatMessage
from .core import do_stream_response, do_json_response, do_openai_stream_response
from dotenv import load_dotenv
from typing import Any, List

APP_TITLE = "MPKani OpenWebUI Integration"
APP_VERSION = "1.1.6"
DEFAULT_MODEL = "gpt-4.1"

load_dotenv()

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
_log = logging.getLogger(__name__)

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
async def chat_completions(request: ChatRequest, raw: Request):
    # Support multipart/form-data (OpenAI SDK style for binary images)
    content_type = raw.headers.get("content-type", "")
    _log.info("/v1/chat/completions: content-type=%s stream=%s model=%s msgs=%s input_present=%s",
              content_type, getattr(request, "stream", None), getattr(request, "model", None),
              len(request.messages or []), bool(getattr(request, "input", None)))
    print(f"/v1/chat/completions: content-type={content_type} stream={getattr(request, 'stream', None)} model={getattr(request, 'model', None)} msgs={len(request.messages or [])} input_present={bool(getattr(request, 'input', None))}", flush=True)
    if "multipart/form-data" in content_type:
        try:
            import json as _json
            import base64 as _base64
            form = await raw.form()

            # Extract messages (JSON string) if provided
            messages_field = form.get("messages")
            messages = request.messages
            if isinstance(messages_field, str):
                try:
                    parsed = _json.loads(messages_field)
                    messages = parsed if isinstance(parsed, list) else messages
                except Exception:
                    _log.warning("/v1/chat/completions: failed to parse multipart messages JSON")
                    print("/v1/chat/completions: failed to parse multipart messages JSON", flush=True)

            # Extract model/params (prefer form values if present)
            model = (form.get("model") or request.model) or DEFAULT_MODEL
            temperature = request.temperature
            if form.get("temperature") is not None:
                try:
                    temperature = float(form.get("temperature"))
                except Exception:
                    _log.warning("/v1/chat/completions: invalid temperature in multipart; using body/default")
                    print("/v1/chat/completions: invalid temperature in multipart; using body/default", flush=True)
            max_tokens = request.max_tokens
            if form.get("max_tokens") is not None:
                try:
                    max_tokens = int(form.get("max_tokens"))
                except Exception:
                    _log.warning("/v1/chat/completions: invalid max_tokens in multipart; using body/default")
                    print("/v1/chat/completions: invalid max_tokens in multipart; using body/default", flush=True)
            stream = False
            if form.get("stream") is not None:
                stream_val = str(form.get("stream")).lower()
                stream = stream_val in ("1", "true", "yes", "on")

            # Build input list from any uploaded files
            input_list: list[dict] = []
            for key, value in form.multi_items():
                if isinstance(value, UploadFile):
                    file_bytes = await value.read()
                    if not file_bytes:
                        _log.warning("/v1/chat/completions: empty upload for field '%s'", key)
                        print(f"/v1/chat/completions: empty upload for field '{key}'", flush=True)
                        continue
                    mime = value.content_type or "image/png"
                    b64 = _base64.b64encode(file_bytes).decode("ascii")
                    data_url = f"data:{mime};base64,{b64}"
                    input_list.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
            _log.info("/v1/chat/completions: multipart parsed files=%s, messages=%s", len(input_list), len(messages or []))
            print(f"/v1/chat/completions: multipart parsed files={len(input_list)} messages={len(messages or [])}", flush=True)

            # Construct a new ChatRequest merging multipart data with JSON body
            merged = ChatRequest(
                messages=messages,
                stream=stream,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                input=input_list or request.input,
            )

            # If streaming requested and images present → OpenAI streaming; else JSON pass-through
            has_imgs_multipart = bool(merged.input) or any(isinstance(m.content, list) for m in merged.messages)
            if merged.stream and has_imgs_multipart:
                _log.info("/v1/chat/completions: OpenAI streaming path selected (multipart images present)")
                print("/v1/chat/completions: OpenAI streaming path selected (multipart images present)", flush=True)
                return await do_openai_stream_response(merged, merged.model or DEFAULT_MODEL)

            _log.info("/v1/chat/completions: forwarding multipart to JSON path model=%s msgs=%s input=%s",
                      merged.model, len(merged.messages or []), len(merged.input or []))
            print(f"/v1/chat/completions: forwarding multipart to JSON path model={merged.model} msgs={len(merged.messages or [])} input={len(merged.input or []) if merged.input else 0}", flush=True)
            return await do_json_response(merged, merged.model or DEFAULT_MODEL)
        except Exception as e:
            _log.exception("/v1/chat/completions: multipart handling error")
            raise HTTPException(status_code=400, detail=f"Malformed multipart request: {e}")
    # Allow image-only requests: if messages are empty but `input` exists, accept.
    if not request.messages and not request.input:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Try to discover image data in alternative fields when input is missing
    extra_input_parts: list[dict] = []
    try:
        body = await raw.json()
        if isinstance(body, dict):
            print(f"/v1/chat/completions: json keys={list(body.keys())}", flush=True)
            candidate_keys = ["images", "image", "attachments", "files", "file"]
            for key in candidate_keys:
                val = body.get(key)
                if not val:
                    continue
                items = val if isinstance(val, list) else [val]
                for it in items:
                    try:
                        if isinstance(it, str):
                            if it.startswith("data:") or it.startswith("http"):
                                extra_input_parts.append({"type": "image_url", "image_url": {"url": it, "detail": "high"}})
                        elif isinstance(it, dict):
                            url = it.get("url") if isinstance(it.get("url"), str) else None
                            if url:
                                extra_input_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                            else:
                                b64 = it.get("b64") or it.get("data") or it.get("base64")
                                mime = it.get("mime_type") or it.get("media_type") or it.get("type") or "image/png"
                                if isinstance(b64, str):
                                    extra_input_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}})
                    except Exception:
                        pass
    except Exception:
        pass

    if extra_input_parts and not request.input:
        merged = ChatRequest(
            messages=request.messages,
            stream=False,
            model=request.model or DEFAULT_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            input=extra_input_parts,
        )
        print(f"/v1/chat/completions: merged extra image parts count={len(extra_input_parts)}", flush=True)
        return await do_json_response(merged, merged.model or DEFAULT_MODEL)

    # Only enforce 'last message must be user' when there are no images provided.
    has_input = bool(request.input)
    has_image = False
    if not has_input and request.messages:
        # Check ONLY the last (current) user message for images, not the entire conversation
        parts_summary: list[str] = []
        last_message = request.messages[-1]
        if isinstance(last_message.content, list):
            for part in last_message.content:
                part_dict = part if isinstance(part, dict) else None
                if part_dict is None:
                    try:
                        from pydantic import BaseModel as _BM
                        if isinstance(part, _BM):
                            part_dict = part.model_dump()
                    except Exception:
                        part_dict = None
                if isinstance(part_dict, dict):
                    t = part_dict.get("type")
                    if t in ("image_url", "input_image", "image"):
                        has_image = True
                        parts_summary.append(f"{last_message.role}:{t}")
                        break
                    if any(k in part_dict for k in ("image_url", "b64", "data", "url")):
                        has_image = True
                        parts_summary.append(f"{last_message.role}:embedded_image")
                        break
        if parts_summary:
            print(f"/v1/chat/completions: detected image parts in CURRENT message {parts_summary}", flush=True)
        else:
            print("/v1/chat/completions: no image-like parts detected in CURRENT message", flush=True)

    if request.messages and not has_input and not has_image:
        last_message = request.messages[-1]
        if last_message.role.lower() != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")

    model = request.model or DEFAULT_MODEL

    # If streaming was requested but there are images present, override to JSON path
    # so we can forward to OpenAI with binary/base64 intact.
    if request.stream:
        # If input or image parts exist, prefer JSON path; else stream as usual
        if not has_input and not has_image:
            _log.info("/v1/chat/completions: streaming path selected (no images)")
            print("/v1/chat/completions: streaming path selected (no images)", flush=True)
            return await do_stream_response(request.messages, model)
        # Images present and stream requested → OpenAI streaming
        _log.info("/v1/chat/completions: OpenAI streaming path selected (images present)")
        print("/v1/chat/completions: OpenAI streaming path selected (images present)", flush=True)
        return await do_openai_stream_response(request, model)

    # Non-streaming or image-present: return a standard JSON object; pass full request
    _log.info("/v1/chat/completions: forwarding JSON body to JSON path model=%s msgs=%s input=%s",
              model, len(request.messages or []), len(request.input or [] if request.input else []))
    print(f"/v1/chat/completions: forwarding JSON body to JSON path model={model} msgs={len(request.messages or [])} input={len(request.input or []) if request.input else 0}", flush=True)
    return await do_json_response(request, model)  # type: ignore[arg-type]


@app.post("/v1/responses")
async def responses(raw: Request):
    """Minimal OpenAI Responses API compatibility for OpenWebUI.

    Accepts bodies like:
      { model, input: string|array(parts), stream?, temperature?, max_tokens?, attachments?/images?/files? }
    Converts to a Chat Completions-style request and forwards to JSON handler.
    """
    try:
        body = await raw.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    model = body.get("model") or DEFAULT_MODEL
    stream_req = bool(body.get("stream"))
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")

    print(f"/v1/responses: keys={list(body.keys())} stream={stream_req} model={model}", flush=True)

    # Normalize messages (if provided)
    raw_messages = body.get("messages")
    norm_messages: List[ChatMessage] = []
    if isinstance(raw_messages, list):
        for m in raw_messages:
            if isinstance(m, dict):
                try:
                    norm_messages.append(ChatMessage.model_validate(m))
                except Exception:
                    pass

    # Collect additional user parts from input and attachment-like fields
    extra_parts: List[dict] = []
    def add_text(t: str):
        extra_parts.append({"type": "text", "text": t})
    def add_image_url(url: str, detail: str = "high"):
        extra_parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})

    input_field = body.get("input")
    if isinstance(input_field, str) and input_field.strip():
        add_text(input_field)
    elif isinstance(input_field, list):
        for part in input_field:
            if isinstance(part, dict):
                extra_parts.append(part)
            elif isinstance(part, str):
                add_text(part)

    for key in ("images", "image", "attachments", "files", "file"):
        val = body.get(key)
        if not val:
            continue
        items = val if isinstance(val, list) else [val]
        for it in items:
            try:
                if isinstance(it, str):
                    if it.startswith("data:") or it.startswith("http"):
                        add_image_url(it)
                elif isinstance(it, dict):
                    url = it.get("url") if isinstance(it.get("url"), str) else None
                    if url:
                        add_image_url(url)
                    else:
                        b64 = it.get("b64") or it.get("data") or it.get("base64")
                        mime = it.get("mime_type") or it.get("media_type") or it.get("type") or "image/png"
                        if isinstance(b64, str):
                            add_image_url(f"data:{mime};base64,{b64}")
            except Exception:
                pass

    # Merge extra parts into last user message or create one
    has_images = any(isinstance(p, dict) and p.get("type") == "image_url" for p in extra_parts)
    if extra_parts:
        if norm_messages and (norm_messages[-1].role or "").lower() == "user":
            last = norm_messages[-1]
            if isinstance(last.content, list):
                last.content = last.content + extra_parts  # type: ignore[operator]
            elif isinstance(last.content, str):
                last.content = [{"type": "text", "text": last.content}] + extra_parts  # type: ignore[assignment]
            else:
                last.content = extra_parts  # type: ignore[assignment]
        else:
            norm_messages.append(ChatMessage(role="user", content=extra_parts))

    # If we didn't see images in extra parts, scan normalized messages for image parts
    if not has_images:
        for m in norm_messages:
            content = m.content
            if isinstance(content, list):
                for part in content:
                    part_dict = part if isinstance(part, dict) else None
                    if part_dict is None:
                        try:
                            from pydantic import BaseModel as _BM
                            if isinstance(part, _BM):
                                part_dict = part.model_dump()
                        except Exception:
                            part_dict = None
                    if isinstance(part_dict, dict):
                        ptype = part_dict.get("type")
                        if ptype in ("image_url", "input_image", "image"):
                            has_images = True
                            break
                        if any(k in part_dict for k in ("image_url", "b64", "data", "url")):
                            has_images = True
                            break
            if has_images:
                break

    # If still no messages, synthesize a simple user message
    if not norm_messages:
        norm_messages.append(ChatMessage(role="user", content=[{"type": "text", "text": "Describe this image"}]))

    # Route: images → JSON (OpenAI pass-through); text-only → stream or JSON via Kani
    if has_images:
        if stream_req:
            # Stream directly from OpenAI for images
            print(f"/v1/responses: OpenAI streaming (images present)", flush=True)
            return await do_openai_stream_response(ChatRequest(
                messages=norm_messages,
                stream=True,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ), model)
        req = ChatRequest(
            messages=norm_messages,
            stream=False,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            input=None,
        )
        print(f"/v1/responses: forwarding to JSON path with parts={len(extra_parts)} (images present)", flush=True)
        return await do_json_response(req, model)
    else:
        # No images: preserve streaming behavior
        if stream_req:
            print(f"/v1/responses: streaming via Kani (text-only)", flush=True)
            return await do_stream_response(norm_messages, model)
        req = ChatRequest(
            messages=norm_messages,
            stream=False,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f"/v1/responses: forwarding to JSON path (text-only)", flush=True)
        return await do_json_response(req, model)
