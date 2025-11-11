from __future__ import annotations
import asyncio
import json
import time
import re
import logging
from typing import Any, List, Dict
from .schemas import ChatMessage
from kani import ChatMessage as KChatMessage

_log = logging.getLogger(__name__)

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
            # Handle Pydantic models by converting to dict
            if hasattr(part, 'model_dump'):
                part = part.model_dump()
            elif not isinstance(part, dict):
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


def extract_images_from_content(content: Any) -> List[Dict[str, Any]]:
    """Extract all images from message content.
    
    Returns a list of image dictionaries with 'url' or 'data' keys.
    """
    if content is None or isinstance(content, str):
        return []
    
    if not isinstance(content, list):
        return []
    
    images = []
    for part in content:
        # Handle Pydantic models by converting to dict
        if hasattr(part, 'model_dump'):
            part = part.model_dump()
        elif not isinstance(part, dict):
            continue
        
        ptype = part.get("type")
        
        # Handle image_url type
        if ptype == "image_url":
            image_field = part.get("image_url")
            if isinstance(image_field, str):
                images.append({"url": image_field})
            elif isinstance(image_field, dict):
                url_val = image_field.get("url")
                if isinstance(url_val, str):
                    detail = image_field.get("detail", "high")
                    images.append({"url": url_val, "detail": detail})
        
        # Handle other image formats (base64, etc.)
        elif ptype in ("input_image", "image"):
            image_obj = part.get("image") or part.get("data") or part.get("b64")
            if isinstance(image_obj, dict):
                data_str = image_obj.get("data") or image_obj.get("b64") or image_obj.get("base64")
                mime = image_obj.get("mime_type") or image_obj.get("media_type") or "image/png"
                if isinstance(data_str, str):
                    images.append({"data": data_str, "mime": mime})
                # Check for URL in the object
                if isinstance(image_obj.get("url"), str):
                    images.append({"url": image_obj["url"]})
            elif isinstance(image_obj, str):
                images.append({"data": image_obj, "mime": "image/png"})
    
    return images


async def download_image_as_base64(url: str, timeout: int = 15) -> tuple[str, str]:
    """Download an image from URL and convert to base64.
    
    Uses multiple strategies to bypass anti-bot protections:
    - Realistic browser headers
    - Multiple user agents (tries different ones if blocked)
    - Referrer headers
    - Accept headers
    
    Args:
        url: Image URL to download
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (base64_data, mime_type)
    """
    import httpx
    import base64
    from urllib.parse import urlparse
    
    print(f"download_image_as_base64: Downloading {url[:100]}...", flush=True)
    
    # Extract domain for referrer
    parsed_url = urlparse(url)
    domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Try multiple user agents in case one is blocked
    user_agents = [
        # Chrome on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Firefox on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Safari on macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        # Chrome on macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    last_error = None
    
    for idx, user_agent in enumerate(user_agents, 1):
        try:
            # Build realistic browser headers
            headers = {
                'User-Agent': user_agent,
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': domain,
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'same-origin',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            if idx > 1:
                print(f"download_image_as_base64: Trying user agent {idx}/{len(user_agents)}...", flush=True)
            
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # Get content type
                content_type = response.headers.get('content-type', 'image/png')
                if ';' in content_type:
                    content_type = content_type.split(';')[0].strip()
                
                # Verify it's actually an image
                if not content_type.startswith('image/') and not content_type.startswith('application/'):
                    # Some servers return wrong content-type, try to infer from URL
                    url_lower = url.lower()
                    if url_lower.endswith('.jpg') or url_lower.endswith('.jpeg'):
                        content_type = 'image/jpeg'
                    elif url_lower.endswith('.png'):
                        content_type = 'image/png'
                    elif url_lower.endswith('.gif'):
                        content_type = 'image/gif'
                    elif url_lower.endswith('.webp'):
                        content_type = 'image/webp'
                    else:
                        content_type = 'image/png'  # Default fallback
                
                # Convert to base64
                image_data = response.content
                
                # Basic validation - check if we got HTML instead of an image
                if len(image_data) > 0:
                    # Check if it starts with HTML tags (common for error pages)
                    data_preview = image_data[:100].lower()
                    if b'<html' in data_preview or b'<!doctype' in data_preview:
                        raise ValueError("Received HTML instead of image (likely an error page)")
                
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                print(f"download_image_as_base64: Successfully downloaded {len(image_data)} bytes, mime: {content_type}", flush=True)
                return base64_data, content_type
                
        except httpx.HTTPStatusError as e:
            last_error = e
            status_code = e.response.status_code
            print(f"download_image_as_base64: HTTP {status_code} with user agent {idx}", flush=True)
            
            # If it's a 403/401, try next user agent
            if status_code in (403, 401) and idx < len(user_agents):
                continue
            else:
                # Other errors or last user agent - give up
                break
                
        except Exception as e:
            last_error = e
            print(f"download_image_as_base64: Error with user agent {idx}: {e}", flush=True)
            # Try next user agent
            if idx < len(user_agents):
                continue
            else:
                break
    
    # All attempts failed
    error_msg = str(last_error) if last_error else "Unknown error"
    print(f"download_image_as_base64: All download attempts failed. Last error: {error_msg}", flush=True)
    raise Exception(f"Failed to download image after {len(user_agents)} attempts: {error_msg}")


async def describe_image_with_gpt4o(image_info: Dict[str, Any], user_text: str = "") -> str:
    """Use OpenAI's GPT-4o API directly to describe an image.
    
    Args:
        image_info: Dictionary with 'url' or 'data' key for the image
        user_text: Optional text provided alongside the image
        
    Returns:
        Text description of the image
    """
    import os
    from openai import AsyncOpenAI
    
    print(f"describe_image_with_gpt4o: Starting, user_text: {user_text[:50] if user_text else '(none)'}...", flush=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("describe_image_with_gpt4o: ERROR - OpenAI API key not configured", flush=True)
        return "[Image: Unable to process - OpenAI API key not configured]"
    
    print(f"describe_image_with_gpt4o: API key found: {api_key[:10]}...", flush=True)
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Build the content for the vision request
    content_parts = []
    
    # Add user text if provided
    prompt_text = "Describe this image in detail, including:"
    prompt_text += "\n- What the image shows (objects, scenes, diagrams, charts, etc.)"
    prompt_text += "\n- Any text visible in the image (transcribe it accurately)"
    prompt_text += "\n- Key details, patterns, or important features"
    prompt_text += "\n- Scientific or technical information if present"
    prompt_text += "\n- Any other information that might be helpful"
    prompt_text += "Use external important knowledge to check if this image is accurate. For example, the text might be blurry and show different characters or numbers (6 instead of 8 for example) than what make sense. In that case, correct the text and provide the correct information."
    
    if user_text:
        prompt_text = f"User query: {user_text}\n\n{prompt_text}"
    
    content_parts.append({
        "type": "text",
        "text": prompt_text
    })
    
    # Prepare image data
    image_url_for_api = None
    
    if "url" in image_info:
        url = image_info["url"]
        print(f"describe_image_with_gpt4o: Using image URL: {url[:100]}...", flush=True)
        
        # Try URL first, but prepare to fall back to downloading
        if url.startswith("data:"):
            # It's already a data URL
            image_url_for_api = url
            print("describe_image_with_gpt4o: Image is already a data URL", flush=True)
        else:
            # Try to use URL directly first
            image_url_for_api = url
            
    elif "data" in image_info:
        # Convert base64 data to data URL if needed
        data_str = image_info["data"]
        mime = image_info.get("mime", "image/png")
        print(f"describe_image_with_gpt4o: Using base64 data, length: {len(data_str)}, mime: {mime}", flush=True)
        if not data_str.startswith("data:"):
            data_str = f"data:{mime};base64,{data_str}"
        image_url_for_api = data_str
        print(f"describe_image_with_gpt4o: Data URL created: {data_str[:100]}...", flush=True)
    else:
        print("describe_image_with_gpt4o: ERROR - No url or data in image_info", flush=True)
        return "[Image: Unable to process - invalid image format]"
    
    # Add image to content
    content_parts.append({
        "type": "image_url",
        "image_url": {
            "url": image_url_for_api,
            "detail": image_info.get("detail", "high") if not image_url_for_api.startswith("data:") else "high"
        }
    })
    
    try:
        print("describe_image_with_gpt4o: Calling OpenAI GPT-4o API...", flush=True)
        # Call GPT-4o with vision
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            max_tokens=500
        )
        
        print("describe_image_with_gpt4o: API call successful", flush=True)
        description = response.choices[0].message.content
        print(f"describe_image_with_gpt4o: Got description: {description[:100]}...", flush=True)
        return f"[Image Description]: {description}"
        
    except Exception as e:
        error_str = str(e)
        _log.error(f"Error describing image with GPT-4o: {e}")
        print(f"describe_image_with_gpt4o: EXCEPTION: {e}", flush=True)
        
        # Check if it's a download error and we have a URL (not already base64)
        if "invalid_image_url" in error_str or "downloading" in error_str.lower():
            if "url" in image_info and not image_info["url"].startswith("data:"):
                print("describe_image_with_gpt4o: URL failed, trying to download and convert to base64...", flush=True)
                try:
                    # Download the image ourselves and retry with base64
                    base64_data, mime_type = await download_image_as_base64(image_info["url"])
                    data_url = f"data:{mime_type};base64,{base64_data}"
                    
                    print(f"describe_image_with_gpt4o: Retrying with base64 data (length: {len(base64_data)})", flush=True)
                    
                    # Update content_parts with base64 data
                    content_parts[-1] = {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high"
                        }
                    }
                    
                    # Retry the API call
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": content_parts
                            }
                        ],
                        max_tokens=500
                    )
                    
                    print("describe_image_with_gpt4o: Retry with base64 successful!", flush=True)
                    description = response.choices[0].message.content
                    print(f"describe_image_with_gpt4o: Got description: {description[:100]}...", flush=True)
                    return f"[Image Description]: {description}"
                    
                except Exception as retry_e:
                    _log.error(f"Failed to download and retry image: {retry_e}")
                    print(f"describe_image_with_gpt4o: Retry also failed: {retry_e}", flush=True)
                    return f"[Image: Unable to process - download failed: {str(retry_e)}]"
        
        import traceback
        traceback.print_exc()
        return f"[Image: Unable to process - {str(e)}]"


async def process_multimodal_content_to_text(content: Any, user_provided_text: str = "") -> str:
    """Convert multimodal content (text + images) to text-only by describing images.
    
    Args:
        content: Message content (str or list with text/image parts)
        user_provided_text: Optional text to help contextualize image descriptions
        
    Returns:
        Text-only content with image descriptions
    """
    try:
        # If already text, return as-is
        if isinstance(content, str):
            print(f"process_multimodal_content_to_text: Content is already text: {content[:100]}...", flush=True)
            return content
        
        if not isinstance(content, list):
            # Try to extract text
            print(f"process_multimodal_content_to_text: Content is not list, extracting text...", flush=True)
            result = extract_text_from_message_content(content)
            print(f"process_multimodal_content_to_text: Extracted text: {result[:100] if result else '(empty)'}...", flush=True)
            return result
        
        print(f"process_multimodal_content_to_text: Processing list content with {len(content)} parts", flush=True)
        
        # Extract text and images separately
        text_parts = []
        images = []
        
        for idx, part in enumerate(content):
            # Handle Pydantic models by converting to dict
            if hasattr(part, 'model_dump'):
                print(f"process_multimodal_content_to_text: Part {idx} is a Pydantic model, converting to dict", flush=True)
                part = part.model_dump()
            elif not isinstance(part, dict):
                print(f"process_multimodal_content_to_text: Part {idx} is not a dict or Pydantic model (type: {type(part).__name__}), skipping", flush=True)
                continue
            
            ptype = part.get("type")
            print(f"process_multimodal_content_to_text: Part {idx} type: {ptype}", flush=True)
            
            # Text part
            if ptype == "text":
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
                    print(f"process_multimodal_content_to_text: Added text part: {text_val[:50]}...", flush=True)
            
            # Image parts
            elif ptype == "image_url":
                image_field = part.get("image_url")
                if isinstance(image_field, str):
                    images.append({"url": image_field})
                    print(f"process_multimodal_content_to_text: Added image (url string): {image_field[:50]}...", flush=True)
                elif isinstance(image_field, dict):
                    url_val = image_field.get("url")
                    if isinstance(url_val, str):
                        detail = image_field.get("detail", "high")
                        images.append({"url": url_val, "detail": detail})
                        print(f"process_multimodal_content_to_text: Added image (url dict): {url_val[:50]}...", flush=True)
            
            elif ptype in ("input_image", "image"):
                image_obj = part.get("image") or part.get("data") or part.get("b64")
                if isinstance(image_obj, dict):
                    data_str = image_obj.get("data") or image_obj.get("b64") or image_obj.get("base64")
                    mime = image_obj.get("mime_type") or image_obj.get("media_type") or "image/png"
                    if isinstance(data_str, str):
                        images.append({"data": data_str, "mime": mime})
                        print(f"process_multimodal_content_to_text: Added image (base64 dict): {len(data_str)} chars", flush=True)
                elif isinstance(image_obj, str):
                    images.append({"data": image_obj, "mime": "image/png"})
                    print(f"process_multimodal_content_to_text: Added image (base64 string): {len(image_obj)} chars", flush=True)
        
        # Build combined text
        combined_text = "\n\n".join(text_parts) if text_parts else user_provided_text
        print(f"process_multimodal_content_to_text: Combined text: {combined_text[:100] if combined_text else '(empty)'}...", flush=True)
        print(f"process_multimodal_content_to_text: Found {len(images)} image(s)", flush=True)
        
        # Process each image and add descriptions
        if images:
            _log.info(f"Processing {len(images)} image(s) with GPT-4o vision...")
            print(f"process_multimodal_content_to_text: Processing {len(images)} image(s) with GPT-4o vision...", flush=True)
            
            image_descriptions = []
            for idx, img_info in enumerate(images, 1):
                try:
                    print(f"process_multimodal_content_to_text: Describing image {idx}...", flush=True)
                    description = await describe_image_with_gpt4o(img_info, combined_text)
                    image_descriptions.append(f"**Image {idx}**:\n{description}")
                    _log.info(f"Successfully described image {idx}/{len(images)}")
                    print(f"process_multimodal_content_to_text: Successfully described image {idx}", flush=True)
                except Exception as e:
                    _log.error(f"Failed to describe image {idx}: {e}")
                    print(f"process_multimodal_content_to_text: ERROR describing image {idx}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    image_descriptions.append(f"**Image {idx}**: [Unable to process image - {str(e)}]")
            
            # Combine text and image descriptions
            if combined_text:
                result = combined_text + "\n\n" + "\n\n".join(image_descriptions)
            else:
                result = "\n\n".join(image_descriptions)
            
            print(f"process_multimodal_content_to_text: Final result length: {len(result)}", flush=True)
            print(f"process_multimodal_content_to_text: Final result preview: {result[:200]}...", flush=True)
            return result
        
        final_text = combined_text or ""
        print(f"process_multimodal_content_to_text: No images, returning text: {final_text[:100] if final_text else '(empty)'}...", flush=True)
        return final_text
        
    except Exception as e:
        _log.error(f"ERROR in process_multimodal_content_to_text: {e}")
        print(f"process_multimodal_content_to_text: EXCEPTION: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return fallback text extraction
        try:
            return extract_text_from_message_content(content)
        except Exception as e2:
            print(f"process_multimodal_content_to_text: Fallback extraction also failed: {e2}", flush=True)
            return user_provided_text or "[Error processing message content]"
