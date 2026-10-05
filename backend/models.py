from __future__ import annotations

from typing import Any

BACKBONE_MODEL = "gpt-5.4"
BACKBONE_REASONING_EFFORT = "high"

HELPER_MODEL = "gpt-5.4"
HELPER_REASONING_EFFORT = "none"

PROMPT_CACHE_KEY = "lamina"

CARD_PARSER_MODEL = "gpt-6-sol"
CARD_PARSER_REASONING_EFFORT = "none"

_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


def helper_completion_params(max_tokens: int, temperature: float) -> dict[str, Any]:
    if is_reasoning_model(HELPER_MODEL):
        return {
            "model": HELPER_MODEL,
            "reasoning_effort": HELPER_REASONING_EFFORT,
            "max_completion_tokens": max_tokens,
        }
    return {"model": HELPER_MODEL, "max_tokens": max_tokens, "temperature": temperature}
