# kani_client.py
from __future__ import annotations

import os
from typing import Any, Optional
import logging as _log

from dotenv import load_dotenv
from kani import Kani, ChatMessage as KChatMessage  # alias to avoid clashing with pydantic ChatMessage

from kani.engines.openai.engine import OpenAIEngine  # we subclass this

from .handlers import MaterialDetailsHandler, MaterialSearchHandler, NameConversionHandler
from .prompts import KANI_SYSTEM_PROMPT


# --------------------------------------------------------------------------------------
# Engine: disable function token reserve to avoid schema pretty-printer crashes
# --------------------------------------------------------------------------------------
class OpenAIEngineNoFuncReserve(OpenAIEngine):
    """
    Kani's OpenAI engine estimates token reserve by pretty-printing tool schemas.
    Some schema generators produce objects without a top-level "type" (e.g., anyOf),
    which can crash the formatter in certain Kani versions.

    We override the reserve implementation to return 0 and skip that path entirely.
    """
    def _function_token_reserve_impl(self, functions: frozenset) -> int:
        return 0

# --------------------------------------------------------------------------------------
# Engine builder
# --------------------------------------------------------------------------------------
def _build_engine(model: str = "gpt-4.1") -> OpenAIEngine:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    # Use our patched engine class that skips function token reserve
    eng = OpenAIEngineNoFuncReserve(api_key, model=model)
    _log.info(f"[kani_client] Initialized OpenAIEngineNoFuncReserve(model={model})")
    return eng

# --------------------------------------------------------------------------------------
# Kani wrapper
# --------------------------------------------------------------------------------------
class MPKani(MaterialDetailsHandler, MaterialSearchHandler, NameConversionHandler, Kani):
    def __init__(
        self,
        client: Optional[object] = None,
        model: str = "gpt-4.1",
        *,
        chat_history: Optional[list[KChatMessage]] = None,
        always_included_messages: Optional[list[KChatMessage]] = None,
    ) -> None:
        # Initialize MPRester and handlers
        import os
        from mp_api.client import MPRester
        api_key = os.getenv("MP_API_KEY")
        mpr = MPRester(api_key)
        
        # Initialize Kani first
        engine = _build_engine(model)
        Kani.__init__(
            self,
            engine,
            system_prompt=KANI_SYSTEM_PROMPT,
            chat_history=chat_history,
            always_included_messages=always_included_messages,
        )
        
        # Initialize all handler classes
        MaterialDetailsHandler.__init__(self, mpr)
        MaterialSearchHandler.__init__(self, mpr)
        NameConversionHandler.__init__(self, mpr)
        
        self.recent_tool_outputs: list[dict[str, Any]] = []
