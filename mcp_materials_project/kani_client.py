# kani_client.py
from __future__ import annotations

import os
from typing import Any, Optional
import logging as _log
from importlib import import_module, invalidate_caches
from pathlib import Path

from dotenv import load_dotenv
from kani import Kani, ChatRole, ChatMessage as KChatMessage  # alias to avoid clashing with pydantic ChatMessage

from kani.engines.openai.engine import OpenAIEngine  # we subclass this

from .mcp_proxy import MCPProxy
from .codegen import generate_kani
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
# Generated mixin loader
# --------------------------------------------------------------------------------------
def _load_generated_mixin():
    """
    Generate the Kani mixin class from JSON templates and import it.
    Falls back to an empty mixin if anything fails.
    """
    try:
        templates_dir = Path(__file__).parent / "templates"
        auto_dir = Path(__file__).parent / "auto_generated"
        os.makedirs(auto_dir, exist_ok=True)

        init_file = auto_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("__all__ = []\n", encoding="utf-8")

        out_file = auto_dir / "kani.py"
        generated = generate_kani(str(templates_dir), str(out_file))
        _log.info(f"[kani_client] generated mixin module at: {generated}")

        if generated and generated.exists():
            invalidate_caches()
            mod = import_module("mcp_materials_project.auto_generated.kani")
            _log.info(f"[kani_client] imported mixin module: {mod}")
            mixin_cls = getattr(mod, "GeneratedKaniTools", None)
            if isinstance(mixin_cls, type):
                return mixin_cls
    except Exception as e:
        _log.exception("[kani_client] Failed to generate/import mixin; using empty mixin. Error: %s", e)

    class _EmptyMixin:
        pass

    return _EmptyMixin


_GeneratedMixin = _load_generated_mixin()
_log.info(f"[kani_client] Using mixin: {_GeneratedMixin}")


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
# --------------------------------------------------------------------------------------
# Kani wrapper
# --------------------------------------------------------------------------------------
class MPKani(_GeneratedMixin, Kani):
    def __init__(
        self,
        client: Optional[object] = None,
        model: str = "gpt-4.1",
        *,
        chat_history: Optional[list[KChatMessage]] = None,
        always_included_messages: Optional[list[KChatMessage]] = None,
    ) -> None:
        engine = _build_engine(model)
        super().__init__(
            engine,
            system_prompt=KANI_SYSTEM_PROMPT,
            chat_history=chat_history,
            always_included_messages=always_included_messages,
        )
        self._proxy = MCPProxy()
        self.recent_tool_outputs: list[dict[str, Any]] = []
