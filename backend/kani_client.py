# kani_client.py
from __future__ import annotations

import os
from typing import Any, Optional, List, Dict, Set
import logging as _log
import inspect

from dotenv import load_dotenv
from kani import Kani, ChatMessage as KChatMessage  # alias to avoid clashing with pydantic ChatMessage
from kani.ai_function import AIFunction

from kani.engines.openai.engine import OpenAIEngine  # we subclass this

from .handlers import MaterialHandler, SearXNGSearchHandler, BatteryHandler, SemiconductorHandler, AlloyHandler, SuperconductorHandler, MagnetHandler, SolutesHandler, CalPhadHandler
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
# Utility functions for AI function management
# --------------------------------------------------------------------------------------
def get_all_ai_functions() -> List[Dict[str, Any]]:
    """Get metadata for all available AI functions without instantiating Kani."""
    from mp_api.client import MPRester
    from kani.ai_function import AIFunction
    
    # Create a temporary instance to introspect functions
    api_key = os.getenv("MP_API_KEY")
    mpr = MPRester(api_key)
    
    # Create a temporary Kani instance to discover functions
    temp_kani = MPKani(model="gpt-4o-mini")
    
    functions_list = []
    
    # Use Kani's built-in function discovery (it's a dict)
    ai_functions_dict = temp_kani.functions
    
    for func_name, func in ai_functions_dict.items():
        if isinstance(func, AIFunction):
            # Get the actual method for module info
            method = getattr(temp_kani, func.name, None)
            category = _get_function_category(func.name, method) if method else 'Other'
            
            functions_list.append({
                'name': func.name,
                'description': func.desc or '',
                'category': category
            })
    
    # Sort by category and name
    functions_list.sort(key=lambda x: (x['category'], x['name']))
    return functions_list


def _get_function_category(func_name: str, func: Any) -> str:
    """Determine the category of a function based on its name and location."""
    module = inspect.getmodule(func)
    if module:
        module_path = module.__name__
        if 'materials' in module_path:
            return 'Materials'
        elif 'search' in module_path:
            return 'Search'
        elif 'calphad' in module_path:
            return 'CALPHAD'
        elif 'electrochemistry' in module_path or 'battery' in module_path:
            return 'Electrochemistry'
        elif 'semiconductor' in module_path:
            return 'Semiconductors'
        elif 'magnet' in module_path:
            return 'Magnets'
        elif 'superconductor' in module_path:
            return 'Superconductors'
        elif 'alloy' in module_path:
            return 'Alloys'
        elif 'solute' in module_path:
            return 'Solutes'
    return 'Other'


# --------------------------------------------------------------------------------------
# Kani wrapper
# --------------------------------------------------------------------------------------
class MPKani(MaterialHandler, SearXNGSearchHandler, BatteryHandler, CalPhadHandler, SemiconductorHandler, AlloyHandler, SuperconductorHandler, MagnetHandler, SolutesHandler, Kani):
    def __init__(
        self,
        client: Optional[object] = None,
        model: str = "gpt-4.1",
        *,
        chat_history: Optional[list[KChatMessage]] = None,
        always_included_messages: Optional[list[KChatMessage]] = None,
        enabled_functions: Optional[Set[str]] = None,
    ) -> None:
        # Initialize MPRester and handlers
        import os
        from mp_api.client import MPRester
        api_key = os.getenv("MP_API_KEY")
        mpr = MPRester(api_key)
        
        # Store enabled functions filter
        self._enabled_functions = enabled_functions
        
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
        MaterialHandler.__init__(self, mpr)
        SearXNGSearchHandler.__init__(self)
        BatteryHandler.__init__(self, mpr)
        CalPhadHandler.__init__(self)
        SemiconductorHandler.__init__(self, mpr)
        AlloyHandler.__init__(self, mpr)
        SuperconductorHandler.__init__(self, mpr)
        MagnetHandler.__init__(self, mpr)
        SolutesHandler.__init__(self, mpr)
        
        self.recent_tool_outputs: list[dict[str, Any]] = []
        
        # Apply function filtering if enabled_functions is provided
        if self._enabled_functions is not None:
            # Filter the functions dict that was set by Kani.__init__
            self.functions = {
                name: func for name, func in self.functions.items()
                if name in self._enabled_functions
            }
