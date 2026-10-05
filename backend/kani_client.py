# kani_client.py
from __future__ import annotations

import functools
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
from .models import BACKBONE_MODEL, BACKBONE_REASONING_EFFORT, PROMPT_CACHE_KEY, is_reasoning_model

import tiktoken

_LOCAL_TOKENIZER = tiktoken.get_encoding("o200k_base")


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

    async def prompt_len(self, messages, functions=None, **kwargs) -> int:
        total = 0
        for message in messages:
            total += 4 + len(_LOCAL_TOKENIZER.encode(message.text or ""))
            for call in message.tool_calls or []:
                total += len(_LOCAL_TOKENIZER.encode(call.function.name + call.function.arguments))
        return total

# --------------------------------------------------------------------------------------
# Engine builder
# --------------------------------------------------------------------------------------
def _build_engine(model: str = BACKBONE_MODEL) -> OpenAIEngine:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    # Use our patched engine class that skips function token reserve
    if is_reasoning_model(model):
        eng = OpenAIEngineNoFuncReserve(api_key, model=model, api_type="responses")
    else:
        eng = OpenAIEngineNoFuncReserve(api_key, model=model, api_type="chat_completions")
    _log.info(f"[kani_client] Initialized OpenAIEngineNoFuncReserve(model={model})")
    return eng


UNREADABLE_SUMMARY_FIELDS = ("dos", "bandstructure")


def materials_project_client(api_key: Optional[str]):
    from mp_api.client import MPRester

    client = MPRester(api_key)
    summary = client.materials.summary
    search = summary.search
    readable = [field for field in summary.available_fields if field not in UNREADABLE_SUMMARY_FIELDS]

    @functools.wraps(search)
    def search_readable_fields(*args, **kwargs):
        if kwargs.get("fields") is None and kwargs.get("all_fields", True):
            kwargs["fields"] = readable
        return search(*args, **kwargs)

    summary.search = search_readable_fields
    return client

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
    temp_kani = MPKani(model=BACKBONE_MODEL)
    
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
        model: str = BACKBONE_MODEL,
        *,
        system_prompt: str = KANI_SYSTEM_PROMPT,
        chat_history: Optional[list[KChatMessage]] = None,
        always_included_messages: Optional[list[KChatMessage]] = None,
        enabled_functions: Optional[Set[str]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize MPKani with optional function filtering.
        
        Args:
            enabled_functions: Explicit set of function names to enable. If provided,
                only these functions will be available.
            temperature, top_p, seed: LLM hyperparameters
        """
        # Initialize MPRester and handlers
        import os
        api_key = os.getenv("MP_API_KEY")
        mpr = materials_project_client(api_key)
        
        # Store function filtering
        self._enabled_functions = enabled_functions
        
        # Store hyperparameters for use in requests
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._model = model

        # Initialize Kani first
        engine = _build_engine(model)
        Kani.__init__(
            self,
            engine,
            system_prompt=system_prompt,
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
        
        # Apply function filtering if enabled_functions is set
        if self._enabled_functions is not None:
            # Filter the functions dict that was set by Kani.__init__
            original_count = len(self.functions)
            self.functions = {
                name: func for name, func in self.functions.items()
                if name in self._enabled_functions
            }
            filtered_count = len(self.functions)
            if filtered_count < original_count:
                _log.info(f"[MPKani] Function filtering: {filtered_count}/{original_count} functions enabled")
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Get hyperparameters to pass to the engine for each request."""
        if is_reasoning_model(self._model):
            return {
                "reasoning": {"effort": BACKBONE_REASONING_EFFORT},
                "prompt_cache_key": PROMPT_CACHE_KEY,
            }
        params = {
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if self._seed is not None:
            params["seed"] = self._seed
        return params
