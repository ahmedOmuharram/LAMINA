"""
Handler for convert_name_to_symbols endpoint.
"""

import json
import os
import logging
from typing import Any, Dict, Mapping, Annotated

import openai
from kani import ai_function, AIParam
from .base import BaseHandler
from ..prompts import get_name_conversion_prompt

_log = logging.getLogger(__name__)


class NameConversionHandler(BaseHandler):
    """Handler for name conversion endpoints."""
    
    @ai_function(desc="Convert an english worded chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly.", auto_truncate=128000)
    async def convert_name_to_symbols(self, name: Annotated[str, AIParam(desc="The english worded chemical name to convert. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly.")]) -> Dict[str, Any]:
        """Convert an english worded chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly."""
        params = {"name": name}
        result = self.handle_name_conversion(params)
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "convert_name_to_symbols",
                "result": result
            })
        return result

    def handle_name_conversion(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/convert/name_to_symbols endpoint."""
        _log.info(f"GET materials/convert/name_to_symbols with params: {params}")
        
        name = params.get("name")
        if not name:
            return {"total_count": None, "error": {"type": "missing_parameter", "message": "name parameter is required"}}

        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"total_count": None, "error": {"type": "configuration_error", "message": "OPENAI_API_KEY not configured"}}

            client = openai.OpenAI(api_key=api_key)
            conversion_prompt = get_name_conversion_prompt(name)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": conversion_prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            converted_symbols = response.choices[0].message.content.strip()
            _log.info(f"Converted '{name}' to '{converted_symbols}'")

            return {"total_count": None, "input_name": name, "converted_symbols": converted_symbols, "success": True}

        except Exception as e:
            _log.error(f"Error converting name to symbols: {e}")
            return {"total_count": None, "error": {"type": "conversion_error", "message": str(e)}}


def handle_name_conversion(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    # This handler doesn't need MPRester, so we can create a dummy one
    from mp_api.client import MPRester
    handler = NameConversionHandler(MPRester())
    return handler.handle_name_conversion(params)
