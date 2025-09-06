from __future__ import annotations

from typing import Any, Optional
import logging as _log

from .client import MaterialsProjectClient

class MCPProxy:
    """
    Direct client proxy that calls tools without MCP server.
    """

    def __init__(self) -> None:
        self.client = MaterialsProjectClient()

    async def call_tool(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> str:
        _log.info(f"call_tool: {tool_name}, {arguments}")
        arguments = arguments or {}
        
        # Map tool names to client methods
        if tool_name == "convert_name_to_symbols":
            name = arguments.get("name")
            if not name:
                return '{"error": "name parameter is required"}'
            result = self.client.get("materials/convert/name_to_symbols", {"name": name})
        elif tool_name == "get_material":
            result = self.client.get("materials/summary/get_material", arguments)
        elif tool_name == "get_material_by_char":
            result = self.client.get("materials/summary/get_material_by_char", arguments)
        elif tool_name == "get_material_details_by_ids":
            result = self.client.get("materials/summary/get_material_details_by_ids", arguments)
        else:
            return f'{{"error": "Unknown tool: {tool_name}"}}'
        
        # Format response as JSON string
        import json
        try:
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return f'{{"error": "Failed to serialize result: {str(e)}"}}'


