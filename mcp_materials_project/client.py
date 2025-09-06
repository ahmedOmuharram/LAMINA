from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

from mp_api.client import MPRester
import logging as _log

from .handlers import (
    BaseHandler, MaterialDetailsHandler, MaterialSearchHandler, NameConversionHandler
)

# ----------------------------
# Lightweight client wrapper
# ----------------------------
class MaterialsProjectClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key: Optional[str] = api_key or os.getenv("MP_API_KEY") or None
        self.mpr = MPRester(self.api_key)
        
        # Initialize handlers
        self.base_handler = BaseHandler(self.mpr)
        self.details_handler = MaterialDetailsHandler(self.mpr)
        self.search_handler = MaterialSearchHandler(self.mpr)
        self.name_conversion_handler = NameConversionHandler(self.mpr)

    def get(self, path: str = "", params: Optional[Mapping[str, Any]] = None) -> Any:
        params = dict(params or {})
        path_key = (path or "").strip().lstrip("/")

        if path_key.startswith("materials"):
            results = self._handle_materials_endpoint(path_key, params)
            return results

        return {"total_count": None, "error": {"type": "unknown_namespace", "message": f"Unhandled path: {path_key}"}}

    def _handle_materials_endpoint(self, path_key: str, params: Mapping[str, Any]) -> Any:
        """
        Route Materials Project endpoints to appropriate handlers.
        This replaces the monolithic handle_materials_endpoint function.
        """
        # ---- Global pre-validation: catch singleton ranges early for ALL endpoints ----
        singleton_errors = self._validate_range_parameters(params)
        if singleton_errors:
            return {
                "total_count": None,
                "error": {
                    "type": "invalid_parameter",
                    "message": "One or more range parameters are invalid.",
                    "details": singleton_errors
                }
            }

        # ---- Route to appropriate handler ----
        if path_key == "materials/summary/get_material_details_by_ids":
            return self.details_handler.handle_material_details(params)
        
        elif path_key == "materials/summary/get_material":
            return self.search_handler.handle_material_search(params)
        
        elif path_key == "materials/summary/get_material_by_char":
            return self.search_handler.handle_material_by_char(params)
        
        elif path_key == "materials/convert/name_to_symbols":
            return self.name_conversion_handler.handle_name_conversion(params)

        return {"total_count": None, "error": {"type": "unknown_endpoint", "message": f"Unhandled path: {path_key}"}}

    def _validate_range_parameters(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Validate that range parameters have proper (min,max) values."""
        singleton_errors = []
        
        # Check standard range keys
        for key in self.base_handler.RANGE_KEYS:
            if key in params and self.base_handler._is_singleton_range_value(params.get(key)):
                singleton_errors.append({
                    "param": key,
                    "value": params.get(key),
                    "reason": "Expected two numbers (min,max), got one."
                })
        
        # Check num_elements alias
        if "num_elements" in params and self.base_handler._is_singleton_range_value(params.get("num_elements")):
            singleton_errors.append({
                "param": "num_elements",
                "value": params.get("num_elements"),
                "reason": "Expected two numbers (min,max), got one."
            })
        
        return singleton_errors

    @staticmethod
    def to_pretty_json(data: Any) -> str:
        try:
            return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(data)
