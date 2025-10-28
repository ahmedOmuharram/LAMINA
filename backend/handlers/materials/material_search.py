"""
Handler for material search endpoints (get_material, get_material_by_char).
"""

import json
import logging
from typing import Any, Dict, Mapping, List

from ..base.base import BaseHandler
from ..base.result_wrappers import success_result, error_result, ErrorType, Confidence
from .ai_functions import MaterialsAIFunctionsMixin

_log = logging.getLogger(__name__)


class MaterialSearchHandler(MaterialsAIFunctionsMixin, BaseHandler):
    """Handler for material search endpoints."""

    def handle_material_search(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material endpoint."""
        _log.info(f"GET materials/summary/get_material with params: {params}")
        
        try:
            kwargs = self._build_summary_search_kwargs(params)
            if "__errors__" in kwargs:
                return error_result(
                    handler="materials",
                    function="handle_material_search",
                    error="One or more range parameters are invalid: " + str(kwargs["__errors__"]),
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["Materials Project"]
                )

            # Pagination: default page=1, per_page<=10
            page, per_page = self._get_pagination(params)

            # Always compute total count, regardless of chunking/limit
            total = self._total_count_for_summary(kwargs)

            # Do NOT pass any 'limit' param to upstream search; it is internal-only

            docs = self.mpr.materials.summary.search(**kwargs)
            data_all = self._convert_docs_to_dicts(docs)
            data = self._slice_for_page(data_all, page, per_page)

            # Envelope with pagination metadata
            total_pages = None
            try:
                if total is not None and per_page:
                    total_pages = (int(total) + per_page - 1) // per_page
            except Exception:
                total_pages = None

            return success_result(
                handler="materials",
                function="handle_material_search",
                data={
                    "total_count": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "data": data
                },
                citations=["Materials Project"],
                confidence=Confidence.HIGH if data else Confidence.LOW
            )
        except Exception as e:
            _log.error(f"Material search failed: {e}", exc_info=True)
            return error_result(
                handler="materials",
                function="handle_material_search",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"]
            )

    def handle_material_by_char(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material_by_char endpoint."""
        _log.info(f"GET materials/summary/get_material_by_char with params: {params}")
        
        try:
            kwargs = self._build_summary_search_kwargs(params)
            if "__errors__" in kwargs:
                return error_result(
                    handler="materials",
                    function="handle_material_by_char",
                    error="One or more range parameters are invalid: " + str(kwargs["__errors__"]),
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["Materials Project"]
                )

            # Accept either identity selectors OR any numeric/range filters
            selector_keys = {
                "material_ids", "formula", "formula_pretty", "chemsys", "elements", "exclude_elements",
                "spacegroup_number", "spacegroup_symbol", "crystal_system", "magnetic_ordering", "ordering"
            }
            range_selector_keys = set(self.RANGE_KEYS)

            has_selector = any(k in kwargs for k in selector_keys | range_selector_keys)

            if not has_selector:
                return error_result(
                    handler="materials",
                    function="handle_material_by_char",
                    error="Provide at least one selector (e.g., formula/chemsys/elements/material_ids) or a numeric/range filter (e.g., band_gap).",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["Materials Project"]
                )

            # Always include material_id in fields without clobbering others.
            existing_fields = kwargs.get("fields")
            if existing_fields is None:
                # Default to common fields that are usually needed
                kwargs["fields"] = ["material_id", "formula_pretty", "elements", "chemsys"]
            elif isinstance(existing_fields, list):
                if "material_id" not in existing_fields:
                    kwargs["fields"] = existing_fields + ["material_id"]
            else:
                if existing_fields != "material_id":
                    kwargs["fields"] = [existing_fields, "material_id"]

            _log.info(f"get_material_by_char -> summary.search kwargs: {kwargs}")

            # Pagination: default page=1, per_page<=10
            page, per_page = self._get_pagination(params)

            # Always compute total count for the same filter criteria
            total = self._total_count_for_summary(kwargs)

            # Do NOT pass any 'limit' param to upstream search; it is internal-only

            docs = self.mpr.materials.summary.search(**kwargs)
            data_all = self._convert_docs_to_dicts(docs)
            data = self._slice_for_page(data_all, page, per_page)

            # Envelope with pagination metadata
            total_pages = None
            try:
                if total is not None and per_page:
                    total_pages = (int(total) + per_page - 1) // per_page
            except Exception:
                total_pages = None

            return success_result(
                handler="materials",
                function="handle_material_by_char",
                data={
                    "total_count": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "data": data
                },
                citations=["Materials Project"],
                confidence=Confidence.HIGH if data else Confidence.LOW
            )
        except Exception as e:
            _log.error(f"Material search by char failed: {e}", exc_info=True)
            return error_result(
                handler="materials",
                function="handle_material_by_char",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"]
            )


def handle_material_search(handler: BaseHandler, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    if isinstance(handler, MaterialSearchHandler):
        return handler.handle_material_search(params)
    else:
        # Create a new handler instance
        search_handler = MaterialSearchHandler(handler.mpr)
        return search_handler.handle_material_search(params)


def handle_material_by_char(handler: BaseHandler, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    if isinstance(handler, MaterialSearchHandler):
        return handler.handle_material_by_char(params)
    else:
        # Create a new handler instance
        search_handler = MaterialSearchHandler(handler.mpr)
        return search_handler.handle_material_by_char(params)
