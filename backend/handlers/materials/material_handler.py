"""
Materials handler for Materials Project API.

Provides unified access to material search and details endpoints.
"""

import json
import logging
from typing import Any, Dict, List, Mapping

from ..base import BaseHandler
from ..shared import success_result, error_result, ErrorType, Confidence
from ..shared.api_utils import format_field_error
from .ai_functions import MaterialsAIFunctionsMixin

_log = logging.getLogger(__name__)


class MaterialHandler(MaterialsAIFunctionsMixin, BaseHandler):
    """Unified handler for material search and details endpoints."""

    def mp_get_material_details(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material_details_by_ids endpoint."""
        _log.info(f"GET materials/summary/get_material_details_by_ids with params: {params}")

        try:
            fields = self._parse_csv_list(params.get("fields"))
            all_fields = self._parse_bool(params.get("all_fields"))

            material_ids = params.get("material_ids")
            # Handle both list and JSON string formats
            if isinstance(material_ids, str):
                # Try to parse as JSON first
                try:
                    material_ids = json.loads(material_ids)
                except:
                    # Fall back to CSV parsing if JSON fails
                    material_ids = material_ids.split(',') if ',' in material_ids else [material_ids]
            
            # Now material_ids is guaranteed to be a list
            if isinstance(material_ids, list):
                material_ids = [str(material_id).strip() for material_id in material_ids]
            else:
                raise ValueError(f"Invalid material_ids type: {type(material_ids)}")
            
            material_id_csv = ",".join(material_ids)
            _log.info(f"material_id_csv: {material_id_csv}")

            search_kwargs: Dict[str, Any] = {"material_ids": material_id_csv}

            if fields is not None:
                if isinstance(fields, str):
                    fields = [f.strip() for f in fields.split(",")]
                if "material_id" not in fields:
                    fields.append("material_id")
                search_kwargs["fields"] = fields
            elif not all_fields:
                search_kwargs["fields"] = ["material_id"]

            if all_fields is not None:
                search_kwargs["all_fields"] = all_fields

            # Pagination: default page=1, per_page<=10
            page, per_page = self._get_pagination(params)

            # Always compute total_count using the same criteria (excluding fields/paging)
            count_criteria: Dict[str, Any] = {}
            if material_id_csv:
                count_criteria["material_ids"] = material_id_csv
            total = self._call_summary_count(count_criteria)

            docs = self.mpr.materials.summary.search(**search_kwargs)
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
                function="mp_get_material_details",
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
            _log.error(f"Material details fetch failed: {e}", exc_info=True)
            formatted_error = format_field_error(e)
            return error_result(
                handler="materials",
                function="mp_get_material_details",
                error=formatted_error,
                error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"]
            )

    def mp_search_by_composition(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material endpoint."""
        _log.info(f"GET materials/summary/get_material with params: {params}")
        
        try:
            kwargs = self._build_summary_search_kwargs(params)
            if "__errors__" in kwargs:
                return error_result(
                    handler="materials",
                    function="mp_search_by_composition",
                    error="One or more range parameters are invalid: " + str(kwargs["__errors__"]),
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["Materials Project"]
                )

            # Pagination: default page=1, per_page<=10
            page, per_page = self._get_pagination(params)

            # Always compute total count, regardless of chunking/limit
            total = self._total_count_for_summary(kwargs)

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
            formatted_error = format_field_error(e)
            return error_result(
                handler="materials",
                function="handle_material_search",
                error=formatted_error,
                error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"]
            )

    def mp_get_by_characteristic(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material_by_char endpoint."""
        _log.info(f"GET materials/summary/get_material_by_char with params: {params}")
        
        try:
            kwargs = self._build_summary_search_kwargs(params)
            if "__errors__" in kwargs:
                return error_result(
                    handler="materials",
                    function="mp_get_by_characteristic",
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
                    function="mp_get_by_characteristic",
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

            _log.info(f"mp_get_by_characteristic -> summary.search kwargs: {kwargs}")

            # Pagination: default page=1, per_page<=10
            page, per_page = self._get_pagination(params)

            # Always compute total count for the same filter criteria
            total = self._total_count_for_summary(kwargs)

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
            formatted_error = format_field_error(e)
            return error_result(
                handler="materials",
                function="handle_material_by_char",
                error=formatted_error,
                error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"]
            )

