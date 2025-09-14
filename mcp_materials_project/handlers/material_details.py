"""
Handler for get_material_details_by_ids endpoint.
"""

import json
import logging
from typing import Any, Dict, List, Mapping, Annotated

from kani import ai_function, AIParam
from .base import BaseHandler

_log = logging.getLogger(__name__)


class MaterialDetailsHandler(BaseHandler):
    """Handler for material details endpoints."""
    
    @ai_function(desc="Fetch one or more materials by their material IDs and return detailed information about them.", auto_truncate=128000)
    async def get_material_details_by_ids(
        self,
        material_ids: Annotated[List[str], AIParam(desc="List of material IDs, e.g., ['mp-149', 'mp-150', 'mp-151'].")],
        fields: Annotated[List[str], AIParam(desc="List of fields to include. Values include 'builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_Ids'")] = None,
        all_fields: Annotated[bool, AIParam(desc="Whether to return all document fields. Useful if the user wants to know about the material without explicitly asking for certain fields (default True).")] = True,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10
    ) -> Dict[str, Any]:
        """Fetch one or more materials by their material IDs and return detailed information about them."""
        params = {
            "material_ids": material_ids,
            "all_fields": all_fields,
            "page": page,
            "per_page": per_page
        }
        if fields is not None:
            params["fields"] = fields
        
        result = self.handle_material_details(params)
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_material_details_by_ids",
                "result": result
            })
        return result

    def handle_material_details(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material_details_by_ids endpoint."""
        _log.info(f"GET materials/summary/get_material_details_by_ids with params: {params}")

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

        # Do NOT pass any 'limit' param to upstream search; it is internal-only

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

        return {
            "total_count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "data": data,
        }


def handle_material_details(handler: BaseHandler, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    if isinstance(handler, MaterialDetailsHandler):
        return handler.handle_material_details(params)
    else:
        # Create a new handler instance
        details_handler = MaterialDetailsHandler(handler.mpr)
        return details_handler.handle_material_details(params)
