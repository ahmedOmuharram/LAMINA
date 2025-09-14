"""
Handler for material search endpoints (get_material, get_material_by_char).
"""

import json
import logging
from typing import Any, Dict, Mapping, List, Annotated

from kani import ai_function, AIParam
from .base import BaseHandler

_log = logging.getLogger(__name__)


class MaterialSearchHandler(BaseHandler):
    """Handler for material search endpoints."""
    
    @ai_function(desc="Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.", auto_truncate=128000)
    async def get_material(
        self,
        chemsys: Annotated[str, AIParam(desc="Chemical system(s) or comma-separated list (e.g., Li-Fe-O,Si-*). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")] = None,
        formula: Annotated[str, AIParam(desc="Formula(s), anonymized formula, or wildcard(s) (e.g., Li2FeO3,Fe2O3,Fe*O*). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")] = None,
        element: Annotated[str, AIParam(desc="Element(s) or comma-separated list (e.g., Li,Fe,O). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")] = None,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10
    ) -> Dict[str, Any]:
        """Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols."""
        params = {}
        if chemsys is not None:
            params["chemsys"] = chemsys
        if formula is not None:
            params["formula"] = formula
        if element is not None:
            params["element"] = element
        params["page"] = page
        params["per_page"] = per_page
        
        result = self.handle_material_search(params)
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_material",
                "result": result
            })
        return result

    @ai_function(desc="Fetch a material id and formula by a characteristic of the material", auto_truncate=128000)
    async def get_material_by_char(
        self,
        band_gap: Annotated[List[float], AIParam(desc="Min,max range of band gap in eV (e.g., [1.2, 3.0]). Tuple of floats, and both values are required.")] = None,
        crystal_system: Annotated[str, AIParam(desc="Crystal system of material. Available options are 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal' or 'Cubic'.")] = None,
        density: Annotated[List[float], AIParam(desc="Min,max density range. Tuple of floats, and both values are required.")] = None,
        e_electronic: Annotated[List[float], AIParam(desc="Min,max electronic dielectric constant. Tuple of floats, and both values are required.")] = None,
        e_ionic: Annotated[List[float], AIParam(desc="Min,max ionic dielectric constant. Tuple of floats, and both values are required.")] = None,
        e_total: Annotated[List[float], AIParam(desc="Min,max total dielectric constant. Tuple of floats, and both values are required.")] = None,
        efermi: Annotated[List[float], AIParam(desc="Min,max fermi energy in eV. Tuple of floats, and both values are required.")] = None,
        elastic_anisotropy: Annotated[List[float], AIParam(desc="Min,max elastic anisotropy. Tuple of floats, and both values are required.")] = None,
        elements: Annotated[List[str], AIParam(desc="List of elements (e.g., ['Li', 'Fe', 'O']).")] = None,
        energy_above_hull: Annotated[List[float], AIParam(desc="Min,max energy above hull in eV/atom. Tuple of floats, and both values are required.")] = None,
        equilibrium_reaction_energy: Annotated[List[float], AIParam(desc="Min,max equilibrium reaction energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        formation_energy: Annotated[List[float], AIParam(desc="Min,max formation energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        g_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        g_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        g_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        k_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        k_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        k_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        n: Annotated[List[int], AIParam(desc="Min,max number of atoms. Tuple of ints, and both values are required.")] = None,
        nelements: Annotated[List[int], AIParam(desc="Min,max number of elements. Tuple of ints, and both values are required.")] = None,
        num_sites: Annotated[List[int], AIParam(desc="Min,max number of sites. Tuple of ints, and both values are required.")] = None,
        num_magnetic_sites: Annotated[List[int], AIParam(desc="Min,max number of magnetic sites.")] = None,
        num_unique_magnetic_sites: Annotated[List[int], AIParam(desc="Min,max number of unique magnetic sites. Tuple of ints, and both values are required.")] = None,
        piezoelectric_modulus: Annotated[List[float], AIParam(desc="Min,max piezoelectric modulus in C/m^2. Tuple of floats, and both values are required.")] = None,
        poisson_ratio: Annotated[List[float], AIParam(desc="Min,max Poisson's ratio. Tuple of floats, and both values are required.")] = None,
        shape_factor: Annotated[List[float], AIParam(desc="Min,max shape factor. Tuple of floats, and both values are required.")] = None,
        surface_energy_anisotropy: Annotated[List[float], AIParam(desc="Min,max surface energy anisotropy. Tuple of floats, and both values are required.")] = None,
        total_energy: Annotated[List[float], AIParam(desc="Min,max total energy in eV/atom.")] = None,
        total_magnetization: Annotated[List[float], AIParam(desc="Min,max total magnetization in Bohr magnetons/atom.")] = None,
        total_magnetization_normalized_formula_units: Annotated[List[float], AIParam(desc="Min,max total magnetization normalized to formula units in Bohr magnetons/formula unit.")] = None,
        total_magnetization_normalized_vol: Annotated[List[float], AIParam(desc="Min,max total magnetization normalized to volume in Bohr magnetons/bohr^3.")] = None,
        uncorrected_energy: Annotated[List[float], AIParam(desc="Min,max uncorrected energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        volume: Annotated[List[float], AIParam(desc="Min,max volume in bohr^3. Tuple of floats, and both values are required.")] = None,
        weighted_surface_energy: Annotated[List[float], AIParam(desc="Min,max weighted surface energy in eV/ang^2. Tuple of floats, and both values are required.")] = None,
        weighted_work_function: Annotated[List[float], AIParam(desc="Min,max weighted work function in eV. Tuple of floats, and both values are required.")] = None,
        surface_anisotropy: Annotated[List[float], AIParam(desc="Min,max surface anisotropy. Tuple of floats, and both values are required.")] = None,
        has_reconstructed: Annotated[bool, AIParam(desc="Whether the entry has reconstructed surfaces.")] = None,
        is_gap_direct: Annotated[bool, AIParam(desc="Whether the material has a direct band gap.")] = None,
        is_metal: Annotated[bool, AIParam(desc="Whether the material is considered a metal.")] = None,
        is_stable: Annotated[bool, AIParam(desc="Whether the material lies on the convex energy hull.")] = None,
        magnetic_ordering: Annotated[str, AIParam(desc="Magnetic ordering of material. Available options are 'paramagnetic', 'ferromagnetic', 'antiferromagnetic', and 'ferrimagnetic'.")] = None,
        spacegroup_number: Annotated[int, AIParam(desc="Spacegroup number of material.")] = None,
        spacegroup_symbol: Annotated[str, AIParam(desc="Spacegroup symbol of material.")] = None,
        exclude_elements: Annotated[str, AIParam(desc="Elements to exclude (e.g., Li,Fe,O).")] = None,
        possible_species: Annotated[str, AIParam(desc="Possible species of material (e.g., Li,Fe,O).")] = None,
        has_props: Annotated[str, AIParam(desc="Calculated properties available (list of HasProps or strings).")] = None,
        theoretical: Annotated[bool, AIParam(desc="Whether the entry is theoretical (true) or experimental/experimentally observed (false).")] = None,
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin (optional).")] = None,
        pressure: Annotated[float, AIParam(desc="Pressure in GPa (optional).")] = None,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10
    ) -> Dict[str, Any]:
        """Fetch a material id and formula by a characteristic of the material"""
        params = {}
        if band_gap is not None:
            params["band_gap"] = band_gap
        if crystal_system is not None:
            params["crystal_system"] = crystal_system
        if density is not None:
            params["density"] = density
        if e_electronic is not None:
            params["e_electronic"] = e_electronic
        if e_ionic is not None:
            params["e_ionic"] = e_ionic
        if e_total is not None:
            params["e_total"] = e_total
        if efermi is not None:
            params["efermi"] = efermi
        if elastic_anisotropy is not None:
            params["elastic_anisotropy"] = elastic_anisotropy
        if elements is not None:
            params["elements"] = elements
        if energy_above_hull is not None:
            params["energy_above_hull"] = energy_above_hull
        if equilibrium_reaction_energy is not None:
            params["equilibrium_reaction_energy"] = equilibrium_reaction_energy
        if formation_energy is not None:
            params["formation_energy"] = formation_energy
        if g_reuss is not None:
            params["g_reuss"] = g_reuss
        if g_voigt is not None:
            params["g_voigt"] = g_voigt
        if g_vrh is not None:
            params["g_vrh"] = g_vrh
        if k_reuss is not None:
            params["k_reuss"] = k_reuss
        if k_voigt is not None:
            params["k_voigt"] = k_voigt
        if k_vrh is not None:
            params["k_vrh"] = k_vrh
        if n is not None:
            params["n"] = n
        if nelements is not None:
            params["nelements"] = nelements
        if num_sites is not None:
            params["num_sites"] = num_sites
        if num_magnetic_sites is not None:
            params["num_magnetic_sites"] = num_magnetic_sites
        if num_unique_magnetic_sites is not None:
            params["num_unique_magnetic_sites"] = num_unique_magnetic_sites
        if piezoelectric_modulus is not None:
            params["piezoelectric_modulus"] = piezoelectric_modulus
        if poisson_ratio is not None:
            params["poisson_ratio"] = poisson_ratio
        if shape_factor is not None:
            params["shape_factor"] = shape_factor
        if surface_energy_anisotropy is not None:
            params["surface_energy_anisotropy"] = surface_energy_anisotropy
        if total_energy is not None:
            params["total_energy"] = total_energy
        if total_magnetization is not None:
            params["total_magnetization"] = total_magnetization
        if total_magnetization_normalized_formula_units is not None:
            params["total_magnetization_normalized_formula_units"] = total_magnetization_normalized_formula_units
        if total_magnetization_normalized_vol is not None:
            params["total_magnetization_normalized_vol"] = total_magnetization_normalized_vol
        if uncorrected_energy is not None:
            params["uncorrected_energy"] = uncorrected_energy
        if volume is not None:
            params["volume"] = volume
        if weighted_surface_energy is not None:
            params["weighted_surface_energy"] = weighted_surface_energy
        if weighted_work_function is not None:
            params["weighted_work_function"] = weighted_work_function
        if surface_anisotropy is not None:
            params["surface_anisotropy"] = surface_anisotropy
        if has_reconstructed is not None:
            params["has_reconstructed"] = has_reconstructed
        if is_gap_direct is not None:
            params["is_gap_direct"] = is_gap_direct
        if is_metal is not None:
            params["is_metal"] = is_metal
        if is_stable is not None:
            params["is_stable"] = is_stable
        if magnetic_ordering is not None:
            params["magnetic_ordering"] = magnetic_ordering
        if spacegroup_number is not None:
            params["spacegroup_number"] = spacegroup_number
        if spacegroup_symbol is not None:
            params["spacegroup_symbol"] = spacegroup_symbol
        if exclude_elements is not None:
            params["exclude_elements"] = exclude_elements
        if possible_species is not None:
            params["possible_species"] = possible_species
        if has_props is not None:
            params["has_props"] = has_props
        if theoretical is not None:
            params["theoretical"] = theoretical
        if temperature is not None:
            params["temperature"] = temperature
        if pressure is not None:
            params["pressure"] = pressure
        params["page"] = page
        params["per_page"] = per_page
        
        result = self.handle_material_by_char(params)
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_material_by_char",
                "result": result
            })
        return result

    def handle_material_search(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material endpoint."""
        _log.info(f"GET materials/summary/get_material with params: {params}")
        
        kwargs = self._build_summary_search_kwargs(params)
        if "__errors__" in kwargs:
            return {
                "total_count": None,
                "error": {
                    "type": "invalid_parameter",
                    "message": "One or more range parameters are invalid.",
                    "details": kwargs["__errors__"],
                }
            }

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

        return {
            "total_count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "data": data,
        }

    def handle_material_by_char(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle materials/summary/get_material_by_char endpoint."""
        _log.info(f"GET materials/summary/get_material_by_char with params: {params}")
        
        kwargs = self._build_summary_search_kwargs(params)
        if "__errors__" in kwargs:
            return {
                "total_count": None,
                "error": {
                    "type": "invalid_parameter",
                    "message": "One or more range parameters are invalid.",
                    "details": kwargs["__errors__"],
                }
            }

        # Accept either identity selectors OR any numeric/range filters
        selector_keys = {
            "material_ids", "formula", "chemsys", "elements", "exclude_elements",
            "spacegroup_number", "spacegroup_symbol", "crystal_system", "magnetic_ordering"
        }
        range_selector_keys = set(self.RANGE_KEYS)

        has_selector = any(k in kwargs for k in selector_keys | range_selector_keys)

        if not has_selector:
            return {
                "total_count": None,
                "error": {
                    "type": "missing_parameter",
                    "message": "Provide at least one selector (e.g., formula/chemsys/elements/material_ids) "
                               "or a numeric/range filter (e.g., band_gap)."
                }
            }

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

        return {
            "total_count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "data": data,
        }


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
