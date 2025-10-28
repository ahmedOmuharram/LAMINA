"""
AI Functions for Materials Search and Details

This module contains all AI-accessible functions for searching and retrieving
material information from the Materials Project database.
"""

import json
import logging
from typing import Any, Dict, List, Annotated, Optional

from kani import ai_function, AIParam
from ..base.result_wrappers import success_result, error_result, ErrorType, Confidence
from .utils import (
    get_elastic_properties,
    find_alloy_compositions,
    compare_material_properties,
    analyze_doping_effect
)

_log = logging.getLogger(__name__)


class MaterialsAIFunctionsMixin:
    """Mixin class containing AI function methods for Materials handlers."""
    
    @ai_function(desc="Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Use chemical symbols directly (e.g., Li-Fe-O, Fe2O3, Li).", auto_truncate=128000)
    async def get_material(
        self,
        chemsys: Annotated[str, AIParam(desc="Chemical system(s) or comma-separated list (e.g., Li-Fe-O,Si-*). Use chemical symbols directly.")] = None,
        formula: Annotated[str, AIParam(desc="Formula(s), anonymized formula, or wildcard(s) (e.g., Li2FeO3,Fe2O3,Fe*O*). Use chemical symbols directly.")] = None,
        element: Annotated[str, AIParam(desc="Element(s) or comma-separated list (e.g., Li,Fe,O). Use chemical symbols directly.")] = None,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
    ) -> Dict[str, Any]:
        """Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Use chemical symbols directly."""
        params = {}
        if chemsys is not None:
            params["chemsys"] = chemsys
        if formula is not None:
            params["formula"] = formula
        if element is not None:
            params["element"] = element
        params["page"] = page
        params["per_page"] = per_page
        
        util_result = self.handle_material_search(params)
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="get_material",
                error=util_result.get("error", "Material search failed"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="get_material",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH
            )
        
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
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
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
        
        util_result = self.handle_material_by_char(params)
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="get_material_by_char",
                error=util_result.get("error", "Material search by characteristics failed"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="get_material_by_char",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH
            )
        
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_material_by_char",
                "result": result
            })
        return result

    @ai_function(desc="Fetch one or more materials by their material IDs and return detailed information about them.", auto_truncate=128000)
    async def get_material_details_by_ids(
        self,
        material_ids: Annotated[List[str], AIParam(desc="List of material IDs, e.g., ['mp-149', 'mp-150', 'mp-151'].")],
        fields: Annotated[List[str], AIParam(desc="List of fields to include. Values include 'builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_Ids'")] = None,
        all_fields: Annotated[bool, AIParam(desc="Whether to return all document fields. Useful if the user wants to know about the material without explicitly asking for certain fields (default True).")] = True,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
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
        
        util_result = self.handle_material_details(params)
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="get_material_details_by_ids",
                error=util_result.get("error", "Failed to fetch material details"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="get_material_details_by_ids",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH
            )
        
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_material_details_by_ids",
                "result": result
            })
        return result

    @ai_function(desc="Get elastic and mechanical properties (bulk modulus, shear modulus, etc.) for a material.", auto_truncate=128000)
    async def get_elastic_properties(
        self,
        material_id: Annotated[str, AIParam(desc="Material ID (e.g., 'mp-81' for Ag, 'mp-30' for Cu).")]
    ) -> Dict[str, Any]:
        """Get elastic and mechanical properties including bulk modulus, shear modulus, Poisson's ratio, etc."""
        util_result = get_elastic_properties(self.mpr, material_id)
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="get_elastic_properties",
                error=util_result.get("error", "Failed to get elastic properties"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project", "pymatgen"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="get_elastic_properties",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH
            )
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_elastic_properties",
                "result": result
            })
        return result

    @ai_function(desc="Find materials with specific alloy compositions (e.g., Ag-Cu alloys with ~12.5% Cu).", auto_truncate=128000)
    async def find_alloy_compositions(
        self,
        elements: Annotated[List[str], AIParam(desc="List of elements in the alloy, e.g., ['Ag', 'Cu'].")],
        target_composition: Annotated[Optional[Dict[str, float]], AIParam(desc="Target atomic fractions as a dictionary, e.g., {'Ag': 0.875, 'Cu': 0.125} for 12.5% Cu. If None, returns all compositions.")] = None,
        tolerance: Annotated[float, AIParam(desc="Tolerance for composition matching (default 0.05).")] = 0.05,
        is_stable: Annotated[bool, AIParam(desc="Whether to filter for stable materials only (default True).")] = True,
        ehull_max: Annotated[float, AIParam(desc="Maximum energy above hull for metastable entries in eV/atom (default 0.20).")] = 0.20,
        require_binaries: Annotated[bool, AIParam(desc="Whether to require exactly 2 elements (default True).")] = True
    ) -> Dict[str, Any]:
        """Find materials with specific alloy compositions."""
        util_result = find_alloy_compositions(
            self.mpr,
            elements,
            target_composition,
            tolerance,
            is_stable,
            ehull_max,
            require_binaries
        )
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="find_alloy_compositions",
                error=util_result.get("error", "Failed to find alloy compositions"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project", "pymatgen"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="find_alloy_compositions",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH
            )
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "find_alloy_compositions",
                "result": result
            })
        return result

    @ai_function(desc="Compare a specific property (e.g., bulk modulus) between two materials.", auto_truncate=128000)
    async def compare_material_properties(
        self,
        material_id1: Annotated[str, AIParam(desc="First material ID.")],
        material_id2: Annotated[str, AIParam(desc="Second material ID.")],
        property_name: Annotated[str, AIParam(desc="Property to compare: 'bulk_modulus', 'shear_modulus', 'poisson_ratio', etc. (default 'bulk_modulus').")] = "bulk_modulus"
    ) -> Dict[str, Any]:
        """Compare a specific property between two materials and calculate percent change."""
        # Get properties for both materials
        props1 = get_elastic_properties(self.mpr, material_id1)
        props2 = get_elastic_properties(self.mpr, material_id2)
        
        util_result = compare_material_properties(props1, props2, property_name)
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="compare_material_properties",
                error=util_result.get("error", "Failed to compare properties"),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="compare_material_properties",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH
            )
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "compare_material_properties",
                "result": result
            })
        return result

    @ai_function(desc="Analyze the effect of doping a host material with a dopant element on a specific property.", auto_truncate=128000)
    async def analyze_doping_effect(
        self,
        host_element: Annotated[str, AIParam(desc="Host element symbol (e.g., 'Ag').")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element symbol (e.g., 'Cu').")],
        dopant_concentration: Annotated[float, AIParam(desc="Dopant atomic fraction (e.g., 0.125 for 12.5% doping).")],
        property_name: Annotated[str, AIParam(desc="Property to analyze: 'bulk_modulus', 'shear_modulus', etc. (default 'bulk_modulus').")] = "bulk_modulus"
    ) -> Dict[str, Any]:
        """Analyze how doping a host material affects a specific property, comparing pure vs doped materials."""
        util_result = analyze_doping_effect(
            self.mpr,
            host_element,
            dopant_element,
            dopant_concentration,
            property_name
        )
        
        if not util_result.get("success"):
            result = error_result(
                handler="materials",
                function="analyze_doping_effect",
                error=util_result.get("error", "Failed to analyze doping effect"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"]
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="materials",
                function="analyze_doping_effect",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.MEDIUM,
                notes=["Comparison between pure and doped materials"]
            )
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "analyze_doping_effect",
                "result": result
            })
        return result
