"""
AI Functions for Materials Search and Details

This module contains all AI-accessible functions for searching and retrieving
material information from the Materials Project database.
"""

import json
import logging
import time
from typing import Any, Dict, List, Annotated, Optional

from kani import ai_function, AIParam
from ..shared import success_result, error_result, ErrorType, Confidence
from .utils import (
    get_elastic_properties,
    find_closest_alloy_compositions,
    compare_material_properties_by_id,
    analyze_doping_effect
)

_log = logging.getLogger(__name__)


class MaterialsAIFunctionsMixin:
    """Mixin class containing AI function methods for Materials handlers."""
    
    @ai_function(desc="Search materials by their chemical system, formula, or elements. At least one of chemsys, formula, or element must be provided. Use chemical symbols directly (e.g., Li-Fe-O, Fe2O3, Li).", auto_truncate=128000)
    async def mp_search_by_composition(
        self,
        chemsys: Annotated[str, AIParam(desc="Chemical system(s) or comma-separated list (e.g., Li-Fe-O,Si-*). Use chemical symbols directly.")] = None,
        formula: Annotated[str, AIParam(desc="Formula(s), anonymized formula, or wildcard(s) (e.g., Li2FeO3,Fe2O3,Fe*O*). Use chemical symbols directly.")] = None,
        element: Annotated[str, AIParam(desc="Element(s) or comma-separated list (e.g., Li,Fe,O). Use chemical symbols directly.")] = None,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
    ) -> Dict[str, Any]:
        """Search materials by their chemical system, formula, or elements. At least one of chemsys, formula, or element must be provided. Use chemical symbols directly."""
        start_time = time.time()
        
        params = {}
        if chemsys is not None:
            params["chemsys"] = chemsys
        if formula is not None:
            params["formula"] = formula
        if element is not None:
            params["element"] = element
        params["page"] = page
        params["per_page"] = per_page
        
        result = self._handle_search_by_composition(params)
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        # Store the result for tooltip display
        self._track_tool_output("mp_search_by_composition", result)
        return result

    @ai_function(desc="Get materials by their Materials Project IDs. Returns material IDs and basic formula information.", auto_truncate=128000)
    async def mp_get_by_id(
        self,
        material_ids: Annotated[List[str], AIParam(desc="List of material IDs (e.g., ['mp-149', 'mp-30', 'mp-81']).")],
        fields: Annotated[List[str], AIParam(desc="List of fields to include. Basic info: 'material_id' (Materials Project ID), 'formula_pretty' (chemical formula), 'formula_anonymous' (stoichiometry pattern), 'chemsys' (chemical system), 'elements' (list of element symbols), 'num_elements' (number of elements), 'composition' (full composition dict), 'composition_reduced' (reduced composition), 'nsites' (number of sites in unit cell). Structural: 'structure' (crystal structure), 'volume' (unit cell volume in Å³), 'density' (in g/cm³), 'density_atomic' (atomic density), 'symmetry' (symmetry info), 'crystal_system', 'spacegroup_number', 'spacegroup_symbol'. Energetic: 'energy_per_atom' (total energy in eV/atom), 'formation_energy_per_atom' (formation energy in eV/atom), 'energy_above_hull' (stability indicator in eV/atom), 'is_stable' (on convex hull), 'uncorrected_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'decomposes_to'. Electronic: 'band_gap' (in eV), 'cbm' (conduction band minimum), 'vbm' (valence band maximum), 'efermi' (Fermi energy), 'is_gap_direct' (direct vs indirect), 'is_metal' (metallic behavior), 'bandstructure', 'dos' (density of states), 'dos_energy_up', 'dos_energy_down'. Magnetic: 'is_magnetic', 'ordering' (magnetic ordering type), 'total_magnetization' (in μ_B/atom), 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species'. Mechanical: 'bulk_modulus' (in GPa), 'shear_modulus' (in GPa), 'universal_anisotropy', 'homogeneous_poisson' (Poisson's ratio). Dielectric: 'e_total', 'e_ionic', 'e_electronic' (dielectric constants), 'n' (refractive index), 'piezoelectric_modulus', 'e_ij_max'. Surface: 'weighted_surface_energy' (in J/m²), 'weighted_work_function' (in eV), 'surface_anisotropy', 'shape_factor', 'has_reconstructed'. Metadata: 'builder_meta', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'task_ids', 'theoretical', 'has_props', 'possible_species', 'database_ids', 'property_name', 'xas', 'grain_boundaries', 'es_source_calc_id'. If not provided, returns basic fields: material_id, formula_pretty, elements, chemsys.")] = None,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
    ) -> Dict[str, Any]:
        """Get materials by their Materials Project IDs. Returns material IDs and basic formula information."""
        start_time = time.time()
        
        params = {
            "material_ids": material_ids,
            "all_fields": False,  # Only return basic info
            "page": page,
            "per_page": per_page
        }
        
        if fields is not None:
            params["fields"] = fields
        else:
            # Default to basic fields if not specified
            params["fields"] = ["material_id", "formula_pretty", "elements", "chemsys"]
        
        result = self._handle_material_details(params)
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        # Store the result for tooltip display
        self._track_tool_output("mp_get_by_id", result)
        return result

    @ai_function(desc="Fetch a material id and formula by a characteristic of the material", auto_truncate=128000)
    async def mp_get_by_characteristic(
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
        g_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss shear modulus in GPa. Tuple of floats, and both values are required.")] = None,
        g_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt shear modulus in GPa. Tuple of floats, and both values are required.")] = None,
        g_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill shear modulus in GPa. Tuple of floats, and both values are required.")] = None,
        k_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        k_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        k_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill bulk modulus in GPa. Tuple of floats, and both values are required.")] = None,
        n: Annotated[List[float], AIParam(desc="Min,max refractive index. Tuple of floats, and both values are required.")] = None,
        num_elements: Annotated[List[int], AIParam(desc="Min,max number of elements. Tuple of ints, and both values are required.")] = None,
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
        total_magnetization_normalized_vol: Annotated[List[float], AIParam(desc="Min,max total magnetization normalized to volume in μB/Å^3.")] = None,
        uncorrected_energy: Annotated[List[float], AIParam(desc="Min,max uncorrected energy in eV/atom. Tuple of floats, and both values are required.")] = None,
        volume: Annotated[List[float], AIParam(desc="Min,max volume in Å^3. Tuple of floats, and both values are required.")] = None,
        weighted_surface_energy: Annotated[List[float], AIParam(desc="Min,max weighted surface energy in J/m^2. Tuple of floats, and both values are required.")] = None,
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
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
    ) -> Dict[str, Any]:
        """Fetch a material id and formula by a characteristic of the material"""
        start_time = time.time()
        
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
        if num_elements is not None:
            params["num_elements"] = num_elements
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
        params["page"] = page
        params["per_page"] = per_page
        
        result = self._handle_get_by_characteristic(params)
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        # Store the result for tooltip display
        self._track_tool_output("mp_get_by_characteristic", result)
        return result

    @ai_function(desc="Fetch one or more materials by their material IDs and return detailed information about them.", auto_truncate=128000)
    async def mp_get_material_details(
        self,
        material_ids: Annotated[List[str], AIParam(desc="List of material IDs, e.g., ['mp-149', 'mp-150', 'mp-151'].")],
        fields: Annotated[List[str], AIParam(desc="List of fields to include. Basic info: 'material_id' (Materials Project ID), 'formula_pretty' (chemical formula), 'formula_anonymous' (stoichiometry pattern), 'chemsys' (chemical system), 'elements' (list of element symbols), 'num_elements' (number of elements), 'composition' (full composition dict), 'composition_reduced' (reduced composition), 'nsites' (number of sites in unit cell). Structural: 'structure' (crystal structure), 'volume' (unit cell volume in Å³), 'density' (in g/cm³), 'density_atomic' (atomic density), 'symmetry' (symmetry info), 'crystal_system', 'spacegroup_number', 'spacegroup_symbol'. Energetic: 'energy_per_atom' (total energy in eV/atom), 'formation_energy_per_atom' (formation energy in eV/atom), 'energy_above_hull' (stability indicator in eV/atom), 'is_stable' (on convex hull), 'uncorrected_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'decomposes_to'. Electronic: 'band_gap' (in eV), 'cbm' (conduction band minimum), 'vbm' (valence band maximum), 'efermi' (Fermi energy), 'is_gap_direct' (direct vs indirect), 'is_metal' (metallic behavior), 'bandstructure', 'dos' (density of states), 'dos_energy_up', 'dos_energy_down'. Magnetic: 'is_magnetic', 'ordering' (magnetic ordering type), 'total_magnetization' (in μ_B/atom), 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species'. Mechanical: 'bulk_modulus' (in GPa), 'shear_modulus' (in GPa), 'universal_anisotropy', 'homogeneous_poisson' (Poisson's ratio). Dielectric: 'e_total', 'e_ionic', 'e_electronic' (dielectric constants), 'n' (refractive index), 'piezoelectric_modulus', 'e_ij_max'. Surface: 'weighted_surface_energy' (in J/m²), 'weighted_work_function' (in eV), 'surface_anisotropy', 'shape_factor', 'has_reconstructed'. Metadata: 'builder_meta', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'task_ids', 'theoretical', 'has_props', 'possible_species', 'database_ids', 'property_name', 'xas', 'grain_boundaries', 'es_source_calc_id'.")] = None,
        all_fields: Annotated[bool, AIParam(desc="Whether to return all document fields. Useful if the user wants to know about the material without explicitly asking for certain fields (default True).")] = True,
        page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1,
        per_page: Annotated[int, AIParam(desc="Items per page (default 10).")] = 10
    ) -> Dict[str, Any]:
        """Fetch one or more materials by their material IDs and return detailed information about them."""
        start_time = time.time()
        
        params = {
            "material_ids": material_ids,
            "all_fields": all_fields,
            "page": page,
            "per_page": per_page
        }
        if fields is not None:
            params["fields"] = fields
        
        result = self._handle_material_details(params)
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        # Store the result for tooltip display
        self._track_tool_output("mp_get_material_details", result)
        return result

    @ai_function(desc="Get elastic and mechanical properties (bulk modulus, shear modulus, Poisson's ratio, Young's modulus, Pugh ratio) for a material. Supports two modes: (1) By material_id, or (2) By composition (element/formula/chemsys) + structure (spacegroup_number + crystal_system) with theoretical=False. Includes mechanical stability validation, derived properties computed from moduli, and optional tensor-based recomputation with Born stability information when elastic tensor is available.", auto_truncate=128000)
    async def get_elastic_properties(
        self,
        element: Annotated[Optional[str], AIParam(desc="Element(s) or comma-separated list (e.g., 'Li,Fe,O'). Use with spacegroup_number and crystal_system.")] = None,
        formula: Annotated[Optional[str], AIParam(desc="Formula (e.g., 'Li2FeO3', 'Fe2O3'). Use with spacegroup_number and crystal_system.")] = None,
        chemsys: Annotated[Optional[str], AIParam(desc="Chemical system (e.g., 'Li-Fe-O'). Use with spacegroup_number and crystal_system.")] = None,
        spacegroup_number: Annotated[Optional[int], AIParam(desc="Spacegroup number. Required when using composition mode (with element/formula/chemsys).")] = None,
        crystal_system: Annotated[Optional[str], AIParam(desc="Crystal system: 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', or 'Cubic'. Required when using composition mode.")] = None
    ) -> Dict[str, Any]:
        """Get elastic and mechanical properties including bulk modulus, shear modulus, Poisson's ratio, Young's modulus, Pugh ratio, mechanical stability assessment, and derived properties. Supports two modes: (1) Search by material_id, or (2) Search by composition (element/formula/chemsys) + structure (spacegroup_number + crystal_system) with theoretical=False. Automatically validates data quality and flags unphysical values (e.g., negative moduli). When elastic tensor is available, recomputes VRH values and provides Born stability information."""
        start_time = time.time()
        
        result = get_elastic_properties(
            self.mpr,
            element=element,
            formula=formula,
            chemsys=chemsys,
            spacegroup_number=spacegroup_number,
            crystal_system=crystal_system
        )
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        self._track_tool_output("get_elastic_properties", result)
        return result

    @ai_function(desc="Find materials with closest matching alloy compositions (e.g., Ag-Cu alloys near ~12.5% Cu). Returns closest match if exact composition not found.", auto_truncate=128000)
    async def find_closest_alloy_compositions(
        self,
        elements: Annotated[List[str], AIParam(desc="List of elements in the alloy, e.g., ['Ag', 'Cu'].")],
        target_composition: Annotated[Optional[Dict[str, float]], AIParam(desc="Target atomic fractions as a dictionary, e.g., {'Ag': 0.875, 'Cu': 0.125} for 12.5% Cu. If None, returns all compositions.")] = None,
        tolerance: Annotated[float, AIParam(desc="Tolerance for composition matching (default 0.05).")] = 0.05,
        is_stable: Annotated[bool, AIParam(desc="Whether to filter for stable materials only (default True).")] = True,
        ehull_max: Annotated[float, AIParam(desc="Maximum energy above hull for metastable entries in eV/atom (default 0.20).")] = 0.20,
        require_binaries: Annotated[bool, AIParam(desc="Whether to require exactly 2 elements (default True).")] = True
    ) -> Dict[str, Any]:
        """Find materials with closest matching alloy compositions."""
        start_time = time.time()
        
        result = find_closest_alloy_compositions(
            self.mpr,
            elements,
            target_composition,
            tolerance,
            is_stable,
            ehull_max,
            require_binaries
        )
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        self._track_tool_output("find_closest_alloy_compositions", result)
        return result

    @ai_function(desc="Compare a specific property (e.g., bulk modulus) between two materials.", auto_truncate=128000)
    async def compare_material_properties(
        self,
        material_id1: Annotated[str, AIParam(desc="First material ID.")],
        material_id2: Annotated[str, AIParam(desc="Second material ID.")],
        property_name: Annotated[str, AIParam(desc="Property to compare: 'bulk_modulus', 'shear_modulus', 'poisson_ratio', etc. (default 'bulk_modulus').")] = "bulk_modulus"
    ) -> Dict[str, Any]:
        """Compare a specific property between two materials and calculate percent change."""
        start_time = time.time()
        
        result = compare_material_properties_by_id(self.mpr, material_id1, material_id2, property_name)
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        self._track_tool_output("compare_material_properties", result)
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
        start_time = time.time()
        
        result = analyze_doping_effect(
            self.mpr,
            host_element,
            dopant_element,
            dopant_concentration,
            property_name
        )
        
        # Add duration to metadata if not already present
        duration_ms = (time.time() - start_time) * 1000
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "duration_ms" not in result["metadata"]:
                result["metadata"]["duration_ms"] = duration_ms
        
        self._track_tool_output("analyze_doping_effect", result)
        return result
