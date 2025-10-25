"""
AI Functions for Semiconductor and Defect Analysis

This module contains all AI-accessible functions for analyzing semiconductors,
defects, doping, and structural properties.
"""

import json
import logging
from typing import Any, Dict, List, Annotated, Optional

from kani import ai_function, AIParam
from .utils import (
    analyze_octahedral_distortion,
    get_magnetic_properties_detailed,
    compare_magnetic_properties,
    calculate_defect_formation_energy,
    analyze_doping_site_preference,
    analyze_structure_temperature_dependence,
    predict_site_preference
)

_log = logging.getLogger(__name__)


class SemiconductorAIFunctionsMixin:
    """Mixin class containing AI function methods for Semiconductor handlers."""
    
    @ai_function(
        desc="Analyze octahedral distortions in a crystal structure. Useful for understanding structural phase transitions and coordination environment changes with temperature.",
        auto_truncate=128000
    )
    async def analyze_octahedral_distortion_in_material(
        self,
        material_id: Annotated[str, AIParam(desc="Material ID (e.g., 'mp-1021522' for VO2).")],
        central_element: Annotated[str, AIParam(desc="Element at the center of octahedra (e.g., 'V' for vanadium).")],
        neighbor_element: Annotated[Optional[str], AIParam(desc="Element at the corners of octahedra (e.g., 'O' for oxygen). If None, finds most common neighbor.")] = None
    ) -> Dict[str, Any]:
        """
        Analyze octahedral distortions in a material structure.
        
        Returns detailed information about:
        - Bond lengths and deviations
        - Bond angles
        - Distortion parameters
        - Whether octahedra are regular or distorted
        """
        try:
            # Get structure
            docs = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["material_id", "formula_pretty", "structure"]
            )
            
            if not docs:
                return {
                    "success": False,
                    "error": f"Material {material_id} not found"
                }
            
            doc = docs[0]
            structure = doc.structure if hasattr(doc, 'structure') else None
            
            if structure is None:
                return {
                    "success": False,
                    "error": "Structure not available for this material"
                }
            
            result = analyze_octahedral_distortion(structure, central_element, neighbor_element)
            result["material_id"] = material_id
            result["formula"] = doc.formula_pretty if hasattr(doc, 'formula_pretty') else material_id
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "analyze_octahedral_distortion_in_material",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in analyze_octahedral_distortion_in_material: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Get detailed magnetic properties of a material including magnetization, magnetic ordering, and magnetic site information.",
        auto_truncate=128000
    )
    async def get_magnetic_properties(
        self,
        material_id: Annotated[str, AIParam(desc="Material ID (e.g., 'mp-19770' for Fe2O3).")]
    ) -> Dict[str, Any]:
        """
        Get comprehensive magnetic properties for a material.
        
        Returns:
        - Magnetic ordering (ferromagnetic, antiferromagnetic, etc.)
        - Total magnetization
        - Magnetization per volume and per formula unit
        - Number of magnetic sites
        - Magnetic species present
        """
        result = get_magnetic_properties_detailed(self.mpr, material_id)
        result["citations"] = ["Materials Project"]
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_magnetic_properties",
                "result": result
            })
        
        return result
    
    @ai_function(
        desc="Compare magnetic properties between two materials (e.g., undoped vs doped) to assess magnetic enhancement.",
        auto_truncate=128000
    )
    async def compare_magnetic_materials(
        self,
        material_id_1: Annotated[str, AIParam(desc="First material ID (e.g., undoped material).")],
        material_id_2: Annotated[str, AIParam(desc="Second material ID (e.g., doped material).")]
    ) -> Dict[str, Any]:
        """
        Compare magnetic properties between two materials.
        
        Useful for analyzing the effect of doping or composition changes on magnetism.
        Returns comparison of magnetization values and interpretation.
        """
        try:
            props1 = get_magnetic_properties_detailed(self.mpr, material_id_1)
            props2 = get_magnetic_properties_detailed(self.mpr, material_id_2)
            
            result = compare_magnetic_properties(props1, props2)
            result["citations"] = ["Materials Project"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "compare_magnetic_materials",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error comparing magnetic materials: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Analyze defect formation energy for substitutional or interstitial doping. Useful for comparing stability of different defect configurations.",
        auto_truncate=128000
    )
    async def analyze_defect_stability(
        self,
        host_material_id: Annotated[str, AIParam(desc="Material ID of host material (e.g., 'mp-149' for Si).")],
        defect_composition: Annotated[Dict[str, float], AIParam(desc="Composition with defect as a dictionary (e.g., {'Si': 31, 'P': 1} for P in Si).")],
        defect_type: Annotated[str, AIParam(desc="Type of defect: 'substitutional' or 'interstitial'.")] = "substitutional"
    ) -> Dict[str, Any]:
        """
        Analyze defect formation energy and stability.
        
        Compares energy of doped structure vs undoped host to estimate defect formation energy.
        Useful for determining whether interstitial or substitutional doping is more stable.
        """
        result = calculate_defect_formation_energy(
            self.mpr,
            host_material_id,
            defect_composition,
            defect_type
        )
        result["citations"] = ["Materials Project", "pymatgen"]
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "analyze_defect_stability",
                "result": result
            })
        
        return result
    
    @ai_function(
        desc="Analyze doping site preference in compound semiconductors (e.g., N in GaAs: does it prefer Ga or As sites?). Compares energy of dopant at different sublattice sites.",
        auto_truncate=128000
    )
    async def analyze_doping_site_preference(
        self,
        host_formula: Annotated[str, AIParam(desc="Host material formula (e.g., 'GaAs').")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element symbol (e.g., 'N' for nitrogen).")],
        site_a_element: Annotated[str, AIParam(desc="First potential substitution site element (e.g., 'Ga').")],
        site_b_element: Annotated[str, AIParam(desc="Second potential substitution site element (e.g., 'As').")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin (default: 298.15 K for STP).")] = 298.15,
        pressure: Annotated[float, AIParam(desc="Pressure in atm (default: 1.0 atm for STP).")] = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze which sublattice site a dopant prefers in compound semiconductors.
        
        For binary compounds AB, determines whether dopant prefers A sites or B sites
        by comparing energies of materials with dopant at each site.
        
        Returns which site is preferred and the energy difference.
        """
        result = analyze_doping_site_preference(
            self.mpr,
            host_formula,
            dopant_element,
            site_a_element,
            site_b_element,
            temperature,
            pressure
        )
        result["citations"] = ["Materials Project", "pymatgen"]
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "analyze_doping_site_preference",
                "result": result
            })
        
        return result
    
    @ai_function(
        desc="Analyze temperature-dependent structural changes by comparing different polymorphs. Useful for phase transition analysis.",
        auto_truncate=128000
    )
    async def analyze_phase_transition_structures(
        self,
        formula: Annotated[str, AIParam(desc="Chemical formula (e.g., 'VO2').")],
        element_of_interest: Annotated[Optional[str], AIParam(desc="Element to analyze coordination environment for (e.g., 'V').")] = None,
        neighbor_element: Annotated[Optional[str], AIParam(desc="Neighboring element in coordination environment (e.g., 'O').")] = None
    ) -> Dict[str, Any]:
        """
        Analyze structural differences across polymorphs/phases of a material.
        
        Searches for different crystal structures of the same composition and analyzes
        their structural properties including octahedral distortions if applicable.
        
        Useful for understanding temperature-dependent phase transitions.
        """
        result = analyze_structure_temperature_dependence(
            self.mpr,
            formula,
            element_of_interest,
            neighbor_element
        )
        result["citations"] = ["Materials Project", "pymatgen"]
        
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "analyze_phase_transition_structures",
                "result": result
            })
        
        return result
    
    @ai_function(
        desc="Search for doped materials by formula and analyze their properties. Useful for finding Al-doped Fe2O3 or other doped systems.",
        auto_truncate=128000
    )
    async def search_doped_materials(
        self,
        host_elements: Annotated[List[str], AIParam(desc="List of host elements (e.g., ['Fe', 'O'] for Fe2O3).")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element (e.g., 'Al').")],
        max_results: Annotated[int, AIParam(desc="Maximum number of results to return (default: 10).")] = 10
    ) -> Dict[str, Any]:
        """
        Search for materials containing both host elements and dopant.
        
        Returns materials that contain the host elements plus the dopant element,
        sorted by stability (energy above hull).
        """
        try:
            elements = host_elements + [dopant_element]
            
            docs = self.mpr.materials.summary.search(
                elements=elements,
                num_elements=[len(elements), len(elements)],
                fields=[
                    "material_id", "formula_pretty", "composition",
                    "energy_above_hull", "is_stable", "symmetry",
                    "is_magnetic", "ordering", "total_magnetization"
                ]
            )
            
            if not docs:
                return {
                    "success": False,
                    "error": f"No materials found with elements {elements}"
                }
            
            # Sort by energy above hull
            docs = sorted(docs, key=lambda x: getattr(x, 'energy_above_hull', float('inf')))[:max_results]
            
            materials = []
            for doc in docs:
                mat_data = {
                    "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                    "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
                    "composition": dict(doc.composition.as_dict()) if hasattr(doc, 'composition') else None,
                    "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None,
                    "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None
                }
                
                # Add symmetry
                if hasattr(doc, 'symmetry') and doc.symmetry:
                    sym = doc.symmetry
                    if hasattr(sym, 'crystal_system'):
                        mat_data["crystal_system"] = str(sym.crystal_system)
                    if hasattr(sym, 'symbol'):
                        mat_data["space_group"] = str(sym.symbol)
                
                # Add magnetic properties
                if hasattr(doc, 'is_magnetic'):
                    mat_data["is_magnetic"] = doc.is_magnetic
                if hasattr(doc, 'ordering') and doc.ordering:
                    mat_data["magnetic_ordering"] = str(doc.ordering)
                if hasattr(doc, 'total_magnetization') and doc.total_magnetization is not None:
                    mat_data["total_magnetization"] = float(doc.total_magnetization)
                
                materials.append(mat_data)
            
            result = {
                "success": True,
                "host_elements": host_elements,
                "dopant_element": dopant_element,
                "num_materials_found": len(materials),
                "materials": materials,
                "citations": ["Materials Project"]
            }
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "search_doped_materials",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error searching doped materials: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc=(
            "Predict whether a dopant prefers substitutional or interstitial sites in a semiconductor. "
            "By default uses physics-based heuristics; if you pass DFT formation energies, they take precedence."
        ),
        auto_truncate=128000,
    )
    async def predict_defect_site_preference(
        self,
        host: Annotated[str, AIParam(desc='Host element symbol, e.g., Si, Ga, Zn')],
        dopant: Annotated[str, AIParam(desc='Dopant element symbol, e.g., P, B, As')],
        mp_material_id: Annotated[Optional[str], AIParam(desc='Optional Materials Project ID for the host (for structure).')] = None,
        E_sub_eV: Annotated[Optional[float], AIParam(desc='Optional DFT formation energy for substitutional (eV).')] = None,
        E_int_tet_eV: Annotated[Optional[float], AIParam(desc='Optional DFT formation energy for tetra interstitial (eV).')] = None,
        E_int_oct_eV: Annotated[Optional[float], AIParam(desc='Optional DFT formation energy for octa interstitial (eV).')] = None,
    ) -> Dict[str, Any]:
        """
        Predict whether dopant prefers substitutional or interstitial sites.
        
        Uses physics-based heuristics considering:
        - Size mismatch (covalent radii)
        - Valence group differences
        - Electronegativity differences
        - Steric strain for interstitials
        
        If DFT formation energies are provided, they override heuristics.
        """
        try:
            dft = None
            if any(v is not None for v in (E_sub_eV, E_int_tet_eV, E_int_oct_eV)):
                dft = {
                    k: v for k, v in {
                        "sub": E_sub_eV, "int_tet": E_int_tet_eV, "int_oct": E_int_oct_eV
                    }.items() if v is not None
                }
            
            res = predict_site_preference(
                host=host,
                dopant=dopant,
                mpr=self.mpr,
                material_id=mp_material_id,
                dft_formation_energies=dft
            )
            
            result = {
                "success": res.success,
                "host": res.host,
                "dopant": res.dopant,
                "method": res.method,
                "preferred_site": res.preferred_site,
                "formation_energy_proxies_eV": {
                    "substitutional": res.E_sub_eV,
                    "interstitial_tetra": res.E_int_tet_eV,
                    "interstitial_hex": res.E_int_hex_eV  # hex for diamond, oct for fcc/hcp
                },
                "margin_eV": res.margin_eV,
                "verdict": res.verdict,
                "notes": res.notes,
                "diagnostics": res.diagnostics,
                "citations": ["Zhang–Northrup defect formation framework", "Van de Walle–Neugebauer"]
            }
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "predict_defect_site_preference",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in predict_defect_site_preference: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

