"""
AI Functions for Magnet Strength Assessment

This module contains all AI-accessible functions for assessing magnetic material
strength, particularly for permanent magnet applications. It evaluates whether
doping improves the "pull force" of a magnet through comprehensive analysis of:
- Phase stability and magnetic ordering
- Magnetic properties (Br, Hc, (BH)max, Ms, Bs)
- Pull force calculations with standard geometry
- Comparison of baseline vs doped materials
"""

import json
import logging
from typing import Any, Dict, Optional, Annotated

from kani import ai_function, AIParam
from .utils import (
    fetch_phase_and_mp_data,
    estimate_material_properties,
    calculate_pull_force_cylinder,
    assess_stronger_magnet
)

_log = logging.getLogger(__name__)


class MagnetAIFunctionsMixin:
    """Mixin class containing AI function methods for Magnet strength handlers."""
    
    @ai_function(
        desc="Assess whether doping makes a material a stronger permanent magnet. Evaluates pull force, coercivity, and magnetic ordering to determine if doped material can 'pull better' than baseline. Returns comprehensive analysis with verdict.",
        auto_truncate=128000
    )
    async def assess_magnet_strength_with_doping(
        self,
        host_formula: Annotated[str, AIParam(desc="Host material formula (e.g., 'Fe2O3' for hematite).")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element symbol (e.g., 'Al' for aluminum).")],
        doping_fraction: Annotated[float, AIParam(desc="Approximate doping fraction (e.g., 0.1 for 10% doping).")] = 0.1,
        magnet_diameter_mm: Annotated[float, AIParam(desc="Magnet diameter in mm for pull force calculation (default: 10 mm).")] = 10.0,
        magnet_length_mm: Annotated[float, AIParam(desc="Magnet length/height in mm (default: 10 mm).")] = 10.0,
        air_gap_mm: Annotated[float, AIParam(desc="Air gap between magnet and steel plate in mm (default: 0 for contact).")] = 0.0
    ) -> Dict[str, Any]:
        """
        Complete assessment of whether doping improves permanent magnet strength.
        
        This function:
        1. Identifies host and doped material phases, structures, and magnetic ordering
        2. Estimates magnetic properties (Br, Hc, (BH)max, Ms) for both materials
        3. Calculates pull force for a standard cylindrical geometry
        4. Provides verdict: is doped material "stronger" (better pull force)?
        
        "Stronger" means:
        - Higher pull force F (primarily from higher remanence Br)
        - Adequate coercivity Hc to retain magnetization
        - Suitable magnetic ordering (FM or FiM, not AFM/weak-FM)
        
        Returns comprehensive analysis with phase checks, property estimates,
        force calculations, and verdict with reasoning.
        """
        try:
            geometry = {
                "diameter_mm": float(magnet_diameter_mm),
                "length_mm": float(magnet_length_mm),
                "air_gap_mm": float(air_gap_mm)
            }
            
            result = assess_stronger_magnet(
                host_formula=host_formula,
                dopant=dopant_element,
                doping_fraction=float(doping_fraction),
                mpr=self.mpr,
                geometry=geometry,
                baseline_literature=None,
                doped_literature=None
            )
            
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "assess_magnet_strength_with_doping",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in assess_magnet_strength_with_doping: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Get phase and magnetic ordering information for a material. Returns crystal structure, space group, stability, and magnetic ordering type (FM/AFM/FiM/etc.).",
        auto_truncate=128000
    )
    async def get_phase_and_magnetic_ordering(
        self,
        formula: Annotated[str, AIParam(desc="Chemical formula (e.g., 'Fe2O3', 'FeAlO3').")]
    ) -> Dict[str, Any]:
        """
        Get phase information and magnetic ordering for a material.
        
        Returns:
        - Crystal structure (space group, crystal system)
        - Stability (energy above hull, is_stable)
        - Magnetic ordering (FM, AFM, FiM, WF, etc.)
        - Total magnetization
        - Number of magnetic sites
        """
        try:
            result = fetch_phase_and_mp_data(self.mpr, formula)
            result["citations"] = ["Materials Project"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "get_phase_and_magnetic_ordering",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in get_phase_and_magnetic_ordering: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Estimate permanent magnet properties (Br, Hc, (BH)max) from DFT data. Uses Materials Project magnetization to estimate saturation (Bs), remanence (Br), and max energy product. Provides heuristic coercivity estimates.",
        auto_truncate=128000
    )
    async def estimate_permanent_magnet_properties(
        self,
        formula: Annotated[str, AIParam(desc="Chemical formula (e.g., 'Fe2O3').")],
        kappa: Annotated[float, AIParam(desc="Remanence factor (Br/Bs ratio). Default 0.7 for hard magnets. Reduce for soft/weak-FM materials.")] = 0.7
    ) -> Dict[str, Any]:
        """
        Estimate permanent magnet properties from DFT magnetization data.
        
        Estimates:
        - Bs (saturation magnetization) from DFT total magnetization
        - Br (remanence) ≈ κ * Bs
        - (BH)max (max energy product) ≈ Br² / (4μ0)
        - Hc (coercivity) using ordering-based heuristics
        
        IMPORTANT CAVEATS:
        - DFT cannot predict coercivity accurately (microstructure-dependent)
        - Remanence estimates are rough; real values depend on processing
        - Use literature values when available
        """
        try:
            # First get phase and MP data
            mp_data = fetch_phase_and_mp_data(self.mpr, formula)
            
            if not mp_data.get("success"):
                return mp_data
            
            # Estimate properties
            result = estimate_material_properties(
                mp_data=mp_data,
                literature_hint=None,
                kappa=float(kappa)
            )
            
            # Add material identification
            result["material_id"] = mp_data.get("material_id")
            result["formula"] = mp_data.get("formula")
            result["magnetic_ordering"] = mp_data.get("magnetic_ordering")
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "estimate_permanent_magnet_properties",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in estimate_permanent_magnet_properties: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Calculate pull force for a cylindrical permanent magnet contacting steel. Uses remanence (Br) and geometry to estimate force in Newtons. Useful for comparing magnet strength in practical applications.",
        auto_truncate=128000
    )
    async def calculate_magnet_pull_force(
        self,
        remanence_tesla: Annotated[float, AIParam(desc="Remanence Br in Tesla (e.g., 0.2 for 0.2 T).")],
        diameter_mm: Annotated[float, AIParam(desc="Cylinder diameter in mm (default: 10 mm).")] = 10.0,
        length_mm: Annotated[float, AIParam(desc="Cylinder length/height in mm (default: 10 mm).")] = 10.0,
        air_gap_mm: Annotated[float, AIParam(desc="Air gap in mm (default: 0 for contact).")] = 0.0,
        eta: Annotated[float, AIParam(desc="Geometry factor (0.6-0.9, default: 0.7).")] = 0.7
    ) -> Dict[str, Any]:
        """
        Calculate pull force for a cylindrical magnet.
        
        Uses simplified magnetic circuit model:
        - Gap field: Bg ≈ η * Br (at contact)
        - Pull force: F ≈ Bg² * A / (2μ0)
        
        Returns force in Newtons and equivalent weight in kg.
        """
        try:
            result = calculate_pull_force_cylinder(
                Br_T=float(remanence_tesla),
                diameter_mm=float(diameter_mm),
                length_mm=float(length_mm),
                air_gap_mm=float(air_gap_mm),
                eta=float(eta)
            )
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "calculate_magnet_pull_force",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in calculate_magnet_pull_force: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Assess the effect of doping on saturation magnetization (Ms). Compares host and doped materials' Ms values to determine if doping increases, decreases, or maintains Ms. Returns quantitative change analysis.",
        auto_truncate=128000
    )
    async def assess_doping_effect_on_saturation_magnetization(
        self,
        host_formula: Annotated[str, AIParam(desc="Host material formula (e.g., 'Fe2O3', 'Co3O4').")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element symbol (e.g., 'Co', 'Fe', 'Nd').")],
        doping_fraction: Annotated[float, AIParam(desc="Approximate doping fraction (e.g., 0.1 for 10%).")] = 0.1
    ) -> Dict[str, Any]:
        """
        Assess how doping affects saturation magnetization (Ms).
        
        This function:
        1. Gets Ms for the host material
        2. Searches for or estimates Ms for the doped material
        3. Calculates the percentage change in Ms
        4. Analyzes whether the dopant enhances or degrades Ms
        
        Returns:
        - host_Ms: Host saturation magnetization
        - doped_Ms: Doped material saturation magnetization
        - Ms_change_percent: Percentage change in Ms
        - verdict: Whether doping increases/decreases Ms
        - analysis: Detailed reasoning based on magnetic moments and ordering
        """
        try:
            from .utils import (
                fetch_phase_and_mp_data,
                estimate_saturation_magnetization_T,
                analyze_doping_effect_on_ms
            )
            
            result = analyze_doping_effect_on_ms(
                host_formula=host_formula,
                dopant=dopant_element,
                doping_fraction=float(doping_fraction),
                mpr=self.mpr
            )
            
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "assess_doping_effect_on_saturation_magnetization",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in assess_doping_effect_on_saturation_magnetization: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Compare multiple dopants to find which causes the least degradation (or most enhancement) in saturation magnetization. Tests multiple dopant elements and ranks them by Ms retention/improvement.",
        auto_truncate=128000
    )
    async def compare_dopants_for_saturation_magnetization(
        self,
        host_formula: Annotated[str, AIParam(desc="Host material formula (e.g., 'Fe2O3').")],
        dopant_elements: Annotated[list, AIParam(desc="List of dopant elements to compare (e.g., ['Ni', 'Co', 'Nd']).")],
        doping_fraction: Annotated[float, AIParam(desc="Doping fraction to test (default: 0.1).")] = 0.1
    ) -> Dict[str, Any]:
        """
        Compare multiple dopants to find which one produces the least degradation in Ms.
        
        For each dopant:
        1. Calculates or estimates Ms for the doped material
        2. Computes Ms change percentage
        3. Ranks dopants from best (least degradation/most enhancement) to worst
        
        Returns:
        - host_Ms: Baseline Ms value
        - dopant_comparison: List of dopants with their Ms effects, sorted by performance
        - best_dopant: Dopant with least degradation or most enhancement
        - worst_dopant: Dopant with most degradation
        """
        try:
            from .utils import compare_multiple_dopants_ms
            
            result = compare_multiple_dopants_ms(
                host_formula=host_formula,
                dopants=dopant_elements,
                doping_fraction=float(doping_fraction),
                mpr=self.mpr
            )
            
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "compare_dopants_for_saturation_magnetization",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in compare_dopants_for_saturation_magnetization: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Get detailed saturation magnetization data for a material, including Ms in various units (T, A/m, emu/g), magnetic ordering, and magnetic site information.",
        auto_truncate=128000
    )
    async def get_saturation_magnetization_detailed(
        self,
        formula: Annotated[str, AIParam(desc="Chemical formula (e.g., 'Fe2O3', 'Co3O4').")]
    ) -> Dict[str, Any]:
        """
        Get comprehensive saturation magnetization data for a material.
        
        Returns:
        - Ms in multiple units: Tesla, A/m, kA/m, emu/g
        - Bs (= μ0 * Ms) in Tesla
        - Total magnetization per unit cell (μB)
        - Magnetic ordering type
        - Magnetic species and site information
        - Crystal structure
        """
        try:
            from .utils import get_detailed_saturation_magnetization
            
            result = get_detailed_saturation_magnetization(
                mpr=self.mpr,
                formula=formula
            )
            
            result["citations"] = ["Materials Project", "pymatgen"]
            
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "get_saturation_magnetization_detailed",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error in get_saturation_magnetization_detailed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc="Search for doped versions of a host material by specifying host elements and dopant. Returns materials from Materials Project that contain all specified elements, sorted by stability and structural similarity.",
        auto_truncate=128000
    )
    async def search_doped_magnetic_materials(
        self,
        host_elements: Annotated[list, AIParam(desc="List of host elements (e.g., ['Fe', 'O'] for Fe2O3).")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element (e.g., 'Al').")],
        max_results: Annotated[int, AIParam(desc="Maximum number of results (default: 10).")] = 10,
        host_crystal_system: Annotated[Optional[str], AIParam(desc="Optional host crystal system to prefer similar structures.")] = None
    ) -> Dict[str, Any]:
        """
        Search for materials containing host elements plus dopant.
        
        Chemistry-aware search that filters out materials with extra cations.
        Only returns materials where element set ⊆ {host elements} ∪ {dopant}.
        
        Works generically for any host: Fe2O3+Co, BaTiO3+Zr, AlN+Ti, etc.
        """
        try:
            elements = host_elements + [dopant_element]
            
            docs = self.mpr.materials.summary.search(
                elements=elements,
                fields=[
                    "material_id", "formula_pretty", "composition",
                    "energy_above_hull", "is_stable", "symmetry",
                    "is_magnetic", "ordering", "total_magnetization",
                    "volume", "nsites"
                ],
            )
            
            if not docs:
                return {
                    "success": False,
                    "error": f"No materials found with elements {elements}"
                }
            
            # Chemistry constraint: only allow {host elements} ∪ {dopant}
            # This prevents SrPrFeCoO6 from being treated as "Co-doped Fe2O3"
            host_set = set(host_elements)
            allowed_elements = host_set | {dopant_element}
            
            filtered_docs = []
            for doc in docs:
                comp = doc.composition.as_dict()
                doc_elements = set(comp.keys())
                
                # 1. No extra cations beyond dopant
                if not doc_elements.issubset(allowed_elements):
                    continue
                
                # 2. Dopant must be present
                if dopant_element not in comp or comp[dopant_element] <= 0:
                    continue
                
                filtered_docs.append(doc)
            
            # Sort by stability with optional structural similarity bonus
            def stability_rank(doc):
                """
                Ranking function: lower is better.
                Prefer low energy_above_hull AND same crystal system as host.
                """
                e_hull = getattr(doc, 'energy_above_hull', float('inf'))
                if e_hull is None:
                    e_hull = float('inf')
                
                # Small bonus for matching crystal system (keeps same lattice type)
                same_cs_bonus = 0.0
                if host_crystal_system and hasattr(doc, 'symmetry') and doc.symmetry:
                    doc_cs = getattr(doc.symmetry, 'crystal_system', None)
                    if doc_cs and str(doc_cs) == host_crystal_system:
                        same_cs_bonus = -0.5  # Small advantage
                
                return e_hull + same_cs_bonus
            
            filtered_docs = sorted(filtered_docs, key=stability_rank)[:max_results]
            
            materials = []
            for doc in filtered_docs:
                comp_dict = doc.composition.as_dict() if hasattr(doc, 'composition') else {}
                total_atoms = sum(comp_dict.values())
                dopant_frac = comp_dict.get(dopant_element, 0) / total_atoms if total_atoms > 0 else 0
                
                mat_data = {
                    "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                    "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
                    "composition": comp_dict,
                    "dopant_fraction": float(dopant_frac),
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
                    mat_data["total_magnetization_muB"] = float(doc.total_magnetization)
                
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
                    "tool_name": "search_doped_magnetic_materials",
                    "result": result
                })
            
            return result
            
        except Exception as e:
            _log.error(f"Error searching doped magnetic materials: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

