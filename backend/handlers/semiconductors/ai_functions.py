"""
AI Functions for Semiconductor and Defect Analysis

This module contains all AI-accessible functions for analyzing semiconductors,
defects, doping, and structural properties.
"""

import json
import logging
import time
import numpy as np
from typing import Any, Dict, Annotated, Optional

from kani import ai_function, AIParam
from ..shared import success_result, error_result, ErrorType, Confidence
from .utils import (
    analyze_octahedral_distortion,
    get_magnetic_properties_detailed,
    compare_magnetic_properties,
    calculate_defect_formation_energy,
    analyze_doping_site_preference,
    analyze_structure_temperature_dependence,
    predict_site_preference
)
from ..magnets.utils import (
    fetch_phase_and_mp_data,
    assess_doping_effect_on_saturation_magnetization,
    analyze_doping_effect_on_ms
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
        start_time = time.time()
        
        try:
            # Get structure
            docs = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["material_id", "formula_pretty", "structure"]
            )
            
            if not docs:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="semiconductors",
                    function="analyze_octahedral_distortion_in_material",
                    error=f"Material {material_id} not found",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )
            
            doc = docs[0]
            structure = doc.structure if hasattr(doc, 'structure') else None
            
            if structure is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="semiconductors",
                    function="analyze_octahedral_distortion_in_material",
                    error="Structure not available for this material",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )
            
            util_result = analyze_octahedral_distortion(structure, central_element, neighbor_element)
            util_result["material_id"] = material_id
            util_result["formula"] = doc.formula_pretty if hasattr(doc, 'formula_pretty') else material_id
            
            duration_ms = (time.time() - start_time) * 1000
            
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="analyze_octahedral_distortion_in_material",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH,
                notes=["Distortion analysis based on coordination geometry from crystal structure"],
                duration_ms=duration_ms
            )
            
            self._track_tool_output("analyze_octahedral_distortion_in_material", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in analyze_octahedral_distortion_in_material: {e}", exc_info=True)
            return error_result(
                handler="semiconductors",
                function="analyze_octahedral_distortion_in_material",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
    
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
        start_time = time.time()
        
        util_result = get_magnetic_properties_detailed(self.mpr, material_id)
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="semiconductors",
                function="get_magnetic_properties",
                error=util_result.get("error", "Unknown error"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.API_ERROR,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="get_magnetic_properties",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH,
                duration_ms=duration_ms
            )
        
        self._track_tool_output("get_magnetic_properties", result)
        
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
        start_time = time.time()
        
        try:
            props1 = get_magnetic_properties_detailed(self.mpr, material_id_1)
            props2 = get_magnetic_properties_detailed(self.mpr, material_id_2)
            
            util_result = compare_magnetic_properties(props1, props2)
            
            duration_ms = (time.time() - start_time) * 1000
            
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="compare_magnetic_materials",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH,
                notes=["Comparison based on DFT magnetization data"],
                duration_ms=duration_ms
            )
            
            self._track_tool_output("compare_magnetic_materials", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error comparing magnetic materials: {e}", exc_info=True)
            return error_result(
                handler="semiconductors",
                function="compare_magnetic_materials",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )
    
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
        start_time = time.time()
        
        util_result = calculate_defect_formation_energy(
            self.mpr,
            host_material_id,
            defect_composition,
            defect_type
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="semiconductors",
                function="analyze_defect_stability",
                error=util_result.get("error", "Unknown error"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="analyze_defect_stability",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.MEDIUM,
                caveats=["Defect formation energy estimated from total energies", "Does not account for charged defects or Fermi level effects"],
                duration_ms=duration_ms
            )
        
        self._track_tool_output("analyze_defect_stability", result)
        
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
        start_time = time.time()
        
        util_result = analyze_doping_site_preference(
            self.mpr,
            host_formula,
            dopant_element,
            site_a_element,
            site_b_element,
            temperature,
            pressure
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="semiconductors",
                function="analyze_doping_site_preference",
                error=util_result.get("error", "Unknown error"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="analyze_doping_site_preference",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.MEDIUM,
                notes=["Site preference determined by comparing total energies"],
                caveats=["Assumes dilute doping limit", "Does not account for defect-defect interactions"],
                duration_ms=duration_ms
            )
        
        self._track_tool_output("analyze_doping_site_preference", result)
        
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
        start_time = time.time()
        
        util_result = analyze_structure_temperature_dependence(
            self.mpr,
            formula,
            element_of_interest,
            neighbor_element
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="semiconductors",
                function="analyze_phase_transition_structures",
                error=util_result.get("error", "Unknown error"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(util_result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="analyze_phase_transition_structures",
                data=data,
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.MEDIUM,
                notes=["Analyzes structural differences across polymorphs of the same composition"],
                duration_ms=duration_ms
            )
        
        self._track_tool_output("analyze_phase_transition_structures", result)
        
        return result
    
    async def _search_same_phase_doped_variants_core(
        self,
        host_formula: str,
        dopant_element: str,
        max_dopant_fraction: float,
        max_results: int
    ) -> Dict[str, Any]:
        """
        CORE HELPER: Search for plausible substitutional doped variants (NO LOGGING).
        
        This is the internal implementation that does NOT touch self.recent_tool_outputs.
        Use this when calling from other tools to avoid polluting the output log.
        """
        try:
            # 1. Get the host reference (stable-ish entry)
            host_mp = fetch_phase_and_mp_data(self.mpr, host_formula)
            if not host_mp.get("success"):
                return {
                    "success": False,
                    "error": f"Could not fetch host phase data for {host_formula}: {host_mp.get('error')}"
                }

            host_phase = host_mp.get("phase", {})
            host_sg = host_phase.get("space_group")
            host_cs = host_phase.get("crystal_system")
            host_comp = host_mp.get("composition", {})

            # Figure out allowed element set = {host elements} ∪ {dopant}
            host_elements = set(host_comp.keys())
            allowed_elements = set(host_elements) | {dopant_element}

            # 2. Broad search: host elements + dopant
            elements = list(host_elements | {dopant_element})
            docs = self.mpr.materials.summary.search(
                elements=elements,
                fields=[
                    "material_id", "formula_pretty", "composition",
                    "energy_above_hull", "is_stable", "symmetry",
                    "is_magnetic", "ordering", "total_magnetization",
                    "volume", "nsites"
                ]
            )

            if not docs:
                return {
                    "success": False,
                    "error": f"No materials found with elements {elements}"
                }

            candidates = []
            for d in docs:
                # --- chemistry gate ---
                comp = d.composition.as_dict() if hasattr(d, "composition") else {}
                doc_elements = set(comp.keys())
                # reject if this introduces any NEW cations not in allowed_elements
                if not doc_elements.issubset(allowed_elements):
                    continue

                # dopant must actually be present
                if dopant_element not in comp or comp[dopant_element] <= 0:
                    continue

                # --- dopant fraction gate (cation sublattice only) ---
                cation_elements = [el for el in comp if el not in ["O", "F", "N", "S", "Cl"]]
                total_cations = sum(comp[el] for el in cation_elements) or 1.0
                dop_frac = comp.get(dopant_element, 0.0) / total_cations
                if dop_frac > max_dopant_fraction:
                    # too much dopant = basically a new compound, not "doped host"
                    continue

                # --- oxygen ratio sanity gate ---
                # We try to keep O:(total cations) similar to host
                host_cat_total = sum(
                    host_comp[el] for el in host_comp if el not in ["O", "F", "N", "S", "Cl"]
                ) or 1.0
                host_O = host_comp.get("O", 0.0)
                host_ratio = host_O / host_cat_total

                this_cat_total = total_cations
                this_O = comp.get("O", 0.0)
                this_ratio = this_O / this_cat_total if this_cat_total > 0 else 0.0

                # allow ±15% relative drift in O:cation ratio
                if host_ratio > 0:
                    rel_dev = abs(this_ratio - host_ratio) / host_ratio
                    if rel_dev > 0.15:
                        # e.g. reject Fe2CoO4 spinel vs Fe2O3 corundum
                        continue

                # --- structural gate (phase similarity) ---
                sg = None
                cs = None
                if hasattr(d, "symmetry") and d.symmetry:
                    sg = getattr(d.symmetry, "symbol", None)
                    cs = getattr(d.symmetry, "crystal_system", None)

                # Stricter phase matching:
                # - If both have space groups, demand exact match
                # - Only fall back to crystal system if space group is missing
                same_phase_like = False
                phase_match_type = None
                
                if host_sg and sg:
                    # Both have explicit space groups - demand exact match
                    if sg == host_sg:
                        same_phase_like = True
                        phase_match_type = "space_group"
                elif host_cs and cs:
                    # Only fall back to crystal system match if at least one lacks space group
                    if str(cs) == str(host_cs):
                        same_phase_like = True
                        phase_match_type = "crystal_system_only"

                if not same_phase_like:
                    # don't claim "doped hematite" if it's actually a totally different lattice
                    continue

                # If it survives all filters, collect it
                entry = {
                    "material_id": getattr(d, "material_id", None),
                    "formula": getattr(d, "formula_pretty", str(d.composition)),
                    "composition": comp,
                    "effective_dopant_fraction_on_cation_lattice": float(dop_frac),
                    "energy_above_hull": float(getattr(d, "energy_above_hull", np.nan)),
                    "is_stable": bool(getattr(d, "is_stable", False)),
                }

                # Add explicit phase similarity note
                if phase_match_type == "space_group":
                    entry["phase_similarity_note"] = f"Matched by space group ({sg})"
                elif phase_match_type == "crystal_system_only":
                    entry["phase_similarity_note"] = (
                        f"Only matched by crystal system ({cs}); may represent a distorted or new phase"
                    )
                else:
                    entry["phase_similarity_note"] = "Unknown phase relationship"

                # stash magnetic info that might be useful later
                if hasattr(d, "is_magnetic"):
                    entry["is_magnetic"] = d.is_magnetic
                if hasattr(d, "ordering") and d.ordering:
                    entry["magnetic_ordering"] = str(d.ordering)
                if hasattr(d, "total_magnetization") and d.total_magnetization is not None:
                    entry["total_magnetization_muB_per_cell"] = float(d.total_magnetization)
                if hasattr(d, "volume"):
                    entry["volume_A3"] = float(d.volume)
                if hasattr(d, "nsites"):
                    entry["nsites"] = int(d.nsites)

                # symmetry
                if sg: entry["space_group"] = str(sg)
                if cs: entry["crystal_system"] = str(cs)

                candidates.append(entry)

            # sort by stability
            candidates.sort(key=lambda x: x.get("energy_above_hull", float("inf")))

            # Keep only candidates that actually matched the exact space group
            strict_candidates = [
                c for c in candidates
                if "phase_similarity_note" in c
                and c["phase_similarity_note"].startswith("Matched by space group")
            ]

            # If strict filter kills everything, we'll fall back to the looser set,
            # but we'll label them as "WARNING: possible new phase".
            phase_warning = None
            if strict_candidates:
                final_candidates = strict_candidates
            else:
                final_candidates = candidates
                phase_warning = (
                    "No doped entries with identical space group. "
                    "Using crystal-system-only matches, which may actually be a new/distorted phase. "
                    "Do NOT claim 'same phase doping' from this."
                )

            result = {
                "success": True,
                "host_formula": host_formula,
                "host_space_group": host_sg,
                "host_crystal_system": host_cs,
                "dopant_element": dopant_element,
                "num_candidates": len(final_candidates),
                "candidates": final_candidates[:max_results],
                "phase_warning": phase_warning,
                "note": (
                    "Strict filtering applied: demands exact space group match when available, "
                    "similar O:cation ratio (±15%), no extra cations, and small dopant fraction. "
                    "Each candidate includes 'phase_similarity_note' field indicating match quality. "
                    "DO NOT compare raw total_magnetization_muB_per_cell across candidates - "
                    "use magnet tools (assess_doping_effect_on_saturation_magnetization) for normalized Ms comparisons."
                ),
                "citations": ["Materials Project"]
            }

            # NO LOGGING in core helper - let the wrapper control that
            return result

        except Exception as e:
            _log.error(f"Error in _search_same_phase_doped_variants_core: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @ai_function(
        desc=(
            "Search for plausible SUBSTITUTIONAL doped variants of a host phase, not completely different compounds. "
            "For example: Fe2O3 with a few % Co on Fe sites (same crystal family), NOT spinels like Fe2CoO4. "
            "Filters candidates by: same crystal system / space group if possible, similar O:cation ratio, "
            "no extra cations beyond the dopant, and small dopant fraction. "
            "\n\nWARNING: Do NOT use this result alone to claim magnetic improvement. "
            "You MUST pass candidates into assess_doping_effect_on_saturation_magnetization to get normalized Ms (% change) "
            "or use compare_dopants_for_saturation_magnetization from the magnet tools."
        ),
        auto_truncate=128000
    )
    async def search_same_phase_doped_variants(
        self,
        host_formula: Annotated[str, AIParam(desc="Exact host formula (e.g., 'Fe2O3' for hematite).")],
        dopant_element: Annotated[str, AIParam(desc="Dopant element symbol (e.g., 'Co').")],
        max_dopant_fraction: Annotated[float, AIParam(desc="Max dopant fraction on the cation sublattice (e.g., 0.15 = 15%).")] = 0.15,
        max_results: Annotated[int, AIParam(desc="Maximum results to return.")] = 10
    ) -> Dict[str, Any]:
        """
        PUBLIC WRAPPER: Search for plausible substitutional doped variants.
        
        This is NOT a broad search - it strictly filters to avoid:
        - Ferrite spinels when you want corundum
        - Perovskites when you want rocksalt
        - New compounds with extra cations
        - Heavily doped phases (>15% dopant by default)
        
        Returns only candidates that preserve:
        - Crystal system / space group similarity
        - O:cation ratio similarity
        - Small dopant fraction on cation sublattice
        """
        start_time = time.time()
        
        # Call the core helper (which does NOT log)
        core_result = await self._search_same_phase_doped_variants_core(
            host_formula,
            dopant_element,
            max_dopant_fraction,
            max_results
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not core_result.get("success"):
            result = error_result(
                handler="semiconductors",
                function="search_same_phase_doped_variants",
                error=core_result.get("error", "Unknown error"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(core_result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in core_result.items() if k != "success"}
            result = success_result(
                handler="semiconductors",
                function="search_same_phase_doped_variants",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH,
                notes=["Filters candidates by crystal system and O:cation ratio similarity", "Only returns substitutional doping with small dopant fraction"],
                warnings=["Do not use this alone to claim magnetic improvement", "Must compare Ms values with proper phase checking"],
                duration_ms=duration_ms
            )
        
        # Log ONLY here, in the public wrapper
        self._track_tool_output("search_same_phase_doped_variants", result)
        
        return result
    
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
        start_time = time.time()
        
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
            
            duration_ms = (time.time() - start_time) * 1000
            
            data = {
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
                "diagnostics": res.diagnostics,
            }
            
            result = success_result(
                handler="semiconductors",
                function="predict_defect_site_preference",
                data=data,
                citations=["Zhang–Northrup defect formation framework", "Van de Walle–Neugebauer"],
                confidence=Confidence.MEDIUM if dft else Confidence.LOW,
                notes=res.notes if res.notes else ["Prediction based on physics-based heuristics" if not dft else "Prediction based on DFT formation energies"],
                caveats=["Heuristic estimates without DFT should be verified experimentally"] if not dft else [],
                duration_ms=duration_ms
            )
            
            self._track_tool_output("predict_defect_site_preference", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in predict_defect_site_preference: {e}", exc_info=True)
            return error_result(
                handler="semiconductors",
                function="predict_defect_site_preference",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Zhang–Northrup defect formation framework", "Van de Walle–Neugebauer"],
                duration_ms=duration_ms
            )
    
    @ai_function(
        desc=(
            "For a given host oxide/semiconductor and a dopant, "
            "find same-phase doped variants, then compute how the saturation magnetization "
            "changes in kA/m. Only trusts results where the doped cell has the SAME space group "
            "as the host. Refuses to claim improvement if the phase changes. "
            "\n\nThis is the ONLY correct way to answer questions about Ms retention or degradation."
        ),
        auto_truncate=128000
    )
    async def evaluate_dopant_effect_on_Ms(
        self,
        host_formula: Annotated[str, AIParam(desc="Host formula, e.g. 'Fe2O3' for hematite")],
        dopant_element: Annotated[str, AIParam(desc="Dopant symbol, e.g. 'Co'")],
        max_dopant_fraction: Annotated[float, AIParam(desc="Max cation fraction, default 0.15")] = 0.15,
    ) -> Dict[str, Any]:
        """
        Evaluate dopant effect on saturation magnetization with proper phase checking.
        
        This function:
        1. Gets the host phase data (correct ground state)
        2. Searches for doped variants that preserve the same space group
        3. Compares Ms in consistent units (kA/m)
        4. Flags any phase changes with strong warnings
        
        Use this instead of raw μB comparisons to get scientifically valid results.
        """
        start_time = time.time()
        
        try:
            host_phase_data = fetch_phase_and_mp_data(self.mpr, host_formula)
            phase_warning_global = None
            if not host_phase_data.get("success"):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="semiconductors",
                    function="evaluate_dopant_effect_on_Ms",
                    error=host_phase_data.get("error", "Failed to fetch host phase data"),
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )

            # find doped same-phase candidates using CORE helper (no logging)
            doped_search = await self._search_same_phase_doped_variants_core(
                host_formula,
                dopant_element,
                max_dopant_fraction,
                max_results=10
            )

            if not doped_search.get("success"):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="semiconductors",
                    function="evaluate_dopant_effect_on_Ms",
                    error=doped_search.get("error", "dopant search failed"),
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )

            if "phase_warning" in doped_search and doped_search["phase_warning"]:
                phase_warning_global = doped_search["phase_warning"]

            summaries = []
            for cand in doped_search["candidates"]:
                mid = cand.get("material_id")
                if not mid:
                    continue

                ms_eval = assess_doping_effect_on_saturation_magnetization(
                    self.mpr,
                    host_formula,
                    mid,
                    host_phase_data=host_phase_data
                )

                # stash metadata so we can interpret later
                ms_eval["candidate_formula"] = cand.get("formula")
                ms_eval["dopant_fraction_estimate"] = cand.get("effective_dopant_fraction_on_cation_lattice")
                ms_eval["phase_similarity_note"] = cand.get("phase_similarity_note")
                summaries.append(ms_eval)

            # >>> NEW FALLBACK BLOCK <<<
            # If no same-space-group candidates survived, fall back to broader analysis
            fallback_info = None
            if not summaries:
                _log.info(
                    f"No same-phase candidates for {host_formula}+{dopant_element}; "
                    f"using fallback analyze_doping_effect_on_ms"
                )
                fallback_analysis = analyze_doping_effect_on_ms(
                    host_formula=host_formula,
                    dopant=dopant_element,
                    doping_fraction=max_dopant_fraction,
                    mpr=self.mpr
                )
                fallback_info = fallback_analysis  # already has Ms_kA_per_m, phase_changed, etc.

            data = {
                "host_formula": host_formula,
                "dopant_element": dopant_element,
                "host_space_group": host_phase_data.get("phase", {}).get("space_group"),
                "results": summaries,
                "fallback_estimate": fallback_info,
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            notes_list = [
                "Only trust entries in 'results' where same_space_group == True and caution is None.",
                "If 'results' is empty but 'fallback_estimate' exists: the dopant seems to force a different phase or required a heuristic Ms estimate.",
                "If same_space_group is False in 'fallback_estimate', it's a new phase, not 'doped host'.",
                "Use percent_change only if it's not None (baseline Ms ≠ 0)."
            ]
            
            warnings_list = []
            if phase_warning_global:
                warnings_list.append(phase_warning_global)
            
            result = success_result(
                handler="semiconductors",
                function="evaluate_dopant_effect_on_Ms",
                data=data,
                citations=["Materials Project"],
                confidence=Confidence.HIGH if summaries else Confidence.LOW,
                notes=notes_list,
                warnings=warnings_list if warnings_list else None,
                caveats=["Only valid for same-phase doping (same space group)", "Different phases require separate analysis"],
                duration_ms=duration_ms
            )
            
            self._track_tool_output("evaluate_dopant_effect_on_Ms", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in evaluate_dopant_effect_on_Ms: {e}", exc_info=True)
            return error_result(
                handler="semiconductors",
                function="evaluate_dopant_effect_on_Ms",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )

