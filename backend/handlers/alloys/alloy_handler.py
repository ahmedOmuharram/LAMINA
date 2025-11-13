"""
Alloy surface and microstructure analysis handler.

This handler provides AI-accessible functions for:
1. Surface diffusion barrier estimation for adatoms on metal surfaces
2. Phase strength and stiffness claims assessment using CALPHAD and Materials Project

The implementation is modular, with specialized utilities organized into separate modules.
"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional, Annotated
from kani import ai_function, AIParam

from ..base import BaseHandler
from ..shared import success_result, error_result, ErrorType, Confidence
from ..shared.constants import (
    ADS_OVER_COH_111,
    ADS_OVER_COH_100,
    ADS_OVER_COH_110,
)

from .atomic_utils import get_cohesive_energy, get_metal_radius
from .surface_utils import (
    evaluate_scenario,
    generate_scenarios,
    generate_warnings
)
from .composition_utils import parse_system_string, parse_composition_string
from .mechanical_utils import get_phase_mechanical_descriptors
from .stiffness_utils import estimate_phase_modulus
from .assessment_utils import assess_mechanical_effects
from .verification_utils import verify_claims
from ..shared.calphad_utils import compute_equilibrium_microstructure

_log = logging.getLogger(__name__)


class AlloyHandler(BaseHandler):
    """
    Alloy surface heuristics for adatom diffusion barriers and microstructure analysis.
    
    This handler provides two main capabilities:
    1. Surface diffusion barrier estimation using descriptor-based models
    2. Comprehensive alloy microstructure and mechanical property assessment
    """

    def __init__(self, mpr_client: Optional[object] = None, **kwargs) -> None:
        # Store mpr_client as mpr for BaseHandler
        if mpr_client is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr_client
        super().__init__(**kwargs)
        # Also set self.mpr if needed (for compatibility)
        if mpr_client is not None:
            self.mpr = mpr_client
        _log.info("AlloyHandler initialized")

    @ai_function(
        desc=(
            "Comprehensive surface diffusion barrier estimation using DFT+NEB calculations across multiple "
            "realistic scenarios (facets, mechanisms, defects, coverage), with uncertainty quantification and kinetics. "
            "By default uses first-principles DFT+NEB for terrace diffusion on (111), (100), (110) facets. "
            "Returns ranked scenarios with P10/P50/P90 uncertainty bands, kinetic rates, and "
            "diffusion coefficients at specified temperature. Use this for quantitative predictions "
            "of all relevant diffusion pathways."
        ),
        auto_truncate=128000,
    )
    async def estimate_surface_diffusion_barrier(
        self,
        adatom_element: Annotated[str, AIParam(desc="Adatom element, e.g., 'Au'.")],
        host_element: Annotated[str, AIParam(desc="Host surface element, e.g., 'Al'.")],
        surface_millers: Annotated[Optional[str], AIParam(desc="Comma-separated surfaces like '111,100,110'. Omit for all three.")] = None,
        include_defects: Annotated[bool, AIParam(desc="Also evaluate step-edge and vacancy-mediated paths")] = True,
        coverage_theta: Annotated[float, AIParam(desc="Adatom coverage (ML, 0–1). Influences crowding")] = 0.0,
        oxide: Annotated[bool, AIParam(desc="If True, apply oxide penalty (e.g., native Al2O3)")] = False,
        temperature_K: Annotated[Optional[float], AIParam(desc="Temperature for kinetics (Arrhenius)")] = 300.0,
        attempt_frequency_Hz: Annotated[float, AIParam(desc="Attempt frequency ν, s^-1")] = 1.0e13,
        jump_distance_A: Annotated[float, AIParam(desc="Jump distance (Å) for D estimate")] = 2.5,
        mc_samples: Annotated[int, AIParam(desc="Monte Carlo samples for uncertainty quantification (0 disables)")] = 200,
        use_dft_neb: Annotated[bool, AIParam(desc="Use DFT+NEB for barrier calculations (default True for terrace diffusion)")] = True,
        dft_backend: Annotated[str, AIParam(desc="Calculator: 'chgnet' or 'gpaw'. Default: 'chgnet'. Do not use 'gpaw' unless you are explicitly told to do so.")] = "chgnet",
        dft_images: Annotated[int, AIParam(desc="Number of NEB images")] = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive surface diffusion analysis with multi-scenario enumeration using DFT+NEB.
        
        This function explores a grid of realistic scenarios:
        - Facets: (111), (100), (110) surfaces (all by default)
        - Mechanisms: hopping vs exchange
        - Defects: terrace vs step-edge vs vacancy-mediated
        - Coverage effects: crowding penalties
        - Surface conditions: clean vs oxide
        
        By default, uses first-principles DFT+NEB calculations for terrace diffusion
        (defect='none') on all requested facets. Defect scenarios use heuristic scaling.
        
        Returns:
        - Ranked list of all scenarios with P10/P50/P90 energy ranges
        - Kinetic rates and diffusion coefficients at specified temperature
        - Per-facet minimum barriers
        - Consensus range across all scenarios
        - Warnings for problematic cases (large size mismatch, oxides, etc.)
        - DFT details (backend, k-points, energy profiles) for NEB calculations
        """
        start_time = time.time()
        dft_kpts = "2,2,1"
    
        try:
            ad = adatom_element.capitalize()
            host = host_element.capitalize()
            
            # Get atomic descriptors
            ecoh_ad, src_ad = get_cohesive_energy(ad)
            ecoh_host, src_host = get_cohesive_energy(host)
            r_ad = get_metal_radius(ad)
            r_host = get_metal_radius(host)
            
            if ecoh_host is None or r_ad is None or r_host is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="alloys",
                    function="estimate_surface_diffusion_barrier",
                    error="Insufficient data: missing cohesive energy or metallic radius for host/adatom.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["mendeleev", "pymatgen"],
                    diagnostics={
                        "missing": {
                            "E_coh_host": ecoh_host,
                            "r_ad": r_ad,
                            "r_host": r_host
                        }
                    },
                    duration_ms=duration_ms
                )
            
            # Determine which facets to evaluate
            if surface_millers:
                facets_req = [s.strip() for s in surface_millers.split(",") if s.strip()]
            else:
                facets_req = ["111", "100", "110"]
            
            # Facet-specific adsorption/cohesion scaling ratios
            ads_over_coh_dict = {
                "111": ADS_OVER_COH_111,
                "100": ADS_OVER_COH_100,
                "110": ADS_OVER_COH_110,
                "unspecified": 0.25
            }
            
            # Parse k-points string to tuple
            try:
                kpts_tuple = tuple(int(k.strip()) for k in dft_kpts.split(","))
                if len(kpts_tuple) != 3:
                    kpts_tuple = (2, 2, 1)  # fallback
            except:
                kpts_tuple = (2, 2, 1)  # fallback
            
            # Generate scenario grid
            scenario_tuples = generate_scenarios(facets_req, include_defects)
            
            # Evaluate all scenarios
            scenarios = []
            for facet, mechanism, defect in scenario_tuples:
                scenario = evaluate_scenario(
                    ecoh_host=ecoh_host,
                    r_ad=r_ad,
                    r_host=r_host,
                    facet_label=facet,
                    mechanism=mechanism,
                    defect=defect,
                    coverage_theta=coverage_theta,
                    oxide=oxide,
                    temperature_K=temperature_K,
                    attempt_frequency_Hz=attempt_frequency_Hz,
                    jump_distance_A=jump_distance_A,
                    mc_samples=mc_samples,
                    ads_over_coh_dict=ads_over_coh_dict,
                    use_dft_neb=use_dft_neb,
                    dft_backend=dft_backend,
                    dft_kpts=kpts_tuple,
                    dft_images=dft_images,
                    adatom_symbol=ad,
                    host_symbol=host
                )
                scenarios.append(scenario)
            
            # Sort by median barrier (P50)
            scenarios.sort(key=lambda s: s["Ea_eV"]["P50"])
            
            # Per-facet minima (lowest barrier for each facet)
            facet_summary = {}
            for s in scenarios:
                f = s["facet"]
                if f not in facet_summary or s["Ea_eV"]["P50"] < facet_summary[f]["Ea_eV"]["P50"]:
                    facet_summary[f] = s
            
            # Consensus range (global P10 to P90)
            all_p10 = min(s["Ea_eV"]["P10"] for s in scenarios)
            all_p90 = max(s["Ea_eV"]["P90"] for s in scenarios)
            consensus = {
                "Ea_eV_min_P10": round(all_p10, 4),
                "Ea_eV_max_P90": round(all_p90, 4)
            }
            
            # Generate warnings
            warnings = generate_warnings(host, r_ad, r_host, oxide)
            
            # Confidence rubric
            confidence = "medium"
            if surface_millers and all(f in ("111", "100", "110") for f in facets_req):
                confidence = "high"
            if oxide and host in ("Al", "Ti", "Mg"):
                confidence = "medium"  # oxide model is coarse
            
            duration_ms = (time.time() - start_time) * 1000
            
            return success_result(
                handler="alloys",
                function="estimate_surface_diffusion_barrier",
                data={
                    "adatom": ad,
                    "host": host,
                    "scenarios_ranked": scenarios,  # Full table sorted by P50
                    "facet_minima": facet_summary,  # Best scenario per facet
                    "consensus_range": consensus,   # Global P10–P90
                    "assumptions": {
                        "include_defects": include_defects,
                        "coverage_theta": coverage_theta,
                        "oxide": oxide,
                        "temperature_K": temperature_K,
                        "attempt_frequency_Hz": attempt_frequency_Hz,
                        "jump_distance_A": jump_distance_A,
                        "mc_samples": mc_samples
                    },
                    "descriptors": {
                        "cohesive_energy_host_eV": round(ecoh_host, 4),
                        "cohesive_energy_adatom_eV": round(ecoh_ad, 4) if ecoh_ad is not None else None,
                        "size_mismatch_rel": round(abs((r_ad - r_host) / r_host), 4),
                        "adatom_radius_A": round(r_ad, 4),
                        "host_radius_A": round(r_host, 4)
                    },
                    "data_sources": {
                        "cohesive_energy": {
                            "adatom": src_ad,
                            "host": src_host,
                        },
                        "radii": "mendeleev metallic radii (pm→Å), else pymatgen (Å)",
                    },
                },
                citations=["mendeleev", "pymatgen"],
                confidence=confidence,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in estimate_surface_diffusion_barrier for {adatom_element} on {host_element}: {e}", exc_info=True)
            return error_result(
                handler="alloys",
                function="estimate_surface_diffusion_barrier",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["mendeleev", "pymatgen"],
                duration_ms=duration_ms
            )

    @ai_function(
        desc=(
            "Assess claims about alloy microstructure, strengthening, embrittlement, and stiffness. "
            "Given a binary alloy system, composition, and temperature, this tool: "
            "(1) calculates which phases are at equilibrium using CALPHAD, "
            "(2) determines which is the matrix vs secondary phases, "
            "(3) estimates mechanical properties (brittleness) using Materials Project elastic data, "
            "(4) assesses whether secondary phases likely strengthen or embrittle the alloy, "
            "(5) estimates the elastic modulus (stiffness) of the matrix phase using rule-of-mixtures, "
            "(6) verifies claims about phase formation, mechanical effects, and stiffness changes. "
            "Works for Al-Mg, Fe-Al, Ni-Al, Ti-Al, and other binary metallic systems."
        ),
        auto_truncate=128000,
    )
    async def assess_phase_strength_and_stiffness_claims(
        self,
        system: Annotated[str, AIParam(desc="Chemical system, e.g. 'Fe-Al', 'Ni-Al', 'Ti-Al' (binary) or 'Al-Mg-Zn', 'Fe-Cr-Ni' (ternary)")],
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Fe30Al70', 'Al88Mg8Zn4'). Numbers are at.%")],
        temperature_K: Annotated[float, AIParam(desc="Temperature in Kelvin")],
        claimed_secondary_phase: Annotated[Optional[str], AIParam(desc="Name of alleged strengthening phase (e.g., 'AL2FE', 'Ni3Al', 'TAU', 'LAVES'). Optional.")] = None,
        claimed_matrix_phase: Annotated[Optional[str], AIParam(desc="Name of alleged matrix phase (e.g., 'BCC_A2', 'FCC_A1'). Optional.")] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive assessment of alloy microstructure and mechanical effects.
        
        Supports both binary and ternary systems.
        
        Returns verification of claims about phase formation, strengthening, embrittlement,
        and stiffness (elastic modulus) changes based on thermodynamic calculations (CALPHAD),
        mechanical property estimates (Materials Project), and rule-of-mixtures stiffness modeling.
        """
        start_time = time.time()
        
        try:
            # Parse system
            system_elems, error = parse_system_string(system)
            if error:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="alloys",
                    function="assess_phase_strength_and_stiffness_claims",
                    error=error,
                    error_type=ErrorType.INVALID_INPUT,
                    suggestions=["Use format like 'Fe-Al' for binary systems or 'Al-Mg-Zn' for ternary systems"],
                    duration_ms=duration_ms
                )
            
            expected_elements = set(system_elems)
            
            # Parse composition
            comp_dict, error = parse_composition_string(composition, expected_elements)
            if error:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="alloys",
                    function="assess_phase_strength_and_stiffness_claims",
                    error=f"Could not parse composition: '{composition}'. {error}",
                    error_type=ErrorType.INVALID_INPUT,
                    suggestions=["Use format like 'Fe30Al70' (concatenated) or 'Al-8Mg-4Zn' (hyphenated)"],
                    duration_ms=duration_ms
                )
            
            _log.info(f"Assessing {system} at composition {comp_dict} and T={temperature_K}K")
            
            # Step 1: Compute equilibrium microstructure using shared utility
            # This handles database loading, equilibrium calculation, and phase identification
            _log.info("Step 1: Computing equilibrium microstructure...")
            
            # Convert composition from at.% to mole fractions
            total = sum(comp_dict.values())
            mole_fractions = {elem: comp_dict[elem] / total for elem in comp_dict}
            
            microstructure = compute_equilibrium_microstructure(
                system_elements=list(system_elems),
                composition=mole_fractions,
                temperature=temperature_K
            )
            
            if not microstructure.get("success"):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="alloys",
                    function="assess_phase_strength_and_stiffness_claims",
                    error=f"Failed to compute equilibrium: {microstructure.get('error')}",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["pycalphad"],
                    diagnostics={"equilibrium_microstructure": microstructure},
                    duration_ms=duration_ms
                )
            
            # Step 2: Get mechanical descriptors for all phases
            _log.info("Step 2: Fetching mechanical descriptors...")
            matrix_desc = get_phase_mechanical_descriptors(
                microstructure["matrix_phase"],
                comp_dict,
                system_elems,
                self.mpr
            )
            
            sec_descs = {}
            for sec_phase in microstructure["secondary_phases"]:
                sec_descs[sec_phase["name"]] = get_phase_mechanical_descriptors(
                    sec_phase["name"],
                    comp_dict,
                    system_elems,
                    self.mpr
                )
            
            # Step 3: Assess mechanical effects (physics-based models: Orowan, Hall-Petch, embrittlement)
            _log.info("Step 3: Assessing mechanical effects with physics-based models...")
            
            # Build phase categories for assessment
            phase_categories = {}
            for phase_name in microstructure["phase_fractions"].keys():
                # Categorize based on phase name patterns
                phase_lower = phase_name.lower()
                if "laves" in phase_lower:
                    phase_categories[phase_name] = "laves"
                elif "tau" in phase_lower or "τ" in phase_lower:
                    phase_categories[phase_name] = "tau"
                elif "gamma" in phase_lower or "γ" in phase_lower:
                    phase_categories[phase_name] = "gamma"
            
            # Compute physics-based assessment (Orowan, coherency, Hall-Petch, embrittlement)
            mech_assessment = assess_mechanical_effects(
                matrix_desc=matrix_desc,
                sec_descs=sec_descs,
                microstructure=microstructure,
                phase_categories=phase_categories
            )
            
            # Step 3b: Assess stiffness (elastic modulus) of matrix phase
            _log.info("Step 3b: Assessing matrix stiffness...")
            stiffness_assessment = estimate_phase_modulus(
                matrix_phase_name=microstructure["matrix_phase"],
                matrix_phase_composition=microstructure.get("matrix_phase_composition", {}),
                temperature_K=temperature_K,
                fallback_to_bulk_composition=comp_dict
            )
            
            # Step 4: Verify claims
            _log.info("Step 4: Verifying claims...")
            claim_check = verify_claims(
                microstructure,
                mech_assessment,
                stiffness_assessment,
                claimed_secondary_phase,
                claimed_matrix_phase
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return success_result(
                handler="alloys",
                function="assess_phase_strength_and_stiffness_claims",
                data={
                    "system": system,
                    "composition": comp_dict,
                    "temperature_K": temperature_K,
                    "equilibrium_microstructure": microstructure,
                    "mechanical_assessment": mech_assessment,
                    "stiffness_assessment": stiffness_assessment,
                    "claim_check": claim_check,
                },
                citations=["pycalphad", "Materials Project"],
                confidence=Confidence.MEDIUM,
                notes=[
                    n for n in [
                        "Equilibrium phases calculated using CALPHAD thermodynamics",
                        "Mechanical properties from Materials Project elastic data (VRH averaging)",
                        "Stiffness changes assessed using ±10% threshold (engineering practice)",
                        f"Yield strength: {mech_assessment.get('yield_strength_MPa')} MPa" if mech_assessment.get('yield_strength_MPa') is not None else None,
                        f"Embrittlement risk score: {mech_assessment.get('embrittlement_score'):.3f}" if mech_assessment.get('embrittlement_score') is not None else None,
                        "Physics-based models: Ashby-Orowan, coherency strengthening, Hall-Petch, Pugh ratio embrittlement"
                    ] if n is not None
                ],
                caveats=[
                    "CALPHAD predictions assume equilibrium conditions",
                    "Actual microstructure may differ due to kinetic effects",
                    "Mechanical property estimates are based on bulk elastic moduli"
                ],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in assess_phase_strength_claim: {e}", exc_info=True)
            return error_result(
                handler="alloys",
                function="assess_phase_strength_and_stiffness_claims",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad", "Materials Project"],
                duration_ms=duration_ms
            )
