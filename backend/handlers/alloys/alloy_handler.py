from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Annotated, Tuple, List
from pathlib import Path

from kani import ai_function, AIParam
from ..base import BaseHandler
from ..constants import (
    KJMOL_PER_EV_PER_ATOM,
    ADS_OVER_COH_111,
    ADS_OVER_COH_100,
    ADS_OVER_COH_110,
    DIFF_OVER_ADS_111,
    DIFF_OVER_ADS_100,
    DIFF_OVER_ADS_110,
    COHESIVE_ENERGY_FALLBACK,
)

_log = logging.getLogger(__name__)

def _kjmol_to_ev(x_kjmol: float) -> float:
    return float(x_kjmol) / KJMOL_PER_EV_PER_ATOM

def _get_cohesive_energy(symbol: str) -> tuple[Optional[float], str]:
    """
    Return cohesive (atomization) energy in eV/atom and a source tag.

    Priority:
      1) mendeleev: evaporation_heat (+ fusion_heat if available) → eV/atom
      2) mendeleev: cohesive_energy if present (infer units if needed)
      3) curated fallback table (eV/atom)
    """
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)

        evap = getattr(e, "evaporation_heat", None)    # kJ/mol
        fus  = getattr(e, "fusion_heat", None)         # kJ/mol (may be None)
        if evap is not None:
            coh_kj = float(evap) + (float(fus) if fus is not None else 0.0)
            return (_kjmol_to_ev(coh_kj), "mendeleev.evaporation_heat(+fusion)")

        # Looser cohesive_energy field (units can vary)
        val = getattr(e, "cohesive_energy", None)
        if val is not None:
            v = float(val)
            # If surprisingly large, assume kJ/mol
            ev = v if v < 20.0 else _kjmol_to_ev(v)
            return (ev, "mendeleev.cohesive_energy")
    except Exception:
        pass

    val = COHESIVE_ENERGY_FALLBACK.get(symbol)
    return (val, "fallback_table") if val is not None else (None, "missing")

def _normalize_surface(surface_miller: Optional[str]) -> Tuple[str, float, str, float]:
    """
    Parse/normalize surface and return:
      (facet, facet_multiplier, mechanism, diff_over_ads)
    Stronger multipliers reflect well-known facet/mechanism differences.
    """
    if not surface_miller:
        # Unspecified facet: assume generic terrace and broaden uncertainty later
        return "unspecified", 1.00, "unknown", 0.18  # midpoint between hopping and exchange

    s = str(surface_miller).lower()
    for token in ("fcc", "bcc", "hcp", "sc", "diamond", "al", "au"):
        s = s.replace(token, "")
    for ch in "()[]{}-–, ":
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch.isdigit())

    if s == "111":
        return "111", 0.75, "hopping", DIFF_OVER_ADS_111
    if s == "100":
        return "100", 1.25, "exchange", DIFF_OVER_ADS_100
    if s == "110":
        return "110", 1.45, "exchange", DIFF_OVER_ADS_110
    return "unspecified", 1.00, "unknown", 0.18

def _get_metal_radius(symbol: str) -> Optional[float]:
    """Approximate metallic/atomic radius in Å (mendeleev pm preferred, else pymatgen)."""
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)
        for attr in ("metallic_radius_c12", "metallic_radius"):
            val = getattr(e, attr, None)
            if val is not None:
                return float(val) / 100.0  # pm → Å
    except Exception:
        pass

    try:
        from pymatgen.core import Element as PMGElement  # type: ignore
        el = PMGElement(symbol)
        for attr in ("metallic_radius", "atomic_radius", "atomic_radius_calculated", "covalent_radius"):
            v = getattr(el, attr, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    continue
    except Exception:
        pass
    return None

class AlloyHandler(BaseHandler):
    """Alloy surface heuristics for adatom diffusion barriers with facet/mechanism awareness."""

    def __init__(self, mpr_client: Optional[object] = None) -> None:
        super().__init__(mpr_client)
        _log.info("AlloyHandler initialized")

    @ai_function(
        desc=(
            "Estimate the surface diffusion barrier (activation energy, eV) for an adatom on a metal surface. "
            "Facet (111/100/110) and likely mechanism (hopping vs exchange) are accounted for. "
            "Intended for rough scoping; use DFT+NEB for quantitative work."
        ),
        auto_truncate=128000,
    )
    async def estimate_surface_diffusion_barrier(
        self,
        adatom_element: Annotated[str, AIParam(desc="Adatom element, e.g., 'Au'.")],
        host_element: Annotated[str, AIParam(desc="Host surface element, e.g., 'Al'.")],
        surface_miller: Annotated[Optional[str], AIParam(desc="Surface orientation like '111', '100', '110' (optional).")] = None,
    ) -> Dict[str, Any]:
        try:
            ad = adatom_element.capitalize()
            host = host_element.capitalize()

            ecoh_ad, src_ad = _get_cohesive_energy(ad)
            ecoh_host, src_host = _get_cohesive_energy(host)
            r_ad = _get_metal_radius(ad)
            r_host = _get_metal_radius(host)

            if ecoh_host is None or r_ad is None or r_host is None:
                return {
                    "success": False,
                    "error": "Insufficient data for estimate.",
                    "missing": {
                        "cohesive_energy_host": ecoh_host,
                        "radius_adatom": r_ad,
                        "radius_host": r_host,
                    },
                }

            facet, facet_mult, mechanism, diff_over_ads = _normalize_surface(surface_miller)

            # Choose facet-specific adsorption scaling vs host cohesion
            if facet == "111":
                ads_over_coh = ADS_OVER_COH_111
            elif facet == "100":
                ads_over_coh = ADS_OVER_COH_100
            elif facet == "110":
                ads_over_coh = ADS_OVER_COH_110
            else:
                # unknown facet → mid value
                ads_over_coh = 0.25

            # Size mismatch (relative to substrate) → gentle penalty/boost
            size_mismatch_rel = abs((r_ad - r_host) / r_host)
            size_factor = max(0.7, min(1.10, 1.0 - 0.35 * size_mismatch_rel))

            # Adsorption energy scale (eV)
            e_ads_est = ads_over_coh * ecoh_host

            # Barrier from adsorption scale with facet/mechanism + facet multiplier + size factor
            ea_est = diff_over_ads * e_ads_est
            ea_est *= facet_mult * size_factor

            # Keep within a plausible metal-on-metal window
            ea_est = float(max(0.01, min(2.00, ea_est)))

            # Uncertainty: ×2 if facet known, ×3 otherwise
            spread = 2.0 if facet in ("111", "100", "110") else 3.0
            ea_low = round(max(0.01, ea_est / spread), 4)
            ea_high = round(min(2.0, ea_est * spread), 4)

            # Confidence: facet specified → medium; unspecified → low
            confidence = "high" if facet in ("111", "100", "110") else "low"

            return {
                "success": True,
                "adatom": ad,
                "host": host,
                "surface": facet,
                "surface_input": surface_miller,
                "activation_energy_eV": round(ea_est, 4),
                "energy_range_eV": [ea_low, ea_high],
                "confidence": confidence,
                "likely_mechanism": mechanism,
                "method": "descriptor_model_v2_facetaware",
                "descriptors": {
                    "cohesive_energy_adatom_eV": round(ecoh_ad, 4) if ecoh_ad is not None else None,
                    "cohesive_energy_host_eV": round(ecoh_host, 4),
                    "ads_over_coh_fraction": round(ads_over_coh, 3),
                    "diff_over_ads_fraction": round(diff_over_ads, 3),
                    "facet_multiplier": round(facet_mult, 2),
                    "size_mismatch_rel": round(size_mismatch_rel, 4),
                    "size_factor": round(size_factor, 3),
                },
                "data_sources": {
                    "cohesive_energy": {
                        "adatom": src_ad,
                        "host": src_host,
                    },
                    "radii": "mendeleev metallic radii (pm→Å), else pymatgen (Å)",
                },
                "caveats": (
                    "Heuristic scaling; actual barriers depend on site, mechanism (hopping vs exchange), "
                    "and reconstruction/segregation. Use DFT+NEB for accuracy; consider alloying tendencies."
                ),
            }
        except Exception as e:
            _log.error(f"Error estimating surface diffusion barrier for {adatom_element} on {host_element}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _compute_equilibrium_microstructure(
        self,
        db,
        system_elems: Tuple[str, str],
        composition: Dict[str, float],
        temperature_K: float
    ) -> Dict[str, Any]:
        """
        Compute equilibrium phase fractions at a specific composition and temperature.
        
        Args:
            db: pycalphad Database
            system_elems: tuple like ("Fe", "Al")
            composition: dict like {"Fe": 30.0, "Al": 70.0}  # at.%
            temperature_K: float
            
        Returns:
            {
                "phases": [{"name": "AL5FE2", "fraction": 0.70}, ...],
                "matrix_phase": "AL5FE2" or "BCC_A2",
                "matrix_phase_composition": {"AL": 0.96, "MG": 0.04},  # at.% in matrix
                "secondary_phases": [...],
            }
        """
        try:
            from pycalphad import equilibrium, variables as v
            from ..calphad.phase_diagrams.equilibrium_utils import extract_phase_fractions_from_equilibrium
            from ..calphad.phase_diagrams.database_utils import get_db_elements
            
            # Get actual elements from database and verify
            db_elems = get_db_elements(db)
            
            # Convert all system elements to uppercase for database lookup
            elements_upper = [elem.upper() for elem in system_elems]
            
            # Verify all elements exist in database
            missing_elements = [el for el in elements_upper if el not in db_elems]
            if missing_elements:
                return {
                    "success": False,
                    "error": f"Elements {missing_elements} not found in database. Available: {sorted(db_elems)}",
                    "phases": [],
                    "matrix_phase": None,
                    "matrix_phase_composition": {},
                    "secondary_phases": []
                }
            
            # Use uppercase element names as they appear in database, add VA
            elements = elements_upper + ['VA']
            
            # Get phases for this system - import from CALPHAD handler
            from ..calphad.phase_diagrams.phase_diagrams import CalPhadHandler
            temp_handler = CalPhadHandler()
            phases = temp_handler._filter_phases_for_system(db, tuple(elements_upper))
            
            # Convert composition to mole fractions
            # Normalize to sum to 1.0
            total = sum(composition.values())
            mole_fractions = {elem: composition.get(elem, 0.0) / total for elem in system_elems}
            
            # Build conditions: N-1 composition constraints for N elements
            # Use the last N-1 elements as independent variables
            conditions = {v.T: temperature_K, v.P: 101325, v.N: 1}
            
            for elem in elements_upper[1:]:  # Skip first element (dependent variable)
                # Find corresponding key in composition dict (case-insensitive match)
                comp_key = None
                for key in composition.keys():
                    if key.upper() == elem:
                        comp_key = key
                        break
                
                if comp_key:
                    x_val = mole_fractions[comp_key]
                    conditions[v.X(elem)] = x_val
            
            # Calculate equilibrium
            eq = equilibrium(db, elements, phases, conditions)
            
            # Extract phase fractions
            phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
            
            # Build phase list
            phase_list = [
                {"name": phase, "fraction": frac}
                for phase, frac in sorted(phase_fractions.items(), key=lambda x: -x[1])
                if frac > 0.01
            ]
            
            if not phase_list:
                return {
                    "success": False,
                    "error": "No stable phases found at this condition",
                    "phases": [],
                    "matrix_phase": None,
                    "matrix_phase_composition": {},
                    "secondary_phases": []
                }
            
            # Matrix is the phase with maximum fraction
            matrix_phase = max(phase_fractions.items(), key=lambda x: x[1])[0]
            
            # Extract matrix phase composition (per-phase chemistry)
            matrix_phase_composition = {}
            try:
                eqp = eq.squeeze()
                phase_mask = eqp['Phase'] == matrix_phase
                
                # Get composition of each element in the matrix phase
                for elem in elements_upper:
                    x_data = eqp['X'].sel(component=elem).where(phase_mask, drop=False)
                    x_val = float(x_data.mean().values)
                    if x_val > 1e-6:  # Only record non-negligible amounts
                        matrix_phase_composition[elem] = x_val
                
                # Normalize to sum to 1.0 (exclude VA)
                total_comp = sum(matrix_phase_composition.values())
                if total_comp > 0:
                    matrix_phase_composition = {
                        elem: frac / total_comp 
                        for elem, frac in matrix_phase_composition.items()
                    }
            except Exception as e:
                _log.warning(f"Could not extract matrix phase composition: {e}")
                matrix_phase_composition = {}
            
            # Secondary phases are all others
            secondary_phases = [
                {"name": phase, "fraction": frac}
                for phase, frac in phase_fractions.items()
                if phase != matrix_phase and frac > 0.01
            ]
            
            return {
                "success": True,
                "phases": phase_list,
                "matrix_phase": matrix_phase,
                "matrix_phase_composition": matrix_phase_composition,
                "secondary_phases": secondary_phases,
                "phase_fractions": phase_fractions
            }
            
        except Exception as e:
            _log.error(f"Error computing equilibrium microstructure: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "phases": [],
                "matrix_phase": None,
                "matrix_phase_composition": {},
                "secondary_phases": []
            }
    
    def _get_phase_mech_descriptors(
        self,
        phase_name: str,
        composition_hint: Dict[str, float],
        system_elems: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Get mechanical descriptors for a phase using Materials Project API.
        
        Args:
            phase_name: CALPHAD phase label (e.g., "AL2FE", "BCC_A2")
            composition_hint: dict with element percentages
            system_elems: tuple of elements in system
            
        Returns:
            {
                "is_intermetallic": bool,
                "is_bcc_like": bool,
                "is_fcc_like": bool,
                "bulk_modulus_GPa": float | None,
                "shear_modulus_GPa": float | None,
                "pugh_ratio": float | None,
                "brittle_flag": bool | None,
                "crystal_system": str | None
            }
        """
        try:
            phase_upper = phase_name.upper()
            
            # Classify based on CALPHAD phase label patterns
            is_bcc_like = "BCC" in phase_upper or "A2" in phase_upper
            is_fcc_like = "FCC" in phase_upper or "A1" in phase_upper
            is_hcp_like = "HCP" in phase_upper or "A3" in phase_upper
            
            # Solid solutions (disordered) vs ordered intermetallics
            is_solid_solution = is_bcc_like or is_fcc_like or is_hcp_like
            is_intermetallic = not is_solid_solution
            
            result = {
                "is_intermetallic": is_intermetallic,
                "is_bcc_like": is_bcc_like,
                "is_fcc_like": is_fcc_like,
                "is_hcp_like": is_hcp_like,
                "bulk_modulus_GPa": None,
                "shear_modulus_GPa": None,
                "pugh_ratio": None,
                "brittle_flag": None,
                "crystal_system": None,
                "source": "classification_only"
            }
            
            # Try to get elastic data from Materials Project
            if self.mpr and is_intermetallic:
                try:
                    # Parse stoichiometry from phase name (e.g., AL2FE -> Al2Fe)
                    # This is a simple heuristic parser
                    formula = self._parse_phase_to_formula(phase_name, system_elems)
                    
                    if formula:
                        _log.info(f"Searching MP for phase {phase_name} with formula {formula}")
                        
                        # Search for materials with this composition
                        from mp_api.client import MPRester
                        from pymatgen.core import Composition
                        
                        # Search for materials matching the formula
                        docs = self.mpr.materials.summary.search(
                            formula=formula,
                            fields=["material_id", "formula_pretty", "symmetry", 
                                   "bulk_modulus", "shear_modulus"]
                        )
                        
                        if docs:
                            # Take first match with elasticity data
                            for doc in docs[:3]:  # Check up to 3 matches
                                try:
                                    if hasattr(doc, 'symmetry'):
                                        result["crystal_system"] = doc.symmetry.crystal_system
                                    
                                    # Get elastic properties if available (VRH averages)
                                    bulk_mod = getattr(doc, 'bulk_modulus', None)
                                    shear_mod = getattr(doc, 'shear_modulus', None)
                                    
                                    # bulk_modulus and shear_modulus might be dict with 'vrh' key
                                    if isinstance(bulk_mod, dict):
                                        bulk_mod = bulk_mod.get('vrh', None)
                                    if isinstance(shear_mod, dict):
                                        shear_mod = shear_mod.get('vrh', None)
                                    
                                    if bulk_mod is not None and shear_mod is not None:
                                        result["bulk_modulus_GPa"] = float(bulk_mod)
                                        result["shear_modulus_GPa"] = float(shear_mod)
                                        
                                        # Calculate Pugh ratio (G/B)
                                        # Pugh ratio > 0.57 typically indicates brittle behavior
                                        pugh = float(shear_mod) / float(bulk_mod)
                                        result["pugh_ratio"] = pugh
                                        result["brittle_flag"] = (pugh >= 0.57)
                                        result["source"] = f"MP:{doc.material_id}"
                                        
                                        _log.info(f"Found elastic data for {phase_name}: B={bulk_mod:.1f} GPa, G={shear_mod:.1f} GPa, Pugh={pugh:.3f}")
                                        break
                                except Exception as e:
                                    _log.debug(f"Error processing MP doc: {e}")
                                    continue
                                    
                except Exception as e:
                    _log.warning(f"Could not fetch MP data for {phase_name}: {e}")
            
            # Fallback heuristics if no MP data
            if result["brittle_flag"] is None:
                # Heuristic: Intermetallics are typically more brittle
                # Solid solutions (BCC, FCC, HCP) are typically more ductile
                if is_intermetallic:
                    result["brittle_flag"] = True  # Assume brittle for intermetallics
                    result["source"] = "heuristic:intermetallic"
                elif is_bcc_like:
                    result["brittle_flag"] = False  # BCC can be ductile at high T
                    result["source"] = "heuristic:bcc"
                elif is_fcc_like:
                    result["brittle_flag"] = False  # FCC typically ductile
                    result["source"] = "heuristic:fcc"
                else:
                    result["brittle_flag"] = None  # Unknown
                    result["source"] = "unknown"
            
            # Special cases: Near-equiatomic ordered intermetallics that may be labeled as BCC/FCC
            # but are actually brittle ordered phases
            if is_bcc_like or is_fcc_like:
                A, B = [e.upper() for e in system_elems]
                total = sum(composition_hint.values())
                
                # Get composition fractions (try both cases)
                fracs = {}
                for elem in [A, B]:
                    fracs[elem] = (
                        composition_hint.get(elem, 0.0) + 
                        composition_hint.get(elem.capitalize(), 0.0)
                    ) / total
                
                elem_pair = {A, B}
                
                # Fe-Al near 50:50 → B2-ordered FeAl (brittle)
                if elem_pair == {"FE", "AL"} and 0.40 <= fracs.get("FE", 0) <= 0.60:
                    result["brittle_flag"] = True
                    result["source"] = "heuristic:B2_FeAl_ordered"
                    result["notes"] = "BCC_A2 at ~50:50 Fe-Al is B2-ordered FeAl (brittle due to limited slip systems)"
                    _log.info(f"Special case: {phase_name} at ~50:50 Fe-Al likely B2-ordered FeAl (brittle)")
                
                # Ni-Al near 50:50 → B2-ordered NiAl (brittle)
                elif elem_pair == {"NI", "AL"} and 0.40 <= fracs.get("NI", 0) <= 0.60:
                    result["brittle_flag"] = True
                    result["source"] = "heuristic:B2_NiAl_ordered"
                    result["notes"] = "BCC_A2 at ~50:50 Ni-Al is B2-ordered NiAl (brittle intermetallic)"
                    _log.info(f"Special case: {phase_name} at ~50:50 Ni-Al likely B2-ordered NiAl (brittle)")
                
                # Ti-Al near 50:50 → Ordered TiAl (L10 or B2, both brittle at room T)
                elif elem_pair == {"TI", "AL"} and 0.40 <= fracs.get("TI", 0) <= 0.60:
                    result["brittle_flag"] = True
                    result["source"] = "heuristic:TiAl_ordered"
                    result["notes"] = "FCC/BCC at ~50:50 Ti-Al is likely ordered TiAl (L10/B2, brittle at ambient T)"
                    _log.info(f"Special case: {phase_name} at ~50:50 Ti-Al likely ordered TiAl (brittle)")
            
            return result
            
        except Exception as e:
            _log.error(f"Error getting mechanical descriptors for {phase_name}: {e}", exc_info=True)
            return {
                "is_intermetallic": False,
                "is_bcc_like": False,
                "is_fcc_like": False,
                "bulk_modulus_GPa": None,
                "shear_modulus_GPa": None,
                "pugh_ratio": None,
                "brittle_flag": None,
                "crystal_system": None,
                "source": "error"
            }
    
    def _parse_phase_to_formula(self, phase_name: str, system_elems: Tuple[str, str]) -> Optional[str]:
        """
        Parse CALPHAD phase name to chemical formula.
        Examples: AL2FE -> Al2Fe, AL5FE2 -> Al5Fe2
        """
        try:
            import re
            phase_upper = phase_name.upper()
            
            # Skip solid solution phases
            if any(x in phase_upper for x in ["BCC", "FCC", "HCP", "LIQUID", "A1", "A2", "A3", "B2"]):
                return None
            
            # Try to parse element-number patterns
            A, B = [e.upper() for e in system_elems]
            
            # Look for patterns like AL2FE, AL5FE2, etc.
            # Replace element symbols with proper case
            formula = phase_upper
            for elem in system_elems:
                elem_upper = elem.upper()
                # Replace with proper case (e.g., AL -> Al, FE -> Fe)
                formula = re.sub(elem_upper, elem.capitalize(), formula, flags=re.IGNORECASE)
            
            # Remove common suffixes that aren't part of formula
            for suffix in ["_D03", "_L12", "_C15", "_DELTA"]:
                formula = formula.replace(suffix, "")
            
            # Validate it looks like a formula
            if any(char.isdigit() for char in formula) and any(char.isalpha() for char in formula):
                return formula
            
            return None
            
        except Exception as e:
            _log.debug(f"Could not parse phase name {phase_name}: {e}")
            return None
    
    def _estimate_phase_modulus(
        self,
        matrix_phase_name: str,
        matrix_phase_composition: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Estimate the effective Young's modulus (stiffness) of the matrix phase
        by simple rule-of-mixtures over its elemental makeup.
        
        Args:
            matrix_phase_name: CALPHAD phase label (e.g., "FCC_A1", "BCC_A2")
            matrix_phase_composition: Dict with atomic fractions of elements in the matrix
                                     (e.g., {"AL": 0.96, "MG": 0.04})
        
        Returns:
            {
              "E_matrix_GPa": float or None,
              "E_baseline_GPa": float or None,
              "baseline_element": str or None,
              "relative_change": float or None,  # (E_matrix - E_baseline)/E_baseline
              "percent_change": float or None,   # relative_change * 100
              "assessment": "increase" | "decrease" | "no_significant_change" | "unknown",
              "matrix_composition": Dict[str, float],
              "notes": str
            }
        """
        # Handbook-ish Young's moduli at room temp (isotropic polycrystal, GPa)
        # Sources: ASM Handbook, CRC Materials Science & Engineering Handbook
        ELEMENT_MODULUS_GPA = {
            "AL": 70.0,   # Aluminum
            "MG": 45.0,   # Magnesium
            "CU": 117.0,  # Copper
            "ZN": 83.0,  # Zinc
            "FE": 210.0,  # Iron
            "NI": 170.0,  # Nickel
        }
        
        if not matrix_phase_composition:
            return {
                "E_matrix_GPa": None,
                "E_baseline_GPa": None,
                "baseline_element": None,
                "relative_change": None,
                "percent_change": None,
                "assessment": "unknown",
                "matrix_composition": {},
                "notes": "No matrix composition available."
            }
        
        # Figure out dominant element (highest atomic fraction in matrix phase)
        dominant_elem = max(matrix_phase_composition.keys(),
                           key=lambda el: matrix_phase_composition[el])
        
        # Need its baseline modulus
        if dominant_elem not in ELEMENT_MODULUS_GPA:
            return {
                "E_matrix_GPa": None,
                "E_baseline_GPa": None,
                "baseline_element": dominant_elem,
                "relative_change": None,
                "percent_change": None,
                "assessment": "unknown",
                "matrix_composition": matrix_phase_composition,
                "notes": f"No baseline modulus data available for dominant element {dominant_elem}"
            }
        
        E_baseline = ELEMENT_MODULUS_GPA[dominant_elem]
        
        # Compute rule-of-mixtures modulus: E ≈ Σ(x_i * E_i)
        E_matrix = 0.0
        missing_data = []
        for el, atfrac in matrix_phase_composition.items():
            if el in ELEMENT_MODULUS_GPA:
                E_matrix += atfrac * ELEMENT_MODULUS_GPA[el]
            else:
                missing_data.append(el)
        
        # Calculate relative change
        if E_baseline > 0:
            rel_change = (E_matrix - E_baseline) / E_baseline
            pct_change = rel_change * 100.0
        else:
            rel_change = None
            pct_change = None
        
        # Classify significance
        # Threshold: ±10% is "significant" change
        # This is because real commercial alloys (2xxx, 5xxx, 6xxx, 7xxx Al)
        # all have moduli within ~5% of pure Al despite varying alloying content
        if rel_change is None:
            assessment = "unknown"
        else:
            if rel_change >= 0.10:
                assessment = "increase"
            elif rel_change <= -0.10:
                assessment = "decrease"
            else:
                assessment = "no_significant_change"
        
        notes_parts = [
            f"Dominant element: {dominant_elem} (baseline E = {E_baseline:.1f} GPa).",
            f"Rule-of-mixtures estimate: E_matrix = {E_matrix:.1f} GPa."
        ]
        
        if missing_data:
            notes_parts.append(f"Missing modulus data for: {', '.join(missing_data)}.")
        
        if rel_change is not None:
            notes_parts.append(f"Relative change: {pct_change:+.2f}%.")
        
        return {
            "E_matrix_GPa": round(E_matrix, 2),
            "E_baseline_GPa": E_baseline,
            "baseline_element": dominant_elem,
            "relative_change": round(rel_change, 4) if rel_change is not None else None,
            "percent_change": round(pct_change, 2) if pct_change is not None else None,
            "assessment": assessment,
            "matrix_composition": matrix_phase_composition,
            "notes": " ".join(notes_parts)
        }
    
    def _assess_mech_effect(
        self,
        matrix_desc: Dict[str, Any],
        sec_descs: Dict[str, Dict[str, Any]],
        microstructure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess strengthening and embrittlement based on microstructure and mechanical properties.
        
        Returns:
            {
                "strengthening_likelihood": "high"/"moderate"/"low",
                "embrittlement_risk": "high"/"moderate"/"low",
                "explanations": {...}
            }
        """
        try:
            matrix_name = microstructure["matrix_phase"]
            matrix_frac = microstructure["phase_fractions"].get(matrix_name, 0.0)
            
            total_secondary_frac = 1.0 - matrix_frac
            
            # Check if we have hard intermetallic secondaries
            hard_secondary_present = any(
                desc.get("is_intermetallic", False) 
                for desc in sec_descs.values()
            )
            
            # Strengthening assessment
            # Classic precipitation strengthening: 5-30% hard phase in ductile matrix
            if hard_secondary_present and 0.05 <= total_secondary_frac <= 0.30:
                if not matrix_desc.get("brittle_flag", True):
                    strengthening = "high"
                else:
                    strengthening = "moderate"
            elif hard_secondary_present and 0.01 <= total_secondary_frac < 0.05:
                strengthening = "moderate"
            elif hard_secondary_present and total_secondary_frac > 0.30:
                strengthening = "mixed"  # Too much precipitate
            else:
                strengthening = "low"
            
            # Embrittlement assessment
            brittle_secondaries = []
            for phase_info in microstructure["secondary_phases"]:
                phase_name = phase_info["name"]
                phase_frac = phase_info["fraction"]
                desc = sec_descs.get(phase_name, {})
                
                # Large brittle secondary phases are concerning
                if desc.get("brittle_flag") and phase_frac >= 0.15:
                    brittle_secondaries.append({
                        "name": phase_name,
                        "fraction": phase_frac,
                        "pugh_ratio": desc.get("pugh_ratio")
                    })
            
            # Embrittlement logic
            if matrix_desc.get("brittle_flag"):
                embrittle = "high"
                embrittle_reason = "Matrix phase itself is predicted to be brittle"
            elif brittle_secondaries:
                embrittle = "high"
                embrittle_reason = f"Large fraction of brittle secondary phases: {', '.join(b['name'] for b in brittle_secondaries)}"
            elif hard_secondary_present and total_secondary_frac > 0.40:
                embrittle = "moderate"
                embrittle_reason = "Very high secondary phase fraction may reduce ductility"
            else:
                embrittle = "low"
                embrittle_reason = "Matrix dominates and is ductile; secondary fraction is limited"
            
            # Build explanations
            explanations = {
                "matrix": {
                    "phase": matrix_name,
                    "fraction": round(matrix_frac, 3),
                    "brittle_flag": matrix_desc.get("brittle_flag"),
                    "pugh_ratio": matrix_desc.get("pugh_ratio"),
                    "type": "BCC" if matrix_desc.get("is_bcc_like") else "FCC" if matrix_desc.get("is_fcc_like") else "other",
                    "source": matrix_desc.get("source")
                },
                "secondary": {
                    "total_fraction": round(total_secondary_frac, 3),
                    "hard_intermetallic_present": hard_secondary_present,
                    "brittle_secondaries": brittle_secondaries,
                    "phases": [
                        {
                            "name": p["name"],
                            "fraction": round(p["fraction"], 3),
                            "is_intermetallic": sec_descs.get(p["name"], {}).get("is_intermetallic"),
                            "brittle_flag": sec_descs.get(p["name"], {}).get("brittle_flag"),
                            "pugh_ratio": sec_descs.get(p["name"], {}).get("pugh_ratio"),
                            "source": sec_descs.get(p["name"], {}).get("source")
                        }
                        for p in microstructure["secondary_phases"]
                    ]
                },
                "rationale": {
                    "strength": self._get_strengthening_rationale(
                        strengthening, hard_secondary_present, total_secondary_frac, matrix_desc
                    ),
                    "embrittlement": embrittle_reason
                }
            }
            
            return {
                "strengthening_likelihood": strengthening,
                "embrittlement_risk": embrittle,
                "explanations": explanations
            }
            
        except Exception as e:
            _log.error(f"Error assessing mechanical effect: {e}", exc_info=True)
            return {
                "strengthening_likelihood": "unknown",
                "embrittlement_risk": "unknown",
                "explanations": {"error": str(e)}
            }
    
    def _get_strengthening_rationale(
        self, 
        level: str, 
        has_hard_phase: bool, 
        sec_frac: float,
        matrix_desc: Dict[str, Any]
    ) -> str:
        """Generate human-readable strengthening rationale."""
        if level == "high":
            return (
                f"Secondary hard intermetallic precipitates ({sec_frac*100:.1f}% volume fraction) "
                "in a ductile matrix are expected to impede dislocation motion via "
                "Orowan strengthening, increasing yield strength significantly."
            )
        elif level == "moderate":
            if sec_frac < 0.05:
                return (
                    f"Small volume fraction ({sec_frac*100:.1f}%) of hard phase present. "
                    "Some strengthening expected but limited by low precipitate density."
                )
            else:
                return (
                    "Moderate strengthening expected, though matrix brittleness may limit effectiveness."
                )
        elif level == "mixed":
            return (
                f"Very high secondary phase fraction ({sec_frac*100:.1f}%) exceeds typical "
                "precipitation strengthening regime. The alloy is essentially a multi-phase "
                "material rather than a precipitate-strengthened alloy."
            )
        else:
            return (
                "No significant hard secondary phase fraction detected. "
                "Limited precipitation strengthening expected. Strength primarily from "
                "solid solution strengthening (if present) or base matrix properties."
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
        
        The stiffness assessment uses a ±10% threshold to classify changes as "significant"
        (matching engineering practice where commercial Al alloys have ~same stiffness as pure Al).
        """
        try:
            # Parse system
            parts = system.replace(" ", "").split("-")
            if len(parts) < 2 or len(parts) > 3:
                return {
                    "success": False,
                    "error": f"Invalid system format: '{system}'. Expected 'Element1-Element2' or 'Element1-Element2-Element3' (e.g., 'Fe-Al' or 'Al-Mg-Zn')"
                }
            
            system_elems = tuple(p.capitalize() for p in parts)
            expected_elements = set(system_elems)
            
            # Parse composition (supports multiple formats)
            # Format 1: "Fe30Al70" or "Al88Mg8Zn4" (element immediately followed by number)
            # Format 2: "Al-8Mg-4Zn" (element-number pairs, first element is balance)
            import re
            comp_dict = {}
            
            # Try format 1: concatenated (e.g., "Al88Mg8Zn4")
            matches = re.findall(r'([A-Z][a-z]?)(\d+\.?\d*)', composition)
            
            if matches and len(matches) == len(expected_elements):
                # Format 1 succeeded
                for elem, pct in matches:
                    comp_dict[elem] = float(pct)
            else:
                # Try format 2: hyphen-separated (e.g., "Al-8Mg-4Zn")
                # Split by hyphens, extract element and optional number
                parts = composition.replace(' ', '').split('-')
                
                for part in parts:
                    # Extract element and number from each part
                    match = re.match(r'([A-Z][a-z]?)(\d+\.?\d*)?', part)
                    if match:
                        elem = match.group(1)
                        pct_str = match.group(2)
                        
                        if pct_str:
                            comp_dict[elem] = float(pct_str)
                        else:
                            # No number means balance element (will be calculated)
                            comp_dict[elem] = None
                
                # Calculate balance element
                balance_elem = None
                specified_total = 0.0
                
                for elem, pct in comp_dict.items():
                    if pct is None:
                        balance_elem = elem
                    else:
                        specified_total += pct
                
                if balance_elem and specified_total < 100.0:
                    comp_dict[balance_elem] = 100.0 - specified_total
                elif balance_elem:
                    return {
                        "success": False,
                        "error": f"Specified compositions sum to {specified_total}%, exceeds 100%"
                    }
            
            # Validate composition
            if not comp_dict or set(comp_dict.keys()) != expected_elements:
                return {
                    "success": False,
                    "error": f"Could not parse composition: '{composition}'. Expected format like 'Fe30Al70' or 'Al-8Mg-4Zn'. Got elements: {list(comp_dict.keys()) if comp_dict else 'none'}, expected: {list(expected_elements)}"
                }
            
            # Ensure all values are numeric
            if any(v is None for v in comp_dict.values()):
                return {
                    "success": False,
                    "error": f"Failed to parse all composition values from '{composition}'"
                }
            
            _log.info(f"Assessing {system} at composition {comp_dict} and T={temperature_K}K")
            
            # Load CALPHAD database
            from pycalphad import Database
            from ..calphad.phase_diagrams.phase_diagrams import CalPhadHandler
            
            temp_handler = CalPhadHandler()
            db_path = temp_handler._get_database_path(system, elements=list(system_elems))
            
            if not db_path:
                return {
                    "success": False,
                    "error": f"No thermodynamic database found for {system} system"
                }
            
            db = Database(str(db_path))
            
            # Step 1: Compute equilibrium microstructure
            _log.info("Step 1: Computing equilibrium microstructure...")
            microstructure = self._compute_equilibrium_microstructure(
                db, system_elems, comp_dict, temperature_K
            )
            
            if not microstructure.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to compute equilibrium: {microstructure.get('error')}",
                    "equilibrium_microstructure": microstructure
                }
            
            # Step 2: Get mechanical descriptors for all phases
            _log.info("Step 2: Fetching mechanical descriptors...")
            matrix_desc = self._get_phase_mech_descriptors(
                microstructure["matrix_phase"],
                comp_dict,
                system_elems
            )
            
            sec_descs = {}
            for sec_phase in microstructure["secondary_phases"]:
                sec_descs[sec_phase["name"]] = self._get_phase_mech_descriptors(
                    sec_phase["name"],
                    comp_dict,
                    system_elems
                )
            
            # Step 3: Assess mechanical effects
            _log.info("Step 3: Assessing mechanical effects...")
            mech_assessment = self._assess_mech_effect(matrix_desc, sec_descs, microstructure)
            
            # Step 3b: Assess stiffness (elastic modulus) of matrix phase
            _log.info("Step 3b: Assessing matrix stiffness...")
            stiffness_assessment = self._estimate_phase_modulus(
                microstructure["matrix_phase"],
                microstructure.get("matrix_phase_composition", {})
            )
            
            # Step 4: Verify claims
            _log.info("Step 4: Verifying claims...")
            claim_check = self._verify_claims(
                microstructure,
                mech_assessment,
                stiffness_assessment,
                claimed_secondary_phase,
                claimed_matrix_phase
            )
            
            return {
                "success": True,
                "system": system,
                "composition": comp_dict,
                "temperature_K": temperature_K,
                "equilibrium_microstructure": microstructure,
                "mechanical_assessment": mech_assessment,
                "stiffness_assessment": stiffness_assessment,
                "claim_check": claim_check,
                "citations": ["pycalphad", "Materials Project"]
            }
            
        except Exception as e:
            _log.error(f"Error in assess_phase_strength_claim: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _verify_claims(
        self,
        microstructure: Dict[str, Any],
        mech_assessment: Dict[str, Any],
        stiffness_assessment: Dict[str, Any],
        claimed_secondary: Optional[str],
        claimed_matrix: Optional[str]
    ) -> Dict[str, Any]:
        """Verify user's claims against calculated results."""
        try:
            results = {}
            
            # Check matrix claim
            actual_matrix = microstructure["matrix_phase"]
            if claimed_matrix:
                matrix_matches = (claimed_matrix.upper() == actual_matrix.upper())
                results["matrix_matches_claim"] = matrix_matches
                results["claimed_matrix"] = claimed_matrix
                results["actual_matrix"] = actual_matrix
            else:
                results["matrix_matches_claim"] = None
                results["actual_matrix"] = actual_matrix
            
            # Check secondary phase claim
            secondary_names = [p["name"] for p in microstructure["secondary_phases"]]
            if claimed_secondary:
                claimed_upper = claimed_secondary.upper()
                secondary_present = any(claimed_upper == name.upper() for name in secondary_names)
                results["secondary_phase_present"] = secondary_present
                results["claimed_secondary"] = claimed_secondary
                
                # If present, check if it's a "small fraction"
                if secondary_present:
                    frac = next(
                        (p["fraction"] for p in microstructure["secondary_phases"] 
                         if p["name"].upper() == claimed_upper),
                        0.0
                    )
                    results["secondary_fraction"] = frac
                    # "Small" is typically 5-30% for precipitation strengthening
                    results["secondary_is_small_fraction"] = (0.05 <= frac <= 0.30)
                else:
                    results["secondary_fraction"] = 0.0
                    results["secondary_is_small_fraction"] = False
            else:
                results["secondary_phase_present"] = None
                results["actual_secondary_phases"] = secondary_names
            
            # Overall assessment
            results["strengthening_plausible"] = mech_assessment["strengthening_likelihood"]
            results["embrittlement_risk"] = mech_assessment["embrittlement_risk"]
            
            # Stiffness assessment
            results["stiffness_change"] = stiffness_assessment.get("assessment", "unknown")
            results["stiffness_percent_change"] = stiffness_assessment.get("percent_change")
            results["E_matrix_GPa"] = stiffness_assessment.get("E_matrix_GPa")
            results["E_baseline_GPa"] = stiffness_assessment.get("E_baseline_GPa")
            
            # Generate final interpretation
            interpretation_lines = []
            
            if claimed_matrix:
                if results["matrix_matches_claim"]:
                    interpretation_lines.append(f"✅ Matrix phase claim VERIFIED: {actual_matrix} confirmed as primary phase")
                else:
                    interpretation_lines.append(f"❌ Matrix phase claim NOT VERIFIED: Predicted {actual_matrix}, claimed {claimed_matrix}")
            
            if claimed_secondary:
                if results["secondary_phase_present"]:
                    frac_pct = results["secondary_fraction"] * 100
                    if results["secondary_is_small_fraction"]:
                        interpretation_lines.append(
                            f"✅ Secondary phase claim VERIFIED: {claimed_secondary} present at {frac_pct:.1f}% "
                            "(suitable for precipitation strengthening)"
                        )
                    else:
                        if frac_pct < 5:
                            interpretation_lines.append(
                                f"⚠️ Secondary phase claim PARTIALLY VERIFIED: {claimed_secondary} present but only {frac_pct:.1f}% "
                                "(too small for significant strengthening)"
                            )
                        else:
                            interpretation_lines.append(
                                f"⚠️ Secondary phase claim PARTIALLY VERIFIED: {claimed_secondary} present at {frac_pct:.1f}% "
                                "(exceeds typical precipitation strengthening range, may be co-matrix)"
                            )
                else:
                    interpretation_lines.append(
                        f"❌ Secondary phase claim NOT VERIFIED: {claimed_secondary} not found. "
                        f"Actual secondary phases: {', '.join(secondary_names) if secondary_names else 'none'}"
                    )
            
            # Strengthening assessment
            strength_level = results["strengthening_plausible"]
            if strength_level == "high":
                interpretation_lines.append("✅ STRENGTHENING: High likelihood based on microstructure")
            elif strength_level == "moderate":
                interpretation_lines.append("⚠️ STRENGTHENING: Moderate likelihood")
            elif strength_level == "mixed":
                interpretation_lines.append("⚠️ STRENGTHENING: Mixed (high secondary fraction)")
            else:
                interpretation_lines.append("❌ STRENGTHENING: Low likelihood (insufficient hard phase)")
            
            # Embrittlement assessment
            embritt_level = results["embrittlement_risk"]
            if embritt_level == "low":
                interpretation_lines.append("✅ EMBRITTLEMENT: Low risk - ductile matrix dominates")
            elif embritt_level == "moderate":
                interpretation_lines.append("⚠️ EMBRITTLEMENT: Moderate risk - significant hard phase fraction")
            else:
                interpretation_lines.append("❌ EMBRITTLEMENT: High risk - brittle phases detected")
            
            # Stiffness assessment
            stiffness_change = results["stiffness_change"]
            pct_change = results.get("stiffness_percent_change")
            E_matrix = results.get("E_matrix_GPa")
            E_baseline = results.get("E_baseline_GPa")
            
            if stiffness_change == "increase":
                interpretation_lines.append(
                    f"✅ STIFFNESS: Significant increase detected "
                    f"(E: {E_baseline:.1f} → {E_matrix:.1f} GPa, {pct_change:+.1f}%)"
                )
            elif stiffness_change == "decrease":
                interpretation_lines.append(
                    f"⚠️ STIFFNESS: Significant decrease detected "
                    f"(E: {E_baseline:.1f} → {E_matrix:.1f} GPa, {pct_change:+.1f}%)"
                )
            elif stiffness_change == "no_significant_change":
                interpretation_lines.append(
                    f"ℹ️ STIFFNESS: No significant change "
                    f"(E: {E_baseline:.1f} → {E_matrix:.1f} GPa, {pct_change:+.1f}%, within ±10% threshold)"
                )
            else:
                interpretation_lines.append("❓ STIFFNESS: Could not assess (insufficient data)")
            
            results["final_interpretation"] = "\n".join(interpretation_lines)
            
            return results
            
        except Exception as e:
            _log.error(f"Error verifying claims: {e}", exc_info=True)
            return {"error": str(e)}
