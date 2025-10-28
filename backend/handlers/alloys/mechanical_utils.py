"""
Mechanical property assessment utilities for phases.

This module provides functions to assess mechanical properties of phases,
including brittleness, ductility, and elastic properties using Materials Project data.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional

_log = logging.getLogger(__name__)


def get_phase_mechanical_descriptors(
    phase_name: str,
    composition_hint: Dict[str, float],
    system_elems: Tuple[str, ...],
    mpr_client: Optional[object] = None
) -> Dict[str, Any]:
    """
    Get mechanical descriptors for a phase using Materials Project API.
    
    This function:
    1. Classifies phase type (intermetallic, solid solution, BCC/FCC/HCP)
    2. Attempts to fetch elastic properties from Materials Project
    3. Calculates Pugh ratio to assess brittleness
    4. Falls back to heuristics if no MP data available
    5. Applies special case rules for known brittle ordered phases
    
    Args:
        phase_name: CALPHAD phase label (e.g., "AL2FE", "BCC_A2")
        composition_hint: dict with element percentages
        system_elems: tuple of elements in system
        mpr_client: Materials Project API client
        
    Returns:
        {
            "is_intermetallic": bool,
            "is_bcc_like": bool,
            "is_fcc_like": bool,
            "is_hcp_like": bool,
            "bulk_modulus_GPa": float | None,
            "shear_modulus_GPa": float | None,
            "pugh_ratio": float | None,
            "brittle_flag": bool | None,
            "crystal_system": str | None,
            "source": str,
            "notes": str (optional)
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
        if mpr_client and is_intermetallic:
            try:
                # Parse stoichiometry from phase name (e.g., AL2FE -> Al2Fe)
                from ..shared.calphad_utils import parse_calphad_phase_name
                formula = parse_calphad_phase_name(phase_name, system_elems)
                
                if formula:
                    _log.info(f"Searching MP for phase {phase_name} with formula {formula}")
                    
                    # Search for materials with this composition
                    from mp_api.client import MPRester
                    from pymatgen.core import Composition
                    
                    # Search for materials matching the formula
                    docs = mpr_client.materials.summary.search(
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
        
        # Apply special case rules for known brittle ordered phases
        result = _apply_special_case_rules(
            result, phase_name, is_bcc_like, is_fcc_like, 
            system_elems, composition_hint
        )
        
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


def _apply_special_case_rules(
    result: Dict[str, Any],
    phase_name: str,
    is_bcc_like: bool,
    is_fcc_like: bool,
    system_elems: Tuple[str, ...],
    composition_hint: Dict[str, float]
) -> Dict[str, Any]:
    """
    Apply special case rules for known brittle ordered phases.
    
    Some phases labeled as BCC or FCC in CALPHAD databases are actually
    ordered intermetallics (e.g., B2-ordered FeAl, NiAl) that are brittle.
    This function detects these cases based on composition.
    
    Args:
        result: Current mechanical descriptors dict
        phase_name: CALPHAD phase name
        is_bcc_like: Whether phase is BCC-like
        is_fcc_like: Whether phase is FCC-like
        system_elems: System elements
        composition_hint: Composition dict
        
    Returns:
        Updated result dict with special case rules applied
    """
    # Special cases: Near-equiatomic ordered intermetallics that may be labeled as BCC/FCC
    # but are actually brittle ordered phases
    if is_bcc_like or is_fcc_like:
        A, B = [e.upper() for e in system_elems[:2]]  # Handle binary systems
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

