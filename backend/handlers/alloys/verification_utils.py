"""
Claim verification utilities for alloy microstructure and properties.

This module provides functions to verify user claims about phase formation,
strengthening, embrittlement, and stiffness changes against computed results.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

_log = logging.getLogger(__name__)


def verify_claims(
    microstructure: Dict[str, Any],
    mech_assessment: Dict[str, Any],
    stiffness_assessment: Dict[str, Any],
    claimed_secondary: Optional[str],
    claimed_matrix: Optional[str]
) -> Dict[str, Any]:
    """
    Verify user's claims against calculated results.
    
    This function:
    1. Checks if claimed matrix phase matches calculated matrix
    2. Checks if claimed secondary phase is present and in appropriate fraction
    3. Assesses strengthening and embrittlement
    4. Assesses stiffness changes
    5. Generates final interpretation with checkmarks/warnings
    
    Args:
        microstructure: Equilibrium microstructure results
        mech_assessment: Mechanical assessment results
        stiffness_assessment: Stiffness assessment results
        claimed_secondary: User's claimed secondary phase (optional)
        claimed_matrix: User's claimed matrix phase (optional)
        
    Returns:
        Dict with verification results and final interpretation
    """
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
        
        return results
        
    except Exception as e:
        _log.error(f"Error verifying claims: {e}", exc_info=True)
        return {"error": str(e)}