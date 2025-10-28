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
        
        # Generate final interpretation
        results["final_interpretation"] = _generate_interpretation(
            results, actual_matrix, secondary_names
        )
        
        return results
        
    except Exception as e:
        _log.error(f"Error verifying claims: {e}", exc_info=True)
        return {"error": str(e)}


def _generate_interpretation(
    results: Dict[str, Any],
    actual_matrix: str,
    secondary_names: list
) -> str:
    """
    Generate final human-readable interpretation with checkmarks and warnings.
    
    Args:
        results: Verification results dict
        actual_matrix: Actual matrix phase name
        secondary_names: List of secondary phase names
        
    Returns:
        Multi-line interpretation string with ✅, ⚠️, ❌, ℹ️ symbols
    """
    interpretation_lines = []
    
    # Matrix phase check
    if results.get("claimed_matrix"):
        if results["matrix_matches_claim"]:
            interpretation_lines.append(
                f"✅ Matrix phase claim VERIFIED: {actual_matrix} confirmed as primary phase"
            )
        else:
            interpretation_lines.append(
                f"❌ Matrix phase claim NOT VERIFIED: Predicted {actual_matrix}, "
                f"claimed {results['claimed_matrix']}"
            )
    
    # Secondary phase check
    if results.get("claimed_secondary"):
        if results["secondary_phase_present"]:
            frac_pct = results["secondary_fraction"] * 100
            if results["secondary_is_small_fraction"]:
                interpretation_lines.append(
                    f"✅ Secondary phase claim VERIFIED: {results['claimed_secondary']} "
                    f"present at {frac_pct:.1f}% (suitable for precipitation strengthening)"
                )
            else:
                if frac_pct < 5:
                    interpretation_lines.append(
                        f"⚠️ Secondary phase claim PARTIALLY VERIFIED: "
                        f"{results['claimed_secondary']} present but only {frac_pct:.1f}% "
                        "(too small for significant strengthening)"
                    )
                else:
                    interpretation_lines.append(
                        f"⚠️ Secondary phase claim PARTIALLY VERIFIED: "
                        f"{results['claimed_secondary']} present at {frac_pct:.1f}% "
                        "(exceeds typical precipitation strengthening range, may be co-matrix)"
                    )
        else:
            interpretation_lines.append(
                f"❌ Secondary phase claim NOT VERIFIED: {results['claimed_secondary']} not found. "
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
    
    return "\n".join(interpretation_lines)

