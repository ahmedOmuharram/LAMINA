"""
Elastic modulus (stiffness) estimation utilities for alloy phases.

This module provides functions to estimate Young's modulus using rule-of-mixtures
and assess stiffness changes in alloys.
"""
from __future__ import annotations
import logging
from typing import Dict, Any

_log = logging.getLogger(__name__)


# Handbook-ish Young's moduli at room temp (isotropic polycrystal, GPa)
# Sources: ASM Handbook, CRC Materials Science & Engineering Handbook
ELEMENT_MODULUS_GPA = {
    "AL": 70.0,   # Aluminum
    "MG": 45.0,   # Magnesium
    "CU": 117.0,  # Copper
    "ZN": 83.0,   # Zinc
    "FE": 210.0,  # Iron
    "NI": 170.0,  # Nickel
    "TI": 116.0,  # Titanium
    "CR": 279.0,  # Chromium
    "MN": 191.0,  # Manganese
    "SI": 165.0,  # Silicon (approximate for alloys)
}


def estimate_phase_modulus(
    matrix_phase_name: str,
    matrix_phase_composition: Dict[str, float],
) -> Dict[str, Any]:
    """
    Estimate the effective Young's modulus (stiffness) of the matrix phase
    by simple rule-of-mixtures over its elemental makeup.
    
    This provides a first-order estimate of how alloying affects stiffness.
    Uses a ±10% threshold to classify changes as "significant" (matching
    engineering practice where commercial Al alloys have ~same stiffness as pure Al).
    
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

