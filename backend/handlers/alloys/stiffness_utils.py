"""
Elastic modulus (stiffness) estimation utilities for alloy phases.

This module provides functions to estimate Young's modulus using rule-of-mixtures
and assess stiffness changes in alloys.

NOTE: Main implementation moved to shared/elasticity_utils.py for reusability.
This module now provides a thin wrapper for backward compatibility.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

# Import from shared constants (comprehensive element modulus data)
from backend.handlers.shared.constants.elasticity import ELEMENT_MODULUS_GPA

# Import robust implementation from shared utilities
from backend.handlers.shared.elasticity_utils import (
    estimate_phase_modulus as _estimate_phase_modulus_robust,
    normalize_matrix_composition,
    apply_temperature_correction,
)

_log = logging.getLogger(__name__)


def estimate_phase_modulus(
    matrix_phase_name: str,
    matrix_phase_composition: Dict[str, float],
    temperature_K: Optional[float] = None,
    fallback_to_bulk_composition: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Estimate the effective Young's modulus (stiffness) of the matrix phase
    by rule-of-mixtures over its elemental makeup.
    
    This is a robust implementation that:
    - Normalizes CALPHAD compositions (handles sublattice markers, vacancies)
    - Falls back to bulk composition if phase composition is empty
    - Handles unknown elements gracefully (never returns nulls unless unavoidable)
    - Applies optional temperature corrections
    - Provides detailed diagnostic notes
    
    Uses a ±10% threshold to classify changes as "significant" (matching
    engineering practice where commercial Al alloys have ~same stiffness as pure Al).
    
    Args:
        matrix_phase_name: CALPHAD phase label (e.g., "FCC_A1", "BCC_A2")
        matrix_phase_composition: Dict with atomic fractions of elements in the matrix
                                 (e.g., {"AL": 0.96, "MG": 0.04})
        temperature_K: Temperature in K for temperature correction (optional)
        fallback_to_bulk_composition: Bulk alloy composition to use if phase comp empty
    
    Returns:
        {
          "E_matrix_GPa": float or None,
          "E_baseline_GPa": float or None,
          "baseline_element": str or None,
          "relative_change": float or None,  # (E_matrix - E_baseline)/E_baseline
          "percent_change": float or None,   # relative_change * 100
          "assessment": "increase" | "decrease" | "no_significant_change" | "unknown",
          "matrix_composition": Dict[str, float],  # normalized composition
          "temperature_K": float or None,
          "notes": str
        }
        
    Example:
        >>> # Room temperature estimation
        >>> result = estimate_phase_modulus("FCC_A1", {"AL": 0.96, "MG": 0.04})
        >>> print(result["E_matrix_GPa"])  # ~68.6 GPa
        
        >>> # High temperature with fallback
        >>> result = estimate_phase_modulus(
        ...     "FCC_A1",
        ...     {},  # empty phase composition
        ...     temperature_K=700.0,
        ...     fallback_to_bulk_composition={"AL": 0.8, "ZN": 0.2}
        ... )
        >>> print(result["E_matrix_GPa"])  # ~52.5 GPa (temp-corrected)
    """
    return _estimate_phase_modulus_robust(
        matrix_phase_name=matrix_phase_name,
        matrix_phase_composition=matrix_phase_composition,
        temperature_K=temperature_K,
        fallback_to_bulk_composition=fallback_to_bulk_composition
    )


def estimate_composite_modulus_vrh(
    phase_fractions: Dict[str, float],
    phase_mechanical_descriptors: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Estimate composite elastic moduli using Voigt-Reuss-Hill (VRH) averaging.
    
    This is a physics-based approach for multi-phase materials that properly
    accounts for the elastic properties of each phase. More rigorous than
    simple rule-of-mixtures when dealing with phases of very different stiffness.
    
    Args:
        phase_fractions: Dict mapping phase name to volume fraction
        phase_mechanical_descriptors: Dict mapping phase name to mechanical descriptors
                                     (must contain "bulk_modulus_GPa" and "shear_modulus_GPa")
    
    Returns:
        {
          "B_composite_GPa": float,      # Bulk modulus (VRH average)
          "G_composite_GPa": float,      # Shear modulus (VRH average)
          "E_composite_GPa": float,      # Young's modulus
          "nu_composite": float,         # Poisson ratio
          "method": "VRH",
          "notes": str
        }
        
    Example:
        >>> phase_fractions = {"FCC_A1": 0.85, "AL2FE": 0.15}
        >>> phase_descs = {
        ...     "FCC_A1": {"bulk_modulus_GPa": 76.0, "shear_modulus_GPa": 26.0},
        ...     "AL2FE": {"bulk_modulus_GPa": 170.0, "shear_modulus_GPa": 85.0}
        ... }
        >>> result = estimate_composite_modulus_vrh(phase_fractions, phase_descs)
        >>> print(result["E_composite_GPa"])  # ~82 GPa
    
    References:
        • Watt et al., Phys. Earth Planet. Inter. 10 (1975) — VRH averaging
    """
    try:
        from .mechanical_utils import PhaseElastic, vrh_composite_moduli
        
        # Build PhaseElastic objects from descriptors
        phase_elastic_data = {}
        for phase_name, desc in phase_mechanical_descriptors.items():
            B = desc.get("bulk_modulus_GPa")
            G = desc.get("shear_modulus_GPa")
            if B is None or G is None:
                return {
                    "B_composite_GPa": None,
                    "G_composite_GPa": None,
                    "E_composite_GPa": None,
                    "nu_composite": None,
                    "method": "VRH",
                    "notes": f"Missing elastic data for phase {phase_name}"
                }
            
            phase_elastic_data[phase_name] = PhaseElastic(
                name=phase_name,
                B_GPa=B,
                G_GPa=G,
                source=desc.get("source", "unknown")
            )
        
        # Compute VRH composite moduli
        B, G, E, nu = vrh_composite_moduli(phase_fractions, phase_elastic_data)
        
        return {
            "B_composite_GPa": round(B, 2),
            "G_composite_GPa": round(G, 2),
            "E_composite_GPa": round(E, 2),
            "nu_composite": round(nu, 4),
            "method": "VRH",
            "notes": f"Voigt-Reuss-Hill average over {len(phase_fractions)} phases"
        }
        
    except Exception as e:
        _log.error(f"Error computing VRH composite modulus: {e}", exc_info=True)
        return {
            "B_composite_GPa": None,
            "G_composite_GPa": None,
            "E_composite_GPa": None,
            "nu_composite": None,
            "method": "VRH",
            "notes": f"Error: {str(e)}"
        }


# Export additional utilities for advanced use cases
__all__ = [
    "estimate_phase_modulus",
    "estimate_composite_modulus_vrh",
    "normalize_matrix_composition",
    "apply_temperature_correction",
    "ELEMENT_MODULUS_GPA",
]
