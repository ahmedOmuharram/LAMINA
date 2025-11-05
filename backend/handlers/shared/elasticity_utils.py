"""
Elastic modulus estimation utilities with robust composition handling.

This module provides functions to:
- Normalize and sanitize CALPHAD phase compositions
- Estimate Young's modulus with temperature corrections
- Handle missing data gracefully (never return null unless truly unavoidable)
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

from .constants.elasticity import (
    ELEMENT_MODULUS_GPA,
    TEMP_COEFFICIENT_PER_K,
    ROOM_TEMPERATURE_K,
)

_log = logging.getLogger(__name__)


def normalize_matrix_composition(raw_comp: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize and sanitize matrix phase composition from CALPHAD output.
    
    This handles common issues:
    - Mixed case element symbols → uppercase
    - Sublattice markers (e.g., 'AL#1' → 'AL')
    - Vacancy placeholders ('VA', 'VAC') → removed
    - Zero/negative entries → removed
    - Non-normalized sums → renormalized to 1.0
    
    Args:
        raw_comp: Raw composition dict from CALPHAD (e.g., {"Al": 0.96, "Mg#1": 0.04})
        
    Returns:
        Clean composition dict with uppercase symbols, summing to 1.0
        Returns empty dict if no valid composition remains
        
    Example:
        >>> normalize_matrix_composition({"Al": 0.5, "Fe#2": 0.3, "VA": 0.2})
        {'AL': 0.625, 'FE': 0.375}
    """
    if not raw_comp:
        return {}
    
    cleaned = {}
    
    for k, v in raw_comp.items():
        # Skip invalid values
        if v is None or v <= 0:
            continue
        
        key = str(k).upper().strip()
        
        # Strip sublattice instance markers (e.g., '#1', '#2')
        if '#' in key:
            key = key.split('#')[0]
        
        # Keep only alphabetical characters (handles 'AL1' → 'AL')
        key = ''.join(ch for ch in key if ch.isalpha())
        
        # Skip vacancy placeholders and empty keys
        if key in {"VA", "VAC", "X", ""} or len(key) == 0:
            continue
        
        # Accumulate (handles multiple sublattice entries for same element)
        cleaned[key] = cleaned.get(key, 0.0) + float(v)
    
    # Renormalize to sum = 1.0
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    
    return {el: frac / total for el, frac in cleaned.items()}


def apply_temperature_correction(
    E_GPa: float,
    composition: Dict[str, float],
    temperature_K: Optional[float]
) -> float:
    """
    Apply linear temperature softening correction to modulus.
    
    Most metals soften with increasing temperature. This applies a mixture
    rule for temperature dependence: E(T) ≈ E(RT) * (1 + Σ x_i * α_i * ΔT)
    
    Args:
        E_GPa: Young's modulus at room temperature (GPa)
        composition: Element composition (normalized, uppercase)
        temperature_K: Target temperature (K), or None to skip correction
        
    Returns:
        Temperature-corrected modulus (GPa)
        Returns original value if temperature is None or near room temp
        
    Example:
        >>> # Al modulus at 700K
        >>> E_corrected = apply_temperature_correction(70.0, {"AL": 1.0}, 700.0)
        >>> # Result: ~58.8 GPa (softer at high temperature)
    """
    if E_GPa is None or temperature_K is None:
        return E_GPa
    
    # Skip correction if near room temperature (±5K)
    delta_T = temperature_K - ROOM_TEMPERATURE_K
    if abs(delta_T) < 5.0:
        return E_GPa
    
    # Calculate mixture temperature coefficient from known elements
    mix_coeff = 0.0
    total_weight = 0.0
    
    for elem, x in composition.items():
        if elem in TEMP_COEFFICIENT_PER_K:
            coeff = TEMP_COEFFICIENT_PER_K[elem]
            mix_coeff += x * coeff
            total_weight += x
    
    # If no temperature data available, use a conservative default
    # (typical for metals: -3e-4 per K)
    if total_weight == 0:
        mix_coeff = -3.0e-4
        _log.debug(f"No temperature coefficient data, using default: {mix_coeff:.2e} K⁻¹")
    
    # Apply linear correction
    E_corrected = E_GPa * (1.0 + mix_coeff * delta_T)
    
    # Ensure non-negative (should not happen for reasonable T ranges)
    return max(0.0, E_corrected)


def estimate_phase_modulus(
    matrix_phase_name: str,
    matrix_phase_composition: Dict[str, float],
    temperature_K: Optional[float] = None,
    fallback_to_bulk_composition: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Robustly estimate Young's modulus of a matrix phase.
    
    This function:
    1. Normalizes composition (handles CALPHAD quirks)
    2. Falls back to bulk composition if phase composition is empty
    3. Computes weighted average modulus over known elements
    4. Picks baseline from dominant element (or mixture if unknown)
    5. Applies optional temperature correction
    6. Returns useful results even with partial data
    
    Almost never returns None unless composition is totally unusable.
    
    Args:
        matrix_phase_name: CALPHAD phase label (e.g., "FCC_A1", "AL2FE")
        matrix_phase_composition: Phase composition from CALPHAD
        temperature_K: Temperature for correction (optional)
        fallback_to_bulk_composition: Bulk alloy composition to use if phase comp empty
        
    Returns:
        Dictionary with keys:
        - E_matrix_GPa: Estimated modulus at temperature (float or None)
        - E_baseline_GPa: Baseline modulus for comparison (float or None)
        - baseline_element: Reference element (str or "mixture_of_knowns")
        - relative_change: (E_matrix - E_baseline) / E_baseline (float or None)
        - percent_change: relative_change × 100 (float or None)
        - assessment: "increase" | "decrease" | "no_significant_change" | "unknown"
        - matrix_composition: Normalized composition used
        - temperature_K: Temperature used (if any)
        - notes: Detailed explanation of calculation
        
    Example:
        >>> result = estimate_phase_modulus(
        ...     "FCC_A1",
        ...     {"AL": 0.96, "MG": 0.04},
        ...     temperature_K=700.0
        ... )
        >>> print(result["E_matrix_GPa"])  # ~56.5 GPa (temperature corrected)
    """
    # Step 1: Normalize composition
    comp = normalize_matrix_composition(matrix_phase_composition)
    
    # Step 2: Fallback to bulk composition if needed
    if not comp and fallback_to_bulk_composition:
        _log.info("Matrix phase composition empty, falling back to bulk composition")
        comp = normalize_matrix_composition(fallback_to_bulk_composition)
    
    # Step 3: Give up if still no composition
    if not comp:
        return {
            "E_matrix_GPa": None,
            "E_baseline_GPa": None,
            "baseline_element": None,
            "relative_change": None,
            "percent_change": None,
            "assessment": "unknown",
            "matrix_composition": {},
            "temperature_K": temperature_K,
            "notes": "No usable matrix composition after normalization (including fallback)."
        }
    
    # Step 4: Separate known and unknown elements
    known_elements = {el: x for el, x in comp.items() if el in ELEMENT_MODULUS_GPA}
    unknown_elements = [el for el in comp.keys() if el not in ELEMENT_MODULUS_GPA]
    
    # Step 5: Check if we have any data to work with
    if not known_elements:
        dominant = max(comp, key=comp.get)
        return {
            "E_matrix_GPa": None,
            "E_baseline_GPa": None,
            "baseline_element": dominant,
            "relative_change": None,
            "percent_change": None,
            "assessment": "unknown",
            "matrix_composition": comp,
            "temperature_K": temperature_K,
            "notes": (
                f"No modulus data for any elements in phase {matrix_phase_name}. "
                f"Unknown elements: {', '.join(sorted(unknown_elements))}. "
                f"Dominant element: {dominant}."
            )
        }
    
    # Step 6: Compute weighted average modulus over known elements
    # (ignoring unknown elements - they don't contribute to E_matrix)
    E_matrix_RT = sum(comp[el] * ELEMENT_MODULUS_GPA[el] for el in known_elements)
    
    # Step 7: Determine baseline
    # Use dominant element if its modulus is known; otherwise use mixture baseline
    dominant_element = max(comp, key=comp.get)
    
    if dominant_element in ELEMENT_MODULUS_GPA:
        E_baseline_RT = ELEMENT_MODULUS_GPA[dominant_element]
        baseline_label = dominant_element
    else:
        # Dominant element unknown → use weighted average of knowns as baseline
        # This allows us to still compute a relative change
        total_known_frac = sum(comp[el] for el in known_elements)
        if total_known_frac > 0:
            E_baseline_RT = sum(
                comp[el] * ELEMENT_MODULUS_GPA[el] for el in known_elements
            ) / total_known_frac
            baseline_label = "mixture_of_knowns"
        else:
            # Should not reach here (caught in step 5), but just in case
            E_baseline_RT = E_matrix_RT
            baseline_label = "mixture_of_knowns"
    
    # Step 8: Apply temperature correction if requested
    E_matrix = apply_temperature_correction(E_matrix_RT, comp, temperature_K)
    
    # For baseline, use single-element composition or full composition for mixture
    baseline_comp = {baseline_label: 1.0} if baseline_label in ELEMENT_MODULUS_GPA else comp
    E_baseline = apply_temperature_correction(E_baseline_RT, baseline_comp, temperature_K)
    
    # Step 9: Calculate relative change
    if E_baseline > 0:
        rel_change = (E_matrix - E_baseline) / E_baseline
        pct_change = rel_change * 100.0
    else:
        rel_change = None
        pct_change = None
    
    # Step 10: Assess significance (±10% threshold)
    if rel_change is None:
        assessment = "unknown"
    elif rel_change >= 0.10:
        assessment = "increase"
    elif rel_change <= -0.10:
        assessment = "decrease"
    else:
        assessment = "no_significant_change"
    
    # Step 11: Build explanatory notes
    notes_parts = [
        f"Phase: {matrix_phase_name}.",
        f"Dominant element: {dominant_element}.",
    ]
    
    if unknown_elements:
        notes_parts.append(
            f"Unknown elements (skipped): {', '.join(sorted(unknown_elements))}."
        )
    
    notes_parts.append(f"E(RT) mix = {E_matrix_RT:.1f} GPa.")
    
    if temperature_K and abs(temperature_K - ROOM_TEMPERATURE_K) > 5.0:
        notes_parts.append(
            f"E({temperature_K:.0f}K) ≈ {E_matrix:.1f} GPa (linear T-correction)."
        )
    
    if baseline_label == "mixture_of_knowns":
        notes_parts.append(
            f"Baseline: weighted average of known elements (dominant {dominant_element} has no modulus data)."
        )
    else:
        notes_parts.append(f"Baseline: {baseline_label} = {E_baseline_RT:.1f} GPa.")
    
    if rel_change is not None:
        notes_parts.append(f"Relative change: {pct_change:+.2f}%.")
    
    # Step 12: Return results
    return {
        "E_matrix_GPa": round(E_matrix, 2),
        "E_baseline_GPa": round(E_baseline, 2),
        "baseline_element": baseline_label,
        "relative_change": round(rel_change, 4) if rel_change is not None else None,
        "percent_change": round(pct_change, 2) if pct_change is not None else None,
        "assessment": assessment,
        "matrix_composition": comp,
        "temperature_K": temperature_K,
        "notes": " ".join(notes_parts)
    }

