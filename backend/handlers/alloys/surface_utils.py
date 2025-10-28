"""
Surface science utilities for alloy surface calculations.

This module handles surface-specific calculations including:
- Surface orientation normalization
- Facet-dependent diffusion mechanisms
- Surface diffusion barrier estimation
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
from ..base.constants import (
    DIFF_OVER_ADS_111,
    DIFF_OVER_ADS_100,
    DIFF_OVER_ADS_110,
)

_log = logging.getLogger(__name__)


def normalize_surface(surface_miller: Optional[str]) -> Tuple[str, float, str, float]:
    """
    Parse/normalize surface orientation and return facet-specific parameters.
    
    Args:
        surface_miller: Surface orientation string like '111', '100', '110', 
                       'FCC(111)', 'BCC-110', etc.
                       
    Returns:
        Tuple of (facet, facet_multiplier, mechanism, diff_over_ads):
        - facet: normalized facet string ('111', '100', '110', or 'unspecified')
        - facet_multiplier: scaling factor for diffusion barrier (111 easiest → 0.75)
        - mechanism: likely diffusion mechanism ('hopping', 'exchange', or 'unknown')
        - diff_over_ads: ratio of diffusion barrier to adsorption energy for this facet
        
    Notes:
        Stronger multipliers reflect well-known facet/mechanism differences:
        - (111): Close-packed, typically hopping mechanism, lowest barriers (0.75x)
        - (100): More open, exchange mechanism favored, medium barriers (1.25x)
        - (110): Most open/corrugated, exchange mechanism, highest barriers (1.45x)
    """
    if not surface_miller:
        # Unspecified facet: assume generic terrace and broaden uncertainty later
        return "unspecified", 1.00, "unknown", 0.18  # midpoint between hopping and exchange

    s = str(surface_miller).lower()
    
    # Remove structure type prefixes (fcc, bcc, etc.)
    for token in ("fcc", "bcc", "hcp", "sc", "diamond", "al", "au"):
        s = s.replace(token, "")
    
    # Remove separators and extract digits
    for ch in "()[]{}-–, ":
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch.isdigit())

    # Map to known facets
    if s == "111":
        return "111", 0.75, "hopping", DIFF_OVER_ADS_111
    if s == "100":
        return "100", 1.25, "exchange", DIFF_OVER_ADS_100
    if s == "110":
        return "110", 1.45, "exchange", DIFF_OVER_ADS_110
    
    return "unspecified", 1.00, "unknown", 0.18


def estimate_diffusion_barrier(
    ecoh_host: float,
    r_ad: float,
    r_host: float,
    facet: str,
    facet_mult: float,
    diff_over_ads: float,
    ads_over_coh: float
) -> Tuple[float, float, float]:
    """
    Estimate surface diffusion barrier using descriptor-based model.
    
    This implements a heuristic scaling model:
    - Adsorption energy scales with host cohesive energy (facet-dependent)
    - Diffusion barrier scales with adsorption energy
    - Size mismatch between adatom and host modulates the barrier
    
    Args:
        ecoh_host: Host element cohesive energy (eV/atom)
        r_ad: Adatom radius (Å)
        r_host: Host radius (Å)
        facet: Surface facet ('111', '100', '110', 'unspecified')
        facet_mult: Facet-specific multiplier
        diff_over_ads: Ratio of diffusion to adsorption energy for this facet
        ads_over_coh: Ratio of adsorption to cohesive energy for this facet
        
    Returns:
        Tuple of (ea_est, ea_low, ea_high):
        - ea_est: Estimated activation energy (eV)
        - ea_low: Lower bound estimate (eV)
        - ea_high: Upper bound estimate (eV)
    """
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
    
    return ea_est, ea_low, ea_high

