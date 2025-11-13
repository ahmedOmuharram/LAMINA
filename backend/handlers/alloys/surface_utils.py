"""
Surface science utilities for alloy surface calculations.

This module handles surface-specific calculations including:
- Surface orientation normalization
- Facet-dependent diffusion mechanisms
- Surface diffusion barrier estimation
- Multi-scenario uncertainty quantification
- Kinetics (rates and diffusion coefficients)
"""
from __future__ import annotations
import logging
import math
import random
from typing import Optional, Tuple, Dict, List, Any
from ..shared.constants import (
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


def compute_kinetics(
    ea_eV: float,
    temperature_K: float,
    attempt_frequency_Hz: float = 1.0e13,
    jump_distance_A: float = 2.5
) -> Dict[str, float]:
    """
    Convert activation energy to kinetic rates and surface diffusion coefficient.
    
    Uses Arrhenius equation: k = ν·exp(-Ea/kBT)
    Surface diffusion coefficient from 2D random walk: D ≈ a²·Γ/4
    
    Args:
        ea_eV: Activation energy (eV)
        temperature_K: Temperature (K)
        attempt_frequency_Hz: Attempt frequency ν (s⁻¹), typically 10¹²-10¹⁴ Hz
        jump_distance_A: Jump distance a (Å), typically nearest-neighbor distance
        
    Returns:
        Dict with rate (s⁻¹) and diffusion coefficient D (m²/s)
    """
    kBT = 8.617333262e-5 * temperature_K  # eV (Boltzmann constant in eV/K)
    rate = attempt_frequency_Hz * math.exp(-ea_eV / kBT)
    
    # 2D surface diffusion coefficient (convert Å to m)
    D_m2_s = (jump_distance_A * 1e-10) ** 2 * rate / 4.0
    
    return {
        "rate_s-1": rate,
        "D_m2_s": D_m2_s,
        "T_K": temperature_K,
        "attempt_frequency_Hz": attempt_frequency_Hz,
        "jump_distance_A": jump_distance_A
    }


def evaluate_scenario(
    ecoh_host: float,
    r_ad: float,
    r_host: float,
    facet_label: str,
    mechanism: str,
    defect: str,
    coverage_theta: float,
    oxide: bool,
    temperature_K: Optional[float],
    attempt_frequency_Hz: float,
    jump_distance_A: float,
    mc_samples: int,
    ads_over_coh_dict: Dict[str, float],
    use_dft_neb: bool = False,
    dft_backend: str = "chgnet",
    dft_workdir: str = "neb_runs",
    dft_kpts: Tuple[int, int, int] = (1, 1, 1),
    dft_images: int = 3,
    adatom_symbol: Optional[str] = None,
    host_symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a single diffusion scenario with uncertainty quantification.
    
    Args:
        ecoh_host: Host cohesive energy (eV)
        r_ad: Adatom radius (Å)
        r_host: Host radius (Å)
        facet_label: Facet string ('111', '100', '110', 'unspecified')
        mechanism: 'hopping' or 'exchange'
        defect: 'none', 'step', or 'vacancy'
        coverage_theta: Adatom coverage (ML, 0-1)
        oxide: If True, apply oxide penalty
        temperature_K: Temperature for kinetics (K), None to skip
        attempt_frequency_Hz: Attempt frequency (Hz)
        jump_distance_A: Jump distance (Å)
        mc_samples: Monte Carlo samples for uncertainty (0 to disable)
        ads_over_coh_dict: Facet-specific adsorption/cohesion ratios
        use_dft_neb: If True, use DFT+NEB for barrier calculation (default: False)
        dft_backend: Calculator ('chgnet', 'gpaw', or 'vasp') for NEB (default: 'chgnet')
        dft_workdir: Root directory for NEB runs
        dft_kpts: k-point mesh for DFT (default: (1,1,1) Γ-only for speed)
        dft_images: Number of NEB images (default: 3 for fast mode)
        adatom_symbol: Element symbol for adatom (required if use_dft_neb=True)
        host_symbol: Element symbol for host (required if use_dft_neb=True)
        
    Returns:
        Dict with scenario details, energy ranges (P10/P50/P90), and kinetics
        
    Notes:
        - DFT NEB is only supported for clean terrace diffusion (defect='none')
        - For DFT calculations on facets '111', '100', '110', set use_dft_neb=True
        - If NEB fails, automatically falls back to heuristic with warning
    """
    # ==================== DFT+NEB Path ====================
    # If requested and applicable, run first-principles NEB calculation
    if use_dft_neb and facet_label in ("111", "100", "110") and defect == "none":
        if not adatom_symbol or not host_symbol:
            _log.warning("DFT NEB requires adatom_symbol and host_symbol. Falling back to heuristic.")
        else:
            try:
                from .dft_neb import run_neb
                import os
                
                _log.info(f"Running DFT NEB: {adatom_symbol}/{host_symbol} {facet_label} {mechanism}")
                
                # Create unique workdir for this scenario
                scenario_dir = os.path.join(
                    dft_workdir,
                    f"{host_symbol}_{adatom_symbol}_{facet_label}_{mechanism}"
                )
                
                # Auto-bump images for hopping on smooth facets (avoid missing shallow saddles)
                images_req = dft_images
                if mechanism == "hopping" and facet_label in ("111", "100") and dft_images < 5:
                    images_req = 5
                    _log.info(f"Auto-bumping images to {images_req} for {mechanism} on {facet_label}")
                
                neb_result = run_neb(
                    adatom=adatom_symbol,
                    host=host_symbol,
                    facet=facet_label,
                    mechanism=mechanism,
                    backend=dft_backend,
                    images_n=images_req,
                    kpts=dft_kpts,
                    workdir=scenario_dir,
                    compute_prefactor=False  # Set True for production if you need accurate ν
                )
                
                # Extract DFT barrier
                ea_dft = neb_result["Ea_eV"]
                
                # If the surrogate says "downhill" (Ea~0), fall back to heuristic
                if ea_dft < 1e-3 or "downhill" in " ".join(neb_result.get("notes", [])):
                    _log.warning("DFT NEB returned barrierless path on surrogate; falling back to heuristic scaling.")
                    raise RuntimeError("Downhill path (no TS) on surrogate")
                
                # For single DFT calculation, use tight uncertainty bands (±10%)
                q10 = round(ea_dft * 0.90, 4)
                q50 = round(ea_dft, 4)
                q90 = round(ea_dft * 1.10, 4)
                
                # Use DFT-calculated kinetics if temperature provided
                kinetics = None
                if temperature_K and temperature_K > 0:
                    kBT = 8.617333262e-5 * temperature_K
                    rate = neb_result["prefactor_Hz"] * math.exp(-ea_dft / kBT)
                    D_m2_s = (jump_distance_A * 1e-10) ** 2 * rate / 4.0
                    
                    kinetics = {
                        "P50": {
                            "rate_s-1": rate,
                            "D_m2_s": D_m2_s,
                            "T_K": temperature_K,
                            "attempt_frequency_Hz": neb_result["prefactor_Hz"],
                            "jump_distance_A": jump_distance_A
                        }
                    }
                    # Add uncertainty bands for kinetics
                    rate_low = neb_result["prefactor_Hz"] * math.exp(-q90 / kBT)
                    rate_high = neb_result["prefactor_Hz"] * math.exp(-q10 / kBT)
                    kinetics["P10"] = {
                        "rate_s-1": rate_low,
                        "D_m2_s": (jump_distance_A * 1e-10) ** 2 * rate_low / 4.0,
                        "T_K": temperature_K,
                        "attempt_frequency_Hz": neb_result["prefactor_Hz"],
                        "jump_distance_A": jump_distance_A
                    }
                    kinetics["P90"] = {
                        "rate_s-1": rate_high,
                        "D_m2_s": (jump_distance_A * 1e-10) ** 2 * rate_high / 4.0,
                        "T_K": temperature_K,
                        "attempt_frequency_Hz": neb_result["prefactor_Hz"],
                        "jump_distance_A": jump_distance_A
                    }
                
                _log.info(f"DFT NEB completed: Ea = {ea_dft:.4f} eV")
                
                return {
                    "facet": facet_label,
                    "mechanism": mechanism,
                    "defect": defect,
                    "Ea_eV": {
                        "P10": q10,
                        "P50": q50,
                        "P90": q90
                    },
                    "baseline_Ea_eV": q50,
                    "modifiers": {
                        "mechanism_mult": 1.0,
                        "defect_mult": 1.0,
                        "coverage_mult": 1.0,
                        "oxide_mult": 1.0
                    },
                    "kinetics": kinetics,
                    "provenance": "DFT+NEB",
                    "dft_details": {
                        "backend": dft_backend,
                        "kpts": dft_kpts,
                        "images": dft_images,
                        "ts_image": neb_result["ts_image_index"],
                        "energy_profile_eV": neb_result["energies_eV"],
                        "notes": neb_result["notes"]
                    }
                }
                
            except ImportError as e:
                _log.warning(f"DFT NEB import failed: {e}. Falling back to heuristic.")
            except Exception as e:
                _log.warning(f"DFT NEB calculation failed: {e}. Falling back to heuristic.")
    
    # ==================== Heuristic Path ====================
    # Mechanism-dependent multipliers (facet-specific)
    # Exchange often favored on open surfaces (100/110), hopping on close-packed (111)
    MECH_MULT = {
        ("111", "exchange"): 1.10,
        ("111", "hopping"): 1.00,
        ("100", "exchange"): 0.85,
        ("100", "hopping"): 1.05,
        ("110", "exchange"): 0.90,
        ("110", "hopping"): 1.05,
        ("unspecified", "exchange"): 0.95,
        ("unspecified", "hopping"): 1.00,
    }
    
    # Defect-assisted pathways typically lower barriers
    DEFECT_MULT = {
        "none": 1.00,
        "step": 0.80,      # step-edge diffusion often easier
        "vacancy": 0.70,   # vacancy-mediated exchange can be significantly easier
    }
    
    # Coverage penalty (crowding effects, mild and linear)
    COV_MULT = 1.0 + 0.4 * max(0.0, min(1.0, coverage_theta))
    
    # Oxide penalty for reactive hosts (e.g., Al native oxide)
    OXIDE_MULT = 1.30 if oxide else 1.00
    
    # Monte Carlo uncertainty level (15% std on scaling factors)
    MC_SIGMA = 0.15
    
    # Get facet parameters
    facet_norm, facet_mult, _, diff_over_ads = normalize_surface(facet_label)
    
    # Get adsorption scaling
    ads_over_coh = ads_over_coh_dict.get(facet_norm, 0.25)
    
    # Baseline barrier (no modifiers yet)
    ea_base, _, _ = estimate_diffusion_barrier(
        ecoh_host, r_ad, r_host, facet_norm, facet_mult,
        diff_over_ads, ads_over_coh
    )
    
    # Apply all modifiers
    mech_mult = MECH_MULT.get((facet_norm, mechanism), 1.0)
    ea_adj = ea_base * mech_mult * DEFECT_MULT[defect] * COV_MULT * OXIDE_MULT
    ea_adj = max(0.01, min(2.0, ea_adj))
    
    # Monte Carlo uncertainty quantification
    mc_vals = []
    if mc_samples > 0:
        for _ in range(mc_samples):
            # Lognormal-ish multiplicative noise on each scaling factor
            mm = mech_mult * math.exp(random.gauss(0.0, MC_SIGMA))
            dm = DEFECT_MULT[defect] * math.exp(random.gauss(0.0, MC_SIGMA))
            cm = COV_MULT * math.exp(random.gauss(0.0, MC_SIGMA * 0.5))
            om = OXIDE_MULT * math.exp(random.gauss(0.0, MC_SIGMA * 0.5))
            
            # Small noise on adsorption/diffusion scalings
            doa = diff_over_ads * math.exp(random.gauss(0.0, MC_SIGMA))
            aoc = ads_over_coh * math.exp(random.gauss(0.0, MC_SIGMA))
            
            # Rebuild baseline with perturbed parameters
            ea_j_base, _, _ = estimate_diffusion_barrier(
                ecoh_host, r_ad, r_host, facet_norm, facet_mult, doa, aoc
            )
            ea_j = max(0.01, min(2.0, ea_j_base * mm * dm * cm * om))
            mc_vals.append(ea_j)
    
    # Calculate quantiles
    if mc_vals:
        mc_vals.sort()
        q10 = mc_vals[int(0.10 * len(mc_vals))]
        q50 = mc_vals[int(0.50 * len(mc_vals))]
        q90 = mc_vals[int(0.90 * len(mc_vals))]
    else:
        # Simple ±50% bands if no MC
        q10 = max(0.01, ea_adj * 0.5)
        q50 = ea_adj
        q90 = min(2.0, ea_adj * 1.5)
    
    # Compute kinetics if temperature provided
    kinetics = None
    if temperature_K and temperature_K > 0:
        kinetics = {
            "P10": compute_kinetics(q10, temperature_K, attempt_frequency_Hz, jump_distance_A),
            "P50": compute_kinetics(q50, temperature_K, attempt_frequency_Hz, jump_distance_A),
            "P90": compute_kinetics(q90, temperature_K, attempt_frequency_Hz, jump_distance_A),
        }
    
    return {
        "facet": facet_norm,
        "mechanism": mechanism,
        "defect": defect,
        "Ea_eV": {
            "P10": round(q10, 4),
            "P50": round(q50, 4),
            "P90": round(q90, 4)
        },
        "baseline_Ea_eV": round(ea_base, 4),
        "modifiers": {
            "mechanism_mult": round(mech_mult, 3),
            "defect_mult": DEFECT_MULT[defect],
            "coverage_mult": round(COV_MULT, 3),
            "oxide_mult": OXIDE_MULT
        },
        "kinetics": kinetics,
        "provenance": "heuristic"
    }


def generate_scenarios(
    facets_requested: List[str],
    include_defects: bool
) -> List[Tuple[str, str, str]]:
    """
    Generate list of (facet, mechanism, defect) scenario tuples to evaluate.
    
    Args:
        facets_requested: List of facet strings (e.g., ['111', '100', '110'])
        include_defects: If True, include step and vacancy scenarios
        
    Returns:
        List of (facet, mechanism, defect) tuples
    """
    MECHANISMS = ["hopping", "exchange"]
    DEFECTS = ["none"]
    if include_defects:
        DEFECTS += ["step", "vacancy"]
    
    scenarios = []
    for facet in facets_requested:
        for mechanism in MECHANISMS:
            for defect in DEFECTS:
                scenarios.append((facet, mechanism, defect))
    
    return scenarios


def generate_warnings(
    host: str,
    r_ad: float,
    r_host: float,
    oxide: bool
) -> List[str]:
    """
    Generate warnings for potentially problematic scenarios.
    
    Args:
        host: Host element symbol
        r_ad: Adatom radius (Å)
        r_host: Host radius (Å)
        oxide: Whether oxide conditions are considered
        
    Returns:
        List of warning strings
    """
    warnings = []
    
    # Size mismatch warning
    size_mismatch_rel = abs((r_ad - r_host) / r_host)
    if size_mismatch_rel > 0.25:
        warnings.append(
            f"Large size mismatch ({size_mismatch_rel*100:.1f}%) — "
            "mechanism may switch or cluster diffusion may dominate."
        )
    
    # Oxide warning for reactive metals
    if host in ("Al", "Ti", "Mg") and not oxide:
        warnings.append(
            f"{host} oxidizes readily; clean terraces require UHV. "
            "Consider oxide=True for ambient conditions."
        )
    
    return warnings

