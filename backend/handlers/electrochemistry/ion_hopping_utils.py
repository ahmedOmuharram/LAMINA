"""
Ion hopping barrier estimation utilities for battery electrode materials.

This module provides comprehensive scenario-based ion diffusion barrier estimation
with uncertainty quantification (P10/P50/P90) and kinetics calculations.

Key features:
- Graphite/graphene-specific scenarios (AB-BLG, AA-BLG, staging, pathways)
- Modifiers for stacking, coverage, defects, strain, interlayer spacing
- Monte Carlo uncertainty quantification
- Arrhenius kinetics and diffusion coefficients
- Fallback to structure heuristics for non-graphite materials
"""
from __future__ import annotations
import logging
import math
import random
from typing import Dict, Any, List, Tuple, Optional

_log = logging.getLogger(__name__)

# Physical constants
K_B = 8.617333262145e-5  # eV/K (Boltzmann constant)

# ============================================================================
# Graphite/Graphene Pathway Database
# ============================================================================

# Literature-anchored baseline barriers (eV) for graphitic systems
# Values reflect the landscape from DFT-NEB studies:
# - AB-BLG TH↔TH ≈ 0.07 eV (very fast, dilute)
# - Outside (H↔H) ≈ 0.25 eV
# - AA-BLG in-gallery ≈ 0.34 eV
# - Graphite stage I (LiC6) ≈ 0.25–0.40 eV
# - Graphite stage II (LiC12) ≈ 0.15–0.30 eV
# - Cross-plane hops are typically higher (0.45–0.8 eV)

GRAPHITE_PATH_DB: Dict[str, Dict[str, Any]] = {
    "AB_BLG_in_gallery_TH-TH": {
        "baseline_ea": 0.07,
        "spread": 1.3,
        "hop_A": 2.46,
        "note": "AB-BLG intergallery TH↔TH (dilute, very fast)",
        "citations": [
            "Kganyago & Ngoepe, PRB 68, 205111 (2003): TH site AB-stacked",
            "Zheng et al., Nano Lett. 14, 2345 (2014): AB-BLG ultrafast diffusion"
        ]
    },
    "AB_BLG_outside_H-H": {
        "baseline_ea": 0.25,
        "spread": 1.25,
        "hop_A": 2.46,
        "note": "Outside graphene H↔H (surface adsorption)",
        "citations": [
            "Persson et al., PRB 82, 125416 (2010): surface pathways higher than gallery"
        ]
    },
    "AA_BLG_in_gallery_H-H": {
        "baseline_ea": 0.34,
        "spread": 1.25,
        "hop_A": 2.46,
        "note": "AA-BLG intergallery H↔H (slower than AB-TH)",
        "citations": [
            "Kganyago & Ngoepe, PRB 68, 205111 (2003): AA stacking raises barriers"
        ]
    },
    "Graphite_stageI_in_gallery": {
        "baseline_ea": 0.28,
        "spread": 1.3,
        "hop_A": 2.46,
        "note": "Stage I (LiC6) intragallery in-plane",
        "citations": [
            "Persson et al., PRB 82, 125416 (2010): 293 meV for LiC6",
            "Umegaki et al., PCCP 19, 19058 (2017): 270±5 meV"
        ]
    },
    "Graphite_stageII_in_gallery": {
        "baseline_ea": 0.20,
        "spread": 1.3,
        "hop_A": 2.46,
        "note": "Stage II (LiC12) intragallery in-plane (faster than stage I)",
        "citations": [
            "Persson et al., PRB 82, 125416 (2010): 218-283 meV range",
            "Umegaki et al., PCCP 19, 19058 (2017): 170±20 meV"
        ]
    },
    "Graphite_cross_plane": {
        "baseline_ea": 0.60,
        "spread": 1.4,
        "hop_A": 3.35,
        "note": "Cross-plane hop (interlayer transport, slow)",
        "citations": [
            "Persson et al., PRB 82, 125416 (2010): interlayer much slower"
        ]
    },
    "Graphite_edge_defect": {
        "baseline_ea": 0.18,
        "spread": 1.5,
        "hop_A": 3.0,
        "note": "Edge/defect-assisted channel (lowered barrier)",
        "citations": [
            "Literature: defect channels can reduce barriers by 30-50%"
        ]
    },
}


# ============================================================================
# Modifier Functions
# ============================================================================

def coverage_mult(theta: float) -> float:
    """
    Coverage modifier for gallery occupancy.
    
    Args:
        theta: Fractional gallery occupancy (0 = dilute, 1 = full for that stage)
               Measured relative to stage saturation
    
    Returns:
        Multiplicative factor (repulsion raises barriers with coverage)
    
    Notes:
        - Gentle linear increase with coverage
        - At θ=1 (full stage), barrier is ~40% higher
        - Based on Li-Li repulsion in graphite galleries
    """
    alpha = 0.4  # tunable per material
    return max(0.85, 1.0 + alpha * theta)


def stacking_mult(stacking: str) -> float:
    """
    Stacking modifier for bilayer graphene.
    
    Args:
        stacking: Stacking type ('AB', 'AA', or unknown)
    
    Returns:
        Multiplicative factor
    
    Notes:
        - AB stacking: TH sites (tetrahedral) lowest barrier → 1.0×
        - AA stacking: H sites (hollow) higher barrier → ~4.9× (0.34/0.07)
        - Unknown: slight penalty → 1.1×
    """
    s = stacking.lower()
    if "ab" in s:
        return 1.0
    if "aa" in s:
        return 0.34 / 0.07  # ~4.9× harder than AB-BLG TH↔TH
    return 1.1  # unknown → slight penalty


def defect_mult(defect: str) -> float:
    """
    Defect modifier for alternative pathways.
    
    Args:
        defect: Defect type ('vacancy', 'edge', 'grain_boundary', 'none')
    
    Returns:
        Multiplicative factor (defects typically lower barriers)
    
    Notes:
        - Vacancy/edge/grain boundaries provide easier pathways
        - Can reduce barriers by 10-30%
    """
    d = (defect or "none").lower()
    if d in ("vacancy", "edge", "grain_boundary"):
        return 0.7  # easier pathways
    if d in ("none", "clean"):
        return 1.0
    return 0.9  # generic defect


def strain_mult(eps_percent: float) -> float:
    """
    Strain modifier for lattice deformation.
    
    Args:
        eps_percent: Strain percentage (positive = expansion, negative = compression)
    
    Returns:
        Multiplicative factor
    
    Notes:
        - Interlayer expansion lowers barriers for gallery diffusion
        - ~5% expansion → ~15% drop in barrier
        - Compression raises barriers
        - Clamped to [0.75, 1.25] for physical reasonableness
    """
    beta = -0.03  # per % strain (positive → expansion)
    return max(0.75, min(1.25, 1.0 + beta * eps_percent))


def il_dist_mult(delta_A: float) -> float:
    """
    Interlayer spacing modifier.
    
    Args:
        delta_A: Change in interlayer spacing (Å) vs nominal ~3.35 Å
                 Positive = expansion, negative = compression
    
    Returns:
        Multiplicative factor
    
    Notes:
        - Explicit interlayer spacing control
        - +0.1 Å → few-% drop in barrier
        - Clamped to [0.75, 1.2] for physical reasonableness
    """
    gamma = -0.12  # per Å (tunable)
    return max(0.75, min(1.2, 1.0 + gamma * delta_A))


# ============================================================================
# Kinetics Functions
# ============================================================================

def arrhenius_rate(Ea: float, T: float, nu: float = 1e13) -> float:
    """
    Calculate Arrhenius hopping rate.
    
    Args:
        Ea: Activation energy (eV)
        T: Temperature (K)
        nu: Attempt frequency (Hz), typically ~1e13
    
    Returns:
        Hopping rate (s^-1)
    """
    return float(nu * math.exp(-Ea / (K_B * T)))


def diffusivity_2D(rate: float, hop_A: float) -> float:
    """
    Calculate 2D diffusivity from hopping rate.
    
    Args:
        rate: Hopping rate (s^-1)
        hop_A: Hop distance (Å)
    
    Returns:
        Diffusion coefficient (m^2/s)
    
    Notes:
        D ≈ a² Γ / 4 (square lattice approximation)
        For hexagonal lattice, factor differences are small for scoping
    """
    a_m = hop_A * 1e-10  # Convert Å to m
    return float((a_m ** 2) * rate / 4.0)


def mc_percentiles(samples: List[float]) -> Tuple[float, float, float]:
    """
    Calculate P10, P50 (median), P90 from Monte Carlo samples.
    
    Args:
        samples: List of sampled values
    
    Returns:
        Tuple of (P10, P50, P90)
    """
    xs = sorted(samples)
    
    def pct(p: float) -> float:
        i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
        return xs[i]
    
    return pct(0.10), pct(0.50), pct(0.90)


# ============================================================================
# Scenario Generation
# ============================================================================

def is_graphitic(host: str) -> bool:
    """Check if host material is graphite/graphene family."""
    h = host.lower()
    return ("graphite" in h) or (h.strip() in ("c", "c6", "c12", "lic6", "lic12")) or ("blg" in h) or ("bilayer" in h)


def graphite_scenarios(host: str) -> List[Dict[str, Any]]:
    """
    Generate default portfolio of scenarios for graphite/graphene family.
    
    Args:
        host: Host material string
    
    Returns:
        List of scenario dictionaries with baseline parameters
    """
    h = host.lower()
    scenarios = []
    
    # Try to infer staging hint from formula
    stage = "generic"
    if "lic6" in h or "stage i" in h:
        stage = "stageI"
    elif "lic12" in h or "stage ii" in h:
        stage = "stageII"
    
    base_keys = []
    
    # Bilayer graphene scenarios
    if "blg" in h or "bilayer" in h:
        base_keys += [
            "AB_BLG_in_gallery_TH-TH",
            "AA_BLG_in_gallery_H-H",
            "AB_BLG_outside_H-H"
        ]
    else:
        # Graphite generic
        if stage == "stageI":
            base_keys += [
                "Graphite_stageI_in_gallery",
                "Graphite_cross_plane",
                "Graphite_edge_defect"
            ]
        elif stage == "stageII":
            base_keys += [
                "Graphite_stageII_in_gallery",
                "Graphite_cross_plane",
                "Graphite_edge_defect"
            ]
        else:
            base_keys += [
                "Graphite_stageI_in_gallery",
                "Graphite_stageII_in_gallery",
                "Graphite_cross_plane",
                "Graphite_edge_defect",
                "AB_BLG_outside_H-H"
            ]
    
    for k in base_keys:
        db = GRAPHITE_PATH_DB[k]
        
        # Sensible defaults per key
        stacking = "AB" if "AB" in k else ("AA" if "AA" in k else "AB")
        defect = "none"
        
        # Coverage defaults
        if "BLG" in k:
            theta = 0.0  # Dilute for BLG
        elif "stageI" in k:
            theta = 0.5  # Half-filled stage I
        elif "stageII" in k:
            theta = 0.25  # Quarter-filled stage II
        else:
            theta = 0.1  # Generic low coverage
        
        eps = 0.0
        delta_A = 0.0
        
        scenarios.append({
            "key": k,
            "stacking": stacking,
            "defect": defect,
            "theta": theta,
            "strain_percent": eps,
            "delta_interlayer_A": delta_A,
            "baseline_ea": db["baseline_ea"],
            "hop_A": db["hop_A"],
            "spread": db["spread"],
            "note": db["note"],
            "citations": db.get("citations", [])
        })
    
    return scenarios


def expand_variants(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate small variants around a base scenario.
    
    Varies: coverage, defects, interlayer spacing, strain
    
    Args:
        s: Base scenario dictionary
    
    Returns:
        List of variant scenarios
    """
    variants = []
    
    # Coverage grid: dilute → moderate
    coverage_grid = [0.0, 0.25, 0.5]
    
    # Defect types
    defects = ["none", "edge", "vacancy"]
    
    # Interlayer spacing changes (Å)
    spacings = [0.0, +0.1]  # nominal and expanded
    
    # Strain (%)
    strains = [0.0, +3.0]  # nominal and tensile
    
    for th in coverage_grid:
        for d in defects:
            for dA in spacings:
                for eps in strains:
                    v = dict(s)
                    v.update({
                        "theta": th,
                        "defect": d,
                        "delta_interlayer_A": dA,
                        "strain_percent": eps
                    })
                    variants.append(v)
    
    return variants


# ============================================================================
# Scenario Evaluation
# ============================================================================

def evaluate_ion_hopping_scenario(
    scenario: Dict[str, Any],
    temperatures_K: List[float],
    mc_samples: int,
    attempt_frequency_Hz: float = 1e13
) -> Dict[str, Any]:
    """
    Evaluate a single ion hopping scenario with MC uncertainty quantification.
    
    Args:
        scenario: Scenario dictionary with baseline parameters and modifiers
        temperatures_K: List of temperatures for kinetics calculation
        mc_samples: Number of Monte Carlo samples for uncertainty
        attempt_frequency_Hz: Attempt frequency for Arrhenius rate
    
    Returns:
        Dictionary with scenario context, P10/P50/P90 barriers, modifiers, and kinetics
    """
    # Calculate modifier multipliers
    s_mult_stack = stacking_mult(scenario.get("stacking", "AB"))
    s_mult_def = defect_mult(scenario.get("defect", "none"))
    s_mult_cov = coverage_mult(float(scenario.get("theta", 0.0)))
    s_mult_strain = strain_mult(float(scenario.get("strain_percent", 0.0)))
    s_mult_dA = il_dist_mult(float(scenario.get("delta_interlayer_A", 0.0)))
    
    # Effective multiplier
    eff_mult = s_mult_stack * s_mult_def * s_mult_cov * s_mult_strain * s_mult_dA
    
    # Base barrier with modifiers
    base = float(scenario["baseline_ea"]) * eff_mult
    spread = float(scenario["spread"])  # multiplicative noise window
    
    # Monte Carlo sampling (log-normal-ish via multiplicative jitter)
    samples = []
    for _ in range(int(mc_samples)):
        # Log-normal sampling
        jitter = math.exp(random.uniform(-math.log(spread) / 2, math.log(spread) / 2))
        samples.append(max(0.01, min(2.0, base * jitter)))
    
    # Calculate percentiles
    p10, p50, p90 = mc_percentiles(samples)
    
    # Kinetics at each temperature
    kin = {}
    for T in temperatures_K:
        rate_p10 = arrhenius_rate(p10, T, attempt_frequency_Hz)
        rate_p50 = arrhenius_rate(p50, T, attempt_frequency_Hz)
        rate_p90 = arrhenius_rate(p90, T, attempt_frequency_Hz)
        
        D_p10 = diffusivity_2D(rate_p10, scenario["hop_A"])
        D_p50 = diffusivity_2D(rate_p50, scenario["hop_A"])
        D_p90 = diffusivity_2D(rate_p90, scenario["hop_A"])
        
        kin[str(int(round(T)))] = {
            "rate_s-1": {"P10": rate_p10, "P50": rate_p50, "P90": rate_p90},
            "D_m2_s": {"P10": D_p10, "P50": D_p50, "P90": D_p90},
            "jump_distance_A": scenario["hop_A"]
        }
    
    return {
        "context": {
            "key": scenario["key"],
            "stacking": scenario.get("stacking", "n/a"),
            "defect": scenario.get("defect", "none"),
            "theta": scenario.get("theta", 0.0),
            "strain_percent": scenario.get("strain_percent", 0.0),
            "delta_interlayer_A": scenario.get("delta_interlayer_A", 0.0),
            "note": scenario.get("note", ""),
        },
        "Ea_eV": {
            "P10": round(p10, 4),
            "P50": round(p50, 4),
            "P90": round(p90, 4)
        },
        "baseline_Ea_eV": round(scenario["baseline_ea"], 4),
        "modifiers": {
            "stacking_mult": round(s_mult_stack, 3),
            "defect_mult": round(s_mult_def, 3),
            "coverage_mult": round(s_mult_cov, 3),
            "strain_mult": round(s_mult_strain, 3),
            "spacing_mult": round(s_mult_dA, 3),
        },
        "kinetics": kin,
    }

