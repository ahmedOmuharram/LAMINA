"""
Utility functions for magnet strength assessment and magnetic material analysis.

This module provides utilities for:
- Phase identification and stability checking
- Material magnetic property estimation (Br, Hc, (BH)max, Ms, Bs, Tc)
- Pull force calculations using standard geometries
- Comparison of baseline vs doped magnetic materials
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pymatgen.core import Element, Composition

_log = logging.getLogger(__name__)

# Physical constants
MU_0 = 4e-7 * np.pi  # Vacuum permeability (H/m)
BOHR_MAGNETON = 9.274e-24  # A⋅m²


def fetch_phase_and_mp_data(
    mpr,
    formula: str
) -> Dict[str, Any]:
    """
    Fetch phase, structure, stability, and magnetic data from Materials Project.
    
    Args:
        mpr: MPRester client instance
        formula: Chemical formula (e.g., "Fe2O3" or "FeAlO3")
        
    Returns:
        Dictionary containing:
        - Phase information (space group, crystal system)
        - Stability (energy_above_hull, is_stable)
        - Magnetic ordering type (FM, AFM, FiM, etc.)
        - Total magnetization per cell
        - Structure data
    """
    try:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id", "formula_pretty", "composition", "structure",
                "energy_above_hull", "is_stable", "symmetry",
                "is_magnetic", "ordering", "total_magnetization",
                "total_magnetization_normalized_vol",
                "total_magnetization_normalized_formula_units",
                "num_magnetic_sites", "types_of_magnetic_species",
                "volume", "nsites"
            ],
        )
        
        if not docs:
            return {
                "success": False,
                "error": f"No materials found for formula {formula}"
            }
        
        # Sort by stability (lowest energy above hull first)
        docs = sorted(docs, key=lambda x: getattr(x, 'energy_above_hull', None) or float('inf'))
        doc = docs[0]  # Most stable polymorph
        
        result = {
            "success": True,
            "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
            "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else formula,
            "composition": dict(doc.composition.as_dict()) if hasattr(doc, 'composition') else None,
            "stability": {
                "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None,
                "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
                "unit": "eV/atom"
            }
        }
        
        # Phase/structure info
        if hasattr(doc, 'symmetry') and doc.symmetry:
            sym = doc.symmetry
            result["phase"] = {
                "space_group": str(sym.symbol) if hasattr(sym, 'symbol') else None,
                "crystal_system": str(sym.crystal_system) if hasattr(sym, 'crystal_system') else None,
                "space_group_number": int(sym.number) if hasattr(sym, 'number') else None
            }
        
        # Magnetic ordering
        result["magnetic_ordering"] = {
            "is_magnetic": doc.is_magnetic if hasattr(doc, 'is_magnetic') else None,
            "ordering_type": str(doc.ordering) if hasattr(doc, 'ordering') and doc.ordering else "Unknown"
        }
        
        # Magnetization data
        if hasattr(doc, 'total_magnetization') and doc.total_magnetization is not None:
            result["total_magnetization_muB"] = float(doc.total_magnetization)
        
        if hasattr(doc, 'total_magnetization_normalized_formula_units') and doc.total_magnetization_normalized_formula_units is not None:
            result["magnetization_per_fu_muB"] = float(doc.total_magnetization_normalized_formula_units)
        
        if hasattr(doc, 'total_magnetization_normalized_vol') and doc.total_magnetization_normalized_vol is not None:
            result["magnetization_per_vol_muB_per_bohr3"] = float(doc.total_magnetization_normalized_vol)
        
        # Magnetic sites
        if hasattr(doc, 'num_magnetic_sites'):
            result["num_magnetic_sites"] = doc.num_magnetic_sites
        
        if hasattr(doc, 'types_of_magnetic_species') and doc.types_of_magnetic_species:
            result["magnetic_species"] = [str(s) for s in doc.types_of_magnetic_species]
        
        # Volume and nsites (needed for magnetization calculations)
        if hasattr(doc, 'volume'):
            result["volume_A3"] = float(doc.volume)
        if hasattr(doc, 'nsites'):
            result["nsites"] = int(doc.nsites)
        
        # Get all polymorphs
        if len(docs) > 1:
            result["num_polymorphs"] = len(docs)
            result["note"] = f"Found {len(docs)} polymorphs; returning most stable"
        
        return result
        
    except Exception as e:
        _log.error(f"Error fetching phase and MP data for {formula}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def estimate_saturation_magnetization_T(
    magnetization_muB: Optional[float],
    volume_A3: Optional[float]
) -> Optional[float]:
    """
    Convert magnetization from μB per cell to Tesla (saturation magnetization Bs).
    
    Bs = μ0 * Ms where Ms is magnetization in A/m.
    
    Args:
        magnetization_muB: Total magnetization in Bohr magnetons per unit cell
        volume_A3: Unit cell volume in Ų
        
    Returns:
        Saturation magnetization Bs in Tesla, or None if inputs are missing
    """
    if magnetization_muB is None or volume_A3 is None:
        return None
    
    # Convert volume from Ų to m³
    volume_m3 = volume_A3 * 1e-30
    
    # Magnetization density: μB per m³
    magnetization_density_muB_per_m3 = abs(magnetization_muB) / volume_m3
    
    # Convert to A/m: Ms = (magnetization in μB/m³) * (Bohr magneton in A⋅m²)
    Ms_A_per_m = magnetization_density_muB_per_m3 * BOHR_MAGNETON
    
    # Bs = μ0 * Ms
    Bs_T = MU_0 * Ms_A_per_m
    
    return float(Bs_T)


def estimate_material_properties(
    mp_data: Dict[str, Any],
    literature_hint: Optional[Dict[str, Any]] = None,
    kappa: float = 0.7
) -> Dict[str, Any]:
    """
    Estimate magnetic material properties for permanent magnet assessment.
    
    Priority:
    1. Use literature values if provided (Br, Hc, (BH)max, Tc)
    2. Otherwise estimate from DFT/MP data
    
    Estimation approach:
    - Bs = μ0 * Ms (from DFT magnetization)
    - Br ≈ κ * Bs (κ typically 0.7-0.95 for hard magnets; VERY LOW for AFM/WF)
    - Hc: use literature or heuristics based on ordering
    - (BH)max ≈ Br² / (4μ0) for ideal case
    
    Args:
        mp_data: Dictionary from fetch_phase_and_mp_data
        literature_hint: Optional dict with {"Br": ..., "Hc": ..., "BHmax": ..., "Tc": ...}
        kappa: Remanence factor (Br/Bs ratio). Default 0.7. Reduce for weak-FM or low Hc.
        
    Returns:
        Dictionary with estimated properties:
        - Bs (saturation magnetization, Tesla)
        - Ms (magnetization, A/m)
        - Br (remanence, Tesla)
        - Hc (coercivity, A/m or kA/m)
        - (BH)max (max energy product, kJ/m³)
        - Tc (Curie temperature, K) if available
        - source: "literature" or "estimated"
    """
    try:
        result = {
            "success": True,
            "source": "estimated",
            "assumptions": [],
            "caveats": []
        }
        
        # If literature values provided, use them
        if literature_hint:
            result["source"] = "literature"
            if "Br" in literature_hint:
                result["Br_T"] = float(literature_hint["Br"])
            if "Hc" in literature_hint:
                result["Hc_kA_per_m"] = float(literature_hint["Hc"])
            if "BHmax" in literature_hint:
                result["BHmax_kJ_per_m3"] = float(literature_hint["BHmax"])
            if "Tc" in literature_hint:
                result["Tc_K"] = float(literature_hint["Tc"])
            
            result["assumptions"].append("Using literature values where provided")
            return result
        
        # Otherwise estimate from MP data
        ordering = mp_data.get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        is_magnetic = mp_data.get("magnetic_ordering", {}).get("is_magnetic", False)
        
        # Get magnetization
        magnetization_muB = mp_data.get("total_magnetization_muB")
        volume_A3 = mp_data.get("volume_A3")
        
        # Estimate Bs and Ms
        if magnetization_muB is not None and volume_A3 is not None:
            Bs_T = estimate_saturation_magnetization_T(magnetization_muB, volume_A3)
            result["Bs_T"] = float(Bs_T) if Bs_T else None
            
            if Bs_T:
                Ms_A_per_m = Bs_T / MU_0
                result["Ms_kA_per_m"] = float(Ms_A_per_m / 1000)  # Convert to kA/m
                result["assumptions"].append(f"Bs = μ0 * Ms calculated from DFT magnetization ({magnetization_muB:.2f} μB)")
        else:
            result["Bs_T"] = None
            result["Ms_kA_per_m"] = None
            result["caveats"].append("Magnetization data not available; cannot estimate Bs/Ms")
        
        # Estimate Br from Bs - FIXED for AFM/WF
        if result.get("Bs_T") is not None:
            # Adjust kappa based on ordering
            if ordering in ["FM", "FiM"]:
                kappa_adj = kappa  # Use provided kappa (default 0.7)
            elif ordering in ["AFM", "WF"]:
                # AFM/WF have VERY low remanence (weak-FM canting gives tiny Br)
                kappa_adj = min(0.1, max(0.02, kappa * 0.1))
                result["assumptions"].append(f"AFM/WF: κ reduced to {kappa_adj:.2f} (very low remanence)")
            else:
                kappa_adj = max(0.05, kappa * 0.2)  # Conservative for unknown ordering
                result["assumptions"].append(f"Unknown ordering: κ set to {kappa_adj:.2f} (conservative)")
            
            Br_T = kappa_adj * result["Bs_T"]
            result["Br_T"] = float(Br_T)
            result["assumptions"].append(f"Br ≈ {kappa_adj:.2f} * Bs (remanence factor)")
        else:
            result["Br_T"] = None
        
        # Coercivity heuristics (very rough)
        # Without literature, we can only provide order-of-magnitude estimates
        if ordering == "FM":
            # Soft ferromagnet: low Hc
            result["Hc_kA_per_m"] = 1.0  # ~1 kA/m (soft magnet)
            result["assumptions"].append("Hc ~ 1 kA/m estimated for FM (soft magnet)")
        elif ordering == "FiM":
            # Ferrimagnets can have moderate Hc
            result["Hc_kA_per_m"] = 10.0  # ~10 kA/m
            result["assumptions"].append("Hc ~ 10 kA/m estimated for FiM")
        elif ordering in ["AFM", "WF"]:
            # AFM can have moderate coercivity (but low Br makes it weak for pull)
            result["Hc_kA_per_m"] = 5.0  # ~5 kA/m (reduced from 50; high Hc doesn't help if Br tiny)
            result["assumptions"].append("Hc ~ 5 kA/m estimated for AFM/WF (low Br dominates weakness)")
        else:
            result["Hc_kA_per_m"] = 5.0  # Conservative default
            result["assumptions"].append("Hc ~ 5 kA/m (conservative default)")
        
        result["caveats"].append("Coercivity (Hc) is highly microstructure-dependent; DFT cannot predict it accurately")
        
        # Estimate (BH)max
        if result.get("Br_T") is not None:
            # Theoretical maximum: (BH)max = Br² / (4μ0)
            # In practice, often 50-80% of theoretical for good hard magnets
            BHmax_ideal = (result["Br_T"] ** 2) / (4 * MU_0)
            BHmax_kJ_per_m3 = BHmax_ideal / 1000  # Convert J/m³ to kJ/m³
            result["BHmax_kJ_per_m3"] = float(BHmax_kJ_per_m3)
            result["assumptions"].append("(BH)max ≈ Br² / (4μ0) (theoretical ideal; actual values often 50-80% of this)")
        else:
            result["BHmax_kJ_per_m3"] = None
        
        # Curie temperature (not available from MP summary API typically)
        result["Tc_K"] = None
        result["caveats"].append("Curie temperature (Tc) not available from DFT; requires literature or experiments")
        
        # Overall caveat
        result["caveats"].append("DFT magnetization ≠ experimental remanence; microstructure (grain size, texture, porosity) dominates Hc and Br in real materials")
        
        return result
        
    except Exception as e:
        _log.error(f"Error estimating material properties: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def calculate_pull_force_cylinder(
    Br_T: float,
    diameter_mm: float = 10.0,
    length_mm: float = 10.0,
    air_gap_mm: float = 0.0,
    eta: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate pull force for a cylindrical magnet contacting a steel plate.
    
    Uses a simplified magnetic circuit model:
    - Gap field: Bg ≈ η * Br (at contact)
    - Pull force: F ≈ Bg² * A / (2μ0)
    
    For non-zero air gap, reduces Bg via 1D magnetic circuit approximation.
    
    Args:
        Br_T: Remanence (Tesla)
        diameter_mm: Cylinder diameter (mm). Default: 10 mm
        length_mm: Cylinder length/height (mm). Default: 10 mm
        air_gap_mm: Air gap between magnet and steel (mm). Default: 0 (contact)
        eta: Geometry factor for flux concentration (0.6-0.9). Default: 0.7
        
    Returns:
        Dictionary with:
        - Bg_T: Gap field (Tesla)
        - F_N: Pull force (Newtons)
        - geometry parameters
    """
    try:
        # Convert to SI units
        diameter_m = diameter_mm / 1000
        length_m = length_mm / 1000
        air_gap_m = air_gap_mm / 1000
        
        # Cross-sectional area
        A_m2 = np.pi * (diameter_m / 2) ** 2
        
        # Gap field
        if air_gap_m == 0:
            # Contact case
            Bg_T = eta * Br_T
        else:
            # With air gap: simplified 1D magnetic circuit
            # Bg ≈ Br / (1 + (μ_r * g) / L)
            # Assuming relative permeability μ_r ~ 1 for air, and magnet length L
            # Very simplified; in reality needs proper reluctance calculation
            reduction_factor = 1.0 / (1.0 + (air_gap_m / length_m))
            Bg_T = eta * Br_T * reduction_factor
        
        # Pull force: F = Bg² * A / (2μ0)
        F_N = (Bg_T ** 2) * A_m2 / (2 * MU_0)
        
        result = {
            "success": True,
            "geometry": {
                "shape": "cylinder",
                "diameter_mm": float(diameter_mm),
                "length_mm": float(length_mm),
                "air_gap_mm": float(air_gap_mm),
                "cross_sectional_area_mm2": float(A_m2 * 1e6)
            },
            "parameters": {
                "Br_T": float(Br_T),
                "eta": float(eta),
                "Bg_T": float(Bg_T)
            },
            "force": {
                "F_N": float(F_N),
                "F_kg_equivalent": float(F_N / 9.81),  # Weight equivalent
                "unit": "Newtons"
            }
        }
        
        if air_gap_mm > 0:
            result["note"] = "Air gap reduces pull force; simplified 1D model used"
        else:
            result["note"] = "Contact pull force (no air gap)"
        
        return result
        
    except Exception as e:
        _log.error(f"Error calculating pull force: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def find_same_phase_doped_entry(
    mpr,
    host_mp: Dict[str, Any],
    dopant: str,
    max_x: float = 0.1
) -> Optional[Dict[str, Any]]:
    """
    Try to find a doped entry that:
    (a) contains the dopant
    (b) matches host space group (same phase)
    (c) has small dopant fraction (<= max_x)
    
    This ensures we're comparing substitutional doping within the same crystal structure,
    not different compounds (e.g., α-Fe₂₋ₓAlₓO₃ vs spinel FeAl₂O₄).
    
    Args:
        mpr: MPRester client instance
        host_mp: Host material MP data from fetch_phase_and_mp_data
        dopant: Dopant element symbol
        max_x: Maximum dopant fraction (default: 0.1 for 10%)
        
    Returns:
        Dictionary from fetch_phase_and_mp_data for doped material, or None if not found
    """
    try:
        host_sg = host_mp.get("phase", {}).get("space_group")
        host_elems = list(host_mp.get("composition", {}).keys())
        
        if not host_sg or not host_elems:
            return None
        
        elems = host_elems + [dopant]
        
        docs = mpr.materials.summary.search(
            elements=elems,
            fields=[
                "material_id", "formula_pretty", "composition", "energy_above_hull",
                "is_stable", "symmetry", "is_magnetic", "ordering", "total_magnetization",
                "volume", "nsites"
            ],
        )
        
        if not docs:
            return None
        
        candidates = []
        for d in docs:
            # Check if same space group
            sg = None
            if hasattr(d, 'symmetry') and d.symmetry:
                sg = str(d.symmetry.symbol) if hasattr(d.symmetry, 'symbol') else None
            
            if sg != host_sg:
                continue
            
            # Check dopant fraction
            comp = d.composition.as_dict() if hasattr(d, 'composition') else {}
            tot = sum(comp.values()) or 1
            x_dop = comp.get(dopant, 0) / tot
            
            if 0 < x_dop <= max_x:
                candidates.append(d)
        
        if not candidates:
            return None
        
        # Prefer lowest energy above hull (most stable)
        candidates.sort(key=lambda z: getattr(z, "energy_above_hull", float("inf")))
        best = candidates[0]
        
        # Wrap with fetch_phase_and_mp_data to normalize shape
        best_formula = best.formula_pretty if hasattr(best, 'formula_pretty') else str(best.composition)
        return fetch_phase_and_mp_data(mpr, best_formula)
        
    except Exception as e:
        _log.warning(f"Error in find_same_phase_doped_entry: {e}")
        return None


def estimate_substitutional_doping_from_host(
    host_props: Dict[str, Any],
    host_mp: Dict[str, Any],
    dopant: str,
    x: float
) -> Dict[str, Any]:
    """
    Heuristic model for substitutional doping when no same-phase doped entry exists in MP.
    
    For α-Fe₂O₃ with Al³⁺ substitution (non-magnetic):
    - Ms ↓ roughly ∝ (1 - x) (dilution of magnetic Fe³⁺)
    - κ (remanence factor) very small for AFM/WF and may drop slightly with x
    - Hc tends to ↑ at small x (pinning/anisotropy), then saturate
    
    This avoids comparing to different phases (spinel, ilmenite) for a "doping" claim.
    
    Args:
        host_props: Host material properties from estimate_material_properties
        host_mp: Host MP data
        dopant: Dopant element symbol
        x: Doping fraction (e.g., 0.1 for 10%)
        
    Returns:
        Properties dict in same shape as estimate_material_properties()
    """
    try:
        out = {
            "success": True,
            "source": "heuristic_doping",
            "assumptions": [],
            "caveats": []
        }
        
        ordering = host_mp.get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        Ms0 = host_props.get("Ms_kA_per_m")
        Bs0 = host_props.get("Bs_T")
        Br0 = host_props.get("Br_T")
        Hc0 = host_props.get("Hc_kA_per_m", 1.0)
        
        # If host Ms is missing, apply conservative degradation
        if Ms0 is None or Bs0 is None:
            out["caveats"].append("Host Ms/Bs unknown; applied conservative Br degradation for non-magnetic dopant")
            Br_scale = max(0.0, 1.0 - x) * 0.2  # Harsh down-weight
            out["Br_T"] = (Br0 or 0.05) * Br_scale
            out["Ms_kA_per_m"] = None
            out["Bs_T"] = None
        else:
            # Ms decreases approximately with (1 - x) for non-magnetic substitution
            Ms1 = Ms0 * max(0.0, 1.0 - x)
            out["Ms_kA_per_m"] = float(Ms1)
            Bs1 = Ms1 * 1e3 * MU_0  # Ms in kA/m -> A/m, then Bs = μ0 * Ms
            out["Bs_T"] = float(Bs1)
            
            # For AFM/WF hematite, start κ very small and let it drop a little with x
            if ordering in ["AFM", "WF", "Unknown"]:
                kappa_base = 0.08  # << hard magnet; weak-FM-like remanence
                kappa1 = max(0.02, kappa_base * (1.0 - 0.5 * x))
                out["assumptions"].append(f"AFM/WF host: κ≈{kappa1:.3f} (very low remanence)")
            else:
                # For FM/FiM hosts (generic)
                kappa1 = 0.5 * (1.0 - 0.3 * x)
                out["assumptions"].append(f"κ≈{kappa1:.3f} for {ordering} host")
            
            out["Br_T"] = float(kappa1 * Bs1)
            out["assumptions"].append("Br ≈ κ·Bs; non-magnetic dopant dilutes Ms")
        
        # Coercivity trend: modest increase with small x (peak at a few at.%)
        # Al³⁺ in hematite creates pinning sites and anisotropy → Hc ↑
        Hc1 = Hc0 * (1.0 + 8.0 * min(x, 0.05))  # +40% at 5% Al, capped
        out["Hc_kA_per_m"] = float(Hc1)
        out["assumptions"].append("Hc ↑ for small x due to pinning/anisotropy (heuristic: +8×x capped at 5%)")
        
        # (BH)max ideal from Br
        if out.get("Br_T") is not None:
            out["BHmax_kJ_per_m3"] = float((out["Br_T"] ** 2) / (4 * MU_0) / 1000.0)
        else:
            out["BHmax_kJ_per_m3"] = None
        
        out["caveats"].extend([
            "Heuristic substitution model; microstructure dominates Br/Hc in oxides",
            "Do not compare to different phases (spinel/ilmenite) for a doping claim",
            f"Model assumes non-magnetic dopant ({dopant}) replaces magnetic cation"
        ])
        
        return out
        
    except Exception as e:
        _log.error(f"Error in estimate_substitutional_doping_from_host: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def assess_stronger_magnet(
    host_formula: str,
    dopant: str,
    doping_fraction: float,
    mpr,
    geometry: Optional[Dict[str, float]] = None,
    baseline_literature: Optional[Dict[str, Any]] = None,
    doped_literature: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete assessment: is doped material a stronger permanent magnet than host?
    
    "Stronger" means:
    1. Higher pull force F (primarily determined by Br)
    2. Sufficient coercivity Hc to retain magnetization under load
    3. Curie temperature above operating temperature
    
    Args:
        host_formula: Host material formula (e.g., "Fe2O3")
        dopant: Dopant element symbol (e.g., "Al")
        doping_fraction: Approximate doping fraction (e.g., 0.1 for 10%)
        mpr: MPRester client instance
        geometry: Optional dict with {"diameter_mm": 10, "length_mm": 10, "air_gap_mm": 0}
        baseline_literature: Optional literature properties for baseline
        doped_literature: Optional literature properties for doped material
        
    Returns:
        Complete assessment dictionary with:
        - phase_check: Phase and stability data for both materials
        - baseline: Estimated properties for host
        - doped: Estimated properties for doped material
        - pull_force: Force calculations for both
        - verdict: Is doped material stronger?
        - assumptions and caveats
    """
    try:
        result = {
            "success": True,
            "host_formula": host_formula,
            "dopant": dopant,
            "doping_fraction": float(doping_fraction)
        }
        
        # Default geometry
        if geometry is None:
            geometry = {"diameter_mm": 10.0, "length_mm": 10.0, "air_gap_mm": 0.0}
        
        result["geometry"] = geometry
        
        # 1. Fetch baseline (host) data
        _log.info(f"Fetching data for baseline: {host_formula}")
        baseline_mp = fetch_phase_and_mp_data(mpr, host_formula)
        
        if not baseline_mp.get("success"):
            return {
                "success": False,
                "error": f"Could not find baseline material {host_formula}: {baseline_mp.get('error')}"
            }
        
        result["baseline_phase_check"] = baseline_mp
        
        # 2. Search for doped material with SAME PHASE constraint
        # Don't accept different compounds (spinel, ilmenite); only same-phase substitution
        baseline_comp = Composition(host_formula)
        host_elements = [str(el) for el in baseline_comp.elements]
        
        _log.info(f"Searching for same-phase doped material: {host_elements} + {dopant}")
        
        # Try to find a same-phase doped entry (same space group, small dopant fraction)
        doped_mp = find_same_phase_doped_entry(
            mpr,
            baseline_mp,
            dopant,
            max_x=min(0.15, doping_fraction + 0.05)  # Allow slightly above requested fraction
        )
        
        # 3. Estimate baseline properties
        baseline_props = estimate_material_properties(
            baseline_mp,
            literature_hint=baseline_literature
        )
        result["baseline_properties"] = baseline_props
        
        # 4. Handle doped material: use MP entry if found, else use substitutional heuristic
        if not doped_mp:
            # No same-phase doped entry found; use substitutional heuristic
            _log.info(f"No same-phase doped entry found; using substitutional heuristic")
            
            doped_props = estimate_substitutional_doping_from_host(
                host_props=baseline_props,
                host_mp=baseline_mp,
                dopant=dopant,
                x=float(doping_fraction)
            )
            result["doped_properties"] = doped_props
            
            result["doped_phase_check"] = {
                "success": True,
                "note": f"No same-phase doped entry in MP; used substitutional heuristic on {host_formula}",
                "phase": baseline_mp.get("phase"),
                "composition_proxy": f"{host_formula} with {doping_fraction*100:.1f}% {dopant} substitution (conceptual)",
                "method": "heuristic"
            }
        else:
            # Found a same-phase doped entry
            _log.info(f"Found same-phase doped material: {doped_mp.get('formula')}")
            
            doped_props = estimate_material_properties(
                doped_mp,
                literature_hint=doped_literature
            )
            result["doped_properties"] = doped_props
            result["doped_phase_check"] = doped_mp
        
        # 5. Calculate pull forces
        if baseline_props.get("Br_T") is not None:
            baseline_force = calculate_pull_force_cylinder(
                Br_T=baseline_props["Br_T"],
                **geometry
            )
            result["baseline_pull_force"] = baseline_force
        else:
            result["baseline_pull_force"] = {
                "success": False,
                "error": "Br not available for baseline"
            }
        
        if doped_props.get("Br_T") is not None:
            doped_force = calculate_pull_force_cylinder(
                Br_T=doped_props["Br_T"],
                **geometry
            )
            result["doped_pull_force"] = doped_force
        else:
            result["doped_pull_force"] = {
                "success": False,
                "error": "Br not available for doped material"
            }
        
        # 6. Verdict: Is doped material stronger?
        verdict = assess_verdict(
            baseline_props=baseline_props,
            doped_props=doped_props,
            baseline_force=result.get("baseline_pull_force", {}),
            doped_force=result.get("doped_pull_force", {}),
            baseline_mp=baseline_mp,
            doped_mp=result.get("doped_phase_check", {})  # Use the phase_check we populated
        )
        
        result["verdict"] = verdict
        
        # 7. Collect all assumptions and caveats
        all_assumptions = []
        all_caveats = []
        
        for props in [baseline_props, doped_props]:
            all_assumptions.extend(props.get("assumptions", []))
            all_caveats.extend(props.get("caveats", []))
        
        # Deduplicate
        result["assumptions"] = list(set(all_assumptions))
        result["caveats"] = list(set(all_caveats))
        
        return result
        
    except Exception as e:
        _log.error(f"Error in assess_stronger_magnet: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def assess_verdict(
    baseline_props: Dict[str, Any],
    doped_props: Dict[str, Any],
    baseline_force: Dict[str, Any],
    doped_force: Dict[str, Any],
    baseline_mp: Dict[str, Any],
    doped_mp: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine if doped material is a "stronger" permanent magnet.
    
    "Stronger" = more pull force (↑Br → ↑F), NOT just ↑Hc.
    
    Criteria:
    1. Pull force (F) must increase by at least 5%
    2. Coercivity (Hc) must remain above a minimum threshold (e.g., 0.5 kA/m)
    3. Magnetic ordering should be FM or FiM (not AFM/WF)
    4. Must be same crystal phase (not a different compound)
    
    Args:
        baseline_props: Baseline material properties
        doped_props: Doped material properties
        baseline_force: Baseline pull force results
        doped_force: Doped pull force results
        baseline_mp: Baseline MP data (for ordering and phase)
        doped_mp: Doped MP data (for ordering and phase)
        
    Returns:
        Verdict dictionary with decision and reasoning
    """
    try:
        verdict = {
            "stronger": False,
            "reason": "",
            "confidence": "low",
            "details": {}
        }
        
        # Guard: Refuse phase changes for a "doping" claim
        # If doped_mp has "method: heuristic", it's conceptually the same phase
        if doped_mp and doped_mp.get("method") != "heuristic":
            baseline_sg = baseline_mp.get("phase", {}).get("space_group")
            doped_sg = doped_mp.get("phase", {}).get("space_group")
            
            if baseline_sg and doped_sg and baseline_sg != doped_sg:
                verdict["stronger"] = False
                verdict["reason"] = (
                    f"Doped entry is a different crystal phase (space group {doped_sg} vs {baseline_sg}); "
                    "not substitutional doping. Cannot claim 'stronger magnet by doping'."
                )
                verdict["confidence"] = "high"
                verdict["details"] = {
                    "baseline_space_group": baseline_sg,
                    "doped_space_group": doped_sg,
                    "phase_changed": True
                }
                return verdict
        
        # Check if we have necessary data
        if not baseline_force.get("success") or not doped_force.get("success"):
            verdict["reason"] = "Insufficient data to calculate pull forces"
            return verdict
        
        F_baseline = baseline_force["force"]["F_N"]
        F_doped = doped_force["force"]["F_N"]
        
        # Check pull force increase
        force_increase_pct = ((F_doped - F_baseline) / F_baseline) * 100 if F_baseline > 0 else 0
        verdict["details"]["force_change_percent"] = float(force_increase_pct)
        verdict["details"]["F_baseline_N"] = float(F_baseline)
        verdict["details"]["F_doped_N"] = float(F_doped)
        
        # Check coercivity
        Hc_baseline = baseline_props.get("Hc_kA_per_m", 0)
        Hc_doped = doped_props.get("Hc_kA_per_m", 0)
        Hc_min_threshold = 0.5  # kA/m (minimum for practical permanent magnet)
        
        verdict["details"]["Hc_baseline_kA_per_m"] = float(Hc_baseline)
        verdict["details"]["Hc_doped_kA_per_m"] = float(Hc_doped)
        
        # Check magnetic ordering
        baseline_ordering = baseline_mp.get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        doped_ordering = (doped_mp or {}).get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        
        verdict["details"]["baseline_ordering"] = baseline_ordering
        verdict["details"]["doped_ordering"] = doped_ordering
        
        # Decision logic
        reasons = []
        
        # Criterion 1: Force increase
        if force_increase_pct > 5:
            reasons.append(f"Pull force increased by {force_increase_pct:.1f}%")
            force_criterion = True
        else:
            reasons.append(f"Pull force changed by only {force_increase_pct:.1f}% (need >5% for 'stronger')")
            force_criterion = False
        
        # Criterion 2: Coercivity check
        if Hc_doped >= Hc_min_threshold:
            reasons.append(f"Coercivity {Hc_doped:.1f} kA/m is above minimum threshold")
            hc_criterion = True
        else:
            reasons.append(f"Coercivity {Hc_doped:.1f} kA/m is below minimum threshold ({Hc_min_threshold} kA/m)")
            hc_criterion = False
        
        # Criterion 3: Magnetic ordering
        if doped_ordering in ["FM", "FiM"]:
            reasons.append(f"Magnetic ordering ({doped_ordering}) suitable for permanent magnet")
            ordering_criterion = True
        else:
            reasons.append(f"Magnetic ordering ({doped_ordering}) not ideal for permanent magnet (prefer FM/FiM)")
            ordering_criterion = False
        
        # Final verdict
        if force_criterion and hc_criterion and ordering_criterion:
            verdict["stronger"] = True
            verdict["confidence"] = "medium"
            verdict["reason"] = "Doped material shows improved pull force with adequate coercivity and suitable magnetic ordering"
        elif force_criterion:
            verdict["stronger"] = False
            verdict["confidence"] = "medium"
            verdict["reason"] = "Pull force improved but coercivity or magnetic ordering concerns remain"
        else:
            verdict["stronger"] = False
            verdict["confidence"] = "medium"
            verdict["reason"] = "Pull force did not improve sufficiently"
        
        verdict["detailed_reasoning"] = reasons
        
        # Special case: AFM/WF materials
        if baseline_ordering in ["AFM", "WF"] or doped_ordering in ["AFM", "WF"]:
            verdict["confidence"] = "low"
            verdict["reason"] += " (Note: AFM/Weak-FM materials are generally poor permanent magnets)"
        
        return verdict
        
    except Exception as e:
        _log.error(f"Error in assess_verdict: {e}", exc_info=True)
        return {
            "stronger": False,
            "reason": f"Error in assessment: {str(e)}",
            "confidence": "N/A"
        }

