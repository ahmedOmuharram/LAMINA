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

# Import physical constants and converters from centralized location
from ..base.constants import (
    MU_0,
    BOHR_MAGNETON,
    AVOGADRO,
    MU_B_TO_EMU,
    MUB_PER_BOHR3_TO_KA_PER_M,
)
from ..base.converters import muB_per_bohr3_to_kA_per_m


def select_representative_entry(candidates, requested_formula: str):
    """
    Pick the 'reference structure' for the host.
    
    Rules:
    - Prefer lowest energy_above_hull among entries with the *common ground-state space group*
    - If multiple polymorphs exist, reject obvious outliers
    - For known materials (e.g., hematite Fe2O3), prefer their documented ground-state structure
    
    Special cases are fine. Example: hematite Fe2O3 is corundum-like (R-3c).
    
    Args:
        candidates: List of MP SummaryDoc objects
        requested_formula: The formula we're searching for (for validation)
        
    Returns:
        Best candidate doc, or None if no valid candidates
    """
    try:
        target_formula = requested_formula.replace(" ", "")
        
        # 1. Gather candidates with sane stoichiometry
        same_formula = []
        for d in candidates:
            fpretty = getattr(d, "formula_pretty", None)
            if fpretty and fpretty.replace(" ", "") == target_formula:
                same_formula.append(d)
        
        if not same_formula:
            # Fall back to "close enough" - try reduced composition matching
            try:
                req_comp = Composition(requested_formula).reduced_composition
                for doc in candidates:
                    doc_comp = Composition(doc.formula_pretty).reduced_composition
                    if doc_comp == req_comp:
                        same_formula.append(doc)
            except Exception:
                same_formula = candidates[:]  # ultimate fallback
        
        if not same_formula:
            return candidates[0] if candidates else None
        
        # 2. Heuristic space-group preference map for known hosts
        # This ensures we get the "real" ground state, not some DFT artifact
        preferred_sg = {
            # hematite (α-Fe2O3) corundum structure
            "Fe2O3": {"R-3c", "R-3̅c", "R-3c:", "R-3c:H"},  # variants of rhombohedral corundum
            # wurtzite ZnO
            "ZnO": {"P63mc", "P6_3mc"},
            # rocksalt NiO
            "NiO": {"Fm-3m", "Fm-3m:1"},
            # Add more hand-picked anchors as needed
        }.get(target_formula, None)
        
        # 3. If we have a preferred sg for this formula, filter to that first
        filtered = []
        if preferred_sg:
            for d in same_formula:
                if hasattr(d, "symmetry") and d.symmetry and getattr(d.symmetry, "symbol", None):
                    sg_symbol = str(d.symmetry.symbol)
                    # Check if any preferred variant matches
                    if sg_symbol in preferred_sg or any(sg in sg_symbol for sg in preferred_sg):
                        filtered.append(d)
            if filtered:
                same_formula = filtered
                _log.info(f"Filtered {target_formula} to preferred space group(s): {preferred_sg}")
        
        # 4. Return the one with lowest energy_above_hull
        def hull(e):
            val = getattr(e, "energy_above_hull", None)
            return float(val) if val is not None else float("inf")
        
        best = min(same_formula, key=hull)
        return best
        
    except Exception as e:
        _log.warning(f"Error in select_representative_entry: {e}")
        # Fallback to first candidate
        return candidates[0] if candidates else None


def fetch_phase_and_mp_data(
    mpr,
    formula: str
) -> Dict[str, Any]:
    """
    Fetch phase, structure, stability, and magnetic data from Materials Project.
    
    Uses smart selection to pick the physically reasonable ground state, not just
    the first hit or the most magnetic state.
    
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
        
        # Use smart selector to pick best representative entry
        doc = select_representative_entry(docs, formula)
        
        if not doc:
            return {
                "success": False,
                "error": f"No valid stoichiometric match for {formula}"
            }
        
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
    (b) matches host space group OR crystal system (same phase)
    (c) has small dopant fraction (<= max_x)
    (d) has no extra cations beyond {host elements} ∪ {dopant}
    
    This ensures we're comparing substitutional doping within the same crystal structure,
    not different compounds (e.g., α-Fe₂₋ₓCoₓO₃ vs SrPrFeCoO₆).
    
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
        host_cs = host_mp.get("phase", {}).get("crystal_system")
        host_elems = list(host_mp.get("composition", {}).keys())
        
        if not host_elems:
            return None
        
        # Chemistry constraint
        host_elem_set = set(host_elems)
        allowed_elements = host_elem_set | {dopant}
        
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
            # Chemistry filter: no extra cations
            comp = d.composition.as_dict() if hasattr(d, 'composition') else {}
            doc_elements = set(comp.keys())
            
            if not doc_elements.issubset(allowed_elements):
                continue
            
            # Calculate dopant fraction on CATION sublattice only (ignore O, F, etc.)
            cation_elements = [el for el in comp if el not in ["O", "F", "N", "S", "Cl"]]
            total_cations = sum(comp[el] for el in cation_elements) or 1
            x_dop = comp.get(dopant, 0) / total_cations
            
            if not (0 < x_dop <= max_x):
                continue
            
            # Check stoichiometry similarity: O:cation ratio should be close to host
            # This rejects ferrites (O:cat ≈ 1.33) when host is hematite (O:cat = 1.5)
            host_comp_dict = host_mp.get("composition", {})
            if host_comp_dict:
                host_cations = [el for el in host_comp_dict if el not in ["O", "F", "N", "S", "Cl"]]
                host_total_cations = sum(host_comp_dict[el] for el in host_cations)
                host_O = host_comp_dict.get("O", 0)
                host_ratio = host_O / host_total_cations if host_total_cations > 0 else 0
                
                doc_cations_total = sum(comp[el] for el in cation_elements)
                doc_O = comp.get("O", 0)
                doc_ratio = doc_O / doc_cations_total if doc_cations_total > 0 else 0
                
                # Allow 15% deviation in O:cation ratio
                if host_ratio > 0 and abs(doc_ratio - host_ratio) / host_ratio > 0.15:
                    continue
            
            # Check host identity: remove dopant and check if formula resembles host
            # E.g., Fe2CoO4 - Co = Fe2O4 → Fe1O2, but host Fe2O3 is Fe2O3
            hypothetical_comp = comp.copy()
            if dopant in hypothetical_comp:
                hypothetical_comp[dopant] = 0
            
            # Normalize and compare to host
            try:
                from pymatgen.core import Composition as PmgComp
                hyp_normalized = PmgComp(hypothetical_comp).reduced_composition
                host_normalized = PmgComp(host_comp_dict).reduced_composition
                
                # Check if removing dopant gives back something close to host
                # Allow slight variation but not totally different formula
                hyp_elements = set(hyp_normalized.elements)
                host_elements_set = set(host_normalized.elements)
                
                if hyp_elements != host_elements_set:
                    continue
                
                # Check stoichiometry roughly matches
                mismatch = False
                for el in host_elements_set:
                    host_frac = host_normalized.get_atomic_fraction(el)
                    hyp_frac = hyp_normalized.get_atomic_fraction(el)
                    if abs(hyp_frac - host_frac) > 0.2:  # 20% tolerance
                        mismatch = True
                        break
                
                if mismatch:
                    continue
            except Exception:
                # If comparison fails, be conservative and skip
                continue
            
            # Structure filter: prefer same space group, fallback to same crystal system
            sg = None
            cs = None
            if hasattr(d, 'symmetry') and d.symmetry:
                sg = str(d.symmetry.symbol) if hasattr(d.symmetry, 'symbol') else None
                cs = str(d.symmetry.crystal_system) if hasattr(d.symmetry, 'crystal_system') else None
            
            # Accept if same space group OR same crystal system (if space group not available)
            if host_sg and sg == host_sg:
                candidates.append(d)
            elif host_cs and cs == host_cs:
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


def get_detailed_saturation_magnetization(
    mpr,
    formula: str
) -> Dict[str, Any]:
    """
    Get comprehensive saturation magnetization data for a material.

    Args:
        mpr: MPRester client instance
        formula: Chemical formula (ideally reduced formula)

    Returns:
        Dictionary with Ms in multiple units, magnetic ordering, and site info
    """
    try:
        mp_data = fetch_phase_and_mp_data(mpr, formula)

        if not mp_data.get("success"):
            return mp_data

        result = {
            "success": True,
            "formula": mp_data.get("formula"),
            "material_id": mp_data.get("material_id")
        }

        # Pull raw quantities
        magnetization_muB = mp_data.get("total_magnetization_muB")      # μB / cell
        volume_A3 = mp_data.get("volume_A3")                             # Å^3 / cell
        nsites = mp_data.get("nsites")                                   # atoms / cell

        # Compute saturation magnetization in SI via Bs_T
        if magnetization_muB is not None and volume_A3 is not None:
            Bs_T = estimate_saturation_magnetization_T(
                magnetization_muB,
                volume_A3
            )  # should return B_s in Tesla = μ0 * M_s

            Ms_A_per_m = Bs_T / MU_0 if Bs_T is not None else None

            sat_mag_dict = {
                "Bs_T": float(Bs_T) if Bs_T is not None else None,
                "Ms_A_per_m": float(Ms_A_per_m) if Ms_A_per_m is not None else None,
                "Ms_kA_per_m": float(Ms_A_per_m / 1000.0) if Ms_A_per_m is not None else None,
            }

            # Now compute density and Ms in emu/g using correct cell mass
            try:
                # Use the reported formula from mp_data if available
                formula_for_mass = mp_data.get("formula", formula)
                comp = Composition(formula_for_mass)

                molar_mass = comp.weight  # g / mol for ONE reduced formula unit
                atoms_per_fu = comp.num_atoms  # atoms in reduced formula unit (e.g. 5 for Fe2O3)

                if nsites is not None and atoms_per_fu > 0:
                    # Number of formula units in this calculation cell
                    Z = float(nsites) / float(atoms_per_fu)

                    # mass of full cell in grams
                    mass_per_fu_g = molar_mass / AVOGADRO
                    mass_cell_g = mass_per_fu_g * Z  # g / cell

                    # volume of cell in cm^3
                    volume_cm3 = volume_A3 * 1e-24  # (1 Å^3 = 1e-24 cm^3)

                    # density = mass / volume
                    density = mass_cell_g / volume_cm3 if volume_cm3 > 0 else None

                    # Ms in emu/g:
                    # total magnetic moment per cell (μB) -> emu via μB * 9.274e-21
                    # divide by mass of that same cell in g
                    if magnetization_muB is not None and mass_cell_g > 0:
                        Ms_emu_per_g = (
                            abs(magnetization_muB) * MU_B_TO_EMU / mass_cell_g
                        )
                    else:
                        Ms_emu_per_g = None

                    sat_mag_dict["Ms_emu_per_g"] = float(Ms_emu_per_g) if Ms_emu_per_g is not None else None
                    result["density_g_per_cm3"] = float(density) if density is not None else None
                else:
                    # Couldn't compute Z -> can't get density or Ms_emu_per_g
                    sat_mag_dict["Ms_emu_per_g"] = None
                    result["density_g_per_cm3"] = None

            except Exception as e:
                _log.warning(f"Could not calculate density/Ms_emu_per_g: {e}")
                sat_mag_dict["Ms_emu_per_g"] = None
                result["density_g_per_cm3"] = None

            result["saturation_magnetization"] = sat_mag_dict

        else:
            # Missing required magnetic data
            result["saturation_magnetization"] = None
            result["density_g_per_cm3"] = None

        # Magnetic ordering info
        result["magnetic_ordering"] = mp_data.get("magnetic_ordering")

        # Magnetic sites
        result["magnetic_sites"] = {
            "num_sites": mp_data.get("num_magnetic_sites"),
            "species": mp_data.get("magnetic_species")
        }

        # Structural info
        result["structure"] = mp_data.get("phase")

        # Raw magnetization for reference
        result["total_magnetization_muB"] = magnetization_muB
        result["magnetization_per_fu_muB"] = mp_data.get("magnetization_per_fu_muB")

        return result

    except Exception as e:
        _log.error(f"Error getting detailed saturation magnetization: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def analyze_doping_effect_on_ms(
    host_formula: str,
    dopant: str,
    doping_fraction: float,
    mpr
) -> Dict[str, Any]:
    """
    Analyze the effect of doping on saturation magnetization.

    This function:
    1. Gets host Ms from MP data
    2. Attempts to find a structurally/symmetry-similar doped version
       *with no extra cations beyond the dopant*
    3. Falls back to a theoretical estimate if no clean doped analog is found
    4. Computes % change
    """
    try:
        result = {
            "success": True,
            "host_formula": host_formula,
            "dopant": dopant,
            "doping_fraction": float(doping_fraction)
        }

        # 1. Get host data
        _log.info(f"Analyzing Ms for host: {host_formula}")
        host_mp = fetch_phase_and_mp_data(mpr, host_formula)

        if not host_mp.get("success"):
            return {
                "success": False,
                "error": f"Could not find host material {host_formula}: {host_mp.get('error')}"
            }

        result["host_data"] = host_mp

        # Compute host Ms_kA_per_m and Bs_T
        host_mag_muB = host_mp.get("total_magnetization_muB")
        host_volume = host_mp.get("volume_A3")

        if host_mag_muB is not None and host_volume is not None:
            host_Bs = estimate_saturation_magnetization_T(host_mag_muB, host_volume)
            host_Ms = host_Bs / MU_0 if host_Bs is not None else None
            result["host_Ms_kA_per_m"] = float(host_Ms / 1000.0) if host_Ms is not None else None
            result["host_Bs_T"] = float(host_Bs) if host_Bs is not None else None
        else:
            result["host_Ms_kA_per_m"] = None
            result["host_Bs_T"] = None
            result["warning"] = "Host magnetization data not available"

        # 2. Try to find a doped entry
        host_comp = Composition(host_formula)
        host_elements = {str(el) for el in host_comp.elements}
        allowed_elements = set(host_elements)
        allowed_elements.add(dopant)

        _log.info(f"Searching for doped material similar to {host_formula} with dopant {dopant}")

        doped_mp = find_same_phase_doped_entry(
            mpr,
            host_mp,
            dopant,
            max_x=min(0.2, doping_fraction + 0.1)
        )

        # Sanity check doped candidate
        if doped_mp:
            doped_formula = doped_mp.get("formula")
            try:
                doped_elements = {str(el) for el in Composition(doped_formula).elements}
            except Exception:
                doped_elements = set()

            # Reject doped structures that introduce *new* cations beyond dopant
            # e.g. SrPrFeCoO6 will be rejected if host is Fe2O3 and dopant is Co
            extra_elements = doped_elements - allowed_elements
            if extra_elements:
                _log.info(
                    f"Discarding doped candidate {doped_formula} due to extra elements {extra_elements}"
                )
                doped_mp = None

        if doped_mp:
            _log.info(f"Accepted doped material: {doped_mp.get('formula')}")
            result["doped_data"] = doped_mp
            result["doped_formula"] = doped_mp.get("formula")
            result["used_heuristic"] = False
            
            # Calculate actual dopant fraction on cation sublattice
            doped_comp_dict = doped_mp.get("composition", {})
            if doped_comp_dict:
                cation_elements = [el for el in doped_comp_dict if el not in ["O", "F", "N", "S", "Cl"]]
                total_cations = sum(doped_comp_dict[el] for el in cation_elements)
                actual_x_dop = doped_comp_dict.get(dopant, 0) / total_cations if total_cations > 0 else 0
                result["effective_dopant_fraction_on_cation_lattice"] = float(actual_x_dop)
            
            # Check if this is really the same phase or a transformation
            host_formula_normalized = Composition(host_formula).reduced_formula
            doped_formula_test = doped_mp.get("formula")
            
            # Simple phase change detection: check space group and ordering
            host_sg = host_mp.get("phase", {}).get("space_group")
            doped_sg = doped_mp.get("phase", {}).get("space_group")
            host_ordering = host_mp.get("magnetic_ordering", {}).get("ordering_type")
            doped_ordering = doped_mp.get("magnetic_ordering", {}).get("ordering_type")
            
            phase_changed = False
            if host_sg and doped_sg and host_sg != doped_sg:
                phase_changed = True
                result["phase_change_note"] = f"Space group changed: {host_sg} → {doped_sg}"
            elif host_ordering and doped_ordering and host_ordering != doped_ordering:
                # Significant ordering change (e.g., AFM → FM) might indicate phase change
                if {host_ordering, doped_ordering} & {"FM", "AFM"}:
                    phase_changed = True
                    result["phase_change_note"] = f"Magnetic ordering changed: {host_ordering} → {doped_ordering}"
            
            result["phase_changed"] = phase_changed

            doped_mag_muB = doped_mp.get("total_magnetization_muB")
            doped_volume = doped_mp.get("volume_A3")

            if doped_mag_muB is not None and doped_volume is not None:
                doped_Bs = estimate_saturation_magnetization_T(doped_mag_muB, doped_volume)
                doped_Ms = doped_Bs / MU_0 if doped_Bs is not None else None
                result["doped_Ms_kA_per_m"] = float(doped_Ms / 1000.0) if doped_Ms is not None else None
                result["doped_Bs_T"] = float(doped_Bs) if doped_Bs is not None else None
            else:
                result["doped_Ms_kA_per_m"] = None
                result["doped_Bs_T"] = None

        else:
            # 3. No valid doped analog -> fall back to theory
            _log.info("No acceptable doped material found in MP; using theoretical estimation")
            result["doped_data"] = None
            result["estimation_method"] = "theoretical"
            result["used_heuristic"] = True
            result["phase_changed"] = False  # Heuristic assumes same phase
            result["effective_dopant_fraction_on_cation_lattice"] = float(doping_fraction)

            doped_est = estimate_doped_ms_from_magnetic_moments(
                host_formula=host_formula,
                host_mp=host_mp,
                dopant=dopant,
                doping_fraction=doping_fraction,
                host_Ms_kA_per_m=result.get("host_Ms_kA_per_m")
            )

            result["doped_Ms_kA_per_m"] = doped_est.get("Ms_kA_per_m")
            result["doped_Bs_T"] = doped_est.get("Bs_T")
            result["estimation_notes"] = doped_est.get("notes", [])

        # 4. % change calculation (using magnitudes to avoid sign flip issues)
        host_Ms_val = result.get("host_Ms_kA_per_m")
        doped_Ms_val = result.get("doped_Ms_kA_per_m")

        if (host_Ms_val is not None) and (doped_Ms_val is not None):
            # Compare magnitudes to handle ferrimagnets and sign flips robustly
            host_abs = abs(host_Ms_val)
            doped_abs = abs(doped_Ms_val)
            
            if host_abs > 0:
                change_pct = ((doped_abs - host_abs) / host_abs) * 100.0
                result["Ms_change_percent"] = float(change_pct)

                if change_pct > 5:
                    result["verdict"] = "increases"
                elif change_pct < -5:
                    result["verdict"] = "decreases"
                else:
                    result["verdict"] = "maintains (negligible change)"
            else:
                # AFM / nearly compensated case
                result["Ms_change_percent"] = None
                result["verdict"] = "unclear (host Ms near zero)"
        else:
            result["Ms_change_percent"] = None
            result["verdict"] = "insufficient data"

        # 5. Physical interpretation
        result["analysis"] = analyze_magnetic_moment_contribution(
            host_formula=host_formula,
            host_mp=host_mp,
            dopant=dopant,
            doping_fraction=doping_fraction,
            Ms_change_percent=result.get("Ms_change_percent")
        )

        return result

    except Exception as e:
        _log.error(f"Error analyzing doping effect on Ms: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def estimate_doped_ms_from_magnetic_moments(
    host_formula: str,
    host_mp: Dict[str, Any],
    dopant: str,
    doping_fraction: float,
    host_Ms_kA_per_m: Optional[float]
) -> Dict[str, Any]:
    """
    Estimate doped Ms using a simple magnetic moment model.

    Logic:
    - Identify the magnetic ion in the host (e.g. Fe³⁺ in Fe2O3)
    - Assume dopant substitutes on that site
    - Compare dopant's spin moment vs host ion's spin moment
    - Scale effect ~ linearly with doping_fraction
    - Reduce the effect if the host is AFM (because sublattices mostly cancel)

    Returns:
        {
          "Ms_kA_per_m": float or None,
          "Bs_T": float or None,
          "notes": [...]
        }
    """
    try:
        result = {"notes": []}

        host_ordering = host_mp.get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        magnetic_species = host_mp.get("magnetic_species", []) or []

        # Typical spin-only moments in μB for common oxidation states.
        # NOTE: these are heuristic; high-spin assumed where ambiguous.
        magnetic_moments = {
            "Ti": {"3+": 1.0, "4+": 0.0},
            "V":  {"3+": 2.0, "4+": 1.0, "5+": 0.0},
            "Cr": {"3+": 3.0, "4+": 2.0},
            "Mn": {"2+": 5.0, "3+": 4.0, "4+": 3.0},
            "Fe": {"2+": 4.0, "3+": 5.0},
            "Co": {"2+": 3.0, "3+": 4.0},
            "Ni": {"2+": 2.0, "3+": 1.0},
            "Cu": {"2+": 1.0},
            "Nd": {"3+": 3.6},
            "Sm": {"3+": 1.5},
            "Gd": {"3+": 7.9},
            "Dy": {"3+": 10.6},
        }

        host_comp = Composition(host_formula)

        # Get oxidation state guess and convert keys to strings like "Fe": +3
        try:
            oxi_states_list = host_comp.oxi_state_guesses()
            if oxi_states_list:
                raw_oxi = oxi_states_list[0]  # dict(Element -> oxi int)
                host_oxi = {str(el): raw_oxi[el] for el in raw_oxi}
            else:
                host_oxi = {}
        except Exception:
            host_oxi = {}

        replaced_ion = None
        replaced_moment = 0.0
        dopant_moment = 0.0

        # Pick the first magnetic species that we can assign a spin moment to
        for species in magnetic_species:
            if species in host_oxi:
                oxi_int = int(round(host_oxi[species]))
                oxi_key = f"{oxi_int}+"
                if species in magnetic_moments and oxi_key in magnetic_moments[species]:
                    replaced_ion = species
                    replaced_moment = magnetic_moments[species][oxi_key]
                    break

        # Estimate dopant moment in the same oxidation state as the replaced ion
        if replaced_ion and replaced_ion in host_oxi:
            oxi_int = int(round(host_oxi[replaced_ion]))
            oxi_key = f"{oxi_int}+"
            if dopant in magnetic_moments and oxi_key in magnetic_moments[dopant]:
                dopant_moment = magnetic_moments[dopant][oxi_key]

        if host_Ms_kA_per_m is not None and replaced_ion:
            base = replaced_moment if abs(replaced_moment) > 1e-6 else 1.0
            moment_diff = dopant_moment - replaced_moment

            # naive linear scaling with doping fraction
            Ms_change_factor = 1.0 + (moment_diff / base) * doping_fraction

            # Damp for AFM systems OR weak moment systems (nearly compensated)
            # This catches real AFM, FiM with tiny net moment, weakly canted systems
            host_abs = abs(host_Ms_kA_per_m)
            is_weak_moment = host_abs < 10.0  # 10 kA/m is tiny compared to real ferromagnets
            
            if host_ordering == "AFM" or is_weak_moment:
                Ms_change_factor = 1.0 + (moment_diff / base) * doping_fraction * 0.5
                if host_ordering == "AFM":
                    result["notes"].append("AFM host: damping factor 0.5 applied to Ms change")
                if is_weak_moment:
                    result["notes"].append(f"Nearly compensated host (Ms={host_abs:.1f} kA/m): damping factor 0.5 applied")

            doped_Ms = host_Ms_kA_per_m * Ms_change_factor  # kA/m

            result["Ms_kA_per_m"] = float(doped_Ms)
            # Convert kA/m -> A/m -> Tesla:  (kA/m * 1000) * mu0
            result["Bs_T"] = float(doped_Ms * 1000.0 * MU_0)

            result["notes"].append(f"Replaced ion: {replaced_ion}")
            result["notes"].append(f"Estimated {replaced_ion} moment: {replaced_moment} μB")
            result["notes"].append(f"Estimated {dopant} moment: {dopant_moment} μB")
            result["notes"].append(f"Moment difference: {moment_diff:.2f} μB")
            result["notes"].append(f"Theoretical Ms change factor: {Ms_change_factor:.3f}")
        else:
            # Fallback heuristic if we couldn't map oxidation states well
            if host_Ms_kA_per_m is not None:
                if dopant in ["Co", "Fe", "Nd", "Gd", "Dy"]:
                    guessed = host_Ms_kA_per_m * 1.05
                    note = f"Assumed slight increase (dopant {dopant} is strongly magnetic)"
                else:
                    guessed = host_Ms_kA_per_m * 0.95
                    note = f"Assumed slight decrease (dopant {dopant} is weaker / non-magnetic)"

                result["Ms_kA_per_m"] = float(guessed)
                result["Bs_T"] = float(guessed * 1000.0 * MU_0)
                result["notes"].append(note)
            else:
                result["Ms_kA_per_m"] = None
                result["Bs_T"] = None

        result["notes"].append("Simple spin-moment substitution model; ignores local distortion, canting, clustering, etc.")

        return result

    except Exception as e:
        _log.error(f"Error estimating doped Ms: {e}", exc_info=True)
        return {
            "Ms_kA_per_m": None,
            "Bs_T": None,
            "notes": [f"Error: {str(e)}"]
        }


def analyze_magnetic_moment_contribution(
    host_formula: str,
    host_mp: Dict[str, Any],
    dopant: str,
    doping_fraction: float,
    Ms_change_percent: Optional[float]
) -> Dict[str, Any]:
    """
    Provide physical analysis of why doping affects Ms.
    
    Returns:
        Dictionary with reasoning about magnetic moment contributions
    """
    try:
        analysis = {
            "physical_mechanism": [],
            "expected_behavior": "",
            "confidence": "medium"
        }
        
        host_ordering = host_mp.get("magnetic_ordering", {}).get("ordering_type", "Unknown")
        
        # Common strongly magnetic elements
        strong_magnetic = ["Fe", "Co", "Ni", "Gd", "Nd", "Dy", "Sm"]
        weak_magnetic = ["Mn", "Cr", "Cu"]
        non_magnetic = ["Al", "Zn", "Mg", "Ca", "Ti", "Zr"]
        
        if dopant in strong_magnetic:
            analysis["dopant_type"] = "strongly magnetic"
            if Ms_change_percent and Ms_change_percent > 0:
                analysis["physical_mechanism"].append(
                    f"{dopant} has significant magnetic moment and contributes to net magnetization"
                )
            else:
                analysis["physical_mechanism"].append(
                    f"{dopant} is magnetic but may occupy unfavorable sites or disrupt magnetic ordering"
                )
        elif dopant in weak_magnetic:
            analysis["dopant_type"] = "weakly magnetic"
            analysis["physical_mechanism"].append(
                f"{dopant} has some magnetic moment but weaker than typical ferromagnetic ions"
            )
        elif dopant in non_magnetic:
            analysis["dopant_type"] = "non-magnetic"
            analysis["physical_mechanism"].append(
                f"{dopant} is non-magnetic and dilutes the magnetic sublattice"
            )
        else:
            analysis["dopant_type"] = "unknown magnetic character"
        
        # Ordering-specific analysis
        if host_ordering == "FM":
            analysis["physical_mechanism"].append(
                "Host is ferromagnetic: all moments align. Doping can enhance or reduce Ms depending on dopant moment."
            )
        elif host_ordering == "FiM":
            analysis["physical_mechanism"].append(
                "Host is ferrimagnetic: sublattices with antiparallel moments. Dopant substitution affects sublattice imbalance."
            )
        elif host_ordering == "AFM":
            analysis["physical_mechanism"].append(
                "Host is antiferromagnetic: sublattices cancel almost completely. Small net Ms from imperfect cancellation or spin canting."
            )
            analysis["physical_mechanism"].append(
                "Doping can enhance Ms if it breaks symmetry or introduces spin canting."
            )
        
        # Expected behavior summary
        if Ms_change_percent is not None:
            if Ms_change_percent > 10:
                analysis["expected_behavior"] = "Significant increase in Ms"
            elif Ms_change_percent > 0:
                analysis["expected_behavior"] = "Modest increase in Ms"
            elif Ms_change_percent > -10:
                analysis["expected_behavior"] = "Small decrease in Ms"
            else:
                analysis["expected_behavior"] = "Significant decrease in Ms"
        else:
            analysis["expected_behavior"] = "Insufficient data to determine"
        
        return analysis
        
    except Exception as e:
        _log.error(f"Error in magnetic moment analysis: {e}", exc_info=True)
        return {
            "physical_mechanism": [],
            "expected_behavior": "Error in analysis",
            "confidence": "none"
        }


def compare_multiple_dopants_ms(
    host_formula: str,
    dopants: list,
    doping_fraction: float,
    mpr
) -> Dict[str, Any]:
    """
    Compare multiple dopants to find which causes least degradation in Ms.
    
    Args:
        host_formula: Host material formula
        dopants: List of dopant elements
        doping_fraction: Doping fraction to test
        mpr: MPRester client instance
        
    Returns:
        Comparison results with ranking of dopants
    """
    try:
        result = {
            "success": True,
            "host_formula": host_formula,
            "doping_fraction": float(doping_fraction),
            "dopants_tested": dopants
        }
        
        # Get host Ms first
        host_mp = fetch_phase_and_mp_data(mpr, host_formula)
        if not host_mp.get("success"):
            return {
                "success": False,
                "error": f"Could not find host material: {host_mp.get('error')}"
            }
        
        host_mag_muB = host_mp.get("total_magnetization_muB")
        host_volume = host_mp.get("volume_A3")
        
        if host_mag_muB is not None and host_volume is not None:
            host_Bs = estimate_saturation_magnetization_T(host_mag_muB, host_volume)
            host_Ms = host_Bs / MU_0 if host_Bs is not None else None
            result["host_Ms_kA_per_m"] = float(host_Ms / 1000.0) if host_Ms is not None else None
        else:
            result["host_Ms_kA_per_m"] = None
        
        # Analyze each dopant
        dopant_results = []
        
        for dopant in dopants:
            _log.info(f"Analyzing dopant: {dopant}")
            
            dopant_analysis = analyze_doping_effect_on_ms(
                host_formula=host_formula,
                dopant=dopant,
                doping_fraction=doping_fraction,
                mpr=mpr
            )
            
            if dopant_analysis.get("success"):
                dopant_result = {
                    "dopant": dopant,
                    "doped_Ms_kA_per_m": dopant_analysis.get("doped_Ms_kA_per_m"),
                    "Ms_change_percent": dopant_analysis.get("Ms_change_percent"),
                    "verdict": dopant_analysis.get("verdict"),
                    "analysis": dopant_analysis.get("analysis", {}),
                    "phase_changed": dopant_analysis.get("phase_changed", False),
                    "used_heuristic": dopant_analysis.get("used_heuristic", False),
                    "effective_dopant_fraction": dopant_analysis.get("effective_dopant_fraction_on_cation_lattice", doping_fraction)
                }
                
                # Add warning notes
                notes = []
                if dopant_result["phase_changed"]:
                    phase_note = dopant_analysis.get("phase_change_note", "Phase transformation detected")
                    notes.append(f"WARNING: {phase_note} - not substitutional doping")
                if dopant_result["used_heuristic"]:
                    notes.append("Used theoretical estimation (no MP data for doped phase)")
                
                if notes:
                    dopant_result["notes"] = notes
                
                dopant_results.append(dopant_result)
            else:
                dopant_results.append({
                    "dopant": dopant,
                    "error": dopant_analysis.get("error")
                })
        
        # Sort by Ms change (least degradation = highest change %)
        # Filter out results with errors
        valid_results = [r for r in dopant_results if r.get("Ms_change_percent") is not None]
        
        if valid_results:
            # Sort by Ms change (descending)
            valid_results.sort(key=lambda x: x["Ms_change_percent"], reverse=True)
            result["dopant_comparison"] = valid_results
            
            # Separate same-phase from phase-changed dopants
            same_phase = [d for d in valid_results if not d.get("phase_changed", False)]
            all_dopants = valid_results
            
            # 1. Rank dopants that keep the same phase
            if same_phase:
                best_same_phase = max(same_phase, key=lambda x: x["Ms_change_percent"])
                worst_same_phase = min(same_phase, key=lambda x: x["Ms_change_percent"])
                
                result["best_dopant_same_phase"] = {
                    "element": best_same_phase["dopant"],
                    "Ms_change_percent": best_same_phase["Ms_change_percent"],
                    "reason": "Highest Ms with no phase change"
                }
                result["worst_dopant_same_phase"] = {
                    "element": worst_same_phase["dopant"],
                    "Ms_change_percent": worst_same_phase["Ms_change_percent"],
                    "reason": "Largest Ms drop with no phase change"
                }
            else:
                result["best_dopant_same_phase"] = None
                result["worst_dopant_same_phase"] = None
            
            # 2. Overall best (any phase) - but label clearly
            best_any = max(all_dopants, key=lambda x: x["Ms_change_percent"])
            result["best_dopant_any_phase"] = {
                "element": best_any["dopant"],
                "Ms_change_percent": best_any["Ms_change_percent"],
                "reason": (
                    "Most enhancement, BUT PHASE CHANGED"
                    if best_any.get("phase_changed", False)
                    else "Most enhancement (same phase)"
                )
            }
            
            # Keep old 'best_dopant' for backwards compatibility, but use same_phase if available
            if same_phase:
                result["best_dopant"] = result["best_dopant_same_phase"]
                result["worst_dopant"] = result["worst_dopant_same_phase"]
            else:
                result["best_dopant"] = result["best_dopant_any_phase"]
                worst_any = min(all_dopants, key=lambda x: x["Ms_change_percent"])
                result["worst_dopant"] = {
                    "element": worst_any["dopant"],
                    "Ms_change_percent": worst_any["Ms_change_percent"],
                    "reason": "Lowest Ms (but phase may have changed)"
                }
            
            # 3. Human-readable summary
            if same_phase:
                best_elem = result["best_dopant_same_phase"]["element"]
                best_pct = result["best_dopant_same_phase"]["Ms_change_percent"]
                result["summary"] = (
                    f"Among same-phase dopants at ~{doping_fraction*100:.0f}% "
                    f"on {host_formula}, {best_elem} causes the smallest "
                    f"Ms change ({best_pct:+.2f}%)."
                )
            else:
                result["summary"] = (
                    "All dopants that gave large Ms changes also altered the phase; "
                    "no same-phase substitutional dopant yielded a strong Ms increase."
                )
        else:
            result["dopant_comparison"] = dopant_results
            result["error"] = "Could not obtain valid Ms data for any dopant"
        
        return result
        
    except Exception as e:
        _log.error(f"Error comparing dopants for Ms: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def assess_doping_effect_on_saturation_magnetization(
    mpr,
    host_formula: str,
    doped_entry_material_id: str,
    host_phase_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare Ms(host) vs Ms(doped_candidate) *in consistent units*.

    REQUIREMENTS:
    - Same space group as host OR we downrate the claim
    - Use MP's normalized magnetization per volume for cross-phase comparisons

    Args:
        mpr: MPRester client instance
        host_formula: Host material formula (e.g., "Fe2O3")
        doped_entry_material_id: Material ID of doped candidate (e.g., "mp-1234")
        host_phase_data: Optional pre-fetched host data from fetch_phase_and_mp_data

    Returns:
        Dictionary with:
        - success: bool
        - Ms_host_kA_per_m: Host magnetization in kA/m
        - Ms_doped_kA_per_m: Doped magnetization in kA/m
        - delta_kA_per_m: Absolute change
        - percent_change: Percentage change (or None if baseline ~0)
        - same_space_group: bool
        - caution: Warning message if phases don't match
    """
    try:
        # 1. get host data if not provided
        if host_phase_data is None:
            host_phase_data = fetch_phase_and_mp_data(mpr, host_formula)
        if not host_phase_data.get("success"):
            return {"success": False, "error": "host lookup failed"}

        # 2. get doped entry data
        doped_docs = mpr.materials.summary.search(
            material_ids=[doped_entry_material_id],
            fields=[
                "material_id", "formula_pretty", "symmetry",
                "total_magnetization_normalized_vol",
                "total_magnetization_normalized_formula_units",
                "is_magnetic", "ordering"
            ]
        )
        if not doped_docs:
            return {"success": False, "error": "doped entry lookup failed"}
        doped_doc = doped_docs[0]

        # 3. pull normalized volumes
        host_mp_Mv = host_phase_data.get("magnetization_per_vol_muB_per_bohr3")
        doped_Mv = getattr(doped_doc, "total_magnetization_normalized_vol", None)

        if host_mp_Mv is None or doped_Mv is None:
            return {
                "success": False,
                "error": "missing normalized magnetization_per_volume for one or both phases"
            }

        Ms_host = muB_per_bohr3_to_kA_per_m(host_mp_Mv)
        Ms_doped = muB_per_bohr3_to_kA_per_m(float(doped_Mv))

        # 4. same-phase check
        host_sg = None
        if "phase" in host_phase_data and host_phase_data["phase"]:
            host_sg = host_phase_data["phase"].get("space_group")
        doped_sg = None
        if hasattr(doped_doc, "symmetry") and doped_doc.symmetry:
            doped_sg = getattr(doped_doc.symmetry, "symbol", None)

        same_sg = (host_sg is not None and doped_sg is not None and host_sg == doped_sg)

        # 5. compute percent change (careful when Ms_host ~ 0)
        if abs(Ms_host) > 1e-9:
            pct = (Ms_doped - Ms_host) / abs(Ms_host) * 100.0
        else:
            pct = None  # can't do % if baseline is ~0

        return {
            "success": True,
            "host_formula": host_formula,
            "doped_material_id": doped_entry_material_id,
            "Ms_host_kA_per_m": Ms_host,
            "Ms_doped_kA_per_m": Ms_doped,
            "delta_kA_per_m": Ms_doped - Ms_host,
            "percent_change": pct,
            "same_space_group": same_sg,
            "caution": None if same_sg else (
                "Doped structure has different space group from host, so this is probably a different phase. "
                "You CANNOT call this 'improved magnetization of the host phase'."
            )
        }

    except Exception as e:
        _log.error(f"Error in assess_doping_effect_on_saturation_magnetization: {e}", exc_info=True)
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

