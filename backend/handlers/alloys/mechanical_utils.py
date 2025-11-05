"""
Physics-based mechanical property assessment for alloy phases.

This module provides research-grade mechanical property calculations using:
- Voigt-Reuss-Hill (VRH) averaging for elastic properties
- Materials Project elastic tensor data
- Pugh ratio (G/B) for ductility assessment
- Phase-resolved elastic moduli with proper tensor handling

References:
    - Watt et al., Phys. Earth Planet. Inter. 10 (1975) — VRH averaging
    - Pugh, Phil. Mag. 45 (1954) — G/B ratio for ductility
    - Materials Project API for elastic tensors
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------

@dataclass
class PhaseElastic:
    """Container for elastic properties of a phase (polycrystalline or single-crystal).
    
    If full elastic tensor C (GPa) is provided, Voigt/Reuss/VRH are computed. Otherwise,
    isotropic B (bulk) and G (shear) are used directly.
    """
    name: str
    C: Optional[List[List[float]]] = None  # 6x6 in GPa (Voigt notation)
    B_GPa: Optional[float] = None
    G_GPa: Optional[float] = None
    E_GPa: Optional[float] = None
    nu: Optional[float] = None
    source: str = "unknown"
    
    def ensure_isotropic(self) -> None:
        """Compute E and nu from B and G if not already set."""
        if self.E_GPa is None and self.B_GPa is not None and self.G_GPa is not None:
            # E = 9BG/(3B+G), nu = (3B - 2G)/(2(3B+G))
            B, G = self.B_GPa, self.G_GPa
            denom = (3.0*B + G)
            if denom > 1e-9:
                self.E_GPa = 9.0*B*G/denom
                self.nu = (3.0*B - 2.0*G)/(2.0*denom)
    
    @property
    def pugh(self) -> Optional[float]:
        """Pugh ratio G/B. Values > 0.57 typically indicate brittle behavior."""
        if self.B_GPa is None or self.G_GPa is None:
            return None
        return float(self.G_GPa)/float(self.B_GPa)


# --------------------------------------------------------------------------------------
# Utility physics
# --------------------------------------------------------------------------------------

def voigt_reuss_hill_from_C(C: List[List[float]]) -> Tuple[float, float]:
    """Compute isotropic bulk (B) and shear (G) moduli from a 6×6 stiffness tensor (Voigt).
    
    Returns (B_VRH, G_VRH) in GPa.
    
    References:
        • Watt et al., Phys. Earth Planet. Inter. 10 (1975) — VRH averaging
        • Nye, "Physical Properties of Crystals" (classic background)
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C must be 6x6 Voigt matrix")
    
    # Voigt averages
    B_V = (C[0, 0] + C[1, 1] + C[2, 2] + 2.0*(C[0, 1] + C[1, 2] + C[0, 2]))/9.0
    G_V = (
        C[0, 0] + C[1, 1] + C[2, 2]
        - (C[0, 1] + C[1, 2] + C[0, 2])
        + 3.0*(C[3, 3] + C[4, 4] + C[5, 5])
    )/15.0
    
    # Reuss averages via compliance S = C^{-1}
    S = np.linalg.inv(C)
    B_R = 1.0/(S[0, 0] + S[1, 1] + S[2, 2] + 2.0*(S[0, 1] + S[1, 2] + S[0, 2]))
    G_R = 15.0/(4.0*(S[0, 0] + S[1, 1] + S[2, 2]) - 4.0*(S[0, 1] + S[1, 2] + S[0, 2]) + 3.0*(S[3, 3] + S[4, 4] + S[5, 5]))
    
    B = 0.5*(B_V + B_R)
    G = 0.5*(G_V + G_R)
    return float(B), float(G)


def E_nu_from_BG(B: float, G: float) -> Tuple[float, float]:
    """Compute Young's modulus E and Poisson ratio nu from bulk B and shear G moduli."""
    if (3.0*B + G) <= 1e-12:
        raise ValueError("Invalid B,G for isotropic conversion")
    E = 9.0*B*G/(3.0*B + G)
    nu = (3.0*B - 2.0*G)/(2.0*(3.0*B + G))
    return float(E), float(nu)


def pugh_ratio(B: Optional[float], G: Optional[float]) -> Optional[float]:
    """Calculate Pugh ratio G/B. Values > 0.57 typically indicate brittle behavior."""
    if B is None or G is None:
        return None
    return float(G)/float(B)


def cauchy_pressure_indicator(C: Optional[List[List[float]]]) -> Optional[float]:
    """Return C12 − C44 if tensor is available; negative often correlates with brittleness
    in many metals/ceramics (Pettifor)."""
    if not C:
        return None
    try:
        return float(C[0][1] - C[3][3])
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Elastic data acquisition
# --------------------------------------------------------------------------------------

class ElasticDataClient:
    """Fetch/construct elastic properties for phases.
    
    Preferred sources (in order):
        1) Provided C (6×6) tensor (GPa) → VRH → B,G,E,ν
        2) Materials Project elastic data via formula search (requires MP API client)
        3) Provided isotropic B,G
        4) Built-in heuristic defaults per crystal family (last resort; discouraged)
    """
    def __init__(self, mpr_client: Optional[object] = None):
        self._mpr = mpr_client
    
    def from_phase_name(
        self, 
        phase_name: str,
        composition_hint: Dict[str, float],
        system_elems: Tuple[str, ...],
        provided_elastic: Optional[PhaseElastic] = None
    ) -> PhaseElastic:
        """Get elastic properties for a phase by name.
        
        Args:
            phase_name: CALPHAD phase label (e.g., "AL2FE", "BCC_A2")
            composition_hint: Dict with element percentages
            system_elems: Tuple of elements in system
            provided_elastic: Optional pre-computed elastic data
            
        Returns:
            PhaseElastic object with B, G, E, nu, Pugh ratio
        """
        # 1) If tensor or B,G given directly
        if provided_elastic and provided_elastic.C:
            B, G = voigt_reuss_hill_from_C(provided_elastic.C)
            E, nu = E_nu_from_BG(B, G)
            return PhaseElastic(
                name=phase_name, C=provided_elastic.C, B_GPa=B, G_GPa=G, E_GPa=E, nu=nu,
                source=f"user_tensor:{provided_elastic.source}"
            )
        
        if provided_elastic and (provided_elastic.B_GPa is not None) and (provided_elastic.G_GPa is not None):
            e = PhaseElastic(
                name=phase_name, B_GPa=provided_elastic.B_GPa, G_GPa=provided_elastic.G_GPa,
                E_GPa=provided_elastic.E_GPa, nu=provided_elastic.nu, source=provided_elastic.source
            )
            e.ensure_isotropic()
            return e
        
        # 2) Materials Project lookup
        if self._mpr:
            try:
                from ..shared.calphad_utils import parse_calphad_phase_name
                formula = parse_calphad_phase_name(phase_name, system_elems)
                
                if formula:
                    _log.info(f"Searching MP for phase {phase_name} with formula {formula}")
                    
                    # Search for materials matching the formula
                    docs = self._mpr.materials.summary.search(
                        formula=formula,
                        fields=["material_id", "formula_pretty", "symmetry", 
                               "bulk_modulus", "shear_modulus"]
                    )
                    
                    if docs:
                        # Take first match with elasticity data
                        for doc in docs[:3]:  # Check up to 3 matches
                            try:
                                # Get elastic properties if available (VRH averages)
                                bulk_mod = getattr(doc, 'bulk_modulus', None)
                                shear_mod = getattr(doc, 'shear_modulus', None)
                                
                                # bulk_modulus and shear_modulus might be dict with 'vrh' key
                                if isinstance(bulk_mod, dict):
                                    bulk_mod = bulk_mod.get('vrh', None)
                                if isinstance(shear_mod, dict):
                                    shear_mod = shear_mod.get('vrh', None)
                                
                                if bulk_mod is not None and shear_mod is not None:
                                    B = float(bulk_mod)
                                    G = float(shear_mod)
                                    E, nu = E_nu_from_BG(B, G)
                                    
                                    _log.info(f"Found elastic data for {phase_name}: B={B:.1f} GPa, G={G:.1f} GPa, Pugh={G/B:.3f}")
                                    
                                    return PhaseElastic(
                                        name=phase_name, B_GPa=B, G_GPa=G, E_GPa=E, nu=nu,
                                        source=f"MP:{doc.material_id}"
                                    )
                            except Exception as e:
                                _log.debug(f"Error processing MP doc: {e}")
                                continue
            except Exception as e:
                _log.warning(f"Could not fetch MP data for {phase_name}: {e}")
        
        # 3) Heuristic fallbacks (do not use for final claims; marked as heuristic)
        # Typical orders of magnitude at RT
        category = _categorize_phase(phase_name)
        defaults = {
            "fcc": (76.0, 26.0),   # Al-like (B,G) GPa
            "bcc": (170.0, 82.0),  # Fe-like
            "hcp": (160.0, 45.0),
            "laves": (180.0, 90.0),
            "tau": (160.0, 80.0),
            "gamma": (110.0, 45.0)
        }
        B, G = defaults.get(category, (120.0, 60.0))
        E, nu = E_nu_from_BG(B, G)
        
        _log.info(f"Using heuristic elastic data for {phase_name} (category={category}): B={B:.1f} GPa, G={G:.1f} GPa")
        
        return PhaseElastic(name=phase_name, B_GPa=B, G_GPa=G, E_GPa=E, nu=nu, source=f"heuristic:{category}")


def _categorize_phase(name: str) -> str:
    """Categorize a phase by name patterns."""
    n = name.lower()
    if "laves" in n:
        return "laves"
    if "tau" in n or "τ" in n:
        return "tau"
    if "gamma" in n or "γ" in n:
        return "gamma"
    if "fcc" in n or "a1" in n:
        return "fcc"
    if "bcc" in n or "a2" in n:
        return "bcc"
    if "hcp" in n or "a3" in n:
        return "hcp"
    return "other"


def get_phase_mechanical_descriptors(
    phase_name: str,
    composition_hint: Dict[str, float],
    system_elems: Tuple[str, ...],
    mpr_client: Optional[object] = None
) -> Dict[str, Any]:
    """
    Get physics-based mechanical descriptors for a phase using Materials Project API.
    
    This function:
    1. Classifies phase type (intermetallic, solid solution, BCC/FCC/HCP)
    2. Attempts to fetch elastic properties from Materials Project
    3. Calculates Pugh ratio (G/B) to assess brittleness (> 0.57 = brittle)
    4. Falls back to physics-based heuristics if no MP data available
    
    Args:
        phase_name: CALPHAD phase label (e.g., "AL2FE", "BCC_A2")
        composition_hint: dict with element percentages
        system_elems: tuple of elements in system
        mpr_client: Materials Project API client
        
    Returns:
        {
            "name": str,
            "is_intermetallic": bool,
            "is_bcc_like": bool,
            "is_fcc_like": bool,
            "is_hcp_like": bool,
            "category": str,
            "bulk_modulus_GPa": float | None,
            "shear_modulus_GPa": float | None,
            "E_GPa": float | None,
            "nu": float | None,
            "pugh_ratio": float | None,
            "brittle_flag": bool | None,
            "cauchy_pressure": float | None,
            "source": str,
        }
    """
    try:
        client = ElasticDataClient(mpr_client)
        elastic = client.from_phase_name(phase_name, composition_hint, system_elems)
        
        category = _categorize_phase(phase_name)
        is_bcc_like = category == "bcc"
        is_fcc_like = category == "fcc"
        is_hcp_like = category == "hcp"
        is_intermetallic = category in ("laves", "tau", "gamma") or not (is_bcc_like or is_fcc_like or is_hcp_like)
        
        # Determine brittleness from Pugh ratio if available
        brittle_flag = None
        if elastic.pugh is not None:
            # Pugh criterion: G/B > 0.57 → brittle, < 0.57 → ductile
            brittle_flag = (elastic.pugh >= 0.57)
        else:
            # Fallback: intermetallics typically brittle, FCC typically ductile
            if is_intermetallic:
                brittle_flag = True
            elif is_fcc_like:
                brittle_flag = False
            elif is_bcc_like:
                brittle_flag = False  # BCC can be ductile at high T
        
        return {
            "name": phase_name,
            "is_intermetallic": is_intermetallic,
            "is_bcc_like": is_bcc_like,
            "is_fcc_like": is_fcc_like,
            "is_hcp_like": is_hcp_like,
            "category": category,
            "bulk_modulus_GPa": elastic.B_GPa,
            "shear_modulus_GPa": elastic.G_GPa,
            "E_GPa": elastic.E_GPa,
            "nu": elastic.nu,
            "pugh_ratio": elastic.pugh,
            "brittle_flag": brittle_flag,
            "cauchy_pressure": cauchy_pressure_indicator(elastic.C),
            "source": elastic.source,
        }
        
    except Exception as e:
        _log.error(f"Error getting mechanical descriptors for {phase_name}: {e}", exc_info=True)
        return {
            "name": phase_name,
            "is_intermetallic": False,
            "is_bcc_like": False,
            "is_fcc_like": False,
            "is_hcp_like": False,
            "category": "unknown",
            "bulk_modulus_GPa": None,
            "shear_modulus_GPa": None,
            "E_GPa": None,
            "nu": None,
            "pugh_ratio": None,
            "brittle_flag": None,
            "cauchy_pressure": None,
            "source": "error"
        }


# --------------------------------------------------------------------------------------
# VRH composite moduli for multi-phase materials
# --------------------------------------------------------------------------------------

def vrh_composite_moduli(
    phase_fractions: Dict[str, float],
    phase_elastic_data: Dict[str, PhaseElastic]
) -> Tuple[float, float, float, float]:
    """Voigt–Reuss–Hill mixture for polycrystal composite.
    
    Returns (B_VRH, G_VRH, E_VRH, nu_VRH).
    
    Implementation: compute Voigt and Reuss bounds for phases treated as
    isotropic constituents using their B,G and volume fractions f_i, then average.
    
    Args:
        phase_fractions: Dict mapping phase name to volume fraction
        phase_elastic_data: Dict mapping phase name to PhaseElastic
        
    Returns:
        (B_VRH, G_VRH, E_VRH, nu_VRH) in GPa (and dimensionless for nu)
    """
    # Collect B,G,f for each phase
    Bs, Gs, fs = [], [], []
    for phase_name, frac in phase_fractions.items():
        e = phase_elastic_data.get(phase_name)
        if not e:
            raise ValueError(f"Phase {phase_name} lacks elastic data")
        B, G = e.B_GPa, e.G_GPa
        if (B is None) or (G is None):
            raise ValueError(f"Phase {phase_name} lacks B or G modulus")
        Bs.append(B)
        Gs.append(G)
        fs.append(frac)
    
    # Normalize fractions
    s = sum(fs)
    if s <= 0.0:
        raise ValueError("Zero total phase fraction")
    fs = [f/s for f in fs]
    
    # Voigt bounds: linear in volume fraction
    B_V = sum(f*B for f, B in zip(fs, Bs))
    G_V = sum(f*G for f, G in zip(fs, Gs))
    
    # Reuss bounds
    B_R = 1.0 / sum(f/(B+1e-12) for f, B in zip(fs, Bs))
    G_R = 1.0 / sum(f/(G+1e-12) for f, G in zip(fs, Gs))
    
    B = 0.5*(B_V + B_R)
    G = 0.5*(G_V + G_R)
    E, nu = E_nu_from_BG(B, G)
    return B, G, E, nu
