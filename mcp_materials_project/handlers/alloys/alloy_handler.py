from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Annotated

from kani import ai_function, AIParam

from ..base import BaseHandler

# --- constants for conversions and simple calibration ---
KJMOL_PER_EV_PER_ATOM = 96.4853321233  # 1 eV/atom = 96.485... kJ/mol
# Heuristic: E_ads ≈ 0.25 * E_coh (very rough, EMT/BOC style scaling)
# and E_diff ≈ 0.1–0.2 * E_ads (rule of thumb). Use the midpoint 0.15.
ADS_FROM_COH = 0.25
DIFF_FROM_ADS = 0.15
BASE_SCALE = ADS_FROM_COH * DIFF_FROM_ADS  # ≈ 0.0375

_log = logging.getLogger(__name__)


def _get_cohesive_energy(symbol: str) -> Optional[float]:
    """Return cohesive energy in eV/atom for an element if available.

    Strategy:
      1) Prefer mendeleev.evaporation_heat (kJ/mol) -> convert to eV/atom.
      2) If a 'cohesive_energy' attribute exists, convert if it looks like kJ/mol.
      3) Fallback to small curated table (eV/atom).
    """
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)

        # (1) evaporation_heat is documented and unit-specified in kJ/mol.
        evap = getattr(e, "evaporation_heat", None)
        if evap is not None:
            return float(evap) / KJMOL_PER_EV_PER_ATOM  # -> eV/atom

        # (2) loose 'cohesive_energy' if present; normalize units conservatively
        val = getattr(e, "cohesive_energy", None)
        if val is not None:
            v = float(val)
            # If value is large (>20), assume it's kJ/mol and convert
            return v if v < 20.0 else v / KJMOL_PER_EV_PER_ATOM
    except Exception:
        pass

    # (3) last-resort small fallback (extend as needed) in eV/atom
    COHESIVE_FALLBACK: dict[str, float] = {
        "Al": 3.39,
        "Au": 3.81,
        "Cu": 3.49,
        "Ag": 2.95,
        "Ni": 4.44,
        "Pt": 5.84,
        "Pd": 3.89,
        "Fe": 4.28,
        "Ti": 4.85,
        "Mg": 1.51,
    }
    return COHESIVE_FALLBACK.get(symbol)


def _get_metal_radius(symbol: str) -> Optional[float]:
    """Return an approximate metallic/atomic radius in Å.

    Prefers mendeleev metallic radii (pm, convert to Å), then pymatgen.
    """
    # mendeleev first (pm -> Å)
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)
        for attr in ("metallic_radius_c12", "metallic_radius"):
            val = getattr(e, attr, None)
            if val is not None:
                return float(val) / 100.0  # pm -> Å (docs specify pm)
    except Exception:
        pass

    # Fallbacks from pymatgen (reported in Å)
    try:
        from pymatgen.core import Element as PMGElement
        el = PMGElement(symbol)
        for attr in ("metallic_radius", "atomic_radius", "atomic_radius_calculated", "covalent_radius"):
            v = getattr(el, attr, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    continue
    except Exception:
        pass
    return None


class AlloyHandler(BaseHandler):
    """Handler for alloy surface analyses including heuristic diffusion barriers."""

    def __init__(self, mpr_client: Optional[object] = None) -> None:
        super().__init__(mpr_client)
        _log.info("AlloyHandler initialized")

    @ai_function(
        desc=(
            "Calculate/estimate the surface diffusion barrier (activation energy in eV) for an adatom "
            "diffusing on a metal surface. Use this when asked about diffusion barriers, adatom mobility, "
            "or questions like 'What is the barrier for Au on Al?' or 'How easily does X diffuse on Y?'. "
            "Returns a descriptor-based estimate using cohesive energies and size mismatch."
        ),
        auto_truncate=128000,
    )
    async def estimate_surface_diffusion_barrier(
        self,
        adatom_element: Annotated[str, AIParam(desc="Adatom element symbol, e.g., 'Au'.")],
        host_element: Annotated[str, AIParam(desc="Host surface element symbol, e.g., 'Al'.")],
        surface_miller: Annotated[Optional[str], AIParam(desc="Surface orientation like '111', '100', '110'. Optional.")] = None,
    ) -> Dict[str, Any]:
        """Return a coarse, unit-consistent diffusion barrier estimate in eV with rationale."""
        try:
            ad = adatom_element.capitalize()
            host = host_element.capitalize()

            ecoh_ad = _get_cohesive_energy(ad)
            ecoh_host = _get_cohesive_energy(host)
            r_ad = _get_metal_radius(ad)
            r_host = _get_metal_radius(host)

            if ecoh_ad is None or ecoh_host is None or r_ad is None or r_host is None:
                missing = {
                    "cohesive_energy_adatom": ecoh_ad,
                    "cohesive_energy_host": ecoh_host,
                    "radius_adatom": r_ad,
                    "radius_host": r_host,
                }
                return {
                    "success": False,
                    "error": "Insufficient data for estimate (see 'missing')",
                    "missing": missing,
                }

            # size mismatch (dimensionless) using sum in denominator (bounded in [0,1))
            size_mismatch = abs(r_ad - r_host) / max(1e-6, (r_ad + r_host))
            size_factor = max(0.5, min(1.05, 1.0 - 0.3 * size_mismatch))  # gentle penalty, light boost cap

            # base reference scale from the *weaker* cohesion
            ecoh_ref = min(ecoh_ad, ecoh_host)

            # Adsorption estimate then diffusion fraction
            e_ads_est = ADS_FROM_COH * ecoh_ref
            ea_est = DIFF_FROM_ADS * e_ads_est  # ≈ 0.0375 * ecoh_ref

            # Surface orientation modifier: (111) < (100) < (110) on average
            surf_mod = 1.0
            if surface_miller:
                s = str(surface_miller).strip().replace("(", "").replace(")", "")
                if s == "111":
                    surf_mod = 0.85
                elif s == "100":
                    surf_mod = 1.00
                elif s == "110":
                    surf_mod = 1.15

            ea_est *= size_factor * surf_mod

            # Keep within a plausible metal-adatom window, but less aggressive
            ea_est = float(max(0.01, min(2.00, ea_est)))

            return {
                "success": True,
                "adatom": ad,
                "host": host,
                "surface": surface_miller,
                "activation_energy_eV": round(ea_est, 4),
                "method": "descriptor_model_v2",
                "descriptors": {
                    "cohesive_energy_adatom_eV": round(ecoh_ad, 4),
                    "cohesive_energy_host_eV": round(ecoh_host, 4),
                    "estimated_adsorption_energy_eV": round(e_ads_est, 4),
                    "diff_over_ads_fraction": DIFF_FROM_ADS,
                    "ads_over_coh_fraction": ADS_FROM_COH,
                    "size_mismatch": round(size_mismatch, 6),
                    "surface_modifier": float(surf_mod),
                },
                "notes": (
                    "Heuristic: Ea ≈ 0.1–0.2 × E_ads and E_ads ~ 0.25 × E_coh (very rough). "
                    "Cohesive energies via evaporation_heat (kJ/mol) converted to eV/atom. "
                    "For accuracy, use DFT+NEB on the explicit surface with the correct adsorption sites."
                ),
            }

        except Exception as e:
            _log.error(f"Error estimating surface diffusion barrier for {adatom_element} on {host_element}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
