from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Annotated, Tuple

from kani import ai_function, AIParam
from ..base import BaseHandler

KJMOL_PER_EV_PER_ATOM = 96.4853321233  # 1 eV/atom = 96.485... kJ/mol

# Heuristic anchors (literature/“rules of thumb”):
#   E_ads ≈ 0.20–0.30 * E_coh(host)   (broad metal-on-metal scaling)
#   E_diff ≈ f_mech * E_ads, where f_mech ≈ 0.12 (hopping) or 0.22–0.30 (exchange)
# We’ll pick facet-specific midpoints and still return a wide uncertainty band later.
ADS_OVER_COH_111 = 0.22  # close-packed surfaces bind a bit more weakly on average
ADS_OVER_COH_100 = 0.26  # (100) often stronger site competition + exchange tendency
ADS_OVER_COH_110 = 0.28  # open surfaces ~stronger corrugation

# Facet-dependent diffusion/adsorption fractions (representative midpoints)
DIFF_OVER_ADS_111 = 0.12  # hopping-dominated
DIFF_OVER_ADS_100 = 0.24  # exchange-dominated for many metals on fcc(100)
DIFF_OVER_ADS_110 = 0.28  # often even “rougher”/higher

_log = logging.getLogger(__name__)

def _kjmol_to_ev(x_kjmol: float) -> float:
    return float(x_kjmol) / KJMOL_PER_EV_PER_ATOM

def _get_cohesive_energy(symbol: str) -> tuple[Optional[float], str]:
    """
    Return cohesive (atomization) energy in eV/atom and a source tag.

    Priority:
      1) mendeleev: evaporation_heat (+ fusion_heat if available) → eV/atom
      2) mendeleev: cohesive_energy if present (infer units if needed)
      3) curated fallback table (eV/atom)
    """
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)

        evap = getattr(e, "evaporation_heat", None)    # kJ/mol
        fus  = getattr(e, "fusion_heat", None)         # kJ/mol (may be None)
        if evap is not None:
            coh_kj = float(evap) + (float(fus) if fus is not None else 0.0)
            return (_kjmol_to_ev(coh_kj), "mendeleev.evaporation_heat(+fusion)")

        # Looser cohesive_energy field (units can vary)
        val = getattr(e, "cohesive_energy", None)
        if val is not None:
            v = float(val)
            # If surprisingly large, assume kJ/mol
            ev = v if v < 20.0 else _kjmol_to_ev(v)
            return (ev, "mendeleev.cohesive_energy")
    except Exception:
        pass

    COHESIVE_FALLBACK = {
        # Typical cohesive energies (eV/atom) near 0 K (solid → atoms):
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
    val = COHESIVE_FALLBACK.get(symbol)
    return (val, "fallback_table") if val is not None else (None, "missing")

def _normalize_surface(surface_miller: Optional[str]) -> Tuple[str, float, str, float]:
    """
    Parse/normalize surface and return:
      (facet, facet_multiplier, mechanism, diff_over_ads)
    Stronger multipliers reflect well-known facet/mechanism differences.
    """
    if not surface_miller:
        # Unspecified facet: assume generic terrace and broaden uncertainty later
        return "unspecified", 1.00, "unknown", 0.18  # midpoint between hopping and exchange

    s = str(surface_miller).lower()
    for token in ("fcc", "bcc", "hcp", "sc", "diamond", "al", "au"):
        s = s.replace(token, "")
    for ch in "()[]{}-–, ":
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch.isdigit())

    if s == "111":
        return "111", 0.75, "hopping", DIFF_OVER_ADS_111
    if s == "100":
        return "100", 1.25, "exchange", DIFF_OVER_ADS_100
    if s == "110":
        return "110", 1.45, "exchange", DIFF_OVER_ADS_110
    return "unspecified", 1.00, "unknown", 0.18

def _get_metal_radius(symbol: str) -> Optional[float]:
    """Approximate metallic/atomic radius in Å (mendeleev pm preferred, else pymatgen)."""
    try:
        from mendeleev import element as md_element  # type: ignore
        e = md_element(symbol)
        for attr in ("metallic_radius_c12", "metallic_radius"):
            val = getattr(e, attr, None)
            if val is not None:
                return float(val) / 100.0  # pm → Å
    except Exception:
        pass

    try:
        from pymatgen.core import Element as PMGElement  # type: ignore
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
    """Alloy surface heuristics for adatom diffusion barriers with facet/mechanism awareness."""

    def __init__(self, mpr_client: Optional[object] = None) -> None:
        super().__init__(mpr_client)
        _log.info("AlloyHandler initialized")

    @ai_function(
        desc=(
            "Estimate the surface diffusion barrier (activation energy, eV) for an adatom on a metal surface. "
            "Facet (111/100/110) and likely mechanism (hopping vs exchange) are accounted for. "
            "Intended for rough scoping; use DFT+NEB for quantitative work."
        ),
        auto_truncate=128000,
    )
    async def estimate_surface_diffusion_barrier(
        self,
        adatom_element: Annotated[str, AIParam(desc="Adatom element, e.g., 'Au'.")],
        host_element: Annotated[str, AIParam(desc="Host surface element, e.g., 'Al'.")],
        surface_miller: Annotated[Optional[str], AIParam(desc="Surface orientation like '111', '100', '110' (optional).")] = None,
    ) -> Dict[str, Any]:
        try:
            ad = adatom_element.capitalize()
            host = host_element.capitalize()

            ecoh_ad, src_ad = _get_cohesive_energy(ad)
            ecoh_host, src_host = _get_cohesive_energy(host)
            r_ad = _get_metal_radius(ad)
            r_host = _get_metal_radius(host)

            if ecoh_host is None or r_ad is None or r_host is None:
                return {
                    "success": False,
                    "error": "Insufficient data for estimate.",
                    "missing": {
                        "cohesive_energy_host": ecoh_host,
                        "radius_adatom": r_ad,
                        "radius_host": r_host,
                    },
                }

            facet, facet_mult, mechanism, diff_over_ads = _normalize_surface(surface_miller)

            # Choose facet-specific adsorption scaling vs host cohesion
            if facet == "111":
                ads_over_coh = ADS_OVER_COH_111
            elif facet == "100":
                ads_over_coh = ADS_OVER_COH_100
            elif facet == "110":
                ads_over_coh = ADS_OVER_COH_110
            else:
                # unknown facet → mid value
                ads_over_coh = 0.25

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

            # Confidence: facet specified → medium; unspecified → low
            confidence = "high" if facet in ("111", "100", "110") else "low"

            return {
                "success": True,
                "adatom": ad,
                "host": host,
                "surface": facet,
                "surface_input": surface_miller,
                "activation_energy_eV": round(ea_est, 4),
                "energy_range_eV": [ea_low, ea_high],
                "confidence": confidence,
                "likely_mechanism": mechanism,
                "method": "descriptor_model_v2_facetaware",
                "descriptors": {
                    "cohesive_energy_adatom_eV": round(ecoh_ad, 4) if ecoh_ad is not None else None,
                    "cohesive_energy_host_eV": round(ecoh_host, 4),
                    "ads_over_coh_fraction": round(ads_over_coh, 3),
                    "diff_over_ads_fraction": round(diff_over_ads, 3),
                    "facet_multiplier": round(facet_mult, 2),
                    "size_mismatch_rel": round(size_mismatch_rel, 4),
                    "size_factor": round(size_factor, 3),
                },
                "data_sources": {
                    "cohesive_energy": {
                        "adatom": src_ad,
                        "host": src_host,
                    },
                    "radii": "mendeleev metallic radii (pm→Å), else pymatgen (Å)",
                },
                "caveats": (
                    "Heuristic scaling; actual barriers depend on site, mechanism (hopping vs exchange), "
                    "and reconstruction/segregation. Use DFT+NEB for accuracy; consider alloying tendencies."
                ),
            }
        except Exception as e:
            _log.error(f"Error estimating surface diffusion barrier for {adatom_element} on {host_element}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
