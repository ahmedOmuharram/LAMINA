"""
Atomic and elemental data utilities for alloy calculations.

This module provides functions to retrieve atomic properties like cohesive energy
and metallic radius from various sources (mendeleev, pymatgen).

These utilities are generally applicable and could be used by other handlers
dealing with elements and their properties.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
from ..base.converters import kjmol_to_ev as _kjmol_to_ev
from ..base.constants import COHESIVE_ENERGY_FALLBACK

_log = logging.getLogger(__name__)

def get_cohesive_energy(symbol: str) -> Tuple[Optional[float], str]:
    """
    Return cohesive (atomization) energy in eV/atom and a source tag.

    Priority:
      1) mendeleev: evaporation_heat (+ fusion_heat if available) → eV/atom
      2) mendeleev: cohesive_energy if present (infer units if needed)
      3) curated fallback table (eV/atom)
      
    Args:
        symbol: Element symbol (e.g., 'Al', 'Fe')
        
    Returns:
        Tuple of (cohesive_energy_eV, source_string)
        Returns (None, "missing") if data not available
        
    Note:
        This utility is generally applicable to any handler needing cohesive energy data.
        Consider moving to handlers/shared/ or handlers/base/ if used by multiple handlers.
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

    val = COHESIVE_ENERGY_FALLBACK.get(symbol)
    return (val, "fallback_table") if val is not None else (None, "missing")


def get_metal_radius(symbol: str) -> Optional[float]:
    """
    Approximate metallic/atomic radius in Å.
    
    Tries mendeleev metallic radius (pm converted to Å), then falls back to
    pymatgen atomic radius properties.
    
    Args:
        symbol: Element symbol (e.g., 'Al', 'Fe')
        
    Returns:
        Radius in Angstroms, or None if not available
        
    Note:
        This utility is generally applicable to any handler needing atomic radii.
        Consider moving to handlers/shared/ or handlers/base/ if used by multiple handlers.
    """
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

