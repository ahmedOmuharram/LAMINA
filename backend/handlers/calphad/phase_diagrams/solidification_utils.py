"""
Solidification simulation utilities for CALPHAD.

Implements Scheil-Gulliver style solidification modeling to predict
as-cast microstructures, not infinite-time equilibrium.
"""

import numpy as np
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pycalphad import Database

from ...shared.calphad_utils import (
    extract_phase_fractions_from_equilibrium,
    compute_equilibrium
)

_log = logging.getLogger(__name__)

# Scheil-Gulliver cache configuration
SCHEIL_CACHE_VERSION = "v1.0.0"  # Increment to invalidate old cache
SCHEIL_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "scheil_cache"


def _generate_scheil_cache_key(
    db_name: str,
    elements: List[str],
    composition: Dict[str, float],
    T_start: float,
    T_end: float,
    dT: float,
) -> str:
    """
    Generate a cache key for Scheil-Gulliver results.
    
    Rounds compositions to 3 decimal places for better cache hit rate.
    """
    # Sort elements for consistent ordering
    sorted_elements = sorted(elements)
    
    # Normalize composition keys to uppercase for case-insensitive matching
    # This handles cases where composition={'Mg': 2.67} but elements=['MG']
    comp_normalized = {k.upper(): v for k, v in composition.items()}
    
    # Round composition to 3 decimal places
    rounded_comp = {el.upper(): round(comp_normalized.get(el.upper(), 0.0), 3) for el in sorted_elements}
    
    # Create a deterministic string representation
    comp_str = "_".join([f"{el}{rounded_comp[el]:.3f}" for el in sorted_elements if rounded_comp[el] > 0.001])
    
    # Include simulation parameters that affect results
    key_parts = [
        SCHEIL_CACHE_VERSION,
        db_name,
        comp_str,
        f"T{T_start:.1f}-{T_end:.1f}",
        f"dT{dT:.1f}"
    ]
    
    cache_key = "_".join(key_parts)
    
    # Hash if too long (filesystem limits)
    if len(cache_key) > 200:
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_key = f"{SCHEIL_CACHE_VERSION}_{db_name}_{cache_key_hash}"
    
    return cache_key


def _load_scheil_from_cache(cache_key: str) -> Optional[Dict[str, float]]:
    """Load Scheil-Gulliver results from cache."""
    try:
        cache_file = SCHEIL_CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                _log.info(f"✓ Loaded Scheil results from cache: {cache_key}")
                return cached_data
    except Exception as e:
        _log.warning(f"Failed to load from cache {cache_key}: {e}")
    return None


def _save_scheil_to_cache(cache_key: str, results: Dict[str, float]) -> None:
    """Save Scheil-Gulliver results to cache."""
    try:
        # Create cache directory if it doesn't exist
        SCHEIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_file = SCHEIL_CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        _log.info(f"Saved Scheil results to cache: {cache_key}")
    except Exception as e:
        _log.warning(f"Failed to save to cache {cache_key}: {e}")


def find_liquidus_solidus_temperatures(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    T_start: float = 1200.0,
    T_end: float = 200.0,
    n_points: int = 50
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find liquidus and solidus temperatures by sweeping temperature.
    
    Liquidus: Temperature where first solid forms (liquid fraction drops below 0.99)
    Solidus: Temperature where last liquid disappears (liquid fraction drops below 0.01)
    
    Args:
        db: PyCalphad Database
        elements: List of elements (with VA if needed)
        phases: List of phase names
        composition: Dict of {element: mole_fraction}
        T_start: Starting temperature (high, above liquidus)
        T_end: Ending temperature (low, below solidus)
        n_points: Number of temperature points to check
        
    Returns:
        (T_liquidus, T_solidus) or (None, None) if not found
    """
    try:
        # Filter out elements with negligible composition (< 1e-6)
        active_elements = {el: frac for el, frac in composition.items() if frac > 1e-6}
        
        _log.debug(f"Finding liquidus/solidus with composition: {active_elements}")
        
        temperatures = np.linspace(T_start, T_end, n_points)
        
        T_liquidus = None
        T_solidus = None
        previous_liquid_frac = 1.0  # Start assuming fully liquid at high T
        
        for T in temperatures:
            try:
                # Use compute_equilibrium utility to handle composition normalization,
                # element handling, condition building, and equilibrium calculation
                eq = compute_equilibrium(
                    db=db,
                    elements=elements,
                    phases=phases,
                    composition=active_elements,
                    temperature=T,
                    pressure=101325
                )
                
                if eq is None:
                    _log.debug(f"Equilibrium failed at T={T}K")
                    continue
                
                phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                liquid_frac = phase_fractions.get("LIQUID", 0.0)
                
                # Liquidus: first T (going high→low) where liquid < 0.99 (some solid forms)
                if T_liquidus is None and liquid_frac < 0.99:
                    T_liquidus = T
                    _log.debug(f"Liquidus detected at {T}K (liquid={liquid_frac:.3f})")
                
                # Solidus: first T (going high→low) where liquid drops below 0.01
                # We detect the transition from liquid present to liquid absent
                if T_solidus is None and liquid_frac < 0.01 and previous_liquid_frac >= 0.01:
                    # Liquid just disappeared - solidus is at previous temperature
                    # Use linear interpolation for better accuracy
                    if len(temperatures) > 1:
                        T_solidus = T + (temperatures[1] - temperatures[0]) * 0.5
                    else:
                        T_solidus = T
                    _log.debug(f"Solidus detected at ~{T_solidus}K (liquid went from {previous_liquid_frac:.3f} to {liquid_frac:.3f})")
                
                previous_liquid_frac = liquid_frac
                
                # Early exit if we found both
                if T_liquidus is not None and T_solidus is not None:
                    break
                    
            except Exception as e:
                _log.debug(f"Equilibrium calculation failed at T={T}K: {e}")
                continue
        
        # If we never found solidus but found liquidus, estimate it
        if T_liquidus is not None and T_solidus is None:
            # Check if liquid is still present at the lowest temperature
            if previous_liquid_frac > 0.01:
                _log.warning(f"Solidus not found - liquid still present at {T_end}K (frac={previous_liquid_frac:.3f})")
                _log.warning("Consider lowering T_end or increasing n_points")
            else:
                # Liquid disappeared but we didn't catch the transition
                # This can happen if n_points is too small
                _log.warning(f"Solidus not precisely detected (liquid={previous_liquid_frac:.3f} at T_end)")
                # Use the lowest temperature where we have data
                T_solidus = T_end
        
        _log.info(f"Found liquidus={T_liquidus}K, solidus={T_solidus}K")
        return T_liquidus, T_solidus
        
    except Exception as e:
        _log.error(f"Error finding liquidus/solidus: {e}", exc_info=True)
        return None, None


def simulate_scheil_gulliver(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    T_start: float = 1200.0,
    T_end: float = 200.0,
    dT: float = 5.0,
    df_solid: float = 0.01,
    pressure: float = 101325,
    use_cache: bool = True,
) -> Dict[str, float]:
    """
    Perform a true Scheil-Gulliver solidification simulation.
    
    Assumes:
        - Perfect mixing in the liquid
        - No diffusion in the solid
    Returns final solid phase fractions after all liquid solidifies.
    
    CACHING: Results are cached based on database, composition (rounded to 3 decimals),
    and simulation parameters. Set use_cache=False to bypass cache.
    
    Args:
        db: PyCalphad Database
        elements: List of elements
        phases: List of phase names
        composition: Dict of {element: mole_fraction}
        T_start: Starting temperature (K), above liquidus
        T_end: Ending temperature (K), below solidus
        dT: Temperature step (K)
        df_solid: Solid fraction increment per step
        pressure: Pressure in Pa
        use_cache: If True, check cache before computing (default: True)
        
    Returns:
        Dict of {phase_name: final_fraction} after complete solidification
    """
    # Try to load from cache
    if use_cache:
        db_name = getattr(db, 'name', 'unknown')
        cache_key = _generate_scheil_cache_key(
            db_name=db_name,
            elements=elements,
            composition=composition,
            T_start=T_start,
            T_end=T_end,
            dT=dT
        )
        
        cached_result = _load_scheil_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
    
    # Initialize liquid composition
    xL = {el: frac for el, frac in composition.items() if frac > 1e-10}
    total = sum(xL.values())
    xL = {k: v / total for k, v in xL.items()}
    
    f_liquid = 1.0
    solid_totals = {p: 0.0 for p in phases if p != "LIQUID"}
    T = T_start
    
    _log.info(f"Starting Scheil-Gulliver simulation from {T_start}K to {T_end}K")
    n_steps = 0
    max_steps = int((T_start - T_end) / dT) + 100  # Safety limit
    
    while T > T_end and f_liquid > 1e-4 and n_steps < max_steps:
        n_steps += 1
        
        # Equilibrium at current temperature with current liquid composition
        eq = compute_equilibrium(
            db=db,
            elements=elements,
            phases=phases,
            composition=xL,
            temperature=T,
            pressure=pressure,
        )
        
        if eq is None:
            T -= dT
            continue
        
        phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
        fL_eq = phase_fractions.get("LIQUID", 0.0)
        solid_phases = {ph: frac for ph, frac in phase_fractions.items() 
                       if ph != "LIQUID" and frac > 1e-6}
        
        if fL_eq > 0.999 or not solid_phases:
            # Still fully liquid, no solid formation yet
            T -= dT
            continue
        
        # Normalize solid fractions (relative to total solid, not to unity)
        total_solid_eq = sum(solid_phases.values())
        if total_solid_eq < 1e-6:
            T -= dT
            continue
            
        solid_rel = {ph: f / total_solid_eq for ph, f in solid_phases.items()}
        
        # Extract phase compositions from equilibrium
        # For each solid phase, get its elemental composition
        try:
            # Get composition arrays from equilibrium result
            # eq.X has dimensions: (component, phase, ...)
            phase_comps = {}
            for phase_name in solid_rel.keys():
                try:
                    # Extract composition for this phase
                    comp_dict = {}
                    for el in elements:
                        if el == 'VA':  # Skip vacancies
                            continue
                        try:
                            # Get the composition value for this element in this phase
                            val = float(eq.X.sel(component=el, phase=phase_name).values.flatten()[0])
                            comp_dict[el] = val
                        except (KeyError, IndexError):
                            comp_dict[el] = 0.0
                    phase_comps[phase_name] = comp_dict
                except Exception as e:
                    _log.debug(f"Could not extract composition for phase {phase_name}: {e}")
                    # Use current liquid composition as fallback
                    phase_comps[phase_name] = xL.copy()
            
            # Also get current liquid composition
            liquid_comp = {}
            for el in elements:
                if el == 'VA':
                    continue
                try:
                    val = float(eq.X.sel(component=el, phase="LIQUID").values.flatten()[0])
                    liquid_comp[el] = val
                except (KeyError, IndexError):
                    liquid_comp[el] = xL.get(el, 0.0)
            
        except Exception as e:
            _log.debug(f"Phase composition extraction failed at T={T}K: {e}")
            # Fallback: assume solid has same composition as liquid
            phase_comps = {ph: xL.copy() for ph in solid_rel.keys()}
            liquid_comp = xL.copy()
        
        # Mass balance update: remove solid, update liquid composition
        df = min(df_solid, f_liquid)
        f_liquid_new = f_liquid - df
        
        if f_liquid_new < 1e-6:
            # Almost done - add remaining liquid to solids
            for ph, frac in solid_rel.items():
                solid_totals[ph] += f_liquid * frac
            f_liquid = 0.0
            break
        
        # Update liquid composition via mass balance
        new_liquid = {}
        for el in xL.keys():
            if el == 'VA':
                continue
            # Amount removed to solid
            removed = sum(df * solid_rel[ph] * phase_comps[ph].get(el, 0.0) 
                         for ph in solid_rel)
            # Remaining in liquid
            numer = f_liquid * xL[el] - removed
            new_liquid[el] = max(0.0, numer / max(f_liquid_new, 1e-12))
        
        # Normalize liquid composition
        total = sum(new_liquid.values())
        if total > 1e-12:
            xL = {k: v / total for k, v in new_liquid.items()}
        
        # Accumulate solid phases
        for ph, frac in solid_rel.items():
            solid_totals[ph] += df * frac
        
        f_liquid = f_liquid_new
        T -= dT
    
    _log.info(f"Scheil-Gulliver completed after {n_steps} steps, final liquid fraction: {f_liquid:.4f}")
    
    # Normalize solid totals to 1.0
    total_solid = sum(solid_totals.values())
    if total_solid > 1e-6:
        solid_totals = {k: v / total_solid for k, v in solid_totals.items()}
    else:
        _log.warning("No solid phases formed during Scheil-Gulliver simulation")
        # Cache the empty result as well (avoid recomputation)
        if use_cache:
            _save_scheil_to_cache(cache_key, {})
        return {}
    
    # Remove phases with negligible fractions
    solid_totals = {k: v for k, v in solid_totals.items() if v > 1e-4}
    
    # Save to cache before returning
    if use_cache:
        _save_scheil_to_cache(cache_key, solid_totals)
    
    return solid_totals


def simulate_as_cast_microstructure_simple(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float]
) -> Tuple[Dict[str, float], Optional[float]]:
    """
    As-cast microstructure prediction using Scheil-Gulliver solidification.
    
    This uses a more physically accurate Scheil-Gulliver model that assumes:
    - Perfect mixing in the liquid (rapid diffusion)
    - No diffusion in the solid (compositions frozen upon solidification)
    
    This better represents real casting conditions than equilibrium calculations.
    
    Args:
        db: PyCalphad Database
        elements: List of elements
        phases: List of phase names
        composition: Dict of {element: mole_fraction}
        
    Returns:
        Tuple of (phase_fractions dict, T_ascast in K) or ({}, None) on failure
    """
    try:
        # Check if this is a pure element (only one element with fraction > 0.99)
        active_elements = {el: frac for el, frac in composition.items() if frac > 1e-6}
        is_pure_element = len(active_elements) == 1 and list(active_elements.values())[0] > 0.99
        
        if is_pure_element:
            _log.info(f"Pure element detected: {list(active_elements.keys())[0]}")
            _log.info("Pure elements have no solidification range - using equilibrium at 300K")
            
            # For pure elements, just use room temperature equilibrium
            solid_phases = [p for p in phases if p != "LIQUID"]
            
            eq = compute_equilibrium(
                db=db,
                elements=elements,
                phases=solid_phases,
                composition=active_elements,
                temperature=300.0,
                pressure=101325
            )
            
            if eq is None:
                _log.error("Equilibrium calculation failed at 300K for pure element")
                return {}, None
            
            phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
            
            if not phase_fractions:
                _log.error(f"No phases found for pure element at 300K")
                return {}, None
            
            _log.info(f"Pure element equilibrium: {phase_fractions}")
            return phase_fractions, 300.0
        
        # For alloys, find liquidus temperature to set proper starting temperature
        T_liquidus, T_solidus = find_liquidus_solidus_temperatures(
            db, elements, phases, composition, n_points=30
        )
        
        if T_liquidus is None:
            _log.warning("Could not find liquidus temperature")
            T_start = 1200.0  # Default fallback
        else:
            T_start = T_liquidus + 50.0  # Start 50K above liquidus
            _log.info(f"Starting Scheil-Gulliver at {T_start}K (liquidus={T_liquidus}K)")
        
        # Run Scheil-Gulliver solidification simulation
        phase_fractions = simulate_scheil_gulliver(
            db=db,
            elements=elements,
            phases=phases,
            composition=composition,
            T_start=T_start,
            T_end=200.0,
            dT=5.0,
            df_solid=0.01,
            pressure=101325
        )
        
        if not phase_fractions:
            _log.error("Scheil-Gulliver simulation produced no phases")
            return {}, None
        
        # Representative temperature: approximate solidus
        T_ascast = T_solidus if T_solidus else 500.0
        
        _log.info(f"As-cast microstructure (Scheil-Gulliver): {len(phase_fractions)} phases")
        _log.info(f"Phase fractions: {phase_fractions}")
        
        return phase_fractions, T_ascast
        
    except Exception as e:
        _log.error(f"As-cast simulation failed: {e}", exc_info=True)
        return {}, None