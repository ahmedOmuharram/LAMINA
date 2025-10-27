"""
Solidification simulation utilities for CALPHAD.

Implements Scheil-Gulliver style solidification modeling to predict
as-cast microstructures, not infinite-time equilibrium.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pycalphad import Database, equilibrium
import pycalphad.variables as v

from .equilibrium_utils import extract_phase_fractions_from_equilibrium

_log = logging.getLogger(__name__)


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
        # Setting X(element) = 0.0 exactly can cause pycalphad failures
        active_elements = {el: frac for el, frac in composition.items() if frac > 1e-6}
        comp_elements = list(active_elements.keys())
        
        # Build element list for pycalphad (only active elements + VA)
        # This ensures degrees of freedom match the composition constraints
        elements_for_eq = comp_elements + ['VA']
        
        _log.debug(f"Finding liquidus/solidus with active elements: {comp_elements}")
        
        temperatures = np.linspace(T_start, T_end, n_points)
        
        T_liquidus = None
        T_solidus = None
        
        for T in temperatures:
            try:
                # Build conditions (N-1 composition constraints)
                conditions = {
                    v.T: T,
                    v.P: 101325,
                    v.N: 1.0
                }
                
                if len(comp_elements) > 1:
                    for el in comp_elements[1:]:
                        conditions[v.X(el)] = active_elements[el]
                
                eq = equilibrium(db, elements_for_eq, phases, conditions)
                phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                
                liquid_frac = phase_fractions.get("LIQUID", 0.0)
                
                # Liquidus: highest T where liquid < 1.0 (some solid forms)
                if liquid_frac < 0.99 and T_liquidus is None:
                    T_liquidus = T
                
                # Solidus: lowest T where liquid > 0.01 (last liquid disappears below this)
                if liquid_frac > 0.01:
                    T_solidus = T
                    
            except Exception as e:
                _log.debug(f"Equilibrium failed at T={T}K: {e}")
                continue
        
        _log.info(f"Found liquidus~{T_liquidus}K, solidus~{T_solidus}K")
        return T_liquidus, T_solidus
        
    except Exception as e:
        _log.error(f"Error finding liquidus/solidus: {e}")
        return None, None


def simulate_as_cast_microstructure_simple(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float]
) -> Tuple[Dict[str, float], Optional[float]]:
    """
    Simple as-cast microstructure prediction: equilibrium just below solidus.
    
    This is a simplified approach that:
    1. Finds the solidus temperature
    2. Runs equilibrium ~20K below solidus (after last liquid freezes)
    3. Excludes LIQUID from phases
    
    This approximates "what you get after solidification finishes" without
    the infinite solid-state diffusion time that 300K equilibrium assumes.
    
    Args:
        db: PyCalphad Database
        elements: List of elements
        phases: List of phase names
        composition: Dict of {element: mole_fraction}
        
    Returns:
        Tuple of (phase_fractions dict, T_ascast in K) or ({}, None) on failure
    """
    try:
        # Find solidus
        T_liquidus, T_solidus = find_liquidus_solidus_temperatures(
            db, elements, phases, composition
        )
        
        if T_solidus is None:
            _log.warning("Could not find solidus temperature via liquidus/solidus sweep")
            _log.warning("Falling back to direct equilibrium at 500K")
            # Fallback: just run equilibrium at a reasonable temperature
            T_ascast = 500.0
        else:
            # Equilibrate just below solidus (after last liquid is gone)
            T_ascast = T_solidus - 20.0  # 20K below solidus
        
        # Remove LIQUID from phases for this calculation
        solid_phases = [p for p in phases if p != "LIQUID"]
        
        _log.info(f"Simulating as-cast microstructure at T={T_ascast}K (solidus={T_solidus}K)")
        
        # Filter out elements with negligible composition (< 1e-6)
        # Setting X(element) = 0.0 exactly can cause pycalphad failures
        active_elements = {el: frac for el, frac in composition.items() if frac > 1e-6}
        comp_elements = list(active_elements.keys())
        
        # Build element list for pycalphad (only active elements + VA)
        # This ensures degrees of freedom match the composition constraints
        elements_for_eq = comp_elements + ['VA']
        
        _log.debug(f"Active elements: {comp_elements}, pycalphad elements: {elements_for_eq}")
        
        # Build conditions
        conditions = {
            v.T: T_ascast,
            v.P: 101325,
            v.N: 1.0
        }
        
        if len(comp_elements) > 1:
            for el in comp_elements[1:]:
                conditions[v.X(el)] = active_elements[el]
        
        eq = equilibrium(db, elements_for_eq, solid_phases, conditions)
        phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
        
        _log.info(f"As-cast equilibrium: {len(phase_fractions)} phases at {T_ascast:.1f}K")
        if not phase_fractions:
            _log.error(f"No phases found in as-cast equilibrium at T={T_ascast}K, composition={composition}")
        
        return phase_fractions, T_ascast
        
    except Exception as e:
        _log.error(f"As-cast simulation failed: {e}")
        return {}, None


def mechanical_desirability_score(
    phase_fractions: Dict[str, float],
    phase_categories: Dict[str, str]
) -> Tuple[float, str]:
    """
    Heuristic mechanical desirability assessment for cast alloys.
    
    Rules of thumb:
    - High FCC matrix (>85%) with modest intermetallics (<15%): good ductility/toughness → +1
    - Very high intermetallic content (>20%): likely brittle → -1
    - High Laves/continuous network phases (>15%): poor ductility → -1
    - Otherwise: mixed/depends → 0
    
    Args:
        phase_fractions: Dict of {phase_name: fraction}
        phase_categories: Dict of {phase_name: category_string}
        
    Returns:
        (score: -1/0/+1, interpretation: str)
    """
    # Sum fractions by category
    fcc_frac = sum(frac for phase, frac in phase_fractions.items() 
                   if phase_categories.get(phase, "").lower() in ["primary_fcc", "fcc"])
    
    tau_frac = sum(frac for phase, frac in phase_fractions.items()
                   if phase_categories.get(phase, "").lower() in ["tau", "tau_phase"])
    
    laves_frac = sum(frac for phase, frac in phase_fractions.items()
                     if phase_categories.get(phase, "").lower() == "laves")
    
    gamma_frac = sum(frac for phase, frac in phase_fractions.items()
                     if phase_categories.get(phase, "").lower() == "gamma")
    
    # Total intermetallic content
    intermetallic_total = tau_frac + laves_frac + gamma_frac
    
    # Scoring logic
    if fcc_frac > 0.85 and intermetallic_total < 0.15:
        score = +1.0
        interpretation = "Likely acceptable: High ductile FCC matrix with modest intermetallic strengthening"
    
    elif intermetallic_total > 0.25:
        score = -1.0
        interpretation = "Likely undesirable: Excessive intermetallic content reduces ductility/toughness"
    
    elif laves_frac > 0.15:
        score = -1.0
        interpretation = "Likely undesirable: High Laves fraction tends to form brittle networks"
    
    elif tau_frac > 0.20:
        score = -0.5
        interpretation = "Marginal: High tau content may reduce ductility depending on morphology"
    
    elif fcc_frac > 0.75 and intermetallic_total < 0.20:
        score = +0.5
        interpretation = "Generally acceptable: Good matrix with moderate intermetallic content"
    
    else:
        score = 0.0
        interpretation = "Mixed: Depends on exact morphology and application requirements"
    
    return score, interpretation

