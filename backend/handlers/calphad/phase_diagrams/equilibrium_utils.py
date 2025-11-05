"""
Equilibrium calculation utilities for CALPHAD phase diagrams.

Functions for calculating equilibrium states at specific points or grids.

NOTE: For equilibrium calculations and phase fraction extraction, use the functions from
shared.calphad_utils (compute_equilibrium, extract_phase_fractions, get_phase_fraction).
This module contains only grid-based equilibrium calculations and other specialized functions.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pycalphad import Database, equilibrium
import pycalphad.variables as v

_log = logging.getLogger(__name__)

def calculate_coarse_equilibrium_grid(
    db: Database,
    A: str,
    B: str,
    phases: List[str],
    T_range: Tuple[float, float],
    nx: int = 101,
    nT: int = 161
) -> Optional[Any]:
    """
    Calculate a coarse equilibrium grid for phase diagram analysis.
    
    Args:
        db: PyCalphad Database instance
        A: First element symbol
        B: Second element symbol  
        phases: List of phase names
        T_range: Temperature range as (min, max) in K
        nx: Number of composition points
        nT: Number of temperature points
        
    Returns:
        Equilibrium result or None if calculation fails
    """
    try:
        elements = [A, B, 'VA']
        T_lo, T_hi = T_range
        
        # Build callables for better performance if available
        eq_kwargs = {}
        
        eq = equilibrium(
            db, elements, phases,
            {
                v.X(B): (0, 1, nx),
                v.T: (T_lo, T_hi, nT),
                v.P: 101325,
                v.N: 1
            },
            **eq_kwargs
        )
        
        return eq
        
    except Exception as e:
        _log.error(f"Coarse equilibrium grid calculation failed: {e}")
        return None


def extract_stable_phases(eq: Any, composition: float, temperature: float) -> List[str]:
    """
    Extract list of stable phases at a given point.
    
    Args:
        eq: Equilibrium result
        composition: Composition value (mole fraction)
        temperature: Temperature in K
        
    Returns:
        List of stable phase names
    """
    try:
        # Find nearest point in equilibrium data
        # This is a simplified version - actual implementation would need
        # proper interpolation or nearest-neighbor search
        
        stable_phases = []
        
        # Extract phase data
        if hasattr(eq, 'Phase') and hasattr(eq, 'NP'):
            phase_array = eq.Phase.values
            phase_fraction = eq.NP.values
            
            # Get phases with non-zero fraction
            for phase in np.unique(phase_array):
                # Convert numpy.str_ to regular str to avoid type errors
                phase_str = str(phase)
                if phase_str == '':
                    continue
                    
                mask = (phase_array == phase)
                fractions = phase_fraction[mask]
                
                if np.any(fractions > 1e-6):  # Threshold for "present"
                    stable_phases.append(phase_str)
        
        return stable_phases
        
    except Exception as e:
        _log.error(f"Error extracting stable phases: {e}")
        return []


def calculate_phase_fractions_at_temperature(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    temperature: float
) -> Dict[str, float]:
    """
    Calculate phase fractions at a specific temperature.
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols
        phases: List of phase names
        composition: Dictionary of element: mole_fraction
        temperature: Temperature in K
        
    Returns:
        Dictionary of phase_name: fraction
    """
    from ...shared.calphad_utils import compute_equilibrium, extract_phase_fractions
    
    eq = compute_equilibrium(db, elements, phases, composition, temperature)
    
    if eq is None:
        return {}
    
    # Use shared phase fraction extraction (handles vertex summing properly)
    return extract_phase_fractions(eq, tolerance=1e-6)

