"""
Equilibrium calculation utilities for CALPHAD phase diagrams.

Functions for calculating equilibrium states at specific points or grids.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pycalphad import Database, equilibrium
import pycalphad.variables as v

_log = logging.getLogger(__name__)


def get_phase_fraction(phases_dict: Dict[str, float], target: str) -> float:
    """
    Sum phase fractions by base name, handling PyCalphad instance suffixes.
    
    PyCalphad often labels phase instances as SIC#1, AL4C3#2, etc.
    This function sums all instances with the same base name.
    
    Args:
        phases_dict: Dictionary of phase_name: fraction from equilibrium
        target: Base phase name to look for (e.g., 'SIC', 'AL4C3')
        
    Returns:
        Total fraction summed across all instances
        
    Example:
        phases = {'SIC#1': 0.3, 'SIC#2': 0.2, 'FCC_A1': 0.5}
        get_phase_fraction(phases, 'SIC')  # Returns 0.5
    """
    base = target.upper()
    tot = 0.0
    for k, v in phases_dict.items():
        if k is None: 
            continue
        # Split by '#' to get base name, compare case-insensitively
        if str(k).split('#')[0].upper() == base:
            tot += float(v)
    return tot


def extract_phase_fractions_from_equilibrium(eq: Any, tolerance: float = 1e-4) -> Dict[str, float]:
    """
    Properly extract phase fractions from equilibrium result, handling multiple vertices.
    
    In two-phase regions, equilibrium returns multiple vertices; we need to sum NP 
    over the vertex dimension per phase (not rely on 1-to-1 raveled index).
    
    Args:
        eq: Equilibrium result from pycalphad
        tolerance: Minimum fraction to include (default 1e-4 for better boundary handling)
        
    Returns:
        Dictionary mapping phase names to fractions
    """
    try:
        if not hasattr(eq, 'Phase') or not hasattr(eq, 'NP'):
            return {}
        
        # Squeeze singleton coords first
        eqp = eq.squeeze()
        
        # Group by phase and sum over vertex dimension
        vertex_dims = [d for d in eqp['NP'].dims if d in ('vertex',)]
        
        if vertex_dims:
            # Use xarray groupby to sum fractions per phase across vertices
            frac_by_phase = (
                eqp['NP']
                .groupby(eqp['Phase'])
                .sum(dim=vertex_dims)
            )
        else:
            # No vertex dimension, simple extraction
            frac_by_phase = eqp['NP'].groupby(eqp['Phase']).sum()
        
        # Convert to dictionary and filter by tolerance
        phase_fractions = {}
        for phase in frac_by_phase.coords['Phase'].values:
            if not phase or phase == '':
                continue
            
            frac = float(frac_by_phase.sel(Phase=phase).values)
            
            # Use slightly looser tolerance for reporting (handles boundary noise better)
            if not np.isnan(frac) and frac > tolerance:
                phase_fractions[str(phase)] = frac
        
        return phase_fractions
        
    except Exception as e:
        _log.warning(f"Error extracting phase fractions: {e}")
        # Fallback to simple approach
        try:
            phase_fractions = {}
            phase_array = eq.Phase.values
            np_array = eq.NP.values
            
            for phase in np.unique(phase_array):
                if phase == '' or not isinstance(phase, str):
                    continue
                    
                mask = (phase_array == phase)
                fraction = np.sum(np_array[mask])
                
                if not np.isnan(fraction) and fraction > tolerance:
                    phase_fractions[str(phase)] = float(fraction)
            
            return phase_fractions
        except:
            return {}


def calculate_equilibrium_at_point(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    temperature: float,
    pressure: float = 101325
) -> Optional[Any]:
    """
    Calculate equilibrium at a specific point.
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols
        phases: List of phase names to consider
        composition: Dictionary of element: mole_fraction
        temperature: Temperature in K
        pressure: Pressure in Pa (default: 101325)
        
    Returns:
        Equilibrium result or None if calculation fails
    """
    try:
        # Build conditions
        conditions = {
            v.T: temperature,
            v.P: pressure,
            v.N: 1.0
        }
        
        # Add composition conditions
        for el, frac in composition.items():
            conditions[v.X(el)] = frac
        
        # Calculate equilibrium
        eq = equilibrium(db, elements, phases, conditions)
        
        return eq
        
    except Exception as e:
        _log.error(f"Equilibrium calculation failed: {e}")
        return None


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
                if phase == '' or not isinstance(phase, str):
                    continue
                    
                mask = (phase_array == phase)
                fractions = phase_fraction[mask]
                
                if np.any(fractions > 1e-6):  # Threshold for "present"
                    stable_phases.append(phase)
        
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
    eq = calculate_equilibrium_at_point(db, elements, phases, composition, temperature)
    
    if eq is None:
        return {}
    
    try:
        phase_fractions = {}
        
        if hasattr(eq, 'Phase') and hasattr(eq, 'NP'):
            phase_array = eq.Phase.values
            np_array = eq.NP.values
            
            for phase in np.unique(phase_array):
                if phase == '' or not isinstance(phase, str):
                    continue
                    
                mask = (phase_array == phase)
                fraction = np.sum(np_array[mask])
                
                if fraction > 1e-6:
                    phase_fractions[phase] = float(fraction)
        
        return phase_fractions
        
    except Exception as e:
        _log.error(f"Error calculating phase fractions: {e}")
        return {}

