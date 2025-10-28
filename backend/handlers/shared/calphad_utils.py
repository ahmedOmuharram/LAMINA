"""
Shared CALPHAD utilities for thermodynamic calculations.

This module consolidates common CALPHAD operations used across multiple handlers:
- Database loading with fallback patterns
- Equilibrium calculations at specific conditions
- Phase fraction extraction
- Phase composition extraction
- Phase name parsing and classification

Used by:
- handlers/alloys/alloy_handler.py
- handlers/solutes/ai_functions.py
- handlers/calphad/phase_diagrams/* (various modules)
"""
import logging
import re
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np

try:
    from pycalphad import Database, equilibrium
    import pycalphad.variables as v
except ImportError:
    Database = None
    equilibrium = None
    v = None

_log = logging.getLogger(__name__)


# =============================================================================
# DATABASE LOADING UTILITIES
# =============================================================================

def find_tdb_database(
    system_elements: List[str],
    tdb_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Find appropriate TDB database for given elements.
    
    This implements a common pattern used across handlers for locating
    thermodynamic databases with various naming conventions.
    
    Args:
        system_elements: List of element symbols (e.g., ['AL', 'FE'])
        tdb_dir: Directory to search for TDB files. If None, uses default location.
        
    Returns:
        Path to TDB file, or None if not found
        
    Example:
        >>> db_path = find_tdb_database(['AL', 'MG'])
        >>> if db_path:
        ...     db = Database(str(db_path))
    """
    if tdb_dir is None:
        # Default TDB directory (adjust based on project structure)
        tdb_dir = Path(__file__).parent.parent.parent.parent / "tdbs"
    
    if not tdb_dir.exists():
        _log.warning(f"TDB directory does not exist: {tdb_dir}")
        return None
    
    # Normalize element symbols to uppercase
    elements_upper = [el.upper() for el in system_elements]
    system_str = "-".join(elements_upper)
    
    # Try multiple naming patterns (most specific to most general)
    possible_patterns = []
    
    # 1. Exact system match (e.g., "AL-FE.tdb")
    possible_patterns.append(f"{system_str}.tdb")
    
    # 2. Reverse order (e.g., "FE-AL.tdb" if we're looking for AL-FE)
    if len(elements_upper) == 2:
        reverse_system = f"{elements_upper[1]}-{elements_upper[0]}"
        possible_patterns.append(f"{reverse_system}.tdb")
    
    # 3. Materials Commons format (e.g., "mc_al_v2037_pycal.tdb")
    for elem in elements_upper:
        possible_patterns.append(f"mc_{elem.lower()}_*_pycal.tdb")
    
    # 4. COST507 database (common for many alloy systems)
    possible_patterns.append("COST507.tdb")
    possible_patterns.append("cost507.tdb")
    
    # Try to find matching files
    for pattern in possible_patterns:
        if '*' in pattern:
            # Use glob for patterns with wildcards
            matches = list(tdb_dir.glob(pattern))
            if matches:
                _log.info(f"Found TDB for {system_str}: {matches[0].name}")
                return matches[0]
        else:
            # Direct path check
            candidate = tdb_dir / pattern
            if candidate.exists():
                _log.info(f"Found TDB for {system_str}: {pattern}")
                return candidate
    
    # Fallback: return first available .tdb file
    candidates = list(tdb_dir.glob("*.tdb"))
    if candidates:
        _log.warning(f"No specific TDB for {system_str}, using fallback: {candidates[0].name}")
        return candidates[0]
    
    _log.error(f"No TDB database found for {system_str} in {tdb_dir}")
    return None


def load_tdb_database(
    system_elements: List[str],
    tdb_dir: Optional[Path] = None
) -> Optional[Database]:
    """
    Load TDB database for given element system.
    
    Combines database finding and loading in one step.
    
    Args:
        system_elements: List of element symbols
        tdb_dir: Optional directory containing TDB files
        
    Returns:
        PyCalphad Database instance or None if loading fails
        
    Example:
        >>> db = load_tdb_database(['AL', 'ZN'])
        >>> if db:
        ...     print(f"Loaded database with elements: {db.elements}")
    """
    if Database is None:
        _log.error("pycalphad not installed")
        return None
    
    db_path = find_tdb_database(system_elements, tdb_dir)
    if not db_path:
        return None
    
    try:
        db = Database(str(db_path))
        _log.info(f"Loaded database: {db_path.name}")
        return db
    except Exception as e:
        _log.error(f"Failed to load database {db_path}: {e}")
        return None


# =============================================================================
# EQUILIBRIUM CALCULATION UTILITIES
# =============================================================================

def compute_equilibrium(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    temperature: float,
    pressure: float = 101325
) -> Optional[Any]:
    """
    Calculate equilibrium at a specific point.
    
    Unified equilibrium calculation used across multiple handlers.
    Handles composition constraints correctly for multi-component systems.
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols (VA will be added automatically)
        phases: List of phase names to consider
        composition: Dictionary of element: mole_fraction (should sum to 1.0)
        temperature: Temperature in K
        pressure: Pressure in Pa (default: 101325)
        
    Returns:
        Equilibrium result xarray Dataset or None if calculation fails
        
    Example:
        >>> eq = compute_equilibrium(
        ...     db, ['AL', 'ZN'], phases,
        ...     {'AL': 0.3, 'ZN': 0.7},
        ...     temperature=700
        ... )
    """
    if equilibrium is None or v is None:
        _log.error("pycalphad not installed")
        return None
    
    try:
        # Ensure VA is in elements list (required for many phases)
        elements_with_va = list(elements)
        if 'VA' not in elements_with_va:
            elements_with_va.append('VA')
        
        # Build conditions
        conditions = {
            v.T: temperature,
            v.P: pressure,
            v.N: 1.0
        }
        
        # Add composition conditions (N-1 independent constraints for N components)
        # Skip the first element as it's the dependent variable
        comp_elements = list(composition.keys())
        if len(comp_elements) > 1:
            # For multicomponent systems, specify all but the first element
            for el in comp_elements[1:]:
                conditions[v.X(el)] = composition[el]
        
        # Calculate equilibrium
        _log.debug(f"Running equilibrium: T={temperature}K, composition={composition}")
        eq = equilibrium(db, elements_with_va, phases, conditions)
        
        return eq
        
    except Exception as e:
        _log.error(f"Equilibrium calculation failed: {e}")
        _log.error(f"  Elements: {elements_with_va}")
        _log.error(f"  Temperature: {temperature}K")
        _log.error(f"  Composition: {composition}")
        return None


def extract_phase_fractions(eq: Any, tolerance: float = 1e-4) -> Dict[str, float]:
    """
    Extract phase fractions from equilibrium result.
    
    Properly handles multi-vertex equilibrium results (e.g., two-phase regions)
    by summing phase fractions across vertices.
    
    NOTE: This is a wrapper around the more sophisticated implementation in
    handlers/calphad/phase_diagrams/equilibrium_utils.py. If you need more
    control, use that module directly.
    
    Args:
        eq: Equilibrium result from pycalphad
        tolerance: Minimum fraction to include (default 1e-4)
        
    Returns:
        Dictionary mapping phase names to fractions
        
    Example:
        >>> eq = compute_equilibrium(...)
        >>> fractions = extract_phase_fractions(eq)
        >>> print(fractions)  # {'FCC_A1': 0.7, 'AL2FE': 0.3}
    """
    try:
        if not hasattr(eq, 'Phase') or not hasattr(eq, 'NP'):
            return {}
        
        # Squeeze singleton coords first
        eqp = eq.squeeze()
        
        # Group by phase and sum over vertex dimension if present
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


def get_phase_composition(
    eq: Any,
    phase_name: str,
    elements: List[str]
) -> Dict[str, float]:
    """
    Extract composition of a specific phase from equilibrium result.
    
    Returns the mole fractions of each element within the specified phase.
    
    Args:
        eq: Equilibrium result from pycalphad
        phase_name: Name of phase to extract composition for
        elements: List of element symbols to extract
        
    Returns:
        Dictionary of element: mole_fraction in the phase
        
    Example:
        >>> composition = get_phase_composition(eq, 'FCC_A1', ['AL', 'MG'])
        >>> print(composition)  # {'AL': 0.96, 'MG': 0.04}
    """
    try:
        eqp = eq.squeeze()
        phase_mask = eqp['Phase'] == phase_name
        
        phase_composition = {}
        for elem in elements:
            try:
                x_data = eqp['X'].sel(component=elem).where(phase_mask, drop=False)
                x_val = float(x_data.mean().values)
                if x_val > 1e-6:  # Only record non-negligible amounts
                    phase_composition[elem] = x_val
            except Exception as e:
                _log.debug(f"Could not extract {elem} composition from {phase_name}: {e}")
                continue
        
        # Normalize to sum to 1.0
        total_comp = sum(phase_composition.values())
        if total_comp > 0:
            phase_composition = {
                elem: frac / total_comp 
                for elem, frac in phase_composition.items()
            }
        
        return phase_composition
        
    except Exception as e:
        _log.error(f"Error extracting phase composition: {e}")
        return {}


# =============================================================================
# PHASE NAME PARSING UTILITIES
# =============================================================================

def parse_calphad_phase_name(
    phase_name: str,
    system_elements: Tuple[str, ...]
) -> Optional[str]:
    """
    Parse CALPHAD phase name to chemical formula.
    
    Converts phase names like "AL2FE" to "Al2Fe" or "AL5FE2" to "Al5Fe2".
    Skips solid solution phases like "FCC_A1", "BCC_A2", etc.
    
    Args:
        phase_name: CALPHAD phase name (e.g., "AL2FE", "FCC_A1")
        system_elements: Tuple of element symbols in the system
        
    Returns:
        Chemical formula string or None if not applicable
        
    Example:
        >>> parse_calphad_phase_name("AL2FE", ("Al", "Fe"))
        'Al2Fe'
        >>> parse_calphad_phase_name("FCC_A1", ("Al", "Cu"))
        None  # Solid solution, not a compound
    """
    try:
        phase_upper = phase_name.upper()
        
        # Skip solid solution phases
        solid_solution_patterns = [
            "BCC", "FCC", "HCP", "LIQUID", 
            "_A1", "_A2", "_A3", "_B2", "_B1",
            "DIAMOND", "GRAPHITE"
        ]
        if any(pattern in phase_upper for pattern in solid_solution_patterns):
            return None
        
        # Try to parse element-number patterns
        # Replace element symbols with proper case
        formula = phase_upper
        for elem in system_elements:
            elem_upper = elem.upper()
            # Replace with proper case (e.g., AL -> Al, FE -> Fe)
            formula = re.sub(
                elem_upper, 
                elem.capitalize(), 
                formula, 
                flags=re.IGNORECASE
            )
        
        # Remove common phase structure suffixes
        structure_suffixes = [
            "_D03", "_L12", "_C15", "_DELTA", "_C14", "_C36",
            "_B82", "_A15", "_C11B", "_DO3"
        ]
        for suffix in structure_suffixes:
            formula = formula.replace(suffix, "")
        
        # Validate it looks like a formula (has both letters and numbers)
        if any(char.isdigit() for char in formula) and any(char.isalpha() for char in formula):
            return formula
        
        return None
        
    except Exception as e:
        _log.debug(f"Error parsing phase name {phase_name}: {e}")
        return None


def classify_phase_type(phase_name: str) -> str:
    """
    Classify CALPHAD phase by type.
    
    Args:
        phase_name: Phase name from database
        
    Returns:
        One of: 'liquid', 'solid_solution', 'intermetallic', 'terminal', 'unknown'
        
    Example:
        >>> classify_phase_type("LIQUID")
        'liquid'
        >>> classify_phase_type("FCC_A1")
        'solid_solution'
        >>> classify_phase_type("AL2FE")
        'intermetallic'
    """
    phase_upper = phase_name.upper()
    
    # Liquid phase
    if "LIQUID" in phase_upper:
        return "liquid"
    
    # Solid solution phases (BCC, FCC, HCP, etc.)
    solid_solution_patterns = [
        "BCC_A2", "FCC_A1", "HCP_A3", 
        "BCC_B2", "DIAMOND_A4",
        "_A1", "_A2", "_A3", "_B2"
    ]
    if any(pattern in phase_upper for pattern in solid_solution_patterns):
        return "solid_solution"
    
    # Terminal phases (pure element structures)
    terminal_patterns = [
        "DIAMOND", "GRAPHITE", "SIGMA"
    ]
    if any(pattern in phase_upper for pattern in terminal_patterns):
        return "terminal"
    
    # Intermetallic compounds (contain element symbols and numbers)
    if any(char.isdigit() for char in phase_name):
        return "intermetallic"
    
    return "unknown"


def get_phase_fraction_by_base_name(
    phases_dict: Dict[str, float],
    target: str
) -> float:
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
        >>> phases = {'SIC#1': 0.3, 'SIC#2': 0.2, 'FCC_A1': 0.5}
        >>> get_phase_fraction_by_base_name(phases, 'SIC')
        0.5
    """
    base = target.upper()
    total = 0.0
    for k, v in phases_dict.items():
        if k is None:
            continue
        # Split by '#' to get base name, compare case-insensitively
        if str(k).split('#')[0].upper() == base:
            total += float(v)
    return total


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def verify_elements_in_database(
    db: Database,
    elements: List[str]
) -> Tuple[bool, List[str]]:
    """
    Verify that all required elements exist in the database.
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols to verify
        
    Returns:
        Tuple of (all_present: bool, missing: List[str])
        
    Example:
        >>> db = Database('AL-ZN.tdb')
        >>> ok, missing = verify_elements_in_database(db, ['AL', 'ZN', 'MG'])
        >>> if not ok:
        ...     print(f"Missing elements: {missing}")
    """
    # Get elements from database (excluding VA)
    db_elements = {el.upper() for el in getattr(db, "elements", set()) if el.upper() != "VA"}
    
    # Normalize input elements
    elements_upper = [el.upper() for el in elements]
    
    # Find missing elements
    missing = [el for el in elements_upper if el not in db_elements]
    
    return (len(missing) == 0, missing)


def normalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize composition dictionary to sum to 1.0.
    
    Args:
        composition: Dictionary of element: fraction
        
    Returns:
        Normalized composition dictionary
        
    Example:
        >>> normalize_composition({'AL': 30, 'ZN': 70})
        {'AL': 0.3, 'ZN': 0.7}
    """
    total = sum(composition.values())
    if total == 0:
        return composition
    
    return {
        elem: frac / total 
        for elem, frac in composition.items()
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_equilibrium_microstructure(
    system_elements: List[str],
    composition: Dict[str, float],
    temperature: float,
    tdb_dir: Optional[Path] = None,
    pressure: float = 101325
) -> Dict[str, Any]:
    """
    High-level convenience function for equilibrium calculation.
    
    Combines database loading, phase filtering, equilibrium calculation,
    and result extraction into one step.
    
    Args:
        system_elements: List of element symbols
        composition: Dictionary of element: mole_fraction
        temperature: Temperature in K
        tdb_dir: Optional TDB directory
        pressure: Pressure in Pa
        
    Returns:
        Dictionary with keys:
        - success: bool
        - phases: List[Dict] with name and fraction
        - matrix_phase: str (phase with largest fraction)
        - matrix_phase_composition: Dict[str, float]
        - secondary_phases: List[Dict]
        - error: str (if success=False)
        
    Example:
        >>> result = compute_equilibrium_microstructure(
        ...     ['AL', 'ZN'],
        ...     {'AL': 0.5, 'ZN': 0.5},
        ...     temperature=700
        ... )
        >>> if result['success']:
        ...     print(f"Matrix phase: {result['matrix_phase']}")
    """
    try:
        # Load database
        db = load_tdb_database(system_elements, tdb_dir)
        if db is None:
            return {
                "success": False,
                "error": f"Could not load database for {system_elements}",
                "phases": [],
                "matrix_phase": None,
                "matrix_phase_composition": {},
                "secondary_phases": []
            }
        
        # Verify elements
        ok, missing = verify_elements_in_database(db, system_elements)
        if not ok:
            return {
                "success": False,
                "error": f"Elements {missing} not found in database",
                "phases": [],
                "matrix_phase": None,
                "matrix_phase_composition": {},
                "secondary_phases": []
            }
        
        # Get all phases (simplified - for more control, use CalPhadHandler)
        phases = list(db.phases.keys())
        
        # Normalize composition
        composition = normalize_composition(composition)
        
        # Calculate equilibrium
        eq = compute_equilibrium(
            db, system_elements, phases, 
            composition, temperature, pressure
        )
        
        if eq is None:
            return {
                "success": False,
                "error": "Equilibrium calculation failed",
                "phases": [],
                "matrix_phase": None,
                "matrix_phase_composition": {},
                "secondary_phases": []
            }
        
        # Extract phase fractions
        phase_fractions = extract_phase_fractions(eq, tolerance=1e-4)
        
        if not phase_fractions:
            return {
                "success": False,
                "error": "No stable phases found",
                "phases": [],
                "matrix_phase": None,
                "matrix_phase_composition": {},
                "secondary_phases": []
            }
        
        # Build phase list
        phase_list = [
            {"name": phase, "fraction": frac}
            for phase, frac in sorted(phase_fractions.items(), key=lambda x: -x[1])
            if frac > 0.01
        ]
        
        # Matrix is the phase with maximum fraction
        matrix_phase = max(phase_fractions.items(), key=lambda x: x[1])[0]
        
        # Extract matrix phase composition
        matrix_phase_composition = get_phase_composition(eq, matrix_phase, system_elements)
        
        # Secondary phases
        secondary_phases = [
            {"name": phase, "fraction": frac}
            for phase, frac in phase_fractions.items()
            if phase != matrix_phase and frac > 0.01
        ]
        
        return {
            "success": True,
            "phases": phase_list,
            "matrix_phase": matrix_phase,
            "matrix_phase_composition": matrix_phase_composition,
            "secondary_phases": secondary_phases,
            "phase_fractions": phase_fractions
        }
        
    except Exception as e:
        _log.error(f"Error in compute_equilibrium_microstructure: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "phases": [],
            "matrix_phase": None,
            "matrix_phase_composition": {},
            "secondary_phases": []
        }

