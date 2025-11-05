"""
Shared CALPHAD utilities for thermodynamic calculations.

This module consolidates common CALPHAD operations used across multiple handlers:
- Database loading with fallback patterns and element-based selection
- Equilibrium calculations at specific conditions (point and grid)
- Phase fraction extraction (with vertex handling for two-phase regions)
- Phase composition extraction
- Phase name parsing and classification
- Element normalization and composition parsing

Used by:
- handlers/alloys/alloy_handler.py
- handlers/solutes/ai_functions.py
- handlers/calphad/phase_diagrams/* (various modules)

NOTE: This module maintains BOTH specialized and generalized versions of functions
where different handlers require different behaviors.
"""
import logging
import re
from typing import Optional, Dict, List, Tuple, Any, Set
from pathlib import Path
import numpy as np

try:
    from pycalphad import Database
    import pycalphad.variables as v
except ImportError:
    Database = None
    _pycalphad_equilibrium_original = None
    v = None

import warnings
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from pycalphad.core.workspace import Workspace
from pycalphad.core.light_dataset import LightDataset
import numpy as np
from pycalphad.property_framework import as_property


_log = logging.getLogger(__name__)


# =============================================================================
# DATABASE LOADING UTILITIES
# =============================================================================

def find_tdb_database(
    system_elements: List[str]
) -> Optional[Path]:
    """
    Find appropriate TDB database for given elements with element-based selection.
    
    This implements an enhanced pattern that:
    1. Tries exact system matches first
    2. Uses element-specific databases (e.g., COST507 for Al-Mg-Zn, C, N, B, Li)
    3. Falls back to Materials Commons and general databases
    
    This provides comprehensive database selection with element-based matching.
    
    Args:
        system_elements: List of element symbols (e.g., ['AL', 'FE'])
        
    Returns:
        Path to TDB file, or None if not found
        
    Example:
        >>> db_path = find_tdb_database(['AL', 'MG'])
        >>> if db_path:
        ...     db = Database(str(db_path))
    """
    # TDB directory location
    tdb_dir = Path(__file__).parent.parent.parent / "tdbs"
    
    if not tdb_dir.exists():
        _log.warning(f"TDB directory does not exist: {tdb_dir}")
        return None
    
    # Get all available .tdb files
    candidates = sorted([p for p in tdb_dir.glob("*.tdb") if p.is_file()])
    if not candidates:
        _log.warning(f"No .tdb files found in {tdb_dir}")
        return None
    
    elements_upper = [el.strip().upper() for el in system_elements]
    elements_set = set(elements_upper)
    system_str = "-".join(elements_upper)
    
    # =========================================================================
    # PHASE 1: Element-specific database selection
    # =========================================================================
    
    # COST507.tdb for Al-Mg-Zn ternary system (has clean tau phase data)
    if elements_set == {'AL', 'MG', 'ZN'} or elements_set <= {'AL', 'MG', 'ZN', 'VA'}:
        for p in candidates:
            if "COST507" in p.name or "cost507" in p.name.lower():
                _log.info(f"Selected {p.name} for Al-Mg-Zn system")
                return p
    
    # COST507.tdb for systems with C, N, B, or Li
    if any(el in elements_upper for el in ['C', 'N', 'B', 'LI']):
        for p in candidates:
            if "COST507" in p.name or "cost507" in p.name.lower():
                _log.info(f"Selected {p.name} for elements {elements_upper}")
                return p
    
    # =========================================================================
    # PHASE 2: Exact system match patterns
    # =========================================================================
    
    # 1. Exact system match (e.g., "AL-FE.tdb")
    exact_match = tdb_dir / f"{system_str}.tdb"
    if exact_match.exists():
        _log.info(f"Found exact match TDB: {exact_match.name}")
        return exact_match
    
    # 2. Reverse order (e.g., "FE-AL.tdb" if we're looking for AL-FE)
    if len(elements_upper) == 2:
        reverse_system = f"{elements_upper[1]}-{elements_upper[0]}"
        reverse_match = tdb_dir / f"{reverse_system}.tdb"
        if reverse_match.exists():
            _log.info(f"Found reverse match TDB: {reverse_match.name}")
            return reverse_match
    
    # 3. Materials Commons format (e.g., "mc_al_v2037_pycal.tdb")
    for elem in elements_upper:
        matches = list(tdb_dir.glob(f"mc_{elem.lower()}_*_pycal.tdb"))
        if matches:
            _log.info(f"Found Materials Commons TDB for {elem}: {matches[0].name}")
            return matches[0]
    
    # =========================================================================
    # PHASE 3: Default/fallback selection
    # =========================================================================
    
    # Prefer pycal versions for Al systems
    for p in candidates:
        if "pycal" in p.name.lower() and "al" in p.name.lower():
            _log.info(f"Selected {p.name} (default Al database)")
            return p
    
    # Final fallback: return first available .tdb file
    _log.warning(f"No specific TDB for {system_str}, using fallback: {candidates[0].name}")
    return candidates[0]


def load_tdb_database(
    system_elements: List[str]
) -> Optional[Database]:
    """
    Load TDB database for given element system.
    
    Combines database finding and loading in one step.
    
    Args:
        system_elements: List of element symbols
        
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
    
    db_path = find_tdb_database(system_elements)
    if not db_path:
        return None
    
    try:
        db = Database(str(db_path))
        _log.info(f"Loaded database: {db_path.name}")
        
        # FIX: Convert all phase names in the database to regular Python strings
        # Pycalphad internally reads from db.phases dict, which may contain numpy.str_ keys
        # We need to rebuild the phases dict with regular string keys
        import numpy as np
        if hasattr(db, 'phases') and isinstance(db.phases, dict):
            # Log the first few phase names and their types for debugging
            sample_phases = list(db.phases.keys())[:3]
            _log.info(f"Sample phase names and types: {[(p, type(p).__name__) for p in sample_phases]}")
            
            # Check if any keys are numpy.str_ using multiple detection methods
            has_numpy_str = False
            numpy_str_phases = []
            for k in db.phases.keys():
                # Check both numpy.str_ and np.str_ (they might be different)
                if type(k).__name__ == 'str_' or isinstance(k, np.str_) or 'numpy' in str(type(k)):
                    has_numpy_str = True
                    numpy_str_phases.append(k)
            
            if has_numpy_str:
                _log.warning(f"Database has {len(numpy_str_phases)} numpy.str_ phase names - converting to regular Python str")
                # Rebuild phases dict with string keys
                new_phases = {}
                for key, value in db.phases.items():
                    str_key = str(key)  # Convert numpy.str_ to str
                    new_phases[str_key] = value
                db.phases = new_phases
                _log.info(f"Converted {len(new_phases)} phase names to regular Python str")
                # Verify conversion
                sample_new = list(db.phases.keys())[:3]
                _log.info(f"After conversion, sample types: {[(p, type(p).__name__) for p in sample_new]}")
            else:
                _log.info(f"All {len(db.phases)} phase names are already regular Python str")
        
        return db
    except Exception as e:
        _log.error(f"Failed to load database {db_path}: {e}")
        return None

def custom_equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, calc_opts=None, to_xarray=True,
                parameters=None, solver=None, phase_records=None, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.

    Copied from pycalphad.core.equilibrium.equilibrium, but with the numpy.str_ conversion fix.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str or list of str, optional
        Additional equilibrium model properties (e.g., CPM, HM, etc.) to compute.
        These must be defined as attributes in the Model class of each phase.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    verbose : bool, optional
        Print details of calculations. Useful for debugging.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    to_xarray : bool
        Whether to return an xarray Dataset (True, default) or an EquilibriumResult.
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.Solver.
    phase_records : Optional[Mapping[str, PhaseRecord]]
        Mapping of phase names to PhaseRecord objects with `'GM'` output. Must include
        all active phases. The `model` argument must be a mapping of phase names to
        instances of Model objects.

    Returns
    -------
    Structured equilibrium calculation

    Examples
    --------
    None yet.
    """
    # CRITICAL FIX: Monkey-patch PhaseRecordFactory.get() AND __getitem__() to convert numpy.str_ to str
    # This prevents the Cython PhaseRecord from receiving numpy.str_ which it rejects
    # NOTE: __getitem__ = get creates a reference at class definition time, so we must patch both!
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    original_prf_get = PhaseRecordFactory.get
    original_prf_getitem = PhaseRecordFactory.__getitem__
    
    def patched_prf_get(self, phase_name):
        phase_name_type = type(phase_name).__name__
        if phase_name_type == 'str_' or isinstance(phase_name, np.str_):
            _log.debug(f"custom_equilibrium: Converting numpy.str_ '{phase_name}' to str in .get()")
            phase_name = str(phase_name)
        return original_prf_get(self, phase_name)
    
    def patched_prf_getitem(self, phase_name):
        phase_name_type = type(phase_name).__name__
        if phase_name_type == 'str_' or isinstance(phase_name, np.str_):
            _log.debug(f"custom_equilibrium: Converting numpy.str_ '{phase_name}' to str in __getitem__()")
            phase_name = str(phase_name)
        return original_prf_getitem(self, phase_name)
    
    PhaseRecordFactory.get = patched_prf_get
    PhaseRecordFactory.__getitem__ = patched_prf_getitem
    _log.debug("custom_equilibrium: Monkey-patched PhaseRecordFactory.get() and __getitem__()")
    
    try:
        if output is None:
            output = set()
        elif (not isinstance(output, Iterable)) or isinstance(output, str):
            output = [output]
        
        # CRITICAL FIX: Convert ALL strings to regular Python str to avoid numpy.str_ issues
        # Convert phases - use list with dtype=object to prevent numpy string conversion
        _log.debug(f"custom_equilibrium: Input phase types: {[type(p).__name__ for p in (phases if isinstance(phases, (list, tuple)) else phases.keys())][:5]}")
        
        if isinstance(phases, (list, tuple)):
            # Create numpy array with dtype=object to force Python str, not numpy.str_
            phases_array = np.array([str(p) for p in phases], dtype=object)
            phases = list(phases_array)  # Convert back to list
            _log.debug(f"custom_equilibrium: Converted phase types: {[type(p).__name__ for p in phases][:5]}")
        elif isinstance(phases, dict):
            phases = {str(k): v for k, v in phases.items()}
        
        # Convert components - use list with dtype=object
        if isinstance(comps, (list, tuple)):
            comps_array = np.array([str(c) for c in comps], dtype=object)
            comps = list(comps_array)
        
        # AGGRESSIVE FIX: Force convert ALL phase names in the database to regular str
        # This ensures Workspace doesn't encounter numpy.str_ when it accesses db.phases internally
        if hasattr(dbf, 'phases') and isinstance(dbf.phases, dict):
            # Always rebuild to ensure no numpy.str_ sneaks in
            new_phases = {}
            for key, value in dbf.phases.items():
                str_key = str(key)
                new_phases[str_key] = value
            dbf.phases = new_phases
            _log.debug(f"custom_equilibrium: Rebuilt db.phases dict with {len(new_phases)} regular str keys")
        
        _log.debug(f"custom_equilibrium: calling Workspace with {len(phases)} phases, {len(comps)} components (all converted to str)")
        
        # Call Workspace - this internally may create numpy.str_ in xarray coordinates
        wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions, models=model, parameters=parameters,
                        verbose=verbose, calc_opts=calc_opts, solver=solver, phase_record_factory=phase_records)
        
        # CRITICAL POST-FIX: Immediately after Workspace creation, fix any numpy.str_ in the internal equilibrium result
        # Access the internal eq property and fix Phase coordinates
        if hasattr(wks, 'eq') and hasattr(wks.eq, 'Phase'):
            try:
                phase_coord = wks.eq.Phase
                if hasattr(phase_coord, 'values'):
                    phase_values = phase_coord.values
                    if any(type(p).__name__ == 'str_' or isinstance(p, np.str_) for p in phase_values):
                        _log.warning(f"custom_equilibrium: Workspace created numpy.str_ in Phase coords - fixing immediately")
                        # Fix the Phase coordinate by reassigning with dtype=object
                        fixed_phases = np.array([str(p) for p in phase_values], dtype=object)
                        wks.eq.Phase = fixed_phases
                        _log.debug(f"custom_equilibrium: Fixed Phase coords to regular Python str")
            except Exception as e:
                _log.debug(f"custom_equilibrium: Could not pre-fix Phase coords: {e}")
        
        # Compute equilibrium values of any additional user-specified properties
        # We already computed these properties so don't recompute them
        properties = wks.eq
        conds_keys = [str(k) for k in properties.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        output = sorted(set(output) - {'GM', 'MU'})
        for out in output:
            cprop = as_property(out)
            out = str(cprop)
            result_array = np.zeros(properties.GM.shape) # Will not work for non-scalar properties
            for index, composition_sets in wks.enumerate_composition_sets():
                cur_conds = OrderedDict(zip(conds_keys,
                                            [np.asarray(properties.coords[b][a], dtype=np.float64)
                                            for a, b in zip(index, conds_keys)]))
                chemical_potentials = properties.MU[index]
                result_array[index] = cprop.compute_property(composition_sets, cur_conds, chemical_potentials)
            result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
            properties.merge(result, inplace=True, compat='equals')
        if to_xarray:
            properties = wks.eq.get_dataset()
        
        # CRITICAL FIX: Convert any numpy.str_ in Phase coordinates back to regular Python str
        # This prevents downstream Cython errors when accessing phase data
        if hasattr(properties, 'coords') and 'Phase' in properties.coords:
            phase_coords = properties.coords['Phase'].values
            # Check if any phase coordinates are numpy.str_
            has_numpy_str = any(isinstance(p, np.str_) or type(p).__name__ == 'str_' for p in phase_coords)
            if has_numpy_str:
                _log.warning(f"custom_equilibrium: Found numpy.str_ in result Phase coords, converting to str")
                # Convert numpy.str_ to regular str
                new_phase_coords = np.array([str(p) for p in phase_coords], dtype=object)
                properties = properties.assign_coords(Phase=new_phase_coords)
                _log.debug(f"custom_equilibrium: Phase coords after conversion: {[type(p).__name__ for p in properties.coords['Phase'].values[:3]]}")
        
        properties.attrs['created'] = datetime.now().isoformat()
        if len(kwargs) > 0:
            warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
        return properties
    finally:
        # Restore the original PhaseRecordFactory methods
        PhaseRecordFactory.get = original_prf_get
        PhaseRecordFactory.__getitem__ = original_prf_getitem
        _log.debug("custom_equilibrium: Restored original PhaseRecordFactory methods")

# =============================================================================
# EQUILIBRIUM CALCULATION UTILITIES
# =============================================================================

def compute_equilibrium(
    db: Database,
    elements: List[str],
    phases: List[str],
    composition: Dict[str, float],
    temperature: float,
    pressure: float = 101325,
    calc_opts: Optional[dict] = None,      # <— allow tuning density later
) -> Optional[Any]:
    try:
        # Only the requested system elements (uppercase) – *not* all DB elements
        sys_elems = sorted({el.upper() for el in elements if el.upper() != 'VA'})
        calc_elements = sys_elems + ['VA']

        # Complete composition over system elements only
        complete_composition = {el: 0.0 for el in sys_elems}
        for k, vfrac in composition.items():
            kU = k.upper()
            if kU in complete_composition:
                complete_composition[kU] = float(vfrac)

        # Build conditions (N-1 independent X constraints)
        conditions = {v.T: temperature, v.P: pressure, v.N: 1.0}
        if len(sys_elems) > 1:
            for el in sys_elems[1:]:
                conditions[v.X(el)] = complete_composition.get(el, 0.0)

        # Reasonable default calc options (you can override via calc_opts)
        co = {"pdens": 120}
        if calc_opts:
            co.update(calc_opts)

        _log.info(f"Running equilibrium with {len(phases)} phases, "
                  f"T={temperature}K, components={calc_elements}, "
                  f"X-conds={{...}}")
        eq = custom_equilibrium(db, calc_elements, phases, conditions, calc_opts=co)
        return eq

    except Exception as e:
        _log.error(f"Equilibrium calculation failed: {e}")
        _log.error(f"  Components: {calc_elements if 'calc_elements' in locals() else elements}")
        _log.error(f"  Phases ({len(phases)}): {phases}")
        _log.error(f"  Conditions: {conditions if 'conditions' in locals() else 'n/a'}")
        _log.error(f"  Composition: {complete_composition if 'complete_composition' in locals() else composition}")
        return None

def extract_phase_fractions(eq: Any, tolerance: float = 1e-4) -> Dict[str, float]:
    """
    Extract phase fractions from equilibrium result with proper vertex handling.
    
    Properly handles:
    - Multi-vertex equilibrium results (e.g., two-phase regions)
    - Singleton coordinate squeezing
    - Fallback for legacy/simple equilibrium results
    
    In two-phase regions, equilibrium returns multiple vertices; we sum NP 
    over the vertex dimension per phase (not rely on 1-to-1 raveled index).
    
    Args:
        eq: Equilibrium result from pycalphad
        tolerance: Minimum fraction to include (default 1e-4 for better boundary handling)
        
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
            # This is critical for two-phase regions where we have multiple vertices
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
            # Convert numpy.str_ to regular str to avoid type errors
            phase_str = str(phase)
            if not phase_str or phase_str == '':
                continue
            
            frac = float(frac_by_phase.sel(Phase=phase_str).values)
            
            # Use slightly looser tolerance for reporting (handles boundary noise better)
            if not np.isnan(frac) and frac > tolerance:
                phase_fractions[phase_str] = frac
        
        return phase_fractions
        
    except Exception as e:
        _log.warning(f"Error extracting phase fractions (trying fallback): {e}")
        # Fallback to simple approach for legacy/incompatible equilibrium results
        try:
            phase_fractions = {}
            phase_array = eq.Phase.values
            np_array = eq.NP.values
            
            for phase in np.unique(phase_array):
                # Convert numpy.str_ to regular str to avoid type errors
                phase_str = str(phase)
                if phase_str == '':
                    continue
                    
                mask = (phase_array == phase)
                fraction = np.sum(np_array[mask])
                
                if not np.isnan(fraction) and fraction > tolerance:
                    phase_fractions[phase_str] = float(fraction)
            
            return phase_fractions
        except Exception as fallback_error:
            _log.error(f"Fallback phase fraction extraction also failed: {fallback_error}")
            return {}


def extract_phase_fractions_from_equilibrium(eq: Any, tolerance: float = 1e-4) -> Dict[str, float]:
    """
    Legacy alias for extract_phase_fractions (backward compatibility).
    
    This function name is used in equilibrium_utils.py and some AI function mixins.
    Redirects to the canonical extract_phase_fractions() implementation.
    
    Args:
        eq: Equilibrium result from pycalphad
        tolerance: Minimum fraction to include
        
    Returns:
        Dictionary mapping phase names to fractions
    """
    return extract_phase_fractions(eq, tolerance)


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
        
        # Find indices where this phase exists
        phase_mask = eqp['Phase'] == phase_name
        
        # Check if phase exists at all
        if not phase_mask.any():
            _log.warning(f"Phase {phase_name} not found in equilibrium result")
            return {}
        
        phase_composition = {}
        for elem in elements:
            try:
                # Get composition data for this element
                x_data = eqp['X'].sel(component=elem)
                
                # Extract values only where this phase exists
                # Use .where() with drop=False to keep dimensions, then compute mean over valid values
                masked_data = x_data.where(phase_mask)
                
                # Get the mean, ignoring NaN values
                x_val = float(masked_data.mean(skipna=True).values)
                
                # Handle case where all values were NaN
                if np.isnan(x_val):
                    _log.debug(f"No valid composition data for {elem} in {phase_name}")
                    continue
                
                if x_val > 1e-6:  # Only record non-negligible amounts
                    phase_composition[elem] = x_val
                    _log.debug(f"Extracted {elem} composition from {phase_name}: {x_val:.4f}")
                    
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
            _log.info(f"Phase {phase_name} composition (normalized): {phase_composition}")
        else:
            # This warning occurs when a phase exists in equilibrium but has no extractable
            # composition data. Common causes:
            # 1. Phase is marginally stable (very small fraction, near machine precision)
            # 2. Numerical issues in equilibrium calculation for this phase
            # 3. Phase at boundary conditions where composition data is uncertain
            # This is handled gracefully - we return empty dict and calculations continue.
            _log.warning(f"No valid composition data extracted for phase {phase_name}. Total composition: {total_comp}")
        
        return phase_composition
        
    except Exception as e:
        _log.error(f"Error extracting phase composition for {phase_name}: {e}", exc_info=True)
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
        db = load_tdb_database(system_elements)
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
        # Convert phase names to regular Python strings to avoid numpy.str_ issues
        phases = [str(p) for p in db.phases.keys()]
        
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

