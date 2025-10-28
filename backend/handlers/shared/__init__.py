"""
Shared utilities used across multiple handlers.

This module contains:
- converters: Unit conversion utilities (energy, composition, magnetic, length, temperature)
- result_wrappers: Standardized result formatting for AI functions
- constants: Physical constants, conversion factors, and reference data
- calphad_utils: CALPHAD/thermodynamic calculation functions

These utilities are shared across handlers and don't belong to any specific domain.
"""

# Core utilities
from . import calphad_utils
from . import converters
from . import result_wrappers
from . import constants

# Re-export commonly used CALPHAD functions
from .calphad_utils import (
    find_tdb_database,
    load_tdb_database,
    compute_equilibrium,
    extract_phase_fractions,
    extract_phase_fractions_from_equilibrium,
    get_phase_fraction_by_base_name,
    get_phase_composition,
    parse_calphad_phase_name,
    classify_phase_type,
    verify_elements_in_database,
    normalize_composition,
    compute_equilibrium_microstructure,
)

# Re-export commonly used converters
from .converters import (
    # Energy conversions
    kjmol_to_ev,
    ev_to_kjmol,
    
    # Composition conversions
    weight_to_mole_fraction,
    atpct_to_molefrac,
    
    # Magnetic unit conversions
    muB_per_bohr3_to_kA_per_m,
    kA_per_m_to_muB_per_bohr3,
    
    # Length conversions
    pm_to_angstrom,
    angstrom_to_pm,
    to_angstrom,
    
    # Temperature conversions
    celsius_to_kelvin,
    kelvin_to_celsius,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
)

# Re-export result wrapper utilities
from .result_wrappers import (
    success_result,
    error_result,
    simple_success,
    simple_error,
    with_timing,
    ErrorType,
    Confidence,
    ensure_list,
    merge_citations,
)

__all__ = [
    # Modules
    "calphad_utils",
    "converters",
    "result_wrappers",
    "constants",
    
    # CALPHAD utilities
    "find_tdb_database",
    "load_tdb_database",
    "compute_equilibrium",
    "extract_phase_fractions",
    "extract_phase_fractions_from_equilibrium",
    "get_phase_fraction_by_base_name",
    "get_phase_composition",
    "parse_calphad_phase_name",
    "classify_phase_type",
    "verify_elements_in_database",
    "normalize_composition",
    "compute_equilibrium_microstructure",
    
    # Energy conversions
    "kjmol_to_ev",
    "ev_to_kjmol",
    
    # Composition conversions
    "weight_to_mole_fraction",
    "atpct_to_molefrac",
    
    # Magnetic conversions
    "muB_per_bohr3_to_kA_per_m",
    "kA_per_m_to_muB_per_bohr3",
    
    # Length conversions
    "pm_to_angstrom",
    "angstrom_to_pm",
    "to_angstrom",
    
    # Temperature conversions
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_kelvin",
    "kelvin_to_fahrenheit",
    
    # Result wrappers
    "success_result",
    "error_result",
    "simple_success",
    "simple_error",
    "with_timing",
    "ErrorType",
    "Confidence",
    "ensure_list",
    "merge_citations",
]
