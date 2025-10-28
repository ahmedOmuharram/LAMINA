"""
Shared utilities used across multiple handlers.

Contains:
- calphad_utils: Common CALPHAD/thermodynamic calculation functions
  including database loading, equilibrium calculations, phase fraction
  extraction, and element normalization.
"""

from . import calphad_utils

# Re-export commonly used functions for convenience
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

__all__ = [
    "calphad_utils",
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
]
