"""
Utility functions for CALPHAD phase diagrams.

This module re-exports utilities from specialized sub-modules for backward compatibility.
"""

# Re-export from database_utils for backward compatibility
from .database_utils import (
    is_excluded_phase as _is_excluded_phase,
    upper_symbol as _upper_symbol,
    get_db_elements as _db_elements,
    compose_alias_map as _compose_alias_map,
    pick_tdb_path as _pick_tdb_path,
    load_database,
    map_phase_name,
    EXCLUDE_PHASE_PATTERNS,
    PHASE_NAME_MAP
)

__all__ = [
    '_is_excluded_phase',
    '_upper_symbol',
    '_db_elements',
    '_compose_alias_map',
    '_pick_tdb_path',
    'load_database',
    'map_phase_name',
    'EXCLUDE_PHASE_PATTERNS',
    'PHASE_NAME_MAP',
]
