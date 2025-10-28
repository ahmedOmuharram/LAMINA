"""
Utility functions for CALPHAD phase diagrams.

Re-exports utilities from specialized sub-modules and shared calphad_utils.
For database loading, use shared.calphad_utils.load_tdb_database() directly.
"""

from .database_utils import (
    is_excluded_phase,
    get_db_elements,
    compose_alias_map,
    map_phase_name,
    EXCLUDE_PHASE_PATTERNS,
    PHASE_NAME_MAP
)
from ...shared.calphad_utils import find_tdb_database, load_tdb_database

__all__ = [
    'is_excluded_phase',
    'get_db_elements',
    'compose_alias_map',
    'find_tdb_database',
    'load_tdb_database',
    'map_phase_name',
    'EXCLUDE_PHASE_PATTERNS',
    'PHASE_NAME_MAP',
]
