"""
CALPHAD phase diagram generation module.

Provides phase diagram calculation and plotting functionality using pycalphad.
"""

from .phase_diagrams import CalPhadHandler
from .database_utils import (
    is_excluded_phase,
    EXCLUDE_PHASE_PATTERNS,
    upper_symbol,
    get_db_elements,
    compose_alias_map,
    pick_tdb_path
)
from .consts import weight_to_mole_fraction
from ...constants import ELEMENT_ALIASES
from .plotting import PlottingMixin
from .analysis import AnalysisMixin
from .ai_functions import AIFunctionsMixin

# Backward compatibility aliases
_is_excluded_phase = is_excluded_phase
_upper_symbol = upper_symbol
_db_elements = get_db_elements
_compose_alias_map = compose_alias_map
_pick_tdb_path = pick_tdb_path

__all__ = [
    'CalPhadHandler',
    'PlottingMixin',
    'AnalysisMixin',
    'AIFunctionsMixin',
    # New clean names
    'is_excluded_phase',
    'EXCLUDE_PHASE_PATTERNS',
    'upper_symbol',
    'get_db_elements',
    'compose_alias_map',
    'pick_tdb_path',
    # Constants
    'ELEMENT_ALIASES',
    # Utility functions
    'weight_to_mole_fraction',
    # Backward compatibility
    '_is_excluded_phase',
    '_upper_symbol',
    '_db_elements',
    '_compose_alias_map',
    '_pick_tdb_path',
]
