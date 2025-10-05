"""
CALPHAD phase diagram generation module.

Provides phase diagram calculation and plotting functionality using pycalphad.
"""

from .phase_diagrams import CalPhadHandler
from .utils import _is_excluded_phase, EXCLUDE_PHASE_PATTERNS, _upper_symbol, _db_elements, _compose_alias_map, _pick_tdb_path
from .consts import _ATOMIC_MASS, STATIC_ALIASES
from .plotting import PlottingMixin
from .analysis import AnalysisMixin
from .ai_functions import AIFunctionsMixin

__all__ = [
    'CalPhadHandler', 
    '_is_excluded_phase', 
    'EXCLUDE_PHASE_PATTERNS',
    '_upper_symbol',
    '_db_elements', 
    '_compose_alias_map',
    '_pick_tdb_path',
    '_ATOMIC_MASS',
    'STATIC_ALIASES',
    'PlottingMixin',
    'AnalysisMixin',
    'AIFunctionsMixin'
]
