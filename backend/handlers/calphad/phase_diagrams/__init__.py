"""
CALPHAD phase diagram generation module.

Provides phase diagram calculation and plotting functionality using pycalphad.
"""

from .phase_diagrams import CalPhadHandler
from .database_utils import (
    is_excluded_phase,
    EXCLUDE_PHASE_PATTERNS,
    get_db_elements,
    compose_alias_map,
)
from ...shared.calphad_utils import find_tdb_database
from ...base.converters import weight_to_mole_fraction
from ...base.constants import ELEMENT_ALIASES
from .plotting import PlottingMixin
from .analysis import AnalysisMixin
from .ai_functions import AIFunctionsMixin

__all__ = [
    'CalPhadHandler',
    'PlottingMixin',
    'AnalysisMixin',
    'AIFunctionsMixin',
    'is_excluded_phase',
    'EXCLUDE_PHASE_PATTERNS',
    'get_db_elements',
    'compose_alias_map',
    'find_tdb_database',
    'ELEMENT_ALIASES',
    'weight_to_mole_fraction',
]
