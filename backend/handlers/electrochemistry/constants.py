"""
Constants for electrochemistry calculations - now imports from centralized constants.

This file is kept for backward compatibility but imports from handlers.constants.
"""

import logging

from ..shared.constants import FARADAY_CONSTANT

_log = logging.getLogger(__name__)

# Check PyMatGen availability
try:
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    _log.warning("PyMatGen battery modules not available")
    PYMATGEN_AVAILABLE = False

