"""
Constants for electrochemistry calculations.
"""

# Faraday constant in C/mol
FARADAY_CONSTANT = 96485.3321233100184

# Check PyMatGen availability
import logging

_log = logging.getLogger(__name__)

try:
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    _log.warning("PyMatGen battery modules not available")
    PYMATGEN_AVAILABLE = False

