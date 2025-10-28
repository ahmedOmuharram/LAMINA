"""
Material Project API endpoint handlers.

This package contains organized handlers for different aspects of materials science:
- materials/: Materials Project database search and details
- search/: Web and scientific literature search
- electrochemistry/: Battery and electrode calculations
- calphad/: Phase diagram calculations
- semiconductors/: Semiconductor defect and doping analysis
- magnets/: Permanent magnet strength assessment and doping effects
- solutes/: Lattice parameter effects of substitutional solutes in fcc matrices
- alloys/: Alloy surface and microstructure analysis
- superconductors/: Superconductor materials analysis
"""

from .base.base import BaseHandler, InvalidRangeError
from .shared.constants import RANGE_KEYS
from .materials import MaterialHandler
from .search import SearXNGSearchHandler
from .electrochemistry import BatteryHandler
from .alloys import AlloyHandler
from .superconductors import SuperconductorHandler
from .semiconductors import SemiconductorHandler
from .magnets import MagnetHandler
from .solutes import SolutesHandler
from .calphad import CalPhadHandler

__all__ = [
    # Base classes
    "BaseHandler",
    "InvalidRangeError", 
    "RANGE_KEYS",
    
    # Materials handlers
    "MaterialHandler",
    
    # Search handlers
    "SearXNGSearchHandler",
    
    # Electrochemistry handlers
    "BatteryHandler",
    
    # Alloys handlers
    "AlloyHandler",
    
    # Superconductor handlers
    "SuperconductorHandler",
    
    # Semiconductor handlers
    "SemiconductorHandler",
    
    # Magnet handlers
    "MagnetHandler",
    
    # Solutes handlers
    "SolutesHandler",
    
    # CALPHAD handlers
    "CalPhadHandler",
]
