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
"""

from .base.base import BaseHandler, InvalidRangeError
from .base.constants import RANGE_KEYS
from .materials import (
    MaterialDetailsHandler, 
    MaterialSearchHandler,
    handle_material_details, 
    handle_material_search,
    handle_material_by_char
)
from .search import (
    SearXNGSearchHandler, 
    handle_searxng_search, 
    handle_searxng_engine_stats
)
from .electrochemistry import BatteryHandler
from .alloys import AlloyHandler
from .superconductors import SuperconductorHandler
from .semiconductors import SemiconductorHandler, create_semiconductor_handler
from .magnets import MagnetHandler, create_magnet_handler
from .solutes import SolutesHandler

__all__ = [
    # Base classes
    "BaseHandler",
    "InvalidRangeError", 
    "RANGE_KEYS",
    
    # Materials handlers
    "MaterialDetailsHandler",
    "MaterialSearchHandler", 
    "handle_material_details", 
    "handle_material_search",
    "handle_material_by_char",
    
    # Search handlers
    "SearXNGSearchHandler",
    "handle_searxng_search",
    "handle_searxng_engine_stats",
    
    # Electrochemistry handlers
    "BatteryHandler",
    
    # Alloys handlers
    "AlloyHandler",
    
    # Superconductor handlers
    "SuperconductorHandler",
    
    # Semiconductor handlers
    "SemiconductorHandler",
    "create_semiconductor_handler",
    
    # Magnet handlers
    "MagnetHandler",
    "create_magnet_handler",
    
    # Solutes handlers
    "SolutesHandler",
]
