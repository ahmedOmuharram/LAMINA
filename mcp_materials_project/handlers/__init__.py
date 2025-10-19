"""
Material Project API endpoint handlers.

This package contains organized handlers for different aspects of materials science:
- materials/: Materials Project database search and details
- search/: Web and scientific literature search
- electrochemistry/: Battery and electrode calculations
- calphad/: Phase diagram calculations
"""

from .base import BaseHandler, InvalidRangeError, RANGE_KEYS
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
]
