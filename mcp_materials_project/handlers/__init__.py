"""
Material Project API endpoint handlers.

This package contains the individual endpoint handlers that were previously
all crammed into the monolithic handle_materials_endpoint function.
"""

from .base import BaseHandler, InvalidRangeError, RANGE_KEYS
from .material_details import MaterialDetailsHandler, handle_material_details
from .material_search import MaterialSearchHandler, handle_material_search, handle_material_by_char
from .searxng_search import SearXNGSearchHandler, handle_searxng_search, handle_searxng_engine_stats

__all__ = [
    "BaseHandler",
    "InvalidRangeError", 
    "RANGE_KEYS",
    "MaterialDetailsHandler",
    "MaterialSearchHandler", 
    "SearXNGSearchHandler",
    "handle_material_details", 
    "handle_material_search",
    "handle_material_by_char",
    "handle_searxng_search",
    "handle_searxng_engine_stats",
]
