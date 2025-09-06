"""
Material Project API endpoint handlers.

This package contains the individual endpoint handlers that were previously
all crammed into the monolithic handle_materials_endpoint function.
"""

from .base import BaseHandler, InvalidRangeError, RANGE_KEYS
from .material_details import MaterialDetailsHandler, handle_material_details
from .material_search import MaterialSearchHandler, handle_material_search, handle_material_by_char
from .name_conversion import NameConversionHandler, handle_name_conversion

__all__ = [
    "BaseHandler",
    "InvalidRangeError", 
    "RANGE_KEYS",
    "MaterialDetailsHandler",
    "MaterialSearchHandler", 
    "NameConversionHandler",
    "handle_material_details", 
    "handle_material_search",
    "handle_material_by_char",
    "handle_name_conversion",
]
