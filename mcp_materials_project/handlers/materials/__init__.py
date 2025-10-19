"""
Materials handlers for Materials Project API.

This module provides handlers for material search and details endpoints.
"""

from .material_details import MaterialDetailsHandler, handle_material_details
from .material_search import MaterialSearchHandler, handle_material_search, handle_material_by_char

__all__ = [
    "MaterialDetailsHandler",
    "MaterialSearchHandler",
    "handle_material_details",
    "handle_material_search",
    "handle_material_by_char",
]

