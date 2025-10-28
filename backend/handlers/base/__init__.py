"""
Base handler module containing the BaseHandler class.

This module provides the foundation for all specialized handlers in the system.
BaseHandler provides common functionality like:
- Materials Project API client management
- Pagination utilities
- Parameter parsing and validation
- Tool output tracking

For shared utilities (converters, constants, result wrappers), see handlers.shared
"""

from .base import BaseHandler, InvalidRangeError
from .decorators import track_tool_output, auto_track

__all__ = [
    'BaseHandler',
    'InvalidRangeError',
    'track_tool_output',
    'auto_track',
]
