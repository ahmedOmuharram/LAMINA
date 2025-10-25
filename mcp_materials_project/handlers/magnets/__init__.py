"""
Magnet Strength Assessment Module

Provides comprehensive tools for assessing permanent magnet strength,
particularly for evaluating whether doping improves pull force capabilities.
"""

from .magnet_handler import MagnetHandler, create_magnet_handler
from .ai_functions import MagnetAIFunctionsMixin

__all__ = [
    "MagnetHandler",
    "create_magnet_handler",
    "MagnetAIFunctionsMixin",
]

