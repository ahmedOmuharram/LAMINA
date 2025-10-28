"""
Semiconductor and Defect Analysis Module

This module provides comprehensive tools for analyzing semiconductors, defects,
and doping effects in materials.

Key Features:
- Structural analysis (octahedral distortions, bond lengths)
- Magnetic property analysis and comparison
- Defect formation energy calculations
- Doping site preference determination
- Phase transition analysis
"""

from .semiconductor_handler import SemiconductorHandler

__all__ = [
    "SemiconductorHandler",
]

