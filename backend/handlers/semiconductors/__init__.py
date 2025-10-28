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

from .semiconductor_handler import SemiconductorHandler, create_semiconductor_handler
from .ai_functions import SemiconductorAIFunctionsMixin
from .utils import (
    analyze_octahedral_distortion,
    get_magnetic_properties_detailed,
    compare_magnetic_properties,
    calculate_defect_formation_energy,
    analyze_doping_site_preference,
    analyze_structure_temperature_dependence
)

__all__ = [
    "SemiconductorHandler",
    "create_semiconductor_handler",
    "SemiconductorAIFunctionsMixin",
    "analyze_octahedral_distortion",
    "get_magnetic_properties_detailed",
    "compare_magnetic_properties",
    "calculate_defect_formation_energy",
    "analyze_doping_site_preference",
    "analyze_structure_temperature_dependence",
]

