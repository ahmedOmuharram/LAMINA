"""
Base handler module containing:
- BaseHandler: Abstract base class for all handlers
- constants: Physical constants, conversion factors, and reference data
- converters: Unit conversion utilities

This module provides the foundation for all specialized handlers in the system.
"""

from .base import BaseHandler

# Re-export constants for convenience
from .constants import *

# Re-export converters
from .converters import (
    # Energy conversions
    kjmol_to_ev,
    ev_to_kjmol,
    
    # Composition conversions
    weight_to_mole_fraction,
    atpct_to_molefrac,
    
    # Magnetic unit conversions
    muB_per_bohr3_to_kA_per_m,
    kA_per_m_to_muB_per_bohr3,
    
    # Length conversions
    pm_to_angstrom,
    angstrom_to_pm,
    to_angstrom,
    
    # Temperature conversions
    celsius_to_kelvin,
    kelvin_to_celsius,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
)

__all__ = [
    'BaseHandler',
    # Energy conversions
    'kjmol_to_ev',
    'ev_to_kjmol',
    # Composition conversions
    'weight_to_mole_fraction',
    'atpct_to_molefrac',
    # Magnetic conversions
    'muB_per_bohr3_to_kA_per_m',
    'kA_per_m_to_muB_per_bohr3',
    # Length conversions
    'pm_to_angstrom',
    'angstrom_to_pm',
    'to_angstrom',
    # Temperature conversions
    'celsius_to_kelvin',
    'kelvin_to_celsius',
    'fahrenheit_to_kelvin',
    'kelvin_to_fahrenheit',
]

