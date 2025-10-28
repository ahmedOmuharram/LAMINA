"""
Centralized constants for all handlers.

This module provides a single import point for all physical, chemical,
and computational constants used across the handler modules.

Organization:
    - physical.py: Fundamental physical constants (Faraday, Bohr magneton, etc.)
    - atomic.py: Metallic radii, lattice parameters
    - elements.py: Element name aliases and mappings
    - energetics.py: Energy conversion factors and cohesive energies
    - surfaces.py: Surface adsorption and diffusion scaling factors
    - phases.py: CALPHAD phase classifications and categories
    - database.py: Database exclusion patterns and phase name mappings
    - electrochemistry.py: Battery diffusion barriers and structure defaults
    - magnetics.py: Magnetic conversion factors
    - api.py: API query parameter constants

Usage:
    from handlers.constants import FARADAY_CONSTANT, METALLIC_RADII_PM
    from handlers.constants import PhaseCategory, PHASE_CLASSIFICATION
    from handlers.constants.atomic import FCC_LATTICE_PARAMS_A
"""

# Physical constants
from .physical import (
    FARADAY_CONSTANT,
    MU_0,
    BOHR_MAGNETON,
    AVOGADRO,
    MU_B_TO_EMU,
    MUB_PER_BOHR3_TO_KA_PER_M,
)

# Atomic properties
from .atomic import (
    METALLIC_RADII_PM,
    FCC_LATTICE_PARAMS_A,
)

# Element mappings
from .elements import (
    ELEMENT_ALIASES,
)

# Energetics
from .energetics import (
    KJMOL_PER_EV_PER_ATOM,
    COHESIVE_ENERGY_FALLBACK,
)

# Surface science
from .surfaces import (
    ADS_OVER_COH_111,
    ADS_OVER_COH_100,
    ADS_OVER_COH_110,
    DIFF_OVER_ADS_111,
    DIFF_OVER_ADS_100,
    DIFF_OVER_ADS_110,
)

# Phase classification
from .phases import (
    PhaseCategory,
    PHASE_CLASSIFICATION,
)

# Database utilities
from .database import (
    EXCLUDE_PHASE_PATTERNS,
    PHASE_NAME_MAP,
)

# Electrochemistry
from .electrochemistry import (
    KNOWN_DIFFUSION_BARRIERS,
    STRUCTURE_DIFFUSION_DEFAULTS,
)

# Magnetics (re-exported for convenience)
from .magnetics import (
    MU_B_TO_EMU as MAGNETICS_MU_B_TO_EMU,
    MUB_PER_BOHR3_TO_KA_PER_M as MAGNETICS_MUB_PER_BOHR3_TO_KA_PER_M,
)

# API constants
from .api import (
    RANGE_KEYS,
)

__all__ = [
    # Physical
    "FARADAY_CONSTANT",
    "MU_0",
    "BOHR_MAGNETON",
    "AVOGADRO",
    "MU_B_TO_EMU",
    "MUB_PER_BOHR3_TO_KA_PER_M",
    # Atomic
    "METALLIC_RADII_PM",
    "FCC_LATTICE_PARAMS_A",
    # Elements
    "ELEMENT_ALIASES",
    # Energetics
    "KJMOL_PER_EV_PER_ATOM",
    "COHESIVE_ENERGY_FALLBACK",
    # Surfaces
    "ADS_OVER_COH_111",
    "ADS_OVER_COH_100",
    "ADS_OVER_COH_110",
    "DIFF_OVER_ADS_111",
    "DIFF_OVER_ADS_100",
    "DIFF_OVER_ADS_110",
    # Phases
    "PhaseCategory",
    "PHASE_CLASSIFICATION",
    # Database
    "EXCLUDE_PHASE_PATTERNS",
    "PHASE_NAME_MAP",
    # Electrochemistry
    "KNOWN_DIFFUSION_BARRIERS",
    "STRUCTURE_DIFFUSION_DEFAULTS",
    # API
    "RANGE_KEYS",
]

