"""
Energy conversion factors and cohesive energy data.

DO NOT change these values without proper scientific justification.
"""

# ============================================================================
# Energy Conversion Factors
# ============================================================================

# 1 eV/atom = 96.485... kJ/mol
# This is the conversion between atomic-scale energies and bulk thermodynamics
KJMOL_PER_EV_PER_ATOM = 96.4853321233

# ============================================================================
# Cohesive Energies (eV/atom)
# ============================================================================

# Typical cohesive energies (eV/atom) near 0 K (solid → atoms)
# Used as fallback when database values are unavailable
# References: Kittel, Introduction to Solid State Physics; CRC Handbook
COHESIVE_ENERGY_FALLBACK = {
    "Al": 3.39,
    "Au": 3.81,
    "Cu": 3.49,
    "Ag": 2.95,
    "Ni": 4.44,
    "Pt": 5.84,
    "Pd": 3.89,
    "Fe": 4.28,
    "Ti": 4.85,
    "Mg": 1.51,
}

