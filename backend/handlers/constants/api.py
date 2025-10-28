"""
API-specific constants: query parameters and field mappings.

Defines which query parameters accept range queries (min/max) for
Materials Project API interactions.

DO NOT change these without understanding API behavior.
"""

# ============================================================================
# Range Query Parameters
# ============================================================================

# Which query params are treated as ranges (min,max)
# These parameters support range-based filtering in Materials Project queries
RANGE_KEYS = [
    "band_gap", "density", "e_electronic", "e_ionic", "e_total", "efermi",
    "elastic_anisotropy", "energy_above_hull", "equilibrium_reaction_energy",
    "formation_energy", "g_reuss", "g_voigt", "g_vrh", "k_reuss", "k_voigt",
    "k_vrh", "n", "nelements", "num_sites", "num_magnetic_sites",
    "num_unique_magnetic_sites", "piezoelectric_modulus", "poisson_ratio",
    "shape_factor", "surface_energy_anisotropy", "total_energy",
    "total_magnetization", "total_magnetization_normalized_formula_units",
    "total_magnetization_normalized_vol", "uncorrected_energy", "volume",
    "weighted_surface_energy", "weighted_work_function",
]

