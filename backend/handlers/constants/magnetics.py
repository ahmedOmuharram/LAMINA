"""
Magnetics-specific conversion factors and constants.

Derived from fundamental physical constants in physical.py but grouped here
for convenience in magnetic property calculations.

DO NOT change these values.
"""

import numpy as np

# Import fundamental constants
from .physical import MU_0, BOHR_MAGNETON, AVOGADRO

# ============================================================================
# Magnetic Conversion Factors
# ============================================================================

# Re-export fundamental magnetic constants for convenience
# (Original definitions in physical.py)
__MU_0 = MU_0                           # Vacuum permeability (H/m)
__BOHR_MAGNETON = BOHR_MAGNETON         # Bohr magneton (A⋅m²)
__AVOGADRO = AVOGADRO                   # Avogadro's number (1/mol)

# Bohr magneton to emu conversion
MU_B_TO_EMU = 9.274e-21  # emu per μB

# Magnetization per volume conversion factor
# Materials Project reports in μB / bohr³
# 1 μB = 9.274e-24 A·m²
# 1 bohr = 0.529177 Å = 0.529177e-10 m
# 1 bohr³ ≈ 1.4818e-31 m³
# => 1 (μB / bohr³) ≈ 6.2584e7 A/m ≈ 6.2584e4 kA/m
MUB_PER_BOHR3_TO_KA_PER_M = 6.258412893e4  # kA/m per (μB/bohr³)

