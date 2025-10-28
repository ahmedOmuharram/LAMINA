"""
Fundamental physical constants used across handlers.

All values from NIST/CODATA or standard physics references.
DO NOT change these values without proper scientific justification.
"""

import numpy as np

# ============================================================================
# Electromagnetic Constants
# ============================================================================

# Faraday constant in C/mol (charge per mole of electrons)
FARADAY_CONSTANT = 96485.3321233100184

# Vacuum permeability in H/m (μ₀)
MU_0 = 4e-7 * np.pi

# ============================================================================
# Atomic/Quantum Constants
# ============================================================================

# Bohr magneton in A⋅m²
BOHR_MAGNETON = 9.274e-24

# Avogadro's number in 1/mol
AVOGADRO = 6.02214076e23

# ============================================================================
# Conversion Factors - Magnetics
# ============================================================================

# Bohr magneton to emu conversion
MU_B_TO_EMU = 9.274e-21  # emu per μB

# Magnetization per volume conversion factor
# Materials Project reports in μB / bohr³
# 1 μB = 9.274e-24 A·m²
# 1 bohr = 0.529177 Å = 0.529177e-10 m
# 1 bohr³ ≈ 1.4818e-31 m³
# => 1 (μB / bohr³) ≈ 6.2584e7 A/m ≈ 6.2584e4 kA/m
MUB_PER_BOHR3_TO_KA_PER_M = 6.258412893e4  # kA/m per (μB/bohr³)

