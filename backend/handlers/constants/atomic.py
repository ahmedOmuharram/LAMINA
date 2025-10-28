"""
Atomic properties: masses, radii, and lattice parameters.

All values from standard references (NIST, crystallographic databases).
DO NOT change these values.
"""
# ============================================================================
# Metallic Radii (pm)
# ============================================================================

# Metallic radii (12-fold/close-packed "metallic radius") in pm.
# These are standard tabulated metallic radii.
# References: Kittel, Introduction to Solid State Physics; 
#             Ashcroft & Mermin, Solid State Physics
METALLIC_RADII_PM = {
    "AL": 143.0,
    "MG": 160.0,
    "CU": 128.0,
    "ZN": 134.0,
    "NI": 124.0,
    "FE": 126.0,
    "CO": 125.0,
    "CR": 128.0,
    "MN": 127.0,
    "SI": 117.0,
    "TI": 147.0,
    "V": 134.0,
    "ZR": 160.0,
    "NB": 146.0,
}

# ============================================================================
# Lattice Parameters (Å)
# ============================================================================

# Lattice parameters in Å at ~300 K for common fcc matrices.
# References: Standard crystallographic databases (ICSD, Pearson's)
FCC_LATTICE_PARAMS_A = {
    # Al: ~4.046 Å at ~300 K
    "AL": 4.046,
    # Ni: ~3.499 Å at ~300 K (typical experimental a_Ni≈3.52 Å at RT)
    "NI": 3.52,
    # Cu: ~3.597 Å at ~300 K (fcc Cu lattice parameter near 3.595-3.61 Å at RT)
    "CU": 3.60,
}

