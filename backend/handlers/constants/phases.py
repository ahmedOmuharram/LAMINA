"""
CALPHAD phase classification and categorization.

Maps raw phase names from thermodynamic databases to metallurgical categories
and readable names for interpretation.

DO NOT change these mappings without metallurgical justification.
"""

from enum import Enum


# ============================================================================
# Phase Categories
# ============================================================================

class PhaseCategory(Enum):
    """Metallurgical phase categories for interpretation."""
    PRIMARY_FCC = "primary_fcc"           # FCC solid solution (Al-rich, Cu-rich, etc.)
    PRIMARY_BCC = "primary_bcc"           # BCC solid solution (Fe-rich, Ti-beta, etc.)
    PRIMARY_HCP = "primary_hcp"           # HCP solid solution (Mg-rich, Zn-rich, Ti-alpha)
    TAU_PHASE = "tau"                     # Ternary tau phase (Al-Mg-Zn)
    GAMMA = "gamma"                       # Gamma phase (Al12Mg17, Ni3Al, etc.)
    LAVES = "laves"                       # Laves-type phases (MgZn2, etc.)
    SIGMA = "sigma"                       # Sigma phase (Fe-Cr)
    CARBIDE = "carbide"                   # Carbides (M23C6, M7C3, etc.)
    NITRIDE = "nitride"                   # Nitrides (AlN, TiN, etc.)
    PRECIPITATE = "precipitate"           # Generic precipitate
    INTERMETALLIC = "intermetallic"       # Generic intermetallic
    LIQUID = "liquid"                     # Liquid phase
    OTHER = "other"                       # Unclassified


# ============================================================================
# Phase Classification
# ============================================================================

# Database-specific phase classification rules
# Maps base phase names (without #1, #2 suffixes) to (readable_name, category, structure)
PHASE_CLASSIFICATION = {
    # ===== FCC phases =====
    "FCC_A1": ("fcc solid solution", PhaseCategory.PRIMARY_FCC, "fcc"),
    "FCC": ("fcc solid solution", PhaseCategory.PRIMARY_FCC, "fcc"),
    "ALUMINUM": ("Al-rich fcc", PhaseCategory.PRIMARY_FCC, "fcc"),
    "AL(FCC)": ("Al-rich fcc", PhaseCategory.PRIMARY_FCC, "fcc"),
    "CU(FCC)": ("Cu-rich fcc", PhaseCategory.PRIMARY_FCC, "fcc"),
    "NI(FCC)": ("Ni-rich fcc", PhaseCategory.PRIMARY_FCC, "fcc"),
    
    # ===== BCC phases =====
    "BCC_A2": ("bcc solid solution", PhaseCategory.PRIMARY_BCC, "bcc"),
    "BCC": ("bcc solid solution", PhaseCategory.PRIMARY_BCC, "bcc"),
    "FERRITE": ("ferrite (bcc)", PhaseCategory.PRIMARY_BCC, "bcc"),
    "BCC_B2": ("ordered bcc (B2)", PhaseCategory.PRIMARY_BCC, "bcc-ordered"),
    
    # ===== HCP phases =====
    "HCP_A3": ("hcp solid solution", PhaseCategory.PRIMARY_HCP, "hcp"),
    "HCP": ("hcp solid solution", PhaseCategory.PRIMARY_HCP, "hcp"),
    "HCP_ZN": ("Zn-rich hcp", PhaseCategory.PRIMARY_HCP, "hcp"),
    "HCP_MG": ("Mg-rich hcp", PhaseCategory.PRIMARY_HCP, "hcp"),
    
    # ===== Al-Mg-Zn ternary phases =====
    "TAU_PHASE": ("tau phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "TAU": ("tau phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "T_PHASE": ("T phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "T": ("T phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "MGALZN_T": ("tau phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "AL2MG3ZN3": ("tau phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    "MG32(AL,ZN)49": ("tau phase (Al-Mg-Zn)", PhaseCategory.TAU_PHASE, "complex"),
    
    # ===== Gamma phases (Al-Mg, Ni-Al, etc.) =====
    "GAMMA": ("gamma phase", PhaseCategory.GAMMA, "complex"),
    "AL12MG17": ("gamma-Al12Mg17", PhaseCategory.GAMMA, "bcc-like"),
    "ALMG_GAMMA": ("gamma-Al12Mg17", PhaseCategory.GAMMA, "bcc-like"),
    "GAMMA_PRIME": ("gamma' precipitate", PhaseCategory.GAMMA, "fcc-ordered"),
    "NI3AL": ("Ni3Al (gamma')", PhaseCategory.GAMMA, "fcc-ordered"),
    
    # ===== Laves phases =====
    "C14_LAVES": ("Laves C14 (MgZn2)", PhaseCategory.LAVES, "hexagonal"),
    "C15_LAVES": ("Laves C15", PhaseCategory.LAVES, "cubic"),
    "C36_LAVES": ("Laves C36", PhaseCategory.LAVES, "hexagonal"),
    "LAVES_C14": ("Laves C14 (COST507)", PhaseCategory.LAVES, "hexagonal"),
    "MGZN2": ("MgZn2 Laves", PhaseCategory.LAVES, "hexagonal"),
    "MGZN2_C14": ("MgZn2 C14", PhaseCategory.LAVES, "hexagonal"),
    "MG7ZN3": ("Mg7Zn3 (related to Laves)", PhaseCategory.LAVES, "hexagonal"),
    
    # ===== Intermetallics =====
    "AL3MG2": ("Al3Mg2 (beta)", PhaseCategory.INTERMETALLIC, "fcc-like"),
    "BETA": ("beta phase", PhaseCategory.INTERMETALLIC, "complex"),
    "AL3NI": ("Al3Ni", PhaseCategory.INTERMETALLIC, "orthorhombic"),
    "AL3NI2": ("Al3Ni2", PhaseCategory.INTERMETALLIC, "hexagonal"),
    "THETA": ("theta phase (Al2Cu)", PhaseCategory.INTERMETALLIC, "tetragonal"),
    "AL2CU": ("Al2Cu (theta)", PhaseCategory.INTERMETALLIC, "tetragonal"),
    
    # ===== Sigma phase =====
    "SIGMA": ("sigma phase", PhaseCategory.SIGMA, "tetragonal"),
    
    # ===== Liquid =====
    "LIQUID": ("liquid", PhaseCategory.LIQUID, None),
    
    # Add more as needed for your databases
}

