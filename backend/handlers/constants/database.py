"""
Database-specific utilities: exclusion patterns and phase name mappings.

These control which phases are filtered from analysis and how phase names
are normalized for display.

DO NOT change these without understanding database-specific behavior.
"""

# ============================================================================
# Phase Exclusion Patterns
# ============================================================================

# Exclude clearly non-equilibrium / cluster / GP phases by default
# These are regex patterns matched against phase names
EXCLUDE_PHASE_PATTERNS = (
    r'^GP_', r'_GP', r'^CL_', r'_DP$', r'^B_PRIME', r'^PRE_', r'^THETA_PRIME$',
    r'^TH_DP', r'^U1_PHASE$', r'^U2_PHASE$',
    r'^FCCAL$', r'^FCCMG$', r'^FCCSI$',  # Helper phases for thermodynamic calculations only
)

# ============================================================================
# Phase Name Mappings
# ============================================================================

# Phase name mapping for better readability
# Maps database-specific phase names to standard readable names
PHASE_NAME_MAP = {
    'CSI': 'SiC',
    'C_SIC': 'SiC',
    'AL4C3': 'Al4C3',
    'AL3NI': 'Al3Ni',
    'AL3NI2': 'Al3Ni2',
}

