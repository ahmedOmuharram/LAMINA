"""
Elastic modulus constants for elements and temperature corrections.

Young's modulus values at room temperature (298K) for isotropic polycrystalline materials.
Sources: ASM Handbook, CRC Materials Science & Engineering Handbook, SpringerMaterials.
"""

# ============================================================================
# Young's Modulus (GPa) - Room Temperature (298K)
# ============================================================================

# Comprehensive element moduli map (GPa)
# Covers most metallic elements used in CALPHAD databases
ELEMENT_MODULUS_GPA = {
    # Light metals
    "AL": 70.0,    # Aluminum
    "MG": 45.0,    # Magnesium
    "BE": 287.0,   # Beryllium
    "LI": 4.9,     # Lithium
    "NA": 10.0,    # Sodium
    "K": 3.2,      # Potassium
    "CA": 20.0,    # Calcium
    "SR": 15.7,    # Strontium
    "BA": 13.0,    # Barium
    
    # Common alloying elements
    "CU": 117.0,   # Copper
    "ZN": 83.0,    # Zinc
    "FE": 210.0,   # Iron
    "NI": 170.0,   # Nickel
    "TI": 116.0,   # Titanium
    "CR": 279.0,   # Chromium
    "MN": 191.0,   # Manganese
    "SI": 165.0,   # Silicon
    "CO": 211.0,   # Cobalt
    
    # Refractory metals
    "MO": 329.0,   # Molybdenum
    "W": 411.0,    # Tungsten
    "V": 128.0,    # Vanadium
    "NB": 105.0,   # Niobium
    "TA": 186.0,   # Tantalum
    "ZR": 99.0,    # Zirconium
    "HF": 128.0,   # Hafnium
    "RE": 69.0,    # Rhenium
    
    # Precious metals
    "PD": 121.0,   # Palladium
    "PT": 168.0,   # Platinum
    "AG": 83.0,    # Silver
    "AU": 78.0,    # Gold
    "RH": 379.0,   # Rhodium
    "IR": 528.0,   # Iridium
    "RU": 447.0,   # Ruthenium
    "OS": 550.0,   # Osmium
    
    # Post-transition metals
    "SN": 50.0,    # Tin
    "PB": 16.0,    # Lead
    "GA": 70.0,    # Gallium
    "GE": 103.0,   # Germanium
    "IN": 11.0,    # Indium
    "TL": 8.0,     # Thallium
    "BI": 32.0,    # Bismuth
    "SB": 55.0,    # Antimony
    
    # Rare earths (commonly in CALPHAD databases)
    "Y": 63.0,     # Yttrium
    "LA": 37.0,    # Lanthanum
    "CE": 34.0,    # Cerium
    "PR": 37.0,    # Praseodymium
    "ND": 41.0,    # Neodymium
    "SM": 50.0,    # Samarium
    "EU": 18.0,    # Europium
    "GD": 55.0,    # Gadolinium
    "TB": 56.0,    # Terbium
    "DY": 61.0,    # Dysprosium
    "HO": 65.0,    # Holmium
    "ER": 70.0,    # Erbium
    "TM": 74.0,    # Thulium
    "YB": 24.0,    # Ytterbium
    "LU": 69.0,    # Lutetium
    "SC": 74.0,    # Scandium
    
    # Other elements that may appear in alloys
    "CD": 50.0,    # Cadmium
    "HG": 25.0,    # Mercury (solid at low T)
    "AS": 58.0,    # Arsenic
    "SE": 58.0,    # Selenium
    "TE": 43.0,    # Tellurium
    
    # Special cases
    "H": 0.0,      # Hydrogen (not applicable for bulk modulus)
    "VA": 0.0,     # Vacancy (CALPHAD placeholder)
}

# ============================================================================
# Temperature Dependence of Young's Modulus
# ============================================================================

# Linear temperature coefficient: dE/dT per degree K
# Approximate values: E(T) ≈ E(RT) * (1 + SLOPE * (T - 298))
# Most metals soften with increasing temperature (negative slope)
# Values in units of (1/K), typical range: -5e-4 to -2e-4
TEMP_COEFFICIENT_PER_K = {
    # Light metals
    "AL": -4.0e-4,   # Aluminum: ~-0.04% per K
    "MG": -3.5e-4,   # Magnesium
    "BE": -2.0e-4,   # Beryllium
    "LI": -5.0e-4,   # Lithium
    
    # Common alloying elements
    "CU": -3.0e-4,   # Copper
    "ZN": -3.5e-4,   # Zinc
    "FE": -2.5e-4,   # Iron
    "NI": -2.0e-4,   # Nickel
    "TI": -2.5e-4,   # Titanium
    "CR": -1.8e-4,   # Chromium (harder, less T-dependent)
    "MN": -2.5e-4,   # Manganese
    "SI": -1.5e-4,   # Silicon (very stiff)
    "CO": -2.0e-4,   # Cobalt
    
    # Refractory metals (higher melting point → less T-dependence)
    "MO": -1.5e-4,   # Molybdenum
    "W": -1.2e-4,    # Tungsten
    "V": -2.0e-4,    # Vanadium
    "NB": -2.5e-4,   # Niobium
    "TA": -1.8e-4,   # Tantalum
    "ZR": -2.5e-4,   # Zirconium
    "HF": -2.0e-4,   # Hafnium
    
    # Precious metals
    "PD": -3.0e-4,   # Palladium
    "PT": -2.5e-4,   # Platinum
    "AG": -3.5e-4,   # Silver
    "AU": -3.5e-4,   # Gold
    
    # Post-transition metals (softer → more T-dependent)
    "SN": -4.5e-4,   # Tin
    "PB": -5.0e-4,   # Lead
    "GA": -4.0e-4,   # Gallium
    "IN": -5.0e-4,   # Indium
    
    # Rare earths
    "Y": -3.0e-4,    # Yttrium
    "LA": -3.5e-4,   # Lanthanum
    "CE": -3.5e-4,   # Cerium
    "ND": -3.0e-4,   # Neodymium
    "GD": -2.5e-4,   # Gadolinium
}

# ============================================================================
# Reference Temperature
# ============================================================================

ROOM_TEMPERATURE_K = 298.15  # Standard reference temperature (25°C)

# ============================================================================
# Utility Sets
# ============================================================================

# Elements with known modulus data
KNOWN_MODULUS_ELEMENTS = set(ELEMENT_MODULUS_GPA.keys())

# Elements with known temperature dependence
KNOWN_TEMP_COEFF_ELEMENTS = set(TEMP_COEFFICIENT_PER_K.keys())

