# Periodic masses (g/mol) for common elements
# From standard atomic weights (NIST)
_ATOMIC_MASS = {
    "AL": 26.98154,  # Aluminum
    "B": 10.81,      # Boron
    "C": 12.011,     # Carbon
    "CR": 51.996,    # Chromium
    "CU": 63.546,    # Copper
    "FE": 55.847,    # Iron
    "LI": 6.94,      # Lithium
    "MG": 24.305,    # Magnesium
    "MN": 54.9380,   # Manganese
    "N": 14.007,     # Nitrogen
    "NI": 58.69,     # Nickel
    "O": 15.999,     # Oxygen
    "SC": 44.956,    # Scandium
    "SI": 28.0855,   # Silicon
    "TI": 47.88,     # Titanium
    "ZN": 65.38,     # Zinc
    "ZR": 91.224     # Zirconium
}


def weight_to_mole_fraction(weight_fractions: dict) -> dict:
    """
    Convert weight fractions to mole fractions.
    
    Args:
        weight_fractions: Dict mapping element symbols to weight fractions (must sum to 1)
        
    Returns:
        Dict mapping element symbols to mole fractions
        
    Raises:
        ValueError: If atomic mass not available for an element
    """
    # Check all elements have atomic masses
    for elem in weight_fractions.keys():
        if elem.upper() not in _ATOMIC_MASS:
            raise ValueError(
                f"Atomic mass not available for element '{elem}'. "
                f"Available elements: {sorted(_ATOMIC_MASS.keys())}"
            )
    
    # Calculate moles for each element
    moles = {}
    for elem, wt_frac in weight_fractions.items():
        elem_upper = elem.upper()
        moles[elem_upper] = wt_frac / _ATOMIC_MASS[elem_upper]
    
    # Total moles
    total_moles = sum(moles.values())
    
    # Convert to mole fractions
    mole_fractions = {}
    for elem, mol in moles.items():
        mole_fractions[elem] = mol / total_moles
    
    return mole_fractions

# Element name aliases for common variations
STATIC_ALIASES = {
    # Vacuum
    "vac": "VA", "va": "VA", "vacuum": "VA",
    
    # Aluminum
    "al": "AL", "aluminum": "AL", "aluminium": "AL",
    
    # Chromium
    "cr": "CR", "chromium": "CR", "chrome": "CR",
    
    # Copper
    "cu": "CU", "copper": "CU",
    
    # Iron
    "fe": "FE", "iron": "FE",
    
    # Magnesium
    "mg": "MG", "magnesium": "MG",
    
    # Manganese
    "mn": "MN", "manganese": "MN",
    
    # Nickel
    "ni": "NI", "nickel": "NI",
    
    # Scandium
    "sc": "SC", "scandium": "SC",
    
    # Silicon
    "si": "SI", "silicon": "SI",
    
    # Titanium
    "ti": "TI", "titanium": "TI",
    
    # Zinc
    "zn": "ZN", "zinc": "ZN",
    
    # Zirconium
    "zr": "ZR", "zirconium": "ZR"
}