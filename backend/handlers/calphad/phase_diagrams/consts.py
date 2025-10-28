"""
Utility functions for CALPHAD phase diagrams.
"""

from mendeleev import element


def weight_to_mole_fraction(weight_fractions: dict) -> dict:
    """
    Convert weight fractions to mole fractions using mendeleev for atomic masses.
    
    Args:
        weight_fractions: Dict mapping element symbols to weight fractions (must sum to 1)
        
    Returns:
        Dict mapping element symbols to mole fractions
        
    Raises:
        ValueError: If atomic mass not available for an element
    """
    # Calculate moles for each element
    moles = {}
    for elem, wt_frac in weight_fractions.items():
        try:
            # Normalize to capitalized format (e.g., 'AL' -> 'Al', 'FE' -> 'Fe')
            elem_normalized = elem.capitalize()
            elem_obj = element(elem_normalized)
            moles[elem] = wt_frac / elem_obj.mass
        except Exception as e:
            raise ValueError(f"Could not get atomic mass for element '{elem}': {e}")
    
    # Total moles
    total_moles = sum(moles.values())
    
    # Convert to mole fractions
    mole_fractions = {}
    for elem, mol in moles.items():
        mole_fractions[elem] = mol / total_moles
    
    return mole_fractions