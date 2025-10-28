"""
Unit conversion utilities used across handlers.

This module provides conversion functions for various units used in materials science:
- Energy conversions (kJ/mol <-> eV/atom)
- Composition conversions (weight% <-> mole%, atomic% <-> mole fractions)
- Magnetic unit conversions (μB/bohr³ <-> kA/m)
- Length conversions (pm <-> Angstrom)

All conversion factors are derived from fundamental constants defined in the
constants submodule.
"""

from typing import Dict, Optional
from .constants.energetics import KJMOL_PER_EV_PER_ATOM
from .constants.physical import MUB_PER_BOHR3_TO_KA_PER_M


# ============================================================================
# Energy Conversions
# ============================================================================

def kjmol_to_ev(energy_kjmol: float) -> float:
    """
    Convert energy from kJ/mol to eV/atom.
    
    Args:
        energy_kjmol: Energy in kJ/mol
        
    Returns:
        Energy in eV/atom
        
    Example:
        >>> kjmol_to_ev(96.485)
        1.0
    """
    return float(energy_kjmol) / KJMOL_PER_EV_PER_ATOM


def ev_to_kjmol(energy_ev: float) -> float:
    """
    Convert energy from eV/atom to kJ/mol.
    
    Args:
        energy_ev: Energy in eV/atom
        
    Returns:
        Energy in kJ/mol
        
    Example:
        >>> ev_to_kjmol(1.0)
        96.485
    """
    return float(energy_ev) * KJMOL_PER_EV_PER_ATOM


# ============================================================================
# Composition Conversions
# ============================================================================

def weight_to_mole_fraction(weight_fractions: Dict[str, float]) -> Dict[str, float]:
    """
    Convert weight fractions to mole fractions using atomic masses.
    
    Uses mendeleev library to get accurate atomic masses.
    
    Args:
        weight_fractions: Dict mapping element symbols to weight fractions (must sum to 1)
        
    Returns:
        Dict mapping element symbols to mole fractions
        
    Raises:
        ValueError: If atomic mass not available for an element
        
    Example:
        >>> weight_to_mole_fraction({"AL": 0.5, "ZN": 0.5})
        {"AL": 0.584, "ZN": 0.416}
    """
    from mendeleev import element
    
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


def atpct_to_molefrac(composition_atpct: Dict[str, float]) -> Dict[str, float]:
    """
    Convert atomic percent to mole fractions.
    
    For substitutional alloys, at% ≈ mol%, so this just normalizes to sum=1.
    
    Args:
        composition_atpct: e.g., {"AL": 88, "MG": 8, "ZN": 4}
        
    Returns:
        Normalized mole fractions: {"AL": 0.88, "MG": 0.08, "ZN": 0.04}
        
    Example:
        >>> atpct_to_molefrac({"AL": 88, "MG": 8, "ZN": 4})
        {"AL": 0.88, "MG": 0.08, "ZN": 0.04}
    """
    total = sum(composition_atpct.values())
    if total == 0:
        return composition_atpct
    return {el: val / total for el, val in composition_atpct.items()}


# ============================================================================
# Magnetic Unit Conversions
# ============================================================================

def muB_per_bohr3_to_kA_per_m(val_muB_per_bohr3: float) -> float:
    """
    Convert magnetization from μB/bohr³ to kA/m (kiloampere per meter).
    
    This is the CORRECT way to compare magnetization across different materials,
    as it normalizes to a consistent volume unit. Materials Project reports
    magnetization_per_volume in μB/bohr³.
    
    Conversion factor derivation:
    - 1 μB = 9.274e-24 A·m²
    - 1 bohr = 0.529177 Å = 0.529177e-10 m
    - 1 bohr³ ≈ 1.4818e-31 m³
    - => 1 (μB / bohr³) ≈ 6.2584e7 A/m ≈ 6.2584e4 kA/m
    
    Args:
        val_muB_per_bohr3: Magnetization per volume in μB/bohr³ (from Materials Project)
        
    Returns:
        Magnetization in kA/m
        
    Example:
        >>> muB_per_bohr3_to_kA_per_m(1.0)
        62584.13
    """
    return float(val_muB_per_bohr3 * MUB_PER_BOHR3_TO_KA_PER_M)


def kA_per_m_to_muB_per_bohr3(val_kA_per_m: float) -> float:
    """
    Convert magnetization from kA/m to μB/bohr³.
    
    Inverse of muB_per_bohr3_to_kA_per_m.
    
    Args:
        val_kA_per_m: Magnetization in kA/m
        
    Returns:
        Magnetization per volume in μB/bohr³
        
    Example:
        >>> kA_per_m_to_muB_per_bohr3(62584.13)
        1.0
    """
    return float(val_kA_per_m / MUB_PER_BOHR3_TO_KA_PER_M)


# ============================================================================
# Length Conversions
# ============================================================================

def pm_to_angstrom(length_pm: float) -> float:
    """
    Convert length from picometers to Angstroms.
    
    Args:
        length_pm: Length in picometers (pm)
        
    Returns:
        Length in Angstroms (Å)
        
    Example:
        >>> pm_to_angstrom(100.0)
        1.0
    """
    return length_pm / 100.0


def angstrom_to_pm(length_A: float) -> float:
    """
    Convert length from Angstroms to picometers.
    
    Args:
        length_A: Length in Angstroms (Å)
        
    Returns:
        Length in picometers (pm)
        
    Example:
        >>> angstrom_to_pm(1.0)
        100.0
    """
    return length_A * 100.0


def to_angstrom(val: Optional[float]) -> Optional[float]:
    """
    Smart conversion to Angstroms for ambiguous units.
    
    If the value is > 4.5, assumes it's in picometers and converts.
    Otherwise, assumes it's already in Angstroms.
    
    This heuristic is useful when dealing with data sources that don't
    clearly specify units (e.g., some element databases).
    
    Args:
        val: Length value (possibly in pm or Å)
        
    Returns:
        Length in Angstroms, or None if input is None
        
    Example:
        >>> to_angstrom(100.0)  # Likely pm
        1.0
        >>> to_angstrom(1.5)    # Likely already Å
        1.5
    """
    if val is None:
        return None
    return val / 100.0 if val > 4.5 else val


# ============================================================================
# Temperature Conversions
# ============================================================================

def celsius_to_kelvin(temp_celsius: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.
    
    Args:
        temp_celsius: Temperature in °C
        
    Returns:
        Temperature in K
        
    Example:
        >>> celsius_to_kelvin(0.0)
        273.15
    """
    return temp_celsius + 273.15


def kelvin_to_celsius(temp_kelvin: float) -> float:
    """
    Convert temperature from Kelvin to Celsius.
    
    Args:
        temp_kelvin: Temperature in K
        
    Returns:
        Temperature in °C
        
    Example:
        >>> kelvin_to_celsius(273.15)
        0.0
    """
    return temp_kelvin - 273.15


def fahrenheit_to_kelvin(temp_fahrenheit: float) -> float:
    """
    Convert temperature from Fahrenheit to Kelvin.
    
    Args:
        temp_fahrenheit: Temperature in °F
        
    Returns:
        Temperature in K
        
    Example:
        >>> fahrenheit_to_kelvin(32.0)
        273.15
    """
    return (temp_fahrenheit - 32.0) * 5.0 / 9.0 + 273.15


def kelvin_to_fahrenheit(temp_kelvin: float) -> float:
    """
    Convert temperature from Kelvin to Fahrenheit.
    
    Args:
        temp_kelvin: Temperature in K
        
    Returns:
        Temperature in °F
        
    Example:
        >>> kelvin_to_fahrenheit(273.15)
        32.0
    """
    return (temp_kelvin - 273.15) * 9.0 / 5.0 + 32.0

