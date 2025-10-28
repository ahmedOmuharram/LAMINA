"""
Composition parsing utilities for alloy systems.

This module provides functions to parse composition strings in various formats
and validate them against expected element sets.

Note: These utilities could be generally useful for other handlers dealing with
multi-component systems (ternary alloys, quaternary systems, etc.).
"""
from __future__ import annotations
import logging
import re
from typing import Dict, Tuple, Optional, Set

_log = logging.getLogger(__name__)


def parse_composition_string(
    composition: str,
    expected_elements: Set[str]
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Parse composition string in multiple formats.
    
    Supported formats:
    - Format 1: "Fe30Al70" or "Al88Mg8Zn4" (element immediately followed by number)
    - Format 2: "Al-8Mg-4Zn" (element-number pairs, first element is balance)
    - Format 3: "Fe-30Al-70" (element-number pairs with all values specified)
    
    Args:
        composition: Composition string to parse
        expected_elements: Set of expected element symbols (capitalized)
        
    Returns:
        Tuple of (composition_dict, error_message):
        - composition_dict: {"Element": at.%, ...} or None if parsing failed
        - error_message: Error description or None if successful
        
    Examples:
        >>> parse_composition_string("Fe30Al70", {"Fe", "Al"})
        ({"Fe": 30.0, "Al": 70.0}, None)
        
        >>> parse_composition_string("Al-8Mg-4Zn", {"Al", "Mg", "Zn"})
        ({"Al": 88.0, "Mg": 8.0, "Zn": 4.0}, None)
    """
    comp_dict = {}
    
    # Try format 1: concatenated (e.g., "Al88Mg8Zn4")
    matches = re.findall(r'([A-Z][a-z]?)(\d+\.?\d*)', composition)
    
    if matches and len(matches) == len(expected_elements):
        # Format 1 succeeded
        for elem, pct in matches:
            comp_dict[elem] = float(pct)
        
        # Validate elements match expected
        if set(comp_dict.keys()) == expected_elements:
            return comp_dict, None
        else:
            return None, f"Parsed elements {set(comp_dict.keys())} don't match expected {expected_elements}"
    
    # Try format 2/3: hyphen-separated (e.g., "Al-8Mg-4Zn" or "Fe-30Al-70")
    parts = composition.replace(' ', '').split('-')
    
    for part in parts:
        # Extract element and number from each part
        match = re.match(r'([A-Z][a-z]?)(\d+\.?\d*)?', part)
        if match:
            elem = match.group(1)
            pct_str = match.group(2)
            
            if pct_str:
                comp_dict[elem] = float(pct_str)
            else:
                # No number means balance element (will be calculated)
                comp_dict[elem] = None
    
    # Calculate balance element if present
    balance_elem = None
    specified_total = 0.0
    
    for elem, pct in comp_dict.items():
        if pct is None:
            balance_elem = elem
        else:
            specified_total += pct
    
    if balance_elem and specified_total < 100.0:
        comp_dict[balance_elem] = 100.0 - specified_total
    elif balance_elem:
        return None, f"Specified compositions sum to {specified_total}%, exceeds 100%"
    
    # Validate composition
    if not comp_dict:
        return None, "Could not parse any element-composition pairs"
    
    if set(comp_dict.keys()) != expected_elements:
        return None, f"Parsed elements {set(comp_dict.keys())} don't match expected {expected_elements}"
    
    # Ensure all values are numeric
    if any(v is None for v in comp_dict.values()):
        return None, "Failed to parse all composition values"
    
    return comp_dict, None


def parse_system_string(system: str) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    """
    Parse system string (e.g., 'Fe-Al', 'Al-Mg-Zn').
    
    Args:
        system: System string with elements separated by hyphens
        
    Returns:
        Tuple of (element_tuple, error_message):
        - element_tuple: ("Fe", "Al") or ("Al", "Mg", "Zn"), capitalized
        - error_message: Error description or None if successful
    """
    parts = system.replace(" ", "").split("-")
    
    if len(parts) < 2:
        return None, f"Invalid system format: '{system}'. Expected at least 2 elements separated by hyphens"
    
    if len(parts) > 3:
        return None, f"System '{system}' has {len(parts)} elements. Currently only binary and ternary systems are supported"
    
    # Capitalize element symbols
    system_elems = tuple(p.capitalize() for p in parts)
    
    return system_elems, None

