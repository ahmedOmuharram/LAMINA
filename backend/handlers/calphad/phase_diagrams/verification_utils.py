"""
Utility functions for CALPHAD verification and fact-checking.

Contains helper functions for:
- Element symbol normalization
- Composition string parsing
- Phase categorization
- Phase filtering for multicomponent systems
"""
import re
import logging
from typing import Dict, List, Optional

from pycalphad import Database

_log = logging.getLogger(__name__)


def parse_composition_string(comp_str: str, elements: List[str]) -> Optional[Dict[str, float]]:
    """
    Parse composition string into {element: at%} dict.
    
    Supports multiple formats:
    - "Al88Mg8Zn4" -> {"AL": 88, "MG": 8, "ZN": 4}
    - "88Al-8Mg-4Zn" -> {"AL": 88, "MG": 8, "ZN": 4}
    - "Al-8Mg-4Zn" -> {"AL": remaining, "MG": 8, "ZN": 4}
    
    Args:
        comp_str: Composition string
        elements: List of element symbols expected in the composition
        
    Returns:
        Dictionary mapping element symbols to atomic percentages, or None if parsing fails
    """
    _log.info(f"Parsing composition: '{comp_str}' with elements {elements}")
    comp_dict = {}
    comp_str = comp_str.replace(' ', '').replace('_', '').upper()
    _log.info(f"Cleaned composition string: '{comp_str}'")
    
    # Try pattern like "AL88MG8ZN4" or "88AL8MG4ZN"
    for element in elements:
        el_upper = element.upper()
        # Look for patterns like "AL88" or "88AL"
        pattern1 = rf"{el_upper}(\d+\.?\d*)"  # AL88
        pattern2 = rf"(\d+\.?\d*){el_upper}"  # 88AL
        
        match = re.search(pattern1, comp_str)
        if not match:
            match = re.search(pattern2, comp_str)
        
        if match:
            comp_dict[el_upper] = float(match.group(1))
    
    # If we found some but not all elements, it might be hyphen-separated format
    # Clear and try the split method instead
    if comp_dict and len(comp_dict) < len(elements):
        _log.info(f"Found {len(comp_dict)}/{len(elements)} elements in first pass, trying split method")
        comp_dict = {}
    
    # If we didn't find explicit percentages, try parsing "Al-8Mg-4Zn" format
    if not comp_dict:
        parts = re.split(r'[-,]', comp_str)
        _log.info(f"Split parts: {parts}")
        for part in parts:
            _log.info(f"Processing part: '{part}'")
            # Try two patterns: "AL8" or "8AL"
            # Pattern 1: Element followed by number (AL8)
            match = re.match(r'^([A-Z][A-Z]?)(\d+\.?\d*)$', part)
            if match:
                el = match.group(1)
                pct = float(match.group(2))
                _log.info(f"  Pattern 1 matched: el={el}, pct={pct}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = pct
                continue
            
            # Pattern 2: Number followed by element (8AL)
            match = re.match(r'^(\d+\.?\d*)([A-Z][A-Z]?)$', part)
            if match:
                pct = float(match.group(1))
                el = match.group(2)
                _log.info(f"  Pattern 2 matched: pct={pct}, el={el}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = pct
                continue
            
            # Pattern 3: Element only (no number - will be balance)
            match = re.match(r'^([A-Z][A-Z]?)$', part)
            if match:
                el = match.group(1)
                _log.info(f"  Pattern 3 matched: el={el}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = None
                continue
            
            _log.warning(f"  No pattern matched for part '{part}'")
    
    # Handle case where one element doesn't have a number (it's the balance)
    if None in comp_dict.values():
        specified_total = sum(v for v in comp_dict.values() if v is not None)
        _log.info(f"Calculating balance: specified_total={specified_total}%, balance={100.0-specified_total}%")
        for el, val in comp_dict.items():
            if val is None:
                comp_dict[el] = 100.0 - specified_total
    
    _log.info(f"Parsed composition dict (before normalization): {comp_dict}")
    
    # Validate
    if not comp_dict:
        return None
    
    total = sum(comp_dict.values())
    
    # Normalize composition to sum to 100%
    if abs(total - 100.0) > 0.1:
        # Try normalizing if close to 1.0 (mole fractions given)
        if 0.9 < total < 1.1:
            comp_dict = {el: val * 100 for el, val in comp_dict.items()}
        elif total > 0:
            # Normalize to 100% if it's off (e.g., 96% -> scale to 100%)
            _log.info(f"Normalizing composition from {total}% to 100%")
            comp_dict = {el: (val / total) * 100.0 for el, val in comp_dict.items()}
        else:
            _log.error(f"Invalid composition: total = {total}%")
            return None
    
    _log.info(f"Final parsed composition: {comp_dict} (total: {sum(comp_dict.values()):.1f}%)")
    return comp_dict if comp_dict else None


def map_phase_to_category(phase_name: str):
    """
    Map phase name to PhaseCategory enum.
    
    Args:
        phase_name: Phase name (e.g., 'fcc', 'tau', 'laves')
        
    Returns:
        PhaseCategory enum value
    """
    from .fact_checker import PhaseCategory
    
    phase_lower = phase_name.lower().strip()
    
    # Map common names to categories
    mapping = {
        'fcc': PhaseCategory.PRIMARY_FCC,
        'fcc_a1': PhaseCategory.PRIMARY_FCC,
        'bcc': PhaseCategory.PRIMARY_BCC,
        'bcc_a2': PhaseCategory.PRIMARY_BCC,
        'hcp': PhaseCategory.PRIMARY_HCP,
        'hcp_a3': PhaseCategory.PRIMARY_HCP,
        'tau': PhaseCategory.TAU_PHASE,
        'tau_phase': PhaseCategory.TAU_PHASE,
        't_phase': PhaseCategory.TAU_PHASE,
        't': PhaseCategory.TAU_PHASE,
        'mgalzn_t': PhaseCategory.TAU_PHASE,
        'al2mg3zn3': PhaseCategory.TAU_PHASE,
        'gamma': PhaseCategory.GAMMA,
        'gamma_prime': PhaseCategory.GAMMA,
        'laves': PhaseCategory.LAVES,
        'c14': PhaseCategory.LAVES,
        'c15': PhaseCategory.LAVES,
        'mgzn2': PhaseCategory.LAVES,
        'sigma': PhaseCategory.SIGMA,
        'liquid': PhaseCategory.LIQUID,
    }
    
    return mapping.get(phase_lower, PhaseCategory.OTHER)


def get_phases_for_elements(db: Database, elements: List[str], phase_elements_func) -> List[str]:
    """
    Get relevant phases for a set of elements - only phases with EXACTLY our elements (no extras).
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols
        phase_elements_func: Function to extract elements from a phase (typically self._phase_elements)
        
    Returns:
        List of phase names that contain only the specified elements
    """
    # For binary systems, caller should use _filter_phases_for_system instead
    if len(elements) == 2:
        _log.warning("get_phases_for_elements called for binary system - consider using _filter_phases_for_system instead")
    
    # For ternary/higher, include ONLY phases that contain exclusively our elements (+ VA)
    all_phases = list(db.phases.keys())
    relevant = []
    allowed_elements = set(el.upper() for el in elements) | {"VA"}  # Our elements + vacancy
    
    _log.info(f"Filtering phases for elements: {elements}, allowed: {allowed_elements}")
    
    for phase_name in all_phases:
        # Always include LIQUID (will be excluded at low T in caller)
        if phase_name == "LIQUID":
            relevant.append(phase_name)
            continue
        
        # Get elements in this phase
        phase_els = phase_elements_func(db, phase_name)
        
        # Only include if phase contains ONLY our elements (no extras like Cr, Cu)
        # Phase must have at least one of our elements AND no forbidden elements
        has_our_elements = any(el in allowed_elements for el in phase_els)
        has_forbidden_elements = any(el not in allowed_elements for el in phase_els)
        
        # Special case: always include common phases and known ternary phases
        is_common_phase = phase_name in ["FCC_A1", "BCC_A2", "BCC_B2", "HCP_A3", "HCP_ZN"]
        is_ternary_phase = phase_name in ["TAU", "TAU_PHASE", "T_PHASE", "PHI", "VPHASE"]
        
        if (has_our_elements and not has_forbidden_elements) or is_common_phase or is_ternary_phase:
            relevant.append(phase_name)
            _log.debug(f"  ✓ Including phase {phase_name}: elements={phase_els}")
        else:
            _log.debug(f"  ✗ Excluding phase {phase_name}: elements={phase_els} (forbidden elements present)")
    
    _log.info(f"Selected {len(relevant)} phases for {'-'.join(elements)} system")
    return relevant if relevant else all_phases

