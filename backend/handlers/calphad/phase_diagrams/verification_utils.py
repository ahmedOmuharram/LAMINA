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
from functools import lru_cache

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
    _log.debug(f"Parsing composition: '{comp_str}' with elements {elements}")
    comp_dict = {}
    comp_str = comp_str.replace(' ', '').replace('_', '').upper()
    _log.debug(f"Cleaned composition string: '{comp_str}'")
    
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
        _log.debug(f"Found {len(comp_dict)}/{len(elements)} elements in first pass, trying split method")
        comp_dict = {}
    
    # If we didn't find explicit percentages, try parsing "Al-8Mg-4Zn" format
    if not comp_dict:
        parts = re.split(r'[-,]', comp_str)
        _log.debug(f"Split parts: {parts}")
        for part in parts:
            _log.debug(f"Processing part: '{part}'")
            # Try two patterns: "AL8" or "8AL"
            # Pattern 1: Element followed by number (AL8)
            match = re.match(r'^([A-Z][A-Z]?)(\d+\.?\d*)$', part)
            if match:
                el = match.group(1)
                pct = float(match.group(2))
                _log.debug(f"  Pattern 1 matched: el={el}, pct={pct}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = pct
                continue
            
            # Pattern 2: Number followed by element (8AL)
            match = re.match(r'^(\d+\.?\d*)([A-Z][A-Z]?)$', part)
            if match:
                pct = float(match.group(1))
                el = match.group(2)
                _log.debug(f"  Pattern 2 matched: pct={pct}, el={el}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = pct
                continue
            
            # Pattern 3: Element only (no number - will be balance)
            match = re.match(r'^([A-Z][A-Z]?)$', part)
            if match:
                el = match.group(1)
                _log.debug(f"  Pattern 3 matched: el={el}, in_elements={el in [e.upper() for e in elements]}")
                if el in [e.upper() for e in elements]:
                    comp_dict[el] = None
                continue
            
            _log.debug(f"  No pattern matched for part '{part}'")
    
    # Handle case where one element doesn't have a number (it's the balance)
    if None in comp_dict.values():
        specified_total = sum(v for v in comp_dict.values() if v is not None)
        _log.debug(f"Calculating balance: specified_total={specified_total}%, balance={100.0-specified_total}%")
        for el, val in comp_dict.items():
            if val is None:
                comp_dict[el] = 100.0 - specified_total
    
    _log.debug(f"Parsed composition dict (before normalization): {comp_dict}")
    
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
            _log.debug(f"Normalizing composition from {total}% to 100%")
            comp_dict = {el: (val / total) * 100.0 for el, val in comp_dict.items()}
        else:
            _log.error(f"Invalid composition: total = {total}%")
            return None
    
    _log.debug(f"Final parsed composition: {comp_dict} (total: {sum(comp_dict.values()):.1f}%)")
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

def _phase_constituent_sets(db: Database, phase_name: str):
    """Return a list[set[str]]: one set of occupant symbols per sublattice."""
    cons = db.phases[phase_name].constituents  # tuple of tuples of Species
    subl_sets = []
    for subl in cons:
        s = set()
        for sp in subl:
            # Species can be elements (AL) or VA; in metallic COST507 that’s typical.
            # Fall back to .name when .element not present.
            sym = getattr(getattr(sp, 'element', None), 'symbol', None) or getattr(sp, 'name', str(sp))
            s.add(sym.upper())
        subl_sets.append(s)
    return subl_sets

def _phase_is_usable(db: Database, phase_name: str, allowed: set[str]) -> bool:
    """Usable if every sublattice offers at least one allowed occupant."""
    for subl in _phase_constituent_sets(db, phase_name):
        if len(subl & allowed) == 0:
            return False
    return True

@lru_cache(maxsize=64)
def _get_phases_for_elements_cached(
    db_id: int, 
    elems_tuple: tuple, 
    all_phases_tuple: tuple,
    always_include_tuple: tuple,
    always_exclude_tuple: tuple
) -> tuple:
    """
    Cached implementation of phase filtering.
    Returns tuple of phase names that are usable for the given elements.
    
    Args:
        db_id: id(db) to distinguish different databases
        elems_tuple: Tuple of element symbols (for hashability)
        all_phases_tuple: Tuple of all phase names in database
        always_include_tuple: Tuple of phases to always include
        always_exclude_tuple: Tuple of phases to always exclude
    
    Note: This is cached separately because Database objects aren't hashable.
    The caller must ensure db_id matches the actual database being used.
    """
    # This function signature is cached, but we need the actual Database object
    # to check phase constituents. The caller will need to pass it via a closure
    # or thread-local storage. For now, we'll just return a marker.
    # The actual implementation is in get_phases_for_elements.
    return ()  # Placeholder - actual logic is in get_phases_for_elements

def get_phases_for_elements(
    db: Database,
    elements: List[str],
    phase_elements_func=None,  # kept for API compatibility, no longer used
    always_include: Optional[List[str]] = None,
    always_exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Select phases that are *usable* for the given elements: each sublattice
    must have at least one occupant from allowed (elements + VA).
    
    Results are cached based on database identity and element set for performance.
    """
    allowed = {el.upper() for el in elements} | {"VA"}
    # Convert phase names to regular Python strings to avoid numpy.str_ issues
    all_phases = [str(p) for p in db.phases.keys()]
    always_include = set((always_include or []) + ["LIQUID", "FCC_A1", "HCP_A3", "BCC_A2", "BCC_B2"])
    always_exclude = set(always_exclude or []) | {"GAS"}  # add others if you want

    # Try to use cache key based on db identity and elements
    cache_key = (id(db), tuple(sorted(el.upper() for el in elements)))
    
    # Check if we have a cached result (using a simple dict cache on the function)
    # Cache version to invalidate old entries with numpy.str_ objects
    CACHE_VERSION = "v0.0.6_prf_getitem_patch"
    if not hasattr(get_phases_for_elements, '_cache'):
        get_phases_for_elements._cache = {}
        get_phases_for_elements._cache_version = CACHE_VERSION
        _log.info(f"Initialized phase cache with version {CACHE_VERSION}")
    elif getattr(get_phases_for_elements, '_cache_version', None) != CACHE_VERSION:
        # Invalidate old cache if version changed
        old_version = getattr(get_phases_for_elements, '_cache_version', 'unknown')
        old_size = len(get_phases_for_elements._cache)
        get_phases_for_elements._cache = {}
        get_phases_for_elements._cache_version = CACHE_VERSION
        _log.warning(f"Cache version changed from {old_version} to {CACHE_VERSION} - cleared {old_size} old entries")
    
    if cache_key in get_phases_for_elements._cache:
        cached_result = get_phases_for_elements._cache[cache_key]
        _log.debug(f"Using cached phase list for {'-'.join(elements)}: {len(cached_result)} phases")
        
        # Safety check: ensure cached results are regular Python strings
        import numpy as np
        if any(isinstance(p, np.str_) for p in cached_result):
            _log.error(f"CRITICAL: Cached phases contain numpy.str_ objects! Re-filtering...")
            # Don't return the bad cache, fall through to regenerate
        else:
            return cached_result

    relevant = []
    _log.debug(f"Filtering phases for elements: {elements}, allowed: {allowed}")

    for ph in all_phases:
        if ph in always_exclude:
            continue
        if ph in always_include:
            relevant.append(ph)
            continue

        usable = False
        try:
            usable = _phase_is_usable(db, ph, allowed)
        except Exception as e:
            _log.debug(f"  ! Could not inspect constituents for {ph}: {e}")

        if usable:
            relevant.append(ph)

    result = relevant if relevant else all_phases
    _log.debug(f"Selected {len(result)} phases for {'-'.join(elements)} system")
    
    # Ensure all phase names are regular Python strings before caching
    result = [str(p) for p in result]
    
    # Final safety check: verify no numpy.str_ objects
    import numpy as np
    if any(isinstance(p, np.str_) for p in result):
        _log.error(f"CRITICAL BUG: Result still contains numpy.str_ after str() conversion!")
        # Force convert again
        result = [str(p) if isinstance(p, np.str_) else p for p in result]
    
    # Cache the result
    get_phases_for_elements._cache[cache_key] = result
    
    # Limit cache size to prevent memory issues
    if len(get_phases_for_elements._cache) > 64:
        # Remove oldest entry (first in dict)
        first_key = next(iter(get_phases_for_elements._cache))
        del get_phases_for_elements._cache[first_key]
    
    return result
