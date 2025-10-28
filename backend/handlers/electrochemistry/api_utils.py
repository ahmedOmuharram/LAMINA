"""
API interaction utilities for electrochemistry handlers.

Functions for building queries and processing Materials Project API responses.
"""
import logging
from typing import Optional, List, Dict, Any, Set
from pymatgen.core import Composition

_log = logging.getLogger(__name__)


def build_electrode_query_params(
    formula: Optional[str],
    elements: Optional[str],
    working_ion: str,
    min_capacity: Optional[float],
    max_capacity: Optional[float],
    min_voltage: Optional[float],
    max_voltage: Optional[float]
) -> Dict[str, Any]:
    """
    Build query parameters for electrode search.
    
    Uses chemsys (exact chemical system) for binary/ternary element queries
    to avoid OR-logic that returns materials containing any of the elements.
    
    Args:
        formula: Chemical formula to search for
        elements: Comma-separated elements (e.g., "Al,Mg" for binary system)
        working_ion: Working ion symbol
        min_capacity: Minimum gravimetric capacity
        max_capacity: Maximum gravimetric capacity
        min_voltage: Minimum voltage
        max_voltage: Maximum voltage
        
    Returns:
        Dictionary of query parameters for MP API
    """
    query_params = {}
    
    if formula:
        query_params['formula'] = formula
    
    if elements:
        element_list = [e.strip() for e in elements.split(',')]
        
        # For small element sets (binary/ternary), use chemsys to enforce exact system
        # This avoids OR-logic where elements=['Al','Mg'] returns Mg-only materials
        if len(element_list) <= 3:
            # Sort elements for standard chemsys format (e.g., "Al-Mg")
            chemsys = "-".join(sorted(element_list))
            query_params['chemsys'] = chemsys
            _log.info(f"Using chemsys='{chemsys}' for exact {len(element_list)}-component system")
        else:
            # For larger systems, still use elements list
            query_params['elements'] = element_list
            _log.warning(f"Using elements list for {len(element_list)}-component system; consider post-filtering")
    
    if working_ion:
        query_params['working_ion'] = working_ion
    
    # Capacity range
    if min_capacity is not None or max_capacity is not None:
        query_params['capacity_grav'] = (
            min_capacity or 0,
            max_capacity or 10000
        )
    
    # Voltage range  
    if min_voltage is not None or max_voltage is not None:
        query_params['average_voltage'] = (
            min_voltage or 0,
            max_voltage or 10
        )
    
    return query_params


def get_framework_elements(framework: Any) -> Set[str]:
    """
    Extract element symbols from a framework composition.
    
    Args:
        framework: Framework composition (string, Composition, or object with __str__)
        
    Returns:
        Set of element symbols in the framework
        
    Example:
        get_framework_elements("AlMg") -> {'Al', 'Mg'}
        get_framework_elements("MgSbPd") -> {'Mg', 'Sb', 'Pd'}
    """
    try:
        if isinstance(framework, Composition):
            comp = framework
        else:
            comp = Composition(str(framework))
        return {el.symbol for el in comp.elements}
    except Exception as e:
        _log.warning(f"Could not parse framework composition '{framework}': {e}")
        return set()


def filter_electrodes_by_framework(
    electrodes: List[Dict[str, Any]],
    allowed_elements: Set[str]
) -> List[Dict[str, Any]]:
    """
    Filter electrodes to only those whose framework contains only allowed elements.
    
    This is critical for verifying query results, as MP's elements parameter
    may use OR logic and return materials with only a subset of requested elements.
    
    Args:
        electrodes: List of electrode data dictionaries
        allowed_elements: Set of allowed element symbols (e.g., {'Al', 'Mg'})
        
    Returns:
        Filtered list containing only electrodes with frameworks ⊆ allowed_elements
        
    Example:
        allowed = {'Al', 'Mg'}
        electrodes with 'AlMg' framework → kept
        electrodes with 'MgSbPd' framework → filtered out (contains Sb, Pd)
    """
    filtered = []
    for electrode in electrodes:
        framework = electrode.get('framework')
        if not framework:
            _log.debug(f"Electrode {electrode.get('battery_id')} has no framework, skipping")
            continue
        
        framework_els = get_framework_elements(framework)
        if framework_els <= allowed_elements:  # subset check
            filtered.append(electrode)
            _log.debug(f"Kept electrode {electrode.get('battery_id')}: framework {framework} ⊆ {allowed_elements}")
        else:
            extra = framework_els - allowed_elements
            _log.info(f"Filtered out electrode {electrode.get('battery_id')}: framework {framework} contains unwanted elements {extra}")
    
    return filtered


def process_electrode_documents(results: List[Any], working_ion: str) -> List[Dict[str, Any]]:
    """
    Process electrode documents from Materials Project API into standardized format.
    
    Args:
        results: Raw electrode documents from MP API
        working_ion: Working ion symbol
        
    Returns:
        List of processed electrode data dictionaries
    """
    electrode_data = []
    
    for doc in results:
        try:
            # Handle both pydantic models and dicts
            if hasattr(doc, 'model_dump'):
                data = doc.model_dump()
            elif hasattr(doc, 'dict'):
                data = doc.dict()
            else:
                data = doc
            
            electrode_data.append({
                'battery_id': getattr(doc, 'battery_id', None) or data.get('battery_id'),
                'material_id': getattr(doc, 'id_discharge', None) or data.get('id_discharge'),
                'formula': getattr(doc, 'battery_formula', None) or data.get('battery_formula'),
                'formula_discharge': getattr(doc, 'formula_discharge', None) or data.get('formula_discharge'),
                'formula_charge': getattr(doc, 'formula_charge', None) or data.get('formula_charge'),
                'working_ion': working_ion,
                'average_voltage': getattr(doc, 'average_voltage', None) or data.get('average_voltage'),
                'max_voltage_step': getattr(doc, 'max_voltage_step', None) or data.get('max_voltage_step'),
                'capacity_grav': getattr(doc, 'capacity_grav', None) or data.get('capacity_grav'),
                'capacity_vol': getattr(doc, 'capacity_vol', None) or data.get('capacity_vol'),
                'energy_grav': getattr(doc, 'energy_grav', None) or data.get('energy_grav'),
                'energy_vol': getattr(doc, 'energy_vol', None) or data.get('energy_vol'),
                'fracA_charge': getattr(doc, 'fracA_charge', None) or data.get('fracA_charge'),
                'fracA_discharge': getattr(doc, 'fracA_discharge', None) or data.get('fracA_discharge'),
                'framework': getattr(doc, 'framework', None) or data.get('framework'),
            })
        except Exception as e:
            _log.warning(f"Error processing electrode doc: {e}")
            continue
    
    return electrode_data


def extract_voltage_profile(electrode: Any) -> Dict[str, Any]:
    """
    Extract voltage profile data from an electrode document.
    
    Args:
        electrode: Electrode document from MP API
        
    Returns:
        Dictionary with profile data
    """
    voltage_pairs = getattr(electrode, 'voltage_pairs', None)
    
    profile_data = {
        "formula": getattr(electrode, 'formula', None),
        "working_ion": getattr(electrode, 'working_ion', None),
        "average_voltage": getattr(electrode, 'average_voltage', None),
        "max_voltage": getattr(electrode, 'max_voltage', None),
        "min_voltage": getattr(electrode, 'min_voltage', None),
        "capacity_grav": getattr(electrode, 'capacity_grav', None),
        "voltage_profile": []
    }
    
    if voltage_pairs:
        for pair in voltage_pairs:
            if hasattr(pair, 'voltage'):
                profile_data["voltage_profile"].append({
                    "voltage": getattr(pair, 'voltage', None),
                    "capacity": getattr(pair, 'capacity_grav', None),
                    "x_charge": getattr(pair, 'x_charge', None),
                    "x_discharge": getattr(pair, 'x_discharge', None),
                })
    
    return profile_data

