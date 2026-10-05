"""
Tool Family Classification

Maps each @ai_function to one or more tool families based on their internal capabilities.

Families:
1. Thermodynamics (CALPHAD) - phase diagrams, equilibrium, phase fractions, Scheil
2. Materials Project/DFT/Surrogates - MP search, stability, E_hull, elastic/electronic, CHGNet
3. Electrochemistry - voltage profiles, insertion/conversion, capacity/energy
4. Magnetism/Defects/Alloys - Ms, Tc, defect sites, diffusion barriers, mechanical scores
5. Semiconductors - octahedral distortion, phase transitions, doping
6. Superconductors - cuprate c-axis stability, octahedral coordination
7. Solutes - lattice parameter engineering, Vegard's law, CALPHAD-validated solubility
8. Search - web/literature search
"""

# Family constants
CALPHAD = "Thermodynamics"
MP_DFT = "Materials Project/DFT"
ELECTROCHEM = "Electrochemistry"
MAG_DEFECT_ALLOY = "Magnetism/Defects/Alloys"
SEMICONDUCTORS = "Semiconductors"
SUPERCONDUCTORS = "Superconductors"
SOLUTES = "Solutes"
SEARCH = "Search"

# Mapping: function_name -> list of families
# If a function uses multiple backends internally, it gets multiple families
TOOL_FAMILIES = {
    # =========================================================================
    # CALPHAD Phase Diagrams (ai_functions_core.py)
    # =========================================================================
    "plot_binary_phase_diagram": [CALPHAD],
    "plot_composition_temperature": [CALPHAD],
    "analyze_last_generated_plot": [CALPHAD],
    
    # =========================================================================
    # CALPHAD Calculations (ai_functions_calculations.py)
    # =========================================================================
    "calculate_equilibrium_at_point": [CALPHAD],
    "compute_scheil_solidification": [CALPHAD],
    "compare_scheil_and_equilibrium": [CALPHAD],
    "find_invariant_reactions": [CALPHAD],
    "calculate_phase_fractions_vs_temperature": [CALPHAD],
    
    # =========================================================================
    # CALPHAD Verification (ai_functions_verification.py)
    # =========================================================================
    "verify_phase_formation_across_composition": [CALPHAD],
    "sweep_microstructure_claim_over_region": [CALPHAD],
    
    # =========================================================================
    # Materials Project / DFT (materials/ai_functions.py)
    # =========================================================================
    "mp_search_by_composition": [MP_DFT],
    "mp_get_by_id": [MP_DFT],
    "mp_get_by_characteristic": [MP_DFT],
    "get_elastic_modulus": [MP_DFT],
    "compare_material_properties": [MP_DFT],
    "analyze_doping_effect": [MP_DFT],
    
    # =========================================================================
    # Electrochemistry (electrochemistry/ai_functions.py)
    # =========================================================================
    "search_battery_electrodes": [ELECTROCHEM, MP_DFT],
    "calculate_voltage_from_formation_energy": [ELECTROCHEM, MP_DFT],
    "get_voltage_profile": [ELECTROCHEM, MP_DFT],
    "compare_electrode_materials": [ELECTROCHEM, MP_DFT],
    "check_composition_stability": [ELECTROCHEM, MP_DFT],
    "analyze_anode_viability": [ELECTROCHEM, MP_DFT],
    "analyze_lithiation_mechanism": [ELECTROCHEM, MP_DFT],
    "estimate_graphite_intercalation_barrier": [ELECTROCHEM, MP_DFT],  # CHGNet NEB
    
    # =========================================================================
    # Magnetism / Defects / Alloys (magnets/, alloys/)
    # =========================================================================
    # Magnets
    "assess_magnet_strength_with_doping": [MAG_DEFECT_ALLOY, MP_DFT],
    "compare_saturation_magnetization": [MAG_DEFECT_ALLOY, MP_DFT],
    "analyze_doping_effect_on_magnetization": [MAG_DEFECT_ALLOY, MP_DFT],
    
    # Alloys - surface diffusion
    "estimate_surface_diffusion_barrier": [MAG_DEFECT_ALLOY, MP_DFT],  # Uses CHGNet NEB
    
    # Alloys - mechanical/microstructure (uses CALPHAD + MP)
    "assess_phase_strength_and_stiffness_claims": [MAG_DEFECT_ALLOY, CALPHAD, MP_DFT],
    
    # =========================================================================
    # Semiconductors (semiconductors/ai_functions.py)
    # =========================================================================
    "analyze_octahedral_distortion_in_material": [SEMICONDUCTORS, MP_DFT],
    "analyze_structure_temperature_dependence": [SEMICONDUCTORS, MP_DFT],
    "analyze_doping_site_preference": [SEMICONDUCTORS, MP_DFT],
    "predict_site_preference": [SEMICONDUCTORS],
    "get_magnetic_properties": [SEMICONDUCTORS, MP_DFT],
    "compare_magnetic_materials": [SEMICONDUCTORS, MP_DFT],
    
    # =========================================================================
    # Superconductors (superconductors/ai_functions.py)
    # =========================================================================
    "analyze_cuprate_octahedral_stability": [SUPERCONDUCTORS, MP_DFT],
    
    # =========================================================================
    # Solutes (solutes/ai_functions.py)
    # Uses CALPHAD for solubility validation + Vegard's law for lattice effects
    # =========================================================================
    "analyze_solute_lattice_effect": [SOLUTES, CALPHAD],
    "compare_solute_lattice_effects": [SOLUTES, CALPHAD],
    "calculate_solute_lattice_effect": [SOLUTES],  # Direct calculation, no CALPHAD call
    "get_solute_reference_data": [SOLUTES],
    
    # =========================================================================
    # Search (search/ai_functions.py)
    # =========================================================================
    "search_web": [SEARCH],
}


def get_tool_families(function_name: str) -> list:
    """
    Get the list of families for a given tool function.
    
    Args:
        function_name: Name of the @ai_function (e.g., 'plot_binary_phase_diagram')
        
    Returns:
        List of family strings the tool belongs to.
        Returns ['unknown'] if function is not in the mapping.
    """
    return TOOL_FAMILIES.get(function_name, ["unknown"])


def get_primary_family(function_name: str) -> str:
    """
    Get the primary (first) family for a given tool function.
    
    Args:
        function_name: Name of the @ai_function
        
    Returns:
        Primary family string (first in the list).
    """
    families = get_tool_families(function_name)
    return families[0] if families else "unknown"


def tool_uses_family(function_name: str, family: str) -> bool:
    """
    Check if a tool uses a specific family.
    
    Args:
        function_name: Name of the @ai_function
        family: Family to check (e.g., 'CALPHAD', 'Materials Project/DFT')
        
    Returns:
        True if the tool uses this family.
    """
    return family in get_tool_families(function_name)


# Alias mapping from old handler names to new families
# Used for backwards compatibility and miss rate calculation
HANDLER_TO_FAMILIES = {
    # Thermodynamics / CALPHAD
    "calphad": [CALPHAD],
    "CALPHAD": [CALPHAD],
    "Thermodynamics": [CALPHAD],
    
    # Materials Project / DFT / Surrogates
    "materials": [MP_DFT],
    "Materials": [MP_DFT],
    "Materials Project": [MP_DFT],
    "Materials Project/DFT": [MP_DFT],
    "DFT": [MP_DFT],
    "CHGNet": [MP_DFT],
    
    # Electrochemistry
    "electrochemistry": [ELECTROCHEM],
    "Electrochemistry": [ELECTROCHEM],
    
    # Magnetism / Defects / Alloys
    "magnets": [MAG_DEFECT_ALLOY],
    "Magnets": [MAG_DEFECT_ALLOY],
    "alloys": [MAG_DEFECT_ALLOY],
    "Alloys": [MAG_DEFECT_ALLOY],
    "Magnetism/Defects/Alloys": [MAG_DEFECT_ALLOY],
    "defects": [MAG_DEFECT_ALLOY],
    
    # Semiconductors
    "semiconductors": [SEMICONDUCTORS],
    "Semiconductors": [SEMICONDUCTORS],
    
    # Superconductors
    "superconductors": [SUPERCONDUCTORS],
    "Superconductors": [SUPERCONDUCTORS],
    
    # Solutes
    "solutes": [SOLUTES],
    "Solutes": [SOLUTES],
    
    # Search
    "search": [SEARCH],
    "Search": [SEARCH],
    
    # Unknown fallback
    "unknown": ["unknown"],
}


def normalize_family(family_name: str) -> list:
    """
    Normalize a handler or family name to the canonical family list.
    
    Args:
        family_name: Handler name or family name (e.g., 'calphad', 'CALPHAD', 'Thermodynamics')
        
    Returns:
        List of canonical family names.
    """
    # Direct match
    if family_name in [CALPHAD, MP_DFT, ELECTROCHEM, MAG_DEFECT_ALLOY, 
                       SEMICONDUCTORS, SUPERCONDUCTORS, SOLUTES, SEARCH]:
        return [family_name]
    
    # Alias lookup
    return HANDLER_TO_FAMILIES.get(family_name, [family_name])

