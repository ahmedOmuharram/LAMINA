"""
Data Source Classification for H4 (Multi-source aggregation)

This module tracks the ACTUAL data sources used by each @ai_function:

PRIMARY SOURCES (computational):
- MP: Materials Project API for DFT data
- CALPHAD: pycalphad thermodynamic databases  
- CHGNet: ML surrogate for NEB barriers

SECONDARY SOURCES (fallbacks/supplements):
- LITERATURE: Published reference values (element moduli, diffusion barriers, etc.)
- HEURISTIC: Empirical models (Vegard's law, Orowan strengthening, scaling relations)
- CORRELATION: Empirical correlations (Tc vs structure, E_ads ~ E_coh)

For H4:
- MULTI-SOURCE: All sources enabled, including cross-source tools
- SINGLE-SOURCE: Only single primary source tools, NO fallbacks
"""

from typing import Set, Dict, Optional
from dataclasses import dataclass, field

# =============================================================================
# Source Constants
# =============================================================================

# Primary computational sources
MP = "MP"
CALPHAD = "CALPHAD"
CHGNET = "CHGNet"

# Secondary/fallback sources
LITERATURE = "LITERATURE"
HEURISTIC = "HEURISTIC"
CORRELATION = "CORRELATION"

PRIMARY_SOURCES = {MP, CALPHAD, CHGNET}
SECONDARY_SOURCES = {LITERATURE, HEURISTIC, CORRELATION}
ALL_SOURCES = PRIMARY_SOURCES | SECONDARY_SOURCES


@dataclass
class FunctionSources:
    """Describes all data sources used by a function."""
    primary: Set[str] = field(default_factory=set)     # Required computational sources
    secondary: Set[str] = field(default_factory=set)   # Fallback/supplementary sources
    description: str = ""
    
    @property
    def is_multi_primary(self) -> bool:
        return len(self.primary) > 1
    
    @property
    def uses_secondary(self) -> bool:
        return len(self.secondary) > 0


# =============================================================================
# Complete Function → Sources Mapping
# Based on actual code inspection of each @ai_function
# =============================================================================

FUNCTION_SOURCES: Dict[str, FunctionSources] = {
    # =========================================================================
    # CALPHAD Phase Diagrams (pure CALPHAD, no fallbacks)
    # =========================================================================
    "plot_binary_phase_diagram": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD phase diagram generation"
    ),
    "plot_composition_temperature": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD phase stability vs temperature"
    ),
    "analyze_last_generated_plot": FunctionSources(
        primary={CALPHAD},
        description="Analyzes cached CALPHAD plot"
    ),
    "calculate_equilibrium_at_point": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD equilibrium calculation"
    ),
    "compute_scheil_solidification": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD Scheil solidification"
    ),
    "compare_scheil_and_equilibrium": FunctionSources(
        primary={CALPHAD},
        description="Compares CALPHAD Scheil vs equilibrium"
    ),
    "find_invariant_reactions": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD invariant reaction finder"
    ),
    "calculate_phase_fractions_vs_temperature": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD phase fractions vs T"
    ),
    "analyze_phase_fraction_trend": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD phase fraction trend analysis"
    ),
    "verify_phase_formation_across_composition": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD phase formation verification"
    ),
    "sweep_microstructure_claim_over_region": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD composition space sweep"
    ),
    "fact_check_microstructure_claim": FunctionSources(
        primary={CALPHAD},
        description="CALPHAD microstructure fact checking"
    ),
    
    # =========================================================================
    # Materials Project (pure MP API)
    # =========================================================================
    "mp_search_by_composition": FunctionSources(
        primary={MP},
        description="MP database search"
    ),
    "mp_get_by_id": FunctionSources(
        primary={MP},
        description="MP get by ID"
    ),
    "mp_get_by_characteristic": FunctionSources(
        primary={MP},
        description="MP search by characteristic"
    ),
    "mp_get_material_details": FunctionSources(
        primary={MP},
        description="MP material details"
    ),
    "find_closest_alloy_compositions": FunctionSources(
        primary={MP},
        description="MP alloy composition finder"
    ),
    "compare_material_properties": FunctionSources(
        primary={MP},
        description="MP property comparison"
    ),
    "analyze_doping_effect": FunctionSources(
        primary={MP},
        description="MP doping effect analysis"
    ),
    
    # =========================================================================
    # Materials Project + Literature Fallbacks
    # =========================================================================
    "get_elastic_properties": FunctionSources(
        primary={MP},
        secondary={LITERATURE},
        description="MP elastic tensor + literature element moduli fallback"
    ),
    
    # =========================================================================
    # Electrochemistry - Various source combinations
    # =========================================================================
    "search_battery_electrodes": FunctionSources(
        primary={MP},
        description="MP electrode database search"
    ),
    "calculate_voltage_from_formation_energy": FunctionSources(
        primary={MP},
        description="MP convex hull voltage calculation"
    ),
    "get_voltage_profile": FunctionSources(
        primary={MP},
        description="MP voltage profile"
    ),
    "compare_electrode_materials": FunctionSources(
        primary={MP},
        description="MP electrode comparison"
    ),
    "check_composition_stability": FunctionSources(
        primary={MP},
        description="MP stability check (E_hull)"
    ),
    "analyze_anode_viability": FunctionSources(
        primary={MP},
        description="MP anode viability analysis"
    ),
    "analyze_lithiation_mechanism": FunctionSources(
        primary={MP},
        description="MP lithiation mechanism analysis"
    ),
    # These use CHGNet NEB but get structures from MP
    "estimate_graphite_intercalation_barrier": FunctionSources(
        primary={CHGNET},
        secondary={LITERATURE},
        description="CHGNet NEB for graphite, literature fallback barriers"
    ),
    "estimate_ion_hopping_barrier": FunctionSources(
        primary={CHGNET},
        secondary={LITERATURE},
        description="CHGNet NEB for ion hopping, structure-based literature defaults"
    ),
    
    # =========================================================================
    # Magnets - MP primary + heuristic models for derived properties
    # =========================================================================
    "assess_magnet_strength_with_doping": FunctionSources(
        primary={MP},
        secondary={HEURISTIC},
        description="MP magnetization + heuristic Hc/Br/pull-force models"
    ),
    "get_phase_and_magnetic_ordering": FunctionSources(
        primary={MP},
        description="MP magnetic ordering data"
    ),
    "estimate_permanent_magnet_properties": FunctionSources(
        primary={MP},
        secondary={HEURISTIC},
        description="MP Ms + heuristic Hc/Br/BHmax models"
    ),
    "calculate_magnet_pull_force": FunctionSources(
        primary=set(),  # Pure physics calculation
        description="Pure physics calculation (no data source)"
    ),
    "assess_doping_effect_on_saturation_magnetization": FunctionSources(
        primary={MP},
        secondary={HEURISTIC},
        description="MP Ms + heuristic dilution model"
    ),
    "compare_dopants_for_saturation_magnetization": FunctionSources(
        primary={MP},
        secondary={HEURISTIC},
        description="MP Ms + heuristic spin moment estimates"
    ),
    "get_saturation_magnetization_detailed": FunctionSources(
        primary={MP},
        description="MP magnetization data"
    ),
    "search_doped_magnetic_materials": FunctionSources(
        primary={MP},
        description="MP doped material search"
    ),
    
    # =========================================================================
    # Alloys - MULTI-SOURCE: Uses CALPHAD + MP + Literature + Heuristics
    # =========================================================================
    "estimate_surface_diffusion_barrier": FunctionSources(
        primary={MP, CHGNET},  # Uses MP for cohesive energies, CHGNet for NEB
        secondary={LITERATURE, CORRELATION, HEURISTIC},  # Literature E_coh, scaling relations, defect models
        description="CHGNet NEB + MP cohesive energies + surface science correlations + defect heuristics"
    ),
    "assess_phase_strength_and_stiffness_claims": FunctionSources(
        primary={CALPHAD, MP},  # CALPHAD for phases, MP for elastic data
        secondary={LITERATURE, HEURISTIC},  # Literature element moduli, Orowan model, Hall-Petch
        description="CALPHAD phases + MP elastic + literature moduli + strengthening models"
    ),
    
    # =========================================================================
    # Semiconductors - MP primary + some heuristics
    # =========================================================================
    "analyze_octahedral_distortion_in_material": FunctionSources(
        primary={MP},
        description="MP structure analysis"
    ),
    "analyze_structure_temperature_dependence": FunctionSources(
        primary={MP},
        secondary={HEURISTIC},
        description="MP structure + thermal expansion heuristics"
    ),
    "analyze_doping_site_preference": FunctionSources(
        primary={MP},
        description="MP doping site analysis"
    ),
    "predict_defect_site_preference": FunctionSources(
        primary={MP},
        description="MP defect site prediction"
    ),
    "get_magnetic_properties": FunctionSources(
        primary={MP},
        description="MP magnetic properties"
    ),
    "compare_magnetic_materials": FunctionSources(
        primary={MP},
        description="MP magnetic material comparison"
    ),
    "analyze_defect_stability": FunctionSources(
        primary={MP},
        description="MP defect stability"
    ),
    "search_same_phase_doped_variants": FunctionSources(
        primary={MP},
        description="MP doped variant search"
    ),
    "analyze_phase_transition_structures": FunctionSources(
        primary={MP},
        description="MP phase transition analysis"
    ),
    
    # =========================================================================
    # Superconductors - MP + Literature Tc correlations
    # =========================================================================
    "analyze_cuprate_octahedral_stability": FunctionSources(
        primary={MP},
        secondary={LITERATURE, CORRELATION},
        description="MP structure + literature cuprate Tc correlations"
    ),
    
    # =========================================================================
    # Solutes - CALPHAD for solubility + Literature Vegard coefficients
    # =========================================================================
    "analyze_solute_lattice_effect": FunctionSources(
        primary={CALPHAD},
        secondary={LITERATURE},
        description="CALPHAD solubility + literature Vegard's law"
    ),
    "compare_solute_lattice_effects": FunctionSources(
        primary={CALPHAD},
        secondary={LITERATURE},
        description="CALPHAD + literature lattice params"
    ),
    "calculate_solute_lattice_effect": FunctionSources(
        primary=set(),  # Pure calculation
        secondary={LITERATURE},
        description="Vegard's law with literature coefficients"
    ),
    "get_solute_reference_data": FunctionSources(
        primary=set(),
        secondary={LITERATURE},
        description="Literature reference data"
    ),
    
    # =========================================================================
    # Search - External (not a materials data source)
    # =========================================================================
    "search_web": FunctionSources(
        primary=set(),
        description="Web/literature search"
    ),
}


# =============================================================================
# Query Functions
# =============================================================================

def get_function_sources(function_name: str) -> FunctionSources:
    """Get sources for a function."""
    return FUNCTION_SOURCES.get(function_name, FunctionSources())


def is_multi_primary(function_name: str) -> bool:
    """Check if function requires multiple primary sources."""
    return get_function_sources(function_name).is_multi_primary


def uses_secondary_sources(function_name: str) -> bool:
    """Check if function uses any secondary/fallback sources."""
    return get_function_sources(function_name).uses_secondary


# =============================================================================
# H4 Configuration Functions
# =============================================================================

def get_h4_multi_source_functions() -> Set[str]:
    """
    H4 MULTI-SOURCE mode: All functions enabled.
    
    - All primary sources available
    - All secondary/fallback sources available
    - Multi-primary functions available
    """
    return set(FUNCTION_SOURCES.keys())


def get_h4_single_source_functions() -> Set[str]:
    """
    H4 SINGLE-SOURCE mode: Restricted function set.
    
    Excludes:
    1. Functions requiring multiple primary sources (e.g., CALPHAD + MP)
    2. Functions using secondary sources will be modified at RUNTIME
       to not use fallbacks (via _allow_fallbacks flag)
    """
    result = set()
    
    for func_name, sources in FUNCTION_SOURCES.items():
        # Include if uses at most ONE primary source
        if len(sources.primary) <= 1:
            result.add(func_name)
    
    return result


def get_multi_primary_functions() -> Set[str]:
    """Get functions requiring multiple primary sources."""
    return {
        func_name for func_name, sources in FUNCTION_SOURCES.items()
        if len(sources.primary) > 1
    }


def get_functions_with_secondary() -> Set[str]:
    """Get functions that use secondary/fallback sources."""
    return {
        func_name for func_name, sources in FUNCTION_SOURCES.items()
        if sources.secondary
    }


# =============================================================================
# Summary
# =============================================================================

def print_summary():
    """Print a summary of all function classifications."""
    print("Data Source Classification Summary")
    print("=" * 70)
    
    print(f"\nTotal functions: {len(FUNCTION_SOURCES)}")
    
    # Multi-primary
    multi = get_multi_primary_functions()
    print(f"\nMulti-primary source functions ({len(multi)}):")
    for f in sorted(multi):
        s = FUNCTION_SOURCES[f]
        print(f"  {f}")
        print(f"    Primary: {s.primary}")
        print(f"    Secondary: {s.secondary}")
        print(f"    Desc: {s.description}")
    
    # Secondary source users
    with_secondary = get_functions_with_secondary()
    print(f"\nFunctions with secondary sources ({len(with_secondary)}):")
    for f in sorted(with_secondary):
        s = FUNCTION_SOURCES[f]
        if f not in multi:  # Don't repeat multi-primary
            print(f"  {f}")
            print(f"    Primary: {s.primary}")
            print(f"    Secondary: {s.secondary}")
    
    print(f"\nH4 Configurations:")
    print(f"  Multi-source: {len(get_h4_multi_source_functions())} functions")
    print(f"  Single-source: {len(get_h4_single_source_functions())} functions")
    
    excluded = get_h4_multi_source_functions() - get_h4_single_source_functions()
    print(f"  Excluded in single-source: {sorted(excluded)}")


if __name__ == "__main__":
    print_summary()
