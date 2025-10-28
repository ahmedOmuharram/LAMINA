"""
AI Functions for Solute Lattice Effects Analysis

This module contains all AI-accessible functions for analyzing how substitutional
solute atoms change the lattice parameter of fcc matrices in the dilute limit.
It supports claims like:
- "Mg in Al causes the largest lattice expansion"
- "Cu in Al causes a moderate lattice contraction"

Uses Vegard's law and Hume-Rothery size mismatch principles with CALPHAD validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Annotated

from kani import ai_function, AIParam
from ..shared import success_result, error_result, ErrorType, Confidence
from .utils import (
    compute_substitutional_lattice_effect,
    rank_solutes_by_expansion,
    FCC_LATTICE_PARAMS_A,
    METALLIC_RADII_PM
)

_log = logging.getLogger(__name__)


def _get_calphad_fcc_composition(
    matrix_element: str,
    solute_element: str,
    solute_atpct: float,
    temperature_K: float
) -> Optional[Dict[str, float]]:
    """
    Internal helper to get FCC_A1 phase composition from CALPHAD equilibrium.
    
    Args:
        matrix_element: Matrix element (e.g., "Al")
        solute_element: Solute element (e.g., "Mg")
        solute_atpct: Solute concentration in at.%
        temperature_K: Temperature in K
        
    Returns:
        Dict of element mole fractions in FCC_A1 phase, or None if failed
    """
    try:
        from pycalphad import Database, equilibrium
        import pycalphad.variables as v
        import numpy as np
        
        mat = matrix_element.upper()
        sol = solute_element.upper()
        
        # Convert at.% to mole fraction
        x_solute = solute_atpct / 100.0
        x_matrix = 1.0 - x_solute
        
        # Find TDB file using shared utility
        from ..shared.calphad_utils import find_tdb_database
        
        db_path = find_tdb_database([mat, sol])
        if not db_path:
            _log.warning(f"No TDB database found for {mat}-{sol} system")
            return None
        
        db = Database(str(db_path))
        
        # Get all phases from database
        phases = list(db.phases.keys())
        
        # Setup elements and conditions
        elements = [mat, sol, 'VA']
        conditions = {
            v.T: temperature_K,
            v.P: 101325,
            v.N: 1.0,
            v.X(sol): x_solute
        }
        
        # Calculate equilibrium
        eq = equilibrium(db, elements, phases, conditions)
        
        # Extract FCC_A1 phase composition
        eq_squeezed = eq.squeeze()
        
        # Find FCC_A1 phase
        phase_array = eq_squeezed.Phase.values
        np_array = eq_squeezed.NP.values
        
        fcc_composition = {}
        
        for idx, phase in enumerate(phase_array):
            if phase and 'FCC_A1' in str(phase):
                frac = float(np_array[idx])
                if frac > 1e-4:  # Phase is present
                    # Extract composition of this phase
                    # Get X values for this phase
                    for elem in [mat, sol]:
                        try:
                            x_elem = float(eq_squeezed.X.sel(component=elem).values[idx])
                            if not np.isnan(x_elem):
                                fcc_composition[elem] = x_elem
                        except:
                            pass
                    
                    if fcc_composition:
                        return fcc_composition
        
        # If we didn't find FCC_A1, return None
        _log.warning(f"FCC_A1 phase not found in equilibrium for {mat}-{sol} at {temperature_K}K")
        return None
        
    except Exception as e:
        _log.error(f"Error getting CALPHAD composition: {e}", exc_info=True)
        return None


class SolutesAIFunctionsMixin:
    """Mixin class containing AI function methods for solute lattice effects analysis."""
    
    @ai_function(
        desc=(
            "PREFERRED for lattice parameter analysis. Analyze how a solute element affects the lattice "
            "parameter of an fcc matrix (e.g., 'Does Mg expand the Al lattice?'). Automatically runs CALPHAD "
            "to validate solubility, then applies Vegard's law and size misfit theory. Returns expansion/"
            "contraction prediction with Hume-Rothery analysis. Use for claims about lattice effects."
        ),
        auto_truncate=128000
    )
    async def analyze_solute_lattice_effect(
        self,
        matrix_element: Annotated[str, AIParam(desc="Matrix element symbol (e.g., 'Al').")],
        solute_element: Annotated[str, AIParam(desc="Solute element symbol (e.g., 'Mg').")],
        solute_atpct: Annotated[float, AIParam(desc="Nominal solute concentration in atomic percent (e.g., 1.0 for 1 at.%).")] = 1.0,
        temperature_K: Annotated[float, AIParam(desc="Temperature in Kelvin (default: 300K).")] = 300.0
    ) -> Dict[str, Any]:
        """
        Analyze lattice parameter effect with automatic CALPHAD validation.
        
        This function:
        1. Runs CALPHAD equilibrium to get FCC_A1 phase composition
        2. Validates that solute is actually dissolved substitutionally
        3. Computes size misfit and Vegard's law prediction
        4. Returns classification and Hume-Rothery analysis
        
        Returns comprehensive analysis that can support or refute claims about
        lattice expansion/contraction effects.
        """
        start_time = time.time()
        
        try:
            import numpy as np
            
            # Get CALPHAD FCC composition
            fcc_comp = _get_calphad_fcc_composition(
                matrix_element, solute_element, solute_atpct, temperature_K
            )
            
            if fcc_comp is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="solutes",
                    function="analyze_solute_lattice_effect",
                    error=f"Could not compute CALPHAD equilibrium for {matrix_element}-{solute_element} system",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["pycalphad", "CALPHAD equilibrium validation"],
                    notes=["This may mean no thermodynamic database is available or FCC phase doesn't form"],
                    duration_ms=duration_ms
                )
            
            # Now call the core physics function
            util_result = compute_substitutional_lattice_effect(
                matrix_element=matrix_element,
                solute_element=solute_element,
                solute_atpct=float(solute_atpct),
                temperature_K=float(temperature_K),
                matrix_phase_name="FCC_A1",
                matrix_phase_composition=fcc_comp,
                min_required_solute_in_matrix_atpct=0.1
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not util_result.get("success"):
                result = error_result(
                    handler="solutes",
                    function="analyze_solute_lattice_effect",
                    error=util_result.get("error", "Computation failed"),
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions",
                        "CALPHAD equilibrium validation"
                    ],
                    duration_ms=duration_ms
                )
            else:
                data = {k: v for k, v in util_result.items() if k != "success"}
                result = success_result(
                    handler="solutes",
                    function="analyze_solute_lattice_effect",
                    data=data,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions",
                        "CALPHAD equilibrium validation"
                    ],
                    confidence=Confidence.MEDIUM,
                    notes=["CALPHAD-validated solubility", "Vegard's law applied in dilute limit"],
                    duration_ms=duration_ms
                )
            
            self._track_tool_output("analyze_solute_lattice_effect", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in analyze_solute_lattice_effect: {e}", exc_info=True)
            return error_result(
                handler="solutes",
                function="analyze_solute_lattice_effect",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=[
                    "Vegard's law for dilute substitutional alloys",
                    "Hume-Rothery rules for solid solutions"
                ],
                duration_ms=duration_ms
            )
    
    @ai_function(
        desc=(
            "PREFERRED for comparing multiple solutes. Rank multiple solute elements by their lattice "
            "expansion effect on an fcc matrix (e.g., 'Which causes largest expansion: Mg, Cu, or Zn in Al?'). "
            "Automatically runs CALPHAD for each solute, validates solubility, and ranks by Δa/a. "
            "Returns sorted ranking with largest expander identified."
        ),
        auto_truncate=128000
    )
    async def compare_solute_lattice_effects(
        self,
        matrix_element: Annotated[str, AIParam(desc="Matrix element symbol (e.g., 'Al').")],
        solute_elements: Annotated[List[str], AIParam(desc="List of solute elements to compare (e.g., ['Mg', 'Cu', 'Zn']).")],
        solute_atpct: Annotated[float, AIParam(desc="Nominal solute concentration for all solutes in at.% (default: 1.0).")] = 1.0,
        temperature_K: Annotated[float, AIParam(desc="Temperature in Kelvin (default: 300K).")] = 300.0
    ) -> Dict[str, Any]:
        """
        Compare and rank multiple solutes by their lattice expansion effects.
        
        This function:
        1. Runs CALPHAD equilibrium for each solute independently
        2. Validates solubility for each
        3. Computes lattice effects for all valid solutes
        4. Ranks by predicted Δa/a (largest expansion first)
        5. Identifies the largest expander
        
        Perfect for answering comparative claims like:
        - "Does Mg cause larger expansion than Cu in Al?"
        - "Which solute expands the Al lattice most?"
        
        Returns ranking with detailed analysis for each solute.
        """
        start_time = time.time()
        
        try:
            import numpy as np
            
            # Get CALPHAD compositions for each solute
            calphad_compositions = {}
            for solute in solute_elements:
                fcc_comp = _get_calphad_fcc_composition(
                    matrix_element, solute, solute_atpct, temperature_K
                )
                if fcc_comp is not None:
                    calphad_compositions[solute.upper()] = fcc_comp
                else:
                    # Store empty dict to indicate failure
                    calphad_compositions[solute.upper()] = {}
            
            # Now call the ranking function
            util_result = rank_solutes_by_expansion(
                matrix_element=matrix_element,
                solute_elements=solute_elements,
                solute_atpct=float(solute_atpct),
                temperature_K=float(temperature_K),
                calphad_matrix_phase_name="FCC_A1",
                calphad_matrix_phase_compositions=calphad_compositions
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not util_result.get("success"):
                result = error_result(
                    handler="solutes",
                    function="compare_solute_lattice_effects",
                    error=util_result.get("error", "Comparison failed"),
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions",
                        "CALPHAD equilibrium validation"
                    ],
                    duration_ms=duration_ms
                )
            else:
                data = {k: v for k, v in util_result.items() if k != "success"}
                result = success_result(
                    handler="solutes",
                    function="compare_solute_lattice_effects",
                    data=data,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions",
                        "Comparative size misfit analysis",
                        "CALPHAD equilibrium validation"
                    ],
                    confidence=Confidence.HIGH,
                    notes=["Solutes ranked by lattice expansion effect", "CALPHAD-validated for each solute"],
                    duration_ms=duration_ms
                )
            
            self._track_tool_output("compare_solute_lattice_effects", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in compare_solute_lattice_effects: {e}", exc_info=True)
            return error_result(
                handler="solutes",
                function="compare_solute_lattice_effects",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=[
                    "Vegard's law for dilute substitutional alloys",
                    "Hume-Rothery rules for solid solutions"
                ],
                duration_ms=duration_ms
            )
    
    @ai_function(
        desc=(
            "Advanced: Calculate lattice effect when you already have CALPHAD equilibrium data. "
            "Most users should use analyze_solute_lattice_effect instead, which fetches CALPHAD data automatically."
        ),
        auto_truncate=128000
    )
    async def calculate_solute_lattice_effect(
        self,
        matrix_element: Annotated[str, AIParam(desc="Matrix element symbol (e.g., 'Al' for aluminum).")],
        solute_element: Annotated[str, AIParam(desc="Solute element symbol (e.g., 'Mg' for magnesium).")],
        solute_atpct: Annotated[float, AIParam(desc="Nominal solute concentration in atomic percent (e.g., 1.0 for 1 at.%).")],
        temperature_K: Annotated[float, AIParam(desc="Temperature in Kelvin (e.g., 300 for room temperature).")],
        matrix_phase_name: Annotated[str, AIParam(desc="Name of the matrix phase from CALPHAD (must be 'FCC_A1').")],
        matrix_phase_composition: Annotated[
            Dict[str, float], 
            AIParam(desc="Element mole fractions in the matrix phase from CALPHAD equilibrium. E.g., {'AL': 0.995, 'MG': 0.005}.")
        ],
        min_required_solute_atpct: Annotated[
            float, 
            AIParam(desc="Minimum solute concentration in matrix phase (at.%) to consider valid. Default: 0.1")
        ] = 0.1
    ) -> Dict[str, Any]:
        """
        Calculate lattice parameter change due to substitutional solute addition.
        
        This function:
        1. Validates that the solute is dissolved in the fcc matrix phase (from CALPHAD)
        2. Computes size misfit δ_size = (r_solute - r_matrix) / r_matrix
        3. Applies Vegard's law: Δa/a0 ≈ δ_size * x (dilute limit)
        4. Classifies the effect (expands/contracts, large/moderate/negligible)
        5. Reports Hume-Rothery size mismatch criterion (~15% threshold)
        
        Physics basis:
        - fcc geometry: a = 2√2 * r (atomic radius)
        - Vegard's law for dilute substitutional alloys
        - Standard metallic radii for close-packed coordination
        
        Returns comprehensive analysis with lattice parameter predictions,
        size misfit, classification, and applicability assessment.
        """
        start_time = time.time()
        
        try:
            util_result = compute_substitutional_lattice_effect(
                matrix_element=matrix_element,
                solute_element=solute_element,
                solute_atpct=float(solute_atpct),
                temperature_K=float(temperature_K),
                matrix_phase_name=matrix_phase_name,
                matrix_phase_composition=matrix_phase_composition,
                min_required_solute_in_matrix_atpct=float(min_required_solute_atpct)
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not util_result.get("success"):
                result = error_result(
                    handler="solutes",
                    function="calculate_solute_lattice_effect",
                    error=util_result.get("error", "Calculation failed"),
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions"
                    ],
                    duration_ms=duration_ms
                )
            else:
                data = {k: v for k, v in util_result.items() if k != "success"}
                result = success_result(
                    handler="solutes",
                    function="calculate_solute_lattice_effect",
                    data=data,
                    citations=[
                        "Vegard's law for dilute substitutional alloys",
                        "Hume-Rothery rules for solid solutions",
                        "Standard crystallographic data for fcc lattice parameters",
                        "Metallic radii for 12-fold coordination"
                    ],
                    confidence=Confidence.HIGH,
                    notes=["Uses user-provided CALPHAD composition", "Vegard's law in dilute limit"],
                    duration_ms=duration_ms
                )
            
            self._track_tool_output("calculate_solute_lattice_effect", result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in calculate_solute_lattice_effect: {e}", exc_info=True)
            return error_result(
                handler="solutes",
                function="calculate_solute_lattice_effect",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=[
                    "Vegard's law for dilute substitutional alloys",
                    "Hume-Rothery rules for solid solutions"
                ],
                duration_ms=duration_ms
            )
    
    @ai_function(
        desc=(
            "Get reference data for fcc lattice parameters and metallic radii. "
            "Shows which matrix elements and solutes are supported by the analysis tools. "
            "Useful for checking if your material combination is in the reference database."
        ),
        auto_truncate=128000
    )
    async def get_solute_reference_data(
        self
    ) -> Dict[str, Any]:
        """
        Retrieve reference data for lattice parameters and metallic radii.
        
        Returns:
        - Supported fcc matrix elements with lattice parameters
        - Available metallic radii for matrix and solute elements
        - Temperature reference (~300 K for most data)
        """
        start_time = time.time()
        
        try:
            data = {
                "fcc_lattice_parameters_angstrom": FCC_LATTICE_PARAMS_A,
                "metallic_radii_pm": METALLIC_RADII_PM,
                "reference_temperature_K": 300,
                "supported_matrices": list(FCC_LATTICE_PARAMS_A.keys()),
                "supported_elements": list(METALLIC_RADII_PM.keys())
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = success_result(
                handler="solutes",
                function="get_solute_reference_data",
                data=data,
                citations=["Standard crystallographic tables", "Metallic radii compilations"],
                confidence=Confidence.HIGH,
                duration_ms=duration_ms,
                notes=[
                    "Lattice parameters are at ~300 K",
                    "Metallic radii are for 12-fold coordination (close-packed metallic)",
                    "Data sources: standard crystallographic tables and metallic radii compilations"
                ]
            )
            
            self._track_tool_output("get_solute_reference_data", result)
            
            return result
            
        except Exception as e:
            _log.error(f"Error in get_solute_reference_data: {e}", exc_info=True)
            return error_result(
                handler="solutes",
                function="get_solute_reference_data",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR
            )

