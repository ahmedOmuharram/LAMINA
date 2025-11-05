"""
Advanced verification AI function methods for CALPHAD phase diagrams.

Contains sophisticated verification and fact-checking functions:
- verify_phase_formation_across_composition: Verify phase formation across composition ranges
- sweep_microstructure_claim_over_region: Sweep composition space to evaluate claims
- fact_check_microstructure_claim: Fact-check microstructure claims
"""
from typing import Dict, Any, Optional, List
import logging
import time
import asyncio

from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .verification_utils import (
    parse_composition_string,
    map_phase_to_category,
    get_phases_for_elements
)
from ...shared.calphad_utils import load_tdb_database, compute_equilibrium
from ...shared.result_wrappers import success_result, error_result, Confidence, ErrorType

_log = logging.getLogger(__name__)


class VerificationMixin:
    """Mixin class containing advanced verification AI functions for CalPhadHandler."""
    
    @ai_function(desc="Verify phase formation statements across a composition range. Use to check claims like 'beyond X% of element B, phase Y forms' or 'at compositions greater than X%, phase Z appears'. Analyzes which phases form at different compositions.")
    async def verify_phase_formation_across_composition(
        self,
        system: Annotated[str, AIParam(desc="System (e.g., 'Fe-Al', 'Al-Zn' for binary, 'Al-Mg-Zn' for ternary)")],
        phase_name: Annotated[str, AIParam(desc="Phase name: exact database name (e.g., 'MGZN2', 'TAU', 'FCC_A1') or category (e.g., 'Laves', 'tau', 'fcc', 'gamma')")],
        composition_threshold: Annotated[float, AIParam(desc="Composition threshold to verify (e.g., 50.0 for '50 at.%')")],
        threshold_element: Annotated[str, AIParam(desc="Element being thresholded (e.g., 'Al' in 'beyond 50% Al')")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin to check phase formation. Default: 300K")] = 300.0,
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic",
        fixed_element: Annotated[Optional[str], AIParam(desc="For ternary: element to keep fixed (e.g., 'Zn')")] = None,
        fixed_composition: Annotated[Optional[float], AIParam(desc="For ternary: fixed element composition in at.% (e.g., 4.0 for 4%)")] = None
    ) -> str:
        """
        Verify whether a specific phase forms above/below a composition threshold.
        
        This tool checks statements like:
        - Binary: "Beyond 50 at.% Al in Fe-Al, the Al2Fe phase forms"
        - Ternary: "In Al-Mg-Zn with 4% Zn, tau phase forms above 8% Mg"
        
        For ternary systems, specify fixed_element and fixed_composition to hold one element constant
        while varying the threshold_element.
        
        Returns detailed analysis showing:
        - Which phases are present below/above the threshold
        - Whether the specified phase forms above/below threshold
        - Phase fractions across the composition range
        """
        start_time = time.time()
        
        try:
            import numpy as np
            
            # Parse elements from system
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [e.strip().upper() for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]
            
            if len(elements) < 2:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="verify_phase_formation_across_composition",
                    error=f"System must have at least 2 elements. Got: {system}",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            is_ternary = len(elements) >= 3
            system_str = "-".join(elements)
            _log.debug(f"System: {system_str}")
            _log.debug(f"Elements: {elements}")
            _log.debug(f"Is ternary: {is_ternary}")

            # Validate threshold element
            threshold_elem = threshold_element.strip().upper()
            if threshold_elem not in elements:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="verify_phase_formation_across_composition",
                    error=f"Element '{threshold_element}' not found in system {system_str}. Must be one of: {', '.join(elements)}",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # For ternary systems, validate fixed element
            if is_ternary:
                if not fixed_element or fixed_composition is None:
                    duration_ms = (time.time() - start_time) * 1000
                    return error_result(
                        handler="calphad",
                        function="verify_phase_formation_across_composition",
                        error=f"For ternary system {system_str}, must specify fixed_element and fixed_composition",
                        error_type=ErrorType.INVALID_INPUT,
                        citations=["pycalphad"],
                        duration_ms=duration_ms
                    )
                
                fixed_elem = fixed_element.strip().upper()
                if fixed_elem not in elements:
                    duration_ms = (time.time() - start_time) * 1000
                    return error_result(
                        handler="calphad",
                        function="verify_phase_formation_across_composition",
                        error=f"Fixed element '{fixed_element}' not found in system {system_str}",
                        error_type=ErrorType.INVALID_INPUT,
                        citations=["pycalphad"],
                        duration_ms=duration_ms
                    )
                if fixed_elem == threshold_elem:
                    duration_ms = (time.time() - start_time) * 1000
                    return error_result(
                        handler="calphad",
                        function="verify_phase_formation_across_composition",
                        error=f"Fixed element and threshold element cannot be the same",
                        error_type=ErrorType.INVALID_INPUT,
                        citations=["pycalphad"],
                        duration_ms=duration_ms
                    )
                
                # The third element is the balance
                balance_elem = [e for e in elements if e not in [threshold_elem, fixed_elem]][0]
            else:
                # Binary system: one element varies, the other is balance
                balance_elem = [e for e in elements if e != threshold_elem][0]
                fixed_elem = None
            _log.debug(f"Balance element: {balance_elem}")
            _log.debug(f"Fixed element: {fixed_elem}")
            # Load database
            db = load_tdb_database(elements)
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="verify_phase_formation_across_composition",
                    error=f"No thermodynamic database found for {system_str} system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Get phases
            if len(elements) == 2:
                phases = self._filter_phases_for_system(db, tuple(elements[:2]))
            else:
                phases = get_phases_for_elements(db, elements, self._phase_elements)
            _log.debug(f"Phases: {phases}")

            # Normalize phase name and handle category names
            from .fact_checker import PHASE_CLASSIFICATION
            _log.debug(f"Phase classification: {PHASE_CLASSIFICATION}")
            phase_name_upper = phase_name.upper()
            available_phases = sorted(phases)
            
            # Debug: List all available phases
            _log.debug(f"Available phases in {system_str} database: {available_phases}")
            
            # Check if input is a category name (e.g., "Laves", "tau", "gamma")
            # Map it to actual database phase names
            phase_to_check = None
            candidate_phases = []
            
            # First, try direct match with database phase names
            if phase_name_upper in phases or phase_name in phases:
                phase_to_check = phase_name_upper if phase_name_upper in phases else phase_name
                candidate_phases = [phase_to_check]
            else:
                # Try category-based matching using PHASE_CLASSIFICATION
                phase_name_lower = phase_name.lower()
                
                for db_phase_name, (readable_name, category, structure) in PHASE_CLASSIFICATION.items():
                    # Check if input matches the category or readable name
                    category_value = category.value if hasattr(category, 'value') else str(category)
                    if (phase_name_lower in category_value.lower() or 
                        phase_name_lower in readable_name.lower() or
                        category_value.lower() in phase_name_lower):
                        # Check if this database phase is available in current system
                        if db_phase_name in phases:
                            candidate_phases.append(db_phase_name)
                
                if candidate_phases:
                    phase_to_check = candidate_phases[0]  # Use first as representative
                    _log.debug(f"Mapped category '{phase_name}' to database phases: {candidate_phases}")
            
            # Final check: did we find any matching phase?
            if not phase_to_check:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="verify_phase_formation_across_composition",
                    error=f"Phase '{phase_name}' not found in database. Available phases: {', '.join(available_phases)}. Try using exact database names or categories like: fcc, bcc, hcp, tau, laves, gamma",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Determine the set of database phase names to aggregate
            target_names = set(candidate_phases) if candidate_phases else {phase_to_check}
            display_name = phase_name if not candidate_phases or len(candidate_phases) == 1 else f"{phase_name} (any)"
            
            # Sample compositions for the threshold element
            threshold_fraction = composition_threshold / 100.0
            n_points = 21
            threshold_compositions = np.linspace(0.0, 1.0, n_points)
            
            # Add specific points around threshold for precision
            threshold_nearby = [
                max(0.0, threshold_fraction - 0.05),
                max(0.0, threshold_fraction - 0.02),
                threshold_fraction,
                min(1.0, threshold_fraction + 0.02),
                min(1.0, threshold_fraction + 0.05)
            ]
            threshold_compositions = np.unique(np.sort(np.concatenate([threshold_compositions, threshold_nearby])))
            
            # Build elements list for pycalphad (include VA)
            pycalphad_elements = elements + ['VA']
            results = []
            
            _log.debug(f"Checking phase '{phase_to_check}' formation across {len(threshold_compositions)} compositions at T={temperature}K")
            if is_ternary:
                _log.debug(f"  Ternary: varying {threshold_elem}, fixing {fixed_elem}={fixed_composition}%, balance={balance_elem}")
            
            for x_threshold in threshold_compositions:
                try:
                    # Build composition dict
                    comp_dict = {}
                    comp_dict[threshold_elem] = x_threshold
                    
                    if is_ternary:
                        # Fixed element
                        comp_dict[fixed_elem] = fixed_composition / 100.0
                        # Balance element (automatically calculated by pycalphad as 1 - others)
                        comp_dict[balance_elem] = 1.0 - x_threshold - (fixed_composition / 100.0)
                        
                        # Skip if balance is negative or zero
                        if comp_dict[balance_elem] <= 0:
                            continue
                    else:
                        # Binary: balance is just 1 - threshold
                        comp_dict[balance_elem] = 1.0 - x_threshold
                    
                    # For pycalphad equilibrium, only specify N-1 composition constraints
                    eq = compute_equilibrium(db, pycalphad_elements, phases, comp_dict, temperature)
                    
                    if eq is None:
                        _log.warning(f"Equilibrium calculation failed at {threshold_elem}={x_threshold:.3f}")
                        continue
                    
                    # Extract phase fractions
                    from ...shared.calphad_utils import extract_phase_fractions_from_equilibrium
                    phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    # Helper to strip instance suffixes (e.g., "FCC_A1#2" -> "FCC_A1")
                    def base_name(n):
                        return str(n).split('#')[0].upper()
                    
                    # Aggregate fraction across all target phase names
                    target_names_upper = {t.upper() for t in target_names}
                    phase_fraction = sum(
                        frac for name, frac in phase_fractions.items()
                        if base_name(name) in target_names_upper
                    )
                    phase_present = phase_fraction > 0.01
                    
                    # Store results with at.% for all elements
                    result = {
                        'phase_present': phase_present,
                        'phase_fraction': float(phase_fraction),
                        'all_phases': {k: float(v) for k, v in phase_fractions.items()}
                    }
                    
                    # Add at.% for each element
                    for el in elements:
                        result[f'at_pct_{el}'] = float(comp_dict[el] * 100)
                    
                    results.append(result)
                    
                except Exception as e:
                    _log.warning(f"Equilibrium calculation failed at {threshold_elem}={x_threshold:.3f}: {e}")
                    continue
            
            if not results:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="verify_phase_formation_across_composition",
                    error=f"Failed to calculate equilibrium for any compositions in {system_str} at {temperature}K",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Analyze results around threshold
            threshold_pct = composition_threshold
            threshold_key = f'at_pct_{threshold_elem}'
            
            # Find compositions below and above threshold
            below_threshold = [r for r in results if r[threshold_key] < threshold_pct]
            above_threshold = [r for r in results if r[threshold_key] >= threshold_pct]
            
            # Count how many times the phase appears in each region
            phase_count_below = sum(1 for r in below_threshold if r['phase_present'])
            phase_count_above = sum(1 for r in above_threshold if r['phase_present'])
            
            # Build response - structured JSON
            total_below = len(below_threshold)
            total_above = len(above_threshold)
            
            # Calculate fractions
            fraction_below = (phase_count_below / total_below) if total_below > 0 else 0.0
            fraction_above = (phase_count_above / total_above) if total_above > 0 else 0.0
            
            # Build example compositions
            below_threshold_example = None
            above_threshold_example = None
            
            if total_below > 0 and below_threshold:
                present_examples = [r for r in below_threshold if r['phase_present']]
                example = present_examples[len(present_examples)//2] if present_examples else below_threshold[len(below_threshold)//2]
                
                # Format phases
                example_phases = []
                for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                    if f > 0.01:
                        phase_entry = {"name": p, "fraction": float(f)}
                        if p == 'FCC_A1':
                            host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                            phase_entry["host_element"] = host_el
                        example_phases.append(phase_entry)
                
                below_threshold_example = {
                    "composition": {el: float(example[f'at_pct_{el}']) for el in elements},
                    "phases": example_phases
                }
            
            if total_above > 0 and above_threshold:
                present_examples = [r for r in above_threshold if r['phase_present']]
                example = present_examples[len(present_examples)//2] if present_examples else above_threshold[len(above_threshold)//2]
                
                # Format phases
                example_phases = []
                for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                    if f > 0.01:
                        phase_entry = {"name": p, "fraction": float(f)}
                        if p == 'FCC_A1':
                            host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                            phase_entry["host_element"] = host_el
                        example_phases.append(phase_entry)
                
                above_threshold_example = {
                    "composition": {el: float(example[f'at_pct_{el}']) for el in elements},
                    "phases": example_phases
                }
            
            # Build detailed phase presence table
            detailed_compositions = []
            for r in results[::max(1, len(results)//10)]:  # ~10 representative points
                other_phases_list = []
                for p, f in sorted(r['all_phases'].items(), key=lambda x: -x[1]):
                    if p != phase_to_check and f > 0.05:
                        phase_entry = {"name": p, "fraction": float(f)}
                        if p == 'FCC_A1':
                            host_el = max(elements, key=lambda el: r[f'at_pct_{el}'])
                            phase_entry["host_element"] = host_el
                        other_phases_list.append(phase_entry)
                        if len(other_phases_list) >= 3:
                            break
                
                detailed_compositions.append({
                    "composition": {el: float(r[f'at_pct_{el}']) for el in elements},
                    "phase_present": r['phase_present'],
                    "phase_fraction": float(r['phase_fraction']) if r['phase_fraction'] > 0 else None,
                    "other_major_phases": other_phases_list
                })
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Calculate confidence based on sample size and detection rate
            sample_confidence = min(1.0, len(results) / 20.0)  # More samples = more confidence
            detection_confidence = max(fraction_below, fraction_above)  # Higher detection rate = more confidence
            overall_confidence = (sample_confidence + detection_confidence) / 2.0
            
            if overall_confidence > 0.8:
                conf_level = Confidence.HIGH
            elif overall_confidence > 0.5:
                conf_level = Confidence.MEDIUM
            else:
                conf_level = Confidence.LOW
            
            # Build structured JSON response
            response_data = {
                "system": system_str,
                "phase": display_name,
                "temperature_K": temperature,
                "threshold": {
                    "element": threshold_elem,
                    "value_at_pct": threshold_pct
                },
                "results": {
                    "below_threshold": {
                        "count": phase_count_below,
                        "total_compositions": total_below,
                        "fraction": fraction_below,
                        "example": below_threshold_example
                    },
                    "above_threshold": {
                        "count": phase_count_above,
                        "total_compositions": total_above,
                        "fraction": fraction_above,
                        "example": above_threshold_example
                    },
                    "frequency_difference": fraction_above - fraction_below
                },
                "detailed_compositions": detailed_compositions,
                "total_sampled_points": len(results)
            }
            
            if is_ternary:
                response_data["fixed_element"] = {
                    "element": fixed_elem,
                    "value_at_pct": fixed_composition
                }
            
            return success_result(
                handler="calphad",
                function="verify_phase_formation_across_composition",
                data=response_data,
                citations=["pycalphad"],
                confidence=conf_level,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error verifying phase formation across composition")
            return error_result(
                handler="calphad",
                function="verify_phase_formation_across_composition",
                error=f"Failed to verify phase formation: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
    
    @ai_function(desc="Sweep composition space and evaluate microstructure properties across an entire region. Use this to test claims like 'all Al-Mg-Zn alloys with Mg<8 at.% and Zn<4 at.% form fcc+tau microstructures'. Returns grid of results with phase data.")
    async def sweep_microstructure_claim_over_region(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Mg-Zn')")],
        element_ranges: Annotated[str, AIParam(desc="JSON dict of element ranges, e.g. '{\"MG\": [0, 8], \"ZN\": [0, 4]}'. Units determined by composition_type parameter (default: atomic percent). Remaining composition is balance element (usually first in system).")],
        claim_type: Annotated[str, AIParam(desc="Type of claim: 'two_phase', 'three_phase', 'phase_fraction'")],
        expected_phases: Annotated[Optional[str], AIParam(desc="Expected phases (e.g., 'fcc+tau'). Required for two_phase and three_phase claims.")] = None,
        phase_to_check: Annotated[Optional[str], AIParam(desc="Specific phase to check. Required for phase_fraction claims.")] = None,
        min_fraction: Annotated[Optional[float], AIParam(desc="Minimum phase fraction (0-1)")] = None,
        max_fraction: Annotated[Optional[float], AIParam(desc="Maximum phase fraction (0-1)")] = None,
        grid_points: Annotated[int, AIParam(desc="Number of grid points per element (default: 4 = 16 total points for binary variation)")] = 4,
        composition_type: Annotated[str, AIParam(desc="Composition units: 'atomic' (at.%, DEFAULT) or 'weight' (wt.%)")] = "atomic",
        process_type: Annotated[str, AIParam(desc="Process model: 'as_cast' or 'equilibrium_300K'")] = "as_cast",
        require_mechanical_desirability: Annotated[bool, AIParam(desc="If true, also require positive mechanical desirability score")] = False,
        stop_on_first_violation: Annotated[bool, AIParam(desc="If true, stop as soon as a point contradicts the claim (faster for universal claims)")] = False
    ) -> Dict[str, Any]:
        """
        Sweep over a composition region and test whether a claim holds universally.
        
        This answers: "Does the claim hold for ALL compositions in the stated range?"
        not just "Does it hold for this one specific composition?"
        
        **Composition Units**: By default uses atomic percent (at.%). 
        For example, element_ranges='{"MG": [0, 8], "ZN": [0, 4]}' means:
        - Mg varies from 0 to 8 at.%
        - Zn varies from 0 to 4 at.%
        - Al (balance) = 100 - Mg - Zn at.%
        
        Example: "All lightweight Al-Mg-Zn with Mg<8 at.% and Zn<4 at.% form desirable fcc+tau"
        → Sweep Mg=[0,8), Zn=[0,4) in atomic percent and check if fcc+tau with tau≤20% everywhere.
        
        Note: Weight percent (wt.%) is not yet fully implemented.
        """
        start_time = time.time()
        
        try:
            import json
            import numpy as np
            
            # Validate claim type and required parameters
            claim_type_lower = claim_type.lower()
            
            # --- Validate required args by claim type ---
            if claim_type_lower == "phase_fraction":
                if not phase_to_check:
                    return {
                        "success": False,
                        "error": "phase_to_check is required for phase_fraction claims "
                                 "(e.g. phase_to_check='tau', min_fraction=0.05, max_fraction=0.20)",
                        "citations": ["pycalphad"]
                    }
            
            elif claim_type_lower == "two_phase":
                if not expected_phases:
                    return {
                        "success": False,
                        "error": "expected_phases is required for two_phase claims "
                                 "(e.g. expected_phases='fcc+theta', max_fraction=0.20)",
                        "citations": ["pycalphad"]
                    }
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                
                # Auto-correct claim_type based on actual number of phases
                if len(phase_list) == 3:
                    claim_type_lower = "three_phase"
                elif len(phase_list) != 2:
                    return {
                        "success": False,
                        "error": (
                            "two_phase claim requires exactly 2 phases (matrix+secondary).\n"
                            f"You gave expected_phases={expected_phases!r} which parsed to {len(phase_list)} phase(s).\n"
                            "Examples: expected_phases='fcc+tau' or 'fcc,theta'\n"
                            "If you're trying to say 'phase X exists between 5%-50%', "
                            "use claim_type='phase_fraction', "
                            "phase_to_check='X', min_fraction=0.05, max_fraction=0.5 instead."
                        ),
                        "citations": ["pycalphad"]
                    }
            
            elif claim_type_lower == "three_phase":
                if not expected_phases:
                    return {
                        "success": False,
                        "error": "expected_phases is required for three_phase claims "
                                 "(e.g. expected_phases='fcc+tau+lambda')",
                        "citations": ["pycalphad"]
                    }
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                
                # Auto-correct claim_type based on actual number of phases
                if len(phase_list) == 2:
                    claim_type_lower = "two_phase"
                elif len(phase_list) != 3:
                    return {
                        "success": False,
                        "error": (
                            "three_phase claim requires exactly 3 phases.\n"
                            f"You gave expected_phases={expected_phases!r} which parsed to {len(phase_list)} phase(s).\n"
                            "Example: expected_phases='fcc+tau+gamma'"
                        ),
                        "citations": ["pycalphad"]
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported claim_type '{claim_type}'. "
                             "Use 'two_phase', 'three_phase', or 'phase_fraction'.",
                    "citations": ["pycalphad"]
                }
            
            # Validate composition type
            comp_type = composition_type.lower()
            if comp_type not in ["atomic", "weight"]:
                return {"success": False, "error": f"composition_type must be 'atomic' or 'weight', got '{composition_type}'", "citations": ["pycalphad"]}
            
            if comp_type == "weight":
                return {"success": False, "error": "Weight percent (wt.%) is not yet implemented. Please use composition_type='atomic' (at.%)", "citations": ["pycalphad"]}
            
            # Parse system
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [e.strip().upper() for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]
            
            if len(elements) < 2:
                return {"success": False, "error": f"Need at least 2 elements in system '{system}'", "citations": ["pycalphad"]}
            
            # Parse element ranges
            try:
                ranges_dict = json.loads(element_ranges) if isinstance(element_ranges, str) else element_ranges
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid element_ranges JSON: {e}", "citations": ["pycalphad"]}
            
            # Determine balance element (first one not in ranges)
            balance_element = None
            for el in elements:
                if el not in ranges_dict:
                    balance_element = el
                    break
            
            if not balance_element:
                return {"success": False, "error": "Need one balance element not specified in ranges", "citations": ["pycalphad"]}
            
            _log.debug(f"Sweeping {system} with ranges {ranges_dict}, balance={balance_element}")
            
            # Pre-load database and phases once for the entire grid (performance optimization)
            _log.debug(f"Pre-loading thermodynamic database for {'-'.join(elements)} system...")
            db = load_tdb_database(elements)
            if db is None:
                return {
                    "success": False,
                    "error": f"No thermodynamic database found for {'-'.join(elements)}",
                    "citations": ["pycalphad"]
                }
            
            phases = get_phases_for_elements(db, elements, self._phase_elements)
            _log.debug(f"Pre-loaded {len(phases)} phases: {', '.join(phases[:10])}{'...' if len(phases) > 10 else ''}")
            
            # Build grid
            grid_elements = list(ranges_dict.keys())
            if len(grid_elements) == 1:
                # 1D sweep
                el1 = grid_elements[0]
                el1_min, el1_max = ranges_dict[el1]
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                grid_compositions = []
                
                for v1 in el1_vals:
                    # Normalize all element names to uppercase to avoid case sensitivity issues
                    comp = {balance_element.upper(): float(100 - v1), el1.upper(): float(v1)}
                    # Add other elements as 0
                    for el in elements:
                        el_upper = el.upper()
                        if el_upper not in comp:
                            comp[el_upper] = 0.0
                    grid_compositions.append(comp)
                    
            elif len(grid_elements) == 2:
                # 2D sweep
                el1, el2 = grid_elements
                el1_min, el1_max = ranges_dict[el1]
                el2_min, el2_max = ranges_dict[el2]
                
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                el2_vals = np.linspace(el2_min, el2_max - 1e-6, grid_points)
                
                # Determine if we should skip binary edges
                # Skip if: (1) ternary system, (2) both ranges start from 0
                is_ternary = len(elements) >= 3
                skip_binary_edges = is_ternary and el1_min == 0 and el2_min == 0
                
                if skip_binary_edges:
                    _log.info(f"Ternary system detected: will skip binary edge compositions (where {el1}=0 or {el2}=0)")
                
                grid_compositions = []
                for v1 in el1_vals:
                    for v2 in el2_vals:
                        balance_val = 100 - v1 - v2
                        if balance_val >= 0:  # Valid composition
                            # Skip binary edges in ternary systems
                            # (compositions where one varying element is exactly 0)
                            if skip_binary_edges and (v1 == 0 or v2 == 0):
                                _log.debug(f"Skipping binary edge: {el1}={v1:.3f}, {el2}={v2:.3f}")
                                continue
                            
                            # Normalize all element names to uppercase to avoid case sensitivity issues
                            comp = {
                                balance_element.upper(): float(balance_val), 
                                el1.upper(): float(v1), 
                                el2.upper(): float(v2)
                            }
                            # Add other elements as 0
                            for el in elements:
                                el_upper = el.upper()
                                if el_upper not in comp:
                                    comp[el_upper] = 0.0
                            grid_compositions.append(comp)
            else:
                return {"success": False, "error": "Only supports 1D or 2D sweeps currently", "citations": ["pycalphad"]}
            
            _log.info(f"Generated {len(grid_compositions)} grid points")
            
            # Log all generated compositions for debugging
            for idx, comp in enumerate(grid_compositions):
                comp_summary = ", ".join([f"{el}={comp[el]:.3f}" for el in elements if comp[el] > 0.001])
                _log.info(f"Grid point {idx+1}/{len(grid_compositions)}: {comp_summary}")
            
            # Evaluate claim at each grid point - PARALLEL EXECUTION
            # Create coroutines for all grid points
            async def process_grid_point(comp_dict):
                """Process a single grid point and return structured result."""
                # Format composition string with separators for parser compatibility
                # Use "-" separator when we have decimals to avoid ambiguity
                comp_parts = [f"{el}{comp_dict[el]:.1f}" for el in elements if comp_dict[el] > 0.01]
                comp_str = "-".join(comp_parts)
                
                # Call the single-point checker (reuse pre-loaded db and phases)
                result = await self.fact_check_microstructure_claim(
                    system=system,
                    composition=comp_str,
                    claim_type=claim_type,
                    expected_phases=expected_phases,
                    phase_to_check=phase_to_check,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                    process_type=process_type,
                    temperature=None,
                    composition_constraints=None,  # Don't gate - we're generating valid points
                    preloaded_db=db,  # Reuse pre-loaded database
                    preloaded_phases=phases  # Reuse pre-filtered phase list
                )
                
                # Extract phase information from supporting_data (handles both "phases" and "all_phases" keys)
                supporting_data = result.get("supporting_data", {})
                error_msg = result.get("error")
                
                # Debug: log if we're missing phase data
                if not supporting_data and not result.get("success", True):
                    _log.warning(f"Point {comp_str}: fact_check failed - {error_msg}")
                
                # Check which key exists - don't use 'or' because empty list is falsy
                if "phases" in supporting_data:
                    phase_list = supporting_data["phases"]
                elif "all_phases" in supporting_data:
                    phase_list = supporting_data["all_phases"]
                else:
                    phase_list = []
                    if supporting_data:  # Has data but no phase keys
                        _log.warning(f"Point {comp_str}: supporting_data has keys {list(supporting_data.keys())} but no phases/all_phases")
                
                grid_entry = {
                    "composition": comp_dict,
                    "composition_str": comp_str,
                    "score": result.get("score", 0),
                    "confidence": result.get("confidence", 0),
                    "phases": phase_list,
                    "reasoning": result.get("claim_analysis", {}).get("reasoning", "")
                }
                
                # Add mechanical score if evaluated
                if require_mechanical_desirability and "mechanical_score" in result:
                    grid_entry["mechanical_score"] = result["mechanical_score"]
                
                # Add error if present
                if error_msg:
                    grid_entry["error"] = error_msg
                
                return grid_entry
            
            # Run all grid points in parallel (with optional early stopping)
            _log.info(f"Starting parallel evaluation of {len(grid_compositions)} grid points...")
            parallel_start = time.time()
            
            if stop_on_first_violation:
                # Use as_completed for early stopping on first violation (score < 0)
                tasks = [process_grid_point(comp) for comp in grid_compositions]
                results_grid = []
                stopped_early = False
                
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results_grid.append(result)
                    
                    # Stop if we find a violation (negative score indicates contradiction)
                    if result.get("score", 0) < 0:
                        _log.info(f"Early stop: Found violation at {result.get('composition_str')} (score={result.get('score')})")
                        stopped_early = True
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
                
                parallel_duration = time.time() - parallel_start
                if stopped_early:
                    _log.info(f"Stopped early after {len(results_grid)}/{len(grid_compositions)} points in {parallel_duration:.2f}s")
                else:
                    _log.info(f"Completed all {len(results_grid)} points in {parallel_duration:.2f}s (no violations found)")
            else:
                # Standard parallel execution (all points)
                results_grid = await asyncio.gather(*[process_grid_point(comp) for comp in grid_compositions])
                parallel_duration = time.time() - parallel_start
                _log.info(f"Completed parallel evaluation in {parallel_duration:.2f}s ({len(grid_compositions)/parallel_duration:.1f} points/sec)")
            
            # Calculate aggregate statistics (data only)
            total_points = len(results_grid)
            scores = [r["score"] for r in results_grid]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            # Format response - structured JSON
            comp_unit = "at.%" if comp_type == "atomic" else "wt.%"
            
            # Build composition region description
            composition_region = {
                el: {"min": r[0], "max": r[1], "unit": comp_unit}
                for el, r in ranges_dict.items()
            }
            
            result_dict = {
                "success": True,
                "system": system,
                "composition_region": composition_region,
                "grid_points": {
                    "total": total_points,
                    "per_element": grid_points
                },
                "process_type": process_type,
                "claim": {
                    "type": claim_type,
                    "phases": expected_phases or phase_to_check
                },
                "statistics": {
                    "avg_score": avg_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "score_range": max_score - min_score
                },
                "grid_results": results_grid[:20],  # Limit to first 20 for response size
                "total_compositions_evaluated": len(results_grid),
                "citations": ["pycalphad"]
            }
            
            # Add mechanical desirability info if enabled
            if require_mechanical_desirability:
                mech_scores = [r.get("mechanical_score", 0) for r in results_grid if "mechanical_score" in r]
                if mech_scores:
                    result_dict["mechanical_statistics"] = {
                        "avg_mechanical_score": sum(mech_scores) / len(mech_scores),
                        "min_mechanical_score": min(mech_scores),
                        "max_mechanical_score": max(mech_scores)
                    }
            
            return result_dict
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception("Error in region sweep fact-check")
            return error_result(
                handler="calphad",
                function="sweep_microstructure_claim_over_region",
                error=f"Region sweep failed: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
    
    @ai_function(desc="Evaluate microstructure properties for multicomponent alloys. Use this to analyze metallurgical assertions like 'Al-8Mg-4Zn forms fcc + tau phase with tau < 20%' or 'eutectic composition shows >20% intermetallics'. Returns detailed analysis with score (-2 to +2) and supporting thermodynamic data.")
    async def fact_check_microstructure_claim(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Mg-Zn', 'Fe-Cr-Ni')")],
        composition: Annotated[str, AIParam(desc="Composition in at.% (e.g., 'Al88Mg8Zn4', '88Al-8Mg-4Zn', or 'Al-8Mg-4Zn'). No wildcards or special characters.")],
        claim_type: Annotated[str, AIParam(desc="Type of claim: 'two_phase', 'three_phase', 'phase_fraction', or 'custom'")],
        expected_phases: Annotated[Optional[str], AIParam(desc="Expected phases (e.g., 'fcc+tau', 'fcc+laves+gamma'). Comma or + separated.")] = None,
        phase_to_check: Annotated[Optional[str], AIParam(desc="Specific phase to check fraction (e.g., 'tau', 'gamma', 'laves')")] = None,
        min_fraction: Annotated[Optional[float], AIParam(desc="Minimum required phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        max_fraction: Annotated[Optional[float], AIParam(desc="Maximum allowed phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        process_type: Annotated[str, AIParam(desc="Process model: 'as_cast' (after solidification, DEFAULT) or 'equilibrium_300K' (infinite-time room-temp equilibrium)")] = "as_cast",
        temperature: Annotated[Optional[float], AIParam(desc="Temperature in K for evaluation (only used if process_type='equilibrium_300K')")] = None,
        composition_constraints: Annotated[Optional[str], AIParam(desc="Composition constraints as JSON string, For example, if the claim is about an Al-Mg-Zn alloy, and the claim is that the alloy has less than 8% Mg and less than 4% Zn, the composition_constraints would be '{\"MG\": {\"lt\": 8.0}, \"ZN\": {\"lt\": 4.0}}'. Supports: lt, lte, gt, gte, between:[min,max]")] = None,
        preloaded_db: Any = None,  # Internal: pre-loaded Database for performance (not exposed to AI)
        preloaded_phases: Any = None  # Internal: pre-computed phase list for performance (not exposed to AI)
    ) -> Dict[str, Any]:
        """
        Evaluate microstructure properties using CALPHAD thermodynamic calculations.
        
        This function provides detailed thermodynamic analysis of multicomponent alloys by:
        1. Simulating the processing path (as-cast solidification or full equilibrium)
        2. Calculating resulting phases and phase fractions
        3. Interpreting CALPHAD phases into metallurgical categories (fcc, tau, gamma, etc.)
        4. Evaluating mechanical desirability based on phase distribution
        5. Returning quantitative data with score and detailed reasoning for model evaluation
        
        **IMPORTANT**: By default uses 'as_cast' process_type which simulates
        slow solidification from the melt (typical casting). This answers:
        "What do I get after the alloy freezes?" NOT "What do I get after infinite
        time at room temperature?" (which would require process_type='equilibrium_300K').
        
        Returns detailed analysis with:
        - score: -2 to +2 (-2 = claim strongly contradicted, +2 = claim strongly supported)
        - confidence: 0-1 (confidence in the analysis)
        - reasoning: Detailed explanation of the microstructure findings
        - mechanical_score: -1 to +1 (brittleness/desirability assessment for as_cast)
        - phases: List of phases with fractions and categories
        - supporting_data: Complete phase fractions and thermodynamic details
        
        Example usage:
        - Claim: "Slowly solidified Al-8Mg-4Zn forms fcc + tau with tau < 20%"
          → claim_type='two_phase', expected_phases='fcc+tau', max_fraction=0.20, process_type='as_cast'
        
        - Claim: "After equilibration at 300K, Al-8Mg-4Zn has fcc + Laves + gamma"
          → claim_type='three_phase', expected_phases='fcc+laves+gamma', process_type='equilibrium_300K'
        
        - Claim: "Eutectic Al-Mg-Zn (~34.5% Mg, 5% Zn) has >20% tau after casting"
          → claim_type='phase_fraction', phase_to_check='tau', min_fraction=0.20, process_type='as_cast'
        """
        start_time = time.time()
        
        try:
            from .fact_checker import (
                AlloyFactChecker, TwoPhaseChecker, ThreePhaseChecker, 
                PhaseFractionChecker, interpret_microstructure
            )
            from ...shared.converters import atpct_to_molefrac
            from .solidification_utils import simulate_as_cast_microstructure_simple
            
            # Parse system to get elements
            parsed = self._normalize_system(system, db=None)
            if len(parsed) < 2:
                return {"success": False, "error": "System must have at least 2 elements", "citations": ["pycalphad"]}
            
            # For ternary systems, extract all 3 elements from system string
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [e.strip().upper() for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]  # Valid element symbols
            
            if len(elements) < 2:
                return {"success": False, "error": f"Could not parse elements from system '{system}'", "citations": ["pycalphad"]}
            
            # Parse composition (e.g., "Al88Mg8Zn4" or "88Al-8Mg-4Zn" or "Al-8Mg-4Zn")
            comp_dict = parse_composition_string(composition, elements)
            if not comp_dict:
                return {"success": False, "error": f"Could not parse composition '{composition}'", "citations": ["pycalphad"]}
            
            # Check composition constraints if provided
            violations = []
            composition_within_bounds = True
            
            if composition_constraints:
                import json
                try:
                    constraints_dict = json.loads(composition_constraints) if isinstance(composition_constraints, str) else composition_constraints
                    
                    for el, rules in constraints_dict.items():
                        el_upper = el.upper()
                        if el_upper not in comp_dict:
                            continue  # Element not in alloy, skip
                        
                        val = comp_dict[el_upper]  # at.%
                        
                        # Check less than
                        if "lt" in rules:
                            if not (val < rules["lt"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not < {rules['lt']:.2f} at.%")
                        
                        # Check less than or equal
                        if "lte" in rules:
                            if not (val <= rules["lte"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not ≤ {rules['lte']:.2f} at.%")
                        
                        # Check greater than
                        if "gt" in rules:
                            if not (val > rules["gt"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not > {rules['gt']:.2f} at.%")
                        
                        # Check greater than or equal
                        if "gte" in rules:
                            if not (val >= rules["gte"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not ≥ {rules['gte']:.2f} at.%")
                        
                        # Check between range
                        if "between" in rules:
                            lo, hi = rules["between"]
                            if not (lo <= val <= hi):
                                violations.append(f"{el_upper}={val:.2f} at.% not in [{lo:.2f}, {hi:.2f}] at.%")
                    
                    composition_within_bounds = (len(violations) == 0)
                    
                    if violations:
                        _log.warning(f"Composition constraint violations: {'; '.join(violations)}")
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    _log.error(f"Failed to parse composition_constraints: {e}")
                    return {"success": False, "error": f"Invalid composition_constraints format: {str(e)}", "citations": ["pycalphad"]}
            
            # Convert to mole fractions
            comp_molefrac = atpct_to_molefrac(comp_dict)
            
            # Load or reuse database (reuse if provided for performance)
            db = preloaded_db or load_tdb_database(elements)
            if db is None:
                return {"success": False, "error": f"No .tdb found for system {system}", "citations": ["pycalphad"]}
            
            # Filter or reuse phases (reuse if provided for performance)
            phases = preloaded_phases or get_phases_for_elements(db, elements, self._phase_elements)
            _log.debug(f"Selected {len(phases)} phases for {'-'.join(elements)} system")
            
            # Determine processing path and phase fractions
            precalc_fractions = None
            process_description = ""
            
            if process_type.lower() == "as_cast":
                # Simulate as-cast microstructure (after slow solidification)
                _log.debug("Using as-cast solidification simulation")
                
                # Warn if user passed temperature but it will be ignored
                if temperature is not None and temperature != 300.0:
                    _log.warning(f"Temperature {temperature}K was provided but will be ignored in as_cast mode. "
                               f"As-cast uses the freezing range temperature from solidification simulation. "
                               f"Use process_type='equilibrium_300K' to honor the specified temperature.")
                
                # Offload CPU-bound solidification simulation to thread for true parallelism
                precalc_fractions, T_ascast = await asyncio.to_thread(
                    simulate_as_cast_microstructure_simple,
                    db, elements, phases, comp_molefrac
                )
                
                if not precalc_fractions or T_ascast is None:
                    return {
                        "success": False,
                        "error": "As-cast solidification simulation failed. Try process_type='equilibrium_300K' instead.",
                        "citations": ["pycalphad"]
                    }
                
                # Use the actual as-cast temperature from simulation
                T_ref = T_ascast
                process_description = f"as-cast after slow solidification from melt (T={T_ascast:.0f}K, ~{T_ascast-273.15:.0f}°C)"
                
                # Add note to description if temperature was overridden
                if temperature is not None and temperature != 300.0:
                    process_description += f" [Note: provided temperature {temperature}K was overridden by solidification model]"
                
            elif process_type.lower() == "equilibrium_300k":
                # Traditional equilibrium at specified temperature
                T_ref = temperature or 300.0
                _log.info(f"Using equilibrium calculation at {T_ref}K")
                
                # For low temperatures (<500K), exclude LIQUID as it's metastable
                if T_ref < 500.0 and "LIQUID" in phases:
                    phases = [p for p in phases if p != "LIQUID"]
                    _log.info(f"Excluded LIQUID phase at low temperature ({T_ref}K)")
                
                process_description = f"equilibrium at {T_ref:.0f}K after infinite diffusion time"
                # precalc_fractions stays None, will be calculated during check()
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown process_type '{process_type}'. Use 'as_cast' or 'equilibrium_300K'.",
                    "citations": ["pycalphad"]
                }
            
            # Create fact-checker (T_ref used only for equilibrium calculations if needed)
            fact_checker = AlloyFactChecker(db, elements, phases, T_ref)
            
            # Add appropriate checker based on claim type
            claim_type_eval = claim_type.lower()
            checker_added = False
            
            if claim_type_eval == "two_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for two_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                
                # Auto-correct claim_type based on actual number of phases
                if len(phase_list) == 3:
                    claim_type_eval = "three_phase"
                elif len(phase_list) != 2:
                    return {"success": False, "error": "two_phase claim requires exactly 2 phases", "citations": ["pycalphad"]}
                
                if claim_type_eval == "two_phase":
                    # Map phase names to categories
                    primary_cat = map_phase_to_category(phase_list[0])
                    secondary_cat = map_phase_to_category(phase_list[1])
                    
                    checker = TwoPhaseChecker(
                        db, elements, phases,
                        primary_category=primary_cat,
                        secondary_category=secondary_cat,
                        secondary_max_fraction=max_fraction or 0.20,
                        temperature=T_ref
                    )
                    fact_checker.add_checker(checker)
                    checker_added = True
                
            if claim_type_eval == "three_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for three_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                
                # Auto-correct claim_type based on actual number of phases
                if len(phase_list) == 2:
                    claim_type_eval = "two_phase"
                    # Map phase names to categories
                    primary_cat = map_phase_to_category(phase_list[0])
                    secondary_cat = map_phase_to_category(phase_list[1])
                    
                    checker = TwoPhaseChecker(
                        db, elements, phases,
                        primary_category=primary_cat,
                        secondary_category=secondary_cat,
                        secondary_max_fraction=max_fraction or 0.20,
                        temperature=T_ref
                    )
                    fact_checker.add_checker(checker)
                    checker_added = True
                elif len(phase_list) == 3:
                    # Map to categories
                    categories = [map_phase_to_category(p) for p in phase_list]
                    
                    checker = ThreePhaseChecker(
                        db, elements, phases,
                        expected_categories=categories,
                        temperature=T_ref
                    )
                    fact_checker.add_checker(checker)
                    checker_added = True
                else:
                    return {"success": False, "error": "three_phase claim requires exactly 3 phases", "citations": ["pycalphad"]}
                
            if claim_type_eval == "phase_fraction":
                if not phase_to_check:
                    return {"success": False, "error": "phase_to_check required for phase_fraction claim", "citations": ["pycalphad"]}
                
                target_cat = map_phase_to_category(phase_to_check.lower())
                
                checker = PhaseFractionChecker(
                    db, elements, phases,
                    target_category=target_cat,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                    temperature=T_ref
                )
                fact_checker.add_checker(checker)
                checker_added = True
            
            if not checker_added:
                return {"success": False, "error": f"Unknown claim_type: {claim_type}. Use 'two_phase', 'three_phase', or 'phase_fraction'", "citations": ["pycalphad"]}
            
            # Evaluate claims (with precalculated fractions if using as_cast)
            results = fact_checker.evaluate_all(comp_molefrac, precalculated_fractions=precalc_fractions)
            report = fact_checker.generate_report(
                comp_molefrac, 
                precalculated_fractions=precalc_fractions,
                process_description=process_description
            )
            
            # Calculate mechanical properties using physics-based models (only for as_cast)
            mech_score = 0.0
            mech_details = None
            
            if process_type.lower() == "as_cast" and precalc_fractions:
                # For as-cast, evaluate mechanical properties using physics-based assessment
                # (Only meaningful for as-cast microstructures, not infinite-time equilibrium)
                try:
                    from ...alloys.mechanical_utils import get_phase_mechanical_descriptors
                    from ...alloys.assessment_utils import assess_mechanical_effects
                    
                    microstructure = interpret_microstructure(precalc_fractions)
                    phase_categories = {p.base_name: p.category.value for p in microstructure}
                    
                    # Build microstructure dict for assessment
                    phase_fractions = {name: frac for name, frac in precalc_fractions.items() if frac > 0.01}
                    
                    # Identify matrix phase (highest fraction)
                    matrix_phase = max(phase_fractions, key=phase_fractions.get) if phase_fractions else None
                    
                    if matrix_phase:
                        # Get mechanical descriptors for all phases
                        # Note: Using empty composition hint since we don't have per-phase composition here
                        matrix_desc = get_phase_mechanical_descriptors(
                            matrix_phase,
                            comp_dict,  # Use bulk composition as hint
                            tuple(elements),
                            None  # No Materials Project client available in verification context
                        )
                        
                        sec_descs = {}
                        for phase_name in phase_fractions:
                            if phase_name != matrix_phase:
                                sec_descs[phase_name] = get_phase_mechanical_descriptors(
                                    phase_name,
                                    comp_dict,
                                    tuple(elements),
                                    None
                                )
                        
                        # Build microstructure structure for assessment
                        microstructure_dict = {
                            "phase_fractions": phase_fractions,
                            "matrix_phase": matrix_phase,
                            "secondary_phases": [
                                {"name": name, "fraction": frac}
                                for name, frac in phase_fractions.items()
                                if name != matrix_phase
                            ]
                        }
                        
                        # Run physics-based assessment
                        assessment = assess_mechanical_effects(
                            matrix_desc=matrix_desc,
                            sec_descs=sec_descs,
                            microstructure=microstructure_dict,
                            phase_categories=phase_categories
                        )
                        
                        # Extract embrittlement score and normalize to [-1, 1] range
                        # Original embrittlement_score is in [0, 1] where 0=ductile, 1=brittle
                        # We want: positive=desirable(ductile), negative=undesirable(brittle)
                        emb_score = assessment.get("embrittlement_score", 0.5)
                        mech_score = 1.0 - 2.0 * emb_score  # Maps [0,1] → [1,-1]
                        
                        mech_details = {
                            "strengthening_likelihood": assessment.get("strengthening_likelihood"),
                            "embrittlement_risk": assessment.get("embrittlement_risk"),
                            "yield_strength_MPa": assessment.get("yield_strength_MPa"),
                            "embrittlement_score": emb_score,
                            "physics_based": True
                        }
                        
                        _log.info(f"Physics-based mechanical assessment: score={mech_score:.3f}, "
                                 f"embrittlement={assessment.get('embrittlement_risk')}, "
                                 f"strengthening={assessment.get('strengthening_likelihood')}")
                    else:
                        _log.warning("No matrix phase identified for mechanical assessment")
                        
                except Exception as e:
                    _log.warning(f"Physics-based mechanical assessment failed: {e}")
                    # Fallback to simple heuristic
                    microstructure = interpret_microstructure(precalc_fractions)
                    phase_categories = {p.base_name: p.category.value for p in microstructure}
                    
                    # Simple heuristic: penalize high brittle phase fractions
                    fcc_frac = sum(frac for name, frac in precalc_fractions.items() 
                                  if 'FCC' in name.upper())
                    brittle_fracs = sum(frac for name, frac in precalc_fractions.items() 
                                       if any(cat in phase_categories.get(name, "").upper() 
                                             for cat in ['LAVES', 'TAU', 'SIGMA']))
                    
                    mech_score = (fcc_frac - brittle_fracs * 2.0)
                    mech_score = max(-1.0, min(1.0, mech_score))
                    
                    mech_details = {
                        "fcc_fraction": fcc_frac,
                        "brittle_fraction": brittle_fracs,
                        "physics_based": False,
                        "error": str(e)
                    }
                    
                    _log.info(f"Fallback mechanical score: {mech_score:.3f}")
            
            # Format response
            if results:
                result = results[0]  # Get first (usually only) result
                
                # Apply composition constraint violations
                if composition_constraints and not composition_within_bounds:
                    # Composition is outside the claim's stated bounds
                    original_score = result.score
                    
                    # If microstructure analysis gave positive score but chemistry is wrong, adjust to mild negative (-1)
                    # If both were already negative, keep the harsh score
                    if original_score > 0:
                        result.score = -1
                    
                    # Append violation info to reasoning
                    result.reasoning += f" | COMPOSITION OUT OF BOUNDS: {'; '.join(violations)}"
                    
                    # Store in supporting data
                    result.supporting_data["composition_within_bounds"] = False
                    result.supporting_data["composition_violations"] = violations
                    
                    _log.warning(f"Score adjusted due to composition constraints: {original_score} → {result.score}")
                else:
                    result.supporting_data["composition_within_bounds"] = True
                    result.supporting_data["composition_violations"] = []
                
                # Format phases_info as structured data
                phases_list = []
                _log.info(f"Supporting data keys: {list(result.supporting_data.keys())}")
                
                if "phases" in result.supporting_data or "all_phases" in result.supporting_data:
                    phase_data = result.supporting_data.get("phases") or result.supporting_data.get("all_phases")
                    _log.info(f"Phase data type: {type(phase_data)}, length: {len(phase_data) if phase_data else 0}")
                    
                    if phase_data:
                        for phase_info in phase_data[:10]:  # Top 10 phases
                            if len(phase_info) >= 3:
                                phases_list.append({
                                    "name": phase_info[0],
                                    "fraction": float(phase_info[1]),
                                    "category": phase_info[2]
                                })
                else:
                    _log.warning("No 'phases' or 'all_phases' key in supporting_data")
                
                # Build structured response
                response_data = {
                    "success": True,
                    "composition": composition,
                    "system": '-'.join(elements),
                    "process": {
                        "type": process_type,
                        "description": process_description
                    },
                    "claim_analysis": {
                        "type": result.claim_text,
                        "score": result.score,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning
                    },
                    "phases": phases_list,
                    "supporting_data": result.supporting_data,
                    "citations": ["pycalphad"],
                    "score": result.score,
                    "confidence": result.confidence
                }
                
                # Add composition constraint status if checked
                if composition_constraints:
                    response_data["composition_constraints"] = {
                        "within_bounds": composition_within_bounds,
                        "violations": violations if not composition_within_bounds else []
                    }
                
                # Add mechanical score if evaluated (only for as_cast)
                if process_type.lower() == "as_cast":
                    mechanical_analysis = {
                        "score": mech_score,
                        "scale": {
                            "min": -1.0,
                            "max": 1.0,
                            "description": "Negative: high embrittlement risk, Positive: good ductility"
                        }
                    }
                    
                    # Add detailed physics-based results if available
                    if mech_details:
                        mechanical_analysis.update(mech_details)
                    
                    response_data["mechanical_analysis"] = mechanical_analysis
                    # Backwards-compatible top-level key
                    response_data["mechanical_score"] = mech_score
                
                # Include full report for debugging/detailed analysis
                response_data["full_report"] = report
                
                return response_data
            else:
                return {"success": False, "error": "No results from fact-checker", "citations": ["pycalphad"]}
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error in fact-check microstructure claim")
            return error_result(
                handler="calphad",
                function="fact_check_microstructure_claim",
                error=f"Fact-check failed: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
