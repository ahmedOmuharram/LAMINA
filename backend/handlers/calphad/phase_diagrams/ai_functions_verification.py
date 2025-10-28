"""
Advanced verification AI function methods for CALPHAD phase diagrams.

Contains sophisticated verification and fact-checking functions:
- verify_phase_formation_across_composition: Verify phase formation across composition ranges
- sweep_microstructure_claim_over_region: Sweep composition space to evaluate claims
- fact_check_microstructure_claim: Fact-check microstructure claims
"""
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time

from pycalphad import Database
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
            from pycalphad import Database, equilibrium, variables as v
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
            
            # Normalize phase name and handle category names
            from .fact_checker import PHASE_CLASSIFICATION
            
            phase_name_upper = phase_name.upper()
            available_phases = sorted(phases)
            
            # Debug: List all available phases
            _log.info(f"Available phases in {system_str} database: {available_phases}")
            
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
                    _log.info(f"Mapped category '{phase_name}' to database phases: {candidate_phases}")
            
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
            
            _log.info(f"Checking phase '{phase_to_check}' formation across {len(threshold_compositions)} compositions at T={temperature}K")
            if is_ternary:
                _log.info(f"  Ternary: varying {threshold_elem}, fixing {fixed_elem}={fixed_composition}%, balance={balance_elem}")
            
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
                    from .equilibrium_utils import extract_phase_fractions_from_equilibrium
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
                        'phase_fraction': phase_fraction,
                        'all_phases': phase_fractions
                    }
                    
                    # Add at.% for each element
                    for el in elements:
                        result[f'at_pct_{el}'] = comp_dict[el] * 100
                    
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
            
            # Build response
            response_lines = [
                f"## Phase Formation Analysis: {system_str} System",
                f"**Phase**: {display_name}",
                f"**Temperature**: {temperature:.1f} K",
                f"**Threshold**: {threshold_pct:.1f} at.% {threshold_elem}",
            ]
            
            if is_ternary:
                response_lines.append(f"**Fixed**: {fixed_composition:.1f} at.% {fixed_elem}")
            
            response_lines.extend([
                "",
                "### Results:",
                ""
            ])
            
            # Summary statistics
            total_below = len(below_threshold)
            total_above = len(above_threshold)
            
            if total_below > 0:
                fraction_below = phase_count_below / total_below
                response_lines.append(f"**Below threshold** (<{threshold_pct:.1f}% {threshold_elem}):")
                response_lines.append(f"  - Phase '{display_name}' present in {phase_count_below}/{total_below} compositions ({fraction_below*100:.1f}%)")
                
                # Show example phases below threshold (pick one where phase is present if possible)
                if below_threshold:
                    # Prefer an example where the phase is actually present
                    present_examples = [r for r in below_threshold if r['phase_present']]
                    example = present_examples[len(present_examples)//2] if present_examples else below_threshold[len(below_threshold)//2]
                    
                    # Format phases with FCC_A1 host labeling
                    phases_str_list = []
                    for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                        if f > 0.01:
                            if p == 'FCC_A1':
                                host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                                phases_str_list.append(f"{p}({host_el}-rich)({f*100:.1f}%)")
                            else:
                                phases_str_list.append(f"{p}({f*100:.1f}%)")
                    phases_str = ", ".join(phases_str_list)
                    
                    # Build composition string dynamically
                    comp_parts = [f"{example[f'at_pct_{el}']:.1f}% {el}" for el in elements]
                    response_lines.append(f"  - Example at {' / '.join(comp_parts)}: {phases_str}")
            
            response_lines.append("")
            
            if total_above > 0:
                fraction_above = phase_count_above / total_above
                response_lines.append(f"**Above threshold** (≥{threshold_pct:.1f}% {threshold_elem}):")
                response_lines.append(f"  - Phase '{display_name}' present in {phase_count_above}/{total_above} compositions ({fraction_above*100:.1f}%)")
                
                # Show example phases above threshold (pick one where phase is present if possible)
                if above_threshold:
                    # Prefer an example where the phase is actually present
                    present_examples = [r for r in above_threshold if r['phase_present']]
                    example = present_examples[len(present_examples)//2] if present_examples else above_threshold[len(above_threshold)//2]
                    
                    # Format phases with FCC_A1 host labeling
                    phases_str_list = []
                    for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                        if f > 0.01:
                            if p == 'FCC_A1':
                                host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                                phases_str_list.append(f"{p}({host_el}-rich)({f*100:.1f}%)")
                            else:
                                phases_str_list.append(f"{p}({f*100:.1f}%)")
                    phases_str = ", ".join(phases_str_list)
                    
                    # Build composition string dynamically
                    comp_parts = [f"{example[f'at_pct_{el}']:.1f}% {el}" for el in elements]
                    response_lines.append(f"  - Example at {' / '.join(comp_parts)}: {phases_str}")
            
            response_lines.append("")
            response_lines.append("### Verification:")
            response_lines.append("")
            
            # Compute frequencies (already calculated above, but ensure they're defined)
            fraction_below = (phase_count_below / total_below) if total_below > 0 else 0.0
            fraction_above = (phase_count_above / total_above) if total_above > 0 else 0.0
            
            # Use frequency-based comparison, not raw counts
            eps = 0.05  # 5 percentage points tolerance to avoid false flips from numerical noise
            
            if fraction_above > 0 and fraction_below == 0:
                response_lines.append(
                    f"✅ **VERIFIED**: Phase '{display_name}' forms above {threshold_pct:.1f}% {threshold_elem} "
                    f"and is absent below this threshold."
                )
            
            elif fraction_above >= eps and (fraction_above - fraction_below) > eps:
                response_lines.append(
                    f"⚠️ **PARTIALLY VERIFIED**: Phase '{display_name}' forms more frequently above "
                    f"{threshold_pct:.1f}% {threshold_elem} ({fraction_above*100:.1f}% vs {fraction_below*100:.1f}% of samples), "
                    f"but it can also appear below."
                )
            
            elif fraction_below >= eps and (fraction_below - fraction_above) > eps:
                response_lines.append(
                    f"❌ **CONTRADICTED**: Phase '{display_name}' is actually more frequent below "
                    f"{threshold_pct:.1f}% {threshold_elem} ({fraction_below*100:.1f}% vs {fraction_above*100:.1f}% of samples), "
                    f"opposite to the claim."
                )
            
            elif fraction_above > 0 and fraction_below > 0 and abs(fraction_above - fraction_below) <= eps:
                response_lines.append(
                    f"❌ **NOT VERIFIED**: Phase '{display_name}' appears both above and below "
                    f"{threshold_pct:.1f}% {threshold_elem} with similar frequency "
                    f"({fraction_above*100:.1f}% vs {fraction_below*100:.1f}% of samples)."
                )
            
            elif fraction_above == 0 and fraction_below > 0:
                response_lines.append(
                    f"❌ **CONTRADICTED**: Phase '{display_name}' actually forms below "
                    f"{threshold_pct:.1f}% {threshold_elem}, not above."
                )
            
            else:
                response_lines.append(
                    f"⚠️ **INCONCLUSIVE**: Phase '{display_name}' was not detected at {temperature:.1f} K "
                    f"across the sampled range. It may form at other temperatures."
                )
            
            # Show detailed composition scan
            response_lines.append("")
            response_lines.append("### Detailed Phase Presence:")
            response_lines.append("")
            response_lines.append(f"_Note: Table shows ~10 representative compositions from {len(results)} total sampled points._")
            response_lines.append("")
            
            # Build table header dynamically
            element_headers = " | ".join([f"{el} at.%" for el in elements])
            header = f"| {element_headers} | {phase_to_check} Present | {phase_to_check} Fraction | Other Major Phases |"
            separator = "|" + "|".join(["---------"] * (len(elements) + 3)) + "|"
            
            response_lines.append(header)
            response_lines.append(separator)
            
            for r in results[::max(1, len(results)//10)]:  # Show ~10 representative points
                present = "✓" if r['phase_present'] else "✗"
                fraction = f"{r['phase_fraction']*100:.1f}%" if r['phase_fraction'] > 0 else "-"
                
                # Build other_phases list with FCC_A1 host element labeling
                other_phases_list = []
                for p, f in sorted(r['all_phases'].items(), key=lambda x: -x[1]):
                    if p != phase_to_check and f > 0.05:
                        # For FCC_A1, indicate which element is the host (majority)
                        if p == 'FCC_A1':
                            # Find majority element at this composition
                            host_el = max(elements, key=lambda el: r[f'at_pct_{el}'])
                            other_phases_list.append(f"{p}({host_el}-rich)")
                        else:
                            other_phases_list.append(p)
                    if len(other_phases_list) >= 3:
                        break
                other_phases = ", ".join(other_phases_list)
                
                # Build row with dynamic element columns
                element_values = " | ".join([f"{r[f'at_pct_{el}']:.1f}" for el in elements])
                response_lines.append(f"| {element_values} | {present} | {fraction} | {other_phases} |")
            
            duration_ms = (time.time() - start_time) * 1000
            return success_result(
                handler="calphad",
                function="verify_phase_formation_across_composition",
                data={
                    "message": "\n".join(response_lines),
                    "system": system_str,
                    "phase": display_name,
                    "threshold": {"element": threshold_elem, "value_at_pct": threshold_pct},
                    "temperature_K": temperature,
                    "verification_summary": {
                        "phase_count_below": phase_count_below,
                        "phase_count_above": phase_count_above,
                        "total_below": total_below,
                        "total_above": total_above,
                        "fraction_below": fraction_below,
                        "fraction_above": fraction_above
                    }
                },
                citations=["pycalphad"],
                confidence=Confidence.HIGH if (fraction_above > 0 and fraction_below == 0) else Confidence.MEDIUM,
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
    
    @ai_function(desc="Sweep composition space and evaluate whether a microstructure claim holds across an entire region. Use this to test universal claims like 'all Al-Mg-Zn alloys with Mg<8 at.% and Zn<4 at.% form desirable fcc+tau microstructures'. Returns grid of results and overall verdict.")
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
        require_mechanical_desirability: Annotated[bool, AIParam(desc="If true, also require positive mechanical desirability score")] = False
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
                if len(phase_list) != 2:
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
                if len(phase_list) != 3:
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
            
            _log.info(f"Sweeping {system} with ranges {ranges_dict}, balance={balance_element}")
            
            # Build grid
            grid_elements = list(ranges_dict.keys())
            if len(grid_elements) == 1:
                # 1D sweep
                el1 = grid_elements[0]
                el1_min, el1_max = ranges_dict[el1]
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                grid_compositions = []
                
                for v1 in el1_vals:
                    comp = {balance_element: 100 - v1, el1: v1}
                    # Add other elements as 0
                    for el in elements:
                        if el not in comp:
                            comp[el] = 0.0
                    grid_compositions.append(comp)
                    
            elif len(grid_elements) == 2:
                # 2D sweep
                el1, el2 = grid_elements
                el1_min, el1_max = ranges_dict[el1]
                el2_min, el2_max = ranges_dict[el2]
                
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                el2_vals = np.linspace(el2_min, el2_max - 1e-6, grid_points)
                
                grid_compositions = []
                for v1 in el1_vals:
                    for v2 in el2_vals:
                        balance_val = 100 - v1 - v2
                        if balance_val >= 0:  # Valid composition
                            comp = {balance_element: balance_val, el1: v1, el2: v2}
                            # Add other elements as 0
                            for el in elements:
                                if el not in comp:
                                    comp[el] = 0.0
                            grid_compositions.append(comp)
            else:
                return {"success": False, "error": "Only supports 1D or 2D sweeps currently", "citations": ["pycalphad"]}
            
            _log.info(f"Generated {len(grid_compositions)} grid points")
            
            # Evaluate claim at each grid point
            results_grid = []
            pass_count = 0
            fail_count = 0
            mech_fail_count = 0
            
            for comp_dict in grid_compositions:
                # Format composition string with separators for parser compatibility
                # Use "-" separator when we have decimals to avoid ambiguity
                comp_parts = [f"{el}{comp_dict[el]:.1f}" for el in elements if comp_dict[el] > 0.01]
                comp_str = "-".join(comp_parts)
                
                # Call the single-point checker
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
                    composition_constraints=None  # Don't gate - we're generating valid points
                )
                
                point_pass = result.get("verdict", False)
                mech_ok = True
                
                if require_mechanical_desirability and "mechanical_score" in result:
                    mech_ok = result["mechanical_score"] > 0
                    if not mech_ok:
                        mech_fail_count += 1
                
                overall_pass = point_pass and mech_ok
                
                if overall_pass:
                    pass_count += 1
                else:
                    fail_count += 1
                
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
                
                results_grid.append({
                    "composition": comp_dict,
                    "composition_str": comp_str,
                    "microstructure_verdict": point_pass,
                    "mechanical_ok": mech_ok if require_mechanical_desirability else None,
                    "overall_pass": overall_pass,
                    "score": result.get("score", 0),
                    "phases": phase_list,
                    "error": error_msg  # Include error for debugging
                })
            
            # Aggregate verdict
            total_points = len(results_grid)
            pass_fraction = pass_count / total_points if total_points > 0 else 0.0
            
            if pass_fraction == 1.0:
                overall_verdict = "UNIVERSALLY SUPPORTED"
                overall_score = 2
                confidence = 1.0
            elif pass_fraction >= 0.9:
                overall_verdict = "MOSTLY SUPPORTED"
                overall_score = 1
                confidence = 0.8
            elif pass_fraction >= 0.5:
                overall_verdict = "MIXED"
                overall_score = 0
                confidence = 0.5
            elif pass_fraction > 0:
                overall_verdict = "MOSTLY REJECTED"
                overall_score = -1
                confidence = 0.7
            else:
                overall_verdict = "UNIVERSALLY REJECTED"
                overall_score = -2
                confidence = 1.0
            
            # Format response
            comp_unit = "at.%" if comp_type == "atomic" else "wt.%"
            message_lines = [
                f"## Region Sweep Fact-Check Result",
                f"",
                f"**System**: {system}",
                f"**Composition Region**: {', '.join([f'{el} ∈ [{r[0]}, {r[1]}) {comp_unit}' for el, r in ranges_dict.items()])}",
                f"**Grid Points**: {total_points} ({grid_points} per element)",
                f"**Process**: {process_type}",
                f"**Claim**: {claim_type} - {expected_phases or phase_to_check}",
                f"",
                f"### Verdict: **{overall_verdict}**",
                f"- **Score**: {overall_score:+d}/2 (confidence: {confidence:.0%})",
                f"- **Pass Rate**: {pass_count}/{total_points} compositions ({pass_fraction:.1%})",
            ]
            
            if require_mechanical_desirability:
                microstructure_pass = pass_count + mech_fail_count  # Points that passed microstructure check
                message_lines.append(f"- **Mechanical Desirability**: Required (score > 0)")
                message_lines.append(f"  - Microstructure match: {microstructure_pass}/{total_points}")
                message_lines.append(f"  - Mechanical failures: {mech_fail_count}/{total_points} (brittle/undesirable phases)")
                message_lines.append(f"  - Overall pass: {pass_count}/{total_points}")
            
            message_lines.extend([
                f"",
                f"### Interpretation:",
            ])
            
            if overall_score >= 1:
                message_lines.append(f"The claim holds across {'all' if overall_score == 2 else 'most of'} the stated composition region. The microstructure prediction is {'universally' if overall_score == 2 else 'generally'} valid.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: Some compositions matched the claimed phases but were rejected due to poor mechanical properties (e.g., brittle intermetallics dominating).")
            elif overall_score == 0:
                message_lines.append(f"The claim holds in some parts of the region but fails in others. The statement is over-generalized.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: {mech_fail_count} compositions matched the phases but had undesirable mechanical properties.")
            else:
                message_lines.append(f"The claim fails across {'all' if overall_score == -2 else 'most of'} the composition region. The statement is incorrect.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: Some failures were due to poor mechanical desirability rather than wrong phase mix.")
            
            # Sample failures with detailed reasons
            failures = [r for r in results_grid if not r["overall_pass"]]
            if failures and len(failures) <= 5:
                message_lines.append(f"")
                message_lines.append(f"**Failed Compositions:**")
                for fail in failures:
                    reason = ""
                    if require_mechanical_desirability and fail.get("mechanical_ok") is False:
                        reason = " (mechanical failure)"
                    elif not fail.get("microstructure_verdict"):
                        reason = " (microstructure mismatch)"
                    message_lines.append(f"- {fail['composition_str']}: Score {fail['score']:+d}{reason}")
            elif failures:
                message_lines.append(f"")
                message_lines.append(f"**Sample Failures** (showing 5 of {len(failures)}):")
                for fail in failures[:5]:
                    reason = ""
                    if require_mechanical_desirability and fail.get("mechanical_ok") is False:
                        reason = " (mechanical failure)"
                    elif not fail.get("microstructure_verdict"):
                        reason = " (microstructure mismatch)"
                    message_lines.append(f"- {fail['composition_str']}: Score {fail['score']:+d}{reason}")
            
            result_dict = {
                "success": True,
                "message": "\n".join(message_lines),
                "overall_verdict": overall_verdict,
                "overall_score": overall_score,
                "confidence": confidence,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "total_points": total_points,
                "pass_fraction": pass_fraction,
                "grid_results": results_grid[:20],  # Limit to first 20 for response size
                "citations": ["pycalphad"]
            }
            
            if require_mechanical_desirability:
                result_dict["mechanical_fail_count"] = mech_fail_count
                result_dict["microstructure_pass_count"] = pass_count + mech_fail_count
            
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
    
    @ai_function(desc="Evaluate microstructure claims for multicomponent alloys. Use this to fact-check metallurgical assertions like 'Al-8Mg-4Zn forms fcc + tau phase with tau < 20%' or 'eutectic composition shows >20% intermetallics'. Returns detailed verdict with score (-2 to +2) and supporting thermodynamic data.")
    async def fact_check_microstructure_claim(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Mg-Zn', 'Fe-Cr-Ni')")],
        composition: Annotated[str, AIParam(desc="Composition in at.% (e.g., 'Al88Mg8Zn4', '88Al-8Mg-4Zn', or 'Al-8Mg-4Zn')")],
        claim_type: Annotated[str, AIParam(desc="Type of claim: 'two_phase', 'three_phase', 'phase_fraction', or 'custom'")],
        expected_phases: Annotated[Optional[str], AIParam(desc="Expected phases (e.g., 'fcc+tau', 'fcc+laves+gamma'). Comma or + separated.")] = None,
        phase_to_check: Annotated[Optional[str], AIParam(desc="Specific phase to check fraction (e.g., 'tau', 'gamma', 'laves')")] = None,
        min_fraction: Annotated[Optional[float], AIParam(desc="Minimum required phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        max_fraction: Annotated[Optional[float], AIParam(desc="Maximum allowed phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        process_type: Annotated[str, AIParam(desc="Process model: 'as_cast' (after solidification, DEFAULT) or 'equilibrium_300K' (infinite-time room-temp equilibrium)")] = "as_cast",
        temperature: Annotated[Optional[float], AIParam(desc="Temperature in K for evaluation (only used if process_type='equilibrium_300K')")] = None,
        composition_constraints: Annotated[Optional[str], AIParam(desc="Composition constraints as JSON string, For example, if the claim is about an Al-Mg-Zn alloy, and the claim is that the alloy has less than 8% Mg and less than 4% Zn, the composition_constraints would be '{\"MG\": {\"lt\": 8.0}, \"ZN\": {\"lt\": 4.0}}'. Supports: lt, lte, gt, gte, between:[min,max]")] = None
    ) -> Dict[str, Any]:
        """
        Evaluate microstructure claims using CALPHAD thermodynamic calculations.
        
        This function acts as an automated "materials expert witness" that can verify
        metallurgical assertions about multicomponent alloys by:
        1. Simulating the processing path (as-cast solidification or full equilibrium)
        2. Calculating resulting phases and phase fractions
        3. Interpreting CALPHAD phases into metallurgical categories (fcc, tau, gamma, etc.)
        4. Evaluating mechanical desirability based on phase distribution
        5. Returning a verdict with score and detailed reasoning
        
        **IMPORTANT**: By default uses 'as_cast' process_type which simulates
        slow solidification from the melt (typical casting). This answers:
        "What do I get after the alloy freezes?" NOT "What do I get after infinite
        time at room temperature?" (which would require process_type='equilibrium_300K').
        
        Returns detailed verdict with:
        - verdict: True/False (claim supported or rejected)
        - score: -2 to +2 (-2 = completely wrong, +2 = fully correct)
        - confidence: 0-1 (confidence in the verdict)
        - reasoning: Explanation of why claim passed/failed
        - mechanical_score: -1/0/+1 (brittleness/desirability assessment)
        - supporting_data: Phase fractions and thermodynamic details
        
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
            from .solidification_utils import (
                simulate_as_cast_microstructure_simple,
                mechanical_desirability_score
            )
            
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
            
            # Select database
            db = load_tdb_database(elements)
            if db is None:
                return {"success": False, "error": f"No .tdb found for system {system}", "citations": ["pycalphad"]}
            
            # Get phases
            phases = get_phases_for_elements(db, elements, self._phase_elements)
            _log.debug(f"Selected {len(phases)} phases for {'-'.join(elements)} system")
            
            # Determine processing path and phase fractions
            precalc_fractions = None
            process_description = ""
            
            if process_type.lower() == "as_cast":
                # Simulate as-cast microstructure (after slow solidification)
                _log.info("Using as-cast solidification simulation")
                
                # Warn if user passed temperature but it will be ignored
                if temperature is not None and temperature != 300.0:
                    _log.warning(f"Temperature {temperature}K was provided but will be ignored in as_cast mode. "
                               f"As-cast uses the freezing range temperature from solidification simulation. "
                               f"Use process_type='equilibrium_300K' to honor the specified temperature.")
                
                precalc_fractions, T_ascast = simulate_as_cast_microstructure_simple(
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
            if claim_type.lower() == "two_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for two_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 2:
                    return {"success": False, "error": "two_phase claim requires exactly 2 phases", "citations": ["pycalphad"]}
                
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
                
            elif claim_type.lower() == "three_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for three_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 3:
                    return {"success": False, "error": "three_phase claim requires exactly 3 phases", "citations": ["pycalphad"]}
                
                # Map to categories
                categories = [map_phase_to_category(p) for p in phase_list]
                
                checker = ThreePhaseChecker(
                    db, elements, phases,
                    expected_categories=categories,
                    temperature=T_ref
                )
                fact_checker.add_checker(checker)
                
            elif claim_type.lower() == "phase_fraction":
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
            else:
                return {"success": False, "error": f"Unknown claim_type: {claim_type}. Use 'two_phase', 'three_phase', or 'phase_fraction'", "citations": ["pycalphad"]}
            
            # Evaluate claims (with precalculated fractions if using as_cast)
            results = fact_checker.evaluate_all(comp_molefrac, precalculated_fractions=precalc_fractions)
            report = fact_checker.generate_report(
                comp_molefrac, 
                precalculated_fractions=precalc_fractions,
                process_description=process_description
            )
            
            # Calculate mechanical desirability score (only for as_cast)
            mech_score = 0.0
            mech_interpretation = "Not evaluated"
            
            if process_type.lower() == "as_cast" and precalc_fractions:
                # For as-cast, evaluate mechanical properties
                # (Only meaningful for as-cast microstructures, not infinite-time equilibrium)
                microstructure = interpret_microstructure(precalc_fractions)
                phase_categories = {p.base_name: p.category.value for p in microstructure}
                mech_score, mech_interpretation = mechanical_desirability_score(
                    precalc_fractions, phase_categories
                )
                _log.info(f"Mechanical desirability: {mech_score} - {mech_interpretation}")
            
            # Format response
            if results:
                result = results[0]  # Get first (usually only) result
                
                # Apply composition constraint violations
                if composition_constraints and not composition_within_bounds:
                    # Composition is outside the claim's stated bounds
                    # Even if microstructure matched, the claim doesn't apply here
                    original_verdict = result.verdict
                    original_score = result.score
                    
                    result.verdict = False
                    
                    # If microstructure was good but chemistry is wrong, mild fail (-1)
                    # If both were wrong, keep the harsh score
                    if original_score > 0:
                        result.score = -1
                    
                    # Append violation info to reasoning
                    result.reasoning += f" | COMPOSITION OUT OF BOUNDS: {'; '.join(violations)}"
                    
                    # Store in supporting data
                    result.supporting_data["composition_within_bounds"] = False
                    result.supporting_data["composition_violations"] = violations
                    
                    _log.warning(f"Verdict adjusted due to composition constraints: {original_verdict} → {result.verdict}, score: {original_score} → {result.score}")
                else:
                    result.supporting_data["composition_within_bounds"] = True
                    result.supporting_data["composition_violations"] = []
                
                # Format supporting data for display
                phases_info = []
                _log.info(f"Supporting data keys: {list(result.supporting_data.keys())}")
                
                if "phases" in result.supporting_data or "all_phases" in result.supporting_data:
                    phase_data = result.supporting_data.get("phases") or result.supporting_data.get("all_phases")
                    _log.info(f"Phase data type: {type(phase_data)}, length: {len(phase_data) if phase_data else 0}")
                    
                    if phase_data:
                        for phase_info in phase_data[:10]:  # Top 10 phases
                            if len(phase_info) >= 3:
                                phases_info.append(f"{phase_info[0]}: {phase_info[1]*100:.1f}% ({phase_info[2]})")
                else:
                    _log.warning("No 'phases' or 'all_phases' key in supporting_data")
                
                verdict_emoji = "✓" if result.verdict else "✗"
                score_text = f"{result.score:+d}/2"
                
                message_lines = [
                    f"## Microstructure Fact-Check Result",
                    f"",
                    f"**Composition**: {composition} ({'-'.join(elements)} system)",
                    f"**Process**: {process_description}",
                    f"**Claim**: {result.claim_text}",
                    f"",
                    f"### {verdict_emoji} Verdict: **{'SUPPORTED' if result.verdict else 'REJECTED'}**",
                    f"- **Score**: {score_text} (confidence: {result.confidence:.0%})",
                    f"- **Reasoning**: {result.reasoning}",
                ]
                
                # Add composition constraint status if checked
                if composition_constraints:
                    if composition_within_bounds:
                        message_lines.append(f"- **Composition Bounds**: ✓ Within stated constraints")
                    else:
                        message_lines.append(f"- **Composition Bounds**: ✗ VIOLATED - {'; '.join(violations)}")
                
                # Add mechanical desirability if evaluated (only for as_cast)
                if process_type.lower() == "as_cast" and (mech_score != 0.0 or mech_interpretation != "Not evaluated"):
                    mech_emoji = "✓" if mech_score > 0 else ("✗" if mech_score < 0 else "○")
                    message_lines.append(f"- **Mechanical Desirability**: {mech_emoji} {mech_interpretation} (score: {mech_score:+.1f})")
                
                if phases_info:
                    message_lines.append(f"\n### Calculated Phase Fractions:")
                    for phase_line in phases_info:
                        message_lines.append(f"- {phase_line}")
                
                message_lines.append(f"\n---")
                message_lines.append(f"\n**Full Report:**\n```\n{report}\n```")
                
                return {
                    "success": True,
                    "message": "\n".join(message_lines),
                    "verdict": result.verdict,
                    "score": result.score,
                    "confidence": result.confidence,
                    "mechanical_score": mech_score,
                    "mechanical_interpretation": mech_interpretation,
                    "process_type": process_type,
                    "supporting_data": result.supporting_data,
                    "citations": ["pycalphad"]
                }
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
