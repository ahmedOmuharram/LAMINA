"""
Calculation AI function methods for CALPHAD phase diagrams.

Contains equilibrium calculation and phase fraction analysis functions:
- calculate_equilibrium_at_point: Calculate equilibrium at specific T and composition
- calculate_phase_fractions_vs_temperature: Calculate phase fractions across temperature range
- analyze_phase_fraction_trend: Analyze phase trends with temperature
"""
import numpy as np
from typing import Optional
import logging
import time

from pycalphad import Database, equilibrium
import pycalphad.variables as v
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .database_utils import map_phase_name
from .solidification_utils import simulate_scheil_gulliver, find_liquidus_solidus_temperatures
from ...shared.calphad_utils import (
    extract_phase_fractions_from_equilibrium,
    extract_phase_fractions,
    get_phase_composition,
    get_phase_fraction_by_base_name,
    load_tdb_database,
    compute_equilibrium,
)
from ...shared.result_wrappers import success_result, error_result, Confidence, ErrorType

_log = logging.getLogger(__name__)


class CalculationsMixin:
    """Mixin class containing calculation AI functions for CalPhadHandler."""
    
    @ai_function(desc="Calculate equilibrium phase fractions at a specific temperature and composition. Use to verify phase amounts at a single condition. Returns detailed phase information including fractions and compositions.")
    async def calculate_equilibrium_at_point(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15', 'Al80Zn20', 'Fe70Cr20Ni10'). Numbers are percentages.")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin")],
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Calculate thermodynamic equilibrium at a specific point (temperature + composition).
        
        Uses compute_equilibrium utility for robust equilibrium calculation.
        
        Args:
            composition: Element-number pairs (e.g., 'Al30Si55C15'). Numbers are percentages.
            composition_type: 'atomic' (default) or 'weight'. Weight% is converted to mole fractions internally.
        
        Returns:
            JSON result with phase fractions and per-phase compositions.
        """
        start_time = time.time()
        
        try:
            # Parse composition string (e.g., "Al30Si55C15" -> {AL: 0.30, SI: 0.55, C: 0.15})
            # Note: Always returns atomic (mole) fractions, converting from weight if needed
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_equilibrium_at_point",
                    error=f"Failed to parse composition: {composition}. Use format like 'Al30Si55C15' or 'Fe70Cr20Ni10'",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_equilibrium_at_point",
                    error=f"No thermodynamic database found for {system_str} system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Calculate equilibrium using robust compute_equilibrium utility
            _log.info(f"Computing equilibrium at {temperature}K for {system_str} with composition {comp_dict}")
            eq = compute_equilibrium(
                db=db,
                elements=elements,
                phases=phases,
                composition=comp_dict,
                temperature=temperature,
                pressure=101325
            )
            
            if eq is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_equilibrium_at_point",
                    error=f"Equilibrium calculation failed at {temperature}K for {composition}",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Extract phase fractions using robust utility (handles vertices properly)
            phase_fractions_dict = extract_phase_fractions(eq, tolerance=1e-4)
            
            if not phase_fractions_dict:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_equilibrium_at_point",
                    error=f"No stable phases found at {temperature}K for {composition}",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Extract phase compositions using robust utility
            phase_info = []
            total_fraction = 0.0
            
            for phase, frac in phase_fractions_dict.items():
                total_fraction += frac
                
                # Use get_phase_composition utility for robust composition extraction
                phase_comp = get_phase_composition(eq, phase, elements)
                
                # Map phase name to readable form (e.g., CSI -> SiC)
                readable_name = map_phase_name(phase)
                
                # Convert composition to percentage
                phase_comp_percent = {
                    elem: round(comp * 100, 2)
                    for elem, comp in phase_comp.items()
                }
                
                phase_info.append({
                    'phase_name': readable_name,
                    'phase_name_raw': phase,
                    'fraction': round(frac, 4),
                    'fraction_percent': round(frac * 100, 2),
                    'composition_atomic_fraction': phase_comp,
                    'composition_atomic_percent': phase_comp_percent
                })
            
            # Sort by fraction (descending)
            phase_info.sort(key=lambda x: x['fraction'], reverse=True)
            
            # Prepare composition info
            composition_info = {
                'input_string': composition,
                'composition_type': composition_type,
                'atomic_fractions': {elem: round(frac, 4) for elem, frac in comp_dict.items()},
                'atomic_percent': {elem: round(frac * 100, 2) for elem, frac in comp_dict.items()},
                'elements': elements,
                'system': system_str
            }
            
            # Temperature info
            temperature_info = {
                'temperature_K': temperature,
                'temperature_C': round(temperature - 273.15, 2)
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return success_result(
                handler="calphad",
                function="calculate_equilibrium_at_point",
                data={
                    "temperature": temperature_info,
                    "composition": composition_info,
                    "phases": phase_info,
                    "total_fraction": round(total_fraction, 4),
                    "n_phases": len(phase_info)
                },
                citations=["pycalphad"],
                confidence=Confidence.HIGH,
                notes=[
                    f"Equilibrium calculated using robust compute_equilibrium utility",
                    f"Found {len(phase_info)} stable phase(s)",
                    f"Total phase fraction: {total_fraction*100:.2f}%"
                ],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error calculating equilibrium at {temperature}K for {composition}")
            return error_result(
                handler="calphad",
                function="calculate_equilibrium_at_point",
                error=f"Failed to calculate equilibrium: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
    
    @ai_function(desc="Calculate how phase fractions change with temperature for a specific composition. Essential for understanding precipitation, dissolution, and phase transformations. Returns phase fraction data across temperature range.")
    async def calculate_phase_fractions_vs_temperature(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15', 'Al80Zn20')")],
        min_temperature: Annotated[float, AIParam(desc="Minimum temperature in Kelvin")],
        max_temperature: Annotated[float, AIParam(desc="Maximum temperature in Kelvin")],
        temperature_step: Annotated[Optional[float], AIParam(desc="Temperature step in Kelvin. Default: 10")] = None,
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Calculate phase fractions as a function of temperature for a fixed composition.
        
        This is the primary tool for understanding:
        - Precipitation behavior (phase fraction increasing with cooling)
        - Dissolution behavior (phase fraction decreasing with heating)
        - Phase transformation temperatures
        - Solvus boundaries
        
        Returns detailed phase fraction data and analysis.
        """
        start_time = time.time()
        
        try:
            # Parse composition (always returns atomic/mole fractions, converting from weight if needed)
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_phase_fractions_vs_temperature",
                    error=f"Failed to parse composition: {composition}",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="calculate_phase_fractions_vs_temperature",
                    error=f"No thermodynamic database found for {system_str} system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Temperature array
            step = temperature_step or 10.0
            temps = np.arange(min_temperature, max_temperature + step, step)
            
            # Calculate equilibrium at each temperature
            phase_fractions = {}  # {phase_name: [fractions]}
            all_phases_seen = set()
            
            elements_with_va = elements + ['VA']
            
            _log.info(f"Calculating equilibrium for {len(temps)} temperature points...")
            
            for T in temps:
                conditions = {v.T: T, v.P: 101325, v.N: 1}
                for i, elem in enumerate(elements[1:], 1):
                    conditions[v.X(elem)] = comp_dict[elem]
                
                try:
                    eq = equilibrium(db, elements_with_va, phases, conditions)
                    
                    # Extract phases at this temperature (properly handling multiple vertices)
                    # Use looser tolerance (1e-4) for better boundary handling
                    temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    for phase in temp_phases.keys():
                        all_phases_seen.add(phase)
                    
                    # Store fractions for all seen phases
                    for phase in all_phases_seen:
                        if phase not in phase_fractions:
                            phase_fractions[phase] = []
                        phase_fractions[phase].append(temp_phases.get(phase, 0.0))
                        
                except Exception as e:
                    _log.warning(f"Equilibrium calculation failed at {T}K: {e}")
                    # Append zeros for this temperature
                    for phase in all_phases_seen:
                        if phase not in phase_fractions:
                            phase_fractions[phase] = []
                        phase_fractions[phase].append(0.0)
            
            # Generate analysis
            comp_str = " ".join([f"{elem}{comp_dict[elem]*100:.0f}" for elem in elements])
            
            response_lines = [
                f"**Phase Fractions vs Temperature for {comp_str}**\n",
                f"**Temperature Range**: {min_temperature:.0f} - {max_temperature:.0f} K ({min_temperature-273.15:.0f} - {max_temperature-273.15:.0f} °C)",
                f"**Composition**: {comp_str} (atomic %)",
                f"**Temperature Points**: {len(temps)}\n",
                "**Phase Evolution**:"
            ]
            
            # Analyze each phase
            for phase, fractions in sorted(phase_fractions.items()):
                max_frac = max(fractions)
                min_frac = min(fractions)
                
                if max_frac < 1e-6:
                    continue  # Skip phases that never appear
                
                # Find where phase appears/disappears
                frac_start = fractions[0]
                frac_end = fractions[-1]
                
                # Determine trend
                if frac_end > frac_start + 0.01:
                    trend = "increasing"
                    change = f"+{(frac_end - frac_start)*100:.2f}%"
                elif frac_start > frac_end + 0.01:
                    trend = "decreasing"
                    change = f"{(frac_end - frac_start)*100:.2f}%"
                else:
                    trend = "stable"
                    change = "~0%"
                
                response_lines.append(
                    f"  • **{phase}**: {frac_start*100:.2f}% → {frac_end*100:.2f}% "
                    f"({trend}, {change})"
                )
            
            # Store data for potential plotting
            setattr(self, '_last_phase_fraction_data', {
                'temperatures': temps.tolist(),
                'phase_fractions': {p: f.copy() for p, f in phase_fractions.items()},
                'composition': comp_dict,
                'composition_str': comp_str
            })
            
            duration_ms = (time.time() - start_time) * 1000
            return success_result(
                handler="calphad",
                function="calculate_phase_fractions_vs_temperature",
                data={
                    "message": "\n".join(response_lines),
                    "composition": comp_str,
                    "temperature_range_K": [min_temperature, max_temperature],
                    "temperature_points": len(temps),
                    "phase_evolution": {phase: {"start": fractions[0], "end": fractions[-1], "max": max(fractions)}
                                      for phase, fractions in phase_fractions.items() if max(fractions) > 1e-6}
                },
                citations=["pycalphad"],
                confidence=Confidence.HIGH,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error calculating phase fractions vs temperature")
            return error_result(
                handler="calphad",
                function="calculate_phase_fractions_vs_temperature",
                error=f"Failed to calculate phase fractions vs temperature: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
    
    @ai_function(desc="Analyze how a specific phase fraction changes with temperature. Provides objective data on precipitation/dissolution behavior with equilibrium and non-equilibrium (Scheil) predictions.")
    async def analyze_phase_fraction_trend(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15')")],
        phase_name: Annotated[str, AIParam(desc="Name of the phase to analyze (e.g., 'AL4C3', 'SIC', 'FCC_A1')")],
        start_temperature: Annotated[float, AIParam(desc="Starting temperature in Kelvin (typically lower temperature)")],
        end_temperature: Annotated[float, AIParam(desc="Ending temperature in Kelvin (typically higher temperature)")],
        include_scheil: Annotated[Optional[bool], AIParam(desc="Include Scheil-Gulliver solidification analysis for as-cast behavior. Default: False")] = False,
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Analyze the trend of a specific phase fraction with temperature.
        
        Provides objective data on:
        - Equilibrium phase fraction as function of temperature
        - Trend direction (increases/decreases with temperature)
        - Temperature ranges where phase is stable
        - Optional: Scheil-Gulliver solidification prediction for as-cast behavior
        
        Returns detailed quantitative analysis without subjective verification.
        """
        start_time = time.time()
        
        try:
            # Parse composition (always returns atomic/mole fractions)
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="analyze_phase_fraction_trend",
                    error=f"Failed to parse composition: {composition}",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="analyze_phase_fraction_trend",
                    error=f"No thermodynamic database found for {system_str} system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Debug: List all available phases in the database
            available_phases = {p.name for p in db.phases.values()}
            _log.info(f"Available phases in database: {sorted(available_phases)}")
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            _log.info(f"Filtered phases for calculation: {sorted(phases)}")
            
            # Normalize phase name
            phase_name_upper = phase_name.upper()
            
            # Check if phase exists in database
            if phase_name_upper not in phases and phase_name not in phases:
                available = ", ".join(sorted(phases))
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="analyze_phase_fraction_trend",
                    error=f"Phase '{phase_name}' not found in database. Available phases: {available}",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            phase_to_track = phase_name_upper if phase_name_upper in phases else phase_name
            
            # Calculate over temperature range with better resolution
            n_points = 100  # Increased resolution for better trend detection
            temps = np.linspace(start_temperature, end_temperature, n_points)
            fractions = []
            successful_calcs = 0
            
            elements_with_va = elements + ['VA']
            
            _log.info(f"Calculating phase fraction trend for {phase_to_track} over {n_points} temperature points")
            
            for T in temps:
                conditions = {v.T: T, v.P: 101325, v.N: 1}
                for i, elem in enumerate(elements[1:], 1):
                    conditions[v.X(elem)] = comp_dict[elem]
                
                try:
                    eq = equilibrium(db, elements_with_va, phases, conditions)
                    
                    # Extract phase fractions properly (handling multiple vertices)
                    # Use looser tolerance (1e-5) to capture trace amounts
                    temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-5)
                    
                    # Find our phase, summing all instances (e.g., SIC#1 + SIC#2)
                    phase_frac = get_phase_fraction_by_base_name(temp_phases, phase_to_track)
                    
                    # Debug: Log phase fractions at sample temperatures
                    if len(fractions) < 3 or len(fractions) == n_points // 2 or len(fractions) == n_points - 1:
                        # Collapse instances to base names for debugging
                        base_phases = {}
                        for pname, frac in temp_phases.items():
                            base_name = str(pname).split('#')[0].upper()
                            base_phases[base_name] = base_phases.get(base_name, 0.0) + frac
                        top_phases = sorted(base_phases.items(), key=lambda x: x[1], reverse=True)[:5]
                        _log.info(f"At {T:.0f}K: top phases = {top_phases}, {phase_to_track} fraction = {phase_frac:.6f}")
                    
                    if phase_frac > 1e-8:  # Track successful detections
                        successful_calcs += 1
                    
                    fractions.append(phase_frac)
                    
                except Exception as e:
                    _log.warning(f"Equilibrium calculation failed at {T:.0f}K: {e}")
                    fractions.append(0.0)
            
            # Analyze trend
            fractions = np.array(fractions)
            frac_start = float(fractions[0])
            frac_end = float(fractions[-1])
            
            max_frac = float(np.max(fractions))
            min_frac = float(np.min(fractions))
            mean_frac = float(np.mean(fractions))
            
            # Count non-zero points
            nonzero_count = np.count_nonzero(fractions > 1e-6)
            
            _log.info(f"Phase {phase_to_track} statistics: successful_calcs={successful_calcs}, nonzero_count={nonzero_count}, max={max_frac:.6f}, mean={mean_frac:.6f}")
            
            # Detect if phase is essentially absent
            if max_frac < 1e-5:
                _log.warning(f"Phase {phase_to_track} has negligible presence (max={max_frac:.2e}) - may not be stable in this composition/T range")
            
            # Compute overall trend using linear regression for robustness
            if nonzero_count >= 3:
                # Use only nonzero points for slope calculation
                nonzero_mask = fractions > 1e-6
                temps_nz = temps[nonzero_mask]
                fracs_nz = fractions[nonzero_mask]
                
                if len(temps_nz) >= 2:
                    # Linear fit
                    slope, intercept = np.polyfit(temps_nz, fracs_nz, 1)
                    # Normalize slope by temperature range for interpretation
                    normalized_slope = slope * (end_temperature - start_temperature)
                else:
                    slope = 0
                    normalized_slope = 0
            else:
                slope = 0
                normalized_slope = 0
            
            # Compute change and trend
            delta = float(frac_end - frac_start)
            
            # Determine trend from slope (more robust than endpoints)
            if abs(normalized_slope) < 0.001:
                trend = "stable"
                trend_desc = "remains relatively stable with temperature"
            elif normalized_slope > 0.001:
                trend = "increases"
                trend_desc = "increases with increasing temperature"
            else:
                trend = "decreases"
                trend_desc = "decreases with increasing temperature"
            
            # Add interpretation
            if trend == "decreases":
                interpretation = f"{phase_to_track} precipitates upon cooling (stable at lower temperatures)"
            elif trend == "increases":
                interpretation = f"{phase_to_track} dissolves upon cooling (stable at higher temperatures)"
            else:
                interpretation = f"{phase_to_track} fraction is relatively temperature-independent"
            
            # Find stability temperature range
            stable_temps = temps[fractions > 1e-4]
            if len(stable_temps) > 0:
                T_stability_start = float(stable_temps[0])
                T_stability_end = float(stable_temps[-1])
            else:
                T_stability_start = None
                T_stability_end = None
            
            # Check if phase is negligible
            is_negligible = max_frac < 1e-5
            
            comp_str = "".join([f"{elem}{comp_dict[elem]*100:.0f}" for elem in elements])
            
            # Scheil-Gulliver analysis for as-cast behavior (optional)
            scheil_data = None
            if include_scheil:
                try:
                    _log.info(f"Running Scheil-Gulliver solidification analysis for {phase_to_track}")
                    
                    # Find liquidus temperature for proper starting point
                    T_liquidus, T_solidus = find_liquidus_solidus_temperatures(
                        db, elements, phases, comp_dict, n_points=30
                    )
                    
                    if T_liquidus is not None:
                        T_start_scheil = T_liquidus + 50.0
                        T_end_scheil = max(200.0, start_temperature - 50.0)
                        
                        # Run Scheil-Gulliver simulation
                        scheil_phases = simulate_scheil_gulliver(
                            db=db,
                            elements=elements,
                            phases=phases,
                            composition=comp_dict,
                            T_start=T_start_scheil,
                            T_end=T_end_scheil,
                            dT=5.0,
                            use_cache=True
                        )
                        
                        # Extract our phase fraction from Scheil results
                        scheil_frac = float(get_phase_fraction_by_base_name(scheil_phases, phase_to_track))
                        
                        # Compare to equilibrium at room temperature
                        room_temp_eq_frac = frac_start if start_temperature < 400 else None
                        
                        # Convert all phase fractions to float for JSON serialization
                        scheil_phases_clean = {str(k): float(v) for k, v in scheil_phases.items()}
                        
                        scheil_data = {
                            "liquidus_K": float(T_liquidus),
                            "liquidus_C": float(T_liquidus - 273.15),
                            "solidus_K": float(T_solidus) if T_solidus else None,
                            "solidus_C": float(T_solidus - 273.15) if T_solidus else None,
                            "phase_fraction_as_cast": scheil_frac,
                            "difference_vs_equilibrium": float(scheil_frac - room_temp_eq_frac) if room_temp_eq_frac is not None else None,
                            "all_phases": scheil_phases_clean,
                            "note": "Scheil-Gulliver predicts non-equilibrium solidification behavior (as-cast microstructure before heat treatment)"
                        }
                    else:
                        scheil_data = {
                            "error": "Could not determine liquidus temperature for Scheil-Gulliver analysis"
                        }
                        
                except Exception as e:
                    _log.warning(f"Scheil-Gulliver analysis failed: {e}")
                    scheil_data = {
                        "error": f"Scheil-Gulliver analysis unavailable: {str(e)}"
                    }
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Build clean JSON-serializable output
            result_data = {
                "phase": phase_to_track,
                "composition": comp_str,
                "system": "-".join(elements),
                "temperature_range": {
                    "start_K": float(start_temperature),
                    "end_K": float(end_temperature),
                    "start_C": float(start_temperature - 273.15),
                    "end_C": float(end_temperature - 273.15)
                },
                "equilibrium_analysis": {
                    "fraction_at_start_T": frac_start,
                    "fraction_at_end_T": frac_end,
                    "max_fraction": max_frac,
                    "min_fraction": min_frac,
                    "mean_fraction": mean_frac,
                    "fraction_change": delta,
                    "nonzero_count": int(nonzero_count),
                    "total_points": int(len(temps))
                },
                "trend": {
                    "direction": trend,
                    "description": trend_desc,
                    "interpretation": interpretation
                },
                "stability_range": {
                    "start_K": T_stability_start,
                    "end_K": T_stability_end,
                    "start_C": float(T_stability_start - 273.15) if T_stability_start else None,
                    "end_C": float(T_stability_end - 273.15) if T_stability_end else None
                } if T_stability_start else None,
                "is_negligible": is_negligible,
                "scheil_analysis": scheil_data
            }
            
            return success_result(
                handler="calphad",
                function="analyze_phase_fraction_trend",
                data=result_data,
                citations=["pycalphad"],
                confidence=Confidence.HIGH if nonzero_count >= 10 else Confidence.MEDIUM,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error analyzing phase fraction trend")
            return error_result(
                handler="calphad",
                function="analyze_phase_fraction_trend",
                error=f"Failed to analyze phase fraction trend: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )

