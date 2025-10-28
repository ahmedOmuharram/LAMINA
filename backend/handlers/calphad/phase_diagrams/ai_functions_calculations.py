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

from pycalphad import Database, equilibrium
import pycalphad.variables as v
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .database_utils import map_phase_name
from ...shared.calphad_utils import (
    extract_phase_fractions_from_equilibrium,
    get_phase_fraction_by_base_name,
    load_tdb_database
)

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
        
        Args:
            composition: Element-number pairs (e.g., 'Al30Si55C15'). Numbers are percentages.
            composition_type: 'atomic' (default) or 'weight'. Weight% is converted to mole fractions internally.
        
        Returns:
            Formatted text with phase fractions and per-phase compositions.
        """
        try:
            # Parse composition string (e.g., "Al30Si55C15" -> {AL: 0.30, SI: 0.55, C: 0.15})
            # Note: Always returns atomic (mole) fractions, converting from weight if needed
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                return f"Failed to parse composition: {composition}. Use format like 'Al30Si55C15' or 'Fe70Cr20Ni10'"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                return f"No thermodynamic database found for {system_str} system."
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Build conditions
            elements_with_va = elements + ['VA']
            conditions = {v.T: temperature, v.P: 101325, v.N: 1}
            
            # Set composition conditions (N-1 independent compositions)
            for i, elem in enumerate(elements[1:], 1):
                conditions[v.X(elem)] = comp_dict[elem]
            
            # Calculate equilibrium
            eq = equilibrium(db, elements_with_va, phases, conditions)
            
            # Extract phase fractions properly (handling multiple vertices in two-phase regions)
            # Use looser tolerance (1e-4) for better boundary handling
            phase_fractions_dict = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
            
            # Extract phase fractions and compositions
            phase_info = []
            total_fraction = 0.0
            
            for phase, frac in phase_fractions_dict.items():
                total_fraction += frac
                
                # Get composition of this phase
                phase_comp = {}
                try:
                    # Squeeze and select data for this phase
                    eqp = eq.squeeze()
                    phase_mask = eqp['Phase'] == phase
                    
                    for elem in elements:
                        # Extract phase composition for this element (average over vertices)
                        x_data = eqp['X'].sel(component=elem).where(phase_mask, drop=False)
                        x_val = float(x_data.mean().values)
                        if not np.isnan(x_val):
                            phase_comp[elem] = x_val
                except Exception as e:
                    _log.warning(f"Could not extract composition for phase {phase}: {e}")
                
                # Map phase name to readable form (e.g., CSI -> SiC)
                readable_name = map_phase_name(phase)
                
                phase_info.append({
                    'phase': readable_name,
                    'fraction': frac,
                    'composition': phase_comp
                })
            
            # Sort by fraction (descending)
            phase_info.sort(key=lambda x: x['fraction'], reverse=True)
            
            # Format response
            comp_str = " ".join([f"{elem}{comp_dict[elem]*100:.1f}" for elem in elements])
            response_lines = [
                f"**Equilibrium at {temperature:.1f} K for {comp_str}**\n",
                f"**Temperature**: {temperature:.1f} K ({temperature-273.15:.1f} °C)",
                f"**Composition**: {comp_str} (atomic %)\n",
                "**Stable Phases**:"
            ]
            
            for pinfo in phase_info:
                phase_name = pinfo['phase']
                frac = pinfo['fraction']
                comp = pinfo['composition']
                
                comp_str_phase = ", ".join([f"{e}: {comp[e]*100:.2f}%" for e in elements if e in comp])
                response_lines.append(f"  • **{phase_name}**: {frac*100:.2f}% ({comp_str_phase})")
            
            if not phase_info:
                response_lines.append("  • No stable phases found (calculation may have failed)")
            
            response_lines.append(f"\n**Total phase fraction**: {total_fraction*100:.2f}%")
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error calculating equilibrium at {temperature}K for {composition}")
            return {"success": False, "error": f"Failed to calculate equilibrium: {str(e)}", "citations": ["pycalphad"]}
    
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
        try:
            # Parse composition (always returns atomic/mole fractions, converting from weight if needed)
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                return f"No thermodynamic database found for {system_str} system."
            
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
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error calculating phase fractions vs temperature")
            return {"success": False, "error": f"Failed to calculate phase fractions vs temperature: {str(e)}", "citations": ["pycalphad"]}
    
    @ai_function(desc="Analyze whether a specific phase increases or decreases with temperature. Use to verify statements about precipitation or dissolution behavior.")
    async def analyze_phase_fraction_trend(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15')")],
        phase_name: Annotated[str, AIParam(desc="Name of the phase to analyze (e.g., 'AL4C3', 'SIC', 'FCC_A1')")],
        min_temperature: Annotated[float, AIParam(desc="Minimum temperature in Kelvin")],
        max_temperature: Annotated[float, AIParam(desc="Maximum temperature in Kelvin")],
        expected_trend: Annotated[Optional[str], AIParam(desc="Expected trend: 'increase', 'decrease', or 'stable'. Optional.")] = None
    ) -> str:
        """
        Analyze the trend of a specific phase fraction with temperature.
        
        This tool is designed to verify statements like:
        - "Phase X increases with decreasing temperature"
        - "Phase Y precipitates upon cooling"
        - "Phase Z dissolves upon heating"
        
        Returns detailed analysis with verification of expected trends.
        """
        try:
            # Parse composition (note: expected_trend doesn't have composition_type, default to atomic)
            # Always returns atomic/mole fractions
            comp_dict = self._parse_multicomponent_composition(composition, composition_type="atomic")
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db = load_tdb_database(elements)
            if db is None:
                return f"No thermodynamic database found for {system_str} system."
            
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
                return f"Phase '{phase_name}' not found in database. Available phases: {available}"
            
            phase_to_track = phase_name_upper if phase_name_upper in phases else phase_name
            
            # Calculate over temperature range
            temps = np.linspace(min_temperature, max_temperature, 50)
            fractions = []
            
            elements_with_va = elements + ['VA']
            
            for T in temps:
                conditions = {v.T: T, v.P: 101325, v.N: 1}
                for i, elem in enumerate(elements[1:], 1):
                    conditions[v.X(elem)] = comp_dict[elem]
                
                try:
                    eq = equilibrium(db, elements_with_va, phases, conditions)
                    
                    # Extract phase fractions properly (handling multiple vertices)
                    # Use looser tolerance (1e-4) for better boundary handling
                    temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    # Find our phase, summing all instances (e.g., SIC#1 + SIC#2)
                    phase_frac = get_phase_fraction_by_base_name(temp_phases, phase_to_track)
                    
                    # Debug: Log phase fractions at a few sample temperatures
                    if len(fractions) < 3 or len(fractions) == len(temps) // 2:
                        # Collapse instances to base names for debugging
                        base_phases = {}
                        for k, v in temp_phases.items():
                            base_name = str(k).split('#')[0].upper()
                            base_phases[base_name] = base_phases.get(base_name, 0.0) + v
                        top_phases = sorted(base_phases.items(), key=lambda x: x[1], reverse=True)[:5]
                        _log.info(f"At {T:.0f}K: top phases = {top_phases}, {phase_to_track} fraction = {phase_frac:.4f}")
                    
                    fractions.append(phase_frac)
                    
                except Exception as e:
                    _log.warning(f"Calculation failed at {T}K: {e}")
                    fractions.append(0.0)
            
            # Analyze trend
            fractions = np.array(fractions)
            frac_low_T = fractions[0]
            frac_high_T = fractions[-1]
            
            max_frac = np.max(fractions)
            min_frac = np.min(fractions)
            
            # Compute overall trend
            delta = frac_high_T - frac_low_T
            
            if abs(delta) < 0.001:
                trend = "stable"
                trend_desc = "remains relatively stable"
            elif delta > 0.001:
                trend = "increases"
                trend_desc = "increases with increasing temperature (decreases upon cooling)"
            else:
                trend = "decreases"
                trend_desc = "decreases with increasing temperature (increases upon cooling)"
            
            comp_str = "".join([f"{elem}{comp_dict[elem]*100:.0f}" for elem in elements])
            
            response_lines = [
                f"**Phase Fraction Analysis: {phase_to_track} in {comp_str}**\n",
                f"**Temperature Range**: {min_temperature:.0f} - {max_temperature:.0f} K ({min_temperature-273.15:.0f} - {max_temperature-273.15:.0f} °C)",
                f"**Phase**: {phase_to_track}",
                f"**Composition**: {comp_str} (atomic %)\n",
                f"**Results**:",
                f"  • Fraction at {min_temperature:.0f} K: {frac_low_T*100:.3f}%",
                f"  • Fraction at {max_temperature:.0f} K: {frac_high_T*100:.3f}%",
                f"  • Change: {delta*100:.3f}% ({'+' if delta > 0 else ''}{delta*100:.3f}%)",
                f"  • Maximum fraction: {max_frac*100:.3f}%",
                f"  • Minimum fraction: {min_frac*100:.3f}%\n",
                f"**Trend**: The phase fraction **{trend_desc}**."
            ]
            
            # Verify against expected trend if provided
            if expected_trend:
                expected_lower = expected_trend.lower()
                matches = False
                
                # Check for "increasing/decreasing with temperature" patterns
                if "decreasing temperature" in expected_lower or "upon cooling" in expected_lower or "with cooling" in expected_lower:
                    # Phase should be higher at LOW temperature (precipitation upon cooling)
                    if "increase" in expected_lower:
                        matches = (frac_low_T > frac_high_T + 0.001)
                    elif "decrease" in expected_lower:
                        matches = (frac_low_T < frac_high_T - 0.001)
                        
                elif "increasing temperature" in expected_lower or "upon heating" in expected_lower or "with heating" in expected_lower:
                    # Phase should be higher at HIGH temperature
                    if "increase" in expected_lower:
                        matches = (frac_high_T > frac_low_T + 0.001)
                    elif "decrease" in expected_lower:
                        matches = (frac_high_T < frac_low_T - 0.001)
                        
                # Simple increase/decrease without temperature reference
                elif "increase" in expected_lower:
                    matches = (trend == "increases")
                elif "decrease" in expected_lower:
                    matches = (trend == "decreases")
                elif "stable" in expected_lower or "constant" in expected_lower:
                    matches = (trend == "stable")
                
                if matches:
                    response_lines.append(f"\n✅ **Verification**: The expected trend ('{expected_trend}') **matches** the calculated behavior.")
                else:
                    response_lines.append(f"\n❌ **Verification**: The expected trend ('{expected_trend}') **does NOT match** the calculated behavior.")
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error analyzing phase fraction trend")
            return {"success": False, "error": f"Failed to analyze phase fraction trend: {str(e)}", "citations": ["pycalphad"]}

