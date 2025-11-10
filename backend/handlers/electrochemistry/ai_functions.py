"""
AI Functions for Battery and Electrochemistry

This module contains all AI-accessible functions for battery electrode analysis,
voltage calculations, and electrochemical properties.
"""
import logging
import time
from typing import Optional, List, Dict, Any

from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from ..shared import success_result, error_result, ErrorType, Confidence
from . import utils
from ..shared.constants import KNOWN_DIFFUSION_BARRIERS, STRUCTURE_DIFFUSION_DEFAULTS
from .ion_hopping_utils import (
    is_graphitic,
    graphite_scenarios,
    expand_variants,
    evaluate_ion_hopping_scenario,
)

_log = logging.getLogger(__name__)


def _classify_electrode_structure(host: str, user_input: Optional[str]) -> str:
    """Classify electrode structure type from formula or user hint."""
    if user_input:
        s = user_input.lower().strip()
        if "layer" in s or "2d" in s:
            return "layered"
        if "1d" in s or "channel" in s:
            return "1D-channel"
        if "3d" in s or "frame" in s or "spinel" in s:
            return "3D"
        if "olivine" in s or "lfp" in s:
            return "olivine"
    
    # Pattern matching on known materials
    h_lower = host.lower()
    
    # Graphite / layered carbons
    if "c" in h_lower and ("6" in h_lower or "graphite" in h_lower):
        return "layered"
    
    # Common layered materials (TMDs, LCO, NCA, etc.)
    if any(x in h_lower for x in ("coo", "nio", "mno2", "tis2", "mos2", "ws2", "nca", "nmc")):
        return "layered"
    
    # Olivines (LiFePO4, etc.)
    if "po4" in h_lower or "fep" in h_lower:
        return "olivine"
    
    # 1D channels (some titanates, vanadates)
    if any(x in h_lower for x in ("tio2", "v2o5", "vo2")):
        return "1D-channel"
    
    # 3D frameworks (spinels, garnets)
    if any(x in h_lower for x in ("mn2o4", "ti4o8", "lto", "llzo")):
        return "3D"
    
    return "unknown"


def _estimate_barrier_from_structure(host: str, ion: str, struct_type: str) -> Dict[str, Any]:
    """Return barrier estimate based on structure class and known literature values."""
    
    # 1) Exact formula match first (e.g., 'C6', 'C12')
    for (mat, i), info in KNOWN_DIFFUSION_BARRIERS.items():
        if ion == i and host.strip().lower() == mat.lower():
            d = {"literature_value": True, "note": info["note"]}
            d.update(info.get("descriptors", {}))
            return {
                "Ea_eV": info["Ea"],
                "range_eV": info["range"],
                "confidence": "high",
                "descriptors": d,
                "citations": info.get("citations", ["Literature benchmark values"]),
            }
    
    # 2) Relaxed contains match (e.g., 'graphite', 'LiC6 (graphite)')
    for (mat, i), info in KNOWN_DIFFUSION_BARRIERS.items():
        if ion == i and mat.lower() in host.lower():
            d = {"literature_value": True, "note": info["note"]}
            d.update(info.get("descriptors", {}))
            return {
                "Ea_eV": info["Ea"],
                "range_eV": info["range"],
                "confidence": "medium" if mat.lower() == "graphite" else "high",
                "descriptors": d,
                "citations": info.get("citations", ["Literature benchmark values"]),
            }
    
    # Structure-based estimates (generic)
    defaults = STRUCTURE_DIFFUSION_DEFAULTS.get(struct_type, STRUCTURE_DIFFUSION_DEFAULTS["unknown"])
    
    return {
        "Ea_eV": defaults["Ea"],
        "range_eV": defaults["range"],
        "confidence": defaults["confidence"],
        "descriptors": {
            "structure_type": struct_type,
            "note": defaults["note"],
        },
        "citations": ["Structure-based heuristics from battery literature"],
    }


class BatteryAIFunctionsMixin:
    """Mixin class containing AI function methods for BatteryHandler."""

    @ai_function(
        desc="Search for battery electrode materials and their voltage profiles. "
             "Use this for questions about battery voltages, electrode materials, "
             "capacity, and electrochemical performance (e.g., 'AlMg anode voltage', "
             "'lithium battery with aluminum anode', 'voltage vs Li/Li+')."
    )
    async def search_battery_electrodes(
        self,
        formula: Annotated[
            Optional[str],
            AIParam(desc="Chemical formula of electrode material (e.g., 'AlMg', 'Al2Mg3', 'LiCoO2')")
        ] = None,
        elements: Annotated[
            Optional[str],
            AIParam(desc="Comma-separated elements in electrode (e.g., 'Al,Mg' or 'Li,Co,O')")
        ] = None,
        working_ion: Annotated[
            Optional[str],
            AIParam(desc="Working ion for the battery (e.g., 'Li', 'Na', 'Mg'). Default: 'Li'")
        ] = "Li",
        min_capacity: Annotated[
            Optional[float],
            AIParam(desc="Minimum gravimetric capacity in mAh/g")
        ] = None,
        max_capacity: Annotated[
            Optional[float],
            AIParam(desc="Maximum gravimetric capacity in mAh/g")
        ] = None,
        min_voltage: Annotated[
            Optional[float],
            AIParam(desc="Minimum average voltage vs working ion in V")
        ] = None,
        max_voltage: Annotated[
            Optional[float],
            AIParam(desc="Maximum average voltage vs working ion in V")
        ] = None,
        max_entries: Annotated[
            Optional[int],
            AIParam(desc="Maximum number of results to return. Default: 10")
        ] = 10
    ) -> Dict[str, Any]:
        """
        Search for battery electrode materials using Materials Project's electrodes database.
        
        Returns voltage profiles, capacities, and electrochemical properties.
        """
        start_time = time.time()
        
        try:
            if not self.mpr:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="electrochemistry",
                    function="search_battery_electrodes",
                    error="MPRester client not initialized",
                    error_type=ErrorType.API_ERROR,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )
            
            # Check if insertion_electrodes endpoint is available
            if not hasattr(self.mpr, 'insertion_electrodes'):
                return await self._fallback_voltage_calculation(
                    formula=formula,
                    elements=elements,
                    working_ion=working_ion
                )
            
            # Build query parameters using utility function
            query_params = utils.build_electrode_query_params(
                formula, elements, working_ion,
                min_capacity, max_capacity, min_voltage, max_voltage
                )
            
            _log.info(f"Searching insertion_electrodes with params: {query_params}")
            
            # Query the insertion_electrodes endpoint
            try:
                results = self.mpr.insertion_electrodes.search(**query_params)
                
                # Limit results
                if results and len(results) > max_entries:
                    results = results[:max_entries]
                
                # Process electrode documents using utility function
                electrode_data = utils.process_electrode_documents(results, working_ion)
                
                # Post-filter by framework composition if elements were specified
                # This is CRITICAL because MP's query may use OR logic
                if elements and electrode_data:
                    el_list = {e.strip() for e in elements.split(",") if e.strip()}
                    original_count = len(electrode_data)
                    electrode_data = utils.filter_electrodes_by_framework(electrode_data, el_list)
                    filtered_count = original_count - len(electrode_data)
                    if filtered_count > 0:
                        _log.info(f"Post-filtered {filtered_count} electrodes with frameworks outside {el_list}")
                
                # If curated DB returns nothing (or all filtered out), fall back to convex-hull computation
                if not electrode_data and (formula or elements):
                    host_formula = formula
                    if not host_formula and elements:
                        el_list = [e.strip() for e in elements.split(",") if e.strip()]
                        # For binary/ternary, create simple formula like "AlMg"
                        host_formula = "".join(el_list) if el_list else None

                    if host_formula:
                        _log.info(f"No curated electrodes found; computing voltage via convex hull for {host_formula}")
                        synth = utils.compute_alloy_voltage_via_hull(
                            self.mpr, host_formula, working_ion=working_ion
                        )
                        if synth.get("success"):
                            electrode_data = [synth["electrode"]]

                duration_ms = (time.time() - start_time) * 1000
                
                return success_result(
                    handler="electrochemistry",
                    function="search_battery_electrodes",
                    data={
                        "count": len(electrode_data),
                        "electrodes": electrode_data,
                        "query": query_params,
                    },
                    citations=["Materials Project", "pymatgen"],
                    confidence=Confidence.HIGH if electrode_data else Confidence.LOW,
                    notes=[
                        f"Found {len(electrode_data)} electrode materials"
                        + (" (computed from convex hull)" if electrode_data and electrode_data[0].get("source") == "computed_from_phase_diagram" else ""),
                        "Voltages are reported vs. the working ion (e.g., Li/Li+)",
                        "capacity_grav is in mAh/g, energy_grav is in Wh/kg",
                        "Framework compositions verified to match requested elements" if elements else ""
                    ],
                    duration_ms=duration_ms
                )
                
            except AttributeError:
                _log.warning("insertion_electrodes.search not available, trying alternative method")
                return await self._fallback_voltage_calculation(
                    formula=formula,
                    elements=elements,
                    working_ion=working_ion
                )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in search_battery_electrodes: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="search_battery_electrodes",
                error=str(e),
                error_type=ErrorType.API_ERROR,
                citations=["Materials Project", "pymatgen"],
                suggestions=["Try using calculate_voltage_from_formation_energy for custom calculations"],
                duration_ms=duration_ms
            )
    

    @ai_function(
        desc="Calculate electrode voltage from MP DFT entries via convex-hull (InsertionElectrode). "
             "Uses only consistent ComputedEntry data from MP for the full chemical system."
    )
    async def calculate_voltage_from_formation_energy(
        self,
        electrode_formula: Annotated[
            str,
            AIParam(desc="Chemical formula of the electrode material (e.g., 'Al3Mg2', 'AlMg')")
        ],
        working_ion: Annotated[
            str,
            AIParam(desc="Working ion element symbol (e.g., 'Li', 'Na', 'Mg')")
        ] = "Li",
        temperature: Annotated[
            Optional[float],
            AIParam(desc="Temperature in Kelvin (currently unused - 0K hull used). Default: 298.15 K")
        ] = 298.15
    ) -> Dict[str, Any]:
        """
        Calculate electrode voltage from convex hull analysis using PyMatGen InsertionElectrode.
        
        Uses two-phase equilibria on the convex hull to compute physically valid voltage curves.
        All data from a single consistent set of ComputedEntry objects from Materials Project.
        
        Returns error if no suitable framework or if voltage is unphysical.
        """
        start_time = time.time()
        
        try:
            # Try InsertionElectrode calculation first
            result = utils.calculate_voltage_from_insertion_electrode(
                self.mpr, electrode_formula, working_ion
            )
            
            # If InsertionElectrode fails or returns None, fallback to hull scan
            if result is None or not result.get("success"):
                synth = utils.compute_alloy_voltage_via_hull(
                    self.mpr, electrode_formula, working_ion=working_ion
                )
                if synth.get("success"):
                    duration_ms = (time.time() - start_time) * 1000
                    e = synth["electrode"]
                    return success_result(
                        handler="electrochemistry",
                        function="calculate_voltage_from_formation_energy",
                        data={
                            "calculation_method": "phase_diagram_line_scan",
                            "calculated_voltage": e["average_voltage"],
                            "chemical_system": e["diagnostics"]["chemsys"],
                            "framework_formula": e["framework"],
                            "voltage_range": {"min": e["min_voltage"], "max": e["max_voltage"], "average": e["average_voltage"]},
                            "capacity_grav": e["capacity_grav"],
                            "energy_grav": e["energy_grav"],
                        },
                        citations=["Materials Project", "pymatgen"],
                        confidence=Confidence.HIGH,
                        notes=[
                            "Voltages from two-phase convex-hull scan along fixed host ratio (0 K)",
                            f"Reported vs. {working_ion}/{working_ion}+; consistent entry set"
                        ],
                        diagnostics=e.get("diagnostics", {}),
                        duration_ms=duration_ms
                    )
                # If both methods fail, return the original error
                duration_ms = (time.time() - start_time) * 1000
                if result:
                    return error_result(
                        handler="electrochemistry",
                        function="calculate_voltage_from_formation_energy",
                        error=result.get("error", "Voltage calculation failed"),
                        error_type=ErrorType.COMPUTATION_ERROR,
                        citations=["Materials Project", "pymatgen"],
                        duration_ms=duration_ms
                    )
                return error_result(
                    handler="electrochemistry",
                    function="calculate_voltage_from_formation_energy",
                    error="Voltage calculation failed",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["Materials Project", "pymatgen"],
                    duration_ms=duration_ms
                )
            
            # Wrap the successful InsertionElectrode result
            duration_ms = (time.time() - start_time) * 1000
            if result.get("success"):
                data = {k: v for k, v in result.items() if k not in ["success", "citations", "notes"]}
                return success_result(
                    handler="electrochemistry",
                    function="calculate_voltage_from_formation_energy",
                    data=data,
                    citations=result.get("citations", ["Materials Project", "pymatgen"]),
                    confidence=Confidence.HIGH,
                    notes=result.get("notes", []),
                    duration_ms=duration_ms
                )
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Voltage calculation failed: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="calculate_voltage_from_formation_energy",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
    

    @ai_function(
        desc="Get detailed voltage profile and phase evolution data for a specific electrode material. "
             "Shows how voltage changes during charge/discharge cycles."
    )
    async def get_voltage_profile(
        self,
        material_id: Annotated[
            str,
            AIParam(desc="Materials Project ID of the electrode (e.g., 'mp-12345') or battery ID")
        ],
        working_ion: Annotated[
            str,
            AIParam(desc="Working ion element symbol (e.g., 'Li', 'Na'). Default: 'Li'")
        ] = "Li"
    ) -> Dict[str, Any]:
        """
        Get detailed voltage profile for a specific electrode material.
        
        Returns the full charge/discharge curve showing voltage vs. capacity.
        """
        start_time = time.time()
        
        try:
            if not self.mpr:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="electrochemistry",
                    function="get_voltage_profile",
                    error="MPRester client not initialized",
                    error_type=ErrorType.API_ERROR,
                    citations=["Materials Project"],
                    duration_ms=duration_ms
                )
            
            # Try to get electrode data
            if hasattr(self.mpr, 'insertion_electrodes'):
                try:
                    # Search by material_id or battery_id
                    results = self.mpr.insertion_electrodes.search(battery_id=material_id)
                    
                    if not results:
                        # Try as a formula
                        results = self.mpr.insertion_electrodes.search(formula=material_id)
                    
                    if results:
                        duration_ms = (time.time() - start_time) * 1000
                        electrode = results[0]
                        
                        # Extract voltage profile using utility function
                        profile_data = utils.extract_voltage_profile(electrode)
                        return success_result(
                            handler="electrochemistry",
                            function="get_voltage_profile",
                            data={
                                **profile_data,
                                "material_id": material_id
                            },
                            citations=["Materials Project", "pymatgen"],
                            confidence=Confidence.HIGH,
                            notes=["Full voltage profile extracted from electrode database"],
                            duration_ms=duration_ms
                        )
                        
                except Exception as e:
                    _log.warning(f"Error getting voltage profile: {e}")
            
            # Fallback: just return summary data
            mat_data = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["material_id", "formula_pretty", "formation_energy_per_atom", 
                       "energy_above_hull"]
            )
            
            if mat_data:
                duration_ms = (time.time() - start_time) * 1000
                mat = mat_data[0]
                return success_result(
                    handler="electrochemistry",
                    function="get_voltage_profile",
                    data={
                        "material_id": str(mat.material_id),
                        "formula": mat.formula_pretty,
                        "formation_energy_per_atom": mat.formation_energy_per_atom,
                        "energy_above_hull": mat.energy_above_hull,
                    },
                    citations=["Materials Project"],
                    confidence=Confidence.LOW,
                    notes=[
                        "Detailed voltage profile not available",
                        "Use search_battery_electrodes to find electrode materials with profiles",
                        "Or use calculate_voltage_from_formation_energy for estimates"
                    ],
                    duration_ms=duration_ms
                )
            
            duration_ms = (time.time() - start_time) * 1000
            return error_result(
                handler="electrochemistry",
                function="get_voltage_profile",
                error=f"Material {material_id} not found",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in get_voltage_profile: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="get_voltage_profile",
                error=str(e),
                error_type=ErrorType.API_ERROR,
                citations=["Materials Project"],
                duration_ms=duration_ms
            )
    

    @ai_function(
        desc="Compare multiple electrode materials for battery applications. "
             "USE THIS for questions like 'does X increase/decrease voltage vs Y', "
             "'compare Al vs AlMg', 'which is better: A or B', 'does alloying affect voltage'. "
             "Returns side-by-side voltages, capacities, and energy densities."
    )
    async def compare_electrode_materials(
        self,
        formulas: Annotated[
            str,
            AIParam(desc="Comma-separated list of chemical formulas to compare (e.g., 'Al,AlMg' or 'LiCoO2,LiFePO4')")
        ],
        working_ion: Annotated[
            str,
            AIParam(desc="Working ion for comparison (e.g., 'Li', 'Na'). Default: 'Li'")
        ] = "Li"
    ) -> Dict[str, Any]:
        """
        Compare multiple electrode materials side-by-side.
        
        Returns voltages, capacities, and energy densities for direct comparison.
        Automatically calculates which material has higher/lower voltage.
        """
        start_time = time.time()
        
        try:
            formula_list = [f.strip() for f in formulas.split(',')]
            
            comparison_results = []
            
            for formula in formula_list:
                # Try electrodes search first
                electrode_result = await self.search_battery_electrodes(
                    formula=formula,
                    working_ion=working_ion,
                    max_entries=1
                )
                
                if electrode_result.get("success") and electrode_result.get("data", {}).get("electrodes"):
                    comparison_results.append({
                        "formula": formula,
                        "data": electrode_result["data"]["electrodes"][0],
                        "source": "electrodes_database"
                    })
                else:
                    # Fallback to formation energy calculation
                    calc_result = await self.calculate_voltage_from_formation_energy(
                        electrode_formula=formula,
                        working_ion=working_ion
                    )
                    
                    if calc_result.get("success"):
                        data = calc_result.get("data", {})
                        comparison_results.append({
                            "formula": formula,
                            "data": {
                                "voltage": data.get("calculated_voltage"),
                                "material_id": data.get("electrode_material", {}).get("material_id"),
                                "formation_energy": data.get("electrode_material", {}).get("formation_energy_per_atom")
                            },
                            "source": "calculated_from_formation_energy"
                        })
                    else:
                        comparison_results.append({
                            "formula": formula,
                            "error": calc_result.get("error"),
                            "source": "failed"
                        })
            
            # Generate comparison summary using utility function
            summary = utils.generate_comparison_summary(comparison_results, working_ion)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return success_result(
                handler="electrochemistry",
                function="compare_electrode_materials",
                data={
                    "working_ion": working_ion,
                    "comparison": comparison_results,
                    "count": len(comparison_results),
                    "summary": summary,
                },
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH,
                notes=[
                    "Voltages are vs. working ion reference (e.g., Li/Li+)",
                    "Data from insertion_electrodes database: pre-computed voltage profiles",
                    "Data from convex hull: thermodynamically rigorous phase diagram calculations",
                    "All data is from Materials Project DFT calculations - no heuristics or estimates"
                ],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error comparing electrodes: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="compare_electrode_materials",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
    

    @ai_function(
        desc="Check if a composition is thermodynamically stable (on convex hull). "
             "USE THIS to determine if a material can exist as a stable phase or if it decomposes. "
             "Returns energy above hull (if entry exists) and decomposition products. "
             "Optionally analyzes battery-relevant stability vs voltage (grand-potential phase diagram). "
             "Essential for questions about 'thermodynamically stable', 'can form', 'stable phase', "
             "'stable as anode', 'battery stability'."
    )
    async def check_composition_stability(
        self,
        composition: Annotated[
            str,
            AIParam(desc="Chemical composition to check (e.g., 'Cu8LiAl', 'Li3Al2', 'Cu80Li10Al10')")
        ],
        battery_analysis: Annotated[
            bool,
            AIParam(desc="Include battery-relevant voltage-dependent analysis (grand-potential PD). Default: False")
        ] = False,
        working_ion: Annotated[
            Optional[str],
            AIParam(desc="Working ion for battery analysis (e.g., 'Li', 'Na'). Only used if battery_analysis=True. Default: 'Li'")
        ] = "Li",
        vmin: Annotated[
            Optional[float],
            AIParam(desc="Minimum voltage (V) for voltage scan. Default: 0.0")
        ] = 0.0,
        vmax: Annotated[
            Optional[float],
            AIParam(desc="Maximum voltage (V) for voltage scan. Default: 1.0")
        ] = 1.0,
        voltage_points: Annotated[
            Optional[int],
            AIParam(desc="Number of voltage points to scan. Default: 11")
        ] = 11
    ) -> Dict[str, Any]:
        """
        Check if a composition is thermodynamically stable.
        
        Returns:
        - equilibrium_0K: Standard 0 K convex hull analysis (energy above hull, decomposition)
        - battery_relevant (optional): Voltage-dependent stability with Li as open species
        - summary: Combined human-readable assessment
        
        The battery_analysis option provides actionable insights for electrode materials:
        - Stable phases at each voltage vs Li/Li+
        - Voltage intervals where decomposition is unchanged
        - Whether single-phase or multiphase equilibrium
        - Theoretical capacity estimates
        """
        start_time = time.time()
        
        # Always run 0 K analysis first
        result_0K = utils.check_composition_stability_detailed(self.mpr, composition)
        
        if not result_0K.get("success"):
            duration_ms = (time.time() - start_time) * 1000
            return error_result(
                handler="electrochemistry",
                function="check_composition_stability",
                error=result_0K.get("error", "Stability check failed"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(result_0K.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
        
        # Prepare response data
        data = {
            "equilibrium_0K": {k: v for k, v in result_0K.items() if k not in ["success", "citations"]}
        }
        
        notes = ["0 K convex hull analysis from DFT calculations"]
        
        # Optionally run battery-relevant voltage analysis
        if battery_analysis:
            _log.info(f"Running battery-relevant voltage analysis for {composition}")
            result_battery = utils.check_anode_stability_vs_voltage(
                self.mpr, 
                composition, 
                vmin=vmin, 
                vmax=vmax, 
                npts=voltage_points,
                open_element=working_ion
            )
            
            if result_battery.get("success"):
                data["battery_relevant"] = {k: v for k, v in result_battery.items() if k not in ["success"]}
                notes.append(f"Battery-relevant analysis with {working_ion} as open species")
                
                # Combined summary
                summary_0K = result_0K.get("notes", [""])[0] if result_0K.get("notes") else ""
                summary_battery = result_battery.get("summary", "")
                
                data["combined_summary"] = (
                    f"**0 K Equilibrium:** {summary_0K}\n\n"
                    f"**Battery Context ({working_ion} as open element):** {summary_battery}"
                )
            else:
                _log.warning(f"Battery analysis failed: {result_battery.get('error')}")
                data["battery_relevant_error"] = result_battery.get("error")
                notes.append("Battery analysis requested but failed")
        else:
            notes.append("For battery-relevant voltage analysis, set battery_analysis=True")
        
        duration_ms = (time.time() - start_time) * 1000
        
        return success_result(
            handler="electrochemistry",
            function="check_composition_stability",
            data=data,
            citations=["Materials Project", "pymatgen"],
            confidence=Confidence.HIGH,
            notes=notes,
            duration_ms=duration_ms
        )
    

    @ai_function(
        desc="Analyze a composition as a potential battery anode, including stability check and voltage. "
             "USE THIS for questions about whether a material 'can form an anode', 'is suitable as anode'. "
             "Checks thermodynamic stability (0 K and voltage-dependent) and calculates voltage if viable. "
             "Provides battery-relevant grand-potential analysis automatically."
    )
    async def analyze_anode_viability(
        self,
        composition: Annotated[
            str,
            AIParam(desc="Chemical composition to analyze (e.g., 'Cu8LiAl', 'AlMg', 'Li3Al2')")
        ],
        working_ion: Annotated[
            str,
            AIParam(desc="Working ion for battery (e.g., 'Li', 'Na'). Default: 'Li'")
        ] = "Li",
        voltage_window: Annotated[
            Optional[str],
            AIParam(desc="Voltage window to scan (e.g., '0-1' for 0-1 V). Default: '0-1'")
        ] = "0-1"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a composition as a battery anode.
        
        Automatically includes:
        1. 0 K thermodynamic stability (convex hull)
        2. Battery-relevant voltage-dependent stability (grand-potential phase diagram)
        3. Voltage calculations vs working ion
        4. Decomposition products and phase evolution
        5. Theoretical capacity estimates
        6. Overall viability assessment with actionable insights
        """
        start_time = time.time()
        
        try:
            if not utils.PYMATGEN_AVAILABLE:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="electrochemistry",
                    function="analyze_anode_viability",
                    error="PyMatGen not available",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["Materials Project", "pymatgen"],
                    duration_ms=duration_ms
                )
            
            from pymatgen.core import Composition
            
            # Parse voltage window
            try:
                vmin, vmax = [float(v) for v in voltage_window.split('-')]
            except:
                vmin, vmax = 0.0, 1.0
            
            # Check stability with battery-relevant analysis
            stability = await self.check_composition_stability(
                composition, 
                battery_analysis=True,
                working_ion=working_ion,
                vmin=vmin,
                vmax=vmax,
                voltage_points=11
            )
            
            if not stability.get("success"):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="electrochemistry",
                    function="analyze_anode_viability",
                    error=stability.get("error", "Stability check failed"),
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["Materials Project", "pymatgen"],
                    duration_ms=duration_ms
                )
            
            # Extract stability data
            stability_data = stability.get("data", {})
            equilibrium_0K = stability_data.get("equilibrium_0K", {})
            battery_relevant = stability_data.get("battery_relevant", {})
            combined_summary = stability_data.get("combined_summary", "")
            
            is_stable_0K = equilibrium_0K.get("is_stable", False)
            e_above_hull = equilibrium_0K.get("energy_above_hull")
            decomp_phases = equilibrium_0K.get("decomposition", [])
            
            # Extract battery-relevant data
            on_hull_at_any_V = battery_relevant.get("on_hull_at_any_V", False)
            capacity_estimate = battery_relevant.get("capacity_estimate")
            voltage_intervals = battery_relevant.get("intervals", [])
            
            # Try to get voltage information for host material
            comp = Composition(composition)
            voltage_analysis = {}
            
            if working_ion not in [el.symbol for el in comp.elements]:
                # Composition doesn't contain working ion - this is the host material
                host_formula = composition
                
                # Try to calculate voltage
                voltage_result = await self.search_battery_electrodes(
                    formula=host_formula,
                    working_ion=working_ion,
                    max_entries=1
                )
                
                if voltage_result.get("success") and voltage_result.get("data", {}).get("electrodes"):
                    voltage_analysis = voltage_result["data"]["electrodes"][0]
                else:
                    voltage_analysis = {"note": "No curated voltage data available"}
            else:
                # Composition already contains working ion - it's a lithiated phase
                voltage_analysis = {
                    "note": f"Composition contains {working_ion} - this is a lithiated phase, not a host anode material"
                }
            
            # Enhanced viability assessment using both 0 K and battery-relevant data
            assessment = {
                "can_form_stable_anode": False,
                "anode_type": "unknown",
                "reasoning": [],
                "actionable_insights": []
            }
            
            if on_hull_at_any_V:
                assessment["can_form_stable_anode"] = True
                assessment["anode_type"] = "single-phase"
                assessment["reasoning"].append(
                    f"{composition} is thermodynamically stable as a single phase at operating voltages ({vmin}-{vmax} V vs {working_ion}/{working_ion}+)"
                )
                assessment["actionable_insights"].append("This material can form as a stable single-phase anode")
                
                if capacity_estimate:
                    cap = capacity_estimate.get("theoretical_capacity_mAh_g", 0)
                    assessment["actionable_insights"].append(
                        f"Theoretical capacity: ~{cap:.0f} mAh/g (equilibrium-based upper bound)"
                    )
            
            elif is_stable_0K:
                assessment["can_form_stable_anode"] = True
                assessment["anode_type"] = "single-phase (0 K stable)"
                assessment["reasoning"].append(
                    f"{composition} is stable at 0 K but may form multiphase equilibrium at operating voltages"
                )
                assessment["actionable_insights"].append(
                    "Stable in synthesis conditions; check voltage-dependent behavior for cycling stability"
                )
            
            else:
                # Check if nearly metastable
                if e_above_hull is not None and e_above_hull < 0.03:
                    assessment["can_form_stable_anode"] = True
                    assessment["anode_type"] = "metastable"
                    assessment["reasoning"].append(
                        f"Nearly stable ({e_above_hull:.6f} eV/atom above hull - within synthesizable tolerance)"
                    )
                    assessment["actionable_insights"].append(
                        "May be synthesizable as metastable phase; kinetic stabilization possible"
                    )
                else:
                    assessment["can_form_stable_anode"] = False
                    assessment["anode_type"] = "multiphase composite"
                    
                    if decomp_phases:
                        decomp_formulas = [p["formula"] for p in decomp_phases]
                        assessment["reasoning"].append(
                            f"Not thermodynamically stable - decomposes into {len(decomp_phases)} phases: {', '.join(decomp_formulas)}"
                        )
                    
                    # Use battery-relevant data for actionable insights
                    if voltage_intervals:
                        first_interval = voltage_intervals[0]
                        phases = [p["formula"] for p in first_interval.get("phases", [])]
                        
                        # Check for inactive backbone elements
                        inactive_elements = ["Cu", "Ni", "Fe", "Co"]
                        has_inactive = any(any(el in phase for el in inactive_elements) for phase in phases)
                        
                        if has_inactive:
                            assessment["actionable_insights"].append(
                                f"Equilibrium at {vmin:.1f}-{vmax:.1f} V: {' + '.join(phases)}"
                            )
                            assessment["actionable_insights"].append(
                                "Forms multiphase composite with inactive backbone (e.g., Cu) - not a single-phase anode but may work as engineered composite"
                            )
                        else:
                            assessment["actionable_insights"].append(
                                f"Equilibrium phases: {' + '.join(phases)}"
                            )
                            assessment["actionable_insights"].append(
                                "Multiphase composite; performance depends on phase connectivity and Li transport"
                            )
                        
                        if capacity_estimate:
                            cap = capacity_estimate.get("theoretical_capacity_mAh_g", 0)
                            assessment["actionable_insights"].append(
                                f"Composite theoretical capacity: ~{cap:.0f} mAh/g"
                            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return success_result(
                handler="electrochemistry",
                function="analyze_anode_viability",
                data={
                    "composition": composition,
                    "working_ion": working_ion,
                    "voltage_window": f"{vmin:.1f}-{vmax:.1f} V",
                    "stability_0K": equilibrium_0K,
                    "battery_relevant": battery_relevant,
                    "voltage_analysis": voltage_analysis,
                    "viability_assessment": assessment,
                    "summary": combined_summary if combined_summary else battery_relevant.get("summary", "")
                },
                citations=["Materials Project", "pymatgen"],
                confidence=Confidence.HIGH,
                notes=[
                    "Comprehensive anode viability analysis with 0 K and voltage-dependent stability",
                    "Battery-relevant grand-potential analysis included automatically",
                    "Actionable insights based on equilibrium phase behavior at operating voltages"
                ],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error analyzing anode viability: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="analyze_anode_viability",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
    

    @ai_function(
        desc="Analyze the lithiation mechanism of a host material. "
             "Reports phase evolution, two-phase vs single-phase reactions, and equilibrium phases at each voltage step. "
             "Includes both strict 0K classification and effective two-phase classification for near-degenerate cases. "
             "USE THIS for questions about 'two-phase reaction', 'lithiation mechanism', 'phase evolution', "
             "'what phases form', 'initial reaction'."
    )
    async def analyze_lithiation_mechanism(
        self,
        host_composition: Annotated[
            str,
            AIParam(desc="Host material composition (e.g., 'AlCu', 'CuAl', 'Al', 'Mg'). Do NOT include Li.")
        ],
        working_ion: Annotated[
            str,
            AIParam(desc="Working ion for battery (e.g., 'Li', 'Na'). Default: 'Li'")
        ] = "Li",
        max_x: Annotated[
            float,
            AIParam(desc="Maximum Li per host atom to analyze. Default: 3.0")
        ] = 3.0,
        room_temp: Annotated[
            bool,
            AIParam(desc="Filter out phases hard to form at room temperature (E_hull > 0.03 eV/atom). Default: True")
        ] = True,
        enforce_ground_states: Annotated[
            bool,
            AIParam(desc="Enforce ground-state phases for Al, Cu, Li, LiAl. Default: True")
        ] = True,
        prefer_two_phase: Annotated[
            bool,
            AIParam(desc="Attempt effective two-phase classification for near-degenerate 3-phase plateaus. Default: True")
        ] = True,
        two_phase_energy_tolerance: Annotated[
            float,
            AIParam(desc="Energy tolerance (eV/atom) for effective two-phase override. Default: 0.05")
        ] = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze the lithiation mechanism by computing the convex hull of G(x) vs x,
        with optional effective two-phase reclassification.
        
        Reports:
        - Voltage plateaus based on hull segments
        - Equilibrium phases from decompositions (strict and effective classifications)
        - Whether reactions are two-phase, three-phase, or effectively two-phase
        - Initial reaction mechanism (strict vs effective)
        - Full lithiation sequence with dual classification
        
        Args:
            room_temp: If True, filter out phases with E_hull > 0.03 eV/atom (hard to form at RT)
            enforce_ground_states: Encourage correct ground states for Al, Cu, Li, LiAl
            prefer_two_phase: Attempt effective two-phase classification when energetically close
            two_phase_energy_tolerance: Energy tolerance for effective override (eV/atom)
        """
        start_time = time.time()
        
        result = utils.analyze_lithiation_mechanism_detailed(
            self.mpr, host_composition, working_ion, max_x, room_temp,
            enforce_ground_states, prefer_two_phase, two_phase_energy_tolerance
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not result.get("success"):
            return error_result(
                handler="electrochemistry",
                function="analyze_lithiation_mechanism",
                error=result.get("error", "Lithiation mechanism analysis failed"),
                error_type=ErrorType.NOT_FOUND if "not found" in str(result.get("error", "")).lower() else ErrorType.COMPUTATION_ERROR,
                citations=["Materials Project", "pymatgen"],
                duration_ms=duration_ms
            )
        
        data = {k: v for k, v in result.items() if k not in ["success", "citations"]}
        return success_result(
            handler="electrochemistry",
            function="analyze_lithiation_mechanism",
            data=data,
            citations=["Materials Project", "pymatgen"],
            confidence=Confidence.HIGH,
            notes=[
                "Lithiation mechanism computed from convex hull analysis",
                "Includes both strict 0K classification and effective two-phase analysis",
                "Ground-state enforcement ensures reliable phase diagram construction"
            ],
            duration_ms=duration_ms
        )

    @ai_function(
        desc=(
            "Comprehensive ion hopping barrier estimation across many physically distinct scenarios "
            "(AB-BLG TH↔TH ~0.07 eV, AA-BLG ~0.34 eV, stage-I/II graphite, intra- vs inter-layer, "
            "dilute vs finite coverage, defect-assisted). Returns ranked scenarios with P10/P50/P90 "
            "barriers plus Arrhenius kinetics per scenario. Use this for thorough analysis of "
            "graphite/graphene systems or when you need uncertainty quantification and multiple pathways."
        ),
        auto_truncate=128000,
    )
    async def estimate_ion_hopping_barrier(
        self,
        host_material: Annotated[str, AIParam(desc="Host (e.g., 'graphite', 'C6', 'LiC6', 'TiS2', 'LiFePO4').")],
        ion: Annotated[str, AIParam(desc="Ion (Li, Na, Mg).")] = "Li",
        structure_type: Annotated[Optional[str], AIParam(desc="layered, 1D-channel, 3D, olivine (optional; default is layered).")] = "layered",
        temperatures_K: Annotated[Optional[str], AIParam(desc="Comma-separated temps, e.g. '298,323'. Default '300'.")] = "300",
        mc_samples: Annotated[int, AIParam(desc="Samples per scenario for P10/P50/P90.")] = 200,
        include_generic_variants: Annotated[bool, AIParam(desc="Also add generic structure-type scenarios.")] = True,
        return_top: Annotated[int, AIParam(desc="How many scenarios to return (sorted by median Ea).")] = 12,
    ) -> Dict[str, Any]:
        """
        Comprehensive suite of ion hopping barrier estimates with uncertainty quantification.
        
        Enumerates many physically distinct cases:
        - For graphite/graphene: AB-BLG, AA-BLG, stage-I/II, in-gallery, cross-plane, defects
        - Modifiers for stacking, coverage, defects, strain, interlayer spacing
        - Monte Carlo uncertainty → P10/P50/P90 barriers
        - Arrhenius kinetics and diffusion coefficients at specified temperatures
        - Falls back to structure heuristics for non-graphite hosts
        """
        t0 = time.time()
        
        try:
            ion_sym = ion.strip().capitalize()
            host = host_material.strip()
            Ts = [float(t.strip()) for t in str(temperatures_K).split(",") if t.strip()] or [300.0]
            
            scenarios: List[Dict[str, Any]] = []
            all_citations = []
            
            # 1) Graphitic rich expansion
            if is_graphitic(host):
                base = graphite_scenarios(host)
                for b in base:
                    # Collect citations from base scenarios
                    if "citations" in b:
                        all_citations.extend(b["citations"])
                    scenarios.extend(expand_variants(b))
            
            # 2) Optional generic variants for non-graphite (or in addition)
            struct_type_norm = _classify_electrode_structure(host, structure_type)
            
            if include_generic_variants:
                # Build a couple of generic path families per structure class
                gen = _estimate_barrier_from_structure(host, ion_sym, struct_type_norm)
                base_Ea = float(gen["Ea_eV"])
                lo, hi = gen["range_eV"]
                hopA = 3.0 if struct_type_norm != "1D-channel" else 3.5
                
                # Add generic citations
                if "citations" in gen:
                    all_citations.extend(gen["citations"])
                
                generic_families = [
                    {
                        "key": "generic_easy",
                        "baseline_ea": max(0.01, 0.8 * base_Ea),
                        "spread": 1.3,
                        "hop_A": hopA,
                        "note": f"{struct_type_norm} easy path",
                        "citations": gen.get("citations", [])
                    },
                    {
                        "key": "generic_bottleneck",
                        "baseline_ea": min(1.5, 1.2 * base_Ea),
                        "spread": 1.4,
                        "hop_A": hopA * 0.9,
                        "note": f"{struct_type_norm} bottleneck",
                        "citations": gen.get("citations", [])
                    },
                ]
                
                for g in generic_families:
                    scenarios.extend(expand_variants({
                        "key": g["key"],
                        "stacking": "n/a",
                        "defect": "none",
                        "theta": 0.1,
                        "strain_percent": 0.0,
                        "delta_interlayer_A": 0.0,
                        "baseline_ea": g["baseline_ea"],
                        "hop_A": g["hop_A"],
                        "spread": g["spread"],
                        "note": g["note"],
                    }))
            
            # 3) Evaluate all scenarios with Monte Carlo + modifiers
            out_scenarios = []
            for s in scenarios:
                evaluated = evaluate_ion_hopping_scenario(
                    scenario=s,
                    temperatures_K=Ts,
                    mc_samples=mc_samples,
                    attempt_frequency_Hz=1e13
                )
                # Add host and ion to context
                evaluated["context"]["host"] = host
                evaluated["context"]["ion"] = ion_sym
                evaluated["context"]["structure_type"] = struct_type_norm
                out_scenarios.append(evaluated)
            
            # 4) Rank and summarize
            out_scenarios.sort(key=lambda d: d["Ea_eV"]["P50"])
            top = out_scenarios[:int(return_top)] if return_top else out_scenarios
            
            # Global range
            ea_min = min(s["Ea_eV"]["P10"] for s in out_scenarios) if out_scenarios else None
            ea_max = max(s["Ea_eV"]["P90"] for s in out_scenarios) if out_scenarios else None
            
            # Family minima (by coarse bucket)
            family_minima: Dict[str, Dict[str, Any]] = {}
            for s in out_scenarios:
                fam = s["context"]["key"].split("_")[0]  # AB, AA, Graphite, generic...
                if fam not in family_minima or s["Ea_eV"]["P50"] < family_minima[fam]["Ea_eV"]["P50"]:
                    family_minima[fam] = s
            
            # Unique citations
            unique_citations = list(set(all_citations))
            
            duration_ms = (time.time() - t0) * 1000
            
            return success_result(
                handler="electrochemistry",
                function="estimate_ion_hopping_barrier_suite",
                data={
                    "host": host,
                    "ion": ion_sym,
                    "structure_type": struct_type_norm,
                    "scenarios_ranked": top,
                    "family_minima": family_minima,
                    "consensus_range": {
                        "Ea_eV_min_P10": round(ea_min, 4) if ea_min is not None else None,
                        "Ea_eV_max_P90": round(ea_max, 4) if ea_max is not None else None
                    },
                    "assumptions": {
                        "temps_K": Ts,
                        "attempt_frequency_Hz": 1e13,
                        "mc_samples": mc_samples,
                        "theta_definition": "gallery fractional occupancy (stage-relative)",
                        "notes": [
                            "Coverage increases barriers modestly; defects/edges lower them.",
                            "Interlayer expansion lowers gallery barriers; compression raises.",
                            "For non-graphite, results come from structure-class heuristics + variants.",
                        ]
                    }
                },
                citations=unique_citations if unique_citations else [
                    "Literature: graphite/BLG Li diffusion pathways (Persson et al., Umegaki et al.)"
                ],
                confidence=Confidence.MEDIUM if is_graphitic(host) else Confidence.LOW,
                notes=[
                    "This suite enumerates multiple site/topology/stacking/coverage/defect cases.",
                    "Use DFT+NEB for quantitative barriers along specific paths you care about.",
                    f"Generated {len(out_scenarios)} total scenarios; returning top {len(top)} by median barrier."
                ],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            _log.error(f"Error in ion hopping barrier suite for {host_material}/{ion}: {e}", exc_info=True)
            return error_result(
                handler="electrochemistry",
                function="estimate_ion_hopping_barrier_suite",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=[],
                duration_ms=duration_ms
            )

