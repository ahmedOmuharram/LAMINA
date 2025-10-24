"""
AI Functions for Battery and Electrochemistry

This module contains all AI-accessible functions for battery electrode analysis,
voltage calculations, and electrochemical properties.
"""
import logging
from typing import Optional, List, Dict, Any

from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from . import utils

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
    
    # Known priors (literature benchmarks in eV)
    KNOWN_BARRIERS = {
        # Graphite (in-plane). Stage-specific values from NEB (Persson 2010) + μSR (Umegaki 2017).
        ("C6", "Li"): {
            "Ea": 0.28,  # stage I (LiC6)
            "range": [0.20, 0.35],
            "note": "Graphite stage I (LiC6), in-plane hopping",
            "citations": [
                "Persson et al., Phys. Rev. B 82, 125416 (2010), Table II (≈293 meV)",
                "Umegaki et al., PCCP 19, 19058 (2017): Ea(C6Li)=270(5) meV"
            ],
            "descriptors": {"stage": "I", "path": "in-plane"}
        },
        ("C12", "Li"): {
            "Ea": 0.20,  # stage II (LiC12)
            "range": [0.15, 0.28],
            "note": "Graphite stage II (LiC12), in-plane hopping",
            "citations": [
                "Persson et al., Phys. Rev. B 82, 125416 (2010), Table II (≈218-283 meV)",
                "Umegaki et al., PCCP 19, 19058 (2017): Ea(C12Li)=170(20) meV"
            ],
            "descriptors": {"stage": "II", "path": "in-plane"}
        },
        ("graphite", "Li"): {
            "Ea": 0.22,  # stage unspecified → midpoint between stage I/II
            "range": [0.15, 0.30],
            "note": "Graphite (stage unspecified), in-plane hopping",
            "citations": [
                "Persson et al., PRB 82, 125416 (2010)",
                "Umegaki et al., PCCP 19, 19058 (2017)"
            ],
            "descriptors": {"stage": "unspecified", "path": "in-plane"}
        },
    }
    
    # 1) Exact formula match first (e.g., 'C6', 'C12')
    for (mat, i), info in KNOWN_BARRIERS.items():
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
    for (mat, i), info in KNOWN_BARRIERS.items():
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
    STRUCTURE_DEFAULTS = {
        "layered": {
            "Ea": 0.30,
            "range": [0.15, 0.50],
            "confidence": "medium",
            "note": "Layered hosts: graphite in-plane ~0.17–0.30 eV (stage dependent); layered oxides/sulfides typically ~0.3–0.5 eV",
        },
        "1D-channel": {
            "Ea": 0.30,
            "range": [0.15, 0.50],
            "confidence": "medium",
            "note": "1D channels: moderate barriers along channel direction",
        },
        "olivine": {
            "Ea": 0.25,
            "range": [0.15, 0.35],
            "confidence": "medium",
            "note": "Olivine structures: 1D channels with typical barriers ~0.2-0.3 eV",
        },
        "3D": {
            "Ea": 0.40,
            "range": [0.20, 0.70],
            "confidence": "low",
            "note": "3D frameworks: higher barriers, more tortuous paths",
        },
        "unknown": {
            "Ea": 0.35,
            "range": [0.10, 0.80],
            "confidence": "low",
            "note": "Structure unknown; wide uncertainty",
        },
    }
    
    defaults = STRUCTURE_DEFAULTS.get(struct_type, STRUCTURE_DEFAULTS["unknown"])
    
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
        try:
            if not self.mpr:
                return {
                    "success": False,
                    "error": "MPRester client not initialized",
                    "citations": ["Materials Project"]
                }
            
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

                return {
                    "success": True,
                    "count": len(electrode_data),
                    "electrodes": electrode_data,
                    "query": query_params,
                    "notes": [
                        f"Found {len(electrode_data)} electrode materials"
                        + (" (computed from convex hull)" if electrode_data and electrode_data[0].get("source") == "computed_from_phase_diagram" else ""),
                        "Voltages are reported vs. the working ion (e.g., Li/Li+)",
                        "capacity_grav is in mAh/g, energy_grav is in Wh/kg",
                        "Framework compositions verified to match requested elements" if elements else ""
                    ],
                    "citations": ["Materials Project", "pymatgen"]
                }
                
            except AttributeError:
                _log.warning("insertion_electrodes.search not available, trying alternative method")
                return await self._fallback_voltage_calculation(
                    formula=formula,
                    elements=elements,
                    working_ion=working_ion
                )
            
        except Exception as e:
            _log.error(f"Error in search_battery_electrodes: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try using calculate_voltage_from_formation_energy for custom calculations",
                "citations": ["Materials Project", "pymatgen"]
            }
    

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
                    e = synth["electrode"]
                    return {
                        "success": True,
                        "calculation_method": "phase_diagram_line_scan",
                        "calculated_voltage": e["average_voltage"],
                        "chemical_system": e["diagnostics"]["chemsys"],
                        "framework_formula": e["framework"],
                        "voltage_range": {"min": e["min_voltage"], "max": e["max_voltage"], "average": e["average_voltage"]},
                        "capacity_grav": e["capacity_grav"],
                        "energy_grav": e["energy_grav"],
                        "citations": ["Materials Project", "pymatgen"],
                        "notes": [
                            "Voltages from two-phase convex-hull scan along fixed host ratio (0 K)",
                            f"Reported vs. {working_ion}/{working_ion}+; consistent entry set"
                        ],
                        "diagnostics": e.get("diagnostics", {})
                    }
                # If both methods fail, return the original error
                return result if result else {"success": False, "error": "Voltage calculation failed"}
            
            return result

        except Exception as e:
            _log.error(f"Voltage calculation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    

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
        try:
            if not self.mpr:
                return {"success": False, "error": "MPRester client not initialized"}
            
            # Try to get electrode data
            if hasattr(self.mpr, 'insertion_electrodes'):
                try:
                    # Search by material_id or battery_id
                    results = self.mpr.insertion_electrodes.search(battery_id=material_id)
                    
                    if not results:
                        # Try as a formula
                        results = self.mpr.insertion_electrodes.search(formula=material_id)
                    
                    if results:
                        electrode = results[0]
                        
                        # Extract voltage profile using utility function
                        profile_data = utils.extract_voltage_profile(electrode)
                        profile_data["success"] = True
                        profile_data["material_id"] = material_id
                        
                        return profile_data
                        
                except Exception as e:
                    _log.warning(f"Error getting voltage profile: {e}")
            
            # Fallback: just return summary data
            mat_data = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["material_id", "formula_pretty", "formation_energy_per_atom", 
                       "energy_above_hull"]
            )
            
            if mat_data:
                mat = mat_data[0]
                return {
                    "success": True,
                    "material_id": str(mat.material_id),
                    "formula": mat.formula_pretty,
                    "formation_energy_per_atom": mat.formation_energy_per_atom,
                    "energy_above_hull": mat.energy_above_hull,
                    "citations": ["Materials Project"],
                    "notes": [
                        "Detailed voltage profile not available",
                        "Use search_battery_electrodes to find electrode materials with profiles",
                        "Or use calculate_voltage_from_formation_energy for estimates"
                    ]
                }
            
            return {
                "success": False,
                "error": f"Material {material_id} not found",
                "citations": ["Materials Project"]
            }
            
        except Exception as e:
            _log.error(f"Error in get_voltage_profile: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "citations": ["Materials Project"]
            }
    

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
                
                if electrode_result.get("success") and electrode_result.get("electrodes"):
                    comparison_results.append({
                        "formula": formula,
                        "data": electrode_result["electrodes"][0],
                        "source": "electrodes_database"
                    })
                else:
                    # Fallback to formation energy calculation
                    calc_result = await self.calculate_voltage_from_formation_energy(
                        electrode_formula=formula,
                        working_ion=working_ion
                    )
                    
                    if calc_result.get("success"):
                        comparison_results.append({
                            "formula": formula,
                            "data": {
                                "voltage": calc_result.get("calculated_voltage"),
                                "material_id": calc_result.get("electrode_material", {}).get("material_id"),
                                "formation_energy": calc_result.get("electrode_material", {}).get("formation_energy_per_atom")
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
            
            return {
                "success": True,
                "working_ion": working_ion,
                "comparison": comparison_results,
                "count": len(comparison_results),
                "summary": summary,
                "citations": ["Materials Project", "pymatgen"],
                "notes": [
                    "Voltages are vs. working ion reference (e.g., Li/Li+)",
                    "Data from insertion_electrodes database: pre-computed voltage profiles",
                    "Data from convex hull: thermodynamically rigorous phase diagram calculations",
                    "All data is from Materials Project DFT calculations - no heuristics or estimates"
                ]
            }
            
        except Exception as e:
            _log.error(f"Error comparing electrodes: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "citations": ["Materials Project", "pymatgen"]
            }
    

    @ai_function(
        desc="Check if a composition is thermodynamically stable (on convex hull). "
             "USE THIS to determine if a material can exist as a stable phase or if it decomposes. "
             "Returns energy above hull (if entry exists) and decomposition products. "
             "Essential for questions about 'thermodynamically stable', 'can form', 'stable phase'."
    )
    async def check_composition_stability(
        self,
        composition: Annotated[
            str,
            AIParam(desc="Chemical composition to check (e.g., 'Cu8LiAl', 'Li3Al2', 'Cu80Li10Al10')")
        ]
    ) -> Dict[str, Any]:
        """
        Check if a composition is thermodynamically stable.
        
        Returns:
        - energy_above_hull: How far above the convex hull (0 = stable, None if no entry)
        - is_stable: Whether an entry exists at this composition and is on the hull
        - decomposition: What phases it decomposes into if unstable
        """
        result = utils.check_composition_stability_detailed(self.mpr, composition)
        result["citations"] = ["Materials Project", "pymatgen"]
        return result
    

    @ai_function(
        desc="Analyze a composition as a potential battery anode, including stability check and voltage. "
             "USE THIS for questions about whether a material 'can form an anode', 'is suitable as anode'. "
             "Checks thermodynamic stability and calculates voltage if viable."
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
        ] = "Li"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a composition as a battery anode.
        
        Checks:
        1. Thermodynamic stability (is it stable or does it decompose?)
        2. If stable: calculate voltage vs working ion
        3. If unstable: analyze decomposition products as potential anode
        4. Overall assessment of viability
        """
        try:
            if not utils.PYMATGEN_AVAILABLE:
                return {"success": False, "error": "PyMatGen not available"}
            
            from pymatgen.core import Composition
            
            # First check stability
            stability = await self.check_composition_stability(composition)
            
            if not stability.get("success"):
                return stability
            
            result = {
                "success": True,
                "composition": composition,
                "working_ion": working_ion,
                "stability_analysis": stability,
                "voltage_analysis": {},
                "viability_assessment": {}
            }
            
            # Try to get voltage information
            is_stable = stability.get("is_stable", False)
            decomp_phases = stability.get("decomposition", [])
            
            # Extract the non-working-ion composition for voltage calculation
            comp = Composition(composition)
            host_elements = [el.symbol for el in comp.elements if el.symbol != working_ion]
            
            if working_ion not in [el.symbol for el in comp.elements]:
                # Composition doesn't contain working ion - this is the host material
                host_formula = composition
                
                # Try to calculate voltage
                voltage_result = await self.search_battery_electrodes(
                    formula=host_formula,
                    working_ion=working_ion,
                    max_entries=1
                )
                
                if voltage_result.get("success") and voltage_result.get("electrodes"):
                    result["voltage_analysis"] = voltage_result["electrodes"][0]
                else:
                    result["voltage_analysis"] = {"error": "No voltage data available for this composition"}
            else:
                # Composition already contains working ion - it's a lithiated phase
                result["voltage_analysis"] = {
                    "note": f"Composition contains {working_ion} - this is a lithiated phase, not a host anode material"
                }
            
            # Assessment
            assessment = {
                "can_form_stable_anode": False,
                "reasoning": []
            }
            
            if is_stable:
                assessment["reasoning"].append(f"{composition} is thermodynamically stable")
                if host_elements:
                    assessment["can_form_stable_anode"] = True
                    assessment["reasoning"].append(f"Can potentially serve as anode material vs {working_ion}/{working_ion}+")
                else:
                    assessment["reasoning"].append(f"Pure {working_ion} - not an anode, but the reference")
            else:
                e_above_hull = stability.get("energy_above_hull")
                assessment["can_form_stable_anode"] = False
                
                if e_above_hull and e_above_hull < 0.1:
                    assessment["reasoning"].append(f"Nearly stable ({e_above_hull:.6f} eV/atom above hull)")
                    assessment["reasoning"].append("May exist as metastable phase under certain conditions")
                else:
                    assessment["reasoning"].append(f"Thermodynamically unstable - decomposes into {len(decomp_phases)} phases")
                
                # Analyze decomposition products
                decomp_formulas = [p["formula"] for p in decomp_phases]
                assessment["reasoning"].append(f"Equilibrium phases: {', '.join(decomp_formulas)}")
                
                # Check if decomposition products include working-ion compounds
                has_working_ion_phases = any(working_ion in formula for formula in decomp_formulas)
                if has_working_ion_phases:
                    assessment["reasoning"].append(
                        f"Decomposition includes {working_ion}-containing phases - a deliberately multiphase composite "
                        "might be engineered, but this composition is not a single-phase stable anode."
                    )
            
            result["viability_assessment"] = assessment
            result["citations"] = ["Materials Project", "pymatgen"]
            
            return result
            
        except Exception as e:
            _log.error(f"Error analyzing anode viability: {e}", exc_info=True)
            return {"success": False, "error": str(e), "citations": ["Materials Project", "pymatgen"]}
    

    @ai_function(
        desc="Analyze the lithiation mechanism of a host material. "
             "Reports phase evolution, two-phase vs single-phase reactions, and equilibrium phases at each voltage step. "
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
        ] = True
    ) -> Dict[str, Any]:
        """
        Analyze the lithiation mechanism by computing the convex hull of G(x) vs x.
        
        Reports:
        - Voltage plateaus based on hull segments (not midpoint phase count)
        - Equilibrium phases from endpoint decompositions
        - Whether reactions are two-state plateaus or single-phase regions
        - Initial reaction mechanism
        - Full lithiation sequence
        
        Args:
            room_temp: If True, filter out phases with E_hull > 0.03 eV/atom (hard to form at RT)
        """
        result = utils.analyze_lithiation_mechanism_detailed(
            self.mpr, host_composition, working_ion, max_x, room_temp
        )
        result["citations"] = ["Materials Project", "pymatgen"]
        return result

    @ai_function(
        desc=(
            "Estimate the ion hopping/diffusion barrier (activation energy, eV) for an intercalating ion "
            "(e.g., Li, Na, Mg) moving between sites in an electrode material (e.g., graphite, LiFePO4). "
            "Use this for questions about ion mobility, diffusion barriers, or lithium hopping in electrodes."
        ),
        auto_truncate=128000,
    )
    async def estimate_ion_hopping_barrier(
        self,
        host_material: Annotated[str, AIParam(desc="Host electrode material formula, e.g., 'C6' (graphite), 'LiFePO4', 'TiS2'.")],
        ion: Annotated[str, AIParam(desc="Intercalating ion, e.g., 'Li', 'Na', 'Mg'.")] = "Li",
        structure_type: Annotated[Optional[str], AIParam(desc="Structure type/dimensionality: 'layered', '1D-channel', '3D', or 'olivine'. Optional.")] = None,
    ) -> Dict[str, Any]:
        """
        Estimate ion hopping barrier in electrode materials using structure-based heuristics.
        """
        try:
            # Normalize inputs
            ion_sym = ion.strip().capitalize()
            host = host_material.strip()
            
            # Classify structure type if not provided or normalize user input
            struct_type = _classify_electrode_structure(host, structure_type)
            
            # Get typical barrier ranges and confidence based on structure + known priors
            barrier_info = _estimate_barrier_from_structure(host, ion_sym, struct_type)
            
            return {
                "success": True,
                "host_material": host,
                "ion": ion_sym,
                "structure_type": struct_type,
                "activation_energy_eV": barrier_info["Ea_eV"],
                "energy_range_eV": barrier_info["range_eV"],
                "confidence": barrier_info["confidence"],
                "method": "structure_heuristic_v1",
                "descriptors": barrier_info.get("descriptors", {}),
                "caveats": (
                    "Heuristic estimate based on material class and structure dimensionality. "
                    "Actual barriers depend on crystallographic pathway, site occupancy, lattice strain, "
                    "and defect concentration. Use DFT+NEB or impedance spectroscopy for accuracy."
                ),
                "citations": barrier_info.get("citations", []),
            }
        except Exception as e:
            _log.error(f"Error estimating ion hopping barrier for {host_material}/{ion}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

