"""
Voltage calculation utilities for electrochemistry handlers.

Functions for computing electrode voltages via convex hull analysis.
"""
import logging
from typing import Dict, Any
import numpy as np

_log = logging.getLogger(__name__)

# Check PyMatGen availability
try:
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    _log.warning("PyMatGen battery modules not available")
    PYMATGEN_AVAILABLE = False


def compute_alloy_voltage_via_hull(
    mpr,
    formula: str,
    working_ion: str = "Li",
    x_max: float = 3.0,
    dx: float = 0.05,
    min_voltage: float = 0.02
) -> Dict[str, Any]:
    """
    Compute an alloy/insertion voltage by scanning the Li–(host) convex hull.
    
    Args:
        mpr: MPRester client instance
        formula: Chemical formula (may include working ion, will be stripped)
        working_ion: Working ion symbol (e.g., "Li")
        x_max: Maximum Li per host atom to scan
        dx: Step size for scanning
        min_voltage: Minimum voltage cutoff (V)
        
    Returns:
        Dictionary with success status and electrode data
        
    Notes:
        - x_host is Li per host atom (sum of non-Li atoms = 1)
        - Voltage: V = -ΔG_form/Δx (eV per Li == V vs Li/Li+)
        - G(x) uses hull formation energy (two-phase equilibrium at 0 K)
        - Stops at Li plating (pure Li metal in equilibrium)
    """
    try:
        if not mpr:
            return {"success": False, "error": "MPRester client not initialized"}

        if not PYMATGEN_AVAILABLE:
            return {"success": False, "error": "PyMatGen not available"}

        # Parse and extract host composition
        input_comp = Composition(formula)
        host_elems = [el.symbol for el in input_comp.elements if el.symbol != working_ion]
        if not host_elems:
            return {"success": False, "error": "Host must contain at least one non-working-ion element"}
        
        host_dict_input = input_comp.get_el_amt_dict()
        host_counts_input = {el: amt for el, amt in host_dict_input.items() if el != working_ion}
        host_only_comp = Composition(host_counts_input)
        host_formula = host_only_comp.reduced_formula
        
        # Get chemical system and entries
        chemsys = "-".join(sorted(set(host_elems + [working_ion])))
        entries = mpr.get_entries_in_chemsys(chemsys)
        if not entries:
            return {"success": False, "error": f"No entries found for {chemsys}"}

        pd = PhaseDiagram(entries)

        # Working ion reference (metal)
        wi_el = Element(working_ion)
        wi_entries = [e for e in entries if len(e.composition.elements) == 1 and wi_el in e.composition]
        if not wi_entries:
            return {"success": False, "error": f"No pure {working_ion} entry in entries"}

        # Normalize host ratios
        total_host = float(sum(host_counts_input.values()))
        host_ratios = {el: amt / total_host for el, amt in host_counts_input.items()}
        n_host_atoms_per_fu = total_host

        # Helper: composition at Li per host atom = x
        def comp_at(x: float) -> Composition:
            comp = {working_ion: x}
            comp.update(host_ratios)
            return Composition(comp)

        # Hull formation energy per host atom
        def G_host(x: float) -> float:
            e_hull_atom = pd.get_hull_energy_per_atom(comp_at(x))
            e_ref_atom = pd.get_reference_energy_per_atom(comp_at(x))
            g_form_atom = e_hull_atom - e_ref_atom
            return g_form_atom * (1.0 + x)

        # Sample energies along the line
        xs = np.arange(0.0, float(x_max) + 1e-9, float(dx))
        Gs = np.array([G_host(float(x)) for x in xs], dtype=float)

        # Build voltage steps
        volts = []
        steps = []
        plating_starts_at_x = None
        
        for i in range(len(xs) - 1):
            x1, x2 = float(xs[i]), float(xs[i + 1])
            dx_i = x2 - x1
            if dx_i <= 0:
                continue
            
            # Check for Li plating
            mid_x = 0.5 * (x1 + x2)
            decomp, _ = pd.get_decomp_and_hull_energy_per_atom(comp_at(mid_x))
            contains_li_metal = any(
                (len(entry.composition.elements) == 1 and entry.composition.reduced_formula == working_ion)
                for entry in decomp.keys()
            )
            if contains_li_metal:
                if plating_starts_at_x is None:
                    plating_starts_at_x = x1
                continue
            
            # Calculate voltage
            V = -(Gs[i + 1] - Gs[i]) / dx_i
            if min_voltage <= V <= 2.0:
                volts.append(float(V))
                steps.append((x1, x2, float(V)))

        if not volts:
            return {"success": False, "error": "No physically plausible Li-insertion steps found"}

        # Calculate metrics
        weights = [(x2 - x1) for (x1, x2, _) in steps]
        avg_v = float(np.average(volts, weights=weights))
        vmin = float(min(volts))
        vmax = float(max(volts))
        x_max_valid = max(x2 for (x1, x2, _) in steps)

        # Capacity estimate (mAh/g)
        x_per_fu = x_max_valid * n_host_atoms_per_fu
        M_host = host_only_comp.weight
        capacity_mAh_g = 26801.0 * x_per_fu / M_host
        energy_Wh_kg = avg_v * capacity_mAh_g
        
        # Build voltage profile
        x_mix_at_max = (n_host_atoms_per_fu * x_max_valid) / (1 + n_host_atoms_per_fu * x_max_valid)
        voltage_profile_steps = []
        
        for (step_x1, step_x2, step_V) in steps:
            x_mix_1 = (n_host_atoms_per_fu * step_x1) / (1 + n_host_atoms_per_fu * step_x1) if step_x1 > 0 else 0.0
            x_mix_2 = (n_host_atoms_per_fu * step_x2) / (1 + n_host_atoms_per_fu * step_x2)
            voltage_profile_steps.append({
                "x_host_1": float(step_x1),
                "x_host_2": float(step_x2),
                "x_mix_1": float(x_mix_1),
                "x_mix_2": float(x_mix_2),
                "V": float(step_V)
            })

        electrode = {
            "battery_id": None,
            "material_id": None,
            "input_formula": formula,
            "host_formula": host_formula,
            "formula": host_formula,
            "formula_discharge": None,
            "formula_charge": None,
            "working_ion": working_ion,
            "average_voltage": float(avg_v),
            "max_voltage_step": float(vmax),
            "min_voltage": float(vmin),
            "max_voltage": float(vmax),
            "capacity_grav": float(capacity_mAh_g),
            "capacity_basis": f"per gram of host ({host_formula})",
            "capacity_vol": None,
            "energy_grav": float(energy_Wh_kg),
            "energy_vol": None,
            "fracA_charge": None,
            "fracA_discharge": None,
            "framework": "-".join(sorted(host_elems)),
            "source": "computed_from_phase_diagram",
            "diagnostics": {
                "chemsys": chemsys,
                "x_max_scanned": float(x_max),
                "x_host_max": float(x_max_valid),
                "x_mix_at_x_host_max": float(x_mix_at_max),
                "dx": float(dx),
                "min_voltage_cutoff": float(min_voltage),
                "n_steps": len(steps),
                "n_host_atoms_per_fu": float(n_host_atoms_per_fu),
                "host_molar_mass_g_per_mol": float(M_host),
                "plating_starts_at_x_host": float(plating_starts_at_x) if plating_starts_at_x is not None else None,
                "note": "x_host = Li per host atom; x_mix = mole fraction of Li in mixture"
            },
            "voltage_profile_steps": voltage_profile_steps
        }

        return {"success": True, "electrode": electrode}

    except Exception as e:
        _log.error(f"compute_alloy_voltage_via_hull error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def calculate_voltage_from_insertion_electrode(
    mpr,
    electrode_formula: str,
    working_ion: str = "Li"
) -> Dict[str, Any]:
    """
    Calculate electrode voltage from convex hull using PyMatGen InsertionElectrode.
    
    Args:
        mpr: MPRester client instance
        electrode_formula: Chemical formula of electrode
        working_ion: Working ion symbol
        
    Returns:
        Dictionary with calculation results
    """
    try:
        if not mpr:
            return {"success": False, "error": "MPRester client not initialized"}

        if not PYMATGEN_AVAILABLE:
            return {"success": False, "error": "PyMatGen battery modules not available"}

        # Build chemical system
        host_comp = Composition(electrode_formula)
        host_elems = sorted({el.symbol for el in host_comp.elements if el.symbol != working_ion})
        all_elems = sorted(set(host_elems + [working_ion]))
        chemsys = "-".join(all_elems)

        _log.info(f"Getting entries for {chemsys} system")

        # Get entries
        entries = mpr.get_entries_in_chemsys(chemsys)
        if not entries:
            return {"success": False, "error": f"No entries found for {chemsys} system"}

        # Identify working ion entry
        wi_el = Element(working_ion)
        wi_entries = [e for e in entries if len(e.composition.elements) == 1 and wi_el in e.composition]
        if not wi_entries:
            return {"success": False, "error": f"No pure {working_ion} entry found in {chemsys} entries"}
        wi_entry = min(wi_entries, key=lambda e: e.energy_per_atom)

        # Build InsertionElectrode
        ie = InsertionElectrode.from_entries(entries=entries, working_ion_entry=wi_entry)

        # Extract matching framework
        frameworks = [fw for fw in ie.get_frameworks() 
                     if sorted({el.symbol for el in fw.composition.elements if el.symbol != working_ion}) == host_elems]
        
        if not frameworks:
            frameworks = [fw for fw in ie.get_frameworks() 
                         if set({el.symbol for el in fw.composition.elements if el.symbol != working_ion}) == set(host_elems)]
        
        if not frameworks:
            return {
                "success": False,
                "error": "No matching framework found for requested host"
            }

        # Get voltage summary
        vdict = ie.get_summary_dict(framework=frameworks[0])
        avg_v = vdict.get("average_voltage", None)
        vmin = vdict.get("min_voltage", None)
        vmax = vdict.get("max_voltage", None)

        # Sanity check
        if avg_v is None or (avg_v < -0.1 or avg_v > 6.0):
            return None  # Signal to try fallback

        return {
            "success": True,
            "calculation_method": "pymatgen_insertion_electrode",
            "calculated_voltage": avg_v,
            "chemical_system": chemsys,
            "framework_formula": frameworks[0].composition.reduced_formula,
            "voltage_range": {"min": vmin, "max": vmax, "average": avg_v},
            "capacity_grav": vdict.get("capacity_grav"),
            "capacity_vol": vdict.get("capacity_vol"),
            "energy_grav": vdict.get("energy_grav"),
            "num_entries_used": len(entries),
            "notes": [
                "Voltages derived from two-phase equilibria on the convex hull (0 K)",
                f"Reported vs. {working_ion}/{working_ion}+; consistent entry set throughout",
                "Based on thermodynamically rigorous phase diagram analysis"
            ],
            "methodology": "PyMatGen InsertionElectrode with convex hull two-phase equilibria"
        }

    except Exception as e:
        _log.error(f"Voltage calculation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

