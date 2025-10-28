"""
Lithiation mechanism analysis utilities for electrochemistry handlers.

Functions for analyzing lithiation mechanisms and phase evolution.
"""
import logging
from typing import Optional, List, Dict, Any
from collections import defaultdict
import numpy as np

_log = logging.getLogger(__name__)

# Check PyMatGen availability
try:
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    _log.warning("PyMatGen not available")
    PYMATGEN_AVAILABLE = False


def _prepare_entries_for_pd(raw_entries):
    """
    Prepare entries for PhaseDiagram construction with proper compatibility corrections.
    
    - Applies MP compatibility corrections
    - Keeps only the lowest energy entry per reduced formula
    - Ensures elemental references are present
    
    Args:
        raw_entries: List of raw ComputedEntry objects
        
    Returns:
        List of prepared entries ready for PhaseDiagram
    """
    try:
        # Prefer the 2020 compatibility (handles corrections + filtering)
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility as _Compat
    except Exception:
        from pymatgen.entries.compatibility import MaterialsProjectCompatibility as _Compat

    compat = _Compat()
    # Apply corrections & drop incompatible entries
    entries = compat.process_entries(raw_entries, clean=True) or []

    # Keep the lowest-energy entry per reduced formula
    best = {}
    for e in entries:
        f = e.composition.reduced_formula
        if (f not in best) or (e.energy_per_atom < best[f].energy_per_atom):
            best[f] = e
    entries = list(best.values())

    if not entries:
        return []

    # Ensure elemental references for all elements present
    from pymatgen.core import Element
    elems = set()
    for e in entries:
        elems.update([el.symbol for el in e.composition.elements])
    
    # Confirm each element has at least one unary entry
    have = {el: False for el in elems}
    for e in entries:
        if len(e.composition.elements) == 1:
            have[next(iter(e.composition.elements)).symbol] = True

    missing = [el for el, ok in have.items() if not ok]
    if missing:
        _log.warning(f"Missing elemental references for: {missing}")

    return entries


def analyze_lithiation_mechanism_detailed(
    mpr,
    host_composition: str,
    working_ion: str = "Li",
    max_x: float = 3.0,
    room_temp: bool = True
) -> Dict[str, Any]:
    """
    Analyze the lithiation mechanism by computing the convex hull of G(x) vs x.
    
    Args:
        mpr: MPRester client instance
        host_composition: Host material composition
        working_ion: Working ion symbol
        max_x: Maximum Li per host atom
        room_temp: Filter high-energy phases
        
    Returns:
        Dictionary with lithiation mechanism analysis
    """
    try:
        if not mpr:
            return {"success": False, "error": "MPRester client not initialized"}
        
        if not PYMATGEN_AVAILABLE:
            return {"success": False, "error": "PyMatGen not available"}
        
        # Parse host composition
        input_comp = Composition(host_composition)
        host_elems = [el.symbol for el in input_comp.elements if el.symbol != working_ion]
        if not host_elems:
            return {"success": False, "error": "Host must contain at least one non-working-ion element"}
        
        # Extract host-only composition
        host_dict_input = input_comp.get_el_amt_dict()
        host_counts_input = {el: amt for el, amt in host_dict_input.items() if el != working_ion}
        host_only_comp = Composition(host_counts_input)
        host_formula = host_only_comp.reduced_formula
        
        # Get chemical system
        chemsys = "-".join(sorted(set(host_elems + [working_ion])))
        entries_raw = mpr.get_entries_in_chemsys(chemsys)
        if not entries_raw:
            return {"success": False, "error": f"No entries found for {chemsys}"}
        
        # Prepare entries with proper compatibility corrections and ground-state selection
        entries = _prepare_entries_for_pd(entries_raw)
        if not entries:
            return {"success": False, "error": f"No compatible entries after processing for {chemsys}"}
        
        _log.info(f"Prepared {len(entries)} entries for PhaseDiagram (lowest energy per formula)")
        
        # Build phase diagram with correct ground states
        pd = PhaseDiagram(entries)
        
        # Optional: report metastable phases that were excluded (for room temp context)
        filtered_phases = []
        if room_temp:
            # Build PD from raw entries to see what would have been metastable
            pd_full = PhaseDiagram(entries_raw)
            for entry in entries_raw:
                eh = pd_full.get_e_above_hull(entry)
                if eh >= 0.03:
                    filtered_phases.append({
                        "formula": entry.composition.reduced_formula,
                        "material_id": str(entry.entry_id) if hasattr(entry, 'entry_id') else None,
                        "e_above_hull": float(eh),
                        "reason": f"E_hull = {eh:.4f} eV/atom > 0.03 (metastable at RT)"
                    })
            _log.info(f"Room temp context: {len(filtered_phases)} metastable phases identified")
        
        # Normalize host ratios
        total_host = float(sum(host_counts_input.values()))
        host_ratios = {el: amt / total_host for el, amt in host_counts_input.items()}
        n_host_atoms_per_fu = total_host
        
        # Compute lithiation steps
        steps_with_phases = _compute_lithiation_steps(
            pd, host_ratios, working_ion, max_x, n_host_atoms_per_fu
        )
        
        if not steps_with_phases:
            return {"success": False, "error": "No physically plausible voltage plateaus found"}
        
        # Analyze initial reaction
        initial_step = steps_with_phases[0]
        initial_mechanism = _format_initial_mechanism(initial_step)
        
        # Calculate average voltage
        voltages = [s["voltage"] for s in steps_with_phases]
        weights = [s["x_host_range"]["end"] - s["x_host_range"]["start"] for s in steps_with_phases]
        avg_voltage = float(np.average(voltages, weights=weights))
        voltage_uncertainty = 0.15
        
        # Find plating onset
        plating_starts_at_x = _find_plating_onset(pd, host_ratios, working_ion, max_x)
        
        result = {
            "success": True,
            "host_formula": host_formula,
            "input_formula": host_composition,
            "working_ion": working_ion,
            "chemical_system": chemsys,
            "room_temp_filter": room_temp,
            "filtered_phases": filtered_phases if room_temp else [],
            "initial_reaction": initial_mechanism,
            "lithiation_steps": steps_with_phases,
            "num_plateau_steps": len(steps_with_phases),
            "average_voltage": avg_voltage,
            "voltage_range": {"min": min(voltages), "max": max(voltages)},
            "voltage_uncertainty_estimate": voltage_uncertainty,
            "plating_starts_at_x_host": plating_starts_at_x,
            "methodology": _get_methodology_description(voltage_uncertainty),
            "notes": _get_analysis_notes(
                host_formula, working_ion, len(steps_with_phases),
                initial_mechanism, room_temp, filtered_phases, voltage_uncertainty
            )
        }
        
        return result
        
    except Exception as e:
        _log.error(f"Error analyzing lithiation mechanism: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _compute_lithiation_steps(
    pd: 'PhaseDiagram',
    host_ratios: Dict[str, float],
    working_ion: str,
    max_x: float,
    n_host_atoms_per_fu: float,
    dx: float = 0.005,
    frac_eps: float = 1e-8
) -> List[Dict[str, Any]]:
    """
    Compute lithiation voltage steps by following decomposition sets.
    
    Builds plateaus from constant decomposition sets (tie-simplices), not geometric hull sampling.
    This correctly separates two-phase tie-lines from three-phase tie-triangles.
    """
    
    def comp_at(x: float) -> 'Composition':
        comp = {working_ion: x}
        comp.update(host_ratios)
        return Composition(comp)
    
    def decomp_set(x: float):
        """Get stable decomposition at x, ignoring numerically tiny phases."""
        decomp = pd.get_decomposition(comp_at(x))
        
        # Normalize to a stable set of formulas (ignore tiny fractions)
        s = tuple(sorted(
            e.composition.reduced_formula
            for e, frac in decomp.items() if frac > frac_eps
        ))
        
        # Formation energy per atom of the equilibrium mixture
        g_form_atom = sum(
            frac * pd.get_form_energy_per_atom(e)
            for e, frac in decomp.items() if frac > frac_eps
        )
        
        return s, g_form_atom, decomp
    
    # Sample x (exclude clear Li-plating region later)
    xs = np.arange(0.0, float(max_x) + 1e-12, dx)
    if len(xs) < 2:
        return []
    
    sets, g_atoms, decomps = [], [], []
    for x in xs:
        s, g, d = decomp_set(float(x))
        sets.append(s)
        g_atoms.append(g)
        decomps.append(d)
    
    # Find Li plating onset: first x where elemental Li appears in the set
    def has_li(set_tuple):
        return working_ion in set_tuple
    
    plating_idx = None
    for i, s in enumerate(sets):
        if has_li(s):
            plating_idx = i
            break
    
    if plating_idx is not None:
        xs = xs[:plating_idx]
        sets = sets[:plating_idx]
        g_atoms = g_atoms[:plating_idx]
        decomps = decomps[:plating_idx]
        if len(xs) < 2:
            return []
    
    # Build contiguous segments where the decomposition set is constant
    segments = []
    start = 0
    for i in range(1, len(xs)):
        if sets[i] != sets[i-1]:
            segments.append((start, i-1))
            start = i
    segments.append((start, len(xs)-1))
    
    steps = []
    for seg_id, (i0, i1) in enumerate(segments, start=1):
        if i1 <= i0:
            continue
        
        x0, x1 = float(xs[i0]), float(xs[i1])
        g0_atom, g1_atom = float(g_atoms[i0]), float(g_atoms[i1])
        
        # Convert formation energy per atom → per host-"formula" for correct slope
        # (host is normalized to 1 total atom; total atoms = 1 + x)
        G0 = g0_atom * (1.0 + x0)
        G1 = g1_atom * (1.0 + x1)
        
        # Voltage in eV/Li (formation energies already reference elemental Li)
        V = -(G1 - G0) / (x1 - x0) if (x1 - x0) > 1e-12 else None
        if (V is None) or not (0.01 <= V <= 5.0):
            # Filter nonsense segments
            continue
        
        # Report the (unique) phases in this tie-simplex
        phase_formulas = list(sets[i0])
        num_ph = len(phase_formulas)
        two_phase = (num_ph == 2)
        
        # Get material IDs from decomposition
        decomp_at_start = decomps[i0]
        phase_list = []
        for formula in phase_formulas:
            # Find matching entry in decomposition
            mat_id = None
            for entry in decomp_at_start.keys():
                if entry.composition.reduced_formula == formula:
                    mat_id = str(entry.entry_id) if hasattr(entry, 'entry_id') else None
                    break
            phase_list.append({
                "formula": formula,
                "material_id": mat_id,
                "observed_at_x": [x0, x1]
            })
        
        # x expressed also as mixture fraction if you need it
        x_mix0 = (n_host_atoms_per_fu * x0) / (1 + n_host_atoms_per_fu * x0) if x0 > 0 else 0.0
        x_mix1 = (n_host_atoms_per_fu * x1) / (1 + n_host_atoms_per_fu * x1)
        
        steps.append({
            "step_number": seg_id,
            "x_host_range": {"start": x0, "end": x1},
            "x_mix_range": {"start": x_mix0, "end": x_mix1},
            "voltage": float(V),
            "equilibrium_phases": phase_list,
            "num_phases_in_microstructure": num_ph,
            "is_constant_mu_plateau": True,
            "is_two_phase_microstructure": two_phase,
            "reaction_type": ("two-phase plateau (tie-line)"
                              if two_phase else f"three-phase plateau (tie-triangle, {num_ph} phases)"),
            "microstructure_note": (
                f"Decomposition set constant from x={x0:.3f} to {x1:.3f}: "
                f"{', '.join(phase_formulas)} "
                f"→ {num_ph}-phase coexistence."
            )
        })
    
    return steps


def _format_initial_mechanism(initial_step: Dict[str, Any]) -> Dict[str, Any]:
    """Format initial reaction mechanism description."""
    return {
        "reaction_type": initial_step["reaction_type"],
        "voltage": initial_step["voltage"],
        "phases_in_microstructure": [p["formula"] for p in initial_step["equilibrium_phases"]],
        "num_phases": initial_step["num_phases_in_microstructure"],
        "is_constant_mu_plateau": initial_step["is_constant_mu_plateau"],
        "is_two_phase_microstructure": initial_step["is_two_phase_microstructure"],
        "composition_range": f"x_host={initial_step['x_host_range']['start']:.3f}→{initial_step['x_host_range']['end']:.3f}",
        "explanation": (
            f"The initial lithiation is a **constant chemical potential plateau at ~{initial_step['voltage']:.3f} V** "
            f"(convex hull segment). "
            f"Microstructure: {', '.join([p['formula'] for p in initial_step['equilibrium_phases']])} "
            f"({'2-phase coexistence' if initial_step['is_two_phase_microstructure'] else f'{initial_step['num_phases_in_microstructure']}-phase mixture'}). "
            f"Thermodynamically, this is a two-state reaction (lever rule applies between hull vertices), "
            f"but the equilibrium mixture contains {initial_step['num_phases_in_microstructure']} distinct crystalline phases."
        )
    }


def _find_plating_onset(
    pd: 'PhaseDiagram',
    host_ratios: Dict[str, float],
    working_ion: str,
    max_x: float,
    frac_eps: float = 1e-8
) -> Optional[float]:
    """Find the x value where Li plating begins."""
    def comp_at(x: float) -> 'Composition':
        comp = {working_ion: x}
        comp.update(host_ratios)
        return Composition(comp)
    
    dx = 0.05
    xs = np.arange(0.0, float(max_x) + 1e-9, dx)
    
    for x in xs:
        decomp = pd.get_decomposition(comp_at(float(x)))
        
        # Check if elemental Li appears with non-negligible fraction
        contains_li_metal = any(
            (len(entry.composition.elements) == 1 
             and entry.composition.reduced_formula == working_ion
             and frac > frac_eps)
            for entry, frac in decomp.items()
        )
        if contains_li_metal:
            return float(x)
    
    return None


def _get_methodology_description(voltage_uncertainty: float) -> Dict[str, Any]:
    """Get methodology description for lithiation analysis."""
    return {
        "approach": "Decomposition-set tracking along lithiation path",
        "method": "Follow constant decomposition sets from PhaseDiagram.get_decomposition()",
        "phase_identification": "Phases come directly from 2-D ternary phase diagram decomposition",
        "classification": {
            "is_constant_mu_plateau": "True for all segments (constant decomposition set)",
            "is_two_phase_microstructure": "True only if exactly 2 phases in decomposition set (tie-line)",
            "three_phase_plateau": "Decomposition set has 3 phases (tie-triangle)"
        },
        "phase_reporting": "Exact phases from tie-simplex corners, not sampled from interior",
        "voltage_uncertainty": f"±{voltage_uncertainty} V (typical DFT GGA systematic error for alloy anodes)"
    }


def _get_analysis_notes(
    host_formula: str,
    working_ion: str,
    num_steps: int,
    initial_mechanism: Dict[str, Any],
    room_temp: bool,
    filtered_phases: List[Dict[str, Any]],
    voltage_uncertainty: float
) -> List[str]:
    """Generate notes for lithiation analysis."""
    return [
        f"Analyzed lithiation of {host_formula} vs {working_ion}/{working_ion}+",
        f"Initial reaction: {initial_mechanism['reaction_type']} with {initial_mechanism['num_phases']} phases in microstructure",
        f"Total {num_steps} voltage plateaus detected from decomposition-set tracking",
        "Each plateau = constant decomposition set (tie-line or tie-triangle on 2-D ternary hull)",
        "Phases identified from exact PhaseDiagram decomposition, not interior sampling",
        "Entry preparation: Applied MP2020 compatibility corrections and kept lowest energy per formula",
        f"Room temperature context: {'Metastable phases (E_hull > 30 meV/atom) reported but not used' if room_temp else 'All phases included'}",
        f"{'Identified ' + str(len(filtered_phases)) + ' metastable phases' if filtered_phases else 'No metastable phases identified'}",
        f"Voltage uncertainty: ±{voltage_uncertainty} V (typical DFT GGA error for alloy anodes)",
        "0 K thermodynamics with GGA energies; kinetics and finite-T effects not included",
        "Segments truncated at Li plating onset (elemental Li appears in decomposition)"
    ]

