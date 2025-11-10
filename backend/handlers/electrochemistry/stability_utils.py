"""
Stability analysis utilities for electrochemistry handlers.

Functions for checking composition stability and decomposition analysis.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

_log = logging.getLogger(__name__)

# Check PyMatGen availability
try:
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    _log.warning("PyMatGen not available")
    PYMATGEN_AVAILABLE = False

# Constants
DEFAULT_METASTABILITY_TOL = 0.03  # eV/atom


def check_composition_stability_detailed(
    mpr,
    composition: str
) -> Dict[str, Any]:
    """
    Check if a composition is thermodynamically stable.
    
    Args:
        mpr: MPRester client instance
        composition: Chemical composition string
        
    Returns:
        Dictionary with stability analysis results
    """
    try:
        if not mpr:
            return {"success": False, "error": "MPRester client not initialized"}
        
        if not PYMATGEN_AVAILABLE:
            return {"success": False, "error": "PyMatGen not available"}
        
        # Parse composition
        try:
            comp = Composition(composition)
        except Exception as e:
            return {"success": False, "error": f"Invalid composition: {e}"}
        
        # Get chemical system
        elements = sorted([el.symbol for el in comp.elements])
        chemsys = "-".join(elements)
        
        _log.info(f"Checking stability of {composition} in {chemsys} system")
        
        # Get all entries
        entries = mpr.get_entries_in_chemsys(chemsys)
        if not entries:
            return {"success": False, "error": f"No entries found for {chemsys} system"}
        
        # Build phase diagram
        pd = PhaseDiagram(entries)
        
        # Get decomposition
        decomp = pd.get_decomposition(comp)
        
        # Find matching entries
        matching_entries = [e for e in entries if e.composition.reduced_formula == comp.reduced_formula]
        e_above_hull = None
        best_entry = None
        
        if matching_entries:
            best_entry = min(matching_entries, key=lambda e: e.energy_per_atom)
            e_above_hull = float(pd.get_e_above_hull(best_entry))
        
        # Calculate decomposition coefficients
        decomp_phases = _calculate_decomposition_coefficients(comp, decomp)
        
        # Stability decision
        is_stable = (e_above_hull is not None and e_above_hull < 1e-6)
        
        result = {
            "success": True,
            "composition": composition,
            "reduced_formula": comp.reduced_formula,
            "chemical_system": chemsys,
            "is_stable": is_stable,
            "energy_above_hull": e_above_hull,
            "decomposition": decomp_phases,
            "material_id": str(best_entry.entry_id) if best_entry else None,
            "notes": []
        }
        
        # Add notes
        if is_stable:
            result["notes"].append(f"{composition} is thermodynamically stable (on convex hull)")
        else:
            result["notes"].append("Off the convex hull; shown decomposition is the 0 K equilibrium mixture")
            if not best_entry:
                result["notes"].append("No entry with this formula in MP; E_above_hull undefined")
            elif e_above_hull is not None:
                result["notes"].append(f"Entry exists but is {e_above_hull:.6f} eV/atom above hull")
        
        result["notes"].append(
            "Decomposition: 'amount' = formula-unit coefficients (atom-balanced); "
            "'phase_fraction' = composition-space fractions (not volume/mass)"
        )
        
        return result
        
    except Exception as e:
        _log.error(f"Error checking composition stability: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _calculate_decomposition_coefficients(
    comp: 'Composition',
    decomp: Dict[Any, float]
) -> List[Dict[str, Any]]:
    """
    Calculate human-friendly decomposition coefficients.
    
    Args:
        comp: Target composition
        decomp: Decomposition dictionary from phase diagram
        
    Returns:
        List of phase dictionaries with amounts and fractions
    """
    from fractions import Fraction
    
    elems = [el.symbol for el in comp.elements]
    
    def counts_vec(cmp):
        reduced = cmp.reduced_composition
        d = reduced.get_el_amt_dict()
        return [float(d.get(el, 0.0)) for el in elems]
    
    # Build matrix
    B = np.array([counts_vec(e.composition) for e in decomp.keys()], dtype=float).T
    t = np.array(counts_vec(comp), dtype=float)
    
    # Solve for coefficients
    x, *_ = np.linalg.lstsq(B, t, rcond=None)
    x[np.isclose(x, 0, atol=1e-10)] = 0.0
    
    # Convert to small integers
    fracs = [Fraction(v).limit_denominator(64) for v in x]
    denoms = np.array([f.denominator for f in fracs], dtype=int)
    lcm = int(np.lcm.reduce(denoms))
    coeffs = [int(f.numerator * (lcm // f.denominator)) for f in fracs]
    g = int(np.gcd.reduce(coeffs)) if any(coeffs) else 1
    coeffs = [c // g for c in coeffs]
    
    # Format decomposition
    decomp_phases = []
    phase_fractions = list(decomp.values())
    
    for (entry, phase_frac), coeff in zip(decomp.items(), coeffs):
        entry_id = str(entry.entry_id) if hasattr(entry, 'entry_id') else None
        decomp_phases.append({
            "formula": entry.composition.reduced_formula,
            "amount": int(coeff),
            "phase_fraction": float(phase_frac),
            "material_id": entry_id
        })
    
    return decomp_phases


def _best_entry_for_formula(entries: List[Any], reduced_formula: str) -> Optional[Any]:
    """Find the lowest-energy entry matching a reduced formula."""
    matches = [e for e in entries if e.composition.reduced_formula == reduced_formula]
    return min(matches, key=lambda e: e.energy_per_atom) if matches else None


def _format_decomp_phases(decomp: Dict[Any, float]) -> List[Dict[str, Any]]:
    """
    Format decomposition dictionary into user-friendly phase list.
    
    Args:
        decomp: Dict[Entry, float] with fractions summing ~1 (in composition space)
        
    Returns:
        List of phase dictionaries sorted by fraction descending
    """
    out = []
    for entry, frac in decomp.items():
        out.append({
            "formula": entry.composition.reduced_formula,
            "phase_fraction": float(frac),
            "material_id": str(getattr(entry, "entry_id", None))
        })
    # Sort by fraction desc for nicer UI
    out.sort(key=lambda d: d["phase_fraction"], reverse=True)
    return out


def _estimate_theoretical_capacity(
    phases: List[Dict[str, Any]],
    open_element: str = "Li"
) -> Optional[Dict[str, Any]]:
    """
    Estimate theoretical capacity for a multiphase mixture.
    
    Args:
        phases: List of phase dicts with formula and phase_fraction
        open_element: Working ion element (e.g., 'Li')
        
    Returns:
        Dict with capacity estimate or None if can't compute
    """
    try:
        if not PYMATGEN_AVAILABLE:
            return None
        
        total_li = 0.0
        total_mass = 0.0
        
        for phase in phases:
            formula = phase["formula"]
            frac = phase["phase_fraction"]
            
            comp = Composition(formula)
            li_content = comp.get_atomic_fraction(Element(open_element))
            molar_mass = comp.weight  # g/mol
            
            # Weighted contribution
            total_li += li_content * frac
            total_mass += molar_mass * frac
        
        if total_mass > 0 and total_li > 0:
            # Capacity = 26801 * n_Li / M (mAh/g)
            # where n_Li is moles of Li per mole of mixture
            capacity_mah_g = 26801 * total_li / total_mass
            
            return {
                "theoretical_capacity_mAh_g": round(capacity_mah_g, 1),
                "li_content": round(total_li, 4),
                "average_molar_mass_g_mol": round(total_mass, 2),
                "note": "Upper-bound snapshot based on equilibrium mixture; not rate-capable value"
            }
        
        return None
        
    except Exception as e:
        _log.warning(f"Could not estimate capacity: {e}")
        return None


def check_anode_stability_vs_voltage(
    mpr,
    composition: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    npts: int = 11,
    open_element: str = "Li",
    metastability_tol: float = DEFAULT_METASTABILITY_TOL
) -> Dict[str, Any]:
    """
    Battery-relevant stability: grand-potential phase diagram with Li as open species.
    Scans voltage and returns stable decomposition + intervals.
    
    Args:
        mpr: MPRester client instance
        composition: Chemical composition to analyze
        vmin: Minimum voltage vs Li/Li+ (V)
        vmax: Maximum voltage vs Li/Li+ (V)
        npts: Number of voltage points to scan
        open_element: Open element symbol (default: "Li")
        metastability_tol: Tolerance for metastability (eV/atom)
        
    Returns:
        Dictionary with voltage-dependent stability analysis:
        {
            success, composition, chemsys,
            voltage_scan: [
                {V, stable_single_phase: bool, phases: [...], note: str}
            ],
            intervals: [
                {V_min, V_max, phases: [...], stable_single_phase: bool}
            ],
            on_hull_at_any_V: bool,
            capacity_estimate: {...},
            summary: str,
            notes: [...]
        }
    """
    try:
        if not mpr:
            return {"success": False, "error": "MPRester client not initialized"}
        
        if not PYMATGEN_AVAILABLE:
            return {"success": False, "error": "PyMatGen not available"}
        
        # Parse composition
        try:
            comp = Composition(composition)
        except Exception as e:
            return {"success": False, "error": f"Invalid composition: {e}"}
        
        # Normalize open element
        open_element = Element(open_element.strip().capitalize()).symbol
        
        elems = {el.symbol for el in comp.elements}
        elems_with_open = sorted(elems | {open_element})
        chemsys = "-".join(elems_with_open)
        
        # Build closed composition (without open element) for GPPD decomposition
        # GPPD operates in the subspace of closed elements; the open element (Li) is
        # treated via its chemical potential μ_Li. We must pass a Li-free composition
        # to get_decomposition(), but the returned phases can still contain Li.
        closed_dict = {
            el: amt
            for el, amt in comp.get_el_amt_dict().items()
            if el != open_element
        }
        
        if not closed_dict:
            return {"success": False, "error": f"Composition contains only the open element {open_element}"}
        
        closed_comp = Composition(closed_dict)
        
        _log.info(f"Battery-relevant stability check for {composition} in {chemsys} (voltage scan)")
        
        # Pull entries for closed+open system
        entries = mpr.get_entries_in_chemsys(chemsys)
        if not entries:
            return {"success": False, "error": f"No entries found for {chemsys}"}
        
        # Reference chemical potential for Li (0 V vs Li): lowest-energy Li entry
        li_entries = [e for e in entries if e.composition.reduced_formula == open_element]
        if not li_entries:
            return {"success": False, "error": f"No {open_element} reference entry; cannot build grand-potential PD."}
        
        mu_li_0V = min(li_entries, key=lambda e: e.energy_per_atom).energy_per_atom
        
        # Voltage scan
        Vs = np.linspace(vmin, vmax, max(2, npts))
        scan = []
        any_single_phase = False
        
        for V in Vs:
            mu = {Element(open_element): mu_li_0V - V}  # μ_Li(V) = μ_Li(0) − V  [eV/atom]
            
            try:
                gppd = GrandPotentialPhaseDiagram(entries, mu)
            except Exception as e:
                _log.warning(f"Could not build grand-potential PD at {V} V: {e}")
                continue
            
            # Decomposition at this V (grand potential space)
            # Use closed_comp (without open element) for GPPD decomposition
            try:
                decomp = gppd.get_decomposition(closed_comp)
            except Exception as e:
                _log.warning(f"GPPD decomposition failed at {V} V: {e}")
                continue
            
            phases = _format_decomp_phases(decomp)
            
            # "Single-phase stable" only if the decomposition is exactly one entry
            # AND that entry matches this composition's reduced formula.
            stable_single = False
            if len(decomp) == 1:
                only_entry = next(iter(decomp.keys()))
                if only_entry.composition.reduced_formula == comp.reduced_formula:
                    stable_single = True
                    any_single_phase = True
            
            scan.append({
                "V": float(V),
                "stable_single_phase": stable_single,
                "phases": phases,
                "note": "grand-potential decomposition at this voltage"
            })
        
        if not scan:
            return {"success": False, "error": "Voltage scan produced no valid results"}
        
        # Merge contiguous V-points with identical phase sets for compact intervals
        def phase_signature(phases):
            """Order-independent signature for phase set."""
            return tuple(sorted((p["formula"], round(p["phase_fraction"], 6)) for p in phases))
        
        intervals = []
        start = 0
        for i in range(1, len(scan) + 1):
            if i == len(scan) or phase_signature(scan[i]["phases"]) != phase_signature(scan[start]["phases"]):
                interval_phases = scan[start]["phases"]
                intervals.append({
                    "V_min": float(scan[start]["V"]),
                    "V_max": float(scan[i-1]["V"]),
                    "phases": interval_phases,
                    "stable_single_phase": all(s["stable_single_phase"] for s in scan[start:i])
                })
                start = i
        
        # Estimate capacity for the first voltage interval (initial state)
        capacity_estimate = None
        if intervals:
            first_interval_phases = intervals[0]["phases"]
            capacity_estimate = _estimate_theoretical_capacity(first_interval_phases, open_element)
        
        # Generate human-readable summary
        summary = _generate_battery_summary(
            composition, intervals, any_single_phase, vmin, vmax, open_element
        )
        
        # Friendly notes
        notes = [
            f"Grand-potential analysis with {open_element} as open species (battery-relevant).",
            f"Voltage window scanned: {vmin:.3f}–{vmax:.3f} V vs {open_element}/{open_element}+.",
            "'stable_single_phase' means an entry exists with this exact reduced formula and is a vertex in G~ space at that V.",
            "If no single-phase is stable, the listed phases are the equilibrium multiphase mixture (thermodynamic composite).",
            "Capacity estimate is upper-bound based on equilibrium; actual performance depends on kinetics and microstructure."
        ]
        
        return {
            "success": True,
            "composition": composition,
            "chemical_system": chemsys,
            "voltage_scan": scan,
            "intervals": intervals,
            "on_hull_at_any_V": any_single_phase,
            "capacity_estimate": capacity_estimate,
            "summary": summary,
            "notes": notes
        }
        
    except Exception as e:
        _log.error(f"Error in battery-relevant stability check: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _generate_battery_summary(
    composition: str,
    intervals: List[Dict[str, Any]],
    any_single_phase: bool,
    vmin: float,
    vmax: float,
    open_element: str = "Li"
) -> str:
    """
    Generate a human-readable summary of battery-relevant stability analysis.
    
    Args:
        composition: Chemical composition analyzed
        intervals: Voltage intervals with stable phases
        any_single_phase: Whether a single-phase is stable at any voltage
        vmin, vmax: Voltage range
        open_element: Working ion element
        
    Returns:
        Human-readable summary paragraph
    """
    if not intervals:
        return f"{composition}: No stable phases found in voltage range {vmin:.2f}–{vmax:.2f} V."
    
    if any_single_phase:
        # Find which interval(s) have single-phase stability
        single_phase_intervals = [iv for iv in intervals if iv["stable_single_phase"]]
        if single_phase_intervals:
            v_range = f"{single_phase_intervals[0]['V_min']:.2f}–{single_phase_intervals[-1]['V_max']:.2f} V"
            return (
                f"{composition} is thermodynamically stable as a single phase at {v_range} vs {open_element}/{open_element}+. "
                f"This material exists on the convex hull and can form as a stable anode phase in this voltage window."
            )
    
    # Multi-phase decomposition
    first_interval = intervals[0]
    phase_names = [p["formula"] for p in first_interval["phases"]]
    
    if len(phase_names) == 1:
        # Edge case: might be a single phase but not matching the exact composition
        return (
            f"{composition} decomposes to {phase_names[0]} at {vmin:.2f}–{vmax:.2f} V vs {open_element}/{open_element}+. "
            f"No single-phase entry with the exact composition {composition} exists on the hull."
        )
    
    phase_str = ", ".join(phase_names[:-1]) + f" + {phase_names[-1]}" if len(phase_names) > 1 else phase_names[0]
    
    # Check if there's a Cu or other inactive phase (common for alloy anodes)
    inactive_elements = ["Cu", "Ni", "Fe", "Co"]  # Common inactive current collectors/backbones
    has_inactive = any(any(el in phase for el in inactive_elements) for phase in phase_names)
    
    summary = (
        f"At {first_interval['V_min']:.2f}–{first_interval['V_max']:.2f} V vs {open_element}/{open_element}+, "
        f"equilibrium phases are {phase_str}. "
        f"No single-phase {composition} appears on the hull; "
        f"thermodynamically this is a multiphase composite"
    )
    
    if has_inactive:
        summary += " with an inactive backbone (e.g., Cu)"
    
    summary += "."
    
    # Add note if phases change with voltage
    if len(intervals) > 1:
        last_interval = intervals[-1]
        last_phase_names = [p["formula"] for p in last_interval["phases"]]
        if set(phase_names) != set(last_phase_names):
            last_phase_str = ", ".join(last_phase_names[:-1]) + f" + {last_phase_names[-1]}" if len(last_phase_names) > 1 else last_phase_names[0]
            summary += (
                f" Phase evolution occurs: at {last_interval['V_min']:.2f}–{last_interval['V_max']:.2f} V, "
                f"equilibrium shifts to {last_phase_str}."
            )
    
    return summary

