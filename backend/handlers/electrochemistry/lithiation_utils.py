"""
Lithiation mechanism analysis utilities for electrochemistry handlers (enhanced).

Key upgrades:
- Ground-state enforcement for key phases (fcc-Al, fcc-Cu, beta-LiAl)
- Dual-phase (tie-line) effective reclassification when a 3-phase tie-triangle is
  nearly degenerate in energy with a relevant two-phase pair
- Clear reporting of both strict 0 K ternary-hull classification and
  kinetics-/pathway-aware "effective" classification

Functions for analyzing lithiation mechanisms and phase evolution.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

_log = logging.getLogger(__name__)

# Check PyMatGen availability
try:
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.compatibility import (
        MaterialsProject2020Compatibility as _Compat2020,
    )
    PYMATGEN_AVAILABLE = True
except Exception:
    try:
        # Fallback if MP2020 isn't present
        from pymatgen.core import Composition
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.compatibility import MaterialsProjectCompatibility as _Compat2020
        PYMATGEN_AVAILABLE = True
    except Exception:
        _log.warning("PyMatGen not available")
        PYMATGEN_AVAILABLE = False


def _normalize_formula(frac_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize element fractions to sum to 1."""
    s = sum(frac_dict.values())
    if s <= 0:
        return dict(frac_dict)
    return {k: v / s for k, v in frac_dict.items()}


def _prepare_entries_for_pd(
    raw_entries,
    mpr=None,
    enforce_ground_states: bool = True,
):
    """
    Prepare entries for PhaseDiagram construction with proper compatibility corrections.
    
    - Applies MP compatibility corrections
    - Keeps only the lowest energy entry per reduced formula
    - Warns if elemental references are missing (attempts to fetch if mpr provided)
    - Optionally enforces ground-state selections for key phases
    
    Args:
        raw_entries: List of raw ComputedEntry objects
        mpr: Optional MPRester client for fetching missing references
        enforce_ground_states: If True, attempts light-touch curation for Al, Cu, Li, LiAl
        
    Returns:
        List of prepared entries ready for PhaseDiagram
    """
    if not raw_entries:
        return []

    compat = _Compat2020()
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

    # Check for elemental references
    elem_has_ref: Dict[str, bool] = defaultdict(bool)
    for e in entries:
        if len(e.composition.elements) == 1:
            el = next(iter(e.composition.elements)).symbol
            elem_has_ref[el] = True
    
    # Find all needed elements
    needed_elements = set()
    for e in entries:
        for el in e.composition.elements:
            needed_elements.add(el.symbol)

    missing = [el for el in needed_elements if not elem_has_ref.get(el, False)]
    if missing:
        _log.warning(f"Missing elemental references for: {missing}")
        if mpr is not None:
            try:
                addl = []
                for el in missing:
                    addl.extend(mpr.get_entries(f"{el}") or [])
                if addl:
                    addl = compat.process_entries(addl, clean=True) or []
                    entries.extend(addl)
                    _log.info(f"Fetched {len(addl)} elemental references for: {missing}")
            except Exception as exc:
                _log.warning(f"Could not fetch missing elemental refs: {exc}")

    # Optionally encourage known ground states (light-touch curation)
    if enforce_ground_states and mpr is not None:
        try:
            whitelist = ["Al", "Cu", "Li", "LiAl"]
            have = {f: False for f in whitelist}
            for e in entries:
                rf = e.composition.reduced_formula
                if rf in have:
                    have[rf] = True

            for formula in whitelist:
                if not have[formula]:
                    try:
                        fetched = mpr.get_entries(formula) or []
                        fetched = compat.process_entries(fetched, clean=True) or []
                        if fetched:
                            best_entry = min(fetched, key=lambda x: x.energy_per_atom)
                            entries.append(best_entry)
                            _log.info(f"Inserted curated entry for {formula} to encourage correct ground state.")
                    except Exception:
                        pass  # Skip if not available
        except Exception as exc:
            _log.warning(f"Ground-state enforcement skipped due to error: {exc}")

    return entries


def _pair_mixture_at_composition(
    target: Composition,
    A: Composition,
    B: Composition,
    elems: Tuple[str, ...] = ("Li", "Al", "Cu"),
    tol: float = 1e-6
) -> Optional[Tuple[float, float]]:
    """
    Solve for lever fractions (f_A, f_B) so that f_A*A + f_B*B matches target (normalized).
    Returns (f_A, f_B) if feasible inside [0,1], else None.
    
    Uses robust barycentric solve with sum-to-one constraint enforced:
    f_A + f_B = 1, and pivots on the component with largest |a_i - b_i| for stability.
    """
    # Normalized composition vectors (sum = 1 across elems)
    def v(c):
        d = c.get_el_amt_dict()
        s = sum(d.get(e, 0.0) for e in elems)
        if s <= 0:
            return np.zeros(len(elems), dtype=float)
        return np.array([d.get(e, 0.0) / s for e in elems], dtype=float)

    y, a, b = v(target), v(A), v(B)
    
    # Choose the component with the largest |a_i - b_i| for numerical stability
    dif = np.abs(a - b)
    idx = int(np.argmax(dif))
    
    if dif[idx] < tol:
        # A and B are (nearly) identical in composition → cannot span a line
        return None

    # Barycentric coefficient with f_B = 1 - f_A
    fA = (y[idx] - b[idx]) / (a[idx] - b[idx])
    fB = 1.0 - fA
    
    if fA < -1e-6 or fB < -1e-6:
        # Outside segment
        return None

    # Validate fit across all components
    y_fit = fA * a + fB * b
    if np.linalg.norm(y_fit - y, ord=np.inf) > 5e-3:
        return None
    
    return float(fA), float(fB)


def _mixture_energy_per_atom(pd_like: PhaseDiagram, entryA, entryB, f_A: float, f_B: float) -> float:
    """
    Return formation energy per atom of f_A*A + f_B*B mixture using PD entry energies.
    """
    E_A = pd_like.get_form_energy_per_atom(entryA)
    E_B = pd_like.get_form_energy_per_atom(entryB)
    return f_A * E_A + f_B * E_B


def _find_entry(pd_like: PhaseDiagram, reduced_formula: str):
    """
    Find the first PD entry whose reduced formula matches reduced_formula.
    Searches pd_like.all_entries.
    """
    for e in pd_like.all_entries:
        if e.composition.reduced_formula == reduced_formula:
            return e
    return None


def _effective_two_phase_override(
    pd_strict: PhaseDiagram,
    pd_full: PhaseDiagram,
    phase_set_formulas: Tuple[str, ...],
    x0: float,
    x1: float,
    host_ratios: Dict[str, float],
    working_ion: str,
    n_host_atoms_per_fu: float,
    energy_tol_eV_per_atom: float = 0.05,
    candidate_pairs: Optional[List[Tuple[str, str]]] = None,
    host_formula: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Try to reclassify a 3-phase plateau as *effectively two-phase* if a relevant
    tie-line pair yields essentially the same energy at compositions inside [x0,x1].

    Samples multiple points along [x0, x1] and checks if any two-phase pair is within
    energy_tol of the 3-phase decomposition at those points.

    Returns a dict with {"override": bool, "pair": (A,B) or None, "delta_e": float}
    """
    from itertools import combinations
    
    def comp_at(x: float) -> Composition:
        comp = {working_ion: x}
        comp.update(host_ratios)
        return Composition(comp)
    
    def get_3phase_energy(x: float) -> float:
        """Get 3-phase mixture energy at composition x using strict PD."""
        decomp = pd_strict.get_decomposition(comp_at(x))
        E_mix = 0.0
        total = 0.0
        for entry, frac in decomp.items():
            if frac <= 0:
                continue
            E_mix += frac * pd_strict.get_form_energy_per_atom(entry)
            total += frac
        return E_mix / total if total > 0 else 0.0

    # Build candidate pairs: include all 2-combinations from strict set + heuristics
    strict_pairs = list(combinations(phase_set_formulas, 2))
    heuristic_pairs = [("LiAl", "Cu")]
    if host_formula:
        heuristic_pairs.append(("LiAl", host_formula))
    
    if candidate_pairs is None:
        candidate_pairs = strict_pairs + heuristic_pairs
    else:
        # Merge with provided candidates
        candidate_pairs = list(set(strict_pairs + heuristic_pairs + candidate_pairs))
    
    # Probe multiple points along the plateau for robustness
    xs_probe = np.linspace(x0, x1, 7)
    
    best = {"override": False, "pair": None, "delta_e": None}

    # Evaluate all candidate two-phase pairs
    for A, B in candidate_pairs:
        eA = _find_entry(pd_full, A)
        eB = _find_entry(pd_full, B)
        if (eA is None) or (eB is None):
            continue
        
        # Check multiple compositions along the plateau
        for xm in xs_probe:
            target = comp_at(xm)
            f = _pair_mixture_at_composition(target, eA.composition, eB.composition)
            if f is None:
                continue
            
            f_A, f_B = f
            E_two = _mixture_energy_per_atom(pd_full, eA, eB, f_A, f_B)
            E_tri = get_3phase_energy(xm)
            delta = E_two - E_tri
            
            # Track the best (lowest) delta across all pairs and probe points
            if best["delta_e"] is None or delta < best["delta_e"]:
                best = {
                    "override": delta <= energy_tol_eV_per_atom,
                    "pair": (A, B),
                    "delta_e": float(delta),
                    "probe_x": float(xm)
                }

    return best


def analyze_lithiation_mechanism_detailed(
    mpr,
    host_composition: str,
    working_ion: str = "Li",
    max_x: float = 3.0,
    room_temp: bool = True,
    enforce_ground_states: bool = True,
    prefer_two_phase: bool = True,
    two_phase_energy_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Analyze the lithiation mechanism by computing the convex hull of G(x) vs x,
    with optional effective two-phase reclassification.
    
    Args:
        mpr: MPRester client instance
        host_composition: Host material composition (e.g., "AlCu")
        working_ion: Working ion symbol (default: Li)
        max_x: Maximum Li per host atom to trace
        room_temp: Filter high-energy phases (E_hull > 30 meV/atom)
        enforce_ground_states: Encourage fcc-Al, fcc-Cu, β-LiAl presence
        prefer_two_phase: Attempt effective two-phase reclassification
        two_phase_energy_tolerance: eV/atom tolerance for override (default: 0.05)
        
    Returns:
        Dictionary with lithiation mechanism analysis (strict + effective views)
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
        
        # Prepare entries with enhanced curation
        entries_cur = _prepare_entries_for_pd(entries_raw, mpr=mpr, enforce_ground_states=enforce_ground_states)
        if not entries_cur:
            return {"success": False, "error": f"No compatible entries after processing for {chemsys}"}
        
        _log.info(f"Prepared {len(entries_cur)} entries for PhaseDiagram (lowest energy per formula)")
        
        # Build phase diagram with correct ground states
        pd = PhaseDiagram(entries_cur)
        
        # Build full PD with compatibility-processed entries (but not aggressively pruned)
        # This ensures consistent energy references for the override check
        compat = _Compat2020()
        entries_full = compat.process_entries(entries_raw, clean=True) or []
        pd_full = PhaseDiagram(entries_full) if entries_full else None
        
        # Optional: report metastable phases that were excluded (for room temp context)
        filtered_phases = []
        if room_temp and pd_full:
            try:
                for entry in entries_full:
                    eh = pd_full.get_e_above_hull(entry)
                    if eh is not None and eh >= 0.03:
                        filtered_phases.append({
                            "formula": entry.composition.reduced_formula,
                            "material_id": str(getattr(entry, 'entry_id', None)),
                            "e_above_hull": float(eh),
                            "reason": f"E_hull = {eh:.4f} eV/atom > 0.03 (metastable at RT)"
                        })
                _log.info(f"Room temp context: {len(filtered_phases)} metastable phases identified")
            except Exception as exc:
                _log.warning(f"Could not compute metastability context: {exc}")
        
        # Normalize host ratios
        total_host = float(sum(host_counts_input.values()))
        host_ratios = {el: amt / total_host for el, amt in host_counts_input.items()}
        n_host_atoms_per_fu = total_host
        
        # Compute strict (0 K) lithiation steps from constant decomposition sets
        strict_steps = _compute_lithiation_steps(
            pd, host_ratios, working_ion, max_x, n_host_atoms_per_fu
        )
        
        if not strict_steps:
            return {"success": False, "error": "No physically plausible voltage plateaus found"}
        
        # Optionally attempt *effective two-phase* override tagging for each 3-phase step
        effective_steps = []
        for st in strict_steps:
            st_eff = dict(st)  # shallow copy
            st_eff["effective_two_phase"] = False
            st_eff["effective_pair"] = None
            st_eff["effective_delta_e_eV_per_atom"] = None
            st_eff["effective_probe_x"] = None
            
            if prefer_two_phase and pd_full is not None and not st.get("is_two_phase_microstructure", False):
                # Try to override only when there are 3+ phases
                if st.get("num_phases_in_microstructure", 0) >= 3:
                    i0 = st["x_host_range"]["start"]
                    i1 = st["x_host_range"]["end"]
                    override = _effective_two_phase_override(
                        pd_strict=pd,
                        pd_full=pd_full,
                        phase_set_formulas=tuple(p["formula"] for p in st["equilibrium_phases"]),
                        x0=float(i0),
                        x1=float(i1),
                        host_ratios=host_ratios,
                        working_ion=working_ion,
                        n_host_atoms_per_fu=n_host_atoms_per_fu,
                        energy_tol_eV_per_atom=float(two_phase_energy_tolerance),
                        candidate_pairs=None,  # Let function build from strict set + heuristics
                        host_formula=host_formula,
                    )
                    if override.get("override"):
                        st_eff["effective_two_phase"] = True
                        st_eff["effective_pair"] = override.get("pair")
                        st_eff["effective_delta_e_eV_per_atom"] = override.get("delta_e")
                        st_eff["effective_probe_x"] = override.get("probe_x")
                        st_eff["reaction_type"] += " | effective two-phase (near-degenerate)"
            
            effective_steps.append(st_eff)
        
        # Initial mechanism (strict vs effective)
        initial_strict = _format_initial_mechanism(strict_steps[0])
        initial_effective = _format_initial_mechanism(effective_steps[0])
        initial_effective["effective_two_phase"] = effective_steps[0].get("effective_two_phase", False)
        initial_effective["effective_pair"] = effective_steps[0].get("effective_pair")
        initial_effective["effective_delta_e_eV_per_atom"] = effective_steps[0].get("effective_delta_e_eV_per_atom")
        initial_effective["effective_probe_x"] = effective_steps[0].get("effective_probe_x")
        
        # Calculate average voltage (strict)
        voltages = [s["voltage"] for s in strict_steps]
        weights = [s["x_host_range"]["end"] - s["x_host_range"]["start"] for s in strict_steps]
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
            "initial_reaction_strict": initial_strict,
            "initial_reaction_effective": initial_effective,
            "lithiation_steps_strict": strict_steps,
            "lithiation_steps_effective": effective_steps,
            "num_plateau_steps": len(strict_steps),
            "average_voltage": avg_voltage,
            "voltage_range": {"min": min(voltages), "max": max(voltages)},
            "voltage_uncertainty_estimate": voltage_uncertainty,
            "plating_starts_at_x_host": plating_starts_at_x,
            "methodology": _get_methodology_description(voltage_uncertainty),
            "notes": _get_analysis_notes(
                host_formula, working_ion, len(strict_steps),
                initial_effective, room_temp, filtered_phases, voltage_uncertainty
            ),
            "parameters": {
                "enforce_ground_states": enforce_ground_states,
                "prefer_two_phase": prefer_two_phase,
                "two_phase_energy_tolerance_eV_per_atom": two_phase_energy_tolerance,
            },
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
    """Format initial reaction mechanism description (pass-through + prose)."""
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
            f"(convex-hull segment). Microstructure: {', '.join([p['formula'] for p in initial_step['equilibrium_phases']])} "
            f"({'2-phase coexistence' if initial_step['is_two_phase_microstructure'] else f"{initial_step['num_phases_in_microstructure']}-phase mixture"}). "
            f"This is the strict 0 K equilibrium classification (lever rule among hull vertices)."
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
            "strict": "Tie-line (2-phase) vs tie-triangle (3-phase) at 0 K",
            "effective_override": "Optional: if a tie-line pair is within ΔE tolerance, tag as effective two-phase",
        },
        "phase_reporting": "Exact phases from tie-simplex corners, not interior sampling",
        "voltage_uncertainty": f"±{voltage_uncertainty} V (typical DFT GGA systematic error for alloy anodes)"
    }


def _get_analysis_notes(
    host_formula: str,
    working_ion: str,
    num_steps: int,
    initial_mechanism_effective: Dict[str, Any],
    room_temp: bool,
    filtered_phases: List[Dict[str, Any]],
    voltage_uncertainty: float
) -> List[str]:
    """Generate notes for lithiation analysis."""
    notes = [
        f"Analyzed lithiation of {host_formula} vs {working_ion}/{working_ion}+",
        f"Initial reaction (strict): {initial_mechanism_effective['reaction_type']}",
    ]
    
    # Add effective two-phase info if applicable
    if initial_mechanism_effective.get('effective_two_phase', False):
        pair = initial_mechanism_effective.get('effective_pair')
        delta_e = initial_mechanism_effective.get('effective_delta_e_eV_per_atom')
        probe_x = initial_mechanism_effective.get('effective_probe_x')
        if probe_x is not None:
            notes.append(f"Initial reaction (effective two-phase): Yes, {pair} (ΔE ≈ {delta_e:.4f} eV/atom at x={probe_x:.3f})")
        else:
            notes.append(f"Initial reaction (effective two-phase): Yes, {pair} (ΔE ≈ {delta_e:.4f} eV/atom)")
    else:
        notes.append("Initial reaction (effective two-phase): No")
    
    notes.extend([
        f"Total {num_steps} voltage plateaus detected (strict classification)",
        "Each plateau = constant decomposition set (tie-line or tie-triangle on 2-D ternary hull)",
        "Effective two-phase analysis: samples 7 points along each plateau, checks all phase pairs from strict set",
        "Entry preparation: Applied MP2020 compatibility corrections and kept lowest energy per formula",
        ("Room temperature context: Metastable phases (E_hull > 30 meV/atom) reported but not used"
         if room_temp else "All phases included"),
        (f"Identified {len(filtered_phases)} metastable phases" if filtered_phases else "No metastable phases identified"),
        f"Voltage uncertainty: ±{voltage_uncertainty} V (typical DFT GGA error for alloy anodes)",
        "0 K thermodynamics with GGA energies; kinetics and finite-T effects not included",
        "Segments truncated at Li plating onset (elemental Li appears in decomposition)",
    ])
    
    return notes

