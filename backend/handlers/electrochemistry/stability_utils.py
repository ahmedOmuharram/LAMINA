"""
Stability analysis utilities for electrochemistry handlers.

Functions for checking composition stability and decomposition analysis.
"""
import logging
from typing import List, Dict, Any
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

