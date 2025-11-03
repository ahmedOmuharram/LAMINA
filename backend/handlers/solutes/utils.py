"""
lattice_effects.py / utils.py

Tools for evaluating how substitutional solute atoms change the lattice
parameter of an fcc matrix in the dilute limit.

These are meant to support claims like:
- "Mg in Al causes the largest lattice expansion"
- "Cu in Al causes a moderate increase" (we can prove if that's wrong)

Physics used:
- fcc geometry: a = 2*sqrt(2)*r  [standard close-packed sphere model for fcc] [1]
- Vegard-like linearization in dilute limit: Δa/a0 ~ x * δ_size  [2]
- δ_size = (r_solute - r_matrix)/r_matrix  (substitutional size misfit) [3]
- Hume-Rothery size mismatch relevance for solubility thresholds (~15%) [4]

[1] basic crystallography of fcc nearest-neighbor packing.
[2] Vegard's law approximation for dilute substitutional alloys.
[3] Classic substitutional misfit parameter used in solid-solution strengthening models.
[4] Hume-Rothery rules for substitutional solid solutions.

We also include:
- Lattice parameter of pure Al at ~300 K: a_Al ≈ 4.046 Å [5]
- Effective "metallic radii" (12-fold coordination / close-packed metallic radius):
    Al ~143 pm, Mg ~160 pm, Cu ~128 pm, Zn ~134 pm. These values are widely
    tabulated for metallic radii in close-packed environments and are
    consistent with using a = 2*sqrt(2)*r for fcc Al giving r_Al ~143 pm. [5][6]

[5] Room-temperature lattice constant of fcc Al is ~4.046 Å; converting via
    r = a/(2√2) gives r_Al ≈ 1.43 Å = 143 pm.
    (Values in this range are reported in crystallographic data for Al.)
[6] Typical metallic radii tables for Mg (~160 pm), Cu (~128 pm),
    Zn (~134 pm), Al (~143 pm), quoted for 12-fold coordination / metallic bonding.

IMPORTANT:
- This model only makes sense if the solute is actually dissolved
  substitutionally in the matrix fcc phase at the temperature of interest.
- You should confirm that separately with CALPHAD (you already have that)
  and then pass that info into these functions.
"""

from math import sqrt
from typing import Dict, Any, List
import logging

_log = logging.getLogger(__name__)


# --- Hardcoded reference data with citations in comments ---
# Import constants from centralized location
from ..shared.constants import FCC_LATTICE_PARAMS_A, METALLIC_RADII_PM


def _fcc_radius_from_lattice(a_angstrom: float) -> float:
    """
    Convert fcc lattice parameter 'a' [Å] to effective atomic radius r [pm].

    Geometry of fcc: atoms touch along the face diagonal.
    => a = 2*sqrt(2)*r  => r = a / (2*sqrt(2))
    Units:
        a in Å
        r in Å
    We'll return pm because our metallic radii table is pm.

    [1] crystallography of fcc packing
    """
    r_angstrom = a_angstrom / (2.0 * sqrt(2.0))
    r_pm = r_angstrom * 100.0  # 1 Å = 100 pm
    return r_pm


def compute_substitutional_lattice_effect(
    matrix_element: str,
    solute_element: str,
    solute_atpct: float,
    temperature_K: float,
    matrix_phase_name: str,
    matrix_phase_composition: Dict[str, float],
    min_required_solute_in_matrix_atpct: float = 0.1,
) -> Dict[str, Any]:
    """
    Core physics tool:
    Estimate how much adding `solute_element` (at `solute_atpct` atomic % overall)
    would change the lattice parameter of an fcc matrix `matrix_element`.

    We assume dilute substitutional solution in fcc_A1 and Vegard-like behavior.

    INPUTS
    -------
    matrix_element : e.g. "Al"
    solute_element : e.g. "Mg"
    solute_atpct   : total alloy loading in at.% (e.g. 1.0 for 1 at.% Mg in Al)
    temperature_K  : temperature of interest (e.g. 300 K)
    matrix_phase_name : name of the matrix phase from CALPHAD
                        (must be "FCC_A1" or we'll refuse)
    matrix_phase_composition : dict of element mole fractions INSIDE the matrix phase,
                               from your CALPHAD output.
                               Example:
                               {
                                   "AL": 0.9948,
                                   "MG": 0.00517
                               }
                               This tells us if the solute is actually present.
    min_required_solute_in_matrix_atpct : threshold (at.%) of solute actually dissolved
                                          in the matrix phase below which we say
                                          "not really in solid solution".

    RETURNS
    --------
    dict with:
      - ok (bool): do we consider the estimate physically applicable?
      - reason (str): why / why not
      - a0_matrix_A (float): lattice param of pure matrix [Å]
      - r_matrix_pm (float): effective matrix radius [pm]
      - r_solute_pm (float): metallic radius of solute [pm]
      - size_misfit_fraction (float): δ_size = (r_solute - r_matrix)/r_matrix
      - predicted_da_over_a (float): Δa/a0 (fraction, dimensionless) for `solute_atpct`
      - predicted_percent_change (float): 100 * predicted_da_over_a
      - classification (str): "expands_lattice (large/moderate/negligible)" etc.

    SCIENCE STEPS
    -------------
    1. Confirm fcc_A1 matrix actually exists and contains non-zero solute.
       If not, we bail: "not applicable at this temperature."
    2. Get base lattice parameter a0 for the pure matrix element (from FCC_LATTICE_PARAMS_A). [5]
    3. Get effective metallic radii for matrix and solute (from METALLIC_RADII_PM). [5][6]
    4. Compute size misfit δ_size = (r_solute - r_matrix)/r_matrix. [3]
    5. Vegard-like linearization:
           Δa/a0 ≈ δ_size * (solute_atpct / 100)
       (dilute limit) [2]
    6. Classify magnitude for human language.


    IMPORTANT:
    - We do NOT numerically use matrix_phase_composition["solute"] inside Δa/a0.
      We only use it as a gate to make sure solute is actually dissolving.
      The predictive scaling is done with 'solute_atpct', so you can ask
      "what if I add 1 at.% Mg?" consistently.
    """

    # Normalize symbols
    mat = matrix_element.strip().upper()
    sol = solute_element.strip().upper()

    # 1. sanity checks
    if matrix_phase_name.upper() != "FCC_A1":
        return {
            "ok": False,
            "reason": f"Matrix phase is {matrix_phase_name}, not FCC_A1. "
                      f"Model only applies to substitutional fcc solutions.",
        }

    if mat not in FCC_LATTICE_PARAMS_A:
        return {
            "ok": False,
            "reason": f"No reference fcc lattice parameter for matrix {mat} in FCC_LATTICE_PARAMS_A.",
        }

    if mat not in METALLIC_RADII_PM:
        return {
            "ok": False,
            "reason": f"No metallic radius for matrix {mat} in METALLIC_RADII_PM.",
        }

    if sol not in METALLIC_RADII_PM:
        return {
            "ok": False,
            "reason": f"No metallic radius for solute {sol} in METALLIC_RADII_PM.",
        }

    # Check that solute is actually present in the matrix phase at all.
    # matrix_phase_composition is mole fraction in that phase (e.g. 0.005 Mg in FCC_A1).
    solute_in_matrix_x = matrix_phase_composition.get(sol, 0.0)
    solute_in_matrix_atpct = solute_in_matrix_x * 100.0

    if solute_in_matrix_atpct < min_required_solute_in_matrix_atpct:
        return {
            "ok": False,
            "reason": (
                f"CALPHAD says only {solute_in_matrix_atpct:.3f} at.% {sol} "
                f"in FCC_A1 at {temperature_K} K, below threshold "
                f"{min_required_solute_in_matrix_atpct} at.%. "
                "So essentially no substitutional solubility in the matrix "
                "at this condition → can't claim a lattice-parameter effect."
            ),
            "solute_in_matrix_atpct": solute_in_matrix_atpct,
        }

    # 2. base lattice parameter for pure matrix
    a0 = FCC_LATTICE_PARAMS_A[mat]  # [Å], ~room-temp [5]

    # 3. radii (pm)
    r_matrix_pm = METALLIC_RADII_PM[mat]  # ~143 pm for Al [5][6]
    r_solute_pm = METALLIC_RADII_PM[sol]  # e.g. Mg ~160 pm [6]

    # sanity cross-check: we *could* recompute r_matrix from a0 to enforce consistency:
    r_matrix_from_a0_pm = _fcc_radius_from_lattice(a0)
    # If you want to be strict, you could average these two or assert they're close.
    # We'll just keep the tabulated r_matrix_pm as the canonical radius in misfit,
    # but note the delta:
    radius_consistency_error = abs(r_matrix_from_a0_pm - r_matrix_pm) / r_matrix_pm

    # 4. size misfit δ_size
    delta_size = (r_solute_pm - r_matrix_pm) / r_matrix_pm  # dimensionless fraction [3]

    # 5. Vegard-like Δa/a0 for the requested solute_atpct
    x = solute_atpct / 100.0  # convert at.% → fraction
    predicted_da_over_a = delta_size * x  # dimensionless [2]
    predicted_percent_change = predicted_da_over_a * 100.0  # %

    # 6. qualitative classification
    # magnitude buckets (tunable heuristic)
    mag = abs(predicted_percent_change)
    if mag < 0.1:
        mag_label = "negligible"
    elif mag < 0.3:
        mag_label = "moderate"
    else:
        mag_label = "large"

    if predicted_da_over_a > 0:
        effect_label = f"expands_lattice ({mag_label})"
    elif predicted_da_over_a < 0:
        effect_label = f"contracts_lattice ({mag_label})"
    else:
        effect_label = "no_change"

    return {
        "ok": True,
        "reason": "substitutional fcc solution appears valid at this T; Vegard estimate applied",
        "matrix": mat,
        "solute": sol,
        "temperature_K": temperature_K,
        "solute_atpct_nominal": solute_atpct,
        "solute_in_matrix_atpct_CALPHAD": solute_in_matrix_atpct,
        "a0_matrix_A": a0,
        "r_matrix_pm": r_matrix_pm,
        "r_matrix_from_a0_pm": r_matrix_from_a0_pm,
        "radii_consistency_fractional_error": radius_consistency_error,
        "r_solute_pm": r_solute_pm,
        "size_misfit_fraction": delta_size,
        "size_misfit_percent": delta_size * 100.0,
        "predicted_da_over_a": predicted_da_over_a,
        "predicted_percent_change": predicted_percent_change,
        "classification": effect_label,
        "hume_rothery_size_mismatch_percent": abs(delta_size) * 100.0,
        "notes": (
            "Δa/a0 ≈ δ_size * x, dilute Vegard limit. "
            "Use only for small at.% solute. "
            "Hume-Rothery says >~15% mismatch hurts solubility; "
            "we report |δ_size|*100 as a proxy."
        ),
    }


def rank_solutes_by_expansion(
    matrix_element: str,
    solute_elements: List[str],
    solute_atpct: float,
    temperature_K: float,
    calphad_matrix_phase_name: str,
    calphad_matrix_phase_compositions: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compare multiple solutes (e.g. ["Mg", "Cu", "Zn"]) in the SAME fcc matrix (e.g. "Al").

    We:
      1. Call compute_substitutional_lattice_effect(...) for each solute.
         You must provide CALPHAD-derived matrix phase composition for each solute
         (because at equilibrium, FCC_A1(Al) with Mg might have ~0.5 at.% Mg
         dissolved, but FCC_A1(Al) with Cu might have basically zero Cu dissolved,
         etc.).

         calphad_matrix_phase_compositions should look like:
         {
            "MG": {"AL": 0.995, "MG": 0.005},
            "CU": {"AL": 0.999, "CU": 0.001},
            "ZN": {"AL": 0.998, "ZN": 0.002}
         }
         These are per-solute runs of CALPHAD at the SAME temperature_K and
         nominal solute_atpct, extracted from your equilibrium tool.

      2. Rank solutes by predicted_da_over_a (largest positive first).
         Positive means lattice expansion; negative means contraction.

    RETURNS
    -------
    dict:
      - results: list of per-solute dicts (from compute_substitutional_lattice_effect)
      - sorted_by_expansion: solute symbols sorted by predicted_da_over_a descending
      - largest_expander: first entry in that sorted list (if any)
      - commentary: human-readable summary of the lattice effects
    """

    per_solute_results = []
    for sol in solute_elements:
        comp_in_matrix = calphad_matrix_phase_compositions.get(sol.upper(), {})
        res = compute_substitutional_lattice_effect(
            matrix_element=matrix_element,
            solute_element=sol,
            solute_atpct=solute_atpct,
            temperature_K=temperature_K,
            matrix_phase_name=calphad_matrix_phase_name,
            matrix_phase_composition=comp_in_matrix,
        )
        res["solute"] = sol.upper()
        per_solute_results.append(res)

    # Only rank the ones that are "ok"
    ok_results = [r for r in per_solute_results if r.get("ok", False)]
    ok_results_sorted = sorted(
        ok_results,
        key=lambda r: r["predicted_da_over_a"],
        reverse=True  # largest positive expansion first
    )

    if ok_results_sorted:
        largest = ok_results_sorted[0]["solute"]
    else:
        largest = None

    # build commentary
    lines = []
    lines.append(f"Matrix: {matrix_element.upper()}, nominal {solute_atpct:.2f} at.% solute, T={temperature_K} K.")
    for r in per_solute_results:
        if not r.get("ok", False):
            lines.append(
                f"- {r['solute']}: not applicable ({r['reason']})"
            )
            continue

        da_pct = r["predicted_percent_change"]
        misfit_pct = r["size_misfit_percent"]
        lines.append(
            f"- {r['solute']}: {r['classification']}, "
            f"Δa/a0 ≈ {da_pct:+.3f}%, size misfit ~{misfit_pct:.1f}%."
        )

    if largest:
        lines.append(f"=> Largest positive lattice expansion among valid solutes: {largest}")
    else:
        lines.append("=> No valid solutes produced a positive lattice expansion under these conditions.")

    return {
        "results": per_solute_results,
        "sorted_by_expansion": [r["solute"] for r in ok_results_sorted],
        "largest_expander": largest,
        "commentary": "\n".join(lines),
    }

