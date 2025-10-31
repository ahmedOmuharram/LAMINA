"""
Utility functions for superconductor analysis.

Provides structure analysis for cuprate superconductors, focusing on
octahedral distortions and c-axis spacing effects.

CRITICAL: Analysis is FAMILY-AWARE. The c-axis/apical-oxygen correlation
varies dramatically between cuprate families:
    - 214 (La₂CuO₄): c tracks apical O; increasing c stabilizes octahedra
    - 123 (YBCO): c tracks CHAIN O; increasing c means chain depletion (worse SC)
    - T′: No apical O by design; c tracks interstitial O reorganization
    - Infinite-layer: No apical O; octahedral stability not applicable
"""
import logging
from typing import Dict, Any, Optional

try:
    from pymatgen.core import Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

from ..shared.constants import CUPRATE_DATA, FAMILY_C_AXIS_RULES

_log = logging.getLogger(__name__)


def _detect_cuprate_family(material_formula: str) -> Optional[str]:
    """
    Detect cuprate family from formula using composition-aware parsing.
    Handles doped formulas like La2-xSrxCuO4, Nd1.85Ce0.15CuO4, YBa2Cu3O7-δ.
    
    Returns:
        family code: "214", "123", "Bi-22n", "T_prime", "infinite_layer", etc.
        None if not recognized
    """
    formula_clean = material_formula.strip().lower().replace(" ", "")
    
    # First check against known cuprates (exact/substring match)
    for key, info in CUPRATE_DATA.items():
        if key.lower() in formula_clean or formula_clean in key.lower():
            return info.get("family")
    
    # Abbrev fallback
    if "lsco" in formula_clean:
        return "214"
    if "ybco" in formula_clean:
        return "123"
    if "ncco" in formula_clean:
        return "T_prime"
    
    # Composition-based detection if pymatgen available
    # Skip if formula contains variables (x, δ, ±) - fall back to substring matching
    has_variables = any(sym in formula_clean for sym in ["-x", "x", "±", "delta", "δ"])
    
    if HAS_PYMATGEN and not has_variables:
        try:
            # Try to parse as Composition directly
            formula_normalized = material_formula.strip()
            
            # Try to parse as Composition
            comp = Composition(formula_normalized)
            elements = set(comp.elements)
            element_syms = {str(e) for e in elements}
            
            # Reduced composition for ratio analysis
            comp_reduced = comp.reduced_composition
            
            # 214-type: A₂CuO₄ (K₂NiF₄ structure)
            # A = La, Sr, Ba, Nd, Pr, etc.
            if 'Cu' in element_syms and 'O' in element_syms:
                A_elements = element_syms - {'Cu', 'O', 'Ce', 'F'}  # Exclude dopants/anions
                if len(A_elements) >= 1:
                    # Check stoichiometry ratios
                    cu_amt = comp_reduced.get('Cu', 0)
                    o_amt = comp_reduced.get('O', 0)
                    
                    # 214-type: A₂CuO₄ → reduced A:Cu:O ~ 2:1:4 or 1:0.5:2
                    # Allow some flexibility for doping
                    if 0.4 <= cu_amt <= 0.6 and 1.5 <= o_amt <= 2.5:
                        # Further check for La, Nd, Pr (common 214 A-sites)
                        if any(el in element_syms for el in ['La', 'Sr', 'Ba', 'Nd', 'Pr', 'Sm', 'Eu', 'Gd']):
                            # Check for T′ markers (electron-doped, Ce presence)
                            if 'Ce' in element_syms or ('Nd' in element_syms and 'La' not in element_syms and 'Sr' not in element_syms):
                                return "T_prime"
                            return "214"
                    
                    # 123-type: YBa₂Cu₃O₇ → Y:Ba:Cu ~ 1:2:3
                    if 'Y' in element_syms or 'Ba' in element_syms:
                        ba_amt = comp_reduced.get('Ba', 0)
                        y_amt = comp_reduced.get('Y', 0)
                        # Rough ratio check: Ba/Cu ~ 2/3, Y/Cu ~ 1/3
                        if 2.5 <= cu_amt <= 3.5:
                            if (0.8 <= ba_amt <= 2.5) and (0.3 <= y_amt <= 1.2):
                                return "123"
                    
                    # Infinite-layer: ACuO₂ → A:Cu:O ~ 1:1:2
                    if 0.8 <= cu_amt <= 1.2 and 1.5 <= o_amt <= 2.5:
                        if any(el in element_syms for el in ['Ca', 'Sr']) and len(element_syms) <= 4:
                            return "infinite_layer"
                    
                    # Bi/Tl/Hg families
                    if 'Bi' in element_syms and 'Sr' in element_syms:
                        return "Bi-22n"
                    if 'Tl' in element_syms and 'Ba' in element_syms:
                        return "Tl-22n"
                    if 'Hg' in element_syms and 'Ba' in element_syms:
                        return "Hg-12n"
        
        except Exception as e:
            _log.debug(f"Composition parsing failed for {material_formula}: {e}")
            pass
    
    # Pattern-based fallback (substring)
    if "la2cuo4" in formula_clean:
        return "214"
    if "yba2cu3" in formula_clean:
        return "123"
    if "bi2sr2" in formula_clean:
        return "Bi-22n"
    if "tl2ba2" in formula_clean:
        return "Tl-22n"
    if "hgba2" in formula_clean:
        return "Hg-12n"
    if "nd2cuo4" in formula_clean:
        return "T_prime"
    if "cacuo2" in formula_clean:
        return "infinite_layer"
    
    return None


def analyze_cuprate_octahedral_stability(
    material_formula: str,
    c_axis_spacing: Optional[float] = None,
    structure_data: Optional[Dict[str, Any]] = None,
    scenario: str = "trend_increase",  # "observed" | "trend_increase" | "trend_decrease"
    trend_probe: float = 0.01,         # ±1% hypothetical change for trend mode
    threshold: Optional[float] = None,  # Auto: 0.02 for MP/unknown, 0.01 for experiment, 0.005 for high-precision
    data_source: str = "unknown",       # "MP" | "experiment" | "high_precision_xrd" | "unknown"
) -> Dict[str, Any]:
    """
    Analyze how c-axis spacing affects Cu-O octahedral stability in cuprates.
    
    FAMILY-AWARE: Returns family-specific verdicts. YBCO and infinite-layer have
    special handling since their c-axis does NOT track apical oxygen.
    
    Args:
        material_formula: Chemical formula (e.g., "La2CuO4", "YBa2Cu3O7", "La1.85Sr0.15CuO4")
        c_axis_spacing: c-axis lattice parameter in Å (optional)
        structure_data: Structure information from Materials Project (optional)
        scenario: "trend_increase" | "trend_decrease" | "observed"
        trend_probe: Fractional change for trend mode (default 0.01 = 1%)
        threshold: Relative change threshold (optional, auto-selected by data_source if None):
            - MP/unknown: 0.02 (2%, accounts for DFT errors)
            - experiment: 0.01 (1%, typical XRD precision)
            - high_precision_xrd: 0.005 (0.5%, for high-quality single-crystal data)
        data_source: "MP" | "experiment" | "high_precision_xrd" | "unknown"
    
    Returns:
        Dictionary with family-aware stability analysis, unified schema:
        - claim: str (the question being asked, always about increasing c)
        - verdict: "TRUE" | "FALSE" | "AMBIGUOUS" | "NOT_APPLICABLE"
        - stability_effect: "stabilized" | "destabilized" | "minimal_change" | etc.
        - family: str (detected family code)
        - threshold_used_percent: float (classification threshold used)
        - citations: List[str] (family-specific references)
    """
    try:
        # Auto-select threshold based on data source
        if threshold is None:
            if data_source == "high_precision_xrd":
                threshold = 0.005  # 0.5% for high-quality single-crystal XRD
            elif data_source == "experiment":
                threshold = 0.01  # 1% for typical experimental XRD
            else:  # "MP" or "unknown"
                threshold = 0.02  # 2% to account for DFT systematic errors
        
        # Normalize formula
        formula_clean = material_formula.strip()
        
        # Detect family
        family = _detect_cuprate_family(formula_clean)
        
        # Get family rules
        family_rules = FAMILY_C_AXIS_RULES.get(family) if family else None
        
        # Check against known cuprates
        cuprate_info = None
        matched_formula = None
        for key, info in CUPRATE_DATA.items():
            if key.lower() in formula_clean.lower():
                cuprate_info = info
                matched_formula = key
                break
        
        # Early exit: Infinite-layer
        if family == "infinite_layer":
            return {
                "success": True,
                "scenario": scenario,
                "material": matched_formula or formula_clean,
                "family": family,
                "analysis_type": "not_applicable",
                "verdict": "NOT_APPLICABLE",
                "explanation": (
                    f"{matched_formula or formula_clean} is an infinite-layer cuprate with NO apical oxygen by design. "
                    "Octahedral stability analysis is not applicable. Structure has square planar CuO₄ coordination "
                    "with minimal c-axis (≈3.2–3.4 Å)."
                ),
                "c_axis_driver": "not_applicable",
                "applicable_to_octahedral_stability": False,
                "citations": [
                    "Standard cuprate crystallography: Infinite-layer has no apical oxygen",
                ],
            }
        
        # If not in known database and no family detected
        if not cuprate_info and not family:
            return {
                "success": True,
                "material": formula_clean,
                "analysis_type": "generic_cuprate",
                "family": "unknown",
                "note": (
                    "Material not in known cuprate database and family not recognized. "
                    "General principle: c-axis effects on octahedral stability are FAMILY-SPECIFIC. "
                    "For 214-type (La₂CuO₄), increasing c correlates with apical O retention. "
                    "For 123-type (YBCO), c tracks chain oxygen (not apical). "
                    "For T′-type and infinite-layer, apical oxygen is absent by design."
                ),
            }
        
        # Get typical c-axis
        typical_c = cuprate_info.get("typical_c", 13.0) if cuprate_info else 13.0
        c_used = c_axis_spacing if c_axis_spacing is not None else typical_c
        
        # Adjust threshold based on data source
        if data_source == "MP":
            effective_threshold = max(threshold, 0.02)  # At least 2% for MP due to DFT errors
        else:
            effective_threshold = threshold
        
        # Calculate relative c-axis change
        delta_c_rel = (c_used - typical_c) / typical_c
        delta_c_abs = c_used - typical_c
        
        # Family-specific analysis
        if family == "123":
            # YBCO: Special handling - c tracks CHAIN oxygen, NOT apical
            if scenario in ("trend_increase", "trend_decrease"):
                baseline = typical_c
                sign = +1 if scenario == "trend_increase" else -1
                probe_rel = sign * abs(trend_probe)
                probe_abs = probe_rel * baseline
                projected = baseline * (1 + probe_rel)
                
                if sign > 0:
                    explanation = (
                        f"In YBa₂Cu₃O₇₋δ (123-type), increasing c from {baseline:.2f} Å to {projected:.2f} Å "
                        f"(+{abs(probe_abs):.3f} Å, +{100*abs(probe_rel):.1f}%) correlates with CHAIN oxygen depletion "
                        f"(O₇ → O₆) in the Cu-O chains between CuO₂ planes, which REDUCES superconducting properties "
                        "and drives orthorhombic → tetragonal structural transition. "
                        "CRITICAL: Apical oxygen above/below planar Cu sites on the CuO₂ planes REMAINS present across "
                        "the entire oxygen range (O₆ to O₇). The c-axis expansion reflects chain oxygen loss from the "
                        "Cu-O chains (Cu1 sites), NOT removal of apical oxygen from plane Cu sites (Cu2). "
                        "Square pyramidal CuO₅ coordination on planes is preserved."
                    )
                    verdict = "FALSE"  # Increasing c does NOT mean apical retention in YBCO
                    stability_effect = "chain_depleted"
                else:
                    explanation = (
                        f"In YBa₂Cu₃O₇₋δ (123-type), decreasing c from {baseline:.2f} Å to {projected:.2f} Å "
                        f"({probe_abs:.3f} Å, {100*probe_rel:.1f}%) correlates with chain oxygen RETENTION "
                        f"(maintaining O₇ stoichiometry), which ENHANCES superconducting properties. "
                        "Apical oxygen on CuO₂ planes (Cu2 sites) remains present."
                    )
                    verdict = "FALSE"  # Claim is about increasing c; decreasing c also doesn't relate to octahedral stability
                    stability_effect = "chain_retained"
                
                return {
                    "success": True,
                    "scenario": scenario,
                    "material": matched_formula or formula_clean,
                    "family": family,
                    "claim": "Does increasing c-axis stabilize Cu–O octahedral coordination?",
                    "verdict": verdict,
                    "baseline_c_axis": round(baseline, 4),
                    "projected_c_axis": round(projected, 4),
                    "hypothetical_delta_angstrom": round(abs(probe_abs), 4),
                    "hypothetical_delta_percent": round(100 * abs(probe_rel), 2),
                    "threshold_used_percent": round(100 * threshold, 1),
                    "data_source": data_source,
                    "coordination": cuprate_info.get("coordination") if cuprate_info else "square pyramidal",
                    "stability_effect": stability_effect,
                    "mechanism": explanation,
                    "c_axis_driver": "chain_oxygen",
                    "applicable_to_octahedral_stability": False,
                    "warning": "YBCO c-axis tracks CHAIN oxygen on Cu1 sites, NOT apical oxygen on Cu2 plane sites",
                    "citations": family_rules["citations"] if family_rules else [
                        "Jorgensen et al., Phys. Rev. B 41, 1863 (1990): YBCO structure vs oxygen content",
                    ],
                }
            else:
                # Observed mode for YBCO
                explanation = (
                    f"In YBa₂Cu₃O₇₋δ, observed c = {c_used:.2f} Å vs typical {typical_c:.2f} Å "
                    f"(Δc = {delta_c_abs:+.3f} Å, {100*delta_c_rel:+.1f}%). "
                    "This change primarily reflects chain oxygen content (O₇₋δ stoichiometry), "
                    "NOT apical oxygen removal from CuO₂ planes. Apical oxygen on planes is retained."
                )
                return {
                    "success": True,
                    "scenario": "observed",
                    "material": matched_formula or formula_clean,
                    "family": family,
                    "c_axis_analyzed": c_used,
                    "c_axis_typical": typical_c,
                    "observed_change_angstrom": round(delta_c_abs, 4),
                    "observed_change_percent": round(100 * delta_c_rel, 2),
                    "coordination": cuprate_info.get("coordination") if cuprate_info else "square pyramidal",
                    "explanation": explanation,
                    "c_axis_driver": "chain_oxygen",
                    "applicable_to_octahedral_stability": False,
                    "warning": "YBCO c-axis tracks CHAIN oxygen, NOT apical oxygen on planes",
                    "citations": family_rules["citations"] if family_rules else [
                        "web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf",
                    ],
                }
        
        # T′-type: No apical oxygen by design
        if family == "T_prime":
            explanation = (
                f"{matched_formula or formula_clean} is a T′-type cuprate with NO apical oxygen by design "
                "(square planar CuO₄ coordination). C-axis changes reflect interstitial oxygen reorganization "
                "during annealing, which can move c in EITHER direction depending on starting oxygen configuration. "
                "The octahedral stability question is not applicable."
            )
            return {
                "success": True,
                "scenario": scenario,
                "material": matched_formula or formula_clean,
                "family": family,
                "analysis_type": "T_prime_no_apical",
                "verdict": "AMBIGUOUS",
                "explanation": explanation,
                "c_axis_driver": "interstitial_oxygen",
                "applicable_to_octahedral_stability": False,
                "citations": family_rules["citations"] if family_rules else [
                    "Avella & Guarino, Phys. Rev. B 105, 014512 (2022): Electron-doped T′ annealing",
                ],
            }
        
        # 214-type (La₂CuO₄) or other families where apical O logic applies
        if scenario in ("trend_increase", "trend_decrease"):
            # Trend mode
            baseline = typical_c
            sign = +1 if scenario == "trend_increase" else -1
            probe_rel = sign * abs(trend_probe)
            probe_abs = probe_rel * baseline
            projected = baseline * (1 + probe_rel)
            
            # Check if family supports octahedral stability analysis
            if family_rules and not family_rules.get("applicable_to_octahedral_stability", True):
                verdict = "AMBIGUOUS"
                explanation = (
                    f"For {matched_formula or formula_clean} ({family}-type), c-axis changes "
                    f"(hypothetical {probe_abs:+.3f} Å, {100*abs(probe_rel):.1f}%) "
                    f"are dominated by {family_rules.get('c_axis_meaning', 'stacking effects')}, "
                    "not apical oxygen retention. Octahedral stability correlation is ambiguous."
                )
                stability_effect = "ambiguous"
            elif sign > 0:
                verdict = "TRUE"
                stability_effect = "stabilized"
                explanation = (
                    f"Hypothetical c-axis increase of {abs(probe_abs):.3f} Å ({100*abs(probe_rel):.1f}%) "
                    f"from baseline {baseline:.2f} Å → {projected:.2f} Å "
                    "correlates with retention of apical oxygen and thus stabilizes CuO₆ octahedra. "
                    "This is the established trend for 214-type (K₂NiF₄ structure) cuprates."
                )
            else:
                verdict = "FALSE"
                stability_effect = "destabilized"
                explanation = (
                    f"Hypothetical c-axis decrease of {abs(probe_abs):.3f} Å ({100*abs(probe_rel):.1f}%) "
                    f"from baseline {baseline:.2f} Å → {projected:.2f} Å "
                    "correlates with removal of apical oxygen, destabilizing CuO₆ octahedra (toward square planar CuO₄). "
                    "This is the established trend for 214-type cuprates (T → T′ transition)."
                )
            
            out = {
                "success": True,
                "scenario": scenario,
                "material": matched_formula or formula_clean,
                "family": family or "214_assumed",
                "claim": "Does increasing c-axis stabilize Cu–O octahedral coordination?",
                "verdict": verdict,
                "baseline_c_axis": round(baseline, 4),
                "projected_c_axis": round(projected, 4),
                "hypothetical_delta_angstrom": round(abs(probe_abs), 4),
                "hypothetical_delta_percent": round(100 * abs(probe_rel), 2),
                "threshold_used_percent": round(100 * threshold, 1),
                "data_source": data_source,
                "coordination": cuprate_info.get("coordination") if cuprate_info else "elongated octahedral",
                "stability_effect": stability_effect,
                "mechanism": explanation,
                "c_axis_driver": family_rules.get("c_axis_driver", "apical_oxygen") if family_rules else "apical_oxygen",
                "applicable_to_octahedral_stability": family_rules.get("applicable_to_octahedral_stability", True) if family_rules else True,
            }
            
            if cuprate_info:
                out["structural_details"] = {
                    "typical_apical_distance_A": cuprate_info.get("apical_distance"),
                    "typical_planar_distance_A": cuprate_info.get("planar_distance"),
                    "note": cuprate_info.get("note"),
                }
            
            if family_rules:
                out["citations"] = family_rules.get("citations", [])
            else:
                out["citations"] = [
                    "Yamamoto et al., Physica C 470, 1383 (2010): T-phase La₂CuO₄ c ≈ 13.15 Å; T′-phase c ≈ 12.55 Å.",
                ]
            
            return out
        
        # Observed mode (for 214 and compatible families)
        if abs(delta_c_rel) > effective_threshold:
            if delta_c_rel > 0:
                stability_effect = "stabilized"
                mechanism = (
                    f"Increasing c-axis by {delta_c_abs:.3f} Å ({100*delta_c_rel:.1f}%) "
                    "correlates with retention of apical oxygen and stabilizes CuO₆ octahedra."
                )
                verdict = "TRUE"
            else:
                stability_effect = "destabilized"
                mechanism = (
                    f"Decreasing c-axis by {abs(delta_c_abs):.3f} Å ({100*abs(delta_c_rel):.1f}%) "
                    "correlates with removal of apical oxygen, destabilizing CuO₆ octahedra."
                )
                verdict = "FALSE"
        else:
            stability_effect = "minimal_change"
            mechanism = (
                f"c-axis change of {delta_c_abs:+.3f} Å ({100*delta_c_rel:+.1f}%) is below threshold "
                f"({100*effective_threshold:.0f}%, adjusted for {data_source} data source) "
                "and cannot reliably indicate octahedral stability change."
            )
            verdict = "AMBIGUOUS"
        
        # Add data source note if from MP
        if data_source == "MP":
            mechanism += " Note: Materials Project DFT values may have ~1–2% systematic error in lattice parameters."
        
        out = {
            "success": True,
            "scenario": "observed",
            "material": matched_formula or formula_clean,
            "family": family or "214_assumed",
            "claim": "Does increasing c-axis stabilize Cu–O octahedral coordination?",
            "verdict": verdict,
            "c_axis_analyzed": c_used,
            "c_axis_typical": typical_c,
            "observed_change_angstrom": round(delta_c_abs, 4),
            "observed_change_percent": round(100 * delta_c_rel, 2),
            "threshold_used_percent": round(100 * effective_threshold, 1),
            "data_source": data_source,
            "coordination": cuprate_info.get("coordination") if cuprate_info else "elongated octahedral",
            "stability_effect": stability_effect,
            "mechanism": mechanism,
            "c_axis_driver": family_rules.get("c_axis_driver", "apical_oxygen") if family_rules else "apical_oxygen",
            "applicable_to_octahedral_stability": family_rules.get("applicable_to_octahedral_stability", True) if family_rules else True,
        }
        
        if cuprate_info:
            out["structural_details"] = {
                "typical_apical_distance_A": cuprate_info.get("apical_distance"),
                "typical_planar_distance_A": cuprate_info.get("planar_distance"),
                "note": cuprate_info.get("note"),
            }
        
        if family_rules:
            out["citations"] = family_rules.get("citations", [])
        else:
            out["citations"] = [
                "Yamamoto et al., Physica C 470, 1383 (2010): T vs T′ phases",
            ]
        
        return out
        
    except Exception as e:
        _log.error(f"Error analyzing cuprate octahedral stability: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

