"""
Utility functions for semiconductor and defect analysis.

This module provides utilities for:
- Structure analysis (octahedral distortions, bond lengths)
- Magnetic property analysis
- Defect formation energy calculations
- Doping site preference analysis
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pymatgen.core import Structure, Element
from ..base.converters import to_angstrom as _to_Angstrom

_log = logging.getLogger(__name__)


def analyze_octahedral_distortion(
    structure: Structure,
    central_element: str,
    neighbor_element: Optional[str] = "O"
) -> Dict[str, Any]:
    """
    Analyze octahedral distortion around sites of `central_element` coordinated by `neighbor_element` (default O).
    Distances are true geometric distances (Å). Distortion metrics include:
      - Baur's bond-length distortion index: Δ = (1/6) * Σ |li - l_avg| / l_avg
      - Average cis- (near 90°) and trans- (near 180°) angle deviations.
    
    Args:
        structure: Pymatgen Structure object
        central_element: Element at center of octahedra (e.g., 'V')
        neighbor_element: Element at corners of octahedra (e.g., 'O'). Default 'O'.
        
    Returns:
        Dictionary with distortion analysis including bond lengths, angles, and distortion metrics
    """
    try:
        result: Dict[str, Any] = {
            "success": True,
            "central_element": central_element,
            "neighbor_element": neighbor_element,
            "octahedra_analyzed": 0,
            "distortions": [],
        }

        # indices of all central sites (e.g., all V)
        central_indices = [i for i, s in enumerate(structure) if s.species_string == central_element]
        if not central_indices:
            return {"success": False, "error": f"No {central_element} sites found in structure"}

        def six_nearest_O_neighbors(center_idx: int) -> Optional[List[Any]]:
            """Return the six nearest O neighbors as PeriodicNeighbor objects, or None if not found.
            Do not deduplicate by site index; periodic images are distinct nearest neighbors.
            """
            center = structure[center_idx]
            # Adaptive search radius: start at 2.2 Å and grow in small steps until we find ≥ 6 neighbors
            r = 2.2
            for _ in range(12):
                neighs = [
                    n for n in structure.get_neighbors(center, r)
                    if n.species_string == (neighbor_element or n.species_string)
                ]
                # Sort by geometric distance
                neighs.sort(key=lambda n: float(n.nn_distance))
                if len(neighs) >= 6:
                    return neighs[:6]  # keep periodic images; do not deduplicate by index
                r += 0.1
            return None

        for cidx in central_indices:
            neighs = six_nearest_O_neighbors(cidx)
            if not neighs or len(neighs) < 6:
                continue

            center = structure[cidx]
            # True V–O distances (Å)
            dists = [float(n.nn_distance) for n in neighs]
            d_avg = float(np.mean(dists))
            d_std = float(np.std(dists))
            # Baur bond-length distortion index (dimensionless)
            baur_delta = float(np.mean([abs(d - d_avg) / d_avg for d in dists]))

            # Build center→neighbor vectors in Cartesian using PeriodicNeighbor coords
            # PeriodicNeighbor.coords already references the correct imaged neighbor
            rc = np.asarray(center.coords, dtype=float)
            vecs = [np.asarray(n.coords, dtype=float) - rc for n in neighs]
            vecs = [v / np.linalg.norm(v) for v in vecs]

            # All pairwise angles (°) and raw cosines
            angles = []
            cosines = []
            for i in range(6):
                for j in range(i + 1, 6):
                    cosang_raw = float(np.dot(vecs[i], vecs[j]))
                    cosang = float(np.clip(cosang_raw, -1.0, 1.0))
                    angles.append(float(np.degrees(np.arccos(cosang))))
                    cosines.append(cosang_raw)

            # Classify near-cis (~90°) and near-trans (~180°) with tighter windows (for sanity checks)
            cis = [a for a in angles if 80.0 <= a <= 100.0]
            trans = [a for a in angles if 170.0 <= a <= 180.0]

            if len(cis) < 9 or len(trans) < 1:
                _log.warning(
                    "Unusual O–%s–%s angle distribution at site %d; check neighbor picking / structure. cis=%d, trans=%d",
                    central_element,
                    neighbor_element or "X",
                    cidx,
                    len(cis),
                    len(trans),
                )

            # Compute deviations using closest-to-ideal sets for robustness
            cis12 = sorted(angles, key=lambda a: abs(a - 90.0))[:12]
            # Select three most anti-parallel pairs via smallest dot products
            cos_sorted = sorted(cosines)  # ascending: most negative first
            trans3_angles = [float(np.degrees(np.arccos(float(np.clip(c, -1.0, 1.0))))) for c in cos_sorted[:3]]

            avg_dev_90 = float(np.mean([abs(a - 90.0) for a in cis12])) if cis12 else None
            avg_dev_180 = float(np.mean([abs(a - 180.0) for a in trans3_angles])) if trans3_angles else None

            # Quadratic elongation (λ) using average bond length as l0
            quad_elong = float(np.mean([(d / d_avg) ** 2 for d in dists])) if d_avg > 0 else None

            # Octahedral angle variance (σ²) from the 12 angles closest to 90°
            cis12 = sorted(angles, key=lambda a: abs(a - 90.0))[:12]
            angle_variance = float(np.mean([(a - 90.0) ** 2 for a in cis12])) if cis12 else None

            # Heuristic: call it "regular" only if both bond-length and angle distortions are tiny
            is_regular = (baur_delta < 0.01) and (avg_dev_90 is not None and avg_dev_90 < 3.0) and \
                         (avg_dev_180 is not None and avg_dev_180 < 3.0)

            result["distortions"].append({
                "site_index": int(cidx),
                "coordination_number": 6,
                "bond_lengths": {
                    "values": dists,
                    "average": d_avg,
                    "std_dev": d_std,
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists)),
                    "unit": "Å",
                },
                "distortion_parameter": baur_delta,  # Baur index
                "quadratic_elongation": quad_elong,
                "octahedral_angle_variance": angle_variance,
                "angles": {
                    "all_angles": angles,                       # 15 values
                    "angles_near_90": cis12,                    # 12 closest to 90°
                    "angles_near_180": trans3_angles,           # 3 most anti-parallel pairs
                    "avg_deviation_from_90": avg_dev_90,
                    "avg_deviation_from_180": avg_dev_180,
                    "unit": "degrees",
                },
                "is_regular": bool(is_regular),
            })
            result["octahedra_analyzed"] += 1

        if result["octahedra_analyzed"] == 0:
            return {"success": False, "error": f"No octahedral {central_element} sites with six {neighbor_element} neighbors were found"}

        # Overall stats
        deltas = [d["distortion_parameter"] for d in result["distortions"]]
        result["overall_distortion"] = {
            "average": float(np.mean(deltas)),
            "std_dev": float(np.std(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        }
        result["has_significant_distortion"] = bool(result["overall_distortion"]["average"] > 0.01)

        return result
        
    except Exception as e:
        _log.error(f"Error in octahedral distortion analysis: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def get_magnetic_properties_detailed(mpr, material_id: str) -> Dict[str, Any]:
    """
    Get detailed magnetic properties for a material.
    
    Args:
        mpr: MPRester client instance
        material_id: Material ID
        
    Returns:
        Dictionary with magnetic properties
    """
    try:
        docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=[
                "material_id", "formula_pretty", "composition",
                "is_magnetic", "ordering", "total_magnetization",
                "total_magnetization_normalized_vol",
                "total_magnetization_normalized_formula_units",
                "num_magnetic_sites", "num_unique_magnetic_sites",
                "types_of_magnetic_species", "symmetry",
                "energy_above_hull", "is_stable"
            ]
        )
        
        if not docs:
            return {
                "success": False,
                "error": f"Material {material_id} not found"
            }
        
        doc = docs[0]
        
        result = {
            "success": True,
            "material_id": material_id,
            "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
            "composition": dict(doc.composition.as_dict()) if hasattr(doc, 'composition') else None,
            "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
            "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None
        }
        
        # Magnetic properties
        result["is_magnetic"] = doc.is_magnetic if hasattr(doc, 'is_magnetic') else None
        result["magnetic_ordering"] = str(doc.ordering) if hasattr(doc, 'ordering') and doc.ordering else None
        
        if hasattr(doc, 'total_magnetization') and doc.total_magnetization is not None:
            result["total_magnetization"] = {
                "value": float(doc.total_magnetization),
                "unit": "μB"
            }
        
        if hasattr(doc, 'total_magnetization_normalized_vol') and doc.total_magnetization_normalized_vol is not None:
            result["magnetization_per_volume"] = {
                "value": float(doc.total_magnetization_normalized_vol),
                "unit": "μB/Bohr³"
            }
        
        if hasattr(doc, 'total_magnetization_normalized_formula_units') and doc.total_magnetization_normalized_formula_units is not None:
            result["magnetization_per_formula_unit"] = {
                "value": float(doc.total_magnetization_normalized_formula_units),
                "unit": "μB/f.u."
            }
        
        result["num_magnetic_sites"] = doc.num_magnetic_sites if hasattr(doc, 'num_magnetic_sites') else None
        result["num_unique_magnetic_sites"] = doc.num_unique_magnetic_sites if hasattr(doc, 'num_unique_magnetic_sites') else None
        
        if hasattr(doc, 'types_of_magnetic_species') and doc.types_of_magnetic_species:
            result["magnetic_species"] = [str(s) for s in doc.types_of_magnetic_species]
        
        return result
        
    except Exception as e:
        _log.error(f"Error getting magnetic properties for {material_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def compare_magnetic_properties(
    undoped_props: Dict[str, Any],
    doped_props: Dict[str, Any]
) -> Dict[str, Any]:
    """
    SAFE VERSION: Compare magnetic properties between undoped and doped materials.
    
    Try to compare normalized Ms (A/m or kA/m) if available.
    Fall back to per-formula-unit (μB/f.u.) comparison.
    REFUSE to compare raw total_magnetization μB/cell across unrelated phases.
    
    WARNING: This does NOT prove better saturation magnetization unless the two materials 
    are the same structure and similar stoichiometry. For Fe₂O₃ doping analysis use 
    assess_doping_effect_on_saturation_magnetization from the magnet tools instead.
    
    Args:
        undoped_props: Magnetic properties of undoped material
        doped_props: Magnetic properties of doped material
        
    Returns:
        Dictionary with comparison
    """
    try:
        if not undoped_props.get("success") or not doped_props.get("success"):
            return {
                "success": False,
                "error": "One or both materials missing property data"
            }
        
        result = {
            "success": True,
            "undoped": {
                "material_id": undoped_props.get("material_id"),
                "formula": undoped_props.get("formula"),
                "is_magnetic": undoped_props.get("is_magnetic"),
                "ordering": undoped_props.get("magnetic_ordering")
            },
            "doped": {
                "material_id": doped_props.get("material_id"),
                "formula": doped_props.get("formula"),
                "is_magnetic": doped_props.get("is_magnetic"),
                "ordering": doped_props.get("magnetic_ordering")
            },
            "warnings": []
        }
        
        # Best: compare magnetization_per_formula_unit (μB/f.u.) if both present
        if (
            "magnetization_per_formula_unit" in undoped_props and
            "magnetization_per_formula_unit" in doped_props
        ):
            m1 = undoped_props["magnetization_per_formula_unit"]["value"]
            m2 = doped_props["magnetization_per_formula_unit"]["value"]
            result["magnetization_per_fu_comparison"] = {
                "undoped": float(m1),
                "doped": float(m2),
                "absolute_change": float(m2 - m1),
                "percent_change": float((m2 - m1) / m1 * 100) if m1 != 0 else None,
                "unit": "μB/f.u."
            }
        else:
            result["warnings"].append(
                "No consistent per-formula-unit magnetization available for both materials."
            )

        # DO NOT compare raw total_magnetization μB/cell unless nsites & volume match.
        # We'll explicitly tell the model not to trust that.
        if (
            "total_magnetization" in undoped_props and
            "total_magnetization" in doped_props
        ):
            result["warnings"].append(
                "Raw total_magnetization (μB/cell) is not reliably comparable across different structures / cell sizes. "
                "Not reported. Use assess_doping_effect_on_saturation_magnetization for normalized Ms comparisons."
            )

        # high-level interpretation: only if we got a sane per-f.u. comparison
        comp = result.get("magnetization_per_fu_comparison")
        if comp and comp.get("absolute_change") is not None:
            improved = comp["absolute_change"] > 0
            result["magnetic_enhancement"] = bool(improved)
            result["interpretation"] = "Doping enhanced per-f.u. magnetic moment" if improved else "Doping reduced per-f.u. magnetic moment"
        else:
            result["magnetic_enhancement"] = None
            result["interpretation"] = (
                "Insufficient normalized data to compare magnetic strength. "
                "Use magnet tools (assess_doping_effect_on_saturation_magnetization) for proper Ms analysis."
            )

        return result
        
    except Exception as e:
        _log.error(f"Error comparing magnetic properties: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def calculate_defect_formation_energy(
    mpr,
    host_material_id: str,
    defect_composition: Dict[str, float],
    defect_type: str = "substitutional"
) -> Dict[str, Any]:
    """
    Calculate or estimate defect formation energy.
    
    For interstitial vs substitutional comparison, we compare:
    - Energy of structure with dopant at substitutional site
    - Energy of structure with dopant at interstitial site
    
    Args:
        mpr: MPRester client instance
        host_material_id: Material ID of host
        defect_composition: Composition with defect (e.g., {"Si": 31, "P": 1})
        defect_type: "substitutional" or "interstitial"
        
    Returns:
        Dictionary with formation energy analysis
    """
    try:
        # Get host material properties
        host_docs = mpr.materials.summary.search(
            material_ids=[host_material_id],
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "structure", "nsites"
            ]
        )
        
        if not host_docs:
            return {
                "success": False,
                "error": f"Host material {host_material_id} not found"
            }
        
        host_doc = host_docs[0]
        host_energy_per_atom = host_doc.energy_per_atom if hasattr(host_doc, 'energy_per_atom') else None
        
        if host_energy_per_atom is None:
            return {
                "success": False,
                "error": "Host material energy not available"
            }
        
        result = {
            "success": True,
            "host_material": {
                "material_id": host_material_id,
                "formula": host_doc.formula_pretty if hasattr(host_doc, 'formula_pretty') else str(host_doc.composition),
                "energy_per_atom": float(host_energy_per_atom),
                "unit": "eV/atom"
            },
            "defect_type": defect_type,
            "defect_composition": defect_composition
        }
        
        # Search for structures with the defect composition
        elements = list(defect_composition.keys())
        total_atoms = sum(defect_composition.values())
        
        # Convert composition to formula-like string
        from collections import Counter
        comp_counter = Counter(defect_composition)
        
        # Search for similar materials
        search_docs = mpr.materials.summary.search(
            elements=elements,
            num_elements=len(elements),
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "nsites", "energy_above_hull"
            ],
        )
        
        # Find best match
        best_match = None
        min_comp_diff = float('inf')
        
        for doc in search_docs:
            comp = doc.composition.as_dict()
            total = sum(comp.values())
            
            # Normalize to same scale
            scale = total_atoms / total
            scaled_comp = {k: v * scale for k, v in comp.items()}
            
            # Calculate composition difference
            diff = sum(abs(scaled_comp.get(el, 0) - defect_composition.get(el, 0)) for el in elements)
            
            if diff < min_comp_diff:
                min_comp_diff = diff
                best_match = doc
        
        if best_match:
            result["defect_structure"] = {
                "material_id": best_match.material_id if hasattr(best_match, 'material_id') else None,
                "formula": best_match.formula_pretty if hasattr(best_match, 'formula_pretty') else str(best_match.composition),
                "energy_per_atom": float(best_match.energy_per_atom) if hasattr(best_match, 'energy_per_atom') else None,
                "energy_above_hull": float(best_match.energy_above_hull) if hasattr(best_match, 'energy_above_hull') and best_match.energy_above_hull is not None else None,
                "composition_difference": float(min_comp_diff)
            }
            
            # Estimate formation energy (simplified)
            if hasattr(best_match, 'energy_per_atom') and best_match.energy_per_atom is not None:
                # Formation energy ≈ E(defect) - E(host) (per defect site)
                # This is simplified; proper calculation needs chemical potentials
                delta_e = (float(best_match.energy_per_atom) - float(host_energy_per_atom)) * total_atoms
                
                result["estimated_formation_energy"] = {
                    "value": float(delta_e),
                    "unit": "eV",
                    "note": "Simplified estimate; proper defect formation energy requires chemical potentials and charged defects calculations"
                }
        else:
            result["defect_structure"] = None
            result["note"] = "No suitable defect structure found in database"
        
        return result
        
    except Exception as e:
        _log.error(f"Error calculating defect formation energy: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def analyze_doping_site_preference(
    mpr,
    host_formula: str,
    dopant_element: str,
    site_a_element: str,
    site_b_element: str,
    temperature: float = 298.15,
    pressure: float = 1.0
) -> Dict[str, Any]:
    """
    Predict which sublattice (site_a_element vs site_b_element) the dopant_element prefers
    in a binary compound host_formula (e.g. GaAs, ZnO, etc.).

    Strategy:
    1. Try to find Ga–As–N-like compounds in Materials Project and infer which
       site N sits on based on valence-chemistry similarity, then compare energies.
    2. If that fails (common for dilute alloys), fall back to a chemistry heuristic:
         - The dopant prefers the site whose group / electronegativity is
           closest to the dopant's.
    
    Args:
        mpr: MPRester client instance
        host_formula: Host material formula (e.g., "GaAs")
        dopant_element: Dopant element (e.g., "N")
        site_a_element: First potential substitution site (e.g., "Ga")
        site_b_element: Second potential substitution site (e.g., "As")
        temperature: Temperature in K (default: 298.15 K = 25°C = STP)
        pressure: Pressure in atm (default: 1.0 atm = STP)
        
    Returns:
        Dictionary with site preference analysis
    """

    def _chem_distance_score(dop, host):
        # lower = "more similar", i.e. easier substitution
        g_d = _valence_group(dop) or 0
        g_h = _valence_group(host) or 0
        chi_d = _pauling_en(dop) or 0.0
        chi_h = _pauling_en(host) or 0.0
        return abs(g_d - g_h) + 0.5 * abs(chi_d - chi_h)

    try:
        # --- 0. Fetch host info (stable GaAs, etc.) ---
        host_docs = mpr.materials.summary.search(
            formula=host_formula,
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "energy_above_hull", "structure",
                "is_stable"
            ],
        )
        if not host_docs:
            return {
                "success": False,
                "error": f"Host material {host_formula} not found"
            }

        # Pick the most stable (lowest energy_above_hull)
        host_doc = sorted(
            host_docs,
            key=lambda x: getattr(x, "energy_above_hull", float("inf"))
        )[0]

        host_energy = getattr(host_doc, "energy_per_atom", None)
        result = {
            "success": True,
            "host_material": {
                "material_id": getattr(host_doc, "material_id", None),
                "formula": getattr(host_doc, "formula_pretty", host_formula),
                "energy_per_atom": float(host_energy) if host_energy is not None else None,
                "unit": "eV/atom",
            },
            "dopant_element": dopant_element,
            "site_a": site_a_element,
            "site_b": site_b_element,
            "temperature_K": temperature,
            "pressure_atm": pressure,
        }

        # --- 1. Pull candidate doped structures (Ga-As-N style) ---
        # We don't force num_elements==3 because MP might store off-stoichiometric
        # or ordered supercells with the same 3 elements but more sites.
        doped_docs = mpr.materials.summary.search(
            elements=[site_a_element, site_b_element, dopant_element],
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "energy_above_hull", "is_stable"
            ],
        )

        site_a_candidates = []
        site_b_candidates = []

        for doc in doped_docs:
            comp = doc.composition.as_dict()
            # skip anything that doesn't actually contain all 3 required species
            if (site_a_element not in comp or
                site_b_element not in comp or
                dopant_element not in comp):
                continue

            # figure out which sublattice dopant is *most like* chemically
            score_a = _chem_distance_score(dopant_element, site_a_element)
            score_b = _chem_distance_score(dopant_element, site_b_element)

            inferred_site = None
            if score_a < score_b:
                inferred_site = "A"
            elif score_b < score_a:
                inferred_site = "B"
            else:
                # tie-breaker: pick site with closer electronegativity
                chi_d = _pauling_en(dopant_element) or 0.0
                chi_a = _pauling_en(site_a_element) or 0.0
                chi_b = _pauling_en(site_b_element) or 0.0
                inferred_site = "A" if abs(chi_d - chi_a) <= abs(chi_d - chi_b) else "B"

            doc_summary = {
                "material_id": getattr(doc, "material_id", None),
                "formula": getattr(doc, "formula_pretty", str(doc.composition)),
                "composition": comp,
                "energy_per_atom": float(getattr(doc, "energy_per_atom", np.nan)),
                "energy_above_hull": float(getattr(doc, "energy_above_hull", np.nan)),
                "is_stable": bool(getattr(doc, "is_stable", False)),
                "inferred_substitution_site": (
                    site_a_element if inferred_site == "A" else site_b_element
                )
            }

            if inferred_site == "A":
                site_a_candidates.append(doc_summary)
            else:
                site_b_candidates.append(doc_summary)

        # --- 2. Pick "best" (lowest energy_above_hull) example for each site ---
        site_a_best = None
        if site_a_candidates:
            site_a_best = min(
                site_a_candidates,
                key=lambda x: x.get("energy_above_hull", float("inf"))
            )

        site_b_best = None
        if site_b_candidates:
            site_b_best = min(
                site_b_candidates,
                key=lambda x: x.get("energy_above_hull", float("inf"))
            )

        result["site_a_substitution"] = site_a_best
        result["site_b_substitution"] = site_b_best

        # --- 3. Decide preference ---
        # Case 3a: we actually have energy info for both
        if site_a_best and site_b_best:
            e_a = site_a_best.get("energy_above_hull", None)
            e_b = site_b_best.get("energy_above_hull", None)

            if e_a is not None and e_b is not None and np.isfinite(e_a) and np.isfinite(e_b):
                preferred = site_a_element if e_a < e_b else site_b_element
                delta = abs(e_a - e_b)

                result["site_preference"] = {
                    "preferred_site": preferred,
                    "energy_difference": float(delta),
                    "unit": "eV/atom",
                    "interpretation": (
                        f"{dopant_element} prefers the {preferred} sublattice "
                        f"(more stable by {delta:.3f} eV/atom among available {site_a_element}–{site_b_element}–{dopant_element} phases)."
                    )
                }
                return result

        # Case 3b: fallback — use pure chemistry heuristic if database was inconclusive
        score_a = _chem_distance_score(dopant_element, site_a_element)
        score_b = _chem_distance_score(dopant_element, site_b_element)

        preferred = site_a_element if score_a < score_b else site_b_element
        score_margin = abs(score_a - score_b)

        result["site_preference"] = {
            "preferred_site": preferred,
            "energy_difference": None,
            "unit": None,
            "interpretation": (
                f"{dopant_element} is chemically closer to {preferred} "
                f"(valence/electronegativity match). "
                f"Score margin ~{score_margin:.2f} (lower score = better match)."
            ),
            "method": "chem-heuristic-fallback"
        }

        if not (site_a_best or site_b_best):
            result["note"] = (
                f"No explicit {site_a_element}–{site_b_element}–{dopant_element} supercells/alloys found in the database. "
                "Returned heuristic site preference based on group/electronegativity matching."
            )
        else:
            result["note"] = (
                "Only one doped sublattice had candidate structures with energies; "
                "fell back to valence/electronegativity heuristic for final comparison."
            )

        return result

    except Exception as e:
        _log.error(f"Error analyzing doping site preference: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def analyze_structure_temperature_dependence(
    mpr,
    formula: str,
    element_of_interest: str,
    neighbor_element: str = None
) -> Dict[str, Any]:
    """
    Analyze temperature-dependent structural changes.
    
    This searches for different polymorphs/phases and analyzes structural differences.
    
    Args:
        mpr: MPRester client instance
        formula: Chemical formula
        element_of_interest: Element to analyze environment for
        neighbor_element: Neighboring element to consider
        
    Returns:
        Dictionary with temperature-dependent structure analysis
    """
    try:
        # Search for all polymorphs of the material
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id", "formula_pretty", "structure",
                "energy_above_hull", "symmetry", "is_stable"
            ]
        )
        
        if not docs:
            return {
                "success": False,
                "error": f"No materials found for formula {formula}"
            }
        
        result = {
            "success": True,
            "formula": formula,
            "num_polymorphs": len(docs),
            "polymorphs": []
        }
        
        for doc in docs[:5]:  # Limit to top 5 polymorphs
            structure = doc.structure if hasattr(doc, 'structure') else None
            
            if structure is None:
                continue
            
            polymorph_data = {
                "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else formula,
                "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None,
                "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None
            }
            
            # Get symmetry info
            if hasattr(doc, 'symmetry') and doc.symmetry:
                sym = doc.symmetry
                if hasattr(sym, 'crystal_system'):
                    polymorph_data["crystal_system"] = str(sym.crystal_system)
                if hasattr(sym, 'symbol'):
                    polymorph_data["space_group"] = str(sym.symbol)
            
            # Analyze octahedral environment if applicable
            if element_of_interest:
                distortion = analyze_octahedral_distortion(
                    structure,
                    element_of_interest,
                    neighbor_element
                )
                
                if distortion.get("success"):
                    polymorph_data["octahedral_analysis"] = {
                        "num_octahedra": int(distortion.get("octahedra_analyzed", 0)),
                        "has_distortion": bool(distortion.get("has_significant_distortion", False)),
                        "average_distortion": float(distortion.get("overall_distortion", {}).get("average")) if distortion.get("overall_distortion", {}).get("average") is not None else None
                    }
            
            result["polymorphs"].append(polymorph_data)
        
        # Sort by energy_above_hull (most stable first)
        result["polymorphs"].sort(key=lambda x: x.get("energy_above_hull", float('inf')))
        
        # Identify ground state (most stable)
        if result["polymorphs"]:
            result["ground_state"] = result["polymorphs"][0]
            
            # Check if ground state has distortion
            if "octahedral_analysis" in result["ground_state"]:
                result["ground_state_distortion"] = bool(result["ground_state"]["octahedral_analysis"]["has_distortion"])
        
        return result
        
    except Exception as e:
        _log.error(f"Error analyzing temperature-dependent structure: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

"""
General-purpose tool to assess whether a dopant prefers substitutional
or interstitial sites in a crystalline semiconductor (e.g., Si:P).

It supports three modes:
  (A) If you provide DFT formation energies -> uses them directly.
  (B) Else, if an MPRester is given -> fetches structure and computes
      geometric/chemical descriptors for a physics-based heuristic.
  (C) Else -> falls back to host-class defaults (still transparent).

Outputs a comparative 'formation-energy proxy' and a verdict.

Design choices & physics:
- Substitutional formation 'cost' ~ bond rehybridization + size mismatch.
- Interstitial formation 'cost' ~ strong steric strain + bond breaking.
- For covalent hosts (diamond/zincblende), interstitial sites are tiny;
  large dopants are heavily penalized (P in Si is a classic example).
- Group-V on Si (substitutional) is a shallow donor; interstitial P
  typically unstable/metastable and converts via kick-out to substitutional.

References (conceptual, not hard-coded):
- Zhang–Northrup / Van de Walle–Neugebauer defect formation framework.
- Site-size/strain arguments; kick-out mechanism in covalent semiconductors.
"""

import math
from dataclasses import dataclass
from typing import Literal

# Use mendeleev for covalent radii/electronegativity if available
try:
    from mendeleev import element as md_element  # type: ignore
except Exception:
    md_element = None  # type: ignore


# ---- Basic data helpers ------------------------------------------------------

def _covalent_radius_A(sym: str) -> Optional[float]:
    """Get covalent radius in Å; supports mendeleev (pm) and pymatgen (Å)."""
    # mendeleev first, but convert pm -> Å
    if md_element:
        try:
            e = md_element(sym)
            # Prefer Pyykkö single-bond; fall back sanely.
            r = getattr(e, "covalent_radius_pyykko", None)
            # Some mendeleev versions store multiple bonds in a dict
            if isinstance(r, dict):
                r = r.get("single") or r.get(1)
            if r is None:
                r = getattr(e, "covalent_radius", None)
            r = _to_Angstrom(float(r)) if r is not None else None
            if r:
                return r
        except Exception:
            pass
    # pymatgen (already in Å)
    try:
        r = Element(sym).average_covalent_radius
        if r:
            return float(r)
    except Exception:
        pass
    # small fallback table, in Å
    fallback = {
        "H": 0.31, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "Si": 1.11, "P": 1.07, "S": 1.05, "Ge": 1.22, "As": 1.19
    }
    return fallback.get(sym)


def _pauling_en(sym: str) -> Optional[float]:
    if md_element:
        try:
            e = md_element(sym)
            return float(e.en_pauling) if e.en_pauling is not None else None
        except Exception:
            pass
    try:
        return Element(sym).X
    except Exception:
        pass
    return None


def _valence_group(sym: str) -> Optional[int]:
    """Return periodic group number for lightweight valence heuristic."""
    if md_element:
        try:
            g = md_element(sym).group_id
            return int(g) if g else None
        except Exception:
            pass
    try:
        return Element(sym).group
    except Exception:
        pass
    # Fallback for a few common species
    f = {"B": 13, "C": 14, "N": 15, "O": 16, "Si": 14, "P": 15, "S": 16, "Ge": 14, "As": 15}
    return f.get(sym)


# ---- Geometry helpers --------------------------------------------------------

@dataclass
class VoidSizes:
    r_tet_A: float
    r_hex_A: float

def _nn_distance_A(struct: Structure) -> float:
    """Approximate nearest-neighbor distance from CrystalNN or simple min distance."""
    # Robust: use CrystalNN and honor periodic images; fall back to radial neighbors
    try:
        from pymatgen.analysis.local_env import CrystalNN
        cnn = CrystalNN()
        nn = cnn.get_nn_info(struct, 0)
        dists = []
        for n in nn:
            j = n['site_index']
            img = n.get('image', (0, 0, 0))
            d = struct.get_distance(0, j, jimage=img)  # <- honors periodic images
            if d > 1e-6:  # exclude self/near-self
                dists.append(d)
        if dists:
            return float(sorted(dists)[0])  # first NN
    except Exception:
        pass
    # Fallback: smallest non-zero distance from distance_matrix
    dm = struct.distance_matrix
    mind = min(dm[i][j] for i in range(len(dm)) for j in range(len(dm)) if i != j and dm[i][j] > 1e-6)
    return float(mind)

def _estimate_void_sizes(struct: Optional[Structure], host_sym: str) -> Tuple[VoidSizes, float]:
    """
    Estimate tetrahedral and hexagonal/octahedral void radii (Å), and return the
    NN distance used (Å) for diagnostics.
    """
    if struct is not None:
        d = _nn_distance_A(struct)
    else:
        r = _covalent_radius_A(host_sym) or 1.1
        d = 2.0 * r

    # Host-specific sanity clamps
    if host_sym in {"Si", "Ge", "C"}:  # diamond
        expected = {"Si": 2.35, "Ge": 2.45, "C": 1.54}.get(host_sym, 2.35)
        if not (0.9 * expected <= d <= 1.1 * expected):
            _log.warning(f"NN distance {d:.2f} Å for {host_sym} outside expected range; using {expected:.2f} Å")
            d = expected
    else:
        if d < 1.5 or d > 4.0:
            d = 2.0 * (_covalent_radius_A(host_sym) or 1.1)

    # Diamond: T ~ 0.225 d, H ~ 0.414 d. Add small floors.
    r_tet = max(0.40, 0.225 * d)
    r_hex = max(0.60, 0.414 * d)

    return VoidSizes(float(r_tet), float(r_hex)), float(d)


# ---- Formation-energy proxies (heuristic) ------------------------------------

@dataclass
class SiteScores:
    Eproxy_sub_eV: float
    Eproxy_int_tet_eV: float
    Eproxy_int_hex_eV: float  # hex for diamond, oct for fcc/hcp
    details: Dict[str, Any]
    diagnostics: Dict[str, Any]

def _formation_proxy_subst(host: str, dopant: str, struct: Optional[Structure]) -> float:
    """
    Substitutional 'formation energy proxy' (lower is better).
    Terms:
      + size_mismatch_penalty ~ k_s * (Δr / r_host)^2
      + valence_mismatch_penalty ~ k_v * max(0, |Δgroup| - 1)  (one electron off is OK for doping)
      + en_mismatch_penalty    ~ k_en * |Δχ|  (large χ difference can be unfavorable in covalents)
    Tuned to keep magnitudes reasonable; absolute values meaningless, comparison is key.
    """
    r_host = _covalent_radius_A(host) or 1.1
    r_dop  = _covalent_radius_A(dopant) or 1.1
    dgrp   = 0
    g_h = _valence_group(host)
    g_d = _valence_group(dopant)
    if g_h and g_d:
        dgrp = abs(g_d - g_h)
    dchi = 0.0
    chi_h = _pauling_en(host)
    chi_d = _pauling_en(dopant)
    if chi_h is not None and chi_d is not None:
        dchi = abs(chi_d - chi_h)

    k_s, k_v, k_en = 3.0, 0.5, 0.3
    size_pen = k_s * ( (r_dop - r_host) / r_host )**2
    val_pen  = k_v * max(0, dgrp - 1)  # allow ±1 as dopant
    en_pen   = k_en * dchi

    # Small bonus for classic shallow dopants in group-IV hosts (e.g., P->Si, B->Si)
    bonus = 0.0
    if host in {"Si", "Ge", "C"} and g_h == 14 and g_d in {13, 15}:
        bonus = -0.5

    return float(size_pen + val_pen + en_pen + bonus)

def _formation_proxy_interstitial(host: str, dopant: str, struct: Optional[Structure]) -> Dict[str, Any]:
    """
    Interstitial proxy for tetra (T) and hex (H) in diamond (or octa in fcc/hcp).
    """
    r_dop = _covalent_radius_A(dopant) or 1.1
    voids, d_used = _estimate_void_sizes(struct, host)
    chi_h = _pauling_en(host)
    chi_d = _pauling_en(dopant)
    dchi = abs(chi_d - chi_h) if (chi_h is not None and chi_d is not None) else 0.0

    # Tuned coefficients: diamond has higher network-disruption baseline.
    if host in {"Si", "Ge", "C"}:
        k_i, k_en, b0 = 14.0, 0.3, 2.5
    else:
        k_i, k_en, b0 = 20.0, 0.3, 1.0

    diagnostics = {
        "d_nn_A": d_used,
        "radii_A": {
            "dopant": r_dop,
            "tet_void": voids.r_tet_A,
            "hex_void": voids.r_hex_A,
        },
        "overfill": {},
        "cap_hit": {},
        "warnings": []
    }

    def strain(r_void, site_name: str):
        r_void = max(r_void, 0.40)  # Å
        overfill_raw = max(0.0, r_dop / r_void - 1.0)
        cap_hit = overfill_raw >= 2.0
        overfill = min(overfill_raw, 2.0)  # soft cap mainly for pathologies
        diagnostics["overfill"][site_name] = float(overfill_raw)
        diagnostics["cap_hit"][site_name] = bool(cap_hit)
        if cap_hit:
            diagnostics["warnings"].append(
                f"{site_name.capitalize()} interstitial strain hit cap (overfill={overfill_raw:.2f}); proxy is a lower bound"
            )
        return k_i * (overfill ** 2)

    E_tet = b0 + strain(voids.r_tet_A, "tet") + k_en * dchi
    E_hex = b0 + strain(voids.r_hex_A, "hex") + k_en * dchi

    return {
        "tet": float(E_tet),
        "hex": float(E_hex),
        "r_tet_A": voids.r_tet_A,
        "r_hex_A": voids.r_hex_A,
        "r_dop_A": r_dop,
        "diagnostics": diagnostics
    }


def _score_sites(host: str, dopant: str, struct: Optional[Structure]) -> SiteScores:
    E_sub = _formation_proxy_subst(host, dopant, struct)
    ints = _formation_proxy_interstitial(host, dopant, struct)
    
    return SiteScores(
        Eproxy_sub_eV=E_sub,
        Eproxy_int_tet_eV=ints["tet"],
        Eproxy_int_hex_eV=ints["hex"],
        details={
            "radii_A": {
                "host": _covalent_radius_A(host),
                "dopant": ints["r_dop_A"],
                "tet_void": ints["r_tet_A"],
                "hex_void": ints["r_hex_A"],
            }
        },
        diagnostics=ints["diagnostics"]
    )


# ---- Public API --------------------------------------------------------------

@dataclass
class SitePreferenceResult:
    success: bool
    host: str
    dopant: str
    method: Literal["DFT-provided", "heuristic-structure", "heuristic-generic"]
    preferred_site: Literal["substitutional", "interstitial-tetra", "interstitial-hex"]
    E_sub_eV: float
    E_int_tet_eV: float
    E_int_hex_eV: float  # hex for diamond, oct for fcc/hcp
    margin_eV: float
    verdict: str
    notes: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None

def predict_site_preference(
    host: str,
    dopant: str,
    mpr: Optional[Any] = None,
    material_id: Optional[str] = None,
    dft_formation_energies: Optional[Dict[str, float]] = None,
) -> SitePreferenceResult:
    """
    Decide whether dopant prefers substitutional or interstitial sites.

    Args:
      host: host element symbol (e.g., "Si")
      dopant: dopant element symbol (e.g., "P")
      mpr: optional MPRester; if given, we fetch a representative bulk structure
      material_id: optional MP id to disambiguate structure
      dft_formation_energies: optional dict with keys in
            {"sub","int_tet","int_oct"} and values (eV). If provided, these
            override heuristics.

    Returns:
      SitePreferenceResult with a clear verdict and margins.
    """
    host = host.strip().capitalize()
    dopant = dopant.strip().capitalize()

    # (A) If DFT energies are given, use them directly
    diagnostics = None
    if dft_formation_energies:
        E_sub = float(dft_formation_energies.get("sub", math.inf))
        E_tet = float(dft_formation_energies.get("int_tet", math.inf))
        E_hex = float(dft_formation_energies.get("int_oct", math.inf))  # accept "int_oct" for backwards compat
        method = "DFT-provided"
        struct = None
    else:
        # Try to fetch a structure
        struct = None
        method = "heuristic-generic"
        if mpr:
            try:
                if material_id:
                    docs = mpr.materials.summary.search(
                        material_ids=[material_id],
                        fields=["structure"],
                    )
                else:
                    docs = mpr.materials.summary.search(
                        formula=f"{host}",
                        fields=["structure", "formula_pretty", "is_stable", "energy_above_hull"],
                    )
                if docs:
                    struct = docs[0].structure
                    method = "heuristic-structure"
            except Exception as e:
                _log.warning(f"MP fetch failed; falling back to generic heuristic: {e}")

        scores = _score_sites(host, dopant, struct)
        E_sub, E_tet, E_hex = scores.Eproxy_sub_eV, scores.Eproxy_int_tet_eV, scores.Eproxy_int_hex_eV
        diagnostics = scores.diagnostics
        
        # Sanity check: if any proxy is unphysical, fall back to structure-free generic heuristic
        if not all(np.isfinite([E_sub, E_tet, E_hex])) or any(v > 100.0 for v in [E_sub, E_tet, E_hex]):
            _log.warning(f"Unphysical proxy values detected (sub={E_sub:.1e}, tet={E_tet:.1e}, hex={E_hex:.1e}); falling back to generic heuristic")
            scores = _score_sites(host, dopant, struct=None)  # generic, structure-free fallback
            E_sub, E_tet, E_hex = scores.Eproxy_sub_eV, scores.Eproxy_int_tet_eV, scores.Eproxy_int_hex_eV
            diagnostics = scores.diagnostics
            method = "heuristic-generic"

    # Pick preference and margin
    site_map = {
        "substitutional": E_sub,
        "interstitial-tetra": E_tet,
        "interstitial-hex": E_hex
    }
    preferred = min(site_map, key=site_map.get)
    others = [v for k, v in site_map.items() if k != preferred]
    margin = min([x - site_map[preferred] for x in others])

    # Craft verdict
    if preferred == "substitutional":
        verdict = (
            f"{dopant} prefers the substitutional site in {host} by ~{margin:.2f} eV "
            f"(vs interstitial). Interstitial configurations are predicted unstable/higher in energy."
        )
    else:
        verdict = (
            f"{dopant} tends to occupy {preferred} over substitutional in {host} by ~{margin:.2f} eV."
        )

    # Build notes with warnings from diagnostics
    notes = {"caveats": [
        "Proxies compare relative stabilities; absolute values are not DFT formation energies.",
        "If you have DFT or experimental defect energies, pass them in dft_formation_energies to override.",
        "Results are most reliable for covalent semiconductors (diamond/zincblende/wurtzite)."
    ]}
    
    if diagnostics and diagnostics.get("warnings"):
        notes["warnings"] = diagnostics["warnings"]
    
    return SitePreferenceResult(
        success=True,
        host=host,
        dopant=dopant,
        method=method,  # "DFT-provided", "heuristic-structure", or "heuristic-generic"
        preferred_site=preferred,
        E_sub_eV=float(E_sub),
        E_int_tet_eV=float(E_tet),
        E_int_hex_eV=float(E_hex),
        margin_eV=float(margin),
        verdict=verdict,
        notes=notes,
        diagnostics=diagnostics
    )
