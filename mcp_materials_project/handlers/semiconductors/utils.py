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
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
    Compare magnetic properties between undoped and doped materials.
    
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
            }
        }
        
        # Compare magnetization
        if "total_magnetization" in undoped_props and "total_magnetization" in doped_props:
            m1 = undoped_props["total_magnetization"]["value"]
            m2 = doped_props["total_magnetization"]["value"]
            
            result["magnetization_comparison"] = {
                "undoped": float(m1),
                "doped": float(m2),
                "absolute_change": float(m2 - m1),
                "percent_change": float((m2 - m1) / m1 * 100) if m1 != 0 else None,
                "unit": "μB"
            }
        
        # Compare magnetization per formula unit
        if "magnetization_per_formula_unit" in undoped_props and "magnetization_per_formula_unit" in doped_props:
            m1 = undoped_props["magnetization_per_formula_unit"]["value"]
            m2 = doped_props["magnetization_per_formula_unit"]["value"]
            
            result["magnetization_per_fu_comparison"] = {
                "undoped": float(m1),
                "doped": float(m2),
                "absolute_change": float(m2 - m1),
                "percent_change": float((m2 - m1) / m1 * 100) if m1 != 0 else None,
                "unit": "μB/f.u."
            }
        
        # Interpretation
        improved = False
        if "magnetization_comparison" in result:
            improved = bool(result["magnetization_comparison"]["absolute_change"] > 0)
        elif "magnetization_per_fu_comparison" in result:
            improved = bool(result["magnetization_per_fu_comparison"]["absolute_change"] > 0)
        
        result["magnetic_enhancement"] = bool(improved)
        
        if improved:
            result["interpretation"] = "Doping enhanced magnetic properties"
        else:
            result["interpretation"] = "Doping reduced magnetic properties"
        
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
            _limit=20
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
    Analyze doping site preference for compound semiconductors.
    
    For example, N doping in GaAs: does N prefer Ga sites or As sites?
    
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
    try:
        # Get host material
        host_docs = mpr.materials.summary.search(
            formula=host_formula,
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "structure", "is_stable"
            ]
        )
        
        if not host_docs:
            return {
                "success": False,
                "error": f"Host material {host_formula} not found"
            }
        
        # Get the most stable entry
        host_doc = sorted(host_docs, key=lambda x: getattr(x, 'energy_above_hull', float('inf')))[0]
        host_energy = host_doc.energy_per_atom if hasattr(host_doc, 'energy_per_atom') else None
        
        if host_energy is None:
            return {
                "success": False,
                "error": "Host material energy not available"
            }
        
        result = {
            "success": True,
            "host_material": {
                "material_id": host_doc.material_id if hasattr(host_doc, 'material_id') else None,
                "formula": host_doc.formula_pretty if hasattr(host_doc, 'formula_pretty') else host_formula,
                "energy_per_atom": float(host_energy),
                "unit": "eV/atom"
            },
            "dopant_element": dopant_element,
            "site_a": site_a_element,
            "site_b": site_b_element,
            "temperature_K": temperature,
            "pressure_atm": pressure
        }
        
        # Search for materials with dopant substituting each site
        # For GaAs with N: look for NGa (N on Ga site) and NAs (N on As site)
        
        # Try to find materials with dopant
        elements = [site_a_element, site_b_element, dopant_element]
        
        # Search for ternary compounds
        doped_docs = mpr.materials.summary.search(
            elements=elements,
            num_elements=3,
            fields=[
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "energy_above_hull", "is_stable"
            ],
            _limit=50
        )
        
        # Analyze compositions to identify substitution sites
        site_a_candidates = []  # Dopant on site A (replaces site_a_element)
        site_b_candidates = []  # Dopant on site B (replaces site_b_element)
        
        for doc in doped_docs:
            comp = doc.composition.as_dict()
            
            # Check if this could be dopant on site A or B
            # Site A: more of site_b_element, less of site_a_element
            # Site B: more of site_a_element, less of site_b_element
            
            a_count = comp.get(site_a_element, 0)
            b_count = comp.get(site_b_element, 0)
            d_count = comp.get(dopant_element, 0)
            
            # Simple heuristic: if dopant count is close to one of the sites
            if d_count > 0:
                if a_count < b_count:  # Dopant likely on A site
                    site_a_candidates.append({
                        "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                        "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
                        "composition": comp,
                        "energy_per_atom": float(doc.energy_per_atom) if hasattr(doc, 'energy_per_atom') and doc.energy_per_atom else None,
                        "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None
                    })
                elif b_count < a_count:  # Dopant likely on B site
                    site_b_candidates.append({
                        "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                        "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
                        "composition": comp,
                        "energy_per_atom": float(doc.energy_per_atom) if hasattr(doc, 'energy_per_atom') and doc.energy_per_atom else None,
                        "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None
                    })
        
        # Find the most stable candidate for each site
        if site_a_candidates:
            site_a_best = min(site_a_candidates, key=lambda x: x.get("energy_above_hull", float('inf')))
            result["site_a_substitution"] = site_a_best
        else:
            result["site_a_substitution"] = None
        
        if site_b_candidates:
            site_b_best = min(site_b_candidates, key=lambda x: x.get("energy_above_hull", float('inf')))
            result["site_b_substitution"] = site_b_best
        else:
            result["site_b_substitution"] = None
        
        # Compare energies to determine preference
        if result["site_a_substitution"] and result["site_b_substitution"]:
            e_a = result["site_a_substitution"]["energy_above_hull"]
            e_b = result["site_b_substitution"]["energy_above_hull"]
            
            if e_a is not None and e_b is not None:
                result["site_preference"] = {
                    "preferred_site": site_a_element if e_a < e_b else site_b_element,
                    "energy_difference": float(abs(e_a - e_b)),
                    "unit": "eV/atom",
                    "interpretation": f"Dopant prefers {site_a_element} sites (more stable by {abs(e_a - e_b):.3f} eV/atom)" if e_a < e_b else f"Dopant prefers {site_b_element} sites (more stable by {abs(e_a - e_b):.3f} eV/atom)"
                }
                
                result["site_a_more_stable"] = bool(e_a < e_b)
            else:
                result["site_preference"] = None
                result["note"] = "Energy comparison not possible (missing energy data)"
        else:
            result["site_preference"] = None
            result["note"] = "Could not find structures for both substitution sites"
        
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

