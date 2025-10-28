"""
Utility functions for materials property analysis.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

_log = logging.getLogger(__name__)


def get_elastic_properties(mpr, material_id: str) -> Dict[str, Any]:
    """
    Get elastic/mechanical properties for a material.
    
    Args:
        mpr: MPRester client instance
        material_id: Material ID
        
    Returns:
        Dictionary with elastic properties including bulk modulus, shear modulus, etc.
    """
    try:
        docs = mpr.materials.summary.search(
            material_ids=material_id,
            fields=[
                "material_id", "formula_pretty", "composition",
                "bulk_modulus", "shear_modulus", "universal_anisotropy",
                "homogeneous_poisson", "energy_above_hull", "is_stable"
            ]
        )
        
        if not docs:
            return {
                "success": False,
                "error": f"Material {material_id} not found"
            }
        
        doc = docs[0]
        
        # Extract bulk modulus data
        bulk_modulus = doc.bulk_modulus if hasattr(doc, 'bulk_modulus') else None
        shear_modulus = doc.shear_modulus if hasattr(doc, 'shear_modulus') else None
        
        result = {
            "success": True,
            "material_id": material_id,
            "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
            "composition": dict(doc.composition.as_dict()) if hasattr(doc, 'composition') else None,
            "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
            "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None
        }
        
        if bulk_modulus:
            # Handle both dict and object formats
            if isinstance(bulk_modulus, dict):
                result["bulk_modulus"] = {
                    "k_vrh": float(bulk_modulus.get('vrh')) if bulk_modulus.get('vrh') is not None else None,
                    "k_voigt": float(bulk_modulus.get('voigt')) if bulk_modulus.get('voigt') is not None else None,
                    "k_reuss": float(bulk_modulus.get('reuss')) if bulk_modulus.get('reuss') is not None else None,
                    "unit": "GPa"
                }
            else:
                result["bulk_modulus"] = {
                    "k_vrh": float(bulk_modulus.vrh) if hasattr(bulk_modulus, 'vrh') and bulk_modulus.vrh is not None else None,
                    "k_voigt": float(bulk_modulus.voigt) if hasattr(bulk_modulus, 'voigt') and bulk_modulus.voigt is not None else None,
                    "k_reuss": float(bulk_modulus.reuss) if hasattr(bulk_modulus, 'reuss') and bulk_modulus.reuss is not None else None,
                    "unit": "GPa"
                }
        else:
            result["bulk_modulus"] = None
            
        if shear_modulus:
            # Handle both dict and object formats
            if isinstance(shear_modulus, dict):
                result["shear_modulus"] = {
                    "g_vrh": float(shear_modulus.get('vrh')) if shear_modulus.get('vrh') is not None else None,
                    "g_voigt": float(shear_modulus.get('voigt')) if shear_modulus.get('voigt') is not None else None,
                    "g_reuss": float(shear_modulus.get('reuss')) if shear_modulus.get('reuss') is not None else None,
                    "unit": "GPa"
                }
            else:
                result["shear_modulus"] = {
                    "g_vrh": float(shear_modulus.vrh) if hasattr(shear_modulus, 'vrh') and shear_modulus.vrh is not None else None,
                    "g_voigt": float(shear_modulus.voigt) if hasattr(shear_modulus, 'voigt') and shear_modulus.voigt is not None else None,
                    "g_reuss": float(shear_modulus.reuss) if hasattr(shear_modulus, 'reuss') and shear_modulus.reuss is not None else None,
                    "unit": "GPa"
                }
        else:
            result["shear_modulus"] = None
            
        if hasattr(doc, 'universal_anisotropy') and doc.universal_anisotropy is not None:
            result["universal_anisotropy"] = float(doc.universal_anisotropy)
        
        if hasattr(doc, 'homogeneous_poisson') and doc.homogeneous_poisson is not None:
            result["poisson_ratio"] = float(doc.homogeneous_poisson)
            
        return result
        
    except Exception as e:
        _log.error(f"Error getting elastic properties for {material_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def find_alloy_compositions(
    mpr,
    elements: List[str],
    target_composition: Optional[Dict[str, float]] = None,
    tolerance: float = 0.05,
    is_stable: bool = True,
    ehull_max: float = 0.20,
    require_binaries: bool = True
) -> Dict[str, Any]:
    """
    Find materials with specific alloy compositions.
    
    Args:
        mpr: MPRester client instance
        elements: List of elements in the alloy (e.g., ['Ag', 'Cu'])
        target_composition: Target atomic fractions (e.g., {'Ag': 0.875, 'Cu': 0.125})
        tolerance: Tolerance for composition matching
        is_stable: Whether to filter for stable materials only
        ehull_max: Maximum energy above hull for metastable entries (eV/atom)
        require_binaries: Whether to require exactly 2 elements
        
    Returns:
        Dictionary with matching materials
    """
    try:
        chemsys = "-".join(sorted(elements))
        
        search_kwargs = {
            "chemsys": chemsys,
            "fields": [
                "material_id", "formula_pretty", "composition",
                "energy_above_hull", "is_stable",
                "bulk_modulus", "shear_modulus"
            ]
        }
        
        if require_binaries:
            search_kwargs["num_elements"] = 2
        
        if is_stable:
            search_kwargs["energy_above_hull"] = (0, 1e-3)
        else:
            search_kwargs["energy_above_hull"] = (0, ehull_max)
        
        docs = mpr.materials.summary.search(**search_kwargs)
        
        if not docs:
            return {
                "success": False,
                "error": f"No materials found for {chemsys}"
            }
        
        materials = []
        all_candidates = []  # Track all for closest-match fallback
        
        for doc in docs:
            comp = doc.composition
            comp_dict = comp.as_dict()
            
            # Calculate atomic fractions
            total_atoms = sum(comp_dict.values())
            fractions = {el: count/total_atoms for el, count in comp_dict.items()}
            
            # Check if composition matches target
            matches_target = True
            max_deviation = 0.0
            if target_composition:
                for el, target_frac in target_composition.items():
                    actual_frac = fractions.get(el, 0.0)
                    deviation = abs(actual_frac - target_frac)
                    max_deviation = max(max_deviation, deviation)
                    if deviation > tolerance:
                        matches_target = False
                
                _log.debug(f"Composition check: {comp.reduced_formula} - Target: {target_composition}, Actual: {fractions}, Max deviation: {max_deviation:.4f}, Tolerance: {tolerance}, Match: {matches_target}")
            
            mat_info = {
                "material_id": doc.material_id if hasattr(doc, 'material_id') else None,
                "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(comp),
                "composition": comp_dict,
                "atomic_fractions": fractions,
                "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None,
                "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
                "max_composition_deviation": max_deviation if target_composition else None
            }
            
            # Add elastic properties if available
            if hasattr(doc, 'bulk_modulus') and doc.bulk_modulus:
                bm = doc.bulk_modulus
                if isinstance(bm, dict):
                    mat_info["bulk_modulus_vrh"] = float(bm.get('vrh')) if bm.get('vrh') is not None else None
                else:
                    mat_info["bulk_modulus_vrh"] = float(bm.vrh) if hasattr(bm, 'vrh') and bm.vrh is not None else None
            
            all_candidates.append(mat_info)
            
            if not target_composition or matches_target:
                materials.append(mat_info)
        
        # Helper: composition distance
        def composition_distance(mat):
            fracs = mat["atomic_fractions"]
            return sum(abs(fracs.get(el, 0.0) - target_composition.get(el, 0.0)) 
                      for el in target_composition)
        
        # Sort materials by composition distance (best match first)
        if target_composition and materials:
            materials.sort(key=composition_distance)
        
        # If no exact matches but we have a target, use closest match
        closest_match_used = False
        best_l1_distance = None
        max_deviation = None
        
        if target_composition and not materials and all_candidates:
            _log.info(f"No materials within tolerance {tolerance}, using closest match")
            closest = min(all_candidates, key=composition_distance)
            closest["closest_match"] = True
            materials = [closest]
            closest_match_used = True
            best_l1_distance = composition_distance(closest)
            max_deviation = closest.get("max_composition_deviation")
            _log.info(f"Using closest match: {closest['formula']} with L1 distance {best_l1_distance:.4f}")
        elif target_composition and materials:
            best_l1_distance = composition_distance(materials[0])
            max_deviation = max(abs(materials[0]["atomic_fractions"].get(el, 0.0) - target_composition.get(el, 0.0))
                               for el in target_composition)
        
        return {
            "success": True,
            "chemical_system": chemsys,
            "target_composition": target_composition,
            "tolerance": tolerance,
            "require_binaries": require_binaries,
            "ehull_window_eV": (0, 1e-3) if is_stable else (0, ehull_max),
            "num_materials_found": len(materials),
            "closest_match_used": closest_match_used,
            "max_composition_deviation": max_deviation,
            "l1_distance_to_target": best_l1_distance,
            "materials": materials
        }
        
    except Exception as e:
        _log.error(f"Error finding alloy compositions: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def compare_material_properties(
    property1: Dict[str, Any],
    property2: Dict[str, Any],
    property_name: str = "bulk_modulus"
) -> Dict[str, Any]:
    """
    Compare a specific property between two materials.
    
    Args:
        property1: Properties of first material (from get_elastic_properties)
        property2: Properties of second material (from get_elastic_properties)
        property_name: Name of property to compare
        
    Returns:
        Dictionary with comparison results
    """
    try:
        if not property1.get("success") or not property2.get("success"):
            return {
                "success": False,
                "error": "One or both materials missing property data"
            }
        
        # Extract the property values
        val1 = None
        val2 = None
        
        if property_name == "bulk_modulus":
            if property1.get("bulk_modulus") and property2.get("bulk_modulus"):
                val1 = property1["bulk_modulus"].get("k_vrh")
                val2 = property2["bulk_modulus"].get("k_vrh")
        elif property_name == "shear_modulus":
            if property1.get("shear_modulus") and property2.get("shear_modulus"):
                val1 = property1["shear_modulus"].get("g_vrh")
                val2 = property2["shear_modulus"].get("g_vrh")
        elif property_name in property1 and property_name in property2:
            val1 = property1.get(property_name)
            val2 = property2.get(property_name)
        
        if val1 is None or val2 is None:
            return {
                "success": False,
                "error": f"Property '{property_name}' not available for one or both materials"
            }
        
        # Calculate differences
        absolute_diff = val2 - val1
        percent_change = (absolute_diff / val1) * 100.0 if val1 != 0 else None
        
        # Determine unit
        unit = None
        if property_name in ("bulk_modulus", "shear_modulus"):
            unit = "GPa"
        
        result = {
            "success": True,
            "property_name": property_name,
            "material1": {
                "id": property1.get("material_id"),
                "formula": property1.get("formula"),
                "value": float(val1),
                "unit": unit
            },
            "material2": {
                "id": property2.get("material_id"),
                "formula": property2.get("formula"),
                "value": float(val2),
                "unit": unit
            },
            "comparison": {
                "absolute_difference": float(absolute_diff),
                "percent_change": float(percent_change) if percent_change is not None else None,
                "ratio": float(val2 / val1) if val1 != 0 else None,
                "unit": unit
            },
            "interpretation": None
        }
        
        # Add interpretation
        if percent_change is not None:
            if abs(percent_change) < 1:
                result["interpretation"] = "Negligible change"
            elif percent_change > 0:
                result["interpretation"] = f"Material 2 has {abs(percent_change):.1f}% higher {property_name}"
            else:
                result["interpretation"] = f"Material 2 has {abs(percent_change):.1f}% lower {property_name}"
        
        return result
        
    except Exception as e:
        _log.error(f"Error comparing properties: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def analyze_doping_effect(
    mpr,
    host_element: str,
    dopant_element: str,
    dopant_concentration: float,
    property_name: str = "bulk_modulus"
) -> Dict[str, Any]:
    """
    Analyze the effect of doping on material properties.
    
    Args:
        mpr: MPRester client instance
        host_element: Host element (e.g., 'Ag')
        dopant_element: Dopant element (e.g., 'Cu')
        dopant_concentration: Dopant atomic fraction (e.g., 0.125 for 12.5%)
        property_name: Property to analyze
        
    Returns:
        Dictionary with doping analysis results
    """
    try:
        # Find pure host material
        host_docs = mpr.materials.summary.search(
            elements=[host_element],
            num_elements=[1, 1],
            is_stable=True,
            fields=[
                "material_id", "formula_pretty", "composition",
                "bulk_modulus", "shear_modulus", "energy_above_hull", "is_stable"
            ]
        )
        
        if not host_docs:
            return {
                "success": False,
                "error": f"Could not find pure {host_element} in database"
            }
        
        # Get the first stable entry
        host_doc = host_docs[0]
        host_id = host_doc.material_id if hasattr(host_doc, 'material_id') else None
        
        if not host_id:
            return {
                "success": False,
                "error": f"Could not get material ID for pure {host_element}"
            }
        
        # Get pure host material properties
        host_props = get_elastic_properties(mpr, host_id)
        
        # Find doped materials
        target_comp = {
            host_element: 1.0 - dopant_concentration,
            dopant_element: dopant_concentration
        }
        
        # Try stable entries first
        alloys = find_alloy_compositions(
            mpr,
            [host_element, dopant_element],
            target_composition=target_comp,
            tolerance=0.05,  # Strict tolerance: ±5 at%
            is_stable=True,
            require_binaries=True
        )
        
        # Fallback: allow metastable entries if no stable ones found
        used_metastable = False
        if not alloys.get("success") or not alloys.get("materials"):
            _log.info(f"No stable {host_element}-{dopant_element} alloys found, trying metastable entries (Ehull ≤ 0.20 eV/atom)")
            alloys = find_alloy_compositions(
                mpr,
                [host_element, dopant_element],
                target_composition=target_comp,
                tolerance=0.05,  # Keep strict tolerance
                is_stable=False,
                ehull_max=0.20,
                require_binaries=True
            )
            used_metastable = True
        
        # Cache for pure dopant properties (avoid duplicate queries)
        _cached_dopant_props = [None]  # Use list for mutability in closure
        
        def get_pure_dopant_props():
            """Get pure dopant properties (cached)."""
            if _cached_dopant_props[0] is None:
                dopant_docs = mpr.materials.summary.search(
                    elements=[dopant_element],
                    num_elements=1,
                    is_stable=True,
                    fields=["material_id", "formula_pretty", "bulk_modulus", "shear_modulus"]
                )
                if dopant_docs:
                    dopant_id = dopant_docs[0].material_id
                    _cached_dopant_props[0] = get_elastic_properties(mpr, dopant_id)
            return _cached_dopant_props[0]
        
        # If no alloys found, compute VRH estimate from pure elements
        if not alloys.get("success") or not alloys.get("materials"):
            _log.info(f"No {host_element}-{dopant_element} alloys found, computing VRH mixture estimate")
            
            dopant_props = get_pure_dopant_props()
            
            if not dopant_props or not dopant_props.get("success"):
                return {
                    "success": False,
                    "error": (
                        f"No {host_element}-{dopant_element} entries found near {dopant_concentration*100:.1f}% "
                        f"{dopant_element} within Ehull ≤ 0.20 eV/atom, and could not find pure {dopant_element} "
                        f"for mixture model fallback."
                    )
                }
            
            # Check if both have bulk modulus data
            if (host_props.get("bulk_modulus") and host_props["bulk_modulus"].get("k_vrh") and
                dopant_props.get("bulk_modulus") and dopant_props["bulk_modulus"].get("k_vrh")):
                
                K_host = host_props["bulk_modulus"]["k_vrh"]
                K_dopant = dopant_props["bulk_modulus"]["k_vrh"]
                x = dopant_concentration
                
                # Voigt-Reuss-Hill bounds
                K_V = (1 - x) * K_host + x * K_dopant  # Voigt (upper bound)
                K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)  # Reuss (lower bound)
                K_est = 0.5 * (K_V + K_R)  # VRH average
                pct = 100 * (K_est - K_host) / K_host
                
                return {
                    "success": True,
                    "host_element": host_element,
                    "dopant_element": dopant_element,
                    "target_dopant_concentration": dopant_concentration,
                    "property_analyzed": property_name,
                    "pure_host": {
                        "material_id": host_props.get("material_id"),
                        "formula": host_props.get("formula"),
                        "bulk_modulus_vrh": float(K_host),
                        "unit": "GPa"
                    },
                    "pure_dopant": {
                        "material_id": dopant_props.get("material_id"),
                        "formula": dopant_props.get("formula"),
                        "bulk_modulus_vrh": float(K_dopant),
                        "unit": "GPa"
                    },
                    "num_alloys_analyzed": 0,
                    "used_metastable_entries": False,
                    "used_mixture_model": True,
                    "mixture_model_estimate": {
                        "method": "Voigt-Reuss-Hill bounds from pure elements",
                        "k_vrh_gpa": float(K_est),
                        "k_voigt_gpa": float(K_V),
                        "k_reuss_gpa": float(K_R),
                        "percent_change": float(pct),
                        "unit": "GPa"
                    },
                    "comparisons": [],
                    "notes": [
                        f"No {host_element}-{dopant_element} alloy entries found within Ehull ≤ 0.20 eV/atom.",
                        f"Used Voigt-Reuss-Hill mixture model with pure {host_element} and {dopant_element}.",
                        f"This provides an estimate assuming ideal mixing; actual alloys may deviate."
                    ],
                    "citations": ["Materials Project", "pymatgen"]
                }
            else:
                return {
                    "success": False,
                    "error": (
                        f"No {host_element}-{dopant_element} entries found near {dopant_concentration*100:.1f}% "
                        f"{dopant_element} within Ehull ≤ 0.20 eV/atom, and elastic data incomplete "
                        f"for mixture model fallback."
                    )
                }
        
        # Get properties for each alloy and compare
        comparisons = []
        for alloy in alloys["materials"]:
            alloy_id = alloy["material_id"]
            alloy_props = get_elastic_properties(mpr, alloy_id)
            
            if alloy_props.get("success"):
                comparison = compare_material_properties(
                    host_props,
                    alloy_props,
                    property_name
                )
                if comparison.get("success"):
                    comparison["alloy_composition"] = alloy["atomic_fractions"]
                    comparison["alloy_energy_above_hull"] = alloy.get("energy_above_hull")
                    comparison["is_closest_match"] = alloy.get("closest_match", False)
                    comparison["composition_deviation"] = alloy.get("max_composition_deviation")
                    
                    # Add requested vs actual composition info
                    comparison["requested_composition"] = target_comp
                    comparison["actual_composition"] = alloy["atomic_fractions"]
                    
                    comparisons.append(comparison)
        
        # Always compute VRH estimate at exact composition for comparison
        vrh_estimate = None
        if (host_props.get("bulk_modulus") and host_props["bulk_modulus"].get("k_vrh")):
            # Get pure dopant properties (cached)
            dopant_props = get_pure_dopant_props()
            
            if (dopant_props and dopant_props.get("success") and 
                dopant_props.get("bulk_modulus") and dopant_props["bulk_modulus"].get("k_vrh")):
                
                K_host = host_props["bulk_modulus"]["k_vrh"]
                K_dopant = dopant_props["bulk_modulus"]["k_vrh"]
                x = dopant_concentration
                
                # Voigt-Reuss-Hill bounds (rigorous for isotropic aggregates)
                K_V = (1 - x) * K_host + x * K_dopant  # Voigt (upper bound)
                K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)  # Reuss (lower bound)
                K_VRH = 0.5 * (K_V + K_R)  # VRH average
                
                pct_voigt = 100 * (K_V - K_host) / K_host
                pct_reuss = 100 * (K_R - K_host) / K_host
                pct_vrh = 100 * (K_VRH - K_host) / K_host
                
                vrh_estimate = {
                    "method": "Voigt-Reuss-Hill bounds from pure elements",
                    "pure_host_k_vrh": float(K_host),
                    "pure_dopant_k_vrh": float(K_dopant),
                    "dopant_concentration": dopant_concentration,
                    "k_voigt_gpa": float(K_V),
                    "k_reuss_gpa": float(K_R),
                    "k_vrh_gpa": float(K_VRH),
                    "percent_change_voigt": float(pct_voigt),
                    "percent_change_reuss": float(pct_reuss),
                    "percent_change_vrh": float(pct_vrh),
                    "unit": "GPa",
                    "note": f"VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element}; actual value lies between Reuss (lower) and Voigt (upper)"
                }
        
        if not comparisons:
            if vrh_estimate:
                # Return VRH estimate as the result
                return {
                    "success": True,
                    "host_element": host_element,
                    "dopant_element": dopant_element,
                    "target_dopant_concentration": dopant_concentration,
                    "requested_composition": target_comp,
                    "property_analyzed": property_name,
                    "pure_host": {
                        "material_id": host_props.get("material_id"),
                        "formula": host_props.get("formula"),
                        "bulk_modulus_vrh": vrh_estimate["pure_host_k_vrh"],
                        "unit": "GPa"
                    },
                    "num_alloys_analyzed": 0,
                    "used_metastable_entries": False,
                    "used_mixture_model": True,
                    "vrh_estimate": vrh_estimate,
                    "comparisons": [],
                    "notes": [
                        f"No database entries found at {dopant_concentration*100:.1f}% {dopant_element}.",
                        f"Using VRH bounds from pure elements as estimate.",
                        f"VRH predicts {vrh_estimate['percent_change_vrh']:.1f}% change at exact composition "
                        f"(range: {vrh_estimate['percent_change_reuss']:.1f}% to {vrh_estimate['percent_change_voigt']:.1f}%)."
                    ],
                    "citations": ["Materials Project", "pymatgen"]
                }
            else:
                return {
                    "success": False,
                    "error": "Could not compare properties for any alloys and VRH estimate unavailable"
                }
        
        result = {
            "success": True,
            "host_element": host_element,
            "dopant_element": dopant_element,
            "target_dopant_concentration": dopant_concentration,
            "requested_composition": target_comp,
            "property_analyzed": property_name,
            "pure_host": {
                "material_id": host_props.get("material_id"),
                "formula": host_props.get("formula")
            },
            "num_alloys_analyzed": len(comparisons),
            "used_metastable_entries": used_metastable,
            "used_closest_match": alloys.get("closest_match_used", False),
            "vrh_estimate": vrh_estimate,
            "comparisons": comparisons,
            "notes": []
        }
        
        if used_metastable:
            result["notes"].append(
                f"No stable {host_element}-{dopant_element} compounds found on convex hull. "
                f"Used metastable entries (Ehull ≤ 0.20 eV/atom) which may represent "
                f"solid solutions or ordered structures."
            )
        
        if alloys.get("closest_match_used") and comparisons:
            closest_comp = comparisons[0]["actual_composition"]
            req_comp_str = ", ".join(f"{el}{target_comp[el]:.3f}" for el in sorted(target_comp.keys()))
            actual_comp_str = ", ".join(f"{el}{closest_comp[el]:.3f}" for el in sorted(closest_comp.keys()))
            l1_dist = alloys.get("l1_distance_to_target", alloys.get("max_composition_deviation", 0))
            result["notes"].insert(0, 
                f"Requested: {req_comp_str}; using closest DB entry: {actual_comp_str} (Δ={l1_dist:.3f})."
            )
        
        if vrh_estimate:
            result["notes"].append(
                f"VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element}: "
                f"{vrh_estimate['percent_change_vrh']:.1f}% (VRH average), "
                f"range {vrh_estimate['percent_change_reuss']:.1f}% (Reuss lower) to "
                f"{vrh_estimate['percent_change_voigt']:.1f}% (Voigt upper). "
                f"These rigorous bounds cap the maximum possible change at this composition."
            )
        
        # Add summary statistics
        percent_changes = [c["comparison"]["percent_change"] for c in comparisons if c["comparison"].get("percent_change") is not None]
        if percent_changes:
            result["summary"] = {
                "avg_percent_change": float(np.mean(percent_changes)),
                "min_percent_change": float(np.min(percent_changes)),
                "max_percent_change": float(np.max(percent_changes)),
                "std_percent_change": float(np.std(percent_changes)) if len(percent_changes) > 1 else 0.0
            }
        
        return result
        
    except Exception as e:
        _log.error(f"Error analyzing doping effect: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

