"""
Utility functions for materials property analysis.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import math

from ..shared import success_result, error_result, ErrorType, Confidence
from ..shared.api_utils import format_field_error

_log = logging.getLogger(__name__)

BULK_MODULI = [
    {"name": "Acetone", "symbol": None, "bulk_modulus_GPa": 0.92},
    {"name": "Alumina", "symbol": None, "bulk_modulus_GPa": [148, 176]},
    {"name": "Aluminum", "symbol": "AL", "bulk_modulus_GPa": [68, 70]},
    {"name": "Benzene", "symbol": None, "bulk_modulus_GPa": 1.05},
    {"name": "Boron Carbide (B4C)", "symbol": None, "bulk_modulus_GPa": [200, 240]},
    {"name": "Brass - cast", "symbol": None, "bulk_modulus_GPa": 116},
    {"name": "Brass (70 - 30)", "symbol": None, "bulk_modulus_GPa": 108},
    {"name": "Carbon Tetrachloride", "symbol": None, "bulk_modulus_GPa": 1.32},
    {"name": "Chalk", "symbol": None, "bulk_modulus_GPa": 9},
    {"name": "Concrete", "symbol": None, "bulk_modulus_GPa": [6, 28]},
    {"name": "Copper", "symbol": "CU", "bulk_modulus_GPa": 123},
    {"name": "Diamond", "symbol": "C", "bulk_modulus_GPa": [540, 640]},
    {"name": "Elektron (magnesium alloy)", "symbol": "MG", "bulk_modulus_GPa": 33},
    {"name": "Ethyl Alcohol", "symbol": None, "bulk_modulus_GPa": 1.06},
    {"name": "Gasoline", "symbol": None, "bulk_modulus_GPa": 1.3},
    {"name": "Glass", "symbol": None, "bulk_modulus_GPa": [35, 55]},
    {"name": "Glycerine / Glycerol", "symbol": None, "bulk_modulus_GPa": 4.35},
    {"name": "Granite", "symbol": None, "bulk_modulus_GPa": 50},
    {"name": "Graphite - 2H (single crystal)", "symbol": "C", "bulk_modulus_GPa": 34},
    {"name": "Hexane", "symbol": None, "bulk_modulus_GPa": 0.6},
    {"name": "Iron - cast", "symbol": "FE", "bulk_modulus_GPa": [58, 107]},
    {"name": "Iron - malleable", "symbol": "FE", "bulk_modulus_GPa": 119},
    {"name": "ISO 32 Mineral Oil", "symbol": None, "bulk_modulus_GPa": 1.8},
    {"name": "Kerosene", "symbol": None, "bulk_modulus_GPa": 1.3},
    {"name": "Limestone", "symbol": None, "bulk_modulus_GPa": 65},
    {"name": "Magnesium alloy", "symbol": "MG", "bulk_modulus_GPa": 33.1},
    {"name": "Mercury", "symbol": "HG", "bulk_modulus_GPa": 28.5},
    {"name": "Methanol", "symbol": None, "bulk_modulus_GPa": 0.82},
    {"name": "Monel metal", "symbol": None, "bulk_modulus_GPa": 155},
    {"name": "Olive Oil", "symbol": None, "bulk_modulus_GPa": 1.6},
    {"name": "Paraffin Oil", "symbol": None, "bulk_modulus_GPa": 1.64},
    {"name": "Petrol", "symbol": None, "bulk_modulus_GPa": [1.07, 1.49]},
    {"name": "Phosphate Ester", "symbol": None, "bulk_modulus_GPa": 3},
    {"name": "Phosphor bronze", "symbol": "CU", "bulk_modulus_GPa": 112},
    {"name": "SAE 30 Oil", "symbol": None, "bulk_modulus_GPa": 1.5},
    {"name": "Sandstone", "symbol": None, "bulk_modulus_GPa": 0.7},
    {"name": "Seawater", "symbol": None, "bulk_modulus_GPa": 2.34},
    {"name": "Shale", "symbol": None, "bulk_modulus_GPa": 10},
    {"name": "Silicone Rubber", "symbol": None, "bulk_modulus_GPa": [1.5, 2]},
    {"name": "Silver", "symbol": "AG", "bulk_modulus_GPa": [96, 106]},
    {"name": "Sodium chloride", "symbol": "NA", "bulk_modulus_GPa": 24.42},
    {"name": "Solder (Tin-Lead)", "symbol": None, "bulk_modulus_GPa": [33, 58]},
    {"name": "Stainless steel (18 - 8)", "symbol": "FE", "bulk_modulus_GPa": 163},
    {"name": "Steel", "symbol": "FE", "bulk_modulus_GPa": [156, 165]},
    {"name": "Steel - cast", "symbol": "FE", "bulk_modulus_GPa": 139},
    {"name": "Steel - cold rolled", "symbol": "FE", "bulk_modulus_GPa": 159},
    {"name": "Sulfuric Acid", "symbol": "S", "bulk_modulus_GPa": 3.0},
    {"name": "Tin", "symbol": "SN", "bulk_modulus_GPa": [42, 60]},
    {"name": "Tobin bronze", "symbol": "CU", "bulk_modulus_GPa": 112},
    {"name": "Toluene / Toluol", "symbol": None, "bulk_modulus_GPa": 1.09},
    {"name": "Tungsten", "symbol": "W", "bulk_modulus_GPa": [307, 314]},
    {"name": "Turpentine", "symbol": None, "bulk_modulus_GPa": 1.28},
    {"name": "Water", "symbol": "H2O", "bulk_modulus_GPa": 2.15},
    {"name": "Water - Glycol", "symbol": None, "bulk_modulus_GPa": 3.4}
]

def _safe_ratio(K, G, eps=1e-12):
    """
    Compute Poisson ratio from bulk modulus K and shear modulus G.
    Avoids division by zero and handles None values.
    
    Args:
        K: Bulk modulus (GPa)
        G: Shear modulus (GPa)
        
    Returns:
        Poisson ratio or None if calculation not possible
    """
    # avoid division by ~zero
    if K is None or G is None:
        return None
    denom = 2.0 * (3.0*K + G)
    if not (math.isfinite(K) and math.isfinite(G) and math.isfinite(denom)):
        return None
    if abs(denom) < eps:
        return None
    return (3.0*K - 2.0*G) / denom


def get_elastic_properties(
    mpr, 
    element: Optional[str] = None,
    formula: Optional[str] = None,
    chemsys: Optional[str] = None,
    spacegroup_number: Optional[int] = None,
    crystal_system: Optional[str] = None,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Get elastic/mechanical properties for a material by composition + structure.
    
    Args:
        mpr: MPRester client instance
        element: Element(s) or comma-separated list (e.g., "Li,Fe,O") - REQUIRED (or formula/chemsys)
        formula: Formula (e.g., "Li2FeO3", "Fe2O3") - REQUIRED (or element/chemsys)
        chemsys: Chemical system (e.g., "Li-Fe-O") - REQUIRED (or element/formula)
        spacegroup_number: Spacegroup number (STRONGLY RECOMMENDED to narrow search)
        crystal_system: Crystal system (Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic) (STRONGLY RECOMMENDED)
        eps: Epsilon value for numerical stability in division checks (default: 1e-12)
        
    Returns:
        Dictionary with elastic properties including bulk modulus, shear modulus, etc.
        Includes derived properties (Poisson ratio, Young's modulus, Pugh ratio),
        mechanical stability assessment, and optional tensor-based recomputation.
        Also includes precalculated bulk moduli from reference data when available.
    """
    try:
        # Validate input modes
        has_composition = any([element, formula, chemsys])
        has_structure = spacegroup_number is not None and crystal_system is not None
        
        if not has_composition:
            return error_result(
                handler="materials",
                function="get_elastic_properties",
                error="Must provide at least one of: element, formula, or chemsys",
                error_type=ErrorType.INVALID_INPUT,
                citations=["Materials Project"]
            )
        
        # Track warnings
        warnings = []
        if not has_structure:
            warnings.append("Structure parameters (spacegroup_number and crystal_system) not provided. Results may include multiple materials with different structures.")
        
        # Extract element symbols for matching against BULK_MODULI
        element_symbols = []
        if element:
            element_symbols = [e.strip().upper() for e in element.split(',') if e.strip()]
        elif chemsys:
            element_symbols = [e.strip().upper() for e in chemsys.split('-') if e.strip()]
        elif formula:
            # Parse formula to extract element symbols (simple regex approach)
            import re
            element_symbols = [e.upper() for e in re.findall(r'([A-Z][a-z]?)', formula)]
        
        # Match against precalculated BULK_MODULI
        precalc_bulk_moduli = []
        for elem_symbol in element_symbols:
            for entry in BULK_MODULI:
                if entry.get("symbol") == elem_symbol:
                    precalc_bulk_moduli.append({
                        "element": elem_symbol,
                        "name": entry["name"],
                        "bulk_modulus_GPa": entry["bulk_modulus_GPa"]
                    })
        
        # Build search parameters
        search_params = {
            "fields": [
                "material_id", "formula_pretty", "composition",
                "bulk_modulus", "shear_modulus", "universal_anisotropy",
                "homogeneous_poisson", "energy_above_hull", "is_stable"
            ]
        }
        
        if element:
            # Convert comma-separated string to list of elements
            # MP API expects a list: ['Al', 'Fe'] not a string 'Al,Fe'
            search_params["elements"] = [e.strip() for e in element.split(',') if e.strip()]
        if formula:
            search_params["formula"] = formula
        if chemsys:
            search_params["chemsys"] = chemsys
        if has_structure:
            search_params["spacegroup_number"] = spacegroup_number
            search_params["crystal_system"] = crystal_system
        search_params["theoretical"] = False
        
        docs = mpr.materials.summary.search(**search_params)
        
        if not docs:
            search_desc = []
            if element:
                search_desc.append(f"element={element}")
            if formula:
                search_desc.append(f"formula={formula}")
            if chemsys:
                search_desc.append(f"chemsys={chemsys}")
            if has_structure:
                search_desc.append(f"spacegroup={spacegroup_number}")
                search_desc.append(f"crystal_system={crystal_system}")
            search_desc.append("theoretical=False")
            error_msg = f"No materials found matching criteria: {', '.join(search_desc)}"
        
            return error_result(
                handler="materials",
                function="get_elastic_properties",
                error=error_msg,
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
        # Sort by material_id (smaller IDs first - more likely experimental)
        def extract_mp_number(mat_id):
            """Extract numeric part from mp-123 -> 123"""
            try:
                if hasattr(mat_id, 'string'):
                    mat_id = mat_id.string
                return int(str(mat_id).split('-')[-1])
            except:
                return float('inf')
        
        docs = sorted(docs, key=lambda d: extract_mp_number(d.material_id if hasattr(d, 'material_id') else 'mp-999999'))
        
        # If multiple materials found, try each until we find one with elasticity data
        if len(docs) > 1:
            _log.info(f"Found {len(docs)} materials matching criteria, sorted by ID (earliest first). Will try each until elasticity data found.")
        
        # Try each doc until we find one with elasticity data
        doc = None
        actual_material_id = None
        bulk_modulus = None
        shear_modulus = None
        
        for candidate_doc in docs:
            candidate_id = candidate_doc.material_id if hasattr(candidate_doc, 'material_id') else None
            candidate_bulk = candidate_doc.bulk_modulus if hasattr(candidate_doc, 'bulk_modulus') else None
            candidate_shear = candidate_doc.shear_modulus if hasattr(candidate_doc, 'shear_modulus') else None
            
            # Check if this doc has elasticity data
            if candidate_bulk is not None or candidate_shear is not None:
                doc = candidate_doc
                actual_material_id = candidate_id
                bulk_modulus = candidate_bulk
                shear_modulus = candidate_shear
                if len(docs) > 1:
                    _log.info(f"Using {candidate_id} (has elasticity data)")
                break
            else:
                _log.debug(f"Skipping {candidate_id} (no elasticity data in summary)")
        
        # If no doc with elasticity data found, use first doc anyway and try elasticity endpoint
        if doc is None:
            _log.info(f"No materials had elasticity data in summary; using first material {docs[0].material_id}")
            doc = docs[0]
            actual_material_id = doc.material_id if hasattr(doc, 'material_id') else None
            bulk_modulus = doc.bulk_modulus if hasattr(doc, 'bulk_modulus') else None
            shear_modulus = doc.shear_modulus if hasattr(doc, 'shear_modulus') else None
        
        data = {
            "material_id": actual_material_id,
            "formula": doc.formula_pretty if hasattr(doc, 'formula_pretty') else str(doc.composition),
            "composition": dict(doc.composition.as_dict()) if hasattr(doc, 'composition') else None,
            "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
            "energy_above_hull": float(doc.energy_above_hull) if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None else None
        }
        
        # Add search mode info
        data["search_mode"] = "composition_structure" if has_structure else "composition_only"
        data["search_criteria"] = {
            "element": element,
            "formula": formula,
            "chemsys": chemsys,
            "spacegroup_number": spacegroup_number if has_structure else None,
            "crystal_system": crystal_system if has_structure else None,
            "theoretical": False
        }
        if len(docs) > 1:
            data["num_matches_found"] = len(docs)
        
        # Add precalculated bulk moduli from reference data
        if precalc_bulk_moduli:
            data["precalculated_bulk_moduli"] = precalc_bulk_moduli
        
        if bulk_modulus:
            # Handle both dict and object formats
            if isinstance(bulk_modulus, dict):
                # API returns dict with k_vrh, k_voigt, k_reuss keys
                data["bulk_modulus"] = {
                    "k_vrh": float(bulk_modulus.get('k_vrh') or bulk_modulus.get('vrh')) if (bulk_modulus.get('k_vrh') is not None or bulk_modulus.get('vrh') is not None) else None,
                    "k_voigt": float(bulk_modulus.get('k_voigt') or bulk_modulus.get('voigt')) if (bulk_modulus.get('k_voigt') is not None or bulk_modulus.get('voigt') is not None) else None,
                    "k_reuss": float(bulk_modulus.get('k_reuss') or bulk_modulus.get('reuss')) if (bulk_modulus.get('k_reuss') is not None or bulk_modulus.get('reuss') is not None) else None,
                    "unit": "GPa"
                }
            else:
                # Object format with vrh, voigt, reuss attributes
                data["bulk_modulus"] = {
                    "k_vrh": float(bulk_modulus.vrh) if hasattr(bulk_modulus, 'vrh') and bulk_modulus.vrh is not None else None,
                    "k_voigt": float(bulk_modulus.voigt) if hasattr(bulk_modulus, 'voigt') and bulk_modulus.voigt is not None else None,
                    "k_reuss": float(bulk_modulus.reuss) if hasattr(bulk_modulus, 'reuss') and bulk_modulus.reuss is not None else None,
                    "unit": "GPa"
                }
        else:
            data["bulk_modulus"] = None
            
        if shear_modulus:
            # Handle both dict and object formats
            if isinstance(shear_modulus, dict):
                # API returns dict with g_vrh, g_voigt, g_reuss keys
                data["shear_modulus"] = {
                    "g_vrh": float(shear_modulus.get('g_vrh') or shear_modulus.get('vrh')) if (shear_modulus.get('g_vrh') is not None or shear_modulus.get('vrh') is not None) else None,
                    "g_voigt": float(shear_modulus.get('g_voigt') or shear_modulus.get('voigt')) if (shear_modulus.get('g_voigt') is not None or shear_modulus.get('voigt') is not None) else None,
                    "g_reuss": float(shear_modulus.get('g_reuss') or shear_modulus.get('reuss')) if (shear_modulus.get('g_reuss') is not None or shear_modulus.get('reuss') is not None) else None,
                    "unit": "GPa"
                }
            else:
                # Object format with vrh, voigt, reuss attributes
                data["shear_modulus"] = {
                    "g_vrh": float(shear_modulus.vrh) if hasattr(shear_modulus, 'vrh') and shear_modulus.vrh is not None else None,
                    "g_voigt": float(shear_modulus.voigt) if hasattr(shear_modulus, 'voigt') and shear_modulus.voigt is not None else None,
                    "g_reuss": float(shear_modulus.reuss) if hasattr(shear_modulus, 'reuss') and shear_modulus.reuss is not None else None,
                    "unit": "GPa"
                }
        else:
            data["shear_modulus"] = None
        
        # Sanity checks: extract K and G for validation
        K = data.get("bulk_modulus", {}).get("k_vrh") if isinstance(data.get("bulk_modulus"), dict) else None
        G = data.get("shear_modulus", {}).get("g_vrh") if isinstance(data.get("shear_modulus"), dict) else None
        
        # Compute Poisson ratio from K and G
        nu = _safe_ratio(K, G, eps)
        
        # Compute Young's modulus and Pugh ratio (using eps for safety like Poisson ratio)
        E = None
        pugh = None
        if K is not None and G is not None:
            den = 3.0*K + G
            if abs(den) > eps and abs(G) > eps:
                E = 9.0*K*G/den
                pugh = K/G
        
        # Validate and flag issues
        flags = []
        if K is not None and K <= 0:
            flags.append("non_positive_bulk_modulus")
        if G is not None and G <= 0:
            flags.append("non_positive_shear_modulus")
        if nu is not None and not (-1.0 < nu < 0.5):
            flags.append("poisson_out_of_bounds")
        if E is not None and E <= 0:
            flags.append("non_positive_youngs_modulus")
        if pugh is not None and pugh < 0:
            flags.append("negative_pugh_ratio")
        
        # Suppress nonsense derived values when G <= 0
        if G is not None and G <= 0:
            data["derived"] = {
                "poisson_from_KG": None,
                "youngs_from_KG": None,
                "pugh_K_over_G": None
            }
            flags.append("derived_suppressed_due_to_non_positive_shear_modulus")
        else:
            # Add derived properties
            data["derived"] = {
                "poisson_from_KG": nu,
                "youngs_from_KG": E,
                "pugh_K_over_G": pugh
            }
        
        # 2) ELASTICITY: fetch tensor & Born stability from the dedicated endpoint
        et_doc = None
        used_pymatgen = False
        
        try:
            # Try with explicit fields (may vary by server version)
            et_docs = mpr.materials.elasticity.search(
                material_ids=actual_material_id,
                fields=["material_id", "elastic_tensor", "K_VRH", "G_VRH", "warnings"]
            )
        except Exception as e:
            # Fallback: no field filtering (avoids "invalid fields" errors)
            _log.debug(f"Elasticity search with fields failed for {actual_material_id}, trying without fields: {e}")
            try:
                et_docs = mpr.materials.elasticity.search(material_ids=actual_material_id)
            except Exception as e2:
                _log.warning(f"Elasticity search failed for {actual_material_id}: {e2}")
                et_docs = []
        
        if et_docs:
            et_doc = et_docs[0]
            
            # Pull VRH from elasticity doc if present (useful cross-check)
            K_vrh_el = getattr(et_doc, "K_VRH", None)
            G_vrh_el = getattr(et_doc, "G_VRH", None)
            if K_vrh_el is not None or G_vrh_el is not None:
                data.setdefault("vrh_from_tensor", {})
                if K_vrh_el is not None:
                    data["vrh_from_tensor"]["k_vrh"] = float(K_vrh_el)
                if G_vrh_el is not None:
                    data["vrh_from_tensor"]["g_vrh"] = float(G_vrh_el)
            
            # Recompute VRH + Born stability from the full elastic tensor if available
            elastic_tensor = getattr(et_doc, "elastic_tensor", None)
            if elastic_tensor is not None:
                try:
                    from pymatgen.analysis.elasticity.elastic import ElasticTensor
                    ET = ElasticTensor(elastic_tensor).voigt_symmetrized
                    
                    # Convert from eV/Å³ to GPa (pymatgen returns moduli in eV/Å³)
                    # Conversion factor: 1 eV/Å³ = 160.21766208 GPa
                    CONVERSION_FACTOR = 160.21766208
                    
                    # Check Born stability (method availability may vary by pymatgen version)
                    is_born_stable = None
                    if hasattr(ET, 'is_stable') and callable(getattr(ET, 'is_stable', None)):
                        try:
                            is_born_stable = bool(ET.is_stable())
                        except Exception:
                            is_born_stable = None
                    
                    data.setdefault("vrh_from_tensor", {})
                    k_vrh_computed = float(ET.k_vrh) * CONVERSION_FACTOR  # Convert eV/Å³ to GPa
                    g_vrh_computed = float(ET.g_vrh) * CONVERSION_FACTOR  # Convert eV/Å³ to GPa
                    
                    # Cross-check with summary values if available (flag if >10% difference)
                    cross_check_note = None
                    if K is not None and abs(k_vrh_computed - K) / max(abs(K), 1e-6) > 0.10:
                        cross_check_note = f"Summary K_VRH ({K:.2f} GPa) differs from tensor recomputed ({k_vrh_computed:.2f} GPa) by >10%"
                    if G is not None and abs(g_vrh_computed - G) / max(abs(G), 1e-6) > 0.10:
                        if cross_check_note:
                            cross_check_note += f"; Summary G_VRH ({G:.2f} GPa) differs from tensor recomputed ({g_vrh_computed:.2f} GPa) by >10%"
                        else:
                            cross_check_note = f"Summary G_VRH ({G:.2f} GPa) differs from tensor recomputed ({g_vrh_computed:.2f} GPa) by >10%"
                    
                    data["vrh_from_tensor"].update({
                        "k_vrh": k_vrh_computed,
                        "g_vrh": g_vrh_computed,
                        "is_born_stable": is_born_stable,
                        "unit": "GPa"
                    })
                    if cross_check_note:
                        data.setdefault("notes", []).append(cross_check_note)
                    
                    used_pymatgen = True
                    
                    # Add Born stability details
                    data["born_details"] = {
                        "is_born_stable": is_born_stable
                    }
                except Exception as tensor_err:
                    _log.warning(f"Could not recompute VRH from elastic tensor for {actual_material_id}: {tensor_err}")
            
            # Surface warnings if the API provides them
            et_warnings = getattr(et_doc, "warnings", None)
            if et_warnings:
                if isinstance(et_warnings, (list, tuple)):
                    data["elasticity_warnings"] = list(et_warnings)
                else:
                    data["elasticity_warnings"] = [et_warnings]
        
        # Heuristic stability check: K > 0, G > 0, -1 < ν < 0.5, E > 0, and K/G ≥ 0
        likely_stable = (
            K is not None and K > 0 and
            G is not None and G > 0 and
            nu is not None and -1.0 < nu < 0.5 and
            E is not None and E > 0 and
            pugh is not None and pugh >= 0
        )
        
        # Add mechanical stability assessment
        data["mechanical_stability"] = {
            "likely_stable": likely_stable,
            "flags": flags or None,
        }

        # Add compact quality summary
        data["data_quality"] = (
            "elastic_tensor_unstable" if "non_positive_shear_modulus" in flags
            else "ok"
        )
        
        if hasattr(doc, 'universal_anisotropy') and doc.universal_anisotropy is not None:
            data["universal_anisotropy"] = float(doc.universal_anisotropy)
        
        if hasattr(doc, 'homogeneous_poisson') and doc.homogeneous_poisson is not None:
            data["poisson_ratio"] = float(doc.homogeneous_poisson)
        
        citations = ["Materials Project"]
        if used_pymatgen:
            citations.append("pymatgen")
        
        # Lower confidence if flags are present
        conf = Confidence.HIGH if not flags else Confidence.MEDIUM
        
        # Prepare notes (including warnings)
        notes = []
        if warnings:
            notes.extend(warnings)
        
        return success_result(
            handler="materials",
            function="get_elastic_properties",
            data=data,
            citations=citations,
            confidence=conf,
            notes=notes if notes else None
        )
        
    except Exception as e:
        error_context =f"composition/structure search"
        _log.error(f"Error getting elastic properties for {error_context}: {e}", exc_info=True)
        formatted_error = format_field_error(e)
        return error_result(
            handler="materials",
            function="get_elastic_properties",
            error=formatted_error,
            error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
            citations=["Materials Project"]
        )


def find_closest_alloy_compositions(
    mpr,
    elements: List[str],
    target_composition: Optional[Dict[str, float]] = None,
    tolerance: float = 0.05,
    is_stable: bool = True,
    ehull_max: float = 0.20,
    require_binaries: bool = True
) -> Dict[str, Any]:
    """
    Find materials with closest matching alloy compositions.
    
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
        # Input validation
        if tolerance < 0:
            return error_result(
                handler="materials",
                function="find_closest_alloy_compositions",
                error=f"Invalid tolerance {tolerance}. Must be non-negative.",
                error_type=ErrorType.INVALID_INPUT,
                citations=["Materials Project"]
            )
        
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
        
        # If no stable materials found, try with broader stability criteria
        used_fallback_stability = False
        if not docs and is_stable:
            _log.info(f"No stable materials found for {chemsys}, trying with metastable entries (Ehull ≤ {ehull_max} eV/atom)")
            search_kwargs["energy_above_hull"] = (0, ehull_max)
            docs = mpr.materials.summary.search(**search_kwargs)
            used_fallback_stability = True
        
        if not docs:
            return error_result(
                handler="materials",
                function="find_closest_alloy_compositions",
                error=f"No materials found for {chemsys}",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
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
        
        return success_result(
            handler="materials",
            function="find_closest_alloy_compositions",
            data={
                "chemical_system": chemsys,
                "target_composition": target_composition,
                "tolerance": tolerance,
                "require_binaries": require_binaries,
                "ehull_window_eV": (0, ehull_max) if used_fallback_stability else ((0, 1e-3) if is_stable else (0, ehull_max)),
                "num_materials_found": len(materials),
                "closest_match_used": closest_match_used,
                "max_composition_deviation": max_deviation,
                "l1_distance_to_target": best_l1_distance,
                "used_metastable_fallback": used_fallback_stability,
                "materials": materials
            },
            citations=["Materials Project"],
            confidence=Confidence.HIGH if materials else Confidence.LOW
        )
        
    except Exception as e:
        _log.error(f"Error finding alloy compositions: {e}", exc_info=True)
        formatted_error = format_field_error(e)
        return error_result(
            handler="materials",
            function="find_closest_alloy_compositions",
            error=formatted_error,
            error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
            citations=["Materials Project"]
        )


def compare_material_properties_by_id(
    mpr,
    material_id1: str,
    material_id2: str,
    property_name: str = "bulk_modulus"
) -> Dict[str, Any]:
    """
    Compare a specific property between two materials by their IDs.
    
    This is a convenience wrapper that fetches materials, extracts their formulas,
    and calls get_elastic_properties() + compare_material_properties().
    
    Args:
        mpr: MPRester client instance
        material_id1: First material ID
        material_id2: Second material ID
        property_name: Name of property to compare
        
    Returns:
        Dictionary with comparison results
    """
    # Fetch materials to get their formulas
    try:
        mat1_docs = mpr.materials.summary.search(
            material_ids=[material_id1],
            fields=["material_id", "formula_pretty"]
        )
        mat2_docs = mpr.materials.summary.search(
            material_ids=[material_id2],
            fields=["material_id", "formula_pretty"]
        )
        
        if not mat1_docs:
            return error_result(
                handler="materials",
                function="compare_materials_by_id",
                error=f"Material {material_id1} not found",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        if not mat2_docs:
            return error_result(
                handler="materials",
                function="compare_materials_by_id",
                error=f"Material {material_id2} not found",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
        formula1 = mat1_docs[0].formula_pretty
        formula2 = mat2_docs[0].formula_pretty
        
        props1 = get_elastic_properties(mpr, formula=formula1)
        props2 = get_elastic_properties(mpr, formula=formula2)
        return compare_material_properties(props1, props2, property_name)
        
    except Exception as e:
        return error_result(
            handler="materials",
            function="compare_materials_by_id",
            error=str(e),
            error_type=ErrorType.COMPUTATION_ERROR,
            citations=["Materials Project"]
        )


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
            return error_result(
                handler="materials",
                function="compare_material_properties",
                error="One or both materials missing property data",
                error_type=ErrorType.INVALID_INPUT,
                citations=["Materials Project"]
            )
        
        # Extract data from standardized results
        data1 = property1.get("data", property1)
        data2 = property2.get("data", property2)
        
        # Extract the property values
        val1 = None
        val2 = None
        
        if property_name == "bulk_modulus":
            if data1.get("bulk_modulus") and data2.get("bulk_modulus"):
                val1 = data1["bulk_modulus"].get("k_vrh")
                val2 = data2["bulk_modulus"].get("k_vrh")
        elif property_name == "shear_modulus":
            if data1.get("shear_modulus") and data2.get("shear_modulus"):
                val1 = data1["shear_modulus"].get("g_vrh")
                val2 = data2["shear_modulus"].get("g_vrh")
        elif property_name in data1 and property_name in data2:
            val1 = data1.get(property_name)
            val2 = data2.get(property_name)
        
        if val1 is None or val2 is None:
            return error_result(
                handler="materials",
                function="compare_material_properties",
                error=f"Property '{property_name}' not available for one or both materials",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
        # Calculate differences
        absolute_diff = val2 - val1
        percent_change = (absolute_diff / val1) * 100.0 if val1 != 0 else None
        
        # Determine unit
        unit = None
        if property_name in ("bulk_modulus", "shear_modulus"):
            unit = "GPa"
        
        data = {
            "property_name": property_name,
            "material1": {
                "id": data1.get("material_id"),
                "formula": data1.get("formula"),
                "value": float(val1),
                "unit": unit
            },
            "material2": {
                "id": data2.get("material_id"),
                "formula": data2.get("formula"),
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
                data["interpretation"] = "Negligible change"
            elif percent_change > 0:
                data["interpretation"] = f"Material 2 has {abs(percent_change):.1f}% higher {property_name}"
            else:
                data["interpretation"] = f"Material 2 has {abs(percent_change):.1f}% lower {property_name}"
        
        return success_result(
            handler="materials",
            function="compare_material_properties",
            data=data,
            citations=["Materials Project"],
            confidence=Confidence.HIGH
        )
        
    except Exception as e:
        _log.error(f"Error comparing properties: {e}", exc_info=True)
        return error_result(
            handler="materials",
            function="compare_material_properties",
            error=str(e),
            error_type=ErrorType.COMPUTATION_ERROR,
            citations=["Materials Project"]
        )


def _get_reference_bulk_modulus(element_symbol: str) -> Optional[float]:
    """
    Get reference bulk modulus for an element from BULK_MODULI array.
    
    Args:
        element_symbol: Element symbol (e.g., 'Ag', 'Cu')
        
    Returns:
        Bulk modulus in GPa (average if range given), or None if not found
    """
    element_upper = element_symbol.strip().upper()
    
    for entry in BULK_MODULI:
        if entry.get("symbol") == element_upper:
            k_value = entry.get("bulk_modulus_GPa")
            if k_value is None:
                continue
            
            # Handle both single values and ranges
            if isinstance(k_value, list):
                # Return average of range
                return sum(k_value) / len(k_value)
            else:
                return float(k_value)
    
    return None


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
        # Input validation
        if not (0 < dopant_concentration < 1):
            return error_result(
                handler="materials",
                function="analyze_doping_effect",
                error=f"Invalid dopant concentration {dopant_concentration}. Must be between 0 and 1 (exclusive).",
                error_type=ErrorType.INVALID_INPUT,
                citations=["Materials Project"]
            )
        
        # Find pure host material
        host_docs = mpr.materials.summary.search(
            elements=[host_element],
            num_elements=1,
            is_stable=True,
            fields=[
                "material_id", "formula_pretty", "composition",
                "bulk_modulus", "shear_modulus", "energy_above_hull", "is_stable"
            ]
        )
        
        if not host_docs:
            return error_result(
                handler="materials",
                function="analyze_doping_effect",
                error=f"Could not find pure {host_element} in database",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
        # Get the first stable entry
        host_doc = host_docs[0]
        host_id = host_doc.material_id if hasattr(host_doc, 'material_id') else None
        
        if not host_id:
            return error_result(
                handler="materials",
                function="analyze_doping_effect",
                error=f"Could not get material ID for pure {host_element}",
                error_type=ErrorType.NOT_FOUND,
                citations=["Materials Project"]
            )
        
        # Get pure host material properties using the host element
        host_props = get_elastic_properties(mpr, element=host_element)
        
        # Find doped materials
        target_comp = {
            host_element: 1.0 - dopant_concentration,
            dopant_element: dopant_concentration
        }
        
        # Try stable entries first
        alloys = find_closest_alloy_compositions(
            mpr,
            [host_element, dopant_element],
            target_composition=target_comp,
            tolerance=0.05,  # Strict tolerance: ±5 at%
            is_stable=True,
            require_binaries=True
        )
        
        # Fallback: allow metastable entries if no stable ones found
        used_metastable = False
        if not alloys.get("success") or not alloys.get("data") or not alloys["data"].get("materials"):
            _log.info(f"No stable {host_element}-{dopant_element} alloys found, trying metastable entries (Ehull ≤ 0.20 eV/atom)")
            alloys = find_closest_alloy_compositions(
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
                # Get elastic properties for pure dopant element
                _cached_dopant_props[0] = get_elastic_properties(mpr, element=dopant_element)
            return _cached_dopant_props[0]
        
        # If no alloys found, compute VRH estimate from pure elements
        if not alloys.get("success") or not alloys.get("data") or not alloys["data"].get("materials"):
            _log.info(f"No {host_element}-{dopant_element} alloys found, computing VRH mixture estimate")
            
            dopant_props = get_pure_dopant_props()
            
            if not dopant_props or not dopant_props.get("success"):
                return error_result(
                    handler="materials",
                    function="analyze_doping_effect",
                    error=(
                        f"No {host_element}-{dopant_element} entries found near {dopant_concentration*100:.1f}% "
                        f"{dopant_element} within Ehull ≤ 0.20 eV/atom, and could not find pure {dopant_element} "
                        f"for mixture model fallback."
                    ),
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"]
                )
            
            # Extract data from standardized results
            host_data = host_props.get("data", host_props)
            dopant_data = dopant_props.get("data", dopant_props)
            
            # Check if both have bulk modulus data
            if (host_data.get("bulk_modulus") and host_data["bulk_modulus"].get("k_vrh") and
                dopant_data.get("bulk_modulus") and dopant_data["bulk_modulus"].get("k_vrh")):
                
                K_host = host_data["bulk_modulus"]["k_vrh"]
                K_dopant = dopant_data["bulk_modulus"]["k_vrh"]
                x = dopant_concentration
                
                # Voigt-Reuss-Hill bounds
                K_V = (1 - x) * K_host + x * K_dopant  # Voigt (upper bound)
                K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)  # Reuss (lower bound)
                K_est = 0.5 * (K_V + K_R)  # VRH average
                pct = 100 * (K_est - K_host) / K_host
                
                return success_result(
                    handler="materials",
                    function="analyze_doping_effect",
                    data={
                        "host_element": host_element,
                        "dopant_element": dopant_element,
                        "target_dopant_concentration": dopant_concentration,
                        "property_analyzed": property_name,
                        "pure_host": {
                            "material_id": host_data.get("material_id"),
                            "formula": host_data.get("formula"),
                            "bulk_modulus_vrh": float(K_host),
                            "unit": "GPa"
                        },
                        "pure_dopant": {
                            "material_id": dopant_data.get("material_id"),
                            "formula": dopant_data.get("formula"),
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
                        "comparisons": []
                    },
                    citations=["Materials Project", "pymatgen"],
                    confidence=Confidence.MEDIUM,
                    notes=[
                        f"No {host_element}-{dopant_element} alloy entries found within Ehull ≤ 0.20 eV/atom.",
                        f"Used Voigt-Reuss-Hill mixture model with pure {host_element} and {dopant_element}.",
                        f"This provides an estimate assuming ideal mixing; actual alloys may deviate."
                    ]
                )
            else:
                return error_result(
                    handler="materials",
                    function="analyze_doping_effect",
                    error=(
                        f"No {host_element}-{dopant_element} entries found near {dopant_concentration*100:.1f}% "
                        f"{dopant_element} within Ehull ≤ 0.20 eV/atom, and elastic data incomplete "
                        f"for mixture model fallback."
                    ),
                    error_type=ErrorType.NOT_FOUND,
                    citations=["Materials Project"]
                )
        
        # Get properties for each alloy and compare
        # Extract data from standardized result
        alloys_data = alloys.get("data", alloys)
        materials_list = alloys_data.get("materials", [])
        
        comparisons = []
        for alloy in materials_list:
            alloy_id = alloy["material_id"]
            alloy_formula = alloy.get("formula", None)
            
            if not alloy_formula:
                _log.warning(f"Alloy {alloy_id} missing formula, skipping")
                continue
            
            # Get elastic properties using formula
            _log.info(f"Getting elastic properties for alloy {alloy_formula} ({alloy_id})")
            alloy_props = get_elastic_properties(mpr, formula=alloy_formula)
            
            if not alloy_props or not alloy_props.get("success"):
                _log.warning(f"No elastic properties found for alloy {alloy_formula} ({alloy_id}): {alloy_props.get('error', 'unknown error') if alloy_props else 'no result'}")
                continue
            
            comparison_result = compare_material_properties(
                host_props,
                alloy_props,
                property_name
            )
            
            if comparison_result.get("success"):
                    # Extract the data from the standardized result
                    comparison_data = comparison_result.get("data", comparison_result)
                    
                    # Create a new comparison entry with additional metadata
                    comparison_entry = {
                        **comparison_data,
                        "alloy_composition": alloy["atomic_fractions"],
                        "alloy_energy_above_hull": alloy.get("energy_above_hull"),
                        "is_closest_match": alloy.get("closest_match", False),
                        "composition_deviation": alloy.get("max_composition_deviation"),
                        "requested_composition": target_comp,
                        "actual_composition": alloy["atomic_fractions"]
                    }
                    
                    comparisons.append(comparison_entry)
        
        # Always compute VRH estimate at exact composition for comparison
        vrh_estimate = None
        host_data = host_props.get("data", host_props)
        if (host_data.get("bulk_modulus") and host_data["bulk_modulus"].get("k_vrh")):
            # Get pure dopant properties (cached)
            dopant_props = get_pure_dopant_props()
            
            dopant_data = dopant_props.get("data", dopant_props) if dopant_props else {}
            if (dopant_props and dopant_props.get("success") and 
                dopant_data.get("bulk_modulus") and dopant_data["bulk_modulus"].get("k_vrh")):
                
                K_host = host_data["bulk_modulus"]["k_vrh"]
                K_dopant = dopant_data["bulk_modulus"]["k_vrh"]
                x = dopant_concentration
                
                # Voigt-Reuss-Hill bounds (rigorous for isotropic aggregates)
                K_V = (1 - x) * K_host + x * K_dopant  # Voigt (upper bound)
                K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)  # Reuss (lower bound)
                K_VRH = 0.5 * (K_V + K_R)  # VRH average
                
                pct_voigt = 100 * (K_V - K_host) / K_host
                pct_reuss = 100 * (K_R - K_host) / K_host
                pct_vrh = 100 * (K_VRH - K_host) / K_host
                
                vrh_estimate = {
                    "method": "Voigt-Reuss-Hill bounds from pure elements (MP DFT)",
                    "source": "materials_project_dft",
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
                    "note": f"VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element} using MP DFT values; actual value lies between Reuss (lower) and Voigt (upper)"
                }
        
        # Compute reference-based VRH estimate from literature values
        vrh_reference_estimate = None
        K_host_ref = _get_reference_bulk_modulus(host_element)
        K_dopant_ref = _get_reference_bulk_modulus(dopant_element)
        
        if K_host_ref is not None and K_dopant_ref is not None:
            x = dopant_concentration
            
            # Voigt-Reuss-Hill bounds
            K_V_ref = (1 - x) * K_host_ref + x * K_dopant_ref
            K_R_ref = 1.0 / ((1 - x) / K_host_ref + x / K_dopant_ref)
            K_VRH_ref = 0.5 * (K_V_ref + K_R_ref)
            
            pct_voigt_ref = 100 * (K_V_ref - K_host_ref) / K_host_ref
            pct_reuss_ref = 100 * (K_R_ref - K_host_ref) / K_host_ref
            pct_vrh_ref = 100 * (K_VRH_ref - K_host_ref) / K_host_ref
            
            vrh_reference_estimate = {
                "method": "Voigt-Reuss-Hill bounds from literature reference values",
                "source": "experimental_literature",
                "pure_host_k_ref": float(K_host_ref),
                "pure_dopant_k_ref": float(K_dopant_ref),
                "dopant_concentration": dopant_concentration,
                "k_voigt_gpa": float(K_V_ref),
                "k_reuss_gpa": float(K_R_ref),
                "k_vrh_gpa": float(K_VRH_ref),
                "percent_change_voigt": float(pct_voigt_ref),
                "percent_change_reuss": float(pct_reuss_ref),
                "percent_change_vrh": float(pct_vrh_ref),
                "unit": "GPa",
                "note": f"VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element} using experimental literature values"
            }
            _log.info(f"Computed reference VRH estimate: {host_element} ({K_host_ref:.1f} GPa) + {dopant_concentration*100:.1f}% {dopant_element} ({K_dopant_ref:.1f} GPa) → {K_VRH_ref:.1f} GPa ({pct_vrh_ref:+.1f}%)")
        else:
            if K_host_ref is None:
                _log.info(f"No reference bulk modulus found for {host_element}")
            if K_dopant_ref is None:
                _log.info(f"No reference bulk modulus found for {dopant_element}")
        
        if not comparisons:
            if vrh_estimate or vrh_reference_estimate:
                # Return VRH estimates as the result
                notes = [f"No database entries found at {dopant_concentration*100:.1f}% {dopant_element}."]
                
                if vrh_estimate:
                    notes.append(f"MP DFT-based VRH predicts {vrh_estimate['percent_change_vrh']:.1f}% change "
                                f"(range: {vrh_estimate['percent_change_reuss']:.1f}% to {vrh_estimate['percent_change_voigt']:.1f}%).")
                
                if vrh_reference_estimate:
                    notes.append(f"Literature-based VRH predicts {vrh_reference_estimate['percent_change_vrh']:.1f}% change "
                                f"(range: {vrh_reference_estimate['percent_change_reuss']:.1f}% to {vrh_reference_estimate['percent_change_voigt']:.1f}%).")
                
                return success_result(
                    handler="materials",
                    function="analyze_doping_effect",
                    data={
                        "host_element": host_element,
                        "dopant_element": dopant_element,
                        "target_dopant_concentration": dopant_concentration,
                        "requested_composition": target_comp,
                        "property_analyzed": property_name,
                        "pure_host": {
                            "material_id": host_data.get("material_id") if vrh_estimate else None,
                            "formula": host_data.get("formula") if vrh_estimate else host_element,
                            "bulk_modulus_vrh_mp": vrh_estimate["pure_host_k_vrh"] if vrh_estimate else None,
                            "bulk_modulus_ref": K_host_ref if vrh_reference_estimate else None,
                            "unit": "GPa"
                        },
                        "num_alloys_analyzed": 0,
                        "used_metastable_entries": False,
                        "used_mixture_model": True,
                        "vrh_estimate_mp_dft": vrh_estimate,
                        "vrh_estimate_literature": vrh_reference_estimate,
                        "comparisons": []
                    },
                    citations=["Materials Project", "pymatgen"] if vrh_estimate else ["Experimental literature references"],
                    confidence=Confidence.MEDIUM,
                    notes=notes
                )
            else:
                return error_result(
                    handler="materials",
                    function="analyze_doping_effect",
                    error="Could not compare properties for any alloys and VRH estimate unavailable",
                    error_type=ErrorType.COMPUTATION_ERROR,
                    citations=["Materials Project"]
                )
        
        result_data = {
            "host_element": host_element,
            "dopant_element": dopant_element,
            "target_dopant_concentration": dopant_concentration,
            "requested_composition": target_comp,
            "property_analyzed": property_name,
            "pure_host": {
                "material_id": host_data.get("material_id"),
                "formula": host_data.get("formula"),
                "bulk_modulus_vrh_mp": host_data.get("bulk_modulus", {}).get("k_vrh") if vrh_estimate else None,
                "bulk_modulus_ref": K_host_ref if vrh_reference_estimate else None
            },
            "num_alloys_analyzed": len(comparisons),
            "used_metastable_entries": used_metastable,
            "used_closest_match": alloys_data.get("closest_match_used", False),
            "vrh_estimate_mp_dft": vrh_estimate,
            "vrh_estimate_literature": vrh_reference_estimate,
            "comparisons": comparisons
        }
        notes = []
        
        if used_metastable:
            notes.append(
                f"No stable {host_element}-{dopant_element} compounds found on convex hull. "
                f"Used metastable entries (Ehull ≤ 0.20 eV/atom) which may represent "
                f"solid solutions or ordered structures."
            )
        
        if alloys_data.get("closest_match_used") and comparisons:
            closest_comp = comparisons[0]["actual_composition"]
            req_comp_str = ", ".join(f"{el}{target_comp[el]:.3f}" for el in sorted(target_comp.keys()))
            actual_comp_str = ", ".join(f"{el}{closest_comp[el]:.3f}" for el in sorted(closest_comp.keys()))
            l1_dist = alloys_data.get("l1_distance_to_target", alloys_data.get("max_composition_deviation", 0))
            notes.insert(0, 
                f"Requested: {req_comp_str}; using closest DB entry: {actual_comp_str} (Δ={l1_dist:.3f})."
            )
        
        if vrh_estimate:
            notes.append(
                f"MP DFT-based VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element}: "
                f"{vrh_estimate['percent_change_vrh']:.1f}% (VRH average), "
                f"range {vrh_estimate['percent_change_reuss']:.1f}% (Reuss lower) to "
                f"{vrh_estimate['percent_change_voigt']:.1f}% (Voigt upper)."
            )
        
        if vrh_reference_estimate:
            notes.append(
                f"Literature-based VRH bounds at exact {dopant_concentration*100:.1f}% {dopant_element}: "
                f"{vrh_reference_estimate['percent_change_vrh']:.1f}% (VRH average), "
                f"range {vrh_reference_estimate['percent_change_reuss']:.1f}% (Reuss lower) to "
                f"{vrh_reference_estimate['percent_change_voigt']:.1f}% (Voigt upper)."
            )
        
        # Add summary statistics
        percent_changes = [c["comparison"]["percent_change"] for c in comparisons if c["comparison"].get("percent_change") is not None]
        if percent_changes:
            result_data["summary"] = {
                "avg_percent_change": float(np.mean(percent_changes)),
                "min_percent_change": float(np.min(percent_changes)),
                "max_percent_change": float(np.max(percent_changes)),
                "std_percent_change": float(np.std(percent_changes)) if len(percent_changes) > 1 else 0.0
            }
        
        return success_result(
            handler="materials",
            function="analyze_doping_effect",
            data=result_data,
            citations=["Materials Project", "pymatgen"],
            confidence=Confidence.HIGH if comparisons else Confidence.MEDIUM,
            notes=notes if notes else None
        )
        
    except Exception as e:
        _log.error(f"Error analyzing doping effect: {e}", exc_info=True)
        formatted_error = format_field_error(e)
        return error_result(
            handler="materials",
            function="analyze_doping_effect",
            error=formatted_error,
            error_type=ErrorType.INVALID_INPUT if "invalid fields" in str(e).lower() or "invalid field" in str(e).lower() else ErrorType.COMPUTATION_ERROR,
            citations=["Materials Project"]
        )

