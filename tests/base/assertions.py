"""
Custom assertions and validators for testing materials handler responses.

This module provides domain-specific assertions that validate response
structures, physical constraints, and data quality.
"""

from typing import Any, Dict, List, Optional


def assert_success_response(response: Dict[str, Any], expected_handler: str = "materials"):
    """
    Assert that response is a successful result with proper structure.
    
    Args:
        response: Response dictionary to validate
        expected_handler: Expected handler name
        
    Raises:
        AssertionError: If response structure is invalid
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert response.get("success") is True, f"Expected success=True, got {response.get('success')}"
    
    # Check handler in metadata (new structure) or at top level (old structure)
    handler = None
    if "metadata" in response and isinstance(response["metadata"], dict):
        handler = response["metadata"].get("handler")
    else:
        handler = response.get("handler")
    
    assert handler == expected_handler, \
        f"Expected handler='{expected_handler}', got '{handler}'"
    assert "data" in response, "Response must contain 'data' field"
    assert "citations" in response, "Response must contain 'citations' field"
    assert isinstance(response["citations"], list), "Citations must be a list"


def assert_error_response(response: Dict[str, Any], expected_handler: str = "materials"):
    """
    Assert that response is an error result with proper structure.
    
    Args:
        response: Response dictionary to validate
        expected_handler: Expected handler name
        
    Raises:
        AssertionError: If response structure is invalid
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert response.get("success") is False, f"Expected success=False, got {response.get('success')}"
    
    # Check handler in metadata (new structure) or at top level (old structure)
    handler = None
    if "metadata" in response and isinstance(response["metadata"], dict):
        handler = response["metadata"].get("handler")
    else:
        handler = response.get("handler")
    
    assert handler == expected_handler, \
        f"Expected handler='{expected_handler}', got '{handler}'"
    assert "error" in response, "Error response must contain 'error' field"
    assert "error_type" in response, "Error response must contain 'error_type' field"


def assert_elastic_properties_structure(data: Dict[str, Any]):
    """
    Assert that elastic properties data has the expected structure.
    
    Args:
        data: Elastic properties data dictionary
        
    Raises:
        AssertionError: If structure is invalid
    """
    assert "material_id" in data, "Must contain material_id"
    assert "formula" in data, "Must contain formula"
    
    # Check for modulus structures if present
    if data.get("bulk_modulus") is not None:
        bm = data["bulk_modulus"]
        assert isinstance(bm, dict), "bulk_modulus must be a dictionary"
        assert "k_vrh" in bm, "bulk_modulus must contain k_vrh"
        assert "unit" in bm, "bulk_modulus must contain unit"
        assert bm["unit"] == "GPa", "bulk_modulus unit must be GPa"
    
    if data.get("shear_modulus") is not None:
        sm = data["shear_modulus"]
        assert isinstance(sm, dict), "shear_modulus must be a dictionary"
        assert "g_vrh" in sm, "shear_modulus must contain g_vrh"
        assert "unit" in sm, "shear_modulus must contain unit"
        assert sm["unit"] == "GPa", "shear_modulus unit must be GPa"
    
    # Check derived properties structure
    if "derived" in data:
        derived = data["derived"]
        assert isinstance(derived, dict), "derived must be a dictionary"
        # These can be None if suppressed
        assert "poisson_from_KG" in derived
        assert "youngs_from_KG" in derived
        assert "pugh_K_over_G" in derived
    
    # Check mechanical stability assessment
    if "mechanical_stability" in data:
        mech = data["mechanical_stability"]
        assert isinstance(mech, dict), "mechanical_stability must be a dictionary"
        assert "likely_stable" in mech, "mechanical_stability must contain likely_stable"
        assert isinstance(mech["likely_stable"], bool), "likely_stable must be boolean"


def assert_physical_constraints_elastic(data: Dict[str, Any], allow_negative: bool = False):
    """
    Assert that elastic properties satisfy physical constraints.
    
    Args:
        data: Elastic properties data dictionary
        allow_negative: If True, allow negative values (for testing bad data)
        
    Raises:
        AssertionError: If physical constraints are violated
    """
    if not allow_negative:
        # Bulk modulus should be positive
        if data.get("bulk_modulus") and data["bulk_modulus"].get("k_vrh") is not None:
            K = data["bulk_modulus"]["k_vrh"]
            assert K > 0, f"Bulk modulus must be positive, got {K}"
        
        # Shear modulus should be positive
        if data.get("shear_modulus") and data["shear_modulus"].get("g_vrh") is not None:
            G = data["shear_modulus"]["g_vrh"]
            assert G > 0, f"Shear modulus must be positive, got {G}"
        
        # Poisson's ratio should be in valid range
        if data.get("derived") and data["derived"].get("poisson_from_KG") is not None:
            nu = data["derived"]["poisson_from_KG"]
            assert -1.0 < nu < 0.5, f"Poisson's ratio must be in (-1, 0.5), got {nu}"
        
        # Young's modulus should be positive
        if data.get("derived") and data["derived"].get("youngs_from_KG") is not None:
            E = data["derived"]["youngs_from_KG"]
            assert E > 0, f"Young's modulus must be positive, got {E}"


def assert_composition_match(
    actual: Dict[str, float],
    target: Dict[str, float],
    tolerance: float = 0.05
):
    """
    Assert that actual composition matches target within tolerance.
    
    Args:
        actual: Actual atomic fractions
        target: Target atomic fractions
        tolerance: Maximum deviation per element
        
    Raises:
        AssertionError: If composition doesn't match
    """
    for element, target_frac in target.items():
        actual_frac = actual.get(element, 0.0)
        deviation = abs(actual_frac - target_frac)
        assert deviation <= tolerance, \
            f"Element {element}: deviation {deviation:.4f} exceeds tolerance {tolerance}"


def assert_energy_stability(
    energy_above_hull: float,
    max_ehull: float = 0.001,
    description: str = "stable"
):
    """
    Assert that energy above hull is within expected range.
    
    Args:
        energy_above_hull: Energy above hull in eV/atom
        max_ehull: Maximum allowed energy above hull
        description: Description of stability requirement
        
    Raises:
        AssertionError: If energy above hull exceeds threshold
    """
    assert energy_above_hull <= max_ehull, \
        f"Material should be {description} (Ehull <= {max_ehull}), got {energy_above_hull}"


def assert_pagination_structure(data: Dict[str, Any]):
    """
    Assert that paginated response has proper structure.
    
    Args:
        data: Data dictionary from response
        
    Raises:
        AssertionError: If pagination structure is invalid
    """
    assert "total_count" in data, "Must contain total_count"
    assert "page" in data, "Must contain page"
    assert "per_page" in data, "Must contain per_page"
    assert "total_pages" in data, "Must contain total_pages"
    assert "data" in data, "Must contain data list"
    
    assert isinstance(data["data"], list), "data must be a list"
    assert isinstance(data["page"], int), "page must be an integer"
    assert isinstance(data["per_page"], int), "per_page must be an integer"
    assert data["page"] >= 1, "page must be >= 1"
    assert data["per_page"] >= 1, "per_page must be >= 1"


def assert_comparison_structure(data: Dict[str, Any]):
    """
    Assert that property comparison has proper structure.
    
    Args:
        data: Comparison data dictionary
        
    Raises:
        AssertionError: If comparison structure is invalid
    """
    assert "property_name" in data, "Must contain property_name"
    assert "material1" in data, "Must contain material1"
    assert "material2" in data, "Must contain material2"
    assert "comparison" in data, "Must contain comparison"
    
    comp = data["comparison"]
    assert "absolute_difference" in comp, "comparison must contain absolute_difference"
    assert "percent_change" in comp, "comparison must contain percent_change"
    assert "ratio" in comp, "comparison must contain ratio"


def assert_doping_analysis_structure(data: Dict[str, Any]):
    """
    Assert that doping analysis has proper structure.
    
    Args:
        data: Doping analysis data dictionary
        
    Raises:
        AssertionError: If doping analysis structure is invalid
    """
    assert "host_element" in data, "Must contain host_element"
    assert "dopant_element" in data, "Must contain dopant_element"
    assert "target_dopant_concentration" in data, "Must contain target_dopant_concentration"
    assert "property_analyzed" in data, "Must contain property_analyzed"
    assert "pure_host" in data, "Must contain pure_host"
    
    # Can have either comparisons or VRH estimate
    has_comparisons = "comparisons" in data and len(data.get("comparisons", [])) > 0
    has_vrh = "vrh_estimate" in data or "mixture_model_estimate" in data
    
    assert has_comparisons or has_vrh, \
        "Must contain either comparisons or VRH/mixture model estimate"


def assert_data_quality_flags(data: Dict[str, Any], expected_flags: Optional[List[str]] = None):
    """
    Assert that data quality flags match expectations.
    
    Args:
        data: Data dictionary with mechanical_stability field
        expected_flags: Expected list of flags (None means any flags acceptable)
        
    Raises:
        AssertionError: If flags don't match expectations
    """
    if "mechanical_stability" in data:
        mech = data["mechanical_stability"]
        actual_flags = mech.get("flags")
        
        if expected_flags is not None:
            if expected_flags:
                assert actual_flags is not None, "Expected flags but got None"
                assert set(actual_flags) == set(expected_flags), \
                    f"Expected flags {expected_flags}, got {actual_flags}"
            else:
                assert actual_flags is None or not actual_flags, \
                    f"Expected no flags, got {actual_flags}"

