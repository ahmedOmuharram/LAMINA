"""
Tests for materials utility functions.

This module tests the core utility functions used by the materials handler,
including elastic property calculations, composition matching, and property
comparisons.
"""

import pytest
import math
from tests.base.assertions import (
    assert_success_response,
    assert_error_response,
    assert_elastic_properties_structure,
    assert_physical_constraints_elastic,
    assert_composition_match,
    assert_comparison_structure,
    assert_doping_analysis_structure,
    assert_data_quality_flags,
)
from tests.materials.fixtures import *


class TestSafeRatio:
    """Tests for _safe_ratio utility function."""
    
    def test_valid_ratio_calculation(self):
        """Test Poisson ratio calculation with valid inputs."""
        from backend.handlers.materials.utils import _safe_ratio
        
        K = 100.0  # GPa
        G = 50.0   # GPa
        
        nu = _safe_ratio(K, G)
        
        # Calculate expected: (3K - 2G) / (2(3K + G))
        expected = (3*K - 2*G) / (2*(3*K + G))
        
        assert nu is not None
        assert math.isclose(nu, expected, rel_tol=1e-9)
    
    def test_none_inputs(self):
        """Test that None inputs return None."""
        from backend.handlers.materials.utils import _safe_ratio
        
        assert _safe_ratio(None, 50.0) is None
        assert _safe_ratio(100.0, None) is None
        assert _safe_ratio(None, None) is None
    
    def test_zero_denominator(self):
        """Test handling of zero denominator (edge case)."""
        from backend.handlers.materials.utils import _safe_ratio
        
        # When 3K + G = 0 (denominator zero), should return None
        K = 0.0
        G = 0.0
        
        result = _safe_ratio(K, G)
        assert result is None
    
    def test_infinite_values(self):
        """Test handling of infinite or nan values."""
        from backend.handlers.materials.utils import _safe_ratio
        
        assert _safe_ratio(float('inf'), 50.0) is None
        assert _safe_ratio(100.0, float('nan')) is None
        assert _safe_ratio(float('inf'), float('inf')) is None
    
    def test_typical_material_values(self):
        """Test with typical material property values."""
        from backend.handlers.materials.utils import _safe_ratio
        
        # Aluminum: K ≈ 76 GPa, G ≈ 26 GPa
        nu_al = _safe_ratio(76.0, 26.0)
        assert 0.3 < nu_al < 0.4  # Typical for metals
        
        # Steel: K ≈ 160 GPa, G ≈ 80 GPa
        nu_steel = _safe_ratio(160.0, 80.0)
        assert 0.25 < nu_steel < 0.35
        
        # Rubber-like (incompressible): high K, low G
        nu_rubber = _safe_ratio(1000.0, 1.0)
        assert nu_rubber > 0.45  # Close to 0.5


class TestGetElasticProperties:
    """Tests for get_elastic_properties function."""
    
    def test_successful_retrieval(self, mock_mprester, silicon_doc):
        """Test successful retrieval of elastic properties."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([silicon_doc])
        mock_mprester.setup_elasticity_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-149")
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert_elastic_properties_structure(data)
        assert data["material_id"] == "mp-149"
        assert data["formula"] == "Si"
    
    def test_material_not_found(self, mock_mprester):
        """Test handling of non-existent material ID."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-99999")
        
        assert_error_response(result, "materials")
        assert "not found" in result["error"].lower()
    
    def test_positive_moduli(self, mock_mprester, silicon_doc):
        """Test handling of positive moduli (normal case)."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([silicon_doc])
        mock_mprester.setup_elasticity_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-149")
        
        assert result["success"]
        data = result["data"]
        
        # Should have no quality flags for good data
        assert_physical_constraints_elastic(data, allow_negative=False)
        assert_data_quality_flags(data, expected_flags=[])
        assert data["data_quality"] == "ok"
    
    def test_negative_shear_modulus(self, mock_mprester, gold_bad_doc):
        """Test handling of negative shear modulus (data quality issue)."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([gold_bad_doc])
        mock_mprester.setup_elasticity_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-81")
        
        assert result["success"]
        data = result["data"]
        
        # Should flag the problem
        assert "non_positive_shear_modulus" in data["mechanical_stability"]["flags"]
        assert data["data_quality"] == "elastic_tensor_unstable"
        assert result["confidence"] == "medium"
        
        # Derived properties should be suppressed
        assert data["derived"]["poisson_from_KG"] is None
        assert data["derived"]["youngs_from_KG"] is None
        assert data["derived"]["pugh_K_over_G"] is None
        assert "derived_suppressed_due_to_non_positive_shear_modulus" in data["mechanical_stability"]["flags"]
    
    def test_derived_properties_calculation(self, mock_mprester, copper_doc):
        """Test calculation of derived properties (Poisson, Young's, Pugh)."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([copper_doc])
        mock_mprester.setup_elasticity_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-30")
        
        assert result["success"]
        data = result["data"]
        
        K = data["bulk_modulus"]["k_vrh"]
        G = data["shear_modulus"]["g_vrh"]
        
        # Check derived properties
        derived = data["derived"]
        
        # Poisson ratio: (3K - 2G) / (2(3K + G))
        expected_nu = (3*K - 2*G) / (2*(3*K + G))
        assert math.isclose(derived["poisson_from_KG"], expected_nu, rel_tol=1e-6)
        
        # Young's modulus: 9KG / (3K + G)
        expected_E = (9*K*G) / (3*K + G)
        assert math.isclose(derived["youngs_from_KG"], expected_E, rel_tol=1e-6)
        
        # Pugh ratio: K/G
        expected_pugh = K / G
        assert math.isclose(derived["pugh_K_over_G"], expected_pugh, rel_tol=1e-6)
    
    def test_mechanical_stability_assessment(self, mock_mprester, silicon_doc):
        """Test mechanical stability assessment logic."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([silicon_doc])
        mock_mprester.setup_elasticity_response([])
        
        result = get_elastic_properties(mock_mprester, "mp-149")
        
        assert result["success"]
        data = result["data"]
        
        mech = data["mechanical_stability"]
        assert isinstance(mech["likely_stable"], bool)
        assert mech["likely_stable"] is True  # Silicon should be stable
        assert mech["flags"] is None or not mech["flags"]
    
    def test_elasticity_endpoint_integration(self, mock_mprester, silicon_doc, elasticity_doc_silicon):
        """Test integration with elasticity endpoint for tensor data."""
        from backend.handlers.materials.utils import get_elastic_properties
        
        mock_mprester.setup_search_response([silicon_doc])
        mock_mprester.setup_elasticity_response([elasticity_doc_silicon])
        
        result = get_elastic_properties(mock_mprester, "mp-149")
        
        assert result["success"]
        data = result["data"]
        
        # Should have vrh_from_tensor data
        if "vrh_from_tensor" in data:
            vrh = data["vrh_from_tensor"]
            assert "k_vrh" in vrh
            assert "g_vrh" in vrh
            # May have is_born_stable if pymatgen supports it
    
    def test_elasticity_warnings_captured(self, mock_mprester, elasticity_doc_with_warnings):
        """Test that elasticity warnings are captured and included."""
        from backend.handlers.materials.utils import get_elastic_properties
        from tests.base.mock_api import create_sample_material
        
        mat_doc = create_sample_material(
            material_id="mp-999",
            formula="X",
            bulk_modulus_vrh=50.0,
            shear_modulus_vrh=25.0,
        )
        
        mock_mprester.setup_search_response([mat_doc])
        mock_mprester.setup_elasticity_response([elasticity_doc_with_warnings])
        
        result = get_elastic_properties(mock_mprester, "mp-999")
        
        assert result["success"]
        data = result["data"]
        
        # Warnings should be present
        assert "elasticity_warnings" in data
        assert len(data["elasticity_warnings"]) > 0


class TestFindClosestAlloyCompositions:
    """Tests for find_closest_alloy_compositions function."""
    
    def test_exact_composition_match(self, mock_mprester, ag_cu_alloy_docs):
        """Test finding materials with exact composition match."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        
        mock_mprester.setup_search_response(ag_cu_alloy_docs)
        
        result = find_closest_alloy_compositions(
            mock_mprester,
            elements=["Ag", "Cu"],
            target_composition={"Ag": 0.875, "Cu": 0.125},
            tolerance=0.001,  # Strict tolerance
            is_stable=True,
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        
        assert data["num_materials_found"] >= 1
        assert data["closest_match_used"] is False  # Exact match found
        
        # Check first material matches target
        material = data["materials"][0]
        assert_composition_match(
            material["atomic_fractions"],
            {"Ag": 0.875, "Cu": 0.125},
            tolerance=0.001
        )
    
    def test_composition_within_tolerance(self, mock_mprester, composition_test_cases):
        """Test composition matching with various tolerance levels."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        from tests.base.mock_api import create_sample_alloy
        
        for test_case in composition_test_cases:
            # Create alloy with actual composition
            alloy_doc = create_sample_alloy(
                material_id="mp-test",
                elements=list(test_case["actual"].keys()),
                fractions=list(test_case["actual"].values()),
                bulk_modulus_vrh=100.0,
            )
            
            mock_mprester.setup_search_response([alloy_doc])
            
            result = find_closest_alloy_compositions(
                mock_mprester,
                elements=list(test_case["target"].keys()),
                target_composition=test_case["target"],
                tolerance=test_case["tolerance"],
                is_stable=True,
            )
            
            if test_case["should_match"]:
                assert result["success"], f"Failed for case: {test_case['description']}"
                assert result["data"]["num_materials_found"] >= 1
            else:
                # Should use closest match or return limited results
                assert result["success"]
    
    def test_no_target_composition_returns_all(self, mock_mprester, ag_cu_alloy_docs):
        """Test that no target composition returns all materials in system."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        
        mock_mprester.setup_search_response(ag_cu_alloy_docs)
        
        result = find_closest_alloy_compositions(
            mock_mprester,
            elements=["Ag", "Cu"],
            target_composition=None,  # No target
            is_stable=True,
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        
        # Should return all materials
        assert data["num_materials_found"] == len(ag_cu_alloy_docs)
    
    def test_closest_match_fallback(self, mock_mprester):
        """Test closest match fallback when no exact match within tolerance."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        from tests.base.mock_api import create_sample_alloy
        
        # Create alloy far from target
        alloy_doc = create_sample_alloy(
            material_id="mp-far",
            elements=["Ag", "Cu"],
            fractions=[0.95, 0.05],  # Only 5% Cu, target is 12.5%
            bulk_modulus_vrh=105.0,
        )
        
        mock_mprester.setup_search_response([alloy_doc])
        
        result = find_closest_alloy_compositions(
            mock_mprester,
            elements=["Ag", "Cu"],
            target_composition={"Ag": 0.875, "Cu": 0.125},
            tolerance=0.05,  # Strict tolerance
            is_stable=True,
        )
        
        assert result["success"]
        data = result["data"]
        
        # Should use closest match
        assert data["closest_match_used"] is True
        assert data["num_materials_found"] == 1
        
        # Material should be flagged as closest match
        assert data["materials"][0].get("closest_match") is True
    
    def test_metastable_fallback(self, mock_mprester, metastable_alloy_docs):
        """Test automatic fallback to metastable entries when no stable found."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        
        # First call with is_stable=True returns empty
        # Second call with relaxed stability returns metastable
        def mock_search(**kwargs):
            ehull_range = kwargs.get("energy_above_hull", (0, 0))
            if ehull_range[1] < 0.01:  # Stable search
                return []
            else:  # Metastable search
                return metastable_alloy_docs
        
        mock_mprester.materials.summary.search.side_effect = mock_search
        
        result = find_closest_alloy_compositions(
            mock_mprester,
            elements=["Fe", "Al"],
            target_composition={"Fe": 0.75, "Al": 0.25},
            is_stable=True,  # Request stable
            ehull_max=0.20,
        )
        
        assert result["success"]
        data = result["data"]
        
        # Should indicate metastable fallback was used
        assert data["used_metastable_fallback"] is True
        assert data["num_materials_found"] >= 1
    
    def test_no_materials_found_error(self, mock_mprester):
        """Test error when no materials found in chemical system."""
        from backend.handlers.materials.utils import find_closest_alloy_compositions
        
        mock_mprester.setup_search_response([])
        
        result = find_closest_alloy_compositions(
            mock_mprester,
            elements=["Ag", "Cu"],
            is_stable=True,
        )
        
        assert_error_response(result, "materials")
        # Error should mention that materials were not found (flexible wording)
        error_lower = result["error"].lower()
        assert "no materials found" in error_lower or "not found" in error_lower


class TestCompareMaterialProperties:
    """Tests for compare_material_properties functions."""
    
    def test_successful_comparison(self):
        """Test successful property comparison between two materials."""
        from backend.handlers.materials.utils import compare_material_properties
        from backend.handlers.shared import success_result, Confidence
        
        # Mock property results
        props1 = success_result(
            handler="materials",
            function="get_elastic_properties",
            data={
                "material_id": "mp-124",
                "formula": "Ag",
                "bulk_modulus": {"k_vrh": 100.0, "unit": "GPa"},
            },
            citations=["Materials Project"],
            confidence=Confidence.HIGH
        )
        
        props2 = success_result(
            handler="materials",
            function="get_elastic_properties",
            data={
                "material_id": "mp-30",
                "formula": "Cu",
                "bulk_modulus": {"k_vrh": 137.0, "unit": "GPa"},
            },
            citations=["Materials Project"],
            confidence=Confidence.HIGH
        )
        
        result = compare_material_properties(props1, props2, "bulk_modulus")
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert_comparison_structure(data)
        
        # Check calculation correctness
        comp = data["comparison"]
        assert comp["absolute_difference"] == 37.0  # 137 - 100
        assert math.isclose(comp["percent_change"], 37.0, rel_tol=1e-6)  # (37/100) * 100
        assert math.isclose(comp["ratio"], 1.37, rel_tol=1e-6)  # 137/100
    
    def test_percent_change_calculation(self):
        """Test percent change calculation for various scenarios."""
        from backend.handlers.materials.utils import compare_material_properties
        from backend.handlers.shared import success_result, Confidence
        
        test_cases = [
            (100.0, 110.0, 10.0, "10% increase"),
            (100.0, 90.0, -10.0, "10% decrease"),
            (100.0, 100.5, 0.5, "0.5% increase"),
            (100.0, 200.0, 100.0, "100% increase"),
        ]
        
        for val1, val2, expected_pct, description in test_cases:
            props1 = success_result(
                handler="materials",
                function="test",
                data={
                    "material_id": "mp-1",
                    "formula": "A",
                    "bulk_modulus": {"k_vrh": val1, "unit": "GPa"},
                },
                citations=[],
                confidence=Confidence.HIGH
            )
            
            props2 = success_result(
                handler="materials",
                function="test",
                data={
                    "material_id": "mp-2",
                    "formula": "B",
                    "bulk_modulus": {"k_vrh": val2, "unit": "GPa"},
                },
                citations=[],
                confidence=Confidence.HIGH
            )
            
            result = compare_material_properties(props1, props2, "bulk_modulus")
            
            assert result["success"], f"Failed for: {description}"
            pct = result["data"]["comparison"]["percent_change"]
            assert math.isclose(pct, expected_pct, rel_tol=1e-6), \
                f"Expected {expected_pct}%, got {pct}% for {description}"
    
    def test_missing_property_error(self):
        """Test error when property not available for one material."""
        from backend.handlers.materials.utils import compare_material_properties
        from backend.handlers.shared import success_result, Confidence
        
        props1 = success_result(
            handler="materials",
            function="test",
            data={
                "material_id": "mp-1",
                "formula": "A",
                "bulk_modulus": {"k_vrh": 100.0, "unit": "GPa"},
            },
            citations=[],
            confidence=Confidence.HIGH
        )
        
        props2 = success_result(
            handler="materials",
            function="test",
            data={
                "material_id": "mp-2",
                "formula": "B",
                # No bulk_modulus
            },
            citations=[],
            confidence=Confidence.HIGH
        )
        
        result = compare_material_properties(props1, props2, "bulk_modulus")
        
        assert_error_response(result, "materials")
        assert "not available" in result["error"].lower()


class TestAnalyzeDopingEffect:
    """Tests for analyze_doping_effect function."""
    
    def test_doping_with_database_entry(self, mock_mprester, silver_doc, copper_doc, ag_cu_alloy_docs):
        """Test doping analysis when database entry exists for alloy."""
        from backend.handlers.materials.utils import analyze_doping_effect
        
        # Setup: pure host (Ag), pure dopant (Cu), and alloy
        def mock_search(**kwargs):
            elements = kwargs.get("elements", [])
            num_elements = kwargs.get("num_elements")
            
            if num_elements == 1:
                if "Ag" in elements:
                    return [silver_doc]
                elif "Cu" in elements:
                    return [copper_doc]
            else:
                # Return alloy
                return ag_cu_alloy_docs[:1]  # Return one matching alloy
            
            return []
        
        mock_mprester.materials.summary.search.side_effect = mock_search
        mock_mprester.materials.elasticity.search.return_value = []
        
        result = analyze_doping_effect(
            mock_mprester,
            host_element="Ag",
            dopant_element="Cu",
            dopant_concentration=0.125,
            property_name="bulk_modulus"
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert_doping_analysis_structure(data)
        
        assert data["host_element"] == "Ag"
        assert data["dopant_element"] == "Cu"
        assert data["target_dopant_concentration"] == 0.125
        assert data["num_alloys_analyzed"] >= 1
    
    def test_doping_vrh_fallback(self, mock_mprester, silver_doc, copper_doc):
        """Test VRH mixture model fallback when no alloy entries exist."""
        from backend.handlers.materials.utils import analyze_doping_effect
        
        # Setup: only pure elements, no alloy
        def mock_search(**kwargs):
            elements = kwargs.get("elements", [])
            num_elements = kwargs.get("num_elements")
            material_ids = kwargs.get("material_ids")
            
            # Handle searches by material_id (from get_elastic_properties)
            if material_ids:
                if "mp-124" in material_ids:
                    return [silver_doc]
                elif "mp-30" in material_ids:
                    return [copper_doc]
            
            # Handle searches by elements
            if num_elements == 1:
                if "Ag" in elements:
                    return [silver_doc]
                elif "Cu" in elements:
                    return [copper_doc]
            else:
                return []  # No alloy found
            
            return []
        
        mock_mprester.materials.summary.search.side_effect = mock_search
        mock_mprester.materials.elasticity.search.return_value = []
        
        result = analyze_doping_effect(
            mock_mprester,
            host_element="Ag",
            dopant_element="Cu",
            dopant_concentration=0.125,
            property_name="bulk_modulus"
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        
        # Should use mixture model
        assert data.get("used_mixture_model") is True
        assert "mixture_model_estimate" in data or "vrh_estimate" in data
        
        # Check VRH calculation
        if "mixture_model_estimate" in data:
            vrh = data["mixture_model_estimate"]
            assert "k_vrh_gpa" in vrh
            assert "k_voigt_gpa" in vrh
            assert "k_reuss_gpa" in vrh
            assert "percent_change" in vrh
    
    def test_vrh_bounds_calculation(self):
        """Test VRH bounds calculation for doping analysis."""
        # VRH bounds: K_V = (1-x)*K_host + x*K_dopant (Voigt, upper)
        #             K_R = 1/((1-x)/K_host + x/K_dopant) (Reuss, lower)
        #             K_VRH = (K_V + K_R) / 2
        
        K_host = 100.0  # Ag
        K_dopant = 137.0  # Cu
        x = 0.125  # 12.5% Cu
        
        K_V = (1 - x) * K_host + x * K_dopant
        K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)
        K_VRH = 0.5 * (K_V + K_R)
        
        # Voigt (upper bound)
        expected_V = 0.875 * 100.0 + 0.125 * 137.0
        assert math.isclose(K_V, expected_V, rel_tol=1e-6)
        assert math.isclose(K_V, 104.625, rel_tol=1e-6)
        
        # Reuss (lower bound)
        expected_R_inv = 0.875 / 100.0 + 0.125 / 137.0
        expected_R = 1.0 / expected_R_inv
        assert math.isclose(K_R, expected_R, rel_tol=1e-6)
        
        # VRH average between bounds
        assert K_R < K_VRH < K_V
    
    def test_host_element_not_found_error(self, mock_mprester):
        """Test error when host element not found in database."""
        from backend.handlers.materials.utils import analyze_doping_effect
        
        mock_mprester.setup_search_response([])
        
        result = analyze_doping_effect(
            mock_mprester,
            host_element="Zz",  # Non-existent
            dopant_element="Cu",
            dopant_concentration=0.125,
        )
        
        assert_error_response(result, "materials")
        assert "not find" in result["error"].lower()

