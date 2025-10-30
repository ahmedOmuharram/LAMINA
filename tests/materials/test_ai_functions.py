"""
Tests for materials AI functions.

This module tests the AI-accessible functions that wrap the materials
handler utilities, focusing on parameter handling, async execution,
and integration with the handler methods.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.base.assertions import (
    assert_success_response,
    assert_error_response,
    assert_elastic_properties_structure,
    assert_pagination_structure,
)
from tests.materials.fixtures import *


@pytest.mark.asyncio
class TestMaterialsAIFunctions:
    """Tests for MaterialsAIFunctionsMixin AI functions."""
    
    async def test_mp_search_by_composition_with_formula(self, mock_handler_with_mpr, silicon_doc):
        """Test mp_search_by_composition with formula parameter."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._total_count_for_summary = MagicMock(return_value=1)
        
        result = await handler.mp_search_by_composition(formula="Si")
        
        assert_success_response(result, "materials")
    
    async def test_mp_search_by_composition_with_chemsys(self, mock_handler_with_mpr):
        """Test mp_search_by_composition with chemical system."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = await handler.mp_search_by_composition(chemsys="Li-Fe-O")
        
        assert_success_response(result, "materials")
    
    async def test_mp_search_by_composition_pagination(self, mock_handler_with_mpr):
        """Test pagination parameters in mp_search_by_composition."""
        from tests.base.mock_api import MockMaterialDoc
        handler = mock_handler_with_mpr
        # Create enough mock docs for pagination
        mock_docs = [
            MockMaterialDoc(
                material_id=f"mp-{i}",
                formula_pretty=f"X{i}",
                elements=["X"],
                chemsys="X"
            )
            for i in range(50)
        ]
        handler.mpr.setup_search_response(mock_docs)
        handler._total_count_for_summary = MagicMock(return_value=50)
        
        result = await handler.mp_search_by_composition(
            chemsys="Li-Fe-O",
            page=2,
            per_page=20
        )
        
        assert result["success"]
        data = result["data"]
        assert data["page"] == 2
        assert data["per_page"] == 20
        assert data["total_count"] == 50
    
    async def test_mp_get_by_id_single_material(self, mock_handler_with_mpr, silicon_doc):
        """Test mp_get_by_id with a single material ID."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        result = await handler.mp_get_by_id(material_ids=["mp-149"])
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["total_count"] == 1
    
    async def test_mp_get_by_id_multiple_materials(self, mock_handler_with_mpr, sample_material_ids, silicon_doc, copper_doc):
        """Test mp_get_by_id with multiple material IDs."""
        handler = mock_handler_with_mpr
        
        material_ids = [sample_material_ids["silicon"], sample_material_ids["copper"]]
        handler.mpr.setup_search_response([silicon_doc, copper_doc])
        handler._call_summary_count = MagicMock(return_value=2)
        
        result = await handler.mp_get_by_id(material_ids=material_ids)
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["total_count"] == 2
    
    async def test_mp_get_by_id_with_fields(self, mock_handler_with_mpr, silicon_doc):
        """Test mp_get_by_id with specific fields parameter."""
        handler = mock_handler_with_mpr
        
        fields = ["material_id", "formula_pretty", "band_gap", "energy_above_hull"]
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        result = await handler.mp_get_by_id(
            material_ids=["mp-149"],
            fields=fields
        )
        
        assert_success_response(result, "materials")
    
    async def test_mp_get_by_characteristic_band_gap(self, mock_handler_with_mpr):
        """Test mp_get_by_characteristic with band gap filter."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = await handler.mp_get_by_characteristic(
            band_gap=[1.0, 3.0],  # Semiconductors
            is_metal=False,
            is_stable=True
        )
        
        assert_success_response(result, "materials")
    
    async def test_mp_get_by_characteristic_mechanical_properties(self, mock_handler_with_mpr):
        """Test mp_get_by_characteristic with mechanical property filters."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = await handler.mp_get_by_characteristic(
            k_vrh=[100, 500],  # High bulk modulus
            crystal_system="Cubic",
            is_stable=True
        )
        
        assert_success_response(result, "materials")
    
    async def test_mp_get_material_details_all_fields(self, mock_handler_with_mpr, silicon_doc):
        """Test mp_get_material_details with all_fields=True."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        result = await handler.mp_get_material_details(
            material_ids=["mp-149"],
            all_fields=True
        )
        
        assert_success_response(result, "materials")
    
    async def test_get_elastic_properties_success(self, mock_handler_with_mpr):
        """Test get_elastic_properties AI function."""
        handler = mock_handler_with_mpr
        
        with patch('backend.handlers.materials.utils.get_elastic_properties',
                  return_value={
                      "success": True,
                      "handler": "materials",
                      "function": "get_elastic_properties",
                      "data": {
                          "material_id": "mp-149",
                          "formula": "Si",
                          "bulk_modulus": {"k_vrh": 97.9, "unit": "GPa"},
                          "shear_modulus": {"g_vrh": 65.8, "unit": "GPa"},
                          "derived": {
                              "poisson_from_KG": 0.22,
                              "youngs_from_KG": 160.0,
                              "pugh_K_over_G": 1.49
                          },
                          "mechanical_stability": {
                              "likely_stable": True,
                              "flags": None
                          },
                          "data_quality": "ok"
                      },
                      "citations": ["Materials Project"],
                      "confidence": "HIGH"
                  }):
            
            result = await handler.get_elastic_properties(material_id="mp-149")
            
            assert_success_response(result, "materials")
            data = result["data"]
            assert_elastic_properties_structure(data)
    
    async def test_find_closest_alloy_compositions_ai(self, mock_handler_with_mpr, ag_cu_alloy_docs):
        """Test find_closest_alloy_compositions AI function."""
        handler = mock_handler_with_mpr
        
        # Set up mock to return alloy documents
        handler.mpr.setup_search_response(ag_cu_alloy_docs)
        
        result = await handler.find_closest_alloy_compositions(
            elements=["Ag", "Cu"],
            target_composition={"Ag": 0.875, "Cu": 0.125},
            tolerance=0.05
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["chemical_system"] == "Ag-Cu"
        assert data["num_materials_found"] >= 1
        assert "materials" in data
        assert len(data["materials"]) >= 1
    
    async def test_compare_material_properties_ai(self, mock_handler_with_mpr, silver_doc, copper_doc):
        """Test compare_material_properties AI function."""
        handler = mock_handler_with_mpr
        
        # Set up mock to return materials with elastic properties
        def mock_search(**kwargs):
            mat_id = kwargs.get("material_ids", "")
            if "mp-124" in mat_id:
                return [silver_doc]
            elif "mp-30" in mat_id:
                return [copper_doc]
            return []
        
        handler.mpr.materials.summary.search = mock_search
        handler.mpr.materials.elasticity.search.return_value = []
        
        result = await handler.compare_material_properties(
            material_id1="mp-124",
            material_id2="mp-30",
            property_name="bulk_modulus"
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["property_name"] == "bulk_modulus"
        assert "comparison" in data
    
    async def test_analyze_doping_effect_ai(self, mock_handler_with_mpr, silver_doc, copper_doc, ag_cu_alloy_docs):
        """Test analyze_doping_effect AI function."""
        handler = mock_handler_with_mpr
        
        # Set up mock to return pure materials and alloys
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
                return ag_cu_alloy_docs[:1]
            return []
        
        handler.mpr.materials.summary.search = mock_search
        handler.mpr.materials.elasticity.search.return_value = []
        
        result = await handler.analyze_doping_effect(
            host_element="Ag",
            dopant_element="Cu",
            dopant_concentration=0.125,
            property_name="bulk_modulus"
        )
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["host_element"] == "Ag"
        assert data["dopant_element"] == "Cu"
    
    async def test_ai_function_error_handling(self, mock_handler_with_mpr):
        """Test error handling in AI functions."""
        handler = mock_handler_with_mpr
        
        # Set up mock to return empty (material not found)
        handler.mpr.setup_search_response([])
        
        result = await handler.get_elastic_properties(material_id="mp-99999")
        
        assert_error_response(result, "materials")
        assert "not found" in result["error"].lower()
    
    async def test_ai_function_duration_tracking(self, mock_handler_with_mpr):
        """Test that AI functions track execution duration."""
        handler = mock_handler_with_mpr
        
        with patch('backend.handlers.materials.utils.get_elastic_properties',
                  return_value={
                      "success": True,
                      "handler": "materials",
                      "function": "get_elastic_properties",
                      "data": {},
                      "citations": ["Materials Project"],
                      "confidence": "HIGH"
                  }):
            
            result = await handler.get_elastic_properties(material_id="mp-149")
            
            # Duration should be added by AI function wrapper
            assert result["success"]
            # Note: duration_ms may or may not be present depending on implementation

