"""
Tests for MaterialHandler class methods.

This module tests the handler-level methods that interface with
the Materials Project API, focusing on parameter parsing,
pagination, and response formatting.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from tests.base.assertions import (
    assert_success_response,
    assert_error_response,
    assert_pagination_structure,
)
from tests.materials.fixtures import *


class TestMaterialHandlerParameterParsing:
    """Tests for parameter parsing in handler methods."""
    
    def test_parse_csv_list(self, mock_handler_with_mpr):
        """Test CSV list parsing."""
        handler = mock_handler_with_mpr
        
        # Test string input
        result = handler._parse_csv_list("mp-149,mp-30,mp-81")
        assert result == ["mp-149", "mp-30", "mp-81"]
        
        # Test list input
        result = handler._parse_csv_list(["mp-149", "mp-30"])
        assert result == ["mp-149", "mp-30"]
        
        # Test None input
        result = handler._parse_csv_list(None)
        assert result is None
        
        # Test empty string
        result = handler._parse_csv_list("")
        assert result is None
    
    def test_parse_bool(self, mock_handler_with_mpr):
        """Test boolean parsing."""
        handler = mock_handler_with_mpr
        
        # Test various true values
        assert handler._parse_bool(True) is True
        assert handler._parse_bool("true") is True
        assert handler._parse_bool("True") is True
        assert handler._parse_bool("1") is True
        assert handler._parse_bool("yes") is True
        assert handler._parse_bool("y") is True
        
        # Test various false values
        assert handler._parse_bool(False) is False
        assert handler._parse_bool("false") is False
        assert handler._parse_bool("False") is False
        assert handler._parse_bool("0") is False
        assert handler._parse_bool("no") is False
        assert handler._parse_bool("n") is False
        
        # Test None
        assert handler._parse_bool(None) is None
    
    def test_parse_int(self, mock_handler_with_mpr):
        """Test integer parsing."""
        handler = mock_handler_with_mpr
        
        assert handler._parse_int(42) == 42
        assert handler._parse_int("42") == 42
        assert handler._parse_int(None) is None
        assert handler._parse_int("") is None
        assert handler._parse_int("invalid") is None
    
    def test_get_pagination_defaults(self, mock_handler_with_mpr):
        """Test pagination defaults."""
        handler = mock_handler_with_mpr
        
        # Empty params
        page, per_page = handler._get_pagination({})
        assert page == 1
        assert per_page == 10
        
        # Custom values
        page, per_page = handler._get_pagination({"page": 2, "per_page": 20})
        assert page == 2
        assert per_page == 20
        
        # Invalid values (should use defaults)
        page, per_page = handler._get_pagination({"page": 0, "per_page": -5})
        assert page == 1
        assert per_page >= 1
    
    def test_slice_for_page(self, mock_handler_with_mpr):
        """Test pagination slicing."""
        handler = mock_handler_with_mpr
        
        items = list(range(100))  # 0-99
        
        # First page
        result = handler._slice_for_page(items, page=1, per_page=10)
        assert result == list(range(0, 10))
        
        # Second page
        result = handler._slice_for_page(items, page=2, per_page=10)
        assert result == list(range(10, 20))
        
        # Last page (partial)
        result = handler._slice_for_page(items, page=10, per_page=10)
        assert result == list(range(90, 100))
        
        # Beyond range
        result = handler._slice_for_page(items, page=20, per_page=10)
        assert result == []


class TestMaterialHandlerSearchByComposition:
    """Tests for mp_search_by_composition handler method."""
    
    def test_search_by_formula(self, mock_handler_with_mpr, silicon_doc):
        """Test search by chemical formula."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        
        # Mock the count method
        handler._total_count_for_summary = MagicMock(return_value=1)
        
        result = handler._handle_search_by_composition({"formula": "Si"})
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert_pagination_structure(data)
        assert data["total_count"] == 1
    
    def test_search_by_chemsys(self, mock_handler_with_mpr, ag_cu_alloy_docs):
        """Test search by chemical system."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response(ag_cu_alloy_docs)
        handler._total_count_for_summary = MagicMock(return_value=len(ag_cu_alloy_docs))
        
        result = handler._handle_search_by_composition({"chemsys": "Ag-Cu"})
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["total_count"] == len(ag_cu_alloy_docs)
    
    def test_search_by_elements(self, mock_handler_with_mpr):
        """Test search by element list."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = handler._handle_search_by_composition({"element": "Li,Fe,O"})
        
        # Should succeed even with no results
        assert_success_response(result, "materials")
    
    def test_search_pagination(self, mock_handler_with_mpr):
        """Test pagination in search results."""
        handler = mock_handler_with_mpr
        
        # Create 25 dummy docs
        docs = [create_sample_material(f"mp-{i}", f"X{i}") for i in range(25)]
        handler.mpr.setup_search_response(docs)
        handler._total_count_for_summary = MagicMock(return_value=25)
        
        # Get page 2 with 10 per page
        result = handler._handle_search_by_composition({
            "chemsys": "X",
            "page": 2,
            "per_page": 10
        })
        
        assert result["success"]
        data = result["data"]
        assert data["page"] == 2
        assert data["per_page"] == 10
        assert data["total_count"] == 25
        assert data["total_pages"] == 3
        assert len(data["data"]) == 10  # Second page has 10 items


class TestMaterialHandlerGetByCharacteristic:
    """Tests for mp_get_by_characteristic handler method."""
    
    def test_get_by_band_gap_range(self, mock_handler_with_mpr):
        """Test search by band gap range."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = handler._handle_get_by_characteristic({
            "band_gap": [1.0, 3.0],
            "is_metal": False
        })
        
        assert_success_response(result, "materials")
    
    def test_get_by_mechanical_properties(self, mock_handler_with_mpr):
        """Test search by mechanical properties."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = handler._handle_get_by_characteristic({
            "k_vrh": [100, 500],
            "g_vrh": [50, 200],
            "is_stable": True
        })
        
        assert_success_response(result, "materials")
    
    def test_get_by_crystal_system(self, mock_handler_with_mpr):
        """Test search by crystal system."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([])
        handler._total_count_for_summary = MagicMock(return_value=0)
        
        result = handler._handle_get_by_characteristic({
            "crystal_system": "Cubic",
            "is_stable": True
        })
        
        assert_success_response(result, "materials")
    
    def test_no_selector_error(self, mock_handler_with_mpr):
        """Test error when no selector provided."""
        handler = mock_handler_with_mpr
        
        result = handler._handle_get_by_characteristic({})
        
        assert_error_response(result, "materials")
        assert "selector" in result["error"].lower()


class TestMaterialHandlerGetMaterialDetails:
    """Tests for mp_get_material_details handler method."""
    
    def test_get_details_single_id(self, mock_handler_with_mpr, silicon_doc):
        """Test getting details for single material ID."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        result = handler._handle_material_details({
            "material_ids": ["mp-149"],
            "all_fields": True
        })
        
        assert_success_response(result, "materials")
        data = result["data"]
        assert data["total_count"] == 1
    
    def test_get_details_multiple_ids(self, mock_handler_with_mpr, sample_materials):
        """Test getting details for multiple material IDs."""
        handler = mock_handler_with_mpr
        
        docs = [sample_materials["silicon"], sample_materials["copper"]]
        handler.mpr.setup_search_response(docs)
        handler._call_summary_count = MagicMock(return_value=2)
        
        result = handler._handle_material_details({
            "material_ids": ["mp-149", "mp-30"],
            "all_fields": True
        })
        
        assert result["success"]
        data = result["data"]
        assert data["total_count"] == 2
    
    def test_get_details_with_fields(self, mock_handler_with_mpr, silicon_doc):
        """Test getting specific fields only."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        result = handler._handle_material_details({
            "material_ids": ["mp-149"],
            "fields": ["material_id", "formula_pretty", "band_gap"],
            "all_fields": False
        })
        
        assert result["success"]
    
    def test_material_ids_json_string(self, mock_handler_with_mpr, silicon_doc):
        """Test parsing material IDs from JSON string."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        # JSON string format
        result = handler._handle_material_details({
            "material_ids": '["mp-149", "mp-30"]',
            "all_fields": True
        })
        
        assert result["success"]
    
    def test_material_ids_csv_string(self, mock_handler_with_mpr, silicon_doc):
        """Test parsing material IDs from CSV string."""
        handler = mock_handler_with_mpr
        handler.mpr.setup_search_response([silicon_doc])
        handler._call_summary_count = MagicMock(return_value=1)
        
        # CSV format
        result = handler._handle_material_details({
            "material_ids": "mp-149,mp-30,mp-81",
            "all_fields": True
        })
        
        assert result["success"]


class TestMaterialHandlerBuildSearchKwargs:
    """Tests for _build_summary_search_kwargs method."""
    
    def test_build_kwargs_with_range_parameters(self, mock_handler_with_mpr):
        """Test building search kwargs with range parameters."""
        handler = mock_handler_with_mpr
        
        params = {
            "band_gap": [1.0, 3.0],
            "k_vrh": [100, 200],
            "is_stable": True
        }
        
        kwargs = handler._build_summary_search_kwargs(params)
        
        # Should not have errors
        assert "__errors__" not in kwargs
        
        # Ranges should be tuples
        if "band_gap" in kwargs:
            assert isinstance(kwargs["band_gap"], tuple)
            assert kwargs["band_gap"] == (1.0, 3.0)
    
    def test_build_kwargs_invalid_range(self, mock_handler_with_mpr):
        """Test error handling for invalid range parameters."""
        handler = mock_handler_with_mpr
        
        params = {
            "band_gap": [1.0]  # Only one value (invalid)
        }
        
        kwargs = handler._build_summary_search_kwargs(params)
        
        # Should have errors
        if "__errors__" in kwargs:
            assert len(kwargs["__errors__"]) > 0


class TestMaterialHandlerConvertDocs:
    """Tests for _convert_docs_to_dicts method."""
    
    def test_convert_single_doc(self, mock_handler_with_mpr, silicon_doc):
        """Test converting a single document to dictionary."""
        handler = mock_handler_with_mpr
        
        docs = [silicon_doc]
        result = handler._convert_docs_to_dicts(docs)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "material_id" in result[0]
    
    def test_convert_multiple_docs(self, mock_handler_with_mpr, sample_materials):
        """Test converting multiple documents."""
        handler = mock_handler_with_mpr
        
        docs = list(sample_materials.values())
        result = handler._convert_docs_to_dicts(docs)
        
        assert len(result) == len(docs)
        for item in result:
            assert isinstance(item, dict)

