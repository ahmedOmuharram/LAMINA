"""
Fixtures specific to materials handler testing.

This module provides fixtures for mocking Materials Project API
responses and creating test data for materials handler tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from tests.base.mock_api import (
    MockMPRester,
    MockMaterialDoc,
    MockElasticityDoc,
    MockModulus,
    create_sample_material,
    create_sample_alloy,
    SAMPLE_MATERIALS,
)


@pytest.fixture
def mock_mprester():
    """Fixture providing a mock MPRester client."""
    return MockMPRester()


@pytest.fixture
def sample_materials():
    """Fixture providing pre-configured sample materials."""
    return SAMPLE_MATERIALS


@pytest.fixture
def silicon_doc():
    """Fixture for silicon material document."""
    return SAMPLE_MATERIALS["silicon"]


@pytest.fixture
def copper_doc():
    """Fixture for copper material document."""
    return SAMPLE_MATERIALS["copper"]


@pytest.fixture
def silver_doc():
    """Fixture for silver material document."""
    return SAMPLE_MATERIALS["silver"]


@pytest.fixture
def gold_bad_doc():
    """Fixture for gold with bad elastic data (negative shear modulus)."""
    return SAMPLE_MATERIALS["gold_bad_data"]


@pytest.fixture
def ag_cu_alloy_docs():
    """Fixture for Ag-Cu alloy test data at various compositions."""
    return [
        create_sample_alloy(
            material_id="mp-1001",
            elements=["Ag", "Cu"],
            fractions=[0.875, 0.125],  # 12.5% Cu
            is_stable=True,
            energy_above_hull=0.0,
            bulk_modulus_vrh=110.0,
        ),
        create_sample_alloy(
            material_id="mp-1002",
            elements=["Ag", "Cu"],
            fractions=[0.75, 0.25],  # 25% Cu
            is_stable=True,
            energy_above_hull=0.0,
            bulk_modulus_vrh=115.0,
        ),
        create_sample_alloy(
            material_id="mp-1003",
            elements=["Ag", "Cu"],
            fractions=[0.50, 0.50],  # 50% Cu
            is_stable=True,
            energy_above_hull=0.0,
            bulk_modulus_vrh=120.0,
        ),
    ]


@pytest.fixture
def metastable_alloy_docs():
    """Fixture for metastable alloy test data."""
    return [
        create_sample_alloy(
            material_id="mp-2001",
            elements=["Fe", "Al"],
            fractions=[0.75, 0.25],
            is_stable=False,
            energy_above_hull=0.05,  # Metastable
            bulk_modulus_vrh=180.0,
        ),
        create_sample_alloy(
            material_id="mp-2002",
            elements=["Fe", "Al"],
            fractions=[0.60, 0.40],
            is_stable=False,
            energy_above_hull=0.15,  # More metastable
            bulk_modulus_vrh=170.0,
        ),
    ]


@pytest.fixture
def elasticity_doc_silicon():
    """Fixture for silicon elasticity data with Born stability."""
    return MockElasticityDoc(
        material_id="mp-149",
        K_VRH=97.9,
        G_VRH=65.8,
        elastic_tensor=[
            [165.7, 63.9, 63.9, 0, 0, 0],
            [63.9, 165.7, 63.9, 0, 0, 0],
            [63.9, 63.9, 165.7, 0, 0, 0],
            [0, 0, 0, 79.6, 0, 0],
            [0, 0, 0, 0, 79.6, 0],
            [0, 0, 0, 0, 0, 79.6],
        ],
        warnings=[],
    )


@pytest.fixture
def elasticity_doc_with_warnings():
    """Fixture for elasticity data with warnings."""
    return MockElasticityDoc(
        material_id="mp-999",
        K_VRH=50.0,
        G_VRH=25.0,
        elastic_tensor=None,
        warnings=["Elastic tensor may be unreliable"],
    )


@pytest.fixture
def mock_handler_with_mpr(mock_mprester):
    """
    Fixture providing a MaterialHandler with mock MPRester.
    
    This uses actual handler class with mocked API client.
    """
    from backend.handlers.materials.material_handler import MaterialHandler
    
    handler = MaterialHandler(mpr=mock_mprester)
    # Mock the _track_tool_output method
    handler._track_tool_output = MagicMock()
    
    return handler


@pytest.fixture
def mock_summary_search(mock_mprester):
    """
    Fixture that provides a function to easily configure mock search responses.
    
    Returns:
        Function that accepts list of docs and configures the mock
    """
    def configure(docs):
        mock_mprester.materials.summary.search.return_value = docs
        return mock_mprester
    
    return configure


@pytest.fixture
def mock_elasticity_search(mock_mprester):
    """
    Fixture that provides a function to easily configure mock elasticity responses.
    
    Returns:
        Function that accepts list of docs and configures the mock
    """
    def configure(docs):
        mock_mprester.materials.elasticity.search.return_value = docs
        return mock_mprester
    
    return configure


@pytest.fixture
def valid_elastic_params():
    """Fixture providing valid parameters for elastic property tests."""
    return {
        "K_positive": 100.0,
        "G_positive": 50.0,
        "K_zero": 0.0,
        "G_zero": 0.0,
        "K_negative": -10.0,
        "G_negative": -5.74,  # Like Au mp-81
    }


@pytest.fixture
def composition_test_cases():
    """Fixture providing test cases for composition matching."""
    return [
        {
            "target": {"Ag": 0.875, "Cu": 0.125},
            "actual": {"Ag": 0.875, "Cu": 0.125},
            "tolerance": 0.05,
            "should_match": True,
            "description": "Exact match",
        },
        {
            "target": {"Ag": 0.875, "Cu": 0.125},
            "actual": {"Ag": 0.90, "Cu": 0.10},
            "tolerance": 0.05,
            "should_match": True,
            "description": "Within tolerance",
        },
        {
            "target": {"Ag": 0.875, "Cu": 0.125},
            "actual": {"Ag": 0.95, "Cu": 0.05},
            "tolerance": 0.05,
            "should_match": False,
            "description": "Outside tolerance",
        },
        {
            "target": {"Ag": 0.875, "Cu": 0.125},
            "actual": {"Ag": 0.88, "Cu": 0.12},
            "tolerance": 0.01,
            "should_match": False,
            "description": "Strict tolerance",
        },
    ]


@pytest.fixture
def stability_test_cases():
    """Fixture providing test cases for stability criteria."""
    return [
        {"ehull": 0.0, "is_stable": True, "description": "Perfectly stable"},
        {"ehull": 0.0005, "is_stable": True, "description": "Essentially stable"},
        {"ehull": 0.010, "is_stable": False, "description": "Marginally metastable"},
        {"ehull": 0.050, "is_stable": False, "description": "Metastable"},
        {"ehull": 0.100, "is_stable": False, "description": "Near synthesis limit"},
        {"ehull": 0.250, "is_stable": False, "description": "Highly metastable"},
    ]

