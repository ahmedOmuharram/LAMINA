"""
Pytest configuration and global fixtures.

This module provides shared fixtures for all test modules including
mock API clients, test data, and utility functions.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def sample_material_ids():
    """Fixture providing sample material IDs for testing."""
    return {
        "silicon": "mp-149",
        "copper": "mp-30",
        "gold": "mp-81",
        "silver": "mp-124",
        "iron": "mp-13",
        "aluminum": "mp-134",
    }


@pytest.fixture
def sample_elements():
    """Fixture providing sample element lists for testing."""
    return {
        "binaries": [
            ["Ag", "Cu"],
            ["Fe", "Al"],
            ["Si", "Ge"],
        ],
        "ternaries": [
            ["Li", "Fe", "O"],
            ["Al", "Mg", "Si"],
        ],
    }


@pytest.fixture
def sample_compositions():
    """Fixture providing sample composition dictionaries."""
    return {
        "ag_cu_alloy": {"Ag": 0.875, "Cu": 0.125},
        "fe_al_alloy": {"Fe": 0.75, "Al": 0.25},
        "li_fe_o": {"Li": 0.33, "Fe": 0.33, "O": 0.34},
    }


@pytest.fixture
def tolerance_values():
    """Fixture providing common tolerance values for composition matching."""
    return {
        "strict": 0.02,
        "standard": 0.05,
        "relaxed": 0.10,
    }


@pytest.fixture
def energy_ranges():
    """Fixture providing energy range values for stability testing."""
    return {
        "stable": (0, 0.001),
        "marginally_stable": (0, 0.010),
        "metastable": (0, 0.050),
        "synthesis_limit": (0, 0.100),
        "broad": (0, 0.200),
    }

