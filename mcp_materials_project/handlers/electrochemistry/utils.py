"""
Utility functions for electrochemistry and battery calculations.

This module re-exports all utility functions from specialized sub-modules.
For backward compatibility, all functions are available at this level.
"""

# Re-export constants
from .constants import FARADAY_CONSTANT, PYMATGEN_AVAILABLE

# Re-export API utilities
from .api_utils import (
    build_electrode_query_params,
    process_electrode_documents,
    extract_voltage_profile
)

# Re-export voltage calculation utilities
from .voltage_utils import (
    compute_alloy_voltage_via_hull,
    calculate_voltage_from_insertion_electrode
)

# Re-export comparison utilities
from .comparison_utils import (
    generate_comparison_summary
)

# Re-export stability utilities
from .stability_utils import (
    check_composition_stability_detailed
)

# Re-export lithiation utilities
from .lithiation_utils import (
    analyze_lithiation_mechanism_detailed
)

__all__ = [
    # Constants
    'FARADAY_CONSTANT',
    'PYMATGEN_AVAILABLE',
    
    # API utilities
    'build_electrode_query_params',
    'process_electrode_documents',
    'extract_voltage_profile',
    
    # Voltage utilities
    'compute_alloy_voltage_via_hull',
    'calculate_voltage_from_insertion_electrode',
    
    # Comparison utilities
    'generate_comparison_summary',
    
    # Stability utilities
    'check_composition_stability_detailed',
    
    # Lithiation utilities
    'analyze_lithiation_mechanism_detailed',
]
